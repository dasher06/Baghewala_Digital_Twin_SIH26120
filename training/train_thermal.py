"""
training/train_thermal.py

Trains the Model 1 STAGE B learned-correction model.

Architecture recap (see spec + physics/thermal_physics.py docstring):
    Stage A (physics, already computed in the dataset as `physics_temp_C`)
        -> Stage B (this script): gradient-boosted regressor predicts the
           RESIDUAL  (temperature_C - physics_temp_C)
        -> Final temperature = physics_temp_C + predicted_residual

Loss
----
The specification calls for:
    L = L_data + lambda * L_physics_residual
where L_physics_residual penalizes violation of energy-balance consistency
between consecutive timesteps (i.e. the corrected trajectory should not
imply non-physical temperature INCREASES during soak/production, when no
new steam is injected).

XGBoost does not expose a fully custom two-term loss with an inter-row
"neighbouring timestep" term through the sklearn API in a simple way, so we
implement this as a two-part prototype scheme that is faithful to the
intent of the spec:
    1. L_data: standard squared-error training of XGBoost on the residual
       target (temperature_C - physics_temp_C). This is the dominant term.
    2. L_physics_residual (soft, applied as a POST-HOC monotonicity-aware
       correction + reported diagnostic): after generating predictions on
       held-out well-cycles, we compute a physics-violation diagnostic --
       the fraction of consecutive-timestep pairs (within a soak/production
       phase, no new steam) where the corrected trajectory INCREASES in
       temperature. We report this as `physics_violation_rate` in the
       training report, and apply a light isotonic-style clipping at
       inference time (see inference/thermal.py: `_enforce_monotonic_decay`)
       so the deployed model never outputs a physically nonsensical
       temperature increase within a no-injection phase. This keeps the
       physics-consistency spirit of "L = L_data + lambda*L_physics_residual"
       while remaining implementable as a straightforward, fast, prototype
       XGBoost pipeline (a full differentiable custom-loss version is a
       natural extension once real data / more dev time is available).

Run:
    cd Baghewala_Digital_Twin
    python training/train_thermal.py

Output:
    models/thermal/stage_b_correction_model.joblib
    models/thermal/thermal_feature_columns.json
    models/thermal/training_report.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic", "thermal_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "thermal")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "steam_volume_bbl",
    "injection_pressure_psi",
    "steam_quality_frac",
    "soak_duration_days",
    "time_since_injection_end_days",
    "prior_cycle_count",
    "thermal_conductivity_W_mK",
    "porosity_frac",
    "formation_thickness_m",
    "baseline_reservoir_temp_C",
    "crude_api_gravity",
    "cum_production_bbl",
    "physics_temp_C",  # Stage A output is itself a useful Stage B feature
]
TARGET_RESIDUAL = "residual_target"  # = temperature_C - physics_temp_C


def physics_violation_rate(df_with_preds: pd.DataFrame) -> float:
    """
    Diagnostic for the physics-residual loss term: fraction of consecutive
    within-cycle timestep pairs where the FINAL corrected prediction
    increases in temperature (non-physical, since no new steam is injected
    during soak/production).
    """
    violations, total = 0, 0
    for _, g in df_with_preds.sort_values("time_since_injection_end_days").groupby(
        ["well_id", "cycle_num"]
    ):
        preds = g["final_temp_pred_C"].values
        diffs = np.diff(preds)
        violations += int(np.sum(diffs > 0.5))  # >0.5C tolerance for noise
        total += len(diffs)
    return violations / total if total > 0 else 0.0


def train():
    df = pd.read_csv(DATA_PATH)
    df[TARGET_RESIDUAL] = df["temperature_C"] - df["physics_temp_C"]

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_RESIDUAL]
    groups = df["well_id"]  # split by WELL so no well leaks across train/test

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=7)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        random_state=7,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # --- Evaluate on residual target directly ---
    resid_pred_test = model.predict(X_test)
    resid_mae = mean_absolute_error(y_test, resid_pred_test)
    resid_r2 = r2_score(y_test, resid_pred_test)

    # --- Evaluate on FINAL temperature (physics + correction) ---
    df_test = df.iloc[test_idx].copy()
    df_test["final_temp_pred_C"] = df_test["physics_temp_C"].values + resid_pred_test
    final_mae = mean_absolute_error(df_test["temperature_C"], df_test["final_temp_pred_C"])
    final_r2 = r2_score(df_test["temperature_C"], df_test["final_temp_pred_C"])

    # baseline: physics-only (no correction) error, to show Stage B's value-add
    physics_only_mae = mean_absolute_error(df_test["temperature_C"], df_test["physics_temp_C"])

    phys_violation = physics_violation_rate(df_test)

    report = {
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "n_train_wells": int(groups.iloc[train_idx].nunique()),
        "n_test_wells": int(groups.iloc[test_idx].nunique()),
        "residual_model_MAE_C": round(float(resid_mae), 4),
        "residual_model_R2": round(float(resid_r2), 4),
        "final_temperature_MAE_C": round(float(final_mae), 4),
        "final_temperature_R2": round(float(final_r2), 4),
        "physics_only_baseline_MAE_C": round(float(physics_only_mae), 4),
        "physics_residual_violation_rate": round(float(phys_violation), 4),
        "note": (
            "physics_residual_violation_rate is the diagnostic used in place of a "
            "differentiable L_physics_residual term (see module docstring). "
            "It is enforced away at inference time via monotonic-decay clipping."
        ),
        "data_is_synthetic": True,
    }

    joblib.dump(model, os.path.join(MODEL_DIR, "stage_b_correction_model.joblib"))
    with open(os.path.join(MODEL_DIR, "thermal_feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    with open(os.path.join(MODEL_DIR, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved model to {MODEL_DIR}/stage_b_correction_model.joblib")
    return report


if __name__ == "__main__":
    train()
