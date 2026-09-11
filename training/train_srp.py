"""
training/train_srp.py

Trains MODEL 3 (SRP Performance Model) STAGE B learned corrections.

Architecture recap (see spec + physics/srp_physics.py docstring):
    Stage A (physics, already computed as `pprl_physics_lbs` /
             `efficiency_physics_frac` in the dataset)
        -> Stage B (this script): THREE separate gradient-boosted
           regressors, one per output, each predicting the residual between
           the physics backbone's estimate and the "observed" (synthetic)
           value:
             1. pprl_residual        = peak_polished_rod_load_lbs - pprl_physics_lbs
             2. efficiency_residual  = pump_efficiency_pct/100 - efficiency_physics_frac
             3. energy_residual      = energy_kwh_per_bbl - energy_from_physics_pprl
                (physics-only energy recomputed from pprl_physics + physics
                 efficiency, for a fair residual baseline)
        -> Final = physics backbone output + predicted residual

Loss
----
L = L_data + lambda * L_API11L_residual

As with Model 1, a fully custom differentiable two-term XGBoost loss is out
of scope for a fast prototype; the practical equivalent implemented here is:
  - L_data: standard squared-error residual training (dominant term)
  - L_API11L_residual: enforced as a SANITY-CLIPPING guard at inference time
    (efficiency in [3%, 98%], PPRL >= 0, energy >= 0) plus a reported
    `mean_relative_deviation_from_physics` diagnostic per output, so any
    case where Stage B pushes far from the API-RP-11L-consistent baseline
    is visible and auditable.

Design requirement: Model 3 must be a FAST CALLABLE FUNCTION (queried
repeatedly by Model 5 later). XGBoost inference on a handful of rows is
sub-millisecond, satisfying this requirement -- see inference/srp.py.

Run:
    cd Baghewala_Digital_Twin
    python training/train_srp.py

Output:
    models/srp/pprl_correction_model.joblib
    models/srp/efficiency_correction_model.joblib
    models/srp/energy_correction_model.joblib
    models/srp/srp_feature_columns.json
    models/srp/training_report.json
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
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic", "srp_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "srp")
os.makedirs(MODEL_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
from physics.srp_physics import SRPPhysicsParams, energy_consumption_kwh_per_bbl

FEATURE_COLUMNS = [
    "viscosity_cP",
    "fluid_level_m",
    "spm",
    "stroke_length_in",
    "vfd_frequency_hz",
    "rod_string_diameter_in",
    "rod_string_length_ft",
    "pump_depth_ft",
    "tubing_size_in",
    "fluid_specific_gravity",
    "pprl_physics_lbs",
    "efficiency_physics_frac",
]


def train():
    df = pd.read_csv(DATA_PATH)

    # physics-only energy baseline for a fair residual comparison
    params = SRPPhysicsParams()
    physics_energy = []
    for _, row in df.iterrows():
        theo_bopd_at_eff = (row["actual_bopd"] / row["efficiency_physics_frac"]) \
            if row["efficiency_physics_frac"] > 1e-6 else 0.0
        actual_bopd_physics = theo_bopd_at_eff * row["efficiency_physics_frac"]
        kwh_bbl, _ = energy_consumption_kwh_per_bbl(
            row["pprl_physics_lbs"], row["stroke_length_in"], row["spm"],
            max(actual_bopd_physics, 1e-3), params,
        )
        physics_energy.append(kwh_bbl)
    df["energy_physics_kwh_per_bbl"] = physics_energy

    df["pprl_residual"] = df["peak_polished_rod_load_lbs"] - df["pprl_physics_lbs"]
    df["efficiency_residual"] = (df["pump_efficiency_pct"] / 100.0) - df["efficiency_physics_frac"]
    df["energy_residual"] = df["energy_kwh_per_bbl"] - df["energy_physics_kwh_per_bbl"]

    X = df[FEATURE_COLUMNS]
    groups = df["well_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=5)
    train_idx, test_idx = next(splitter.split(X, groups=groups))

    targets = {
        "pprl": ("pprl_residual", "pprl_physics_lbs", "peak_polished_rod_load_lbs"),
        "efficiency": ("efficiency_residual", "efficiency_physics_frac", None),  # pct handled separately
        "energy": ("energy_residual", "energy_physics_kwh_per_bbl", "energy_kwh_per_bbl"),
    }

    report = {"n_train_rows": int(len(train_idx)), "n_test_rows": int(len(test_idx)), "outputs": {}}

    for name, (resid_col, physics_col, observed_col) in targets.items():
        y = df[resid_col]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBRegressor(
            n_estimators=350, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.3,
            random_state=5, objective="reg:squarederror",
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}_correction_model.joblib"))

        resid_pred = model.predict(X_test)
        physics_test = df[physics_col].iloc[test_idx].values
        final_pred = physics_test + resid_pred

        if name == "efficiency":
            observed_test = (df["pump_efficiency_pct"] / 100.0).iloc[test_idx].values
        else:
            observed_test = df[observed_col].iloc[test_idx].values

        mae = mean_absolute_error(observed_test, final_pred)
        r2 = r2_score(observed_test, final_pred)
        physics_only_mae = mean_absolute_error(observed_test, physics_test)
        mean_rel_dev = float(np.mean(np.abs(resid_pred) / (np.abs(physics_test) + 1e-6)))

        report["outputs"][name] = {
            "final_MAE": round(float(mae), 4),
            "final_R2": round(float(r2), 4),
            "physics_only_baseline_MAE": round(float(physics_only_mae), 4),
            "mean_relative_deviation_from_physics": round(mean_rel_dev, 4),
        }

    with open(os.path.join(MODEL_DIR, "srp_feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    report["data_is_synthetic"] = True
    with open(os.path.join(MODEL_DIR, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved 3 correction models to {MODEL_DIR}/")
    return report


if __name__ == "__main__":
    train()
