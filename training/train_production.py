"""
training/train_production.py

Trains MODEL 2 (Production Prediction Model): XGBoost regressor predicting
BOPD from current reservoir/CSS/SRP state.

Physics-consistency regularizer (Arps decline-curve consistency)
------------------------------------------------------------------
Per spec: "Soft penalty term based on Arps decline-curve consistency... a
small lambda -- soft nudge, not a hard constraint, since production is
influenced by factors decline curves don't capture: skin damage, sanding,
interventions."

Implementation approach (consistent in spirit with Model 1's approach):
  1. For each well-cycle in the training data, fit a simple Arps decline
     curve (qi, Di, b) to that cycle's OBSERVED bopd-vs-day_count points via
     least squares. This gives an "expected decline-consistent rate"
     `arps_expected_bopd` at each row -- used as an ENGINEERED FEATURE fed
     into XGBoost (so the model can lean on decline-curve shape where it's
     informative, and deviate from it where reservoir/SRP features explain
     the deviation, e.g. skin damage / interventions / sanding are NOT in
     our feature set, so the model must partly fall back to the Arps prior).
  2. After training, we report a `decline_consistency_MAE` diagnostic:
     mean absolute difference between model predictions and the fitted
     Arps curve, PER WELL-CYCLE. A small lambda-style soft penalty is
     approximated by blending: final_prediction = (1-lambda)*xgb_pred +
     lambda*arps_expected_bopd, with a SMALL lambda (default 0.08) so the
     Arps shape provides only a gentle nudge, never a hard constraint --
     directly implementing "L = L_data + lambda_small * L_decline_consistency"
     as a prediction-space blend rather than a loss-space penalty (simpler
     to implement correctly for a prototype, same practical effect).
  3. Feature importance is reported explicitly, to support the pitch
     narrative point that viscosity should be the dominant driver.

Run:
    cd Baghewala_Digital_Twin
    python training/train_production.py

Output:
    models/production/production_model.joblib
    models/production/production_feature_columns.json
    models/production/training_report.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic", "production_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "production")
os.makedirs(MODEL_DIR, exist_ok=True)

LAMBDA_DECLINE_BLEND = 0.08  # small, soft nudge (per spec: "small lambda")

FEATURE_COLUMNS = [
    "temperature_C",
    "viscosity_cP",
    "day_count_in_cycle",
    "cycle_num",
    "spm",
    "stroke_length_in",
    "vfd_frequency_hz",
    "water_cut_frac",
    "cum_production_this_cycle_bbl",
    "lagged_bopd",
    "arps_expected_bopd",  # engineered decline-consistency feature
]
TARGET = "bopd"


def _arps_hyperbolic(t, qi, di, b):
    b = max(b, 1e-4)
    return qi / np.power(1.0 + b * di * np.clip(t, 0, None), 1.0 / b)


def fit_arps_per_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Fit an Arps curve per (well_id, cycle_num) to observed bopd vs day_count,
    add `arps_expected_bopd` column with the fitted-curve value at each row."""
    df = df.copy()
    df["arps_expected_bopd"] = np.nan

    for (well, cyc), g in df.groupby(["well_id", "cycle_num"]):
        t = g["day_count_in_cycle"].values
        q = g["bopd"].values
        try:
            popt, _ = curve_fit(
                _arps_hyperbolic, t, q,
                p0=[max(q.max(), 1.0), 0.05, 0.35],
                bounds=([0, 0.001, 0.01], [10000, 2.0, 1.0]),
                maxfev=5000,
            )
            fitted = _arps_hyperbolic(t, *popt)
        except Exception:
            fitted = np.full_like(q, q.mean())  # fallback if fit fails
        df.loc[g.index, "arps_expected_bopd"] = fitted

    return df


def train():
    df = pd.read_csv(DATA_PATH)
    df = fit_arps_per_cycle(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    groups = df["well_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=11)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = xgb.XGBRegressor(
        n_estimators=450,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=1.2,
        random_state=11,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    raw_pred = model.predict(X_test)
    arps_test = X_test["arps_expected_bopd"].values
    blended_pred = (1 - LAMBDA_DECLINE_BLEND) * raw_pred + LAMBDA_DECLINE_BLEND * arps_test
    blended_pred = np.clip(blended_pred, 0, None)

    raw_mae = mean_absolute_error(y_test, raw_pred)
    blended_mae = mean_absolute_error(y_test, blended_pred)
    blended_r2 = r2_score(y_test, blended_pred)
    decline_consistency_mae = mean_absolute_error(raw_pred, arps_test)

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4).tolist()))
    importances_sorted = dict(sorted(importances.items(), key=lambda kv: -kv[1]))

    report = {
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "n_train_wells": int(groups.iloc[train_idx].nunique()),
        "n_test_wells": int(groups.iloc[test_idx].nunique()),
        "raw_xgb_MAE_bopd": round(float(raw_mae), 3),
        "blended_final_MAE_bopd": round(float(blended_mae), 3),
        "blended_final_R2": round(float(blended_r2), 4),
        "decline_consistency_MAE_bopd": round(float(decline_consistency_mae), 3),
        "lambda_decline_blend": LAMBDA_DECLINE_BLEND,
        "feature_importances": importances_sorted,
        "top_driver": list(importances_sorted.keys())[0],
        "data_is_synthetic": True,
        "note": (
            "Final prediction = (1-lambda)*XGBoost_pred + lambda*Arps_fitted_curve, "
            "a small soft nudge toward decline-consistent shape (see module docstring)."
        ),
    }

    joblib.dump(model, os.path.join(MODEL_DIR, "production_model.joblib"))
    with open(os.path.join(MODEL_DIR, "production_feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    with open(os.path.join(MODEL_DIR, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved model to {MODEL_DIR}/production_model.joblib")
    return report


if __name__ == "__main__":
    train()
