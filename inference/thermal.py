"""
inference/thermal.py

MODEL 1 — Thermal/Reservoir Model — INFERENCE MODULE.

Exposes clean, fast, callable functions for use by:
  - the test scripts / dashboard
  - Model 2 (needs current temperature & viscosity)
  - Model 3 (needs current viscosity)
  - Model 5 (the future optimizer) — will call predict_temperature(...) and
    predict_viscosity(...) repeatedly for many candidate CSS parameter sets.

Public API
----------
    predict_temperature(steam_volume_bbl, injection_pressure_psi,
                         steam_quality_frac, soak_duration_days,
                         time_since_injection_end_days, prior_cycle_count,
                         thermal_conductivity_W_mK, porosity_frac,
                         formation_thickness_m, baseline_reservoir_temp_C,
                         crude_api_gravity, cum_production_bbl=0.0)
        -> float   (predicted temperature, degC)

    predict_temperature_trajectory(..., t_days_array, cum_production_bbl_series=None)
        -> np.ndarray  (predicted temperature trajectory, degC)

    predict_viscosity(temperature_C) -> float or np.ndarray (cP)

    predict_viscosity_trajectory(...) -> combines the two above directly
        from CSS parameters -> viscosity trajectory (cP), the most common
        call pattern Model 2/3/5 will use.
"""

import os
import json
import numpy as np
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "thermal")

import sys
sys.path.insert(0, BASE_DIR)
from physics.thermal_physics import (
    ThermalPhysicsParams,
    temperature_trajectory as _physics_temperature_trajectory,
    temperature_to_viscosity_cP as _physics_viscosity,
)

_MODEL_PATH = os.path.join(MODEL_DIR, "stage_b_correction_model.joblib")
_FEATURES_PATH = os.path.join(MODEL_DIR, "thermal_feature_columns.json")

_correction_model = None
_feature_columns = None


def _load_model():
    global _correction_model, _feature_columns
    if _correction_model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Stage B model not found at {_MODEL_PATH}. "
                f"Run training/train_thermal.py first."
            )
        _correction_model = joblib.load(_MODEL_PATH)
        with open(_FEATURES_PATH) as f:
            _feature_columns = json.load(f)
    return _correction_model, _feature_columns


def _enforce_monotonic_decay(t_days: np.ndarray, temps: np.ndarray,
                              soak_duration_days: float) -> np.ndarray:
    """
    Lightweight physics-consistency guard (see training/train_thermal.py
    docstring): during soak/production (no new steam injected), temperature
    should not INCREASE over time. If the ML-corrected trajectory implies an
    increase, clip it to the previous (non-increasing) value. This is the
    prototype's stand-in for a full differentiable physics-residual loss.
    """
    order = np.argsort(t_days)
    temps_sorted = temps[order].copy()
    for i in range(1, len(temps_sorted)):
        if temps_sorted[i] > temps_sorted[i - 1]:
            temps_sorted[i] = temps_sorted[i - 1]
    out = np.empty_like(temps)
    out[order] = temps_sorted
    return out


def predict_temperature_trajectory(
    steam_volume_bbl: float,
    injection_pressure_psi: float,
    steam_quality_frac: float,
    soak_duration_days: float,
    t_days_array,
    prior_cycle_count: int,
    thermal_conductivity_W_mK: float,
    porosity_frac: float,
    formation_thickness_m: float,
    baseline_reservoir_temp_C: float,
    crude_api_gravity: float,
    cum_production_bbl_series=None,
) -> np.ndarray:
    """
    Full Model 1 pipeline: Stage A physics backbone + Stage B ML correction
    + monotonic-decay physics guard, evaluated across an array of times.

    Returns predicted temperature (degC) at each time in t_days_array.
    """
    model, feature_columns = _load_model()

    t_days_array = np.atleast_1d(np.asarray(t_days_array, dtype=float))
    if cum_production_bbl_series is None:
        cum_production_bbl_series = np.zeros_like(t_days_array)
    else:
        cum_production_bbl_series = np.atleast_1d(
            np.asarray(cum_production_bbl_series, dtype=float)
        )

    physics_temp = _physics_temperature_trajectory(
        t_days=t_days_array,
        phase_days={"soak": soak_duration_days},
        baseline_temp_C=baseline_reservoir_temp_C,
        steam_volume_bbl=steam_volume_bbl,
        injection_pressure_psi=injection_pressure_psi,
        steam_quality_frac=steam_quality_frac,
        soak_duration_days=soak_duration_days,
        thermal_conductivity_W_mK=thermal_conductivity_W_mK,
        porosity_frac=porosity_frac,
        formation_thickness_m=formation_thickness_m,
        prior_cycle_count=prior_cycle_count,
        cum_production_bbl_series=cum_production_bbl_series,
        params=ThermalPhysicsParams(),
    )

    feat_rows = []
    for i, t in enumerate(t_days_array):
        row = {
            "steam_volume_bbl": steam_volume_bbl,
            "injection_pressure_psi": injection_pressure_psi,
            "steam_quality_frac": steam_quality_frac,
            "soak_duration_days": soak_duration_days,
            "time_since_injection_end_days": t,
            "prior_cycle_count": prior_cycle_count,
            "thermal_conductivity_W_mK": thermal_conductivity_W_mK,
            "porosity_frac": porosity_frac,
            "formation_thickness_m": formation_thickness_m,
            "baseline_reservoir_temp_C": baseline_reservoir_temp_C,
            "crude_api_gravity": crude_api_gravity,
            "cum_production_bbl": cum_production_bbl_series[i],
            "physics_temp_C": physics_temp[i],
        }
        feat_rows.append([row[c] for c in feature_columns])

    residual_pred = model.predict(np.array(feat_rows))
    final_temp = physics_temp + residual_pred

    final_temp = _enforce_monotonic_decay(t_days_array, final_temp, soak_duration_days)
    return final_temp


def predict_temperature(
    steam_volume_bbl: float,
    injection_pressure_psi: float,
    steam_quality_frac: float,
    soak_duration_days: float,
    time_since_injection_end_days: float,
    prior_cycle_count: int,
    thermal_conductivity_W_mK: float,
    porosity_frac: float,
    formation_thickness_m: float,
    baseline_reservoir_temp_C: float,
    crude_api_gravity: float,
    cum_production_bbl: float = 0.0,
) -> float:
    """Single-timepoint convenience wrapper around predict_temperature_trajectory."""
    result = predict_temperature_trajectory(
        steam_volume_bbl=steam_volume_bbl,
        injection_pressure_psi=injection_pressure_psi,
        steam_quality_frac=steam_quality_frac,
        soak_duration_days=soak_duration_days,
        t_days_array=[time_since_injection_end_days],
        prior_cycle_count=prior_cycle_count,
        thermal_conductivity_W_mK=thermal_conductivity_W_mK,
        porosity_frac=porosity_frac,
        formation_thickness_m=formation_thickness_m,
        baseline_reservoir_temp_C=baseline_reservoir_temp_C,
        crude_api_gravity=crude_api_gravity,
        cum_production_bbl_series=[cum_production_bbl],
    )
    return float(result[0])


def predict_viscosity(temperature_C):
    """Temperature (degC, scalar or array) -> viscosity (cP)."""
    result = _physics_viscosity(temperature_C)
    if np.isscalar(temperature_C):
        return float(result)
    return result


def predict_viscosity_trajectory(
    steam_volume_bbl: float,
    injection_pressure_psi: float,
    steam_quality_frac: float,
    soak_duration_days: float,
    t_days_array,
    prior_cycle_count: int,
    thermal_conductivity_W_mK: float,
    porosity_frac: float,
    formation_thickness_m: float,
    baseline_reservoir_temp_C: float,
    crude_api_gravity: float,
    cum_production_bbl_series=None,
):
    """
    Convenience: CSS parameters -> viscosity trajectory (cP) directly.
    This is the main entry point Models 2, 3 and 5 will use.

    Returns (temperature_C array, viscosity_cP array).
    """
    temps = predict_temperature_trajectory(
        steam_volume_bbl=steam_volume_bbl,
        injection_pressure_psi=injection_pressure_psi,
        steam_quality_frac=steam_quality_frac,
        soak_duration_days=soak_duration_days,
        t_days_array=t_days_array,
        prior_cycle_count=prior_cycle_count,
        thermal_conductivity_W_mK=thermal_conductivity_W_mK,
        porosity_frac=porosity_frac,
        formation_thickness_m=formation_thickness_m,
        baseline_reservoir_temp_C=baseline_reservoir_temp_C,
        crude_api_gravity=crude_api_gravity,
        cum_production_bbl_series=cum_production_bbl_series,
    )
    visc = predict_viscosity(temps)
    return temps, visc


if __name__ == "__main__":
    t_days = np.array([0, 5, 10, 20, 30, 45, 60])
    temps, visc = predict_viscosity_trajectory(
        steam_volume_bbl=1200,
        injection_pressure_psi=300,
        steam_quality_frac=0.7,
        soak_duration_days=7,
        t_days_array=t_days,
        prior_cycle_count=2,
        thermal_conductivity_W_mK=1.8,
        porosity_frac=0.27,
        formation_thickness_m=12,
        baseline_reservoir_temp_C=47,
        crude_api_gravity=18,
    )
    for t, T, v in zip(t_days, temps, visc):
        print(f"t={t:4d}d  T={T:7.2f}C  visc={v:10.1f}cP")
