"""
inference/srp.py

MODEL 3 — SRP Performance Model — INFERENCE MODULE.

Designed as a FAST CALLABLE FUNCTION, since the future Model 5 optimizer
will call `predict_srp_performance(...)` repeatedly (potentially thousands
of times) while searching over candidate SPM/stroke/VFD combinations.
Models are loaded once (module-level cache) and each call is a handful of
XGBoost tree lookups -- sub-millisecond.

Public API
----------
    predict_srp_performance(viscosity_cP, fluid_level_m, spm,
                             stroke_length_in, vfd_frequency_hz,
                             rod_string_diameter_in, rod_string_length_ft,
                             pump_depth_ft, tubing_size_in,
                             fluid_specific_gravity=0.95)
        -> dict with keys:
             pump_efficiency_pct
             peak_polished_rod_load_lbs
             energy_kwh_per_bbl
             actual_bopd_from_pump   (implied by efficiency x theoretical displacement)
"""

import os
import sys
import json
import numpy as np
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "srp")
sys.path.insert(0, BASE_DIR)

from physics.srp_physics import (
    SRPPhysicsParams,
    peak_polished_rod_load_lbs,
    volumetric_efficiency_frac,
    theoretical_pump_displacement_bopd,
    energy_consumption_kwh_per_bbl,
)

_models = {}
_feature_columns = None
_PHYSICS_PARAMS = SRPPhysicsParams()


def _load_models():
    global _feature_columns
    if not _models:
        for name in ("pprl", "efficiency", "energy"):
            path = os.path.join(MODEL_DIR, f"{name}_correction_model.joblib")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found. Run training/train_srp.py first."
                )
            _models[name] = joblib.load(path)
        with open(os.path.join(MODEL_DIR, "srp_feature_columns.json")) as f:
            _feature_columns = json.load(f)
    return _models, _feature_columns


def predict_srp_performance(
    viscosity_cP: float,
    fluid_level_m: float,
    spm: float,
    stroke_length_in: float,
    vfd_frequency_hz: float,
    rod_string_diameter_in: float,
    rod_string_length_ft: float,
    pump_depth_ft: float,
    tubing_size_in: float,
    fluid_specific_gravity: float = 0.95,
) -> dict:
    """
    Full Model 3 pipeline: API-RP-11L-style physics backbone + Stage B ML
    correction, for ONE candidate SRP setting. Returns a dict of predicted
    performance metrics. This is the function Model 5 will call repeatedly.
    """
    models, feature_columns = _load_models()

    # --- Stage A: physics backbone ---
    pprl_physics = peak_polished_rod_load_lbs(
        spm=spm, stroke_length_in=stroke_length_in,
        rod_string_length_ft=rod_string_length_ft,
        rod_string_diameter_in=rod_string_diameter_in,
        pump_depth_ft=pump_depth_ft, viscosity_cP=viscosity_cP,
        fluid_specific_gravity=fluid_specific_gravity, params=_PHYSICS_PARAMS,
    )
    eff_physics = volumetric_efficiency_frac(
        viscosity_cP=viscosity_cP, spm=spm,
        rod_string_length_ft=rod_string_length_ft, pump_depth_ft=pump_depth_ft,
        params=_PHYSICS_PARAMS,
    )
    theo_bopd = theoretical_pump_displacement_bopd(spm, stroke_length_in, _PHYSICS_PARAMS)
    theo_bopd *= (vfd_frequency_hz / 50.0)
    actual_bopd_physics = theo_bopd * eff_physics
    energy_physics_kwh_bbl, _ = energy_consumption_kwh_per_bbl(
        pprl_physics, stroke_length_in, spm, max(actual_bopd_physics, 1e-3), _PHYSICS_PARAMS
    )

    # --- Stage B: ML correction ---
    row = {
        "viscosity_cP": viscosity_cP,
        "fluid_level_m": fluid_level_m,
        "spm": spm,
        "stroke_length_in": stroke_length_in,
        "vfd_frequency_hz": vfd_frequency_hz,
        "rod_string_diameter_in": rod_string_diameter_in,
        "rod_string_length_ft": rod_string_length_ft,
        "pump_depth_ft": pump_depth_ft,
        "tubing_size_in": tubing_size_in,
        "fluid_specific_gravity": fluid_specific_gravity,
        "pprl_physics_lbs": pprl_physics,
        "efficiency_physics_frac": eff_physics,
    }
    X = np.array([[row[c] for c in feature_columns]])

    pprl_final = pprl_physics + float(models["pprl"].predict(X)[0])
    eff_final = eff_physics + float(models["efficiency"].predict(X)[0])
    energy_final = energy_physics_kwh_bbl + float(models["energy"].predict(X)[0])

    # --- API-RP-11L-consistency guard (physics-residual enforcement) ---
    pprl_final = max(pprl_final, 0.0)
    eff_final = float(np.clip(eff_final, 0.03, 0.98))
    energy_final = max(energy_final, 0.0)

    actual_bopd = theo_bopd * eff_final

    return {
        "pump_efficiency_pct": round(eff_final * 100, 3),
        "peak_polished_rod_load_lbs": round(pprl_final, 1),
        "energy_kwh_per_bbl": round(energy_final, 4),
        "actual_bopd_from_pump": round(actual_bopd, 2),
    }


if __name__ == "__main__":
    fixed_well = dict(
        rod_string_diameter_in=0.875,
        rod_string_length_ft=4000,
        pump_depth_ft=3800,
        tubing_size_in=2.875,
        fluid_level_m=600,
    )
    for visc in [200, 1000, 5000, 20000]:
        result = predict_srp_performance(
            viscosity_cP=visc, spm=5.5, stroke_length_in=100, vfd_frequency_hz=50,
            **fixed_well,
        )
        print(f"visc={visc:6d}cP -> {result}")
