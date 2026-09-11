"""
inference/production.py

MODEL 2 — Production Prediction Model — INFERENCE MODULE.

Public API
----------
    predict_production(temperature_C, viscosity_cP, day_count_in_cycle,
                        cycle_num, spm, stroke_length_in, vfd_frequency_hz,
                        water_cut_frac, cum_production_this_cycle_bbl,
                        lagged_bopd, arps_expected_bopd=None)
        -> float  (predicted BOPD)

    predict_production_trajectory(..., day_counts_array, arps_qi=None,
                        arps_di=0.03, arps_b=0.35)
        -> np.ndarray  (forward BOPD trajectory over the given day_counts)

    economic_cutoff_day(trajectory_days, trajectory_bopd, cutoff_bopd)
        -> float or None  (day at which production first drops below
                            the economic cutoff threshold)

Model 2 consumes Model 1's outputs directly (temperature_C, viscosity_cP),
using THE SAME variable names, per the cross-model compatibility
requirement in the project brief.
"""

import os
import sys
import json
import numpy as np
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "production")
sys.path.insert(0, BASE_DIR)

from physics.production_physics import arps_decline_bopd

_MODEL_PATH = os.path.join(MODEL_DIR, "production_model.joblib")
_FEATURES_PATH = os.path.join(MODEL_DIR, "production_feature_columns.json")

_model = None
_feature_columns = None
LAMBDA_DECLINE_BLEND = 0.08  # must match training/train_production.py


def _load_model():
    global _model, _feature_columns
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Production model not found at {_MODEL_PATH}. "
                f"Run training/train_production.py first."
            )
        _model = joblib.load(_MODEL_PATH)
        with open(_FEATURES_PATH) as f:
            _feature_columns = json.load(f)
    return _model, _feature_columns


def predict_production(
    temperature_C: float,
    viscosity_cP: float,
    day_count_in_cycle: float,
    cycle_num: int,
    spm: float,
    stroke_length_in: float,
    vfd_frequency_hz: float,
    water_cut_frac: float,
    cum_production_this_cycle_bbl: float,
    lagged_bopd: float,
    arps_expected_bopd: float = None,
) -> float:
    """
    Single-point BOPD prediction. If `arps_expected_bopd` is not supplied,
    it defaults to `lagged_bopd` (a reasonable naive decline-consistency
    prior when no fitted Arps curve is available for this exact call).
    """
    model, feature_columns = _load_model()

    if arps_expected_bopd is None:
        arps_expected_bopd = lagged_bopd

    row = {
        "temperature_C": temperature_C,
        "viscosity_cP": viscosity_cP,
        "day_count_in_cycle": day_count_in_cycle,
        "cycle_num": cycle_num,
        "spm": spm,
        "stroke_length_in": stroke_length_in,
        "vfd_frequency_hz": vfd_frequency_hz,
        "water_cut_frac": water_cut_frac,
        "cum_production_this_cycle_bbl": cum_production_this_cycle_bbl,
        "lagged_bopd": lagged_bopd,
        "arps_expected_bopd": arps_expected_bopd,
    }
    X = np.array([[row[c] for c in feature_columns]])
    raw_pred = float(model.predict(X)[0])
    blended = (1 - LAMBDA_DECLINE_BLEND) * raw_pred + LAMBDA_DECLINE_BLEND * arps_expected_bopd
    return max(blended, 0.0)


def predict_production_trajectory(
    temperature_C_array,
    viscosity_cP_array,
    day_counts_array,
    cycle_num: int,
    spm: float,
    stroke_length_in: float,
    vfd_frequency_hz: float,
    water_cut_frac: float,
    arps_qi: float = None,
    arps_di: float = 0.03,
    arps_b: float = 0.35,
) -> np.ndarray:
    """
    Forward-in-time BOPD trajectory. Iterates day-by-day so `lagged_bopd`
    and `cum_production_this_cycle_bbl` are updated self-consistently, and
    uses an Arps curve (qi defaulting to a rough IPR-scale guess if not
    given) as the decline-consistency prior at each step.

    This is the main entry point Model 5 (future optimizer) will use to
    evaluate a candidate CSS+SRP combination's production trajectory.
    """
    temperature_C_array = np.atleast_1d(np.asarray(temperature_C_array, dtype=float))
    viscosity_cP_array = np.atleast_1d(np.asarray(viscosity_cP_array, dtype=float))
    day_counts_array = np.atleast_1d(np.asarray(day_counts_array, dtype=float))

    n = len(day_counts_array)
    bopd_trajectory = np.zeros(n)
    cum_prod = 0.0
    lagged = arps_qi if arps_qi is not None else 50.0  # naive starting guess

    if arps_qi is None:
        arps_qi = 50.0
    arps_expected_series = arps_decline_bopd(arps_qi, arps_di, arps_b, day_counts_array)

    prev_day = 0.0
    for i in range(n):
        pred = predict_production(
            temperature_C=temperature_C_array[i],
            viscosity_cP=viscosity_cP_array[i],
            day_count_in_cycle=day_counts_array[i],
            cycle_num=cycle_num,
            spm=spm,
            stroke_length_in=stroke_length_in,
            vfd_frequency_hz=vfd_frequency_hz,
            water_cut_frac=water_cut_frac,
            cum_production_this_cycle_bbl=cum_prod,
            lagged_bopd=lagged,
            arps_expected_bopd=arps_expected_series[i],
        )
        bopd_trajectory[i] = pred
        dt = max(day_counts_array[i] - prev_day, 0.0)
        cum_prod += pred * dt
        prev_day = day_counts_array[i]
        lagged = pred

    return bopd_trajectory


def economic_cutoff_day(day_counts_array, bopd_trajectory, cutoff_bopd: float = 5.0):
    """
    Returns the first day at which predicted BOPD drops below `cutoff_bopd`
    (a simple economic-limit heuristic), or None if it never does within
    the given trajectory.
    """
    day_counts_array = np.asarray(day_counts_array)
    bopd_trajectory = np.asarray(bopd_trajectory)
    below = np.where(bopd_trajectory < cutoff_bopd)[0]
    if len(below) == 0:
        return None
    return float(day_counts_array[below[0]])


if __name__ == "__main__":
    days = np.array([0, 2, 5, 8, 12, 16, 20, 25, 30])
    # illustrative viscosity trajectory (would normally come from Model 1)
    temps = np.array([95, 90, 82, 74, 65, 60, 56, 53, 51])
    visc = np.array([700, 950, 1600, 2800, 5500, 8200, 12000, 17000, 21000])

    traj = predict_production_trajectory(
        temperature_C_array=temps,
        viscosity_cP_array=visc,
        day_counts_array=days,
        cycle_num=3,
        spm=5.5,
        stroke_length_in=100,
        vfd_frequency_hz=50,
        water_cut_frac=0.15,
        arps_qi=60,
    )
    for d, t, v, q in zip(days, temps, visc, traj):
        print(f"day={d:3d}  T={t:5.1f}C  visc={v:7.1f}cP  BOPD={q:6.2f}")

    cutoff = economic_cutoff_day(days, traj, cutoff_bopd=5.0)
    print(f"\nEconomic cutoff day (BOPD < 5): {cutoff}")
