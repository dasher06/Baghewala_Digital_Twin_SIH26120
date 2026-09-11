"""
tests/test_production.py

Engineering sanity tests for MODEL 2 (Production Prediction Model).

Run:
    cd Baghewala_Digital_Twin
    python tests/test_production.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference.production import (
    predict_production,
    predict_production_trajectory,
    economic_cutoff_day,
)

PASS, FAIL = "PASS", "FAIL"


def check(name, condition):
    status = PASS if condition else FAIL
    print(f"[{status}] {name}")
    return condition


def run_tests():
    results = []

    common = dict(
        day_count_in_cycle=5,
        cycle_num=2,
        spm=5.5,
        stroke_length_in=100,
        vfd_frequency_hz=50,
        water_cut_frac=0.15,
        cum_production_this_cycle_bbl=50,
        lagged_bopd=30,
    )

    # 1. Higher viscosity (all else equal) -> lower production
    q_low_visc = predict_production(temperature_C=90, viscosity_cP=500, **common)
    q_high_visc = predict_production(temperature_C=60, viscosity_cP=15000, **common)
    results.append(check(
        f"Higher viscosity reduces production ({q_low_visc:.1f} -> {q_high_visc:.1f} BOPD)",
        q_high_visc < q_low_visc,
    ))

    # 2. Production trajectory over a cycle should generally trend downward
    #    as viscosity rises (monotonic reservoir cooling scenario)
    days = np.array([0, 5, 10, 15, 20, 25, 30])
    temps = np.linspace(95, 50, len(days))
    visc = np.geomspace(500, 25000, len(days))
    traj = predict_production_trajectory(
        temperature_C_array=temps, viscosity_cP_array=visc, day_counts_array=days,
        cycle_num=2, spm=5.5, stroke_length_in=100, vfd_frequency_hz=50,
        water_cut_frac=0.15, arps_qi=50,
    )
    results.append(check(
        "Production trajectory trends downward as viscosity rises",
        traj[0] > traj[-1],
    ))

    # 3. Economic cutoff day should be within the trajectory's day range
    cutoff = economic_cutoff_day(days, traj, cutoff_bopd=5.0)
    results.append(check(
        f"Economic cutoff day is sensible (got {cutoff})",
        cutoff is None or (days.min() <= cutoff <= days.max()),
    ))

    # 4. Predictions should never be negative
    results.append(check(
        "All predicted BOPD values are non-negative",
        np.all(traj >= 0),
    ))

    # 5. Non-negative water cut sanity: near-100% water cut should reduce production a lot
    q_normal_wc = predict_production(temperature_C=80, viscosity_cP=1500, **{**common, "water_cut_frac": 0.1})
    q_high_wc = predict_production(temperature_C=80, viscosity_cP=1500, **{**common, "water_cut_frac": 0.5})
    results.append(check(
        f"Higher water cut reduces oil production ({q_normal_wc:.1f} -> {q_high_wc:.1f} BOPD)",
        q_high_wc <= q_normal_wc,
    ))

    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} checks passed.")
    if n_pass != len(results):
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
