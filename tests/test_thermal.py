"""
tests/test_thermal.py

Basic sanity tests for MODEL 1 (Thermal/Reservoir Model).
Not a formal pytest validation suite against real data (we have none yet) --
these are ENGINEERING SANITY CHECKS that the model behaves in a physically
reasonable way.

Run:
    cd Baghewala_Digital_Twin
    python tests/test_thermal.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference.thermal import (
    predict_temperature,
    predict_viscosity,
    predict_viscosity_trajectory,
)

PASS, FAIL = "PASS", "FAIL"


def check(name, condition):
    status = PASS if condition else FAIL
    print(f"[{status}] {name}")
    return condition


def run_tests():
    results = []

    # 1. Temperature should decay over time (monotonic non-increase) with no re-injection
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
    results.append(check("Temperature is non-increasing over time", np.all(np.diff(temps) <= 1e-6)))

    # 2. Temperature should approach baseline reservoir temp for large t
    results.append(check(
        "Temperature approaches baseline (~47C) at large t (within 15C at t=60d)",
        abs(temps[-1] - 47.0) < 15.0,
    ))

    # 3. Viscosity should be lowest when hottest, highest when coolest
    results.append(check(
        "Viscosity increases as temperature decreases",
        np.all(np.diff(visc) >= -1e-6),
    ))

    # 4. More steam volume -> higher peak temperature (all else equal)
    t_low = predict_temperature(
        steam_volume_bbl=600, injection_pressure_psi=300, steam_quality_frac=0.7,
        soak_duration_days=7, time_since_injection_end_days=0, prior_cycle_count=1,
        thermal_conductivity_W_mK=1.8, porosity_frac=0.27, formation_thickness_m=12,
        baseline_reservoir_temp_C=47, crude_api_gravity=18,
    )
    t_high = predict_temperature(
        steam_volume_bbl=2000, injection_pressure_psi=300, steam_quality_frac=0.7,
        soak_duration_days=7, time_since_injection_end_days=0, prior_cycle_count=1,
        thermal_conductivity_W_mK=1.8, porosity_frac=0.27, formation_thickness_m=12,
        baseline_reservoir_temp_C=47, crude_api_gravity=18,
    )
    results.append(check(
        f"More steam volume increases peak temp ({t_low:.1f}C -> {t_high:.1f}C)",
        t_high > t_low,
    ))

    # 5. Viscosity correlation sanity: near-baseline temp -> high viscosity (>1000 cP)
    visc_cold = predict_viscosity(47.0)
    results.append(check(
        f"Viscosity at baseline temp (47C) is high (>1000 cP): got {visc_cold:.1f} cP",
        visc_cold > 1000,
    ))

    # 6. Viscosity at hot steam-zone temp is much lower (<200 cP)
    visc_hot = predict_viscosity(160.0)
    results.append(check(
        f"Viscosity at 160C is low (<200 cP): got {visc_hot:.1f} cP",
        visc_hot < 200,
    ))

    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} checks passed.")
    if n_pass != len(results):
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
