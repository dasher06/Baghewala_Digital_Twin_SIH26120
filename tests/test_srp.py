"""
tests/test_srp.py

Engineering sanity tests for MODEL 3 (SRP Performance Model).

Run:
    cd Baghewala_Digital_Twin
    python tests/test_srp.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference.srp import predict_srp_performance

PASS, FAIL = "PASS", "FAIL"

FIXED_WELL = dict(
    rod_string_diameter_in=0.875,
    rod_string_length_ft=4000,
    pump_depth_ft=3800,
    tubing_size_in=2.875,
    fluid_level_m=600,
)


def check(name, condition):
    status = PASS if condition else FAIL
    print(f"[{status}] {name}")
    return condition


def run_tests():
    results = []

    low_visc = predict_srp_performance(viscosity_cP=200, spm=5.5, stroke_length_in=100,
                                        vfd_frequency_hz=50, **FIXED_WELL)
    high_visc = predict_srp_performance(viscosity_cP=20000, spm=5.5, stroke_length_in=100,
                                         vfd_frequency_hz=50, **FIXED_WELL)

    # 1. Higher viscosity -> lower pump efficiency
    results.append(check(
        f"Higher viscosity reduces pump efficiency ({low_visc['pump_efficiency_pct']:.1f}% "
        f"-> {high_visc['pump_efficiency_pct']:.1f}%)",
        high_visc["pump_efficiency_pct"] < low_visc["pump_efficiency_pct"],
    ))

    # 2. Higher viscosity -> higher energy per barrel
    results.append(check(
        f"Higher viscosity increases energy/bbl ({low_visc['energy_kwh_per_bbl']:.2f} "
        f"-> {high_visc['energy_kwh_per_bbl']:.2f} kWh/bbl)",
        high_visc["energy_kwh_per_bbl"] > low_visc["energy_kwh_per_bbl"],
    ))

    # 3. Higher viscosity -> lower actual BOPD from pump (all else equal)
    results.append(check(
        f"Higher viscosity reduces pump throughput ({low_visc['actual_bopd_from_pump']:.1f} "
        f"-> {high_visc['actual_bopd_from_pump']:.1f} BOPD)",
        high_visc["actual_bopd_from_pump"] < low_visc["actual_bopd_from_pump"],
    ))

    # 4. Efficiency should always stay within physically valid [0,100]% bounds
    results.append(check(
        "Pump efficiency stays within [0, 100]%",
        0 <= low_visc["pump_efficiency_pct"] <= 100 and 0 <= high_visc["pump_efficiency_pct"] <= 100,
    ))

    # 5. Higher SPM (faster pumping) -> higher peak polished rod load
    low_spm = predict_srp_performance(viscosity_cP=1000, spm=3, stroke_length_in=100,
                                       vfd_frequency_hz=50, **FIXED_WELL)
    high_spm = predict_srp_performance(viscosity_cP=1000, spm=8, stroke_length_in=100,
                                        vfd_frequency_hz=50, **FIXED_WELL)
    results.append(check(
        f"Higher SPM increases peak polished rod load ({low_spm['peak_polished_rod_load_lbs']:.0f} "
        f"-> {high_spm['peak_polished_rod_load_lbs']:.0f} lbs)",
        high_spm["peak_polished_rod_load_lbs"] > low_spm["peak_polished_rod_load_lbs"],
    ))

    # 6. Speed check: must be fast enough for repeated optimizer queries (<10ms/call)
    predict_srp_performance(viscosity_cP=1000, spm=5, stroke_length_in=100,
                             vfd_frequency_hz=50, **FIXED_WELL)  # warm-up
    t0 = time.time()
    for _ in range(300):
        predict_srp_performance(viscosity_cP=1000, spm=5, stroke_length_in=100,
                                 vfd_frequency_hz=50, **FIXED_WELL)
    avg_ms = (time.time() - t0) / 300 * 1000
    results.append(check(f"Fast callable function (<10ms/call): {avg_ms:.2f}ms/call", avg_ms < 10))

    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} checks passed.")
    if n_pass != len(results):
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
