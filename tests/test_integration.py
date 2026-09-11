"""
tests/test_integration.py

FULL PIPELINE INTEGRATION TEST — chains Models 1 -> 2 -> 3 -> 4 exactly the
way the future Model 5 optimizer will: given a candidate CSS + SRP setting,
evaluate the whole well-to-surface digital twin and print a consolidated
report.

This is the script to run to prove "the models are compatible with each
other" and "share common variable names and units" (per project brief),
NOT a formal unit test suite -- run tests/test_thermal.py, test_production.py,
test_srp.py, test_failure.py individually for per-model checks.

Run:
    cd Baghewala_Digital_Twin
    python tests/test_integration.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.thermal import predict_viscosity_trajectory
from inference.production import predict_production_trajectory, economic_cutoff_day
from inference.srp import predict_srp_performance
from inference.failure import predict_failure_risk_from_card_dict
from physics.rod_dynamics import generate_card


def run_pipeline(candidate: dict, well_fixed: dict, verbose: bool = True):
    """
    candidate: CSS + SRP decision variables (what Model 5 would search over)
        steam_volume_bbl, injection_pressure_psi, steam_quality_frac,
        soak_duration_days, spm, stroke_length_in, vfd_frequency_hz

    well_fixed: fixed well/reservoir/completion parameters
        prior_cycle_count, thermal_conductivity_W_mK, porosity_frac,
        formation_thickness_m, baseline_reservoir_temp_C, crude_api_gravity,
        rod_string_diameter_in, rod_string_length_ft, pump_depth_ft,
        tubing_size_in, fluid_level_m, water_cut_frac

    Returns a consolidated result dict.
    """
    t_days = np.linspace(0, 45, 15)  # days since injection end (soak+production)

    # ---- MODEL 1: thermal/viscosity trajectory ----
    temps, visc = predict_viscosity_trajectory(
        steam_volume_bbl=candidate["steam_volume_bbl"],
        injection_pressure_psi=candidate["injection_pressure_psi"],
        steam_quality_frac=candidate["steam_quality_frac"],
        soak_duration_days=candidate["soak_duration_days"],
        t_days_array=t_days,
        prior_cycle_count=well_fixed["prior_cycle_count"],
        thermal_conductivity_W_mK=well_fixed["thermal_conductivity_W_mK"],
        porosity_frac=well_fixed["porosity_frac"],
        formation_thickness_m=well_fixed["formation_thickness_m"],
        baseline_reservoir_temp_C=well_fixed["baseline_reservoir_temp_C"],
        crude_api_gravity=well_fixed["crude_api_gravity"],
    )

    # production-phase-only slice (after soak)
    prod_mask = t_days > candidate["soak_duration_days"]
    prod_days = t_days[prod_mask] - candidate["soak_duration_days"]
    prod_temps = temps[prod_mask]
    prod_visc = visc[prod_mask]

    # ---- MODEL 2: production trajectory ----
    bopd_traj = predict_production_trajectory(
        temperature_C_array=prod_temps,
        viscosity_cP_array=prod_visc,
        day_counts_array=prod_days,
        cycle_num=well_fixed["prior_cycle_count"] + 1,
        spm=candidate["spm"],
        stroke_length_in=candidate["stroke_length_in"],
        vfd_frequency_hz=candidate["vfd_frequency_hz"],
        water_cut_frac=well_fixed["water_cut_frac"],
        arps_qi=60.0,
    )
    cutoff_day = economic_cutoff_day(prod_days, bopd_traj, cutoff_bopd=5.0)

    # ---- MODEL 3: SRP performance at mid-cycle viscosity ----
    mid_visc = float(np.median(prod_visc))
    srp_result = predict_srp_performance(
        viscosity_cP=mid_visc,
        fluid_level_m=well_fixed["fluid_level_m"],
        spm=candidate["spm"],
        stroke_length_in=candidate["stroke_length_in"],
        vfd_frequency_hz=candidate["vfd_frequency_hz"],
        rod_string_diameter_in=well_fixed["rod_string_diameter_in"],
        rod_string_length_ft=well_fixed["rod_string_length_ft"],
        pump_depth_ft=well_fixed["pump_depth_ft"],
        tubing_size_in=well_fixed["tubing_size_in"],
    )

    # ---- MODEL 4: failure risk from a (synthetic-simulated) live card ----
    # In production this would come from a REAL sensor reading; here we
    # generate an illustrative "mostly normal, mild wear" card for the demo.
    rng = np.random.default_rng(0)
    card = generate_card(
        "normal", viscosity_cP=mid_visc, spm=candidate["spm"],
        stroke_length_in=candidate["stroke_length_in"], rng=rng,
    )
    failure_result = predict_failure_risk_from_card_dict(card)

    trapz_fn = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz -> trapezoid
    total_production_bbl = float(trapz_fn(bopd_traj, prod_days))
    steam_oil_ratio = candidate["steam_volume_bbl"] / max(total_production_bbl, 1e-3)

    result = {
        "temperature_trajectory_C": temps.round(2).tolist(),
        "viscosity_trajectory_cP": visc.round(1).tolist(),
        "production_trajectory_bopd": bopd_traj.round(2).tolist(),
        "economic_cutoff_day": cutoff_day,
        "total_production_bbl_45day": round(total_production_bbl, 1),
        "steam_oil_ratio": round(steam_oil_ratio, 3),
        "srp_performance": srp_result,
        "failure_risk": failure_result,
    }

    if verbose:
        print("=" * 70)
        print("BAGHEWALA DIGITAL TWIN — INTEGRATED PIPELINE RUN (Models 1-4)")
        print("=" * 70)
        print(f"Candidate CSS+SRP setting: {candidate}")
        print(f"\nModel 1 (thermal): peak T={temps.max():.1f}C -> T at day45={temps[-1]:.1f}C")
        print(f"                    peak visc={visc.min():.0f}cP -> final visc={visc[-1]:.0f}cP")
        print(f"Model 2 (production): peak BOPD={bopd_traj.max():.1f}, "
              f"total 45-day production={total_production_bbl:.1f} bbl")
        print(f"                       economic cutoff day: {cutoff_day}")
        print(f"                       Steam-Oil-Ratio (SOR): {steam_oil_ratio:.3f}")
        print(f"Model 3 (SRP): efficiency={srp_result['pump_efficiency_pct']:.1f}%, "
              f"PPRL={srp_result['peak_polished_rod_load_lbs']:.0f}lbs, "
              f"energy={srp_result['energy_kwh_per_bbl']:.2f}kWh/bbl")
        print(f"Model 4 (failure risk): predicted_class={failure_result['predicted_class']}, "
              f"risk_score={failure_result['failure_risk_score']:.3f}")
        print("=" * 70)

    return result


def run_tests():
    candidate = dict(
        steam_volume_bbl=1400,
        injection_pressure_psi=320,
        steam_quality_frac=0.72,
        soak_duration_days=7,
        spm=5.5,
        stroke_length_in=100,
        vfd_frequency_hz=50,
    )
    well_fixed = dict(
        prior_cycle_count=2,
        thermal_conductivity_W_mK=1.8,
        porosity_frac=0.27,
        formation_thickness_m=12,
        baseline_reservoir_temp_C=47,
        crude_api_gravity=18,
        rod_string_diameter_in=0.875,
        rod_string_length_ft=4000,
        pump_depth_ft=3800,
        tubing_size_in=2.875,
        fluid_level_m=600,
        water_cut_frac=0.15,
    )

    result = run_pipeline(candidate, well_fixed)

    checks = []
    checks.append(("Pipeline runs end-to-end without error", True))
    checks.append(("Temperature trajectory is a valid list of floats",
                    len(result["temperature_trajectory_C"]) > 0))
    checks.append(("Production values are all non-negative",
                    all(v >= 0 for v in result["production_trajectory_bopd"])))
    checks.append(("SRP efficiency is within [0,100]%",
                    0 <= result["srp_performance"]["pump_efficiency_pct"] <= 100))
    checks.append(("Failure risk score is within [0,1]",
                    0 <= result["failure_risk"]["failure_risk_score"] <= 1))
    checks.append(("Steam-oil-ratio is a positive finite number",
                    result["steam_oil_ratio"] > 0))

    print("\nIntegration checks:")
    n_pass = 0
    for name, cond in checks:
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}")
        n_pass += int(cond)
    print(f"\n{n_pass}/{len(checks)} checks passed.")
    if n_pass != len(checks):
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
