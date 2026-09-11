"""Model 5: constrained joint CSS + SRP optimizer for the prototype."""

from __future__ import annotations

import os
import sys
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from inference.thermal import (
    predict_temperature_trajectory,
    predict_viscosity,
)
from inference.production import predict_production_trajectory
from inference.srp import predict_srp_performance
from inference.failure import predict_failure_risk_from_card_dict
from physics.rod_dynamics import generate_card


DEFAULT_WELL = dict(
    injection_pressure_psi=200.0,
    steam_quality_frac=0.85,
    prior_cycle_count=3,
    thermal_conductivity_W_mK=2.2,
    porosity_frac=0.28,
    formation_thickness_m=12.0,
    baseline_reservoir_temp_C=47.0,
    crude_api_gravity=18.0,
    fluid_level_m=600.0,
    rod_string_diameter_in=0.875,
    rod_string_length_ft=4000.0,
    pump_depth_ft=3800.0,
    tubing_size_in=2.875,
    fluid_specific_gravity=0.95,
    water_cut_frac=0.15,
    cycle_num=4,
    lagged_bopd=45.0,
    arps_qi=50.0,
)


# Prototype operating ranges.
BOUNDS = {
    "steam_volume_bbl": (600.0, 1400.0),
    "soak_duration_days": (6.0, 18.0),
    "spm": (3.5, 7.0),
    "stroke_length_in": (60.0, 120.0),
    "vfd_frequency_hz": (35.0, 50.0),
}


def _merge(well):
    """Merge user-provided well data with prototype defaults."""
    out = DEFAULT_WELL.copy()

    if well:
        out.update(well)

    return out


def evaluate_scenario(
    params: dict,
    well: dict | None = None,
    risk_threshold: float = 0.45,
) -> dict:
    """
    Evaluate one CSS + SRP operating scenario through Models 1–4.

    This function is the main interface used by:
        - Model 5 optimizer
        - What-If Simulator
        - Streamlit dashboard
    """

    w = _merge(well)

    # ---------------------------------------------------------
    # 1. TIME GRID
    # ---------------------------------------------------------

    t = np.array(
        [
            0.0,
            params["soak_duration_days"],
            params["soak_duration_days"] + 7.0,
            params["soak_duration_days"] + 14.0,
            params["soak_duration_days"] + 30.0,
        ]
    )

    # ---------------------------------------------------------
    # 2. MODEL 1 — THERMAL + VISCOSITY
    # ---------------------------------------------------------

    temps = predict_temperature_trajectory(
        steam_volume_bbl=params["steam_volume_bbl"],
        injection_pressure_psi=w["injection_pressure_psi"],
        steam_quality_frac=w["steam_quality_frac"],
        soak_duration_days=params["soak_duration_days"],
        t_days_array=t,
        prior_cycle_count=w["prior_cycle_count"],
        thermal_conductivity_W_mK=w["thermal_conductivity_W_mK"],
        porosity_frac=w["porosity_frac"],
        formation_thickness_m=w["formation_thickness_m"],
        baseline_reservoir_temp_C=w["baseline_reservoir_temp_C"],
        crude_api_gravity=w["crude_api_gravity"],
    )

    viscosity = predict_viscosity(temps)

    # ---------------------------------------------------------
    # 3. MODEL 2 — PRODUCTION
    # ---------------------------------------------------------

    prod = predict_production_trajectory(
        temperature_C_array=temps,
        viscosity_cP_array=viscosity,
        day_counts_array=t,
        cycle_num=w["cycle_num"],
        spm=params["spm"],
        stroke_length_in=params["stroke_length_in"],
        vfd_frequency_hz=params["vfd_frequency_hz"],
        water_cut_frac=w["water_cut_frac"],
        arps_qi=w["arps_qi"],
    )

    avg_bopd = float(np.mean(prod[-3:]))

    # ---------------------------------------------------------
    # 4. MODEL 3 — SRP PERFORMANCE
    # ---------------------------------------------------------

    srp = predict_srp_performance(
        viscosity_cP=float(viscosity[-1]),
        fluid_level_m=w["fluid_level_m"],
        spm=params["spm"],
        stroke_length_in=params["stroke_length_in"],
        vfd_frequency_hz=params["vfd_frequency_hz"],
        rod_string_diameter_in=w["rod_string_diameter_in"],
        rod_string_length_ft=w["rod_string_length_ft"],
        pump_depth_ft=w["pump_depth_ft"],
        tubing_size_in=w["tubing_size_in"],
        fluid_specific_gravity=w["fluid_specific_gravity"],
    )

    # ---------------------------------------------------------
    # 5. MODEL 4 — FAILURE / ANOMALY RISK
    # ---------------------------------------------------------

    # Deterministic synthetic card for prototype reproducibility.
    rng = np.random.default_rng(2026)

    card = generate_card(
        "normal",
        float(viscosity[-1]),
        params["spm"],
        params["stroke_length_in"],
        rng=rng,
        severity=0.15,
    )

    risk = predict_failure_risk_from_card_dict(card)

    risk_score = float(risk["failure_risk_score"])

    # ---------------------------------------------------------
    # 6. SOR PROXY
    # ---------------------------------------------------------

    total_oil_proxy = max(avg_bopd * 30.0, 1.0)

    sor_proxy = float(
        params["steam_volume_bbl"] / total_oil_proxy
    )

    # ---------------------------------------------------------
    # 7. ENERGY
    # ---------------------------------------------------------

    energy = float(srp["energy_kwh_per_bbl"])

    # ---------------------------------------------------------
    # 8. FEASIBILITY
    # ---------------------------------------------------------

    # Failure risk is the hard safety constraint.
    #
    # Pump efficiency is NOT a hard 30% cutoff because this is
    # synthetic prototype data and the existing Model 3 may
    # legitimately predict lower efficiency for heavy oil.

    feasible = risk_score <= risk_threshold

    return {
        "params": params.copy(),

        "times_days": t,

        "temperature_C": temps,

        "viscosity_cP": viscosity,

        "production_BOPD": prod,

        "predicted_BOPD": avg_bopd,

        "sor_proxy": sor_proxy,

        "pump_efficiency_pct": float(
            srp["pump_efficiency_pct"]
        ),

        "peak_polished_rod_load_lbs": float(
            srp["peak_polished_rod_load_lbs"]
        ),

        "energy_kwh_per_bbl": energy,

        "failure": risk,

        "failure_risk": risk_score,

        "feasible": feasible,
    }


def _score(result: dict, baseline: dict) -> float:
    """
    Calculate a normalized objective score.

    Higher score = better operating point.

    Priorities:
        1. Production
        2. SOR reduction
        3. Energy reduction
        4. Pump efficiency improvement
        5. Failure-risk penalty
    """

    # ---------------------------------------------------------
    # Production improvement
    # ---------------------------------------------------------

    prod_gain = (
        result["predicted_BOPD"]
        - baseline["predicted_BOPD"]
    ) / max(abs(baseline["predicted_BOPD"]), 1.0)

    # ---------------------------------------------------------
    # SOR improvement
    # ---------------------------------------------------------

    sor_gain = (
        baseline["sor_proxy"]
        - result["sor_proxy"]
    ) / max(abs(baseline["sor_proxy"]), 0.1)

    # ---------------------------------------------------------
    # Energy improvement
    # ---------------------------------------------------------

    energy_gain = (
        baseline["energy_kwh_per_bbl"]
        - result["energy_kwh_per_bbl"]
    ) / max(abs(baseline["energy_kwh_per_bbl"]), 0.1)

    # ---------------------------------------------------------
    # Pump efficiency improvement
    # ---------------------------------------------------------

    efficiency_gain = (
        result["pump_efficiency_pct"]
        - baseline["pump_efficiency_pct"]
    ) / max(abs(baseline["pump_efficiency_pct"]), 1.0)

    # ---------------------------------------------------------
    # Failure-risk penalty
    # ---------------------------------------------------------

    risk_penalty = max(
        0.0,
        result["failure_risk"] - 0.35,
    ) * 3.0

    # ---------------------------------------------------------
    # Final weighted score
    # ---------------------------------------------------------

    score = (
        0.50 * prod_gain
        + 0.20 * sor_gain
        + 0.15 * energy_gain
        + 0.15 * efficiency_gain
        - risk_penalty
    )

    return float(score)


def optimize_well(
    well=None,
    current=None,
    n_candidates=60,
    seed=42,
    risk_threshold=0.45,
):
    """
    Constrained random-search joint optimizer.

    The optimizer:
        1. Evaluates the current operating point.
        2. Generates candidate CSS + SRP settings.
        3. Runs Models 1–4 for every candidate.
        4. Removes scenarios exceeding failure-risk threshold.
        5. Scores remaining scenarios.
        6. Returns the best feasible scenario.
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # Current operating point
    # ---------------------------------------------------------

    if current is None:
        current = {
            k: (lo + hi) / 2
            for k, (lo, hi) in BOUNDS.items()
        }

    baseline = evaluate_scenario(
        current,
        well,
        risk_threshold,
    )

    # ---------------------------------------------------------
    # Generate candidate scenarios
    # ---------------------------------------------------------

    candidates = [current]

    for _ in range(max(1, n_candidates - 1)):

        candidate = {
            k: float(rng.uniform(lo, hi))
            for k, (lo, hi) in BOUNDS.items()
        }

        candidates.append(candidate)

    # ---------------------------------------------------------
    # Evaluate candidates
    # ---------------------------------------------------------

    best = None
    best_score = -np.inf

    feasible_count = 0

    for params in candidates:

        result = evaluate_scenario(
            params,
            well,
            risk_threshold,
        )

        if not result["feasible"]:
            continue

        feasible_count += 1

        score = _score(
            result,
            baseline,
        )

        if score > best_score:

            best = result
            best_score = score

    # ---------------------------------------------------------
    # Safety fallback
    # ---------------------------------------------------------

    if best is None:

        best = baseline
        best_score = _score(
            baseline,
            baseline,
        )

        optimization_status = "NO_FEASIBLE_CANDIDATE"

    else:

        optimization_status = "OPTIMAL_FEASIBLE_SCENARIO_FOUND"

    # ---------------------------------------------------------
    # Return optimizer result
    # ---------------------------------------------------------

    return {
        "baseline": baseline,
        "best": best,
        "score": float(best_score),
        "n_candidates": len(candidates),
        "feasible_candidates": feasible_count,
        "risk_threshold": risk_threshold,
        "optimization_status": optimization_status,
    }