"""
data/synthetic/generate_thermal_data.py

Generates SYNTHETIC training data for Model 1 (Thermal/Reservoir Model).

*** THIS DATA IS 100% SYNTHETIC. IT IS NOT REAL BAGHEWALA FIELD DATA. ***

Strategy
--------
1. Sample a population of "wells x cycles" with physically reasonable CSS
   parameters (steam volume, pressure, soak duration, rock properties,
   drawn from ranges consistent with the specification document).
2. Run the deterministic physics backbone (physics/thermal_physics.py) to
   get a "true physics" temperature trajectory for each well-cycle.
3. Add a synthetic "unmodeled real-world deviation" on top of the physics
   trajectory, to emulate the kind of heterogeneity/multi-cycle-drift error
   that Stage B (the ML correction model) is supposed to learn to predict.
   This deviation is built from:
     - a smooth per-cycle random bias (formation heterogeneity effect)
     - a slow drift term correlated with prior_cycle_count (multi-cycle
       thermal history effects not captured by the simple backbone)
     - i.i.d. Gaussian measurement-style noise
4. Save everything to thermal_data.csv with an explicit `is_synthetic`
   column and a `data_source` column so real data can later be appended /
   substituted without changing the schema.

Run:
    cd Baghewala_Digital_Twin
    python data/synthetic/generate_thermal_data.py

Output:
    data/synthetic/thermal_data.csv
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from physics.thermal_physics import (
    ThermalPhysicsParams,
    temperature_trajectory,
    temperature_to_viscosity_cP,
)

RNG_SEED = 42
N_WELL_CYCLES = 220          # number of distinct (well, cycle) combinations
TIMESTEPS_PER_CYCLE = 20     # sampled time points per cycle (soak+production)
OUT_PATH = os.path.join(os.path.dirname(__file__), "thermal_data.csv")


def sample_well_cycle_params(rng: np.random.Generator, well_id: int, cycle_num: int):
    """Sample one well-cycle's CSS/reservoir parameters within plausible ranges."""
    return dict(
        well_id=f"BGW-{well_id:03d}",
        cycle_num=cycle_num,
        steam_volume_bbl=rng.uniform(600, 2200),
        injection_pressure_psi=rng.uniform(150, 450),
        steam_quality_frac=rng.uniform(0.55, 0.85),
        soak_duration_days=rng.uniform(4, 12),
        thermal_conductivity_W_mK=rng.uniform(1.3, 2.4),
        porosity_frac=rng.uniform(0.20, 0.32),
        formation_thickness_m=rng.uniform(6, 20),
        baseline_reservoir_temp_C=rng.uniform(46, 48),
        crude_api_gravity=rng.uniform(17, 19),
        prior_cycle_count=cycle_num - 1,
    )


def synthetic_unmodeled_deviation(rng, n, prior_cycle_count, well_hash):
    """
    Emulate real-world deviation the physics backbone does NOT capture:
    per-well heterogeneity bias + multi-cycle drift + noise.
    Returns an array of degC deviations to ADD to the physics trajectory.
    """
    well_bias_C = ((well_hash % 97) / 97.0 - 0.5) * 6.0     # +/- 3 degC well-level bias
    cycle_drift_C = 0.4 * prior_cycle_count                  # thermal build-up drift
    noise = rng.normal(0, 1.4, size=n)                       # measurement-style noise
    return well_bias_C + cycle_drift_C + noise


def generate():
    rng = np.random.default_rng(RNG_SEED)
    params = ThermalPhysicsParams()
    rows = []

    well_id = 0
    for wc in range(N_WELL_CYCLES):
        if wc % rng.integers(2, 4) == 0:
            well_id += 1
        cycle_num = rng.integers(1, 6)

        p = sample_well_cycle_params(rng, well_id, cycle_num)

        t_days = np.sort(rng.uniform(0, 60, TIMESTEPS_PER_CYCLE))
        # crude production ramp for cum_production series (used only to
        # drive the physics backbone's production-phase extra decay term)
        prod_phase_start = p["soak_duration_days"]
        cum_prod = np.where(
            t_days > prod_phase_start,
            (t_days - prod_phase_start) * rng.uniform(8, 20),
            0.0,
        )

        physics_temp = temperature_trajectory(
            t_days=t_days,
            phase_days={"soak": p["soak_duration_days"]},
            baseline_temp_C=p["baseline_reservoir_temp_C"],
            steam_volume_bbl=p["steam_volume_bbl"],
            injection_pressure_psi=p["injection_pressure_psi"],
            steam_quality_frac=p["steam_quality_frac"],
            soak_duration_days=p["soak_duration_days"],
            thermal_conductivity_W_mK=p["thermal_conductivity_W_mK"],
            porosity_frac=p["porosity_frac"],
            formation_thickness_m=p["formation_thickness_m"],
            prior_cycle_count=p["prior_cycle_count"],
            cum_production_bbl_series=cum_prod,
            params=params,
        )

        deviation = synthetic_unmodeled_deviation(
            rng, len(t_days), p["prior_cycle_count"], well_hash=well_id * 31 + cycle_num
        )
        observed_temp = physics_temp + deviation
        observed_temp = np.clip(observed_temp, p["baseline_reservoir_temp_C"] - 1, 250)

        observed_visc = temperature_to_viscosity_cP(observed_temp)

        for i, t in enumerate(t_days):
            phase = "soak" if t <= prod_phase_start else "production"
            rows.append(
                {
                    "well_id": p["well_id"],
                    "cycle_num": p["cycle_num"],
                    "time_since_injection_end_days": round(float(t), 3),
                    "phase": phase,
                    "steam_volume_bbl": round(p["steam_volume_bbl"], 1),
                    "injection_pressure_psi": round(p["injection_pressure_psi"], 1),
                    "steam_quality_frac": round(p["steam_quality_frac"], 3),
                    "soak_duration_days": round(p["soak_duration_days"], 2),
                    "prior_cycle_count": p["prior_cycle_count"],
                    "thermal_conductivity_W_mK": round(p["thermal_conductivity_W_mK"], 3),
                    "porosity_frac": round(p["porosity_frac"], 3),
                    "formation_thickness_m": round(p["formation_thickness_m"], 2),
                    "baseline_reservoir_temp_C": round(p["baseline_reservoir_temp_C"], 2),
                    "crude_api_gravity": round(p["crude_api_gravity"], 2),
                    "cum_production_bbl": round(float(cum_prod[i]), 2),
                    "physics_temp_C": round(float(physics_temp[i]), 3),
                    "temperature_C": round(float(observed_temp[i]), 3),  # "observed" (synth)
                    "viscosity_cP": round(float(observed_visc[i]), 2),
                    "is_synthetic": True,
                    "data_source": "physics_simulated_synthetic",
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows ({df['well_id'].nunique()} wells, "
          f"{df[['well_id','cycle_num']].drop_duplicates().shape[0]} well-cycles) "
          f"to {OUT_PATH}")
    print(df.head(8).to_string())
    return df


if __name__ == "__main__":
    generate()
