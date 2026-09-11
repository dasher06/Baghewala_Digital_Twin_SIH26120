"""
data/synthetic/generate_srp_data.py

Generates SYNTHETIC training data for Model 3 (SRP Performance Model).

*** THIS DATA IS 100% SYNTHETIC. IT IS NOT REAL BAGHEWALA FIELD DATA. ***

Strategy (per specification's "Data source strategy" for Model 3):
    "Physics-simulated SRP response across a parameter sweep
    (viscosity x SPM x stroke)."

Pipeline:
    1. Sample well-completion parameters (rod string diameter/length, pump
       depth, tubing size) per synthetic well -- these are FIXED per well,
       consistent with the spec calling them "fixed per well."
    2. Sweep viscosity x SPM x stroke x VFD candidate settings (the
       "what-if" combinations Model 5 will later query).
    3. Run the API-RP-11L-style physics backbone (physics/srp_physics.py)
       to get "true physics" PPRL / efficiency / energy.
    4. Add a synthetic "unmodeled real-world deviation" on top (heavy-oil
       friction effects the simplified physics underrepresents -- this is
       exactly what Stage B's ML correction is meant to learn).
    5. Save to srp_data.csv.

Run:
    cd Baghewala_Digital_Twin
    python data/synthetic/generate_srp_data.py

Output:
    data/synthetic/srp_data.csv
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from physics.srp_physics import (
    SRPPhysicsParams,
    peak_polished_rod_load_lbs,
    volumetric_efficiency_frac,
    theoretical_pump_displacement_bopd,
    energy_consumption_kwh_per_bbl,
)

RNG_SEED = 44
N_WELLS = 25
SAMPLES_PER_WELL = 160
OUT_PATH = os.path.join(os.path.dirname(__file__), "srp_data.csv")


def sample_well_completion(rng: np.random.Generator, well_num: int):
    return dict(
        well_id=f"BGW-{well_num:03d}",
        rod_string_diameter_in=rng.choice([0.75, 0.875, 1.0]),
        rod_string_length_ft=rng.uniform(2800, 4800),
        pump_depth_ft=rng.uniform(3000, 5000),
        tubing_size_in=rng.choice([2.375, 2.875, 3.5]),
    )


def generate():
    rng = np.random.default_rng(RNG_SEED)
    params = SRPPhysicsParams()
    rows = []

    for w in range(1, N_WELLS + 1):
        completion = sample_well_completion(rng, w)

        for _ in range(SAMPLES_PER_WELL):
            viscosity_cP = float(np.exp(rng.uniform(np.log(50), np.log(40000))))  # log-uniform sweep
            spm = rng.uniform(2.5, 8.0)
            stroke_length_in = rng.uniform(64, 144)
            vfd_frequency_hz = rng.uniform(30, 60)
            fluid_level_m = rng.uniform(200, 1200)  # pump submergence proxy
            fluid_sg = rng.uniform(0.90, 1.00)

            pprl_physics = peak_polished_rod_load_lbs(
                spm=spm, stroke_length_in=stroke_length_in,
                rod_string_length_ft=completion["rod_string_length_ft"],
                rod_string_diameter_in=completion["rod_string_diameter_in"],
                pump_depth_ft=completion["pump_depth_ft"],
                viscosity_cP=viscosity_cP,
                fluid_specific_gravity=fluid_sg,
                params=params,
            )
            eff_physics = volumetric_efficiency_frac(
                viscosity_cP=viscosity_cP, spm=spm,
                rod_string_length_ft=completion["rod_string_length_ft"],
                pump_depth_ft=completion["pump_depth_ft"],
                params=params,
            )
            theo_bopd = theoretical_pump_displacement_bopd(spm, stroke_length_in, params)
            # VFD frequency scales effective pump speed relative to a 50Hz reference
            theo_bopd *= (vfd_frequency_hz / 50.0)
            actual_bopd_physics = theo_bopd * eff_physics
            kwh_bbl_physics, kwh_stroke_physics = energy_consumption_kwh_per_bbl(
                pprl_physics, stroke_length_in, spm, actual_bopd_physics, params
            )

            # --- synthetic unmodeled deviation (heavy-oil friction effects
            #     underrepresented by the simplified physics) ---
            visc_drag_bias = 0.06 * np.log1p(viscosity_cP / 1000.0)  # extra frac loss
            noise_eff = rng.normal(0, 0.015)
            eff_observed = np.clip(eff_physics - visc_drag_bias + noise_eff, 0.03, 0.97)

            pprl_noise = rng.normal(0, 120)
            pprl_observed = max(pprl_physics * (1 + 0.02 * np.log1p(viscosity_cP / 1000)) + pprl_noise, 500)

            actual_bopd_observed = theo_bopd * eff_observed
            kwh_bbl_observed, kwh_stroke_observed = energy_consumption_kwh_per_bbl(
                pprl_observed, stroke_length_in, spm, actual_bopd_observed, params
            )

            rows.append({
                "well_id": completion["well_id"],
                "viscosity_cP": round(viscosity_cP, 2),
                "fluid_level_m": round(fluid_level_m, 1),
                "spm": round(spm, 3),
                "stroke_length_in": round(stroke_length_in, 2),
                "vfd_frequency_hz": round(vfd_frequency_hz, 2),
                "rod_string_diameter_in": completion["rod_string_diameter_in"],
                "rod_string_length_ft": round(completion["rod_string_length_ft"], 1),
                "pump_depth_ft": round(completion["pump_depth_ft"], 1),
                "tubing_size_in": completion["tubing_size_in"],
                "fluid_specific_gravity": round(fluid_sg, 3),
                "pprl_physics_lbs": round(pprl_physics, 1),
                "efficiency_physics_frac": round(eff_physics, 4),
                "peak_polished_rod_load_lbs": round(pprl_observed, 1),  # "observed" (synth)
                "pump_efficiency_pct": round(eff_observed * 100, 2),
                "energy_kwh_per_bbl": round(kwh_bbl_observed, 4)
                    if np.isfinite(kwh_bbl_observed) else None,
                "energy_kwh_per_stroke": round(kwh_stroke_observed, 6),
                "actual_bopd": round(actual_bopd_observed, 2),
                "is_synthetic": True,
                "data_source": "physics_simulated_synthetic",
            })

    df = pd.DataFrame(rows).dropna(subset=["energy_kwh_per_bbl"])
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows ({df['well_id'].nunique()} wells) to {OUT_PATH}")
    print(df.head(8).to_string())
    return df


if __name__ == "__main__":
    generate()
