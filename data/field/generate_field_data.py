"""Generate the multi-well synthetic field layer for the SIH prototype.

IMPORTANT: every value produced by this script is synthetic and is not an
actual Baghewala well measurement or coordinate.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "field"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2026
N_WELLS = 52
N_OPERATIONAL = 33
rng = np.random.default_rng(SEED)

# Schematic field positions only; deliberately not real coordinates.
well_ids = [f"BGW-{i:03d}" for i in range(1, N_WELLS + 1)]
status = ["operational"] * N_OPERATIONAL + ["non_producing"] * (N_WELLS - N_OPERATIONAL)
rng.shuffle(status)

rows = []
for i, wid in enumerate(well_ids):
    operational = status[i] == "operational"
    rows.append({
        "well_id": wid,
        "status": status[i],
        "field_x_km": round(float(rng.uniform(0, 14)), 3),
        "field_y_km": round(float(rng.uniform(0, 14)), 3),
        "injection_pressure_psi": round(float(rng.uniform(180, 260)), 1),
        "steam_quality_frac": round(float(rng.uniform(0.78, 0.92)), 3),
        "prior_cycle_count": int(rng.integers(1, 7)),
        "thermal_conductivity_W_mK": round(float(rng.uniform(1.7, 2.6)), 3),
        "porosity_frac": round(float(rng.uniform(0.24, 0.32)), 3),
        "formation_thickness_m": round(float(rng.uniform(9, 16)), 2),
        "baseline_reservoir_temp_C": round(float(rng.uniform(43, 51)), 1),
        "crude_api_gravity": round(float(rng.uniform(17, 19)), 2),
        "fluid_level_m": round(float(rng.uniform(400, 900)), 1),
        "rod_string_diameter_in": float(rng.choice([0.75, 0.875, 1.0])),
        "rod_string_length_ft": round(float(rng.uniform(3000, 4600)), 1),
        "pump_depth_ft": round(float(rng.uniform(3300, 4500)), 1),
        "tubing_size_in": float(rng.choice([2.375, 2.875, 3.5])),
        "fluid_specific_gravity": round(float(rng.uniform(0.92, 0.99)), 3),
        "water_cut_frac": round(float(rng.uniform(0.08, 0.28)), 3),
        "cycle_num": int(rng.integers(2, 8)),
        "lagged_bopd": round(float(rng.uniform(20, 70)), 2),
        "arps_qi": round(float(rng.uniform(35, 80)), 2),
        "is_synthetic": True,
        "data_source": "synthetic_field_prototype",
    })
wells = pd.DataFrame(rows)
wells.to_csv(OUT / "wells.csv", index=False)

active = wells[wells.status == "operational"].copy()
current_rows = []
for _, w in active.iterrows():
    current_rows.append({
        "timestamp": "2026-08-26 08:00:00",
        "well_id": w.well_id,
        "steam_volume_bbl": round(float(rng.uniform(700, 1300)), 1),
        "soak_duration_days": round(float(rng.uniform(7, 17)), 1),
        "spm": round(float(rng.uniform(3.75, 6.75) / 0.25) * 0.25, 2),
        "stroke_length_in": round(float(rng.choice(np.arange(70, 116, 5))), 1),
        "vfd_frequency_hz": round(float(rng.uniform(37, 49)), 1),
        "is_synthetic": True,
        "data_source": "synthetic_field_prototype",
    })
operating = pd.DataFrame(current_rows)
operating.to_csv(OUT / "operating_data.csv", index=False)

# 30 daily historical records per operational well.
history_days = pd.date_range("2026-07-28", "2026-08-26", freq="D")
css_rows, prod_rows, srp_rows = [], [], []
for _, w in active.iterrows():
    base_steam = float(rng.uniform(750, 1250))
    base_spm = float(rng.uniform(4.0, 6.5))
    for d in history_days:
        cycle = int(w.cycle_num + ((d.dayofyear - history_days[0].dayofyear) // 7))
        steam = max(600, base_steam + rng.normal(0, 55))
        soak = float(np.clip(rng.normal(12, 2), 6, 18))
        temp = float(w.baseline_reservoir_temp_C + rng.uniform(5, 20))
        prod = float(np.clip(w.lagged_bopd * rng.uniform(0.10, 0.45), 1.0, 25.0))
        spm = float(np.clip(base_spm + rng.normal(0, 0.35), 3.5, 7.0))
        stroke = float(rng.choice(np.arange(70, 116, 5)))
        vfd = float(np.clip(42 + rng.normal(0, 3), 35, 50))
        css_rows.append({"timestamp": d, "well_id": w.well_id, "cycle_num": cycle, "steam_volume_bbl": round(steam,1), "injection_pressure_psi": w.injection_pressure_psi, "steam_quality_frac": w.steam_quality_frac, "soak_duration_days": round(soak,1), "reservoir_temperature_C": round(temp,1), "is_synthetic": True})
        prod_rows.append({"timestamp": d, "well_id": w.well_id, "oil_BOPD": round(prod,2), "water_cut_frac": w.water_cut_frac, "total_liquid_BLPD": round(prod / max(1-w.water_cut_frac,0.1),2), "is_synthetic": True})
        srp_rows.append({"timestamp": d, "well_id": w.well_id, "spm": round(spm,2), "stroke_length_in": stroke, "vfd_frequency_hz": round(vfd,1), "fluid_level_m": w.fluid_level_m, "pump_depth_ft": w.pump_depth_ft, "rod_string_diameter_in": w.rod_string_diameter_in, "rod_string_length_ft": w.rod_string_length_ft, "tubing_size_in": w.tubing_size_in, "is_synthetic": True})

pd.DataFrame(css_rows).to_csv(OUT / "css_history.csv", index=False)
pd.DataFrame(prod_rows).to_csv(OUT / "production_history.csv", index=False)
pd.DataFrame(srp_rows).to_csv(OUT / "srp_history.csv", index=False)

# A simple manifest tells future data engineers what each file is for.
manifest = pd.DataFrame([
    ["wells.csv", "52 well master records; 33 operational; synthetic schematic positions", "field master"],
    ["operating_data.csv", "current operating point per operational well", "Models 1-5 input layer"],
    ["css_history.csv", "daily CSS/thermal history per operational well", "Model 1 / validation"],
    ["production_history.csv", "daily production history per operational well", "Model 2 / validation"],
    ["srp_history.csv", "daily SRP settings per operational well", "Model 3 / validation"],
], columns=["file", "description", "role"])
manifest.to_csv(OUT / "DATA_MANIFEST.csv", index=False)

print(f"Created {len(wells)} wells: {(wells.status == 'operational').sum()} operational, {(wells.status != 'operational').sum()} non-producing")
print(f"Output directory: {OUT}")
