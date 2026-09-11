import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from optimizer.joint_optimizer import evaluate_scenario, optimize_well

FIELD_DIR = os.path.join(BASE_DIR, "data", "field")
WELLS_PATH = os.path.join(FIELD_DIR, "wells.csv")
OPERATING_PATH = os.path.join(FIELD_DIR, "operating_data.csv")

st.set_page_config(page_title="Baghewala Field Digital Twin", page_icon="🛢️", layout="wide")

# -----------------------------
# Data layer
# -----------------------------
@st.cache_data

def load_field_data():
    wells = pd.read_csv(WELLS_PATH)
    operating = pd.read_csv(OPERATING_PATH)
    operating["timestamp"] = pd.to_datetime(operating["timestamp"])
    return wells, operating

wells_df, operating_df = load_field_data()

@st.cache_data(show_spinner=False)
def evaluate_cached(well_id, params_tuple, well_tuple):
    params = dict(params_tuple)
    well = dict(well_tuple)
    return evaluate_scenario(params, well)


def well_dict(row):
    fields = [
        "injection_pressure_psi", "steam_quality_frac", "prior_cycle_count",
        "thermal_conductivity_W_mK", "porosity_frac", "formation_thickness_m",
        "baseline_reservoir_temp_C", "crude_api_gravity", "fluid_level_m",
        "rod_string_diameter_in", "rod_string_length_ft", "pump_depth_ft",
        "tubing_size_in", "fluid_specific_gravity", "water_cut_frac",
        "cycle_num", "lagged_bopd", "arps_qi",
    ]
    out = {}
    for f in fields:
        if f in row.index and pd.notna(row[f]):
            value = row[f]
            out[f] = int(value) if f == "prior_cycle_count" or f == "cycle_num" else float(value)
    return out


def operating_params(row):
    return {
        "steam_volume_bbl": float(row["steam_volume_bbl"]),
        "soak_duration_days": float(row["soak_duration_days"]),
        "spm": float(row["spm"]),
        "stroke_length_in": float(row["stroke_length_in"]),
        "vfd_frequency_hz": float(row["vfd_frequency_hz"]),
    }

# -----------------------------
# Header
# -----------------------------
st.title("🛢️ Baghewala Field Well-to-Surface Digital Twin")
st.caption("SIH Prototype | Multi-well CSS + SRP simulation, 3D well visualization and joint optimization")
st.warning("⚠️ Prototype only: field records and operating values in this package are synthetic/modelled and are NOT actual Baghewala field measurements or real well coordinates.")

# -----------------------------
# Field-level KPIs
# -----------------------------
active = wells_df[wells_df["status"] == "operational"]
nonprod = wells_df[wells_df["status"] != "operational"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total drilled wells", len(wells_df))
k2.metric("Operational wells", len(active))
k3.metric("Non-producing wells", len(nonprod))
k4.metric("Field area", "~200 km²")

# -----------------------------
# Well selector
# -----------------------------
operational_ids = active["well_id"].tolist()
selected_id = st.selectbox("Select operational well", operational_ids, index=0)
selected_well_row = active.loc[active["well_id"] == selected_id].iloc[0]
selected_operating_row = operating_df.loc[operating_df["well_id"] == selected_id].iloc[-1]
selected_well = well_dict(selected_well_row)
current_params = operating_params(selected_operating_row)

# -----------------------------
# Sidebar: current selected well inputs
# -----------------------------
st.sidebar.header(f"⚙️ Current Conditions — {selected_id}")
st.sidebar.caption("Prototype mode: values represent the selected well's current virtual state. In deployment these can come from SCADA/historian data.")

current_steam = st.sidebar.slider("Steam Volume (bbl)", 600.0, 1400.0, current_params["steam_volume_bbl"], 50.0)
current_soak = st.sidebar.slider("Soak Duration (days)", 6.0, 18.0, current_params["soak_duration_days"], 1.0)
current_spm = st.sidebar.slider("SPM", 3.5, 7.0, current_params["spm"], 0.25)
current_stroke = st.sidebar.slider("Stroke Length (in)", 60.0, 120.0, current_params["stroke_length_in"], 5.0)
current_vfd = st.sidebar.slider("VFD Frequency (Hz)", 35.0, 50.0, current_params["vfd_frequency_hz"], 1.0)

current_params = {
    "steam_volume_bbl": current_steam,
    "soak_duration_days": current_soak,
    "spm": current_spm,
    "stroke_length_in": current_stroke,
    "vfd_frequency_hz": current_vfd,
}

params_tuple = tuple(current_params.items())
well_tuple = tuple(selected_well.items())
with st.spinner(f"Loading {selected_id} virtual state..."):
    current_result = evaluate_cached(selected_id, params_tuple, well_tuple)

# -----------------------------
# Tabs
# -----------------------------
tab_field, tab_well, tab_3d, tab_thermal, tab_whatif, tab_opt = st.tabs([
    "🗺️ Field Overview", "📊 Well Digital Twin", "🧊 3D Well", "🌡️ Thermal & Production", "🔬 What-If", "🎯 Joint Optimization"
])

# -----------------------------
# Field overview
# -----------------------------
with tab_field:
    st.subheader("Baghewala Field — Multi-Well View")
    st.write("Each point represents a drilled well. Locations below are a synthetic schematic for the prototype; real coordinates can replace them without changing the dashboard architecture.")

    map_df = wells_df.copy()
    map_df["label"] = map_df.apply(lambda r: f"{r.well_id} — {r.status}", axis=1)
    fig = go.Figure()
    for status, marker_symbol in [("operational", "circle"), ("non_producing", "x")]:
        part = map_df[map_df["status"] == status]
        if len(part):
            fig.add_trace(go.Scatter(
                x=part["field_x_km"], y=part["field_y_km"], mode="markers+text",
                text=part["well_id"], textposition="top center", name=status.replace("_", " ").title(),
                customdata=part[["well_id", "status"]],
                marker=dict(size=10 if status == "operational" else 9, symbol=marker_symbol),
                hovertemplate="%{customdata[0]}<br>Status: %{customdata[1]}<extra></extra>"
            ))
    fig.update_layout(height=520, xaxis_title="Synthetic field X (km)", yaxis_title="Synthetic field Y (km)", legend_title="Well status", margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Field snapshot based on current virtual states.
    snapshots = []
    with st.spinner("Evaluating operational well snapshots..."):
        for _, row in active.iterrows():
            op = operating_df[operating_df["well_id"] == row["well_id"]].iloc[-1]
            params = operating_params(op)
            wd = well_dict(row)
            r = evaluate_scenario(params, wd)
            snapshots.append({
                "well_id": row["well_id"],
                "production_BOPD": r["predicted_BOPD"],
                "temperature_C": r["temperature_C"][-1],
                "viscosity_cP": r["viscosity_cP"][-1],
                "srp_efficiency_pct": r["pump_efficiency_pct"],
                "failure_risk_pct": r["failure_risk"] * 100,
                "energy_kwh_bbl": r["energy_kwh_per_bbl"],
            })
    snap = pd.DataFrame(snapshots)
    f1,f2,f3,f4 = st.columns(4)
    f1.metric("Virtual field production", f"{snap.production_BOPD.sum():.1f} BOPD")
    f2.metric("Avg. temperature", f"{snap.temperature_C.mean():.1f} °C")
    f3.metric("High-risk wells (>30%)", int((snap.failure_risk_pct > 30).sum()))
    f4.metric("Low-production wells (<5 BOPD)", int((snap.production_BOPD < 5).sum()))

    st.subheader("Operational Well Status")
    st.dataframe(snap.sort_values("failure_risk_pct", ascending=False).round(2), use_container_width=True, hide_index=True)

# -----------------------------
# Well digital twin
# -----------------------------
with tab_well:
    st.subheader(f"Current Virtual Well State — {selected_id}")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Production", f"{current_result['predicted_BOPD']:.2f} BOPD")
    c2.metric("Temperature", f"{current_result['temperature_C'][-1]:.1f} °C")
    c3.metric("Viscosity", f"{current_result['viscosity_cP'][-1]:.0f} cP")
    c4.metric("SRP Efficiency", f"{current_result['pump_efficiency_pct']:.1f}%")
    c5.metric("Failure Risk", f"{current_result['failure_risk']*100:.1f}%")
    st.divider()
    a,b,c = st.columns(3)
    a.metric("SOR Proxy", f"{current_result['sor_proxy']:.2f}")
    b.metric("Energy", f"{current_result['energy_kwh_per_bbl']:.2f} kWh/bbl")
    c.metric("Peak Rod Load", f"{current_result['peak_polished_rod_load_lbs']:.0f} lb")
    st.subheader("Selected Well Configuration")
    cfg = {"Well ID": selected_id, "Status": selected_well_row["status"], "Pump depth": f"{selected_well['pump_depth_ft']:.0f} ft", "Rod string": f"{selected_well['rod_string_length_ft']:.0f} ft / {selected_well['rod_string_diameter_in']:.3f} in", "Tubing": f"{selected_well['tubing_size_in']:.3f} in", "API gravity": f"{selected_well['crude_api_gravity']:.1f} °API"}
    st.dataframe(pd.DataFrame(list(cfg.items()), columns=["Parameter","Value"]), use_container_width=True, hide_index=True)

# -----------------------------
# 3D well visualization
# -----------------------------
with tab_3d:
    st.subheader(f"🧊 3D Schematic Well Twin — {selected_id}")
    st.info("Schematic 3D visualization, not to scale. Geometry is synthetic; the displayed temperature, viscosity, SPM, SRP efficiency and failure risk come from the selected well's virtual state.")
    depth = float(selected_well["pump_depth_ft"])
    rod_depth = np.linspace(0, depth, 80)
    x = np.zeros_like(rod_depth)
    y = np.zeros_like(rod_depth)
    z = -rod_depth
    fig3 = go.Figure()
    # tubing/wellbore
    fig3.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0,-depth], mode="lines", line=dict(width=12), name="Wellbore/Reservoir boundary"))
    # rod string
    fig3.add_trace(go.Scatter3d(x=x+0.08, y=y, z=z, mode="lines", line=dict(width=6), name="Sucker rod"))
    # pump
    fig3.add_trace(go.Scatter3d(x=[-0.25,0.25], y=[0,0], z=[-depth,-depth], mode="lines+markers", line=dict(width=14), marker=dict(size=8), name="SRP pump"))
    # reservoir zone
    rz = max(250.0, depth*0.18)
    fig3.add_trace(go.Mesh3d(x=[-1,1,1,-1,-1,1,1,-1], y=[-1,-1,1,1,-1,-1,1,1], z=[-depth-rz,-depth-rz,-depth-rz,-depth-rz,-depth,-depth,-depth,-depth], opacity=0.22, name="Heavy-oil reservoir", showscale=False))
    fig3.add_trace(go.Scatter3d(x=[0.45], y=[0], z=[-depth*0.55], mode="text", text=[f"{selected_id}"], showlegend=False))
    fig3.update_layout(height=650, scene=dict(xaxis_title="X (schematic)", yaxis_title="Y (schematic)", zaxis_title="Depth (ft)", zaxis=dict(autorange="reversed")), margin=dict(l=0,r=0,t=0,b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig3, use_container_width=True)
    q1,q2,q3,q4 = st.columns(4)
    q1.metric("Reservoir temperature", f"{current_result['temperature_C'][-1]:.1f} °C")
    q2.metric("Oil viscosity", f"{current_result['viscosity_cP'][-1]:.0f} cP")
    q3.metric("Pump speed", f"{current_spm:.2f} SPM")
    q4.metric("Failure risk", f"{current_result['failure_risk']*100:.1f}%")
    st.caption("The 3D view is a visualization layer over the numerical digital twin; it is not itself the physical/ML model.")

# -----------------------------
# Thermal & production
# -----------------------------
with tab_thermal:
    st.subheader("🌡️ Reservoir Thermal Behaviour")
    thermal_df = pd.DataFrame({"Day": current_result["times_days"], "Temperature (°C)": current_result["temperature_C"], "Viscosity (cP)": current_result["viscosity_cP"]})
    st.line_chart(thermal_df.set_index("Day")[["Temperature (°C)"]])
    st.subheader("🛢️ Oil Production Prediction")
    production_df = pd.DataFrame({"Day": current_result["times_days"], "Production (BOPD)": current_result["production_BOPD"]})
    st.line_chart(production_df.set_index("Day"))
    st.dataframe(production_df, use_container_width=True, hide_index=True)
    st.info("Model 1 predicts temperature and viscosity after CSS. Model 2 uses those conditions to estimate oil production; Model 3 predicts SRP performance and Model 4 estimates failure risk.")

# -----------------------------
# What-if
# -----------------------------
with tab_whatif:
    st.subheader("🔬 What-If Simulator")
    st.write(f"Test a different CSS + SRP operating point for **{selected_id}** without changing the current virtual well state.")
    c1,c2 = st.columns(2)
    with c1:
        whatif_steam = st.slider("What-If Steam Volume (bbl)",600.0,1400.0,current_steam,50.0,key="wi_steam")
        whatif_soak = st.slider("What-If Soak Duration (days)",6.0,18.0,current_soak,1.0,key="wi_soak")
    with c2:
        whatif_spm = st.slider("What-If SPM",3.5,7.0,current_spm,0.25,key="wi_spm")
        whatif_stroke = st.slider("What-If Stroke Length (in)",60.0,120.0,current_stroke,5.0,key="wi_stroke")
        whatif_vfd = st.slider("What-If VFD Frequency (Hz)",35.0,50.0,current_vfd,1.0,key="wi_vfd")
    whatif_params={"steam_volume_bbl":whatif_steam,"soak_duration_days":whatif_soak,"spm":whatif_spm,"stroke_length_in":whatif_stroke,"vfd_frequency_hz":whatif_vfd}
    if st.button("🚀 SIMULATE WHAT-IF SCENARIO", use_container_width=True):
        with st.spinner("Running Models 1–4..."):
            r=evaluate_scenario(whatif_params, selected_well)
        st.success("Simulation completed.")
        comparison=pd.DataFrame({"Metric":["Production (BOPD)","Temperature (°C)","Viscosity (cP)","SRP Efficiency (%)","Energy (kWh/bbl)","SOR Proxy","Failure Risk (%)"],"Current":[current_result["predicted_BOPD"],current_result["temperature_C"][-1],current_result["viscosity_cP"][-1],current_result["pump_efficiency_pct"],current_result["energy_kwh_per_bbl"],current_result["sor_proxy"],current_result["failure_risk"]*100],"What-If":[r["predicted_BOPD"],r["temperature_C"][-1],r["viscosity_cP"][-1],r["pump_efficiency_pct"],r["energy_kwh_per_bbl"],r["sor_proxy"],r["failure_risk"]*100]})
        st.dataframe(comparison.round(3), use_container_width=True, hide_index=True)
        wp=pd.DataFrame({"Day":r["times_days"],"Production (BOPD)":r["production_BOPD"]})
        st.line_chart(wp.set_index("Day"))

# -----------------------------
# Optimization
# -----------------------------
with tab_opt:
    st.subheader("🎯 Joint CSS + SRP Optimization")
    st.write(f"Model 5 searches operating combinations for **{selected_id}**, using that well's completion and fluid properties.")
    n_candidates=st.slider("Number of scenarios to evaluate",10,100,40,10)
    risk_threshold=st.slider("Maximum Failure Risk",0.10,0.80,0.45,0.05)
    if st.button("🏆 FIND OPTIMAL SETTINGS", use_container_width=True):
        with st.spinner(f"Evaluating {n_candidates} operating scenarios for {selected_id}..."):
            opt=optimize_well(well=selected_well,current=current_params,n_candidates=n_candidates,risk_threshold=risk_threshold)
        best=opt["best"]; baseline=opt["baseline"]
        st.success(opt["optimization_status"].replace("_"," ").title())
        m1,m2=st.columns(2); m1.metric("Feasible scenarios",opt["feasible_candidates"]); m2.metric("Optimizer score",f"{opt['score']:.3f}")
        c1,c2=st.columns(2)
        with c1:
            st.markdown("### ♨️ Recommended CSS")
            st.write(f"**Steam:** {best['params']['steam_volume_bbl']:.1f} bbl")
            st.write(f"**Soak:** {best['params']['soak_duration_days']:.1f} days")
        with c2:
            st.markdown("### ⚙️ Recommended SRP")
            st.write(f"**SPM:** {best['params']['spm']:.2f}")
            st.write(f"**Stroke:** {best['params']['stroke_length_in']:.1f} in")
            st.write(f"**VFD:** {best['params']['vfd_frequency_hz']:.1f} Hz")
        cmp=pd.DataFrame({"Metric":["Production (BOPD)","SRP Efficiency (%)","SOR Proxy","Energy (kWh/bbl)","Failure Risk (%)","Peak Rod Load (lb)"],"Current":[baseline["predicted_BOPD"],baseline["pump_efficiency_pct"],baseline["sor_proxy"],baseline["energy_kwh_per_bbl"],baseline["failure_risk"]*100,baseline["peak_polished_rod_load_lbs"]],"Optimized":[best["predicted_BOPD"],best["pump_efficiency_pct"],best["sor_proxy"],best["energy_kwh_per_bbl"],best["failure_risk"]*100,best["peak_polished_rod_load_lbs"]]})
        st.dataframe(cmp.round(3),use_container_width=True,hide_index=True)
        op=pd.DataFrame({"Day":best["times_days"],"Production (BOPD)":best["production_BOPD"]})
        st.line_chart(op.set_index("Day"))
        st.info("Optimization is based on synthetic prototype data and must be validated against real Baghewala constraints before field use.")

st.divider()
st.caption("Baghewala Field Digital Twin — SIH Prototype | 52 drilled / 33 operational well architecture | Synthetic field data | Models 1–5 + 3D visualization")
