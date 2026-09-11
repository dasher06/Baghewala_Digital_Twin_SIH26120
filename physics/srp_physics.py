"""
physics/srp_physics.py

Deterministic PHYSICS BACKBONE for Model 3 (SRP Performance Model).

PROTOTYPE APPROXIMATION NOTICE
-------------------------------
API RP 11L (American Petroleum Institute Recommended Practice 11L) is the
published industry-standard method for calculating sucker-rod pumping unit
loads, strokes, torque and horsepower from surface parameters (SPM, stroke
length, rod string, pump depth, fluid properties). The FULL API RP 11L
method uses a series of empirical dimensionless design charts/curves
(originally graphical, later curve-fit into polynomial approximations).

This module implements a SIMPLIFIED, polynomial/analytic approximation of
the key API RP 11L relationships -- good enough to produce physically
reasonable, monotonic, engineering-sane behaviour (loads/efficiency respond
correctly in direction and rough magnitude to SPM, stroke, depth, rod
weight and fluid load) for a PROTOTYPE digital twin. It is explicitly NOT a
certified, full API RP 11L implementation and should not be used for actual
mechanical design or rod-string sizing decisions.

Where real Baghewala SRP dynamometer/load-cell data becomes available, the
constants here (and eventually the ML correction layer in Model 3's Stage B)
should be recalibrated.

Core quantities modeled
------------------------
1. Peak Polished Rod Load (PPRL, lbs):
   PPRL = rod_weight_in_air + fluid_load - buoyancy_correction
          + dynamic_load_factor(SPM, stroke, rod_length)

2. Pump (volumetric) efficiency (%):
   Theoretical pump displacement (bbl/day) vs. the ACTUAL fluid volume the
   pump can move given viscosity-driven slippage/friction losses and rod
   stretch (reduces effective plunger stroke at the pump).

3. Energy consumption (kWh/bbl and kWh/stroke):
   From polished-rod work per stroke (load-distance) and surface
   motor/gearbox efficiency assumption.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

STEEL_DENSITY_LB_PER_IN3 = 0.2835
FLUID_SG_BASE = 0.95           # heavy oil + water mix, prototype default
GRAVITY_CONST = 1.0             # working in lbs (weight), not mass units


@dataclass
class SRPPhysicsParams:
    # Rod string weight per foot, lb/ft, for a "typical" sucker rod grade
    # used in this prototype (a real design would look this up per API rod
    # size/grade table -- simplified to a single average value here).
    rod_weight_lb_per_ft: float = 2.9

    # Plunger/pump cross-sectional area assumption (in^2), representative
    # of a common heavy-oil SRP pump bore. In a full design this would be
    # derived from the tubing/pump size input.
    plunger_area_in2: float = 3.0

    # Dynamic load factor sensitivity to SPM^2 * stroke (inertial + rod
    # stretch effects grow roughly with the square of pumping speed).
    dynamic_load_coefficient: float = 0.0009

    # Baseline volumetric efficiency (fraction) at LOW viscosity (light
    # fluid), before viscosity-driven de-rating.
    base_volumetric_efficiency: float = 0.92

    # How strongly high viscosity reduces volumetric efficiency (slippage,
    # incomplete barrel fillage due to slower fluid influx / gas effects).
    viscosity_efficiency_loss_coeff: float = 0.30
    reference_viscosity_cP: float = 300.0

    # Rod stretch effect: higher viscosity + deeper pumps + longer rods
    # increase effective rod stretch, reducing net plunger stroke at depth.
    rod_stretch_coeff: float = 1.8e-9  # (in stretch per (lb * ft) of rod*load)

    # Surface motor/gearbox efficiency assumption for energy calcs
    surface_mechanical_efficiency: float = 0.82


def peak_polished_rod_load_lbs(
    spm: float,
    stroke_length_in: float,
    rod_string_length_ft: float,
    rod_string_diameter_in: float,
    pump_depth_ft: float,
    viscosity_cP: float,
    fluid_specific_gravity: float = FLUID_SG_BASE,
    params: SRPPhysicsParams = SRPPhysicsParams(),
) -> float:
    """
    Simplified API-RP-11L-style Peak Polished Rod Load (PPRL, lbs).

    PPRL = W_rod_in_air + F_fluid - buoyancy + F_dynamic

    W_rod_in_air : total rod string weight (lb)
    F_fluid      : fluid load on the plunger area at pump depth (lb)
    buoyancy     : Archimedes buoyancy correction for rods submerged in
                   wellbore fluid (~ steel_SG-relative reduction)
    F_dynamic    : acceleration/inertial + viscous-drag dynamic load term,
                   growing with SPM^2, stroke, and rod string length, plus
                   an explicit viscosity-drag contribution (heavy oil adds
                   extra dynamic load that light-oil API RP 11L charts
                   underrepresent -- Stage B is meant to correct this
                   further using data).
    """
    rod_area_in2 = np.pi * (rod_string_diameter_in / 2.0) ** 2
    rod_weight_lb_per_ft = params.rod_weight_lb_per_ft * (rod_area_in2 / (np.pi * (0.875 / 2) ** 2))
    # normalize per-ft weight scaling roughly by rod cross-sectional area
    # relative to a common 7/8" reference rod, so bigger rods weigh more.

    w_rod_air = rod_weight_lb_per_ft * rod_string_length_ft

    buoyancy_factor = 1.0 - (fluid_specific_gravity / 7.85)  # steel SG ~7.85
    w_rod_buoyed = w_rod_air * buoyancy_factor

    fluid_load_lb = params.plunger_area_in2 * pump_depth_ft * 0.433 * fluid_specific_gravity
    # 0.433 psi/ft is the fresh-water hydrostatic gradient; scaled by SG.

    dynamic_load = (
        params.dynamic_load_coefficient
        * (spm ** 2)
        * stroke_length_in
        * (rod_string_length_ft / 1000.0)
    )

    # extra viscous drag term: heavy oil adds friction-driven dynamic load
    # not well captured by light-oil API RP 11L charts (prototype-level,
    # small relative to the ML correction Stage B will layer on top)
    viscous_drag_load = 0.015 * np.log1p(viscosity_cP) * stroke_length_in / 10.0

    pprl = w_rod_buoyed + fluid_load_lb + dynamic_load + viscous_drag_load
    return float(pprl)


def volumetric_efficiency_frac(
    viscosity_cP: float,
    spm: float,
    rod_string_length_ft: float,
    pump_depth_ft: float,
    params: SRPPhysicsParams = SRPPhysicsParams(),
) -> float:
    """
    Simplified pump volumetric efficiency (0-1), de-rated from a base
    efficiency by viscosity-driven slippage and rod-stretch effects.
    """
    visc_factor = 1.0 / (
        1.0
        + params.viscosity_efficiency_loss_coeff
        * np.log1p(viscosity_cP / params.reference_viscosity_cP)
    )

    # rod stretch grows with depth * rod length * pumping speed, reducing
    # effective plunger travel and thus volumetric efficiency slightly
    stretch_penalty = params.rod_stretch_coeff * pump_depth_ft * rod_string_length_ft * spm
    stretch_factor = 1.0 / (1.0 + stretch_penalty)

    eff = params.base_volumetric_efficiency * visc_factor * stretch_factor
    return float(np.clip(eff, 0.05, 0.98))


def theoretical_pump_displacement_bopd(
    spm: float, stroke_length_in: float, params: SRPPhysicsParams = SRPPhysicsParams()
) -> float:
    """
    Theoretical (100%-efficiency) pump displacement, BOPD (barrels/day).
    displacement = plunger_area * stroke * SPM * minutes_per_day / (in^3 per bbl)
    """
    in3_per_bbl = 9702.0  # 1 bbl = 9702 cubic inches
    strokes_per_day = spm * 60 * 24
    volume_in3_per_day = params.plunger_area_in2 * stroke_length_in * strokes_per_day
    return volume_in3_per_day / in3_per_bbl


def energy_consumption_kwh_per_bbl(
    pprl_lbs: float,
    stroke_length_in: float,
    spm: float,
    actual_bopd: float,
    params: SRPPhysicsParams = SRPPhysicsParams(),
) -> tuple[float, float]:
    """
    Simplified surface energy consumption estimate.

    Work per stroke (approx) = average_load * stroke_distance
    (using PPRL as a proxy for average dynamic load -- a real API RP 11L
    calc would use the full load-vs-position dynamometer card area; here we
    approximate average load as a fraction of peak load, a standard
    simplification for prototype energy estimates).

    Returns (kwh_per_bbl, kwh_per_stroke)
    """
    avg_load_fraction = 0.55  # typical avg/peak load ratio for a sucker-rod card
    stroke_ft = stroke_length_in / 12.0
    work_per_stroke_ftlb = pprl_lbs * avg_load_fraction * stroke_ft

    ftlb_to_kwh = 3.766e-7
    kwh_per_stroke = (work_per_stroke_ftlb * ftlb_to_kwh) / params.surface_mechanical_efficiency

    strokes_per_day = spm * 60 * 24
    kwh_per_day = kwh_per_stroke * strokes_per_day

    kwh_per_bbl = kwh_per_day / actual_bopd if actual_bopd > 1e-6 else float("inf")
    return float(kwh_per_bbl), float(kwh_per_stroke)


if __name__ == "__main__":
    params = SRPPhysicsParams()
    for visc in [200, 1000, 5000, 20000]:
        pprl = peak_polished_rod_load_lbs(
            spm=5.5, stroke_length_in=100, rod_string_length_ft=4000,
            rod_string_diameter_in=0.875, pump_depth_ft=3800, viscosity_cP=visc,
        )
        eff = volumetric_efficiency_frac(visc, spm=5.5, rod_string_length_ft=4000, pump_depth_ft=3800)
        theo_bopd = theoretical_pump_displacement_bopd(spm=5.5, stroke_length_in=100)
        actual_bopd = theo_bopd * eff
        kwh_bbl, kwh_stroke = energy_consumption_kwh_per_bbl(pprl, 100, 5.5, actual_bopd)
        print(f"visc={visc:6d}cP  PPRL={pprl:8.1f}lb  eff={eff*100:5.1f}%  "
              f"BOPD={actual_bopd:6.1f}  kWh/bbl={kwh_bbl:6.3f}")
