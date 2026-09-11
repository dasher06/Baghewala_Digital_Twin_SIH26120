"""
physics/thermal_physics.py

Deterministic PHYSICS BACKBONE for Model 1 (Thermal/Reservoir Model).

PROTOTYPE APPROXIMATION NOTICE
-------------------------------
This module implements a SIMPLIFIED, engineering-level version of Boberg-Lantz
style CSS thermal decay behaviour, not the full original Boberg-Lantz (1966)
paper with its exact heat-loss geometry, nor a full Marx-Langenheim areal
sweep solution. It is deliberately simplified so it can run as fast, direct
Python code (no external reservoir simulator) and be queried repeatedly by
Model 5 later.

Physical idea being modeled (heat balance during a CSS cycle):
  1. INJECTION: steam is injected into the near-wellbore zone. This zone is
     approximated as heated instantaneously to a "steam zone temperature"
     that depends on injection pressure (via steam saturation temperature).
  2. SOAK: no more heat is added. Heat leaks away from the heated zone
     radially into the surrounding (cooler) formation and to over/underlying
     strata. We model this leak-off with an exponential decay towards the
     original reservoir baseline temperature, similar in spirit to the decay
     term used in Boberg-Lantz-style engineering correlations. The DECAY RATE
     depends on thermal conductivity, formation thickness and (indirectly)
     on how much energy was pumped in relative to the heated rock volume.
  3. PRODUCTION: continued heat loss as above, PLUS accelerated cooling due
     to fluid withdrawal (cool fluids from further out in the reservoir are
     drawn towards the wellbore, and heat is carried out with produced
     fluids). We add a small additional decay term proportional to
     cumulative produced volume to represent this.

None of the constants below are calibrated against real Baghewala core or
log data (we do not have it yet). They are chosen so that:
  - baseline reservoir temperature ~46-48 degC is recovered at t -> infinity
  - steam zone peak temperatures after injection are physically plausible for
    saturated steam at typical CSS injection pressures (150-450 psi range for
    shallow heavy-oil CSS operations)
  - soak-to-production decay timescales are on the order of days to a few
    weeks, consistent with published CSS analog behaviour (Cold Lake /
    Athabasca / Orinoco literature cited in the specification document)

When real Baghewala injection/production/temperature-log data becomes
available, the CONSTANTS in `ThermalPhysicsParams` should be re-fit (e.g. by
nonlinear least squares against observed BHT/near-wellbore temperature logs)
without changing the function signatures below. Model 1's Stage B (ML
correction, see training/train_thermal.py) is precisely the mechanism that
lets the system absorb the parts of real behaviour that this simplified
backbone cannot capture, WITHOUT having to rewrite this physics module.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Steam saturation temperature approximation
# ---------------------------------------------------------------------------
def steam_saturation_temp_C(pressure_psi: float) -> float:
    """
    Approximate saturated-steam temperature (degC) as a function of pressure
    (psia), valid roughly over 50-1000 psi.

    PROTOTYPE APPROXIMATION: fitted log-curve to standard steam-table
    reference points (NOT a full steam-table lookup). Good to within a few
    degC over the CSS-relevant range, which is adequate for a prototype
    physics backbone.
    """
    p = max(pressure_psi, 14.7)  # floor at atmospheric
    # Antoine-like log fit calibrated to steam tables at
    # (100 psi -> 164.96C), (300 psi -> 214.85C), (600 psi -> 254.66C)
    return 91.5 * np.log(p) - 257.0


# ---------------------------------------------------------------------------
# Parameters (defaults are prototype engineering assumptions, see docstring)
# ---------------------------------------------------------------------------
@dataclass
class ThermalPhysicsParams:
    # Base decay rate (1/day) of heated-zone temperature back to baseline
    # during SOAK, before adjustment for conductivity/thickness/energy.
    base_decay_rate_per_day: float = 0.045

    # How strongly thermal conductivity accelerates decay.
    # Higher conductivity (W/m.K) -> faster heat leak-off -> higher decay rate.
    conductivity_sensitivity: float = 0.9

    # How strongly formation (net pay) thickness SLOWS decay.
    # Thicker pay -> more rock mass holds heat -> slower decay.
    thickness_sensitivity: float = 0.35
    reference_thickness_m: float = 10.0

    # How strongly injected steam volume (relative to a reference volume)
    # raises the initial heated-zone temperature rise above baseline.
    reference_steam_volume_bbl: float = 1000.0
    max_temp_rise_C: float = 140.0  # cap for numerical sanity

    # Extra decay acceleration during PRODUCTION due to fluid withdrawal,
    # scaled by cumulative produced volume relative to a reference volume.
    production_decay_sensitivity: float = 0.6
    reference_cum_production_bbl: float = 500.0

    # Prior-cycle thermal memory: each additional prior CSS cycle raises the
    # effective "residual heated" baseline slightly (cumulative thermal
    # buildup over multiple cycles), capped.
    cycle_memory_gain_C: float = 1.5
    cycle_memory_cap_C: float = 12.0


def initial_heated_zone_temp_C(
    baseline_temp_C: float,
    steam_volume_bbl: float,
    injection_pressure_psi: float,
    steam_quality_frac: float,
    prior_cycle_count: int,
    params: ThermalPhysicsParams = ThermalPhysicsParams(),
) -> float:
    """
    Peak near-wellbore temperature immediately at end of injection.

    Approximated as: baseline + fraction-of-(saturation_temp - baseline),
    where the fraction grows with steam volume (more steam -> more of the
    near-wellbore zone reaches close to saturation temperature) and with
    steam quality (higher quality steam carries more latent heat per unit
    mass injected).
    """
    t_sat = steam_saturation_temp_C(injection_pressure_psi)
    max_possible_rise = max(t_sat - baseline_temp_C, 0.0)

    vol_ratio = steam_volume_bbl / params.reference_steam_volume_bbl
    # Saturating response: more steam helps, with diminishing returns
    volume_fraction = 1.0 - np.exp(-1.1 * vol_ratio)

    quality_fraction = 0.5 + 0.5 * np.clip(steam_quality_frac, 0.0, 1.0)

    rise = max_possible_rise * volume_fraction * quality_fraction
    rise = min(rise, params.max_temp_rise_C)

    # multi-cycle thermal memory (near-wellbore rock never fully re-cools)
    memory_bonus = min(
        prior_cycle_count * params.cycle_memory_gain_C,
        params.cycle_memory_cap_C,
    )

    return baseline_temp_C + rise + memory_bonus


def effective_decay_rate_per_day(
    thermal_conductivity_W_mK: float,
    formation_thickness_m: float,
    params: ThermalPhysicsParams = ThermalPhysicsParams(),
) -> float:
    """
    Decay-rate (1/day) modifier from rock properties. Higher conductivity
    speeds up decay; thicker pay slows it down.
    """
    k_ref = 1.8  # W/m.K reference (typical sandstone-range value)
    k_factor = 1.0 + params.conductivity_sensitivity * (
        (thermal_conductivity_W_mK - k_ref) / k_ref
    )
    k_factor = max(k_factor, 0.3)

    thickness_factor = 1.0 / (
        1.0
        + params.thickness_sensitivity
        * (formation_thickness_m / params.reference_thickness_m - 1.0)
    )
    thickness_factor = max(thickness_factor, 0.3)

    return params.base_decay_rate_per_day * k_factor * thickness_factor


def temperature_trajectory(
    t_days: np.ndarray,
    phase_days: dict,
    baseline_temp_C: float,
    steam_volume_bbl: float,
    injection_pressure_psi: float,
    steam_quality_frac: float,
    soak_duration_days: float,
    thermal_conductivity_W_mK: float,
    porosity_frac: float,
    formation_thickness_m: float,
    prior_cycle_count: int,
    cum_production_bbl_series: np.ndarray | None = None,
    params: ThermalPhysicsParams = ThermalPhysicsParams(),
) -> np.ndarray:
    """
    Compute the PHYSICS-BACKBONE temperature trajectory (degC) at each time
    in `t_days`, where t_days=0 marks the END of injection (start of soak).

    Parameters
    ----------
    t_days : array of time-since-injection-end, in days (can be negative
             during injection if you want to model that phase too; this
             prototype focuses on soak+production, t_days >= 0).
    phase_days : dict with keys 'soak' (float) marking when production
                 starts (t_days >= soak_duration_days -> production phase).
    cum_production_bbl_series : optional array same length as t_days giving
                 cumulative produced volume at each timestep, used to add
                 extra decay acceleration during the production phase.
    porosity_frac : included for interface completeness / a future more
                 detailed rock-heat-capacity model; in this simplified
                 backbone it has a small direct effect via heat capacity
                 (higher porosity -> more pore fluid -> slightly higher
                 effective heat capacity -> marginally slower decay).

    Returns
    -------
    np.ndarray of predicted temperature (degC), same shape as t_days.
    """
    t_days = np.asarray(t_days, dtype=float)

    T_peak = initial_heated_zone_temp_C(
        baseline_temp_C,
        steam_volume_bbl,
        injection_pressure_psi,
        steam_quality_frac,
        prior_cycle_count,
        params,
    )

    decay_rate = effective_decay_rate_per_day(
        thermal_conductivity_W_mK, formation_thickness_m, params
    )
    # porosity effect on effective heat capacity (small, prototype-level)
    porosity_factor = 1.0 - 0.25 * (porosity_frac - 0.25)
    decay_rate = decay_rate * max(porosity_factor, 0.5)

    if cum_production_bbl_series is None:
        cum_production_bbl_series = np.zeros_like(t_days)
    else:
        cum_production_bbl_series = np.asarray(cum_production_bbl_series, dtype=float)

    production_extra_decay = (
        params.production_decay_sensitivity
        * cum_production_bbl_series
        / params.reference_cum_production_bbl
    ) / 100.0  # scaled small per-day contribution

    # Exponential relaxation toward baseline, with production phase getting
    # an additional (time-integrated, approximated stepwise) decay boost.
    rise_above_baseline = T_peak - baseline_temp_C
    temps = np.empty_like(t_days)
    cum_extra = 0.0
    prev_t = 0.0
    for i, t in enumerate(t_days):
        dt = max(t - prev_t, 0.0)
        cum_extra += production_extra_decay[i] * dt
        prev_t = t
        eff_rate = decay_rate + cum_extra
        temps[i] = baseline_temp_C + rise_above_baseline * np.exp(-eff_rate * max(t, 0.0))

    return temps


# ---------------------------------------------------------------------------
# Viscosity conversion: Andrade / Arrhenius-type correlation
# ---------------------------------------------------------------------------
@dataclass
class ViscosityCorrelationParams:
    """
    mu(T) = A * exp(B / (T_K))   (Andrade/Arrhenius form)

    A and B are calibrated (prototype-level, NOT Baghewala-specific) against
    2-3 illustrative heavy-oil reference points at ~17-19 API, in the spirit
    of published Cold Lake / Athabasca / Orinoco heavy-oil viscosity-
    temperature behaviour cited in the specification document:
        ~46 degC (baseline reservoir temp)  -> ~50,000 cP  (very viscous, cold)
        ~90 degC (moderate CSS heating)      -> ~800 cP
        ~150 degC (near steam-zone peak)     -> ~40 cP

    These are illustrative reference anchors for a 17-19 API crude, NOT
    measured Baghewala lab data. Replace with real PVT / viscometer data
    once available (see README "Replacing synthetic data" section).
    """
    A: float = 2.3e-9
    B: float = 9800.0
    min_viscosity_cP: float = 5.0
    max_viscosity_cP: float = 200000.0


def temperature_to_viscosity_cP(
    temp_C: np.ndarray, params: ViscosityCorrelationParams = ViscosityCorrelationParams()
) -> np.ndarray:
    """Andrade/Arrhenius-type temperature -> viscosity conversion (cP)."""
    temp_C = np.asarray(temp_C, dtype=float)
    T_K = temp_C + 273.15
    mu = params.A * np.exp(params.B / T_K)
    return np.clip(mu, params.min_viscosity_cP, params.max_viscosity_cP)


if __name__ == "__main__":
    # quick smoke test
    t = np.linspace(0, 60, 13)
    prod = np.linspace(0, 400, 13)
    temps = temperature_trajectory(
        t_days=t,
        phase_days={"soak": 7},
        baseline_temp_C=47.0,
        steam_volume_bbl=1200,
        injection_pressure_psi=280,
        steam_quality_frac=0.7,
        soak_duration_days=7,
        thermal_conductivity_W_mK=1.8,
        porosity_frac=0.27,
        formation_thickness_m=12.0,
        prior_cycle_count=2,
        cum_production_bbl_series=prod,
    )
    visc = temperature_to_viscosity_cP(temps)
    for ti, Ti, vi in zip(t, temps, visc):
        print(f"t={ti:5.1f}d  T={Ti:6.2f}C  visc={vi:9.1f}cP")
