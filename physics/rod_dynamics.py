"""
physics/rod_dynamics.py

Simplified rod-dynamics ("Gibbs wave-equation" style) SYNTHETIC DYNAMOMETER
CARD generator for MODEL 4 (Failure/Anomaly Model).

PROTOTYPE APPROXIMATION NOTICE
-------------------------------
The real Gibbs wave-equation method solves a damped wave equation along the
rod string to relate surface (polished-rod) load/position to DOWNHOLE
pump-card behaviour. That requires a finite-difference PDE solver over the
rod string.

For this prototype, we do NOT implement the full wave-equation PDE solver.
Instead we implement a PARAMETRIC, SHAPE-BASED generator that produces
load-vs-position curves with the qualitatively correct shape for each
failure mode (normal / rod floating / gas interference / pump-off / other
anomaly), parameterized by viscosity, SPM, valve-leakage severity, and
gas-interference ratio, per the specification's instruction that "the
physics lives [in generating training data], not in the loss function."

This module's job is ONLY to be a "physics-flavoured" synthetic-label data
engine for the CNN -- not a certified downhole-card simulator. When real or
public dynamometer card datasets become available, they should be used to
validate / fine-tune the CNN (see spec: "Supplement with a small set of
real/public dynamometer card datasets for validation").

Card shapes implemented
------------------------
- normal:            smooth, roughly parallelogram-ish loop; load rises on
                      upstroke, falls on downstroke, standard sucker-rod
                      pump card shape.
- rod_floating:       load fails to build properly during part of the
                      upstroke because the rod string is temporarily
                      unloaded (fluid pound / compression), producing a
                      flattened / dipped top-left region of the card.
- gas_interference:   gas in the pump chamber compresses instead of lifting
                      fluid, producing a "humped"/rounded bottom card with
                      reduced enclosed area and a characteristic bulge.
- pump_off:           fluid level below pump intake; load stays low and
                      flat for a large portion of the stroke (little to no
                      fluid load picked up), a thin/collapsed card.
- other_anomaly:      catch-all combining irregular/noisy load response
                      (e.g. valve leakage), producing a distorted,
                      asymmetric card shape.
"""

from __future__ import annotations
import numpy as np

CLASS_NAMES = ["normal", "rod_floating", "gas_interference", "pump_off", "other_anomaly"]


def _base_load_curve(position_frac: np.ndarray, peak_load: float, min_load: float) -> np.ndarray:
    """Smooth baseline parallelogram-ish card as a function of normalized
    stroke position [0,1] on the upstroke."""
    # smoothstep-based rise
    s = position_frac
    rise = 3 * s**2 - 2 * s**3  # smoothstep 0->1
    return min_load + (peak_load - min_load) * rise


def generate_card(
    class_name: str,
    viscosity_cP: float,
    spm: float,
    stroke_length_in: float,
    n_points: int = 200,
    rng: np.random.Generator = None,
    severity: float = None,
) -> dict:
    """
    Generate one synthetic dynamometer card (load vs. position) for the
    given failure class and operating context.

    Returns dict with:
        position_in : array of plunger position (inches), 0..stroke_length_in,
                       full up-then-down cycle (closed loop)
        load_lbs    : array of load (lbs), same length
        class_name  : str
        severity    : float in [0,1], severity of the anomaly (0 for normal)
    """
    if rng is None:
        rng = np.random.default_rng()
    if severity is None:
        severity = 0.0 if class_name == "normal" else rng.uniform(0.3, 1.0)

    n_half = n_points // 2
    pos_up = np.linspace(0, 1, n_half)
    pos_down = np.linspace(1, 0, n_half)

    # baseline load range scales mildly with viscosity and SPM (heavier oil,
    # faster pumping -> higher dynamic loads), consistent in spirit with
    # physics/srp_physics.py's dynamic-load behaviour.
    base_min_load = 4000 + 0.05 * viscosity_cP ** 0.5
    base_max_load = 9000 + 0.15 * viscosity_cP ** 0.5 + 40 * spm

    up_load = _base_load_curve(pos_up, base_max_load, base_min_load)
    down_load = _base_load_curve(pos_down, base_max_load, base_min_load) - 0.15 * (base_max_load - base_min_load)
    # downstroke sits below upstroke -> encloses an area (normal pump card)

    if class_name == "normal":
        pass  # keep as-is

    elif class_name == "rod_floating":
        # flatten/dip the load during part of the upstroke (fluid pound)
        dip_region = (pos_up > 0.55) & (pos_up < 0.85)
        up_load[dip_region] -= severity * 0.35 * (base_max_load - base_min_load) * \
            np.sin((pos_up[dip_region] - 0.55) / 0.3 * np.pi)

    elif class_name == "gas_interference":
        # rounded "hump" reducing enclosed area -- gas compresses instead of
        # lifting fluid over part of the stroke
        hump = severity * 0.45 * (base_max_load - base_min_load) * \
            np.sin(np.pi * pos_up) * np.exp(-((pos_up - 0.4) ** 2) / 0.08)
        up_load -= hump
        down_load += 0.3 * hump[::-1]

    elif class_name == "pump_off":
        # load stays low/flat over most of the stroke (little fluid picked up)
        flat_level = base_min_load + (1 - severity) * 0.3 * (base_max_load - base_min_load)
        up_load = np.minimum(up_load, flat_level + 200 * pos_up)
        down_load = np.minimum(down_load, flat_level + 100 * pos_down)

    elif class_name == "other_anomaly":
        # irregular/noisy distortion representing e.g. valve leakage
        distortion = severity * 0.25 * (base_max_load - base_min_load) * \
            rng.normal(0, 1, size=n_half).cumsum() / np.sqrt(n_half)
        up_load += distortion
        down_load += distortion[::-1] * 0.6

    # measurement-style noise on top of any class
    noise_scale = 0.01 * (base_max_load - base_min_load)
    up_load += rng.normal(0, noise_scale, size=n_half)
    down_load += rng.normal(0, noise_scale, size=n_half)

    load_lbs = np.concatenate([up_load, down_load])
    position_in = np.concatenate([pos_up, pos_down]) * stroke_length_in
    load_lbs = np.clip(load_lbs, 200, None)

    return {
        "position_in": position_in,
        "load_lbs": load_lbs,
        "class_name": class_name,
        "severity": float(severity),
    }


def card_derived_features(card: dict) -> dict:
    """
    Compute simple card-derived features (for the optional feature-based
    fallback mentioned in the spec): enclosed area (proxy via shoelace
    formula), peak load, min load.
    """
    x = card["position_in"]
    y = card["load_lbs"]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return {
        "card_area": float(area),
        "peak_load_lbs": float(np.max(y)),
        "min_load_lbs": float(np.min(y)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for cls in CLASS_NAMES:
        card = generate_card(cls, viscosity_cP=2000, spm=5, stroke_length_in=100, rng=rng)
        feats = card_derived_features(card)
        print(f"{cls:18s} area={feats['card_area']:9.1f}  peak={feats['peak_load_lbs']:8.1f}  "
              f"min={feats['min_load_lbs']:8.1f}")
