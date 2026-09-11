"""
data/synthetic/generate_dynacard_data.py

Generates SYNTHETIC LABELED dynamometer card IMAGES for MODEL 4
(Failure/Anomaly Model), per the specification's "Data source strategy":
    "Primary: physics-simulated cards via rod-dynamics (Gibbs) simulator --
    this is your main data engine."

*** THESE CARD IMAGES ARE 100% SYNTHETIC. THEY ARE NOT REAL BAGHEWALA
    DYNAMOMETER CARDS. ***

Pipeline:
    1. For each of the 5 classes (normal, rod_floating, gas_interference,
       pump_off, other_anomaly), generate many cards across a sweep of
       operating context (viscosity, SPM) and severity, using
       physics/rod_dynamics.py.
    2. Render each card as a small, clean grayscale image (load vs.
       position, closed loop, no axes/labels -- mimicking how a rendered
       card would look for CNN input) and save to
       data/synthetic/dynacard_data/images/.
    3. Save a labels.csv mapping image filename -> class, plus the
       operating-context columns (SPM, stroke_length_in) and card-derived
       features (as an optional feature-based fallback per spec).

Run:
    cd Baghewala_Digital_Twin
    python data/synthetic/generate_dynacard_data.py

Output:
    data/synthetic/dynacard_data/images/*.png   (64x64 grayscale)
    data/synthetic/dynacard_data/labels.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from physics.rod_dynamics import generate_card, card_derived_features, CLASS_NAMES

RNG_SEED = 45
N_PER_CLASS = 260
IMG_SIZE_PX = 64
OUT_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(OUT_DIR, "dynacard_data", "images")
LABELS_PATH = os.path.join(OUT_DIR, "dynacard_data", "labels.csv")

os.makedirs(IMAGES_DIR, exist_ok=True)


def render_card_image(card: dict, path: str, img_size_px: int = IMG_SIZE_PX):
    """Render a card as a small, clean grayscale image: just the closed
    load-vs-position loop, no axes/ticks/labels, fixed aspect."""
    fig = plt.figure(figsize=(img_size_px / 100, img_size_px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(card["position_in"], card["load_lbs"], color="black", linewidth=1.4)
    ax.fill(card["position_in"], card["load_lbs"], color="black", alpha=0.06)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def generate():
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    img_idx = 0

    for cls in CLASS_NAMES:
        for _ in range(N_PER_CLASS):
            viscosity_cP = float(np.exp(rng.uniform(np.log(50), np.log(30000))))
            spm = rng.uniform(2.5, 8.0)
            stroke_length_in = rng.uniform(64, 144)
            severity = 0.0 if cls == "normal" else rng.uniform(0.25, 1.0)

            card = generate_card(
                cls, viscosity_cP=viscosity_cP, spm=spm,
                stroke_length_in=stroke_length_in, rng=rng, severity=severity,
            )
            feats = card_derived_features(card)

            fname = f"card_{img_idx:05d}_{cls}.png"
            render_card_image(card, os.path.join(IMAGES_DIR, fname))

            rows.append({
                "filename": fname,
                "class_name": cls,
                "class_id": CLASS_NAMES.index(cls),
                "severity": round(severity, 3),
                "viscosity_cP": round(viscosity_cP, 2),
                "spm": round(spm, 3),
                "stroke_length_in": round(stroke_length_in, 2),
                "card_area": round(feats["card_area"], 2),
                "peak_load_lbs": round(feats["peak_load_lbs"], 2),
                "min_load_lbs": round(feats["min_load_lbs"], 2),
                "is_synthetic": True,
                "data_source": "physics_simulated_synthetic_dynacard",
            })
            img_idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(LABELS_PATH, index=False)
    print(f"Wrote {len(df)} card images to {IMAGES_DIR}")
    print(f"Wrote labels to {LABELS_PATH}")
    print(df["class_name"].value_counts())
    return df


if __name__ == "__main__":
    generate()
