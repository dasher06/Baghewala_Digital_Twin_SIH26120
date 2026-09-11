"""
inference/failure.py

MODEL 4 — Failure/Anomaly Model — INFERENCE MODULE.

Public API
----------
    predict_failure_risk(position_in, load_lbs) -> dict
        Classify a dynamometer card given RAW load-vs-position arrays
        (e.g. from a live SRP sensor / VFD controller). Internally renders
        the card to an image exactly as done for training, then runs the
        CNN.

    predict_failure_risk_from_image(image_path) -> dict
        Classify a card given an already-rendered image file path.

    predict_failure_risk_from_card_dict(card) -> dict
        Convenience wrapper accepting the same `card` dict shape produced
        by physics/rod_dynamics.py's generate_card() (used by tests / demo).

Returns a dict:
    {
        "predicted_class": str,
        "confidence": float,               # confidence of predicted_class
        "class_probabilities": {cls: prob, ...},
        "failure_risk_score": float,       # 0-1, continuous risk score
    }

failure_risk_score definition: 1 - P(class == "normal"). This gives a
continuous 0-1 score suitable for use as an optimizer constraint by Model 5
(e.g. FailureRisk < threshold), per the specification.
"""

import os
import sys
import json
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "failure")
sys.path.insert(0, BASE_DIR)

from training.train_failure import FailureCNN, IMG_SIZE  # reuse exact architecture/preprocessing
from physics.rod_dynamics import generate_card  # noqa: F401 (re-exported for convenience/demo)

_MODEL_PATH = os.path.join(MODEL_DIR, "failure_cnn.pt")
_CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

_model = None
_class_names = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model():
    global _model, _class_names
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"{_MODEL_PATH} not found. Run training/train_failure.py first."
            )
        with open(_CLASS_NAMES_PATH) as f:
            _class_names = json.load(f)
        _model = FailureCNN(len(_class_names))
        _model.load_state_dict(torch.load(_MODEL_PATH, map_location=_device))
        _model.to(_device)
        _model.eval()
    return _model, _class_names


def _render_card_to_array(position_in: np.ndarray, load_lbs: np.ndarray) -> np.ndarray:
    """Render raw (position, load) arrays to the SAME kind of clean image
    representation used during training, without touching disk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    fig = plt.figure(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(position_in, load_lbs, color="black", linewidth=1.4)
    ax.fill(position_in, load_lbs, color="black", alpha=0.06)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = 1.0 - arr
    return arr


def _predict_from_array(arr: np.ndarray) -> dict:
    model, class_names = _load_model()
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(_device)  # (1,1,H,W)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    predicted_class = class_names[pred_idx]
    class_probabilities = {c: round(float(p), 4) for c, p in zip(class_names, probs)}

    normal_idx = class_names.index("normal") if "normal" in class_names else None
    p_normal = probs[normal_idx] if normal_idx is not None else (1 - probs[pred_idx])
    failure_risk_score = round(float(1.0 - p_normal), 4)

    return {
        "predicted_class": predicted_class,
        "confidence": round(float(probs[pred_idx]), 4),
        "class_probabilities": class_probabilities,
        "failure_risk_score": failure_risk_score,
    }


def predict_failure_risk(position_in, load_lbs) -> dict:
    """Classify a dynamometer card from raw load-vs-position arrays."""
    position_in = np.asarray(position_in, dtype=float)
    load_lbs = np.asarray(load_lbs, dtype=float)
    arr = _render_card_to_array(position_in, load_lbs)
    return _predict_from_array(arr)


def predict_failure_risk_from_image(image_path: str) -> dict:
    """Classify a card from an already-rendered image file."""
    img = Image.open(image_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = 1.0 - arr
    return _predict_from_array(arr)


def predict_failure_risk_from_card_dict(card: dict) -> dict:
    """Convenience: accepts the dict shape returned by
    physics/rod_dynamics.py's generate_card()."""
    return predict_failure_risk(card["position_in"], card["load_lbs"])


if __name__ == "__main__":
    rng = np.random.default_rng(99)
    for cls in ["normal", "rod_floating", "gas_interference", "pump_off", "other_anomaly"]:
        card = generate_card(cls, viscosity_cP=2000, spm=5, stroke_length_in=100, rng=rng, severity=0.75)
        result = predict_failure_risk_from_card_dict(card)
        print(f"true={cls:18s} -> predicted={result['predicted_class']:18s} "
              f"(conf={result['confidence']:.3f})  risk={result['failure_risk_score']:.3f}")
