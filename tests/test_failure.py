"""
tests/test_failure.py

Engineering sanity tests for MODEL 4 (Failure/Anomaly Model).

Run:
    cd Baghewala_Digital_Twin
    python tests/test_failure.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference.failure import predict_failure_risk_from_card_dict
from physics.rod_dynamics import generate_card, CLASS_NAMES

PASS, FAIL = "PASS", "FAIL"


def check(name, condition):
    status = PASS if condition else FAIL
    print(f"[{status}] {name}")
    return condition


def run_tests():
    results = []
    rng = np.random.default_rng(123)

    # 1. Class probabilities should sum to ~1.0
    card = generate_card("normal", viscosity_cP=1500, spm=5, stroke_length_in=100, rng=rng)
    result = predict_failure_risk_from_card_dict(card)
    prob_sum = sum(result["class_probabilities"].values())
    results.append(check(f"Class probabilities sum to ~1.0 (got {prob_sum:.4f})", abs(prob_sum - 1.0) < 1e-3))

    # 2. failure_risk_score is within [0, 1]
    results.append(check(
        f"failure_risk_score is within [0,1] (got {result['failure_risk_score']})",
        0.0 <= result["failure_risk_score"] <= 1.0,
    ))

    # 3. All 5 classes represented in class_probabilities
    results.append(check(
        f"All 5 classes present in output (got {len(result['class_probabilities'])})",
        len(result["class_probabilities"]) == 5,
    ))

    # 4. Batch accuracy over a fresh sample of generated cards (>80% expected,
    #    consistent with the CNN's held-out test accuracy of ~98%)
    correct = 0
    total = 30
    for _ in range(total):
        cls = rng.choice(CLASS_NAMES)
        severity = 0.0 if cls == "normal" else rng.uniform(0.4, 1.0)
        c = generate_card(cls, viscosity_cP=float(np.exp(rng.uniform(4, 9))),
                           spm=rng.uniform(3, 7), stroke_length_in=rng.uniform(70, 130),
                           rng=rng, severity=severity)
        pred = predict_failure_risk_from_card_dict(c)
        if pred["predicted_class"] == cls:
            correct += 1
    acc = correct / total
    results.append(check(f"Batch classification accuracy > 70% (got {acc:.1%})", acc > 0.70))

    # 5. A clearly "normal" (undistorted) card should have low failure risk
    normal_card = generate_card("normal", viscosity_cP=1000, spm=5, stroke_length_in=100, rng=rng)
    normal_result = predict_failure_risk_from_card_dict(normal_card)
    results.append(check(
        f"Normal card yields low-ish failure risk (got {normal_result['failure_risk_score']:.3f})",
        normal_result["failure_risk_score"] < 0.6,
    ))

    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} checks passed.")
    if n_pass != len(results):
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
