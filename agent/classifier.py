"""
agent/classifier.py

Zero-shot risk classifier for insurance claims.
Uses sentence-transformers for NLI-based zero-shot classification.
Compatible with PyTorch 2.2.x (Intel Mac).

The model is loaded once and cached at module level.
Subsequent calls reuse the same pipeline instance.

Usage:
    from agent.classifier import classify_risk
    result = classify_risk("My car was stolen from the parking lot.")
    # {"risk_level": "high", "confidence": 0.94}
"""

import numpy as np
from sentence_transformers import CrossEncoder

# Descriptive hypotheses for NLI — more specific wording improves accuracy
# Mapped back to short labels for logging
RISK_HYPOTHESES = [
    "a serious claim involving major damage or large financial loss",
    "a moderate claim involving partial damage or average financial loss",
    "a minor claim involving small damage or low financial loss",
]

RISK_LABELS = ["high", "medium", "low"]

# Module-level cache — loaded once, reused on every call
_model = None


def _get_model():
    """
    Load the CrossEncoder model on first call, return cached instance
    on subsequent calls.

    Model: cross-encoder/nli-deberta-v3-small
    - 180MB
    - Strong NLI quality for zero-shot classification
    - Compatible with PyTorch 2.2.x
    """
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _model


def classify_risk(claim_text: str) -> dict:
    """
    Classify a claim description into high / medium / low risk.

    Args:
        claim_text: The raw claim description string.

    Returns:
        Dictionary with:
          - risk_level (str): "high", "medium", or "low"
          - confidence (float): model confidence, e.g. 0.94
    """
    model = _get_model()

    # Pair claim text with each descriptive hypothesis
    pairs = [[claim_text, hypothesis] for hypothesis in RISK_HYPOTHESES]
    scores = model.predict(pairs)

    # Model returns (n_labels, 3): [contradiction, neutral, entailment]
    # Take entailment column (index 2) as the relevance score per label
    entailment_scores = scores[:, 2]

    # Softmax to convert raw scores to probabilities
    probs = np.exp(entailment_scores) / np.exp(entailment_scores).sum()

    best_idx = int(probs.argmax())
    return {
        "risk_level": RISK_LABELS[best_idx],
        "confidence": round(float(probs[best_idx]), 3),
    }