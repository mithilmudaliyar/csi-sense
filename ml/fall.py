"""Fall detector.

A fall has a distinctive two-part signature in CSI amplitude:

    1. a sharp, large-amplitude disturbance (the body accelerating/impacting)
    2. followed by sudden, prolonged stillness (the body motionless after)

That maps directly onto the engineered features amp_max_abs_diff,
n_disturbance_peaks and post_event_stillness_s. A RandomForest over the
full window-feature vector captures the interaction between "big spike" and
"then still" without hand-tuned thresholds.

Fall is treated as binary per window: fall vs not-fall. Walking, empty and
2people are all negatives (no fall). Because falls are rare and the classes
are imbalanced, class_weight="balanced" is used.

--------------------------------------------------------------------------
OPTIONAL DEEP-LEARNING EXTENSION POINT (torch)
--------------------------------------------------------------------------
The engineered-feature RandomForest above is the default and needs no GPU.
For a raw-time-series model (1D-CNN or LSTM over the amplitude window), the
`build_torch_fall_model` factory below is a scaffold. torch is NOT a base
dependency - install it explicitly (`py -m pip install torch`) and pass
`--use-torch` where wired. See docs/decision-log.md for the rationale
(sklearn-first, torch strictly optional).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FALL_LABEL = "fall"


def to_fall_labels(labels: pd.Series) -> np.ndarray:
    """Binary: fall window -> 1, everything else -> 0."""
    return (labels.astype(str) == FALL_LABEL).astype(int).to_numpy()


def build_fall_model() -> RandomForestClassifier:
    """RandomForest over the full engineered window-feature vector."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=0,
        n_jobs=-1,
    )


# --------------------------------------------------------------------------
# Optional torch scaffold. Imported lazily so the base install never needs it.
# --------------------------------------------------------------------------
def build_torch_fall_model(window_len: int, n_subcarriers: int):  # pragma: no cover
    """EXTENSION POINT: a 1D-CNN over the raw amplitude window.

    Requires torch (optional extra). Returns an untrained nn.Module. Feed it
    (batch, n_subcarriers, window_len) amplitude tensors. This is intentionally
    minimal - a starting point for a raw-sequence fall model once real fall
    captures exist. It is NOT trained or evaluated by the default pipeline.
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # torch is an optional extra
        raise ImportError(
            "torch is an optional extra. Install it with `py -m pip install torch` "
            "to use the raw-sequence fall model. The default RandomForest path "
            "(build_fall_model) needs no torch."
        ) from exc

    class CNNFallDetector(nn.Module):
        def __init__(self, n_sub: int, win: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_sub, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

        def forward(self, x):  # x: (batch, n_sub, win)
            return self.net(x)

    return CNNFallDetector(n_subcarriers, window_len)


__all__ = ["FALL_LABEL", "to_fall_labels", "build_fall_model", "build_torch_fall_model"]
