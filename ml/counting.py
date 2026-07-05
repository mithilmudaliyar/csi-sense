"""People counting - deliberately capped at 0, 1, or 2 people.

SCOPE CAP (see docs/limitations.md): this classifier distinguishes only
0 / 1 / 2 people. Counting beyond 2 with commodity CSI is unreliable and we
do NOT attempt it. The 3-class target reflects that honest limit.

The model consumes concatenated features from TWO RX nodes (spatial
diversity), matching the planned deployment: two ESP32 sniffers at different
points in the room. More people => more independent scatterers => higher
aggregate variance, more disturbance peaks, and different cross-node
structure. A gradient-boosted / random-forest classifier over the combined
feature vector handles this without hand-tuned rules.

If only one node is available at run time, `single_node_fallback_features`
duplicates the single node's features so the same model shape still applies
(with an accuracy penalty - documented, not hidden).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pipeline.features import FEATURE_COLUMNS

MAX_PEOPLE = 2  # HARD CAP - do not raise without real multi-person data
COUNT_CLASSES = (0, 1, 2)


def build_counting_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        class_weight="balanced",
        random_state=0,
        n_jobs=-1,
    )


def two_node_feature_columns() -> list[str]:
    """Canonical concatenated column order: <feature>_n1 then <feature>_n2."""
    return [f"{c}_n1" for c in FEATURE_COLUMNS] + [f"{c}_n2" for c in FEATURE_COLUMNS]


def counting_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = two_node_feature_columns()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Counting frame missing two-node columns: {missing[:4]}...")
    return df[cols].to_numpy(dtype=np.float64)


def single_node_fallback_features(single_node_df: pd.DataFrame) -> pd.DataFrame:
    """Build a two-node-shaped frame from ONE node by duplicating it.

    Documented degradation path for when only one RX node is online. Same
    model input shape, reduced spatial diversity -> expect lower accuracy.
    """
    n1 = single_node_df[list(FEATURE_COLUMNS)].add_suffix("_n1").reset_index(drop=True)
    n2 = single_node_df[list(FEATURE_COLUMNS)].add_suffix("_n2").reset_index(drop=True)
    return pd.concat([n1, n2], axis=1)


__all__ = [
    "MAX_PEOPLE",
    "COUNT_CLASSES",
    "build_counting_model",
    "two_node_feature_columns",
    "counting_feature_matrix",
    "single_node_fallback_features",
]
