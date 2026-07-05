"""Evaluate saved models on a FRESH synthetic test set (different seeds).

Run:
    py -m ml.evaluate

Loads ml/models/*.joblib and scores them on newly generated synthetic
sessions the models were not trained on. Still synthetic - still labeled as
validation-only, not real accuracy.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

from ml import SYNTHETIC_WARNING
from ml.counting import counting_feature_matrix
from ml.dataset import build_single_node_dataset, build_two_node_counting_dataset
from ml.fall import to_fall_labels
from ml.presence import presence_feature_matrix, to_presence_labels
from pipeline.config import DEFAULT_CONFIG
from pipeline.features import feature_matrix

MODELS_DIR = Path(__file__).resolve().parent / "models"


def main() -> int:
    config = DEFAULT_CONFIG
    for name in ("presence", "fall", "counting"):
        if not (MODELS_DIR / f"{name}.joblib").exists():
            print(f"Model {name}.joblib not found - run `py -m ml.train` first.")
            return 1

    presence = joblib.load(MODELS_DIR / "presence.joblib")
    fall = joblib.load(MODELS_DIR / "fall.joblib")
    counting = joblib.load(MODELS_DIR / "counting.joblib")

    # Fresh synthetic test data with unseen seeds.
    single = build_single_node_dataset(config, sessions_per_scenario=3, base_seed=9000)
    two = build_two_node_counting_dataset(config, sessions_per_count=4, base_seed=90000)

    print("=" * 70)
    print(" EVALUATION ON FRESH SYNTHETIC DATA (unseen seeds)")
    print(f" {SYNTHETIC_WARNING}")
    print("=" * 70)

    yp = to_presence_labels(single["label"])
    acc_p = accuracy_score(yp, presence.predict(presence_feature_matrix(single)))
    print(f"\nPresence accuracy: {acc_p:.3f}")
    print("confusion (rows=true empty/present):")
    print(confusion_matrix(yp, presence.predict(presence_feature_matrix(single))))

    yf = to_fall_labels(single["label"])
    acc_f = accuracy_score(yf, fall.predict(feature_matrix(single)))
    print(f"\nFall accuracy: {acc_f:.3f}")
    print("confusion (rows=true not-fall/fall):")
    print(confusion_matrix(yf, fall.predict(feature_matrix(single))))

    yc = two["count"].to_numpy()
    acc_c = accuracy_score(yc, counting.predict(counting_feature_matrix(two)))
    print(f"\nCounting (0/1/2) accuracy: {acc_c:.3f}")
    print("confusion (rows=true 0/1/2):")
    print(confusion_matrix(yc, counting.predict(counting_feature_matrix(two))))

    print("\n" + SYNTHETIC_WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
