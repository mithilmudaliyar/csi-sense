"""Train presence, fall, and people-counting models on SYNTHETIC data.

Run:
    py -m ml.train                 # trains all three, saves to ml/models/
    py -m ml.train --quick         # fewer sessions, faster smoke run

Every metric printed is on synthetic data and is labeled as such. This
validates the code path only - see docs/limitations.md.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from ml import SYNTHETIC_WARNING
from ml.counting import build_counting_model, counting_feature_matrix
from ml.dataset import build_single_node_dataset, build_two_node_counting_dataset
from ml.fall import build_fall_model, to_fall_labels
from ml.presence import build_presence_model, presence_feature_matrix, to_presence_labels
from pipeline.config import DEFAULT_CONFIG, PipelineConfig
from pipeline.features import feature_matrix

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
MODELS_DIR = Path(__file__).resolve().parent / "models"


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print(f" {SYNTHETIC_WARNING}")
    print("=" * 70)


def train_presence(df, seed: int = 0):
    X = presence_feature_matrix(df)
    y = to_presence_labels(df["label"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    model = build_presence_model().fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    _banner("PRESENCE (empty vs occupied) - logistic regression")
    print(f"held-out accuracy: {acc:.3f}")
    print(classification_report(yte, model.predict(Xte), target_names=["empty", "present"], zero_division=0))
    return model, acc


def train_fall(df, seed: int = 0):
    X = feature_matrix(df)
    y = to_fall_labels(df["label"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    model = build_fall_model().fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    _banner("FALL DETECTION (fall vs not-fall) - random forest")
    print(f"held-out accuracy: {acc:.3f}")
    print(classification_report(yte, model.predict(Xte), target_names=["not-fall", "fall"], zero_division=0))
    return model, acc


def train_counting(df, seed: int = 0):
    X = counting_feature_matrix(df)
    y = df["count"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    model = build_counting_model().fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    _banner("PEOPLE COUNTING (0 / 1 / 2 ONLY) - random forest, 2 RX nodes")
    print(f"held-out accuracy: {acc:.3f}")
    print(classification_report(yte, model.predict(Xte), target_names=["0", "1", "2"], zero_division=0))
    return model, acc


def main() -> int:
    ap = argparse.ArgumentParser(description="Train CSI sensing models on synthetic data.")
    ap.add_argument("--quick", action="store_true", help="fewer sessions for a fast smoke run")
    ap.add_argument("--out", default=str(MODELS_DIR), help="output dir for saved models")
    args = ap.parse_args()

    config: PipelineConfig = DEFAULT_CONFIG
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_single = 3 if args.quick else 6
    n_count = 4 if args.quick else 8

    print(f"Building synthetic single-node dataset ({n_single}/scenario)...")
    single = build_single_node_dataset(config, sessions_per_scenario=n_single)
    print(f"  windows: {len(single)}  label counts: {single['label'].value_counts().to_dict()}")

    print(f"Building synthetic two-node counting dataset ({n_count}/count)...")
    counting = build_two_node_counting_dataset(config, sessions_per_count=n_count)
    print(f"  windows: {len(counting)}  count distribution: {counting['count'].value_counts().to_dict()}")

    presence_model, presence_acc = train_presence(single)
    fall_model, fall_acc = train_fall(single)
    counting_model, counting_acc = train_counting(counting)

    joblib.dump(presence_model, out / "presence.joblib")
    joblib.dump(fall_model, out / "fall.joblib")
    joblib.dump(counting_model, out / "counting.joblib")

    print("\n" + "-" * 70)
    print(f"Saved models to {out}/ (presence.joblib, fall.joblib, counting.joblib)")
    print(f"SYNTHETIC held-out accuracies: presence={presence_acc:.3f} "
          f"fall={fall_acc:.3f} counting={counting_acc:.3f}")
    print(SYNTHETIC_WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
