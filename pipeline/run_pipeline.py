"""End-to-end pipeline runner: parse -> preprocess -> features -> classify.

This is the single command that proves the whole software chain works on
synthetic data today, and will run unchanged on a recorded hardware session
tomorrow.

Sources (swappable):
    --synthetic <scenario>   generate a fresh synthetic session in-memory
    --session <path>         load a recorded session file (data/**/*.csv|parquet)

If trained models exist in ml/models/ they are used to classify each window;
otherwise the run still completes and reports the raw feature summary (so the
pipeline is verifiable before any model is trained).

Run:
    py -m pipeline.run_pipeline --synthetic walking
    py -m pipeline.run_pipeline --session data/synthetic/<id>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline.config import DEFAULT_CONFIG, PipelineConfig
from pipeline.features import FEATURE_COLUMNS, extract_features
from pipeline.parser import parse_csi_line
from pipeline.preprocess import preprocess
from pipeline.storage import amplitude_matrix, load_session

MODELS_DIR = Path(__file__).resolve().parents[1] / "ml" / "models"


def _amplitude_from_synthetic(scenario: str, config: PipelineConfig) -> np.ndarray:
    from ml.synthetic import generate_session

    sess = generate_session(scenario, duration_s=20.0,
                            sample_rate_hz=config.window.sample_rate_hz, seed=7)
    amps = [parse_csi_line(ln).amplitude for ln in sess.lines]
    return np.vstack(amps).astype(np.float32)


def _load_models():
    """Load classifiers if present; return dict (possibly empty)."""
    import joblib

    models = {}
    for name in ("presence", "fall", "counting"):
        p = MODELS_DIR / f"{name}.joblib"
        if p.exists():
            models[name] = joblib.load(p)
    return models


def _classify(features, models) -> None:
    from ml.counting import single_node_fallback_features
    from ml.fall import build_fall_model  # noqa: F401  (kept for discoverability)
    from ml.presence import presence_feature_matrix

    if "presence" in models:
        preds = models["presence"].predict(presence_feature_matrix(features))
        frac = float(np.mean(preds))
        verdict = "PRESENCE DETECTED" if frac > 0.5 else "empty"
        print(f"  presence: {verdict} ({frac*100:.0f}% of windows show motion)")
    if "fall" in models:
        from pipeline.features import feature_matrix

        fpred = models["fall"].predict(feature_matrix(features))
        n_fall = int(np.sum(fpred))
        print(f"  fall:     {'FALL DETECTED' if n_fall else 'no fall'} "
              f"({n_fall}/{len(fpred)} windows flagged)")
    if "counting" in models:
        from ml.counting import counting_feature_matrix

        two = single_node_fallback_features(features)
        cpred = models["counting"].predict(counting_feature_matrix(two))
        vals, counts = np.unique(cpred, return_counts=True)
        mode = int(vals[np.argmax(counts)])
        print(f"  counting: ~{mode} people (single-node fallback; per-window {dict(zip(vals.tolist(), counts.tolist()))})")
    if not models:
        print("  (no trained models in ml/models/ - run `py -m ml.train` to enable "
              "classification. Feature extraction below still validates the pipeline.)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the full CSI pipeline end to end.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", choices=["empty", "walking", "fall", "2people"],
                     help="generate a fresh synthetic session of this scenario")
    src.add_argument("--session", help="path to a recorded session .csv/.parquet")
    args = ap.parse_args()

    config = DEFAULT_CONFIG

    if args.synthetic:
        print(f"Source: synthetic '{args.synthetic}' session")
        amp = _amplitude_from_synthetic(args.synthetic, config)
    else:
        print(f"Source: recorded session {args.session}")
        df, meta = load_session(args.session)
        amp = amplitude_matrix(df)
        if meta:
            print(f"  label={meta.get('label')} node={meta.get('node_id')} "
                  f"source={meta.get('source')}")

    print(f"Parsed amplitude matrix: {amp.shape[0]} samples x {amp.shape[1]} subcarriers")
    denoised = preprocess(amp, config)
    print(f"Preprocessed matrix:     {denoised.shape[0]} x {denoised.shape[1]} "
          f"(null subcarriers dropped: {amp.shape[1] - denoised.shape[1]})")

    features = extract_features(denoised, config)
    print(f"Extracted {len(features)} feature windows x {len(FEATURE_COLUMNS)} features")
    summary = features[list(FEATURE_COLUMNS)].mean().round(3)
    print("Mean feature values across windows:")
    for name, val in summary.items():
        print(f"    {name:26s} {val}")

    print("Classification:")
    _classify(features, _load_models())
    print("\nPipeline run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
