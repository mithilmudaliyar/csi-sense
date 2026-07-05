"""Unit tests for preprocessing and feature-extraction correctness."""

from __future__ import annotations

import numpy as np
import pytest

from ml.synthetic import generate_session
from pipeline.features import (
    FEATURE_COLUMNS,
    extract_features,
    feature_matrix,
    window_features,
)
from pipeline.parser import parse_csi_line
from pipeline.preprocess import hampel_filter, preprocess, select_active_subcarriers


def _amp_from_scenario(scenario: str, config, seed: int = 3) -> np.ndarray:
    sess = generate_session(scenario, duration_s=20.0,
                            sample_rate_hz=config.window.sample_rate_hz, seed=seed)
    amp = np.vstack([parse_csi_line(ln).amplitude for ln in sess.lines]).astype(np.float32)
    return preprocess(amp, config)


def test_hampel_replaces_single_spike():
    x = np.ones(51, dtype=float)
    x[25] = 100.0  # gross outlier
    out = hampel_filter(x, window_size=5, n_sigmas=3.0)
    assert out[25] == pytest.approx(1.0, abs=1e-6)
    # Non-outlier samples are untouched.
    assert out[0] == pytest.approx(1.0, abs=1e-6)


def test_hampel_returns_new_array():
    x = np.random.default_rng(0).normal(size=(30, 4))
    out = hampel_filter(x)
    assert out is not x
    assert out.shape == x.shape


def test_select_active_drops_zero_columns():
    x = np.random.default_rng(0).normal(size=(50, 5))
    x[:, 2] = 0.0  # dead subcarrier
    out = select_active_subcarriers(x)
    assert out.shape[1] == 4


def test_window_features_has_all_columns():
    sig = np.random.default_rng(0).normal(size=(100, 8))
    feats = window_features(sig, sample_rate_hz=50.0)
    for col in FEATURE_COLUMNS:
        assert col in feats


def test_walking_has_more_variance_than_empty(default_config):
    empty = _amp_from_scenario("empty", default_config)
    walking = _amp_from_scenario("walking", default_config)
    fe = extract_features(empty, default_config)
    fw = extract_features(walking, default_config)
    assert fw["amp_var"].mean() > fe["amp_var"].mean()


def test_fall_has_large_spike_and_stillness(default_config):
    fall = _amp_from_scenario("fall", default_config)
    empty = _amp_from_scenario("empty", default_config)
    ff = extract_features(fall, default_config)
    ef = extract_features(empty, default_config)
    # A fall session should contain the sharpest single jump of the two.
    assert ff["amp_max_abs_diff"].max() > ef["amp_max_abs_diff"].max()
    # And measurable post-event stillness somewhere in the session.
    assert ff["post_event_stillness_s"].max() > 0.0


def test_feature_matrix_shape_and_order(default_config):
    walking = _amp_from_scenario("walking", default_config)
    df = extract_features(walking, default_config, label="walking")
    X = feature_matrix(df)
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert X.shape[0] == len(df)
    assert np.all(np.isfinite(X))


def test_short_signal_still_produces_one_window(default_config):
    short = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    df = extract_features(short, default_config)
    assert len(df) == 1


def test_labels_attached_when_given(default_config):
    walking = _amp_from_scenario("walking", default_config)
    df = extract_features(walking, default_config, label="walking")
    assert (df["label"] == "walking").all()
