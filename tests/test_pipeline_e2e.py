"""End-to-end pipeline test: synthetic -> parse -> store -> preprocess ->
features -> train -> classify, all without errors."""

from __future__ import annotations

import numpy as np
import pytest

from ml.dataset import build_single_node_dataset, build_two_node_counting_dataset
from ml.presence import build_presence_model, presence_feature_matrix, to_presence_labels
from ml.synthetic import SCENARIOS, generate_session
from pipeline.config import DEFAULT_CONFIG
from pipeline.features import extract_features
from pipeline.parser import parse_csi_line
from pipeline.preprocess import preprocess
from pipeline.serial_reader import CSISerialReader, MockSerial
from pipeline.storage import (
    SessionWriter,
    amplitude_matrix,
    frames_to_dataframe,
    load_session,
)


def test_all_scenarios_generate_and_parse():
    for sc in SCENARIOS:
        sess = generate_session(sc, duration_s=5.0, seed=0)
        assert len(sess.lines) > 0
        frames = [parse_csi_line(ln) for ln in sess.lines]
        assert len(frames) == len(sess.lines)


def test_mock_serial_reader_yields_frames():
    sess = generate_session("walking", duration_s=2.0, seed=1)
    mock = MockSerial(sess.lines)
    reader = CSISerialReader(mock, host_timestamps=False)
    frames = list(reader.frames())
    assert len(frames) == len(sess.lines)
    assert reader.n_malformed == 0
    host_ts, frame = frames[0]
    assert frame.n_subcarriers == 64


def test_session_write_and_reload_roundtrip(tmp_path):
    sess = generate_session("empty", duration_s=3.0, seed=2)
    mock = MockSerial(sess.lines)
    reader = CSISerialReader(mock, host_timestamps=False)
    writer = SessionWriter(out_dir=tmp_path, label="empty", node_id="nodeX",
                           source="synthetic")
    for host_ts, frame in reader.frames():
        writer.append(host_ts, frame)
    meta = writer.close()

    assert writer.data_path.exists()
    assert writer.meta_path.exists()
    df, loaded_meta = load_session(writer.data_path)
    assert loaded_meta["label"] == "empty"
    assert loaded_meta["node_id"] == "nodeX"
    amp = amplitude_matrix(df)
    assert amp.shape[0] == len(sess.lines)
    assert amp.shape[1] == 64


def test_full_chain_parse_preprocess_features():
    sess = generate_session("2people", duration_s=10.0, seed=4)
    amp = np.vstack([parse_csi_line(ln).amplitude for ln in sess.lines])
    denoised = preprocess(amp, DEFAULT_CONFIG)
    feats = extract_features(denoised, DEFAULT_CONFIG, label="2people")
    assert len(feats) > 0
    assert np.all(np.isfinite(feats.select_dtypes("number").to_numpy()))


def test_dataset_build_and_train_smoke():
    """The whole ML training path must run on synthetic data without error."""
    single = build_single_node_dataset(DEFAULT_CONFIG, sessions_per_scenario=2,
                                       duration_s=8.0)
    assert set(single["label"].unique()) == set(SCENARIOS)
    # Presence model trains and predicts.
    X = presence_feature_matrix(single)
    y = to_presence_labels(single["label"])
    model = build_presence_model().fit(X, y)
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
    # Sanity: on synthetic data the model should beat trivial majority.
    assert (preds == y).mean() > 0.6


def test_two_node_counting_dataset_has_both_node_columns():
    two = build_two_node_counting_dataset(DEFAULT_CONFIG, sessions_per_count=2,
                                          duration_s=8.0)
    assert "count" in two.columns
    assert any(c.endswith("_n1") for c in two.columns)
    assert any(c.endswith("_n2") for c in two.columns)
    assert set(two["count"].unique()) <= {0, 1, 2}


def test_frames_to_dataframe_column_shape():
    sess = generate_session("walking", duration_s=2.0, seed=5)
    frames = [(float(i), parse_csi_line(ln)) for i, ln in enumerate(sess.lines)]
    df = frames_to_dataframe(frames)
    amp_cols = [c for c in df.columns if c.startswith("amp_")]
    phase_cols = [c for c in df.columns if c.startswith("phase_")]
    assert len(amp_cols) == 64
    assert len(phase_cols) == 64
