"""Unit tests for EXPERIMENTAL vitals DSP (ml/vitals.py).

These assert the bandpass/FFT/zero-crossing math RECOVERS KNOWN SYNTHETIC
rates and — the point of the P0 fixes — that the module REFUSES rather than
guesses on bad input: short records, dead channels, pure noise, empty or
NaN-poisoned matrices. They say nothing about real-world accuracy on ESP32
phase noise — that needs hardware.
"""

from __future__ import annotations

import numpy as np

from ml.synthetic import VITALS_SCENARIO, generate_session
from ml.vitals import estimate_vitals
from pipeline.parser import parse_csi_line


def _phase_and_truth(seed: int = 7, duration_s: float = 30.0):
    sess = generate_session(VITALS_SCENARIO, duration_s=duration_s,
                            sample_rate_hz=50.0, seed=seed)
    phase = np.vstack([parse_csi_line(ln).phase for ln in sess.lines])
    return phase, sess.ground_truth


def test_recovers_synthetic_breathing_rate():
    phase, truth = _phase_and_truth()
    result = estimate_vitals(phase, 50.0, signal_kind="phase")
    assert result.breathing.bpm is not None
    # Breathing is the credible band; synthetic tone should land close.
    assert abs(result.breathing.bpm - truth["breathing_bpm"]) < 2.0
    assert result.quality_ok is True


def test_recovers_synthetic_heart_rate():
    phase, truth = _phase_and_truth()
    result = estimate_vitals(phase, 50.0, signal_kind="phase")
    assert result.heart.bpm is not None
    assert abs(result.heart.bpm - truth["heart_bpm"]) < 4.0


def test_fft_and_zero_crossing_cross_check_agree():
    phase, _ = _phase_and_truth()
    b = estimate_vitals(phase, 50.0, signal_kind="phase").breathing
    # The two independent estimators must corroborate for a reported number.
    assert b.bpm_fft is not None and b.bpm_zero_crossing is not None
    assert abs(b.bpm_fft - b.bpm_zero_crossing) < 0.15 * b.bpm_fft


def test_refuses_when_record_too_short():
    # 5s < min_duration_s (15s): must refuse BEFORE any DSP, not guess.
    phase, _ = _phase_and_truth(duration_s=5.0)
    result = estimate_vitals(phase, 50.0, signal_kind="phase")
    assert result.breathing.bpm is None
    assert result.heart.bpm is None
    assert result.quality_ok is False


def test_refuses_when_no_subcarrier_carries_signal():
    # Flat channel -> every column is dead after detrend -> no usable signal.
    flat = np.ones((1500, 51))
    result = estimate_vitals(flat, 50.0, signal_kind="phase")
    assert result.n_subcarriers_used == 0
    assert result.breathing.bpm is None
    assert result.quality_ok is False


def test_refuses_pure_gaussian_noise():
    # THE critical false-positive fix: an empty room (pure noise) must never
    # read as a confident vital. Small-sigma white noise is caught by the
    # out-of-band-floor peak_ratio gate; large-sigma phase noise becomes an
    # unwrap random walk and is caught by the column-std gate.
    for seed in (0, 1, 2, 3, 4):
        for scale in (0.3, 1.0):
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, scale, size=(1500, 51))
            result = estimate_vitals(noise, 50.0, signal_kind="phase")
            assert result.breathing.bpm is None, (seed, scale)
            assert result.quality_ok is False, (seed, scale)


def test_refuses_empty_matrix_without_raising():
    # Zero rows previously crashed inside detrend; must refuse instead.
    result = estimate_vitals(np.zeros((0, 4)), 50.0, signal_kind="phase")
    assert result.breathing.bpm is None
    assert result.heart.bpm is None
    assert result.quality_ok is False


def test_refuses_nan_matrix_without_raising():
    # Non-finite samples previously blew up in detrend/sosfiltfilt; NaN
    # columns are dropped and, with nothing left, the estimate is refused.
    result = estimate_vitals(np.full((1500, 4), np.nan), 50.0, signal_kind="phase")
    assert result.n_subcarriers_used == 0
    assert result.breathing.bpm is None
    assert result.quality_ok is False


def test_rejects_bad_signal_kind():
    try:
        estimate_vitals(np.zeros((100, 4)), 50.0, signal_kind="magic")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown signal_kind")
