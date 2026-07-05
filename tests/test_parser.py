"""Unit tests for CSI line parsing against real and mock lines."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.parser import (
    CSIParseError,
    is_csi_line,
    parse_csi_line,
    parse_stream,
)


def test_parses_real_example_line(real_example_line):
    frame = parse_csi_line(real_example_line)
    assert frame.role == "AP"
    assert frame.mac == "3C:71:BF:6D:2A:78"
    assert frame.rssi == -73
    assert frame.channel == 1
    # CONFIRMED quirk: `len` metadata says 384 (full buffer) but LLTF-only
    # capture prints just 128 int8 values -> 64 complex subcarriers.
    assert frame.advertised_len == 384
    assert frame.buf_len == 128
    assert frame.n_subcarriers == 64
    assert frame.csi.dtype == np.complex64


def test_len_field_is_advisory_not_a_hard_check(real_example_line):
    """A printed-count != len must NOT be rejected (real LLTF-only behavior)."""
    frame = parse_csi_line(real_example_line)
    assert frame.buf_len != frame.advertised_len  # 128 vs 384, and still valid


def test_amplitude_matches_tool_formula(real_example_line):
    """Amplitude must equal sqrt(imag^2 + real^2) with ESP-IDF [imag,real] order."""
    frame = parse_csi_line(real_example_line)
    # First slot buffer values are 101 (imag) and -48 (real).
    expected0 = float(np.sqrt(101.0**2 + (-48.0) ** 2))
    assert frame.amplitude[0] == pytest.approx(expected0, rel=1e-5)


def test_phase_matches_tool_formula(real_example_line):
    frame = parse_csi_line(real_example_line)
    expected0 = float(np.arctan2(101.0, -48.0))  # atan2(imag, real)
    assert frame.phase[0] == pytest.approx(expected0, rel=1e-5)


def test_is_csi_line_detection():
    assert is_csi_line("CSI_DATA,STA,AA:BB,...")
    assert not is_csi_line("I (1234) wifi: some esp-idf log line")
    assert not is_csi_line("")


def test_rejects_non_csi_line():
    with pytest.raises(CSIParseError):
        parse_csi_line("I (123) boot: normal log")


def test_rejects_missing_brackets():
    with pytest.raises(CSIParseError):
        parse_csi_line("CSI_DATA,STA,AA,-40,11,1,0,1,1,1,0,0,0,1,-90,0,6,0,1,0,10,0,0,1.0,4,1 2 3 4")


def test_accepts_len_mismatch_lltf_style():
    # Header claims len=384 but only 4 values present: accepted (advisory len).
    ok = "CSI_DATA,STA,AA:BB:CC:DD:EE:FF,-40,11,1,0,1,1,1,0,0,0,1,-90,0,6,0,1,0,10,0,0,1.0,384,[1 2 3 4 ]"
    frame = parse_csi_line(ok)
    assert frame.buf_len == 4
    assert frame.advertised_len == 384
    assert frame.n_subcarriers == 2


def test_rejects_odd_buffer_length():
    bad = "CSI_DATA,STA,AA:BB:CC:DD:EE:FF,-40,11,1,0,1,1,1,0,0,0,1,-90,0,6,0,1,0,10,0,0,1.0,3,[1 2 3 ]"
    with pytest.raises(CSIParseError):
        parse_csi_line(bad)


def test_parse_stream_skips_noise_keeps_valid(real_example_line):
    lines = [
        "I (100) wifi: log noise",
        real_example_line,
        "garbage that is not csi",
        real_example_line,
        "CSI_DATA,STA,truncated line without bracket",
    ]
    frames = parse_stream(lines)
    assert len(frames) == 2
    assert all(f.advertised_len == 384 for f in frames)
    assert all(f.buf_len == 128 for f in frames)


def test_synthetic_line_roundtrips_through_parser():
    """A synthetic line must parse and yield the expected subcarrier count."""
    from ml.synthetic import BUF_LEN, N_SUBCARRIERS, generate_session

    sess = generate_session("walking", duration_s=1.0, sample_rate_hz=50.0, seed=1)
    frame = parse_csi_line(sess.lines[0])
    assert frame.buf_len == BUF_LEN
    assert frame.n_subcarriers == N_SUBCARRIERS
    assert np.all(np.isfinite(frame.amplitude))
