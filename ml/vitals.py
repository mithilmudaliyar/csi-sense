"""EXPERIMENTAL Phase-5 vitals estimation (breathing + heart rate) from CSI.

================================================================================
EXPERIMENTAL — UNVALIDATED. Synthetic-only until real still-subject data
exists. Requires a still / near-still subject at close range. NOT a medical
device — never use these numbers for health decisions.
================================================================================

PC-side only, nothing on-chip. Operates on the WRAPPED CSI phase time-series
(and can fall back to amplitude) from a still subject.

The two vitals are NOT equally credible — the physics differs sharply on a
single-antenna 2.4 GHz ESP32:

- BREATHING (primary output): bandpass 0.1-0.5 Hz -> BPM via FFT peak +
  zero-crossing cross-check, reported only inside 6-30 BPM. Chest motion is
  ~4-12 mm — a strong CSI perturbation. Realistic expectation once real data
  exists: roughly +/-1-3 breaths/min for ONE still subject at 1-3 m
  line-of-sight. This is the credible half of the module.
- HEART RATE (research-frontier, UNRELIABLE on this hardware): bandpass
  0.8-2.0 Hz -> BPM inside 40-120 BPM. Cardiac chest motion is ~0.2-0.5 mm,
  about 20-50x weaker than breathing and buried under breathing harmonics
  and noise. On single-antenna 2.4 GHz ESP32s this is borderline-gimmick:
  easily corrupted by ANY motion. 60 GHz mmWave radar is the appropriate
  tool for reliable HR. Implemented for completeness; never present its
  output with the same confidence as breathing.

Preconditions for either number to mean anything: a SINGLE still subject,
<=~3 m, line-of-sight, no other motion in the room. Vitals do not work while
walking, through walls, or with multiple people. No medical use.

Every estimate carries a confidence/quality flag; refuse rather than guess.

Credit: the breathing/heart-rate bandpass split is the one idea borrowed
from RuView (github.com/ruvnet/RuView). Its pose-estimation, house-mapping,
and through-wall-imaging claims are explicitly NOT adopted here — they are
unsupported for this hardware and out of scope for this honest build.

Synthetic ground truth from ml/synthetic.py's still_vitals scenario proves
the DSP code recovers known rates. It says NOTHING about real-world accuracy
on ESP32 phase noise (CFO/SFO, boot phase offsets) — see docs/limitations.md.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal as sps

from pipeline.config import DEFAULT_CONFIG, VitalsConfig

logger = logging.getLogger(__name__)

EXPERIMENTAL_BANNER = (
    "EXPERIMENTAL — UNVALIDATED. Synthetic-only until real still-subject "
    "data exists. Requires ONE still subject, <=~3 m, line-of-sight, no "
    "other motion. NOT a medical device; do not use for health decisions."
)

# Breathing is the credible output; heart rate is not. State it everywhere.
BREATHING_NOTE = (
    "Primary vitals output. Chest motion ~4-12 mm is a strong CSI signal; "
    "realistic target once real data exists is ~+/-1-3 breaths/min for one "
    "still subject at 1-3 m line-of-sight."
)
HEART_CAVEAT = (
    "RESEARCH-FRONTIER / UNRELIABLE on single-antenna 2.4 GHz ESP32: cardiac "
    "chest motion is ~0.2-0.5 mm (20-50x weaker than breathing), buried under "
    "breathing harmonics and noise, and corrupted by any motion. 60 GHz "
    "mmWave radar is the appropriate tool for reliable heart rate."
)

# Agreement gate between FFT-peak and zero-crossing BPM estimates.
METHOD_AGREEMENT_FRAC = 0.15
# Phase columns with temporal std outside this range are unusable
# (dead null slots below, wrap-flipping noise above).
_MIN_COL_STD_RAD = 1e-4
_MAX_COL_STD_RAD = 1.0


@dataclass(frozen=True)
class BandEstimate:
    """One frequency band's rate estimate with quality accounting.

    ``confidence`` describes DSP signal quality only. ``reliability_note``
    carries the hardware-honesty framing: breathing is the credible output,
    heart rate is research-frontier and unreliable on this hardware — the
    two must never be presented symmetrically.
    """

    name: str                    # "breathing" | "heart"
    bpm: float | None            # reconciled estimate; None = refused
    bpm_fft: float | None
    bpm_zero_crossing: float | None
    peak_ratio: float            # FFT band peak power / median band power
    confidence: str              # "high" | "medium" | "low" (DSP quality only)
    reliability_note: str = ""   # BREATHING_NOTE or HEART_CAVEAT


@dataclass(frozen=True)
class VitalsResult:
    breathing: BandEstimate
    heart: BandEstimate
    sample_rate_hz: float
    duration_s: float
    n_subcarriers_used: int
    signal_kind: str             # "phase" | "amplitude"
    quality_ok: bool             # both bands produced a confident estimate
    banner: str = EXPERIMENTAL_BANNER


def _select_columns(prepared: np.ndarray, max_std: float) -> np.ndarray:
    """Keep columns whose detrended std is in a plausible signal range.

    No lenient fallback above ``max_std``: for phase, a column whose
    unwrapped/detrended std exceeds ~1 rad is wrap-flipping noise that
    unwrap has turned into a random walk (1/f^2 spectrum) — re-admitting
    it made pure noise read as a confident vital. Refuse instead.
    """
    stds = prepared.std(axis=0)
    keep = (stds > _MIN_COL_STD_RAD) & (stds < max_std)
    if not keep.any():
        return prepared[:, :0]
    return prepared[:, keep]


def _prepare(matrix: np.ndarray, signal_kind: str) -> tuple[np.ndarray, int]:
    """Unwrap (phase only) + detrend + collapse subcarriers to one signal."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    # Refuse-not-crash: real captures carry NaN/inf; detrend/sosfiltfilt reject
    # them. Drop any column with a non-finite sample rather than blow up.
    finite = np.isfinite(x).all(axis=0)
    x = x[:, finite]
    if x.shape[1] == 0:
        return np.zeros(x.shape[0]), 0
    if signal_kind == "phase":
        x = np.unwrap(x, axis=0)
    x = sps.detrend(x, axis=0)
    # The rad-based upper bound only makes sense for phase; amplitude is in
    # arbitrary CSI units where a large std is legitimate signal.
    max_std = _MAX_COL_STD_RAD if signal_kind == "phase" else np.inf
    x = _select_columns(x, max_std)
    if x.shape[1] == 0:
        return np.zeros(x.shape[0]), 0
    return x.mean(axis=1), int(x.shape[1])


def _bandpass(sig: np.ndarray, sr: float, band: tuple[float, float],
              order: int) -> np.ndarray | None:
    nyq = sr / 2.0
    lo, hi = band
    if hi >= nyq or lo <= 0:
        logger.warning("Band %s Hz not resolvable at %.1f Hz sampling", band, sr)
        return None
    sos = sps.butter(order, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return sps.sosfiltfilt(sos, sig)


def _fft_peak(sig: np.ndarray, sr: float, band: tuple[float, float],
              zero_pad_factor: int) -> tuple[float | None, float]:
    """Dominant in-band frequency (Hz) + in-band-peak / out-of-band-floor ratio.

    Significance MUST be measured on the UNFILTERED detrended spectrum against
    the OUT-OF-BAND noise floor. Measuring peak/median *inside* a bandpassed
    signal is meaningless: the filter's rolloff guarantees a large in-band
    ratio even for pure noise, so an empty room reads "high confidence". A real
    periodic vital shows an in-band peak that towers over the broadband floor;
    noise shows a ratio near 1.
    """
    n = sig.size
    if n < 8:
        return None, 0.0
    nfft = int(2 ** np.ceil(np.log2(n * max(1, zero_pad_factor))))
    windowed = sig * np.hanning(n)
    spec = np.abs(np.fft.rfft(windowed, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    # Out-of-band floor excludes DC/near-DC bins that detrend residue leaves.
    out_band = (~in_band) & (freqs > band[0] * 0.5)
    if not in_band.any() or not out_band.any():
        return None, 0.0
    peak_idx = int(np.argmax(spec[in_band]))
    peak_power = float(spec[in_band][peak_idx])
    floor = float(np.median(spec[out_band]))
    ratio = peak_power / floor if floor > 0 else 0.0
    return float(freqs[in_band][peak_idx]), ratio


def _zero_crossing_bpm(filtered: np.ndarray, sr: float) -> float | None:
    """Rate from sign-change count: crossings/2 cycles over the duration."""
    if filtered.size < 8:
        return None
    signs = np.sign(filtered)
    signs[signs == 0] = 1
    crossings = int(np.sum(signs[1:] != signs[:-1]))
    duration_s = filtered.size / sr
    if duration_s <= 0 or crossings == 0:
        return None
    return (crossings / 2.0) / duration_s * 60.0


def _estimate_band(
    sig: np.ndarray, sr: float, name: str,
    band: tuple[float, float], bpm_range: tuple[float, float],
    config: VitalsConfig,
) -> BandEstimate:
    note = HEART_CAVEAT if name == "heart" else BREATHING_NOTE
    filtered = _bandpass(sig, sr, tuple(band), config.filter_order)
    if filtered is None:
        return BandEstimate(name, None, None, None, 0.0, "low", note)

    # Peak significance from the UNFILTERED spectrum (see _fft_peak); the
    # filtered signal is used only for the zero-crossing cross-check.
    peak_hz, peak_ratio = _fft_peak(sig, sr, tuple(band),
                                    config.fft_zero_pad_factor)
    bpm_fft = peak_hz * 60.0 if peak_hz is not None else None
    bpm_zc = _zero_crossing_bpm(filtered, sr)

    lo, hi = bpm_range
    fft_ok = bpm_fft is not None and lo <= bpm_fft <= hi
    zc_ok = bpm_zc is not None and lo <= bpm_zc <= hi
    strong_peak = peak_ratio >= config.min_peak_ratio

    # A reported number REQUIRES the in-band peak to clear the out-of-band
    # noise floor (strong_peak). Method agreement only UPGRADES confidence — it
    # can never rescue a weak peak, otherwise filtered noise reads as a vital.
    if fft_ok and strong_peak:
        methods_agree = zc_ok and abs(bpm_fft - bpm_zc) <= METHOD_AGREEMENT_FRAC * bpm_fft
        conf = "high" if methods_agree else "medium"
        return BandEstimate(name, bpm_fft, bpm_fft, bpm_zc, peak_ratio, conf, note)
    # Refuse rather than report a junk number.
    return BandEstimate(name, None, bpm_fft, bpm_zc, peak_ratio, "low", note)


def _refused(sr: float, duration_s: float, n_used: int,
             signal_kind: str) -> VitalsResult:
    """A fully-refused result (both bands None). Refuse rather than guess."""
    return VitalsResult(
        BandEstimate("breathing", None, None, None, 0.0, "low", BREATHING_NOTE),
        BandEstimate("heart", None, None, None, 0.0, "low", HEART_CAVEAT),
        sr, duration_s, n_used, signal_kind, quality_ok=False,
    )


def estimate_vitals(
    matrix: np.ndarray,
    sample_rate_hz: float,
    config: VitalsConfig | None = None,
    signal_kind: str = "phase",
) -> VitalsResult:
    """Estimate breathing + heart rate from a CSI phase (or amplitude) matrix.

    EXPERIMENTAL — UNVALIDATED (see module banner). ``matrix`` is
    (n_samples, n_subcarriers); phase input is expected WRAPPED and is
    unwrapped internally. Returns refused (None) estimates rather than
    guesses when the record is too short or the spectrum is ambiguous.
    """
    logger.warning("ml.vitals: %s", EXPERIMENTAL_BANNER)
    config = config or DEFAULT_CONFIG.vitals
    if signal_kind not in ("phase", "amplitude"):
        raise ValueError("signal_kind must be 'phase' or 'amplitude'")

    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]
    duration_s = n / sample_rate_hz if sample_rate_hz > 0 else 0.0

    # Refuse BEFORE any DSP touches the data — detrend/unwrap raise on 0 rows.
    if n == 0 or duration_s < config.min_duration_s:
        logger.warning("Record too short (%.1fs < %.1fs) — refusing.",
                       duration_s, config.min_duration_s)
        return _refused(sample_rate_hz, duration_s, 0, signal_kind)

    sig, n_used = _prepare(matrix, signal_kind)
    if n_used == 0:
        logger.warning("No usable subcarriers — refusing.")
        return _refused(sample_rate_hz, duration_s, 0, signal_kind)

    breathing = _estimate_band(sig, sample_rate_hz, "breathing",
                               config.breathing_band_hz,
                               config.breathing_bpm_range, config)
    heart = _estimate_band(sig, sample_rate_hz, "heart",
                           config.heart_band_hz,
                           config.heart_bpm_range, config)
    # Honesty: quality gates on BREATHING only. Heart is unreliable on this
    # hardware; requiring a heart estimate made the honest outcome (breathing
    # ok, heart refused) report quality_ok=False while noise reported True.
    quality_ok = breathing.bpm is not None and breathing.confidence != "low"
    return VitalsResult(breathing, heart, sample_rate_hz, duration_s,
                        n_used, signal_kind, quality_ok=quality_ok)


def _print_band(est: BandEstimate, gt_bpm: float | None, tag: str) -> None:
    if est.bpm is None:
        print(f"  {est.name:9s} {tag}: REFUSED (confidence={est.confidence}, "
              f"fft={est.bpm_fft}, zc={est.bpm_zero_crossing}, "
              f"peak_ratio={est.peak_ratio:.1f})")
    else:
        zc = f"{est.bpm_zero_crossing:.1f}" if est.bpm_zero_crossing is not None else "n/a"
        line = (f"  {est.name:9s} {tag}: {est.bpm:6.1f} BPM  "
                f"(fft={est.bpm_fft:.1f}, zero-crossing={zc}, "
                f"peak_ratio={est.peak_ratio:.1f}, DSP confidence={est.confidence})")
        if gt_bpm is not None:
            line += f"  | synthetic ground truth {gt_bpm:.1f} BPM " \
                    f"(err {est.bpm - gt_bpm:+.1f})"
        print(line)
    if est.reliability_note:
        print(f"             ^ {est.reliability_note}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="EXPERIMENTAL vitals (breathing/HR) from a recorded CSI session.",
    )
    ap.add_argument("--session", required=True,
                    help="session .csv/.parquet (e.g. a still_vitals synthetic session)")
    ap.add_argument("--signal", choices=("phase", "amplitude"), default="phase",
                    help="which CSI component to analyze (default: phase)")
    args = ap.parse_args()

    from pipeline.storage import amplitude_matrix, load_session, phase_matrix

    banner_bar = "=" * 78
    print(banner_bar)
    print(EXPERIMENTAL_BANNER)
    print(banner_bar)

    df, meta = load_session(Path(args.session))
    sr = float(meta.get("sample_rate_hz_nominal", 50.0))
    matrix = phase_matrix(df) if args.signal == "phase" else amplitude_matrix(df)
    result = estimate_vitals(matrix, sr, signal_kind=args.signal)

    gt = (meta.get("extra") or {}).get("ground_truth") or {}
    print(f"session: {Path(args.session).name}")
    print(f"signal: {result.signal_kind}, {result.duration_s:.1f}s at "
          f"{result.sample_rate_hz:.0f} Hz nominal, "
          f"{result.n_subcarriers_used} subcarriers used")
    _print_band(result.breathing, gt.get("breathing_bpm"), "[PRIMARY]")
    _print_band(result.heart, gt.get("heart_bpm"), "[UNRELIABLE-ON-THIS-HW]")
    print(f"  quality_ok: {result.quality_ok}")
    if gt:
        print("  (ground truth is SYNTHETIC — recovering it proves the DSP "
              "math only, NOT real-world accuracy)")
    print(banner_bar)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
