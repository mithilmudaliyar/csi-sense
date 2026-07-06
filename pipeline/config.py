"""Central pipeline configuration.

Everything toggleable lives here so preprocessing / windowing choices are
config-driven, not hardcoded. Load defaults, override via JSON file or
keyword arguments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class HampelConfig:
    enabled: bool = True
    window_size: int = 7      # samples on each side
    n_sigmas: float = 3.0


@dataclass(frozen=True)
class PCAConfig:
    enabled: bool = False     # optional denoiser, off by default
    n_components: int = 5     # components kept when reconstructing


@dataclass(frozen=True)
class WaveletConfig:
    enabled: bool = False     # optional denoiser, off by default
    wavelet: str = "db4"
    level: int = 3
    mode: str = "soft"


@dataclass(frozen=True)
class MotionConfig:
    """Motion-intensity meters + coarse-zone heuristic (dashboard view).

    HONESTY: intensity is a per-node signal-disturbance proxy ("activity is
    stronger near node 2"), NOT position. The zone call is an EXPERIMENTAL
    heuristic that needs per-room calibration to mean anything — it is not
    localization and never produces coordinates.
    """

    smoothing_windows: int = 3      # rolling mean over N feature windows
    dominance_margin: float = 0.30  # relative lead required to call a zone
    # ABSOLUTE amp_std below this = "no motion". Calibration knob: the default
    # separates the synthetic empty (~0.08) vs walking (~0.45) scenarios; tune
    # per room against an empty-room baseline before trusting the zone verdict.
    quiet_floor: float = 0.15


@dataclass(frozen=True)
class VitalsConfig:
    """EXPERIMENTAL Phase-5 vitals (breathing / heart rate) — UNVALIDATED.

    Bandpass bands follow the RuView-style breathing/HR split (that project's
    pose/mapping/through-wall claims are explicitly NOT adopted). Synthetic
    validation only until real still-subject captures exist.
    """

    breathing_band_hz: tuple[float, float] = (0.1, 0.5)
    heart_band_hz: tuple[float, float] = (0.8, 2.0)
    breathing_bpm_range: tuple[float, float] = (6.0, 30.0)
    heart_bpm_range: tuple[float, float] = (40.0, 120.0)
    filter_order: int = 4
    min_duration_s: float = 15.0     # below this, estimates are refused
    # In-band FFT peak vs OUT-OF-BAND median floor required to report a BPM.
    # White noise reads ~4-6 on this statistic (max of ~a dozen independent
    # exponential bins / median); a real periodic vital reads orders of
    # magnitude higher. 10.0 refuses noise with margin. Calibration knob.
    min_peak_ratio: float = 10.0
    fft_zero_pad_factor: int = 8     # finer peak interpolation on short records


@dataclass(frozen=True)
class WindowConfig:
    """Sliding-window parameters for feature extraction."""

    window_seconds: float = 2.0
    hop_seconds: float = 0.5
    sample_rate_hz: float = 50.0  # assumed CSI packet rate; measured later on hardware

    @property
    def window_samples(self) -> int:
        return max(1, int(round(self.window_seconds * self.sample_rate_hz)))

    @property
    def hop_samples(self) -> int:
        return max(1, int(round(self.hop_seconds * self.sample_rate_hz)))


@dataclass(frozen=True)
class PipelineConfig:
    hampel: HampelConfig = field(default_factory=HampelConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    vitals: VitalsConfig = field(default_factory=VitalsConfig)
    # Subcarrier selection: guard/null slots at buffer edges carry no signal.
    drop_null_subcarriers: bool = True

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            hampel=HampelConfig(**data.get("hampel", {})),
            pca=PCAConfig(**data.get("pca", {})),
            wavelet=WaveletConfig(**data.get("wavelet", {})),
            window=WindowConfig(**data.get("window", {})),
            motion=MotionConfig(**data.get("motion", {})),
            vitals=_vitals_from_dict(data.get("vitals", {})),
            drop_null_subcarriers=data.get("drop_null_subcarriers", True),
        )

    def with_overrides(self, **kwargs) -> "PipelineConfig":
        return replace(self, **kwargs)


def _vitals_from_dict(data: dict) -> VitalsConfig:
    """Build a VitalsConfig from JSON data, coercing list bands back to tuples."""
    coerced = {
        k: tuple(v) if isinstance(v, list) else v
        for k, v in data.items()
    }
    return VitalsConfig(**coerced)


DEFAULT_CONFIG = PipelineConfig()
