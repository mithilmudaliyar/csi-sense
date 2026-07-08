"""Per-node motion-intensity and coarse-zone heuristics.

HONESTY (the point of this module):

- ``motion_intensity_series`` is an honest visualization of signal
  DISTURBANCE INTENSITY per RX node — "activity is stronger near node 2".
  It is NOT position, NOT coordinates, NOT localization.
- ``classify_zone`` is an EXPERIMENTAL heuristic: it lights up the coarse
  area of whichever node currently sees the strongest disturbance. It
  requires per-room calibration (node placement, empty-room baselines) to
  mean anything, and even then it only ever says "node 1 area / node 2
  area / between-or-uncertain". Never metric position, never house mapping.

Intensity reuses the existing sliding-window feature extractor
(``pipeline.features.extract_features``) — the ``amp_std`` feature is the
motion-energy proxy the classifiers already rely on — so synthetic replay
and live serial share one code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.features import extract_features

# Zone verdict labels. Deliberately vague: coarse areas, not positions.
ZONE_QUIET = "no motion"
ZONE_UNCERTAIN = "between / uncertain"

ZONE_CAPTION = (
    "EXPERIMENTAL heuristic: strongest-disturbance node only. Needs "
    "per-room calibration to mean anything. NOT localization, NOT "
    "coordinates."
)
INTENSITY_CAPTION = "Motion intensity (signal disturbance), not position."


@dataclass(frozen=True)
class ZoneEstimate:
    """Result of the coarse-zone heuristic for one time step."""

    zone: str                 # "<node> area" | ZONE_UNCERTAIN | ZONE_QUIET
    dominant_node: str | None
    lead_fraction: float      # relative lead of the top node over the runner-up
    is_confident: bool        # lead exceeded the configured dominance margin
    caption: str = ZONE_CAPTION


def motion_intensity_series(
    amplitude: np.ndarray,
    config: PipelineConfig,
    node_id: str = "node1",
) -> pd.DataFrame:
    """Rolling motion-intensity for one node's (preprocessed) amplitude matrix.

    One row per sliding window: ``window_start_s``, ``node_id``,
    ``intensity`` (rolling-smoothed ``amp_std``). Honest proxy for how much
    the channel near this node is being disturbed — nothing more.
    """
    feats = extract_features(amplitude, config, node_id=node_id)
    smoothed = (
        feats["amp_std"]
        .rolling(max(1, config.motion.smoothing_windows), min_periods=1)
        .mean()
    )
    return pd.DataFrame(
        {
            "window_start_s": feats["window_start_s"],
            "node_id": node_id,
            "intensity": smoothed.astype(float),
        }
    )


def normalize_intensities(
    series_by_node: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Scale all nodes' intensities by one SHARED max so bars are comparable.

    Per-node normalization would hide exactly the cross-node contrast the
    meters exist to show; a shared scale keeps "node 2 is livelier" honest.
    Returns new frames with an added ``intensity_norm`` column in [0, 1].
    """
    global_max = max(
        (float(df["intensity"].max()) for df in series_by_node.values() if len(df)),
        default=0.0,
    )
    scale = global_max if global_max > 1e-12 else 1.0
    out: dict[str, pd.DataFrame] = {}
    for node, df in series_by_node.items():
        new = df.copy()
        new["intensity_norm"] = new["intensity"] / scale
        out[node] = new
    return out


def latest_intensities(
    series_by_node: Mapping[str, pd.DataFrame],
    column: str = "intensity_norm",
) -> dict[str, float]:
    """The most recent per-node intensity value (for live meters)."""
    return {
        node: float(df[column].iloc[-1]) if len(df) else 0.0
        for node, df in series_by_node.items()
    }


def estimate_weighted_position(
    intensities: Mapping[str, float],
    node_positions: Mapping[str, tuple[float, float]],
) -> tuple[float, float] | None:
    """Intensity-weighted centroid across node (x, y) positions.

    EXPERIMENTAL, same honesty rules as ``classify_zone``: nodes seeing MORE
    disturbance pull the estimate toward them. This is a continuous version
    of the zone heuristic, NOT a calibrated coordinate system — no
    angle-of-arrival or time-of-flight data exists on this hardware to
    support real triangulation. Needs >=2 nodes reporting; returns None
    otherwise. If no node has any measurable disturbance, returns the plain
    (unweighted) centroid rather than an arbitrary corner.
    """
    nodes = [n for n in node_positions if n in intensities]
    if len(nodes) < 2:
        return None
    weights = np.array([max(intensities[n], 0.0) for n in nodes], dtype=float)
    if weights.sum() <= 1e-12:
        weights = np.ones(len(nodes))
    weights = weights / weights.sum()
    pts = np.array([node_positions[n] for n in nodes], dtype=float)
    point = (pts * weights[:, None]).sum(axis=0)
    return float(point[0]), float(point[1])


def classify_zone(
    intensities: Mapping[str, float],
    config: PipelineConfig,
) -> ZoneEstimate:
    """EXPERIMENTAL coarse-zone heuristic from ABSOLUTE per-node intensities.

    Pass RAW ``intensity`` (amp_std), NOT the session-max ``intensity_norm``:
    normalized values are always ~1.0, so the quiet gate could never fire and
    an empty room's front-end noise got a confident "node X area" verdict. The
    quiet gate must see absolute activity to know the room is actually empty.

    Rules (deliberately simple and inspectable):
    - top absolute intensity under ``quiet_floor`` -> ZONE_QUIET
      (``quiet_floor`` is an ABSOLUTE amp_std threshold — a per-room
      calibration knob; the default only separates the synthetic scenarios)
    - fewer than 2 nodes reporting                 -> ZONE_UNCERTAIN
      (one node cannot disambiguate area)
    - top node leads runner-up by >= margin        -> "<node> area"
    - otherwise                                    -> ZONE_UNCERTAIN
    """
    if not intensities:
        return ZoneEstimate(ZONE_QUIET, None, 0.0, False)

    ranked = sorted(intensities.items(), key=lambda kv: kv[1], reverse=True)
    top_node, top = ranked[0]

    if top < config.motion.quiet_floor:
        return ZoneEstimate(ZONE_QUIET, None, 0.0, False)
    if len(ranked) < 2:
        # A single node can say "motion", never "where".
        return ZoneEstimate(ZONE_UNCERTAIN, top_node, 0.0, False)

    runner_up = ranked[1][1]
    lead = (top - runner_up) / max(top, 1e-12)
    if lead >= config.motion.dominance_margin:
        return ZoneEstimate(f"{top_node} area", top_node, float(lead), True)
    return ZoneEstimate(ZONE_UNCERTAIN, top_node, float(lead), False)
