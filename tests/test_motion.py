"""Unit tests for per-node motion intensity + coarse-zone heuristic.

Asserts the honest behaviours: intensity tracks disturbance, a shared scale
keeps cross-node bars comparable, and ``classify_zone`` judges ABSOLUTE
intensities (raw amp_std) — so an empty room actually reads ZONE_QUIET
instead of a confident node call, and the zone only commits when one node
clearly dominates (never from a single node).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.synthetic import generate_two_node_session
from pipeline.motion import (
    ZONE_QUIET,
    ZONE_UNCERTAIN,
    classify_zone,
    latest_intensities,
    motion_intensity_series,
    normalize_intensities,
)
from pipeline.parser import parse_csi_line
from pipeline.preprocess import preprocess


def _two_node_series(default_config, scenario="walking", active="node1", seed=5):
    two = generate_two_node_session(scenario, duration_s=20.0,
                                    sample_rate_hz=default_config.window.sample_rate_hz,
                                    seed=seed, active_node=active)
    series = {}
    for node, sess in two.items():
        amp = np.vstack([parse_csi_line(ln).amplitude for ln in sess.lines]).astype(np.float32)
        series[node] = motion_intensity_series(preprocess(amp, default_config),
                                               default_config, node_id=node)
    return series


def test_intensity_series_columns_and_length(default_config):
    norm = normalize_intensities(_two_node_series(default_config))
    df = norm["node1"]
    for col in ("window_start_s", "node_id", "intensity", "intensity_norm"):
        assert col in df.columns
    assert len(df) > 0


def test_active_node_reads_stronger(default_config):
    norm = normalize_intensities(_two_node_series(default_config, active="node1"))
    latest = latest_intensities(norm)
    # The node nearer the activity should read the higher disturbance.
    assert latest["node1"] > latest["node2"]


def test_zone_points_at_dominant_node(default_config):
    # classify_zone consumes ABSOLUTE intensities (raw amp_std), never the
    # session-max-normalized values (those are ~1.0 by construction).
    series = _two_node_series(default_config, scenario="walking", active="node1")
    zone = classify_zone(latest_intensities(series, column="intensity"),
                         default_config)
    assert zone.dominant_node == "node1"
    assert zone.is_confident
    assert zone.zone == "node1 area"


def test_empty_scene_reads_quiet(default_config):
    # The P0 fix: an empty room's absolute front-end noise sits below
    # quiet_floor, so the verdict is ZONE_QUIET — not a confident node area
    # (which is what normalized inputs used to produce).
    series = _two_node_series(default_config, scenario="empty")
    latest = latest_intensities(series, column="intensity")
    assert max(latest.values()) < default_config.motion.quiet_floor
    zone = classify_zone(latest, default_config)
    assert zone.zone == ZONE_QUIET
    assert not zone.is_confident


def test_normalize_uses_shared_scale():
    series = {
        "node1": pd.DataFrame({"intensity": [1.0, 2.0]}),
        "node2": pd.DataFrame({"intensity": [4.0, 8.0]}),
    }
    norm = normalize_intensities(series)
    # Global max (8.0) maps to 1.0; everything shares that scale.
    assert norm["node2"]["intensity_norm"].max() == 1.0
    assert norm["node1"]["intensity_norm"].max() == 0.25


def test_quiet_when_all_below_floor(default_config):
    zone = classify_zone({"node1": 0.01, "node2": 0.02}, default_config)
    assert zone.zone == ZONE_QUIET


def test_single_node_cannot_localize(default_config):
    # One node can say "motion" but never "where".
    zone = classify_zone({"node1": 0.9}, default_config)
    assert zone.zone == ZONE_UNCERTAIN
    assert not zone.is_confident


def test_close_nodes_stay_uncertain(default_config):
    # Near-tie below the dominance margin must not commit to a node.
    zone = classify_zone({"node1": 0.50, "node2": 0.48}, default_config)
    assert zone.zone == ZONE_UNCERTAIN
