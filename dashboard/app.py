"""Streamlit dashboard for CSI human sensing - synthetic today, live serial later.

Run:
    streamlit run dashboard/app.py

The data source is swappable via the sidebar:
  - Synthetic   : generate a fresh scenario in-memory (works with no hardware)
  - Recorded    : load a saved session from data/ (synthetic or real capture)
  - Live serial : connect to an ESP32 COM port (enabled once hardware arrives)

It shows amplitude-over-time, the extracted-feature summary, and - if models
are trained (ml/models/*.joblib) - the presence / fall / counting verdicts.

Design note: all heavy logic lives in the pipeline/ and ml/ packages. This
file is a thin presentation layer so the same code serves synthetic and live.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make the project root importable when launched via `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import DEFAULT_CONFIG  # noqa: E402
from pipeline.features import FEATURE_COLUMNS, extract_features  # noqa: E402
from pipeline.motion import (  # noqa: E402
    INTENSITY_CAPTION,
    ZONE_CAPTION,
    classify_zone,
    latest_intensities,
    motion_intensity_series,
    normalize_intensities,
)
from pipeline.parser import parse_csi_line  # noqa: E402
from pipeline.preprocess import preprocess  # noqa: E402
from pipeline.storage import (  # noqa: E402
    amplitude_matrix,
    list_sessions,
    load_session,
    phase_matrix,
)

MODELS_DIR = ROOT / "ml" / "models"
DATA_DIRS = [ROOT / "data" / "synthetic", ROOT / "data" / "raw", ROOT / "data" / "processed"]


@st.cache_data(show_spinner=False)
def _synthetic_amplitude(scenario: str, duration_s: float, sr: float, seed: int) -> np.ndarray:
    from ml.synthetic import generate_session

    sess = generate_session(scenario, duration_s=duration_s, sample_rate_hz=sr, seed=seed)
    amps = [parse_csi_line(ln).amplitude for ln in sess.lines]
    return np.vstack(amps).astype(np.float32)


@st.cache_data(show_spinner=False)
def _synthetic_amp_phase(
    scenario: str, duration_s: float, sr: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """Amplitude + WRAPPED phase matrices (for the experimental vitals panel)."""
    from ml.synthetic import generate_session

    sess = generate_session(scenario, duration_s=duration_s, sample_rate_hz=sr, seed=seed)
    frames = [parse_csi_line(ln) for ln in sess.lines]
    amp = np.vstack([f.amplitude for f in frames]).astype(np.float32)
    ph = np.vstack([f.phase for f in frames]).astype(np.float32)
    return amp, ph, sess.ground_truth


@st.cache_data(show_spinner=False)
def _synthetic_two_node(
    scenario: str, duration_s: float, sr: float, seed: int, active_node: str,
) -> dict[str, np.ndarray]:
    """Two SYNTHETIC correlated-but-offset RX views of one scene."""
    from ml.synthetic import generate_two_node_session

    two = generate_two_node_session(
        scenario, duration_s=duration_s, sample_rate_hz=sr,
        seed=seed, active_node=active_node,
    )
    return {
        node: np.vstack([parse_csi_line(ln).amplitude for ln in s.lines]).astype(np.float32)
        for node, s in two.items()
    }


def _motion_panel(amp_by_node: dict[str, np.ndarray], config) -> None:
    """Per-node motion-intensity meters + EXPERIMENTAL coarse-zone widget.

    Honest framing: this shows WHERE THE SIGNAL IS MOST DISTURBED (e.g.
    "activity is stronger near node 2") — motion intensity, not position.
    """
    st.subheader("Motion intensity per node")
    st.caption(f"{INTENSITY_CAPTION} An honest proxy like “activity is "
               f"stronger near node 2” — never a coordinate.")

    series = {
        node: motion_intensity_series(preprocess(amp, config), config, node_id=node)
        for node, amp in amp_by_node.items()
    }
    norm = normalize_intensities(series)
    latest = latest_intensities(norm)                      # 0-1, for the bars
    latest_abs = latest_intensities(series, column="intensity")  # for the zone gate

    cols = st.columns(max(1, len(norm)))
    for col, (node, val) in zip(cols, sorted(latest.items())):
        col.metric(f"{node} activity", f"{val * 100:.0f}%")
        col.progress(float(min(1.0, max(0.0, val))))

    chart = pd.DataFrame(
        {node: df.set_index("window_start_s")["intensity_norm"]
         for node, df in norm.items()}
    )
    st.line_chart(chart, height=180)
    st.caption("Rolling per-window amplitude disturbance, shared scale across nodes.")

    st.markdown("**Coarse-zone indicator — EXPERIMENTAL**")
    if len(norm) >= 2:
        zone = classify_zone(latest_abs, config)
        icon = "🟢" if zone.is_confident else "⚪"
        st.markdown(f"{icon} **{zone.zone}** (lead {zone.lead_fraction * 100:.0f}%)")
    else:
        st.markdown("⚪ needs ≥2 RX nodes — a single node can say *motion*, never *where*")
    st.caption(ZONE_CAPTION)


def _vitals_panel(phase: np.ndarray, sr: float, ground_truth: dict | None) -> None:
    """EXPERIMENTAL Phase-5 vitals panel. Breathing primary; HR unreliable."""
    from ml.vitals import EXPERIMENTAL_BANNER, HEART_CAVEAT, estimate_vitals

    with st.expander("🫁 Vitals — EXPERIMENTAL Phase-5 (UNVALIDATED)", expanded=True):
        st.warning(EXPERIMENTAL_BANNER)
        result = estimate_vitals(phase, sr)
        c1, c2 = st.columns(2)
        b = result.breathing
        c1.metric(
            "Breathing (primary vitals output)",
            f"{b.bpm:.1f} BPM" if b.bpm is not None else "REFUSED",
            f"DSP confidence: {b.confidence}",
            delta_color="off",
        )
        h = result.heart
        c2.metric(
            "Heart rate (UNRELIABLE on this hardware)",
            f"{h.bpm:.1f} BPM" if h.bpm is not None else "REFUSED",
            f"DSP confidence: {h.confidence}",
            delta_color="off",
        )
        st.caption(HEART_CAVEAT)
        if ground_truth:
            st.caption(
                f"Synthetic ground truth: breathing "
                f"{ground_truth.get('breathing_bpm', float('nan')):.1f} BPM, heart "
                f"{ground_truth.get('heart_bpm', float('nan')):.1f} BPM. Recovering these "
                f"proves the DSP math only — it is NOT evidence of real-world accuracy."
            )


def _load_models() -> dict:
    import joblib

    models = {}
    for name in ("presence", "fall", "counting"):
        p = MODELS_DIR / f"{name}.joblib"
        if p.exists():
            try:
                models[name] = joblib.load(p)
            except Exception as exc:  # pragma: no cover - defensive UI path
                st.warning(f"Could not load {name} model: {exc}")
    return models


def _classify(features: pd.DataFrame, models: dict) -> None:
    from pipeline.features import feature_matrix

    cols = st.columns(3)
    if "presence" in models:
        from ml.presence import presence_feature_matrix

        preds = models["presence"].predict(presence_feature_matrix(features))
        frac = float(np.mean(preds))
        cols[0].metric("Presence", "DETECTED" if frac > 0.5 else "empty", f"{frac*100:.0f}% windows")
    if "fall" in models:
        fpred = models["fall"].predict(feature_matrix(features))
        n = int(np.sum(fpred))
        cols[1].metric("Fall", "FALL" if n else "none", f"{n}/{len(fpred)} windows")
    if "counting" in models:
        from ml.counting import counting_feature_matrix, single_node_fallback_features

        two = single_node_fallback_features(features)
        cpred = models["counting"].predict(counting_feature_matrix(two))
        vals, counts = np.unique(cpred, return_counts=True)
        mode = int(vals[np.argmax(counts)])
        cols[2].metric("People (0/1/2)", str(mode), "single-node est.")
    if not models:
        st.info("No trained models found. Run `py -m ml.train` to enable live verdicts. "
                "Feature extraction below still works.")


def main() -> None:
    st.set_page_config(page_title="WiFi CSI Sensing", page_icon="📡", layout="wide")
    st.title("📡 WiFi CSI Human Sensing")
    st.caption("Presence · Fall · People-counting (0/1/2). Synthetic today, live ESP32 serial once hardware arrives.")

    config = DEFAULT_CONFIG
    sr = config.window.sample_rate_hz

    st.sidebar.header("Data source")
    source = st.sidebar.radio("Source", ["Synthetic", "Recorded session", "Live serial (hardware)"])

    amp: np.ndarray | None = None
    meta_label = None

    if source == "Synthetic":
        scenario = st.sidebar.selectbox("Scenario", ["empty", "walking", "fall", "2people", "still_vitals"])
        duration = st.sidebar.slider("Duration (s)", 5, 40, 20)
        seed = st.sidebar.number_input("Seed", value=7, step=1)
        amp = _synthetic_amplitude(scenario, float(duration), sr, int(seed))
        meta_label = scenario
        if st.sidebar.button("Save this session to data/synthetic/"):
            from ml.generate_synthetic import write_scenario

            path = write_scenario(scenario, ROOT / "data" / "synthetic", float(duration), sr, int(seed))
            st.sidebar.success(f"Saved {path.name}")

    elif source == "Recorded session":
        sessions = []
        for d in DATA_DIRS:
            sessions.extend(list_sessions(d))
        if not sessions:
            st.warning("No recorded sessions yet. Generate some with "
                       "`py -m ml.generate_synthetic`, or record from hardware.")
            return
        choice = st.sidebar.selectbox("Session file", [str(p) for p in sessions])
        df, meta = load_session(choice)
        amp = amplitude_matrix(df)
        meta_label = meta.get("label", "?")
        st.sidebar.write(f"label: **{meta_label}**, source: {meta.get('source', '?')}")

    else:  # Live serial
        st.sidebar.text_input("COM port", value="COM5")
        st.sidebar.number_input("Baud", value=921600, step=1)
        st.warning("Live serial capture activates once an ESP32 RX node is connected. "
                   "The reader (pipeline.serial_reader.CSISerialReader.open_port) is "
                   "already implemented and unit-tested against a mock; this panel wires "
                   "to it on hardware arrival. See docs/flashing-guide.md.")
        return

    if amp is None or amp.size == 0:
        st.error("No CSI data to display.")
        return

    # Preprocess + features.
    denoised = preprocess(amp, config)
    features = extract_features(denoised, config)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Amplitude over time")
        t = np.arange(amp.shape[0]) / sr
        mean_amp = denoised.mean(axis=1)
        chart_df = pd.DataFrame({"time_s": t, "mean_amplitude": mean_amp}).set_index("time_s")
        st.line_chart(chart_df, height=260)
        st.caption(f"{amp.shape[0]} samples × {amp.shape[1]} subcarriers "
                   f"({denoised.shape[1]} active after null removal). Nominal {sr:.0f} Hz.")
        # A few individual subcarriers for texture.
        n_show = min(6, denoised.shape[1])
        idx = np.linspace(0, denoised.shape[1] - 1, n_show).astype(int)
        sub_df = pd.DataFrame(denoised[:, idx], columns=[f"sc{i}" for i in idx])
        sub_df["time_s"] = t
        st.line_chart(sub_df.set_index("time_s"), height=220)

    with right:
        st.subheader("Classifier output")
        _classify(features, _load_models())
        st.subheader("Feature summary (window mean)")
        summary = features[list(FEATURE_COLUMNS)].mean().round(3)
        st.dataframe(summary.rename("mean").to_frame(), use_container_width=True)

    # ---- Experimental Phase-5: vitals + coarse motion zone (clearly flagged) ----
    st.divider()
    st.header("🧪 Experimental — Phase-5 (UNVALIDATED)")
    if source == "Synthetic":
        _, phase, gt = _synthetic_amp_phase(scenario, float(duration), sr, int(seed))
        _vitals_panel(phase, sr, gt)
        # Zone needs ≥2 RX nodes; still_vitals has no motion generator, so the
        # motion demo falls back to a walking scene.
        motion_sc = scenario if scenario != "still_vitals" else "walking"
        amp_by_node = _synthetic_two_node(motion_sc, float(duration), sr, int(seed), "node1")
        _motion_panel(amp_by_node, config)
    else:  # Recorded single-node capture: vitals only (zone needs 2 nodes).
        try:
            _vitals_panel(phase_matrix(df), sr, (meta.get("extra") or {}).get("ground_truth"))
        except ValueError:
            st.info("This session has no phase columns — vitals need phase CSI.")
        st.caption("Coarse-zone needs ≥2 RX nodes; unavailable for one recorded node.")

    st.caption("⚠️ Synthetic data validates the software only — not real-world accuracy. "
               "Retrain on real captures after collecting labeled data.")


if __name__ == "__main__":
    main()
