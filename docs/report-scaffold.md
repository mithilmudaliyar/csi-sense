# College Report — Scaffold

Skeleton and prompts for the write-up. This is **structure only** — fill each
section with your own words and your **real** (post-hardware) results. Do not
paste synthetic numbers as if they were real. Suggested length in brackets.

---

## 1. Abstract [~150 words]
- One paragraph: problem (device-free human sensing), approach (WiFi CSI on
  ESP32), what you built, what you measured, honest headline result.

## 2. Introduction [~1 page]
- Motivation: sensing presence/falls without cameras (privacy) or wearables
  (compliance). Use cases: elderly fall alerts, occupancy.
- Problem statement and the three concrete tasks: presence, fall, 0/1/2 count.
- Explicit scope + non-goals (no localization, no vitals, count <= 2). Cite
  docs/limitations.md.
- Contributions bullet list (what *you* did: end-to-end pipeline, 2-node setup,
  honest evaluation).

## 3. Background & Related Work [~1–1.5 pages]
- **What CSI is**: OFDM subcarriers, complex channel estimate, amplitude/phase;
  how human motion perturbs multipath. (Prompt: include a simple equation
  H(f) = |H(f)|e^{j*theta} and one diagram.)
- **CSI vs RSSI**: why per-subcarrier CSI is richer than a single RSSI value.
- **Related work note** (survey, don't overclaim): device-free sensing with
  WiFi CSI — presence/activity recognition, fall detection, crowd/counting.
  Mention representative lines of work (e.g. FallDeFi-style fall detection,
  CSI activity recognition, device-free counting) and the **ESP32-CSI-Tool**
  (Hernandez & Bulut) as the enabling firmware. Contrast Intel 5300/Atheros
  CSI setups with the cheap-ESP32 approach you use.
- Where your project sits: reproducing a small, honest subset on hobby
  hardware.

## 4. System Design [~1.5–2 pages]
- Hardware: 3x ESP32-WROOM-32, CP2102, 1 AP + 2 RX topology (reuse the diagram
  in docs/architecture.md).
- Firmware: ESP32-CSI-Tool roles (active_ap, active_sta), LLTF-only,
  channel/SSID config, 921600 baud. Confirmed serial schema (cite
  decision-log.md).
- Software pipeline: parse -> preprocess -> features -> ML -> dashboard (reuse
  the data-flow figure). Note the design principle: synthetic and live data
  share one code path.
- Data management: labeled sessions, sidecar metadata.

## 5. Methodology [~2–3 pages] — one subsection per task
For **each** task, cover: signal intuition -> features used -> model ->
training protocol -> metrics.

### 5.1 Preprocessing
- Manual Hampel filter (define it: median + MAD, 3-sigma gate); why enabled by
  default. PCA / wavelet options and when to enable. Null-subcarrier removal.

### 5.2 Feature extraction
- Sliding window (size/hop, sample-rate assumption). Table of features
  (pipeline/features.py FEATURE_COLUMNS) with a one-line physical meaning each.

### 5.3 Presence detection
- Variance intuition; logistic regression; threshold fallback + calibration.

### 5.4 Fall detection
- Spike-then-stillness signature; the three key features; RandomForest;
  class imbalance handling; note the optional CNN/LSTM extension.

### 5.5 People counting (0/1/2)
- Two-node feature concatenation; why spatial diversity helps; RandomForest
  3-class; the hard cap at 2 and why.

## 6. Experimental Setup [~1 page]
- Room description, board placement, channel, packet rate, **measured** sample
  rate. Data collection protocol per label (how you performed empty / walking /
  fall / 2-people; how many sessions; durations). Train/test split strategy
  (split by session, not by window, to avoid leakage — call this out).

## 7. Results [PLACEHOLDER — fill with REAL data]
> WARNING: Do not use synthetic numbers here. Collect real captures first.

- Per-task metrics table: accuracy, precision, recall, F1, confusion matrices.
- Presence: ROC or threshold sweep.
- Fall: precision/recall trade-off (false alarms matter).
- Counting: 3x3 confusion matrix for 0/1/2.
- Effect of preprocessing (ablation: Hampel on/off, +wavelet, +PCA).
- Effect of 1 vs 2 nodes on counting.
- Sample-rate sensitivity if you varied it.

*(Placeholder tables/figures to replace:)*
- Table 7.1 — Per-task performance on real data.
- Fig 7.1 — Example amplitude traces per scenario (from the dashboard).
- Fig 7.2 — Confusion matrices.

## 8. Discussion [~1 page]
- What worked, what didn't, where the model was fooled.
- Environment dependence (retraining), sample-rate limits, phase noise.
- Honest failure cases (e.g. slow movement missed, 2 people merging).

## 9. Limitations [~0.5 page]
- Summarize docs/limitations.md: count <= 2, no localization, environment-
  specific, synthetic != real, no vitals.

## 10. Conclusion & Future Work [~0.5 page]
- Recap contributions and honest results.
- Future: more data/rooms, better fall model (CNN/LSTM), zone-based coarse
  localization, Phase-5 vitals as a stretch.

## References
- ESP32-CSI-Tool (Steven M. Hernandez et al.) — see
  firmware/esp32-csi-tool/docs/bibtex/ for ready BibTeX entries.
- WiFi CSI sensing surveys; fall-detection and counting papers you cite in S3.

## Appendices
- A: exact commands to reproduce (from README.md).
- B: confirmed serial schema (docs/decision-log.md).
- C: flashing procedure (docs/flashing-guide.md).
