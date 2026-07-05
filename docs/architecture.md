# System Architecture

WiFi CSI (Channel State Information) based human sensing — presence, fall, and
people-counting (0/1/2) — with no cameras and no wearables. College hobby
project, honestly scoped.

## What CSI gives us

When a WiFi packet travels from a transmitter to a receiver, it takes many
paths (walls, floor, furniture, **and people**). The receiver's PHY estimates
the channel per OFDM subcarrier as a complex number (amplitude + phase). That
per-subcarrier complex vector is the **CSI**. When a person moves, they perturb
the multipath, and the CSI amplitude fluctuates. Different activities leave
different statistical signatures in that fluctuation — which is what we detect.

The original **ESP32** (WROOM-32) exposes CSI through the ESP-IDF WiFi API, so
cheap boards can act as CSI sniffers.

## Topology (3 boards)

```
                 WiFi packets (channel 6)
   ┌──────────┐   ~~~~~~~~~~~~~~~~~~~~~~~>   ┌──────────┐
   │ Board 1  │                              │ Board 2  │  RX node #1
   │ AP / TX  │───────────┬──────────────────│ STA CSI  │──► USB serial ─┐
   │active_ap │           │                  │active_sta│                │
   └──────────┘           │                  └──────────┘                │
                          │                  ┌──────────┐                │
                          └──────────────────│ Board 3  │  RX node #2    │
                                             │ STA CSI  │──► USB serial ─┤
                                             │active_sta│                │
                                             └──────────┘                │
                                                                         ▼
                                                              Host PC (this repo)
```

- **Board 1 (AP/TX):** broadcasts an access point and emits packets at a fixed
  rate so the RX nodes always have traffic to measure. Does not collect CSI.
- **Boards 2 & 3 (RX):** join the AP and log CSI for every received packet to
  USB serial. Two nodes at different positions give **spatial diversity** —
  essential for the people-counting task and helpful for robustness.
- The host PC reads both serial streams and runs the pipeline below.

(A home router can replace Board 1 as the packet source; a dedicated AP board
is more controllable. See `docs/flashing-guide.md`.)

## Data flow

```
 ESP32 RX ──serial(921600)──► parser ──► preprocess ──► features ──► ML ──► dashboard
 (CSI_DATA   CSISerialReader   CSIFrame   Hampel/PCA/   sliding-    presence  Streamlit
  CSV lines)                   (amp,      wavelet +     window      fall      live/replay
                                phase)    null-drop     feature df  counting
                                     │                                  │
                                     └────► SessionWriter ──► data/{raw,processed,synthetic}
                                            (CSV + .meta.json label sidecar)
```

Component map (files):

| Stage | Module | Notes |
|-------|--------|-------|
| Serial read | `pipeline/serial_reader.py` | `readline()`-based; mockable for tests & replay |
| Live capture CLI | `pipeline/capture.py` | serial → labeled session file |
| Parse | `pipeline/parser.py` | exact ESP32-CSI-Tool schema → `CSIFrame` |
| Config | `pipeline/config.py` | all toggles (denoise, window) — nothing hardcoded |
| Preprocess | `pipeline/preprocess.py` | manual Hampel + PCA + wavelet + null-drop |
| Features | `pipeline/features.py` | sliding-window tidy DataFrame for sklearn |
| Storage | `pipeline/storage.py` | CSV/Parquet sessions + JSON label sidecar |
| End-to-end | `pipeline/run_pipeline.py` | parse→preprocess→features→classify CLI |
| Synthetic | `ml/synthetic.py` | fake CSI for 4 scenarios (validates code path) |
| Datasets | `ml/dataset.py` | builds labeled features via the REAL pipeline |
| Models | `ml/presence.py`, `ml/fall.py`, `ml/counting.py` | sklearn; torch optional |
| Train/eval | `ml/train.py`, `ml/evaluate.py` | synthetic; prints honesty banner |
| Dashboard | `dashboard/app.py` | Streamlit; source = synthetic/recorded/live |

## The three tasks

1. **Presence / motion (primary, most reliable).** Motion raises CSI amplitude
   variance. Logistic regression on variance-family features; a training-free
   variance-threshold fallback (`ThresholdPresenceDetector`) works day one from
   a short empty-room calibration.
2. **Fall (secondary).** Signature = a sharp large-amplitude disturbance
   **followed by sudden stillness**. Captured by `amp_max_abs_diff`,
   `n_disturbance_peaks`, and `post_event_stillness_s`; a RandomForest learns
   their interaction. Optional raw-sequence CNN/LSTM extension point in
   `ml/fall.py` (torch, opt-in).
3. **People counting (0/1/2 ONLY).** Concatenated features from both RX nodes;
   RandomForest 3-class. **Hard-capped at 2** — see limitations.

## Not in scope

- **Precise localization / coordinates.** Any future tracking would be coarse
  zone-based only.
- **Counting 3+ people.**
- **Vitals (breathing / heart rate)** — see Phase 5 note below.

## Roadmap / phases

- **Phase 1 — foundation (this repo):** firmware vendored, full software
  pipeline, synthetic validation, dashboard, docs. Done in software before
  hardware.
- **Phase 2 — real presence:** collect labeled empty/occupied captures,
  retrain, validate on real data.
- **Phase 3 — real fall detection:** collect (safely simulated) fall captures,
  tune the fall features/model, measure real precision/recall.
- **Phase 4 — real 2-node counting:** deploy both RX nodes, collect 0/1/2
  captures, retrain the two-node model.
- **Phase 5 — STRETCH: vitals (breathing / heart-rate).** *Out of current
  scope.* CSI phase on a stationary subject can, in principle, reveal
  sub-Hz chest motion (breathing ~0.2–0.5 Hz) via fine-grained phase analysis
  and band-pass filtering. This would need a much higher and steadier sampling
  rate, careful phase sanitization, and a stationary subject. Noted as a
  possible future direction only — not promised.
