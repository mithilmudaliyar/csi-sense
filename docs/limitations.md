# Limitations — read this honestly

This is a college hobby project built on commodity ESP32 hardware. The goal is
a working, honestly-scoped system, not impressive-but-unproven claims. The
following limits are real and deliberate. Do not oversell the system past them.

## People counting caps at 2

The counting model distinguishes **only 0, 1, or 2 people** (`ml/counting.py`,
`MAX_PEOPLE = 2`). Reliably counting 3+ people with commodity CSI on a couple
of ESP32s is not achievable here — bodies interfere and the signatures stop
separating cleanly. We do not attempt it and you should not report accuracy
for it.

## No precise localization

The system detects **that** something is happening, not **where** in
coordinates. There is no positioning, no trilateration, no coordinates. Any
future location work would be coarse **zone-based** at best (e.g. "near node 2"
vs "near node 1"), never metric position.

## Environment-specific — retraining is expected

CSI signatures depend heavily on the physical environment: room geometry,
furniture, wall materials, board placement, WiFi channel, even people's
clothing. **A model trained in one room will likely degrade in another.**
Expect to recollect labeled data and retrain when you move the setup. The
pipeline is built to make that cheap (label at capture time; one command to
rebuild datasets), but it is a real operational cost.

## The current accuracy numbers are SYNTHETIC — not real

Everything trained and evaluated so far uses the **synthetic** data generator
(`ml/synthetic.py`). That validates the *code path only*. The synthetic
generator and the feature extractor share assumptions, so synthetic scores are
optimistic and near-perfect by construction. They say **nothing** about
real-world accuracy. Every train/eval script prints this warning. Real numbers
come only after Phases 2–4 with real captures.

## Vitals (breathing / heart rate) — EXPERIMENTAL, unvalidated

Phase 5 adds a breathing/heart-rate estimator (`ml/vitals.py`), but treat it as
a **research spike, not a feature**:

- **Synthetic-only validation.** It recovers rates planted by `ml/synthetic.py`
  and refuses on an empty room. That proves the DSP path, not real accuracy —
  the same caveat as every other model here.
- **Breathing is the only credible band** on commodity single-antenna 2.4 GHz
  ESP32. **Heart rate is not trusted** on this hardware (SNR too low); the code
  will report it only when FFT and zero-crossing strongly agree, and otherwise
  returns nothing.
- **Refuses rather than guesses.** Short records (< `min_duration_s`), dead
  channels, NaNs, or structureless noise return **no number**, not a confident
  wrong one. The gate (`min_peak_ratio`) is a per-room **calibration knob**, not
  a guarantee — tune it against a real empty-room baseline before trusting any
  output.
- Needs higher, steadier sampling and clean phase; do not report a BPM as fact
  until validated against a real still-subject capture with a reference monitor.

## Motion "zone" is a heuristic, not localization

`pipeline/motion.py` reports per-node **signal-disturbance intensity** and a
coarse "which node's area is livelier" verdict. It is **not** position, **not**
coordinates — see *No precise localization* above. The zone call needs per-room
calibration (`quiet_floor`, node placement) to mean anything and only ever emits
"node N area / uncertain / quiet".

## Hardware / signal caveats

- **Sampling rate is not guaranteed.** It depends on the AP packet rate, serial
  baud (keep 921600+), and USB stack. Low or jittery rates hurt fall detection
  most (the spike-then-still timing blurs). Measure your real rate with
  `firmware/esp32-csi-tool/python_utils/serial_measure_rate.py`.
- **ESP32 phase is noisy.** Raw CSI phase carries hardware offsets (CFO/SFO,
  random phase boot offset). We lean on **amplitude** features primarily; phase
  is stored but treated cautiously.
- **Only the original ESP32** (WROOM-32) exposes this CSI API. ESP32-S2/S3/C3
  do not, in the way this tool needs.
- **Two RX nodes are assumed for counting.** With one node, counting falls back
  to duplicating a single node's features (`single_node_fallback_features`) —
  functional but less accurate, and flagged as such, never silent.

## What IS reasonable to claim

- Presence/motion detection is the most reliable task and should work well
  after real calibration.
- Fall detection is plausible as a secondary feature with tuning, but validate
  precision/recall on real (safely simulated) falls before trusting it.
- 0/1/2 counting is a stretch that needs both nodes and real per-room training.
