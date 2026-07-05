# Decision Log

Every non-obvious engineering choice, with rationale. Newest decisions appended.

## Confirmed ESP32-CSI-Tool serial schema (authoritative)

Read directly from the vendored source, not guessed:
`firmware/esp32-csi-tool/_components/csi_component.h`
(functions `_wifi_csi_cb` and `_print_csi_csv_header`) and cross-checked
against `firmware/esp32-csi-tool/python_utils/example_csi.csv`.

Each CSI packet is **one CSV line**. The header the firmware prints once at
boot is:

```
type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,real_time_set,real_timestamp,len,CSI_DATA
```

Concretely, every data line is:

```
CSI_DATA,<role>,<mac>,<rssi>,<rate>,<sig_mode>,<mcs>,<bandwidth>,<smoothing>,
<not_sounding>,<aggregation>,<stbc>,<fec_coding>,<sgi>,<noise_floor>,
<ampdu_cnt>,<channel>,<secondary_channel>,<local_timestamp>,<ant>,<sig_len>,
<rx_state>,<real_time_set>,<real_timestamp>,<len>,[<int8 int8 int8 ... > ]
```

Key facts the parser relies on:

- Field 0 is the literal string `CSI_DATA` (this is the `type` column).
- **25 comma-separated metadata fields**, then a **bracketed buffer**.
  Note the printed header lists the bracketed buffer column as `CSI_DATA`,
  i.e. the header has 26 names but the last one labels the `[...]` array.
- `role` = the firmware's `project_type` string (e.g. `AP` / `STA`).
- `len` = number of **int8 values** in the buffer (e.g. `128` for LLTF-only,
  `384` for LLTF+HT-LTF+STBC-HT-LTF).
- The buffer is `data->buf` printed verbatim: signed 8-bit integers,
  interleaved **[imaginary, real]** per subcarrier slot. The tool itself
  computes `amplitude = sqrt(buf[2i]^2 + buf[2i+1]^2)` and
  `phase = atan2(buf[2i], buf[2i+1])` — our parser matches this exactly
  (`imag = raw[0::2]`, `real = raw[1::2]`, `csi = real + 1j*imag`).
- `local_timestamp` is the ESP32 microsecond clock (`rx_ctrl.timestamp`);
  `real_timestamp` is the tool's steady-clock seconds (float), only
  meaningful after `SETTIME:` is issued on the device. Because device time
  is unreliable, the host attaches its own epoch timestamp at read time
  (`CSISerialReader.frames` yields `(host_ts, frame)`).
- There is a trailing space before `]` in the buffer; the parser strips it.

**LLTF-only default**: `SHOULD_COLLECT_ONLY_LLTF` defaults to `y`, giving
`len=128` → **64 complex subcarriers**. The synthetic generator therefore
uses 64 subcarriers so synthetic and real data share one shape.

## CRITICAL schema quirk: `len` != printed buffer count (LLTF-only)

Found by unit-testing the parser against the real vendored
`example_csi.csv`: the metadata `len` field says **384**, but the line only
prints **128** int8 values. Reading `csi_component.h` explains it — with
`CONFIG_SHOULD_COLLECT_ONLY_LLTF` (the default) the firmware sets
`data_len = 128` and prints only the first 128 values, while the `len` field
it writes is `data->len` = 384 (the full buffer including HT-LTF slots that
aren't printed).

**Consequence / fix:** the parser must NOT hard-require
`printed_count == len`. My first version did and it rejected 100% of real
LLTF captures. Fixed: `len` is now stored as `advertised_len` (advisory);
decoding uses whatever count is actually printed; `buf_len` = printed count;
the only hard checks are non-empty + even length (plus the missing-`]` and
non-integer guards that catch genuinely truncated serial lines). This is a
concrete example of why the task said "don't guess the schema — read the
source AND test against the real example." Synthetic data uses len=128 with
128 printed values, so `advertised_len == buf_len` there; real hardware will
show 384 vs 128.

## Firmware toolchain: ESP-IDF is primary, not Arduino/PlatformIO

- ESP32-CSI-Tool is a **native ESP-IDF project** (`.cc` sources,
  `CMakeLists.txt`, `Kconfig.projbuild`, `idf.py menuconfig`). It is *not* an
  Arduino sketch. The CSI callback API (`esp_wifi_set_csi_rx_cb`) is exposed
  by ESP-IDF on the original ESP32 only.
- **Decision: use ESP-IDF (`idf.py`) as the primary build/flash path.** This
  is what the tool is written for and the least-friction route.
- PlatformIO can build ESP-IDF projects (`framework = espidf`) and is a fine
  CLI wrapper; a ready `platformio.ini` is provided in
  `firmware/platformio.ini` as an optional convenience. See the flashing
  guide for both paths.
- **Arduino IDE is documented as a fallback only** and is *not recommended*
  for this tool, because porting the ESP-IDF component structure
  (`_components/*.h`, Kconfig options) into an `.ino` is error-prone. The
  flashing guide records the exact ESP-IDF steps as the supported route and
  explains the Arduino situation honestly.

## PlatformIO install attempt

Attempted `py -m pip install platformio` on Python 3.14. Result recorded in
this log at build time (see "PlatformIO install outcome" below). Regardless
of outcome, ESP-IDF via the standalone installer is the primary path, so a
PlatformIO failure is **not** a hard blocker.

## Python 3.14 + package availability

- All base deps installed cleanly on Python 3.14.4 from wheels: numpy 2.5.1,
  pandas, scipy 1.18, matplotlib 3.11, scikit-learn 1.9, PyWavelets 1.8,
  pyserial 3.5, joblib, streamlit 1.58, pyarrow 24, pytest. No source builds
  were needed. No package had to be dropped or substituted.
- **torch is deliberately NOT in the base install.** It is an optional extra
  behind `ml.fall.build_torch_fall_model`, which raises a clear ImportError
  telling the user to `py -m pip install torch` only if they want the raw
  time-series CNN/LSTM path. Rationale: torch is a large dependency, wheel
  availability on brand-new Python versions is historically the last to land,
  and the sklearn RandomForest path is sufficient to validate the system.
  scikit-learn first, torch strictly optional.

## Storage format: CSV default, Parquet optional

- Sessions are written as **CSV by default** so captures are always
  human-inspectable and diffable, with a `.meta.json` sidecar carrying the
  label / node id / source / nominal sample rate.
- Parquet is available (`prefer_parquet=True`) when pyarrow is importable, for
  when capture volume grows. CSV stays the default for transparency at hobby
  scale.
- **Session labeling** is by both filename (`<UTCstamp>_<label>_<node>.csv`)
  and the sidecar. Valid labels: `empty`, `walking`, `fall`, `2people`,
  `unlabeled`.

## Subcarrier layout: real + j*imag, null-slot removal

- ESP-IDF buffer order is `[imag, real]`; parser builds `csi = real + 1j*imag`
  so `np.abs`/`np.angle` give amplitude/phase consistent with the tool.
- LLTF has null/guard subcarriers (DC + band edges) that carry ~no signal.
  `preprocess.select_active_subcarriers` drops slots whose std across the
  session is ~0, so downstream features aren't diluted by dead subcarriers.

## Denoising defaults (config-driven, not hardcoded)

`pipeline/config.py` centralizes every toggle:

- **Hampel filter: ENABLED by default** (window 7 samples/side, 3σ). Robust
  spike/outlier removal that preserves motion edges — safe general default.
  Implemented manually from scratch (project requirement, no library),
  vectorized with a MAD scale factor of 1.4826.
- **PCA denoise: OFF by default.** Useful to isolate dominant correlated
  motion but can erase weak-but-real signals; opt-in per experiment.
- **Wavelet denoise: OFF by default.** `db4`, level 3, soft universal
  threshold. Opt-in; can over-smooth sharp fall spikes if applied blindly,
  which is exactly the signal fall detection needs — hence off by default.
- **drop_null_subcarriers: ON.**

Rationale: start with the least destructive chain (Hampel only), let each
experiment enable heavier denoising deliberately.

## Feature design: interpretable, subcarrier-count-independent

Window features collapse subcarriers to a per-sample motion signal plus a few
cross-subcarrier statistics, so the feature vector length is fixed regardless
of how many subcarriers survive null-removal. Features map directly onto the
physical signatures:

- presence → `amp_var`, `amp_std`, `subcarrier_var_mean`, `corr_across_subcarriers`
- fall → `amp_max_abs_diff` (spike) + `post_event_stillness_s` (then-still) +
  `n_disturbance_peaks`
- counting → all of the above from **two nodes** concatenated

## People counting capped at 2 (hard)

`ml/counting.py` `MAX_PEOPLE = 2`. Distinguishing 0/1/2 is the honest limit of
commodity CSI for a hobby build; 3+ is unreliable and not attempted. The
3-class target and the constant encode that cap. See `docs/limitations.md`.

## ML honesty

Every training/eval script prints `SYNTHETIC VALIDATION - NOT REAL ACCURACY`.
Synthetic data proves the code path only. The near-perfect synthetic scores
are expected (the generator and the features share assumptions) and must not
be read as real-world performance.

## Two-node design & single-node fallback

Counting expects features from two RX nodes (`_n1`/`_n2` suffixes) for spatial
diversity. When only one node is online, `single_node_fallback_features`
duplicates it to preserve model input shape, with a documented accuracy
penalty rather than a silent failure.

## PlatformIO install outcome

**SUCCESS.** `py -m pip install platformio` installed PlatformIO Core 6.1.19
on Python 3.14 with exit 0. It is invokable as `py -m platformio` (or
`.venv/Scripts/platformio.exe`). Note: the install downgraded `starlette`
(1.3.1 → 0.52.1) as a shared transitive dep; **streamlit 1.58.0 still imports
and runs fine** (verified), because streamlit's server is tornado-based, not
starlette. No action needed.

Caveat: PlatformIO is installed and usable, but *building* the ESP-IDF CSI
firmware with it downloads the full espidf platform + Xtensa toolchain
(~1 GB+). That download was intentionally **not** performed now because there
is no board to flash yet and it is a large one-time fetch. The primary
supported path remains the standalone ESP-IDF installer (`idf.py`); a
`firmware/platformio.ini` is provided for whoever prefers the PlatformIO
wrapper. Neither is a blocker — the firmware source is vendored and reviewed,
so on board arrival it is a flash-and-go step documented in
`docs/flashing-guide.md`.
