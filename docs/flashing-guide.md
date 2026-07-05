# Flashing Guide — ESP32-CSI-Tool onto 3 boards

Exact, unambiguous steps to flash the two firmware roles onto your 3
ESP32-WROOM-32 boards (CP2102 USB) and verify CSI is flowing. Do this when
the boards arrive. Nothing here needs a board until Section 3.

**Roles / topology**

| Board | Role project | `SHOULD_COLLECT_CSI` | Purpose |
|-------|--------------|----------------------|---------|
| 1 | `active_ap`  | `n` | TX: broadcasts an AP + sends packets so RX nodes have traffic to measure |
| 2 | `active_sta` | `y` | RX node #1: connects to the AP, logs CSI over serial |
| 3 | `active_sta` | `y` | RX node #2: same firmware, different physical spot (spatial diversity) |

You may alternatively use your home router as the packet source instead of
Board 1; then flash all traffic considerations onto the STA config (point
`ESP_WIFI_SSID`/`PASSWORD` at the router). Using a dedicated AP board is more
controllable, so this guide assumes Board 1 = AP.

The firmware lives (vendored) at `firmware/esp32-csi-tool/`.

---

## 1. Install the toolchain (ESP-IDF — primary path)

ESP32-CSI-Tool is a native **ESP-IDF** project. Install ESP-IDF **v4.3.x**
(the version the tool targets; newer 5.x may need minor fixes — see
Troubleshooting).

**Windows (recommended: the ESP-IDF Tools Installer):**

1. Download the "ESP-IDF Tools Installer" (Offline) from Espressif and run it,
   OR use VS Code → Extensions → **Espressif IDF** → "Configure ESP-IDF
   extension" → choose ESP-IDF **v4.3**.
2. This gives you an "ESP-IDF Command Prompt" (or the VS Code terminal) with
   `idf.py` on PATH and the Xtensa toolchain installed.
3. Verify: open the ESP-IDF prompt and run `idf.py --version`.

> PlatformIO alternative (already installed here as `py -m platformio`):
> see `firmware/platformio.ini`. It downloads its own toolchain (~1 GB) on
> first build. Steps below use `idf.py`; the PlatformIO equivalents are noted
> at the end.

## 2. Identify the COM ports (CP2102)

1. Plug in one board. Open **Device Manager → Ports (COM & LPT)**.
2. A **"Silicon Labs CP210x USB to UART Bridge (COMx)"** entry appears. Note
   the `COMx` number. If it does not appear, install the **CP210x VCP driver**
   from Silicon Labs.
3. Plug each board one at a time and record its COM number. Label the boards
   physically (AP, RX1, RX2) so you don't mix them up.

## 3. Configure and flash Board 1 — AP / TX (`active_ap`)

From the ESP-IDF prompt:

```bash
cd firmware/esp32-csi-tool/active_ap
idf.py set-target esp32
idf.py menuconfig
```

In `menuconfig`:

- **ESP32 CSI Tool Config**
  - `WiFi Channel` → **6** (pick one channel and use the SAME on all 3 boards)
  - `WiFi SSID` → e.g. **`csi_ap`**  (this is the network the RX nodes join)
  - `WiFi Password` → e.g. **`csipassword`** (>= 8 chars for WPA2)
  - `Should this ESP32 collect and print CSI data?` → **n** (AP only transmits)
  - `Packet TX Rate` → **100** (packets/sec; RX CSI rate tracks this)
- **Serial flasher config**
  - `'idf.py monitor' baud rate` → **Custom** → **921600**
- **Component config → Common ESP32-related**
  - `UART console baud rate` → **921600**
- **Partition Table** → keep default (`Single factory app, no OTA`) unless the
  build complains the app is too large; then choose `Single factory app
  (large)`.

Save & exit, then flash (replace `COM7` with the AP board's port):

```bash
idf.py -p COM7 flash monitor
```

You should see the AP start and `AP started. ...` style logs. Leave it
powered. Exit monitor with `Ctrl+]`.

## 4. Configure and flash Boards 2 & 3 — RX nodes (`active_sta`)

Do this **twice**, once per RX board.

```bash
cd ../active_sta
idf.py set-target esp32
idf.py menuconfig
```

In `menuconfig`:

- **ESP32 CSI Tool Config**
  - `WiFi Channel` → **6** (must match the AP)
  - `WiFi SSID` → **`csi_ap`** (must match the AP exactly)
  - `WiFi Password` → **`csipassword`** (must match the AP)
  - `Should this ESP32 collect and print CSI data?` → **y**
  - `(Advanced) Should we only collect LLTF?` → **y** (default; gives
    `len=128` → 64 subcarriers, which is what the pipeline + synthetic data
    assume)
  - `Send CSI data to Serial` → **y**
  - `Send CSI data to SD` → **n** (no SD card)
- Same **921600** baud settings as the AP (both places).

Flash RX node #1 (replace `COM5`):

```bash
idf.py -p COM5 flash monitor
```

Within a few seconds you should see lines beginning with `CSI_DATA,STA,...`.
That is CSI flowing. Exit with `Ctrl+]`. Repeat for RX node #2 on its own COM
port (e.g. `COM6`) — **same firmware, no menuconfig change needed** beyond the
port.

## 5. Verify CSI capture on the host

With an RX node running, capture straight to a labeled session file using the
project's serial reader (baud 921600 default):

```bash
# from the project root, venv activated
py -m pipeline.capture --port COM5 --label empty --node node1 --seconds 30
```

(Empty-room baseline first. Then repeat with `--label walking`, `--label
fall`, `--label 2people`, moving as described.) Each run writes
`data/raw/<UTCstamp>_<label>_<node>.csv` plus a `.meta.json` sidecar.

Quick sanity check without recording (raw echo, Windows):

```bash
py -c "import serial; s=serial.Serial('COM5',921600,timeout=1); [print(s.readline().decode('utf-8','replace').strip()) for _ in range(10)]"
```

You should see `CSI_DATA,STA,<mac>,...,[ ... ]` lines. If you do, the whole
chain (board → serial → parser) is confirmed on real hardware.

Then run the pipeline on a recorded session:

```bash
py -m pipeline.run_pipeline --session data/raw/<the-file>.csv
```

## 6. Setting device time (optional)

Device wall-clock is not set by default; the host attaches its own timestamp,
which is sufficient. If you want on-device UNIX time in `real_timestamp`,
while `idf.py monitor` is running type: `SETTIME: 1751760000` then Enter (use
the current UNIX epoch seconds).

---

## PlatformIO equivalents (optional)

```bash
# configure (per role/env)
py -m platformio run -e rx_sta -t menuconfig
# build + upload + monitor
py -m platformio run -e rx_sta -t upload -t monitor --upload-port COM5
```
See `firmware/platformio.ini`. Set `monitor_speed`/`upload_speed` to 921600
(already set there).

## Troubleshooting

- **No `CSI_DATA` lines:** confirm the RX board's SSID/password/channel match
  the AP exactly; confirm `SHOULD_COLLECT_CSI = y` on the RX; confirm the AP is
  powered and its packet rate > 0.
- **Garbled serial / dropped lines:** baud mismatch. Both the firmware
  (menuconfig UART console baud) and the host reader must be **921600**. Try
  1000000 or 115200 consistently on both if 921600 is unstable on your USB
  stack (record which works in `docs/decision-log.md`).
- **`idf.py` build errors on ESP-IDF 5.x:** the tool targets 4.3.x. Either
  install 4.3.x, or apply the small API renames flagged by the compiler
  (`esp_spi_flash.h` → `spi_flash_mmap.h`, etc.). Prefer 4.3.x to avoid this.
- **Upload fails / port busy:** close any open serial monitor (only one
  process can hold the COM port). Some boards need the BOOT button held during
  the first flash.
- **Low sample rate:** raise the AP `Packet TX Rate`, keep baud at 921600+,
  and prefer LLTF-only. Measure the true rate with
  `firmware/esp32-csi-tool/python_utils/serial_measure_rate.py`.
