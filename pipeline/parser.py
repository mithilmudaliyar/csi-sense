"""Parser for ESP32-CSI-Tool serial output lines.

CONFIRMED FORMAT (read from vendored source at
firmware/esp32-csi-tool/_components/csi_component.h, function _wifi_csi_cb):

Each CSI packet is one CSV line:

    CSI_DATA,<role>,<mac>,<rssi>,<rate>,<sig_mode>,<mcs>,<bandwidth>,
    <smoothing>,<not_sounding>,<aggregation>,<stbc>,<fec_coding>,<sgi>,
    <noise_floor>,<ampdu_cnt>,<channel>,<secondary_channel>,
    <local_timestamp>,<ant>,<sig_len>,<rx_state>,<real_time_set>,
    <real_timestamp>,<len>,[<int8 buffer, space separated> ]

- 25 comma-separated metadata fields, then a bracketed buffer.
- The buffer is the raw esp_wifi CSI buffer: interleaved signed 8-bit
  pairs, IMAGINARY part first then REAL part for each subcarrier slot
  (see ESP-IDF wifi_csi_info_t docs; the tool itself computes
  amplitude = sqrt(buf[2i]^2 + buf[2i+1]^2) and
  phase = atan2(buf[2i], buf[2i+1])).
- len is data->len, the FULL CSI buffer size (e.g. 384). IMPORTANT: with
  CONFIG_SHOULD_COLLECT_ONLY_LLTF (the tool default) the firmware prints
  only the first 128 int8 values (64 LLTF subcarriers) even though len still
  says 384. Confirmed against the vendored example_csi.csv (len=384, 128
  values printed). The parser therefore treats len as advisory and decodes
  whatever count is actually printed. buf_len = printed count,
  advertised_len = the len field.
- local_timestamp is the ESP32 microsecond clock; real_timestamp is
  the tool's steady-clock seconds (float), only meaningful once time
  is set on the device.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

CSI_PREFIX = "CSI_DATA"
N_META_FIELDS = 25  # fields before the bracketed CSI buffer

# Column names for the 25 metadata fields, taken verbatim from
# _print_csi_csv_header() in csi_component.h (minus the trailing
# CSI_DATA buffer column).
META_COLUMNS = (
    "type", "role", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel",
    "local_timestamp", "ant", "sig_len", "rx_state", "real_time_set",
    "real_timestamp", "len",
)


class CSIParseError(ValueError):
    """Raised when a line that claims to be CSI data cannot be parsed."""


@dataclass(frozen=True)
class CSIFrame:
    """One parsed CSI packet."""

    role: str
    mac: str
    rssi: int
    channel: int
    noise_floor: int
    local_timestamp_us: int
    real_timestamp: float
    sig_len: int
    buf_len: int          # number of int8 values ACTUALLY printed in this line
    advertised_len: int   # the `len` metadata field (full buffer size; advisory)
    csi: np.ndarray = field(repr=False)  # complex64, one value per subcarrier slot

    @property
    def n_subcarriers(self) -> int:
        return int(self.csi.shape[0])

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.csi).astype(np.float32)

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.csi).astype(np.float32)


def is_csi_line(line: str) -> bool:
    return line.startswith(CSI_PREFIX + ",")


def parse_csi_line(line: str) -> CSIFrame:
    """Parse a single ESP32-CSI-Tool serial line into a CSIFrame.

    Raises CSIParseError on malformed input. Callers that stream noisy
    serial data should pre-filter with is_csi_line() and catch
    CSIParseError for truncated lines (common at capture start/stop).
    """
    line = line.strip()
    if not is_csi_line(line):
        raise CSIParseError(f"Not a CSI_DATA line: {line[:40]!r}")

    bracket_open = line.find("[")
    bracket_close = line.rfind("]")
    if bracket_open == -1 or bracket_close == -1 or bracket_close < bracket_open:
        raise CSIParseError("Missing or malformed CSI buffer brackets")

    meta_part = line[:bracket_open].rstrip(",")
    meta = meta_part.split(",")
    if len(meta) != N_META_FIELDS:
        raise CSIParseError(
            f"Expected {N_META_FIELDS} metadata fields, got {len(meta)}"
        )

    raw_str = line[bracket_open + 1 : bracket_close].strip()
    if not raw_str:
        raise CSIParseError("Empty CSI buffer")

    try:
        raw = np.array([int(v) for v in raw_str.split()], dtype=np.int16)
    except ValueError as exc:
        raise CSIParseError(f"Non-integer value in CSI buffer: {exc}") from exc

    try:
        advertised_len = int(meta[24])
        # IMPORTANT (confirmed against the vendored example_csi.csv): the `len`
        # metadata field is data->len (the FULL CSI buffer size, e.g. 384), but
        # with CONFIG_SHOULD_COLLECT_ONLY_LLTF (the default) the firmware prints
        # only the first 128 int8 values. So the printed count legitimately
        # differs from `len`. We therefore treat `len` as advisory and validate
        # only what actually matters for decoding: a non-empty, EVEN-length
        # integer buffer. (Truncated serial lines are caught by the missing-`]`
        # and non-integer checks above, and by requiring even length here.)
        if raw.size % 2 != 0:
            raise CSIParseError(f"Odd CSI buffer length {raw.size}")
        raw = raw.astype(np.float32)
        # ESP-IDF layout: [imag, real] per subcarrier slot.
        imag = raw[0::2]
        real = raw[1::2]
        csi = (real + 1j * imag).astype(np.complex64)

        return CSIFrame(
            role=meta[1],
            mac=meta[2],
            rssi=int(meta[3]),
            channel=int(meta[16]),
            noise_floor=int(meta[14]),
            local_timestamp_us=int(meta[18]),
            real_timestamp=float(meta[23]),
            sig_len=int(meta[20]),
            buf_len=int(raw.size),
            advertised_len=advertised_len,
            csi=csi,
        )
    except CSIParseError:
        raise
    except (ValueError, IndexError) as exc:
        raise CSIParseError(f"Malformed metadata field: {exc}") from exc


def parse_stream(lines) -> "list[CSIFrame]":
    """Parse an iterable of raw serial lines, skipping non-CSI/malformed lines."""
    frames: list[CSIFrame] = []
    n_bad = 0
    for line in lines:
        if not is_csi_line(line.strip() if isinstance(line, str) else ""):
            continue
        try:
            frames.append(parse_csi_line(line))
        except CSIParseError as exc:
            n_bad += 1
            logger.debug("Skipping malformed CSI line: %s", exc)
    if n_bad:
        logger.warning("Skipped %d malformed CSI lines", n_bad)
    return frames
