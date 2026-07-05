"""CLI: capture live CSI from an ESP32 serial port into a labeled session.

Used on hardware. Reads CSI_DATA lines from a COM port via the (mock-tested)
CSISerialReader and appends them to data/raw/<UTCstamp>_<label>_<node>.csv
with a metadata sidecar.

Run (once a board is flashed and connected):
    py -m pipeline.capture --port COM5 --label empty --node node1 --seconds 30

Labels: empty | walking | fall | 2people | unlabeled
Stop early with Ctrl+C — whatever was captured is flushed and saved.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pipeline.serial_reader import DEFAULT_BAUDRATE, CSISerialReader
from pipeline.storage import VALID_LABELS, SessionWriter

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> int:
    ap = argparse.ArgumentParser(description="Capture live CSI to a labeled session.")
    ap.add_argument("--port", required=True, help="serial port, e.g. COM5")
    ap.add_argument("--label", required=True, choices=VALID_LABELS, help="activity label")
    ap.add_argument("--node", default="node1", help="RX node id (node1/node2)")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE, help="serial baud (match firmware)")
    ap.add_argument("--seconds", type=float, default=30.0, help="capture duration")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    ap.add_argument("--notes", default="", help="free-text notes for the sidecar")
    args = ap.parse_args()

    reader = CSISerialReader.open_port(args.port, baudrate=args.baud)
    writer = SessionWriter(
        out_dir=args.out, label=args.label, node_id=args.node,
        notes=args.notes, source="serial",
    )
    print(f"Capturing '{args.label}' on {args.port} @ {args.baud} for {args.seconds:.0f}s "
          f"(node={args.node}). Ctrl+C to stop early.")
    deadline = time.time() + args.seconds
    try:
        for host_ts, frame in reader.frames():
            writer.append(host_ts, frame)
            if reader.n_frames % 100 == 0:
                print(f"  {reader.n_frames} frames...", end="\r")
            if time.time() >= deadline:
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    meta = writer.close()
    print(f"\nSaved {reader.n_frames} frames ({reader.n_malformed} malformed skipped) "
          f"to {writer.data_path}")
    print(f"Session id: {meta.session_id}  |  subcarriers: {meta.n_subcarriers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
