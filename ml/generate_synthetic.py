"""CLI: generate synthetic CSI sessions and write them to data/synthetic/.

Each scenario is parsed through the real parser and written with the real
SessionWriter, so the output files are byte-identical in shape to what live
hardware capture will produce (same columns, same .meta.json sidecar).

Run:
    py -m ml.generate_synthetic                  # all 4 scenarios, 20s each
    py -m ml.generate_synthetic --duration 10    # shorter
    py -m ml.generate_synthetic --scenario fall
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.synthetic import SCENARIOS, generate_session
from pipeline.parser import parse_csi_line
from pipeline.storage import SessionWriter

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic"


def write_scenario(scenario: str, out_dir: Path, duration_s: float, sr: float, seed: int) -> Path:
    sess = generate_session(scenario, duration_s=duration_s, sample_rate_hz=sr, seed=seed)
    writer = SessionWriter(
        out_dir=out_dir,
        label=sess.label,
        node_id="synthetic",
        notes=f"Synthetic {scenario} session (seed={seed}). NOT real capture.",
        source="synthetic",
        sample_rate_hz_nominal=sr,
    )
    for i, line in enumerate(sess.lines):
        frame = parse_csi_line(line)
        # Monotonic host timestamps spaced at the nominal rate.
        writer.append(host_ts=i / sr, frame=frame)
    meta = writer.close()
    return writer.data_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic CSI sessions to disk.")
    ap.add_argument("--scenario", choices=SCENARIOS, help="only this scenario (default: all)")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds per session")
    ap.add_argument("--rate", type=float, default=50.0, help="nominal sample rate Hz")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    ap.add_argument("--seed", type=int, default=42, help="base RNG seed")
    args = ap.parse_args()

    out_dir = Path(args.out)
    scenarios = [args.scenario] if args.scenario else list(SCENARIOS)
    written = []
    for i, sc in enumerate(scenarios):
        path = write_scenario(sc, out_dir, args.duration, args.rate, args.seed + i)
        print(f"  {sc:8s} -> {path.name}")
        written.append(path)
    print(f"Wrote {len(written)} synthetic session(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
