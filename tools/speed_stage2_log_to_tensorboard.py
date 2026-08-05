"""Stream speed_stage2 JSONL metrics into TensorBoard event files."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="speed_stage2 torchrun.log")
    parser.add_argument("--out", type=Path, default=None, help="TensorBoard event directory")
    parser.add_argument("--follow", action="store_true", help="Continue polling for appended metrics")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def _write_line(writer: SummaryWriter, line: str, seen_steps: set[int]) -> None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return
    if not isinstance(record, dict) or "iteration" not in record:
        return
    step = int(record["iteration"])
    if step in seen_steps:
        return
    seen_steps.add(step)
    for name, value in record.items():
        if name != "iteration" and isinstance(value, (int, float)):
            writer.add_scalar(f"train/{name}", value, step)
    writer.flush()


def main() -> None:
    args = _parse_args()
    log_path = args.log.expanduser().resolve()
    out_path = (args.out or log_path.parent / "tensorboard").expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_path))
    seen_steps: set[int] = set()
    position = 0
    try:
        while True:
            if log_path.is_file():
                with log_path.open(encoding="utf-8", errors="replace") as stream:
                    stream.seek(position)
                    for line in stream:
                        _write_line(writer, line, seen_steps)
                    position = stream.tell()
            if not args.follow:
                break
            time.sleep(args.poll_seconds)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
