from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat

import joblib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show the structure of a pickle/joblib pkl file.")
    parser.add_argument("path", type=Path, help="Path to the .pkl file.")
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Maximum dict/list items to print per level. Use 0 to print all items.",
    )
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum nested depth to inspect.")
    parser.add_argument("--stats", action="store_true", help="Print min/max for numeric arrays.")
    return parser.parse_args()


def summarize_array(value: np.ndarray, stats: bool) -> str:
    summary = f"{type(value).__name__}, shape={value.shape}, dtype={value.dtype}"
    if stats and value.size and np.issubdtype(value.dtype, np.number):
        summary += f", min={np.nanmin(value):.6g}, max={np.nanmax(value):.6g}"
    return summary


def summarize_value(value: object, max_items: int, max_depth: int, depth: int, stats: bool) -> str:
    next_indent = "  " * (depth + 1)
    limit = None if max_items <= 0 else max_items

    if isinstance(value, np.ndarray):
        return summarize_array(value, stats)

    if isinstance(value, dict):
        if depth >= max_depth:
            keys = list(value.keys()) if limit is None else list(value.keys())[:limit]
            return f"dict, len={len(value)}, keys={keys!r}"
        lines = [f"dict, len={len(value)}"]
        for idx, (key, item) in enumerate(value.items()):
            if limit is not None and idx >= limit:
                lines.append(f"{next_indent}... {len(value) - limit} more item(s)")
                break
            lines.append(
                f"{next_indent}{key!r}: "
                f"{summarize_value(item, max_items, max_depth, depth + 1, stats)}"
            )
        return "\n".join(lines)

    if isinstance(value, (list, tuple)):
        if depth >= max_depth:
            return f"{type(value).__name__}, len={len(value)}"
        lines = [f"{type(value).__name__}, len={len(value)}"]
        items = value if limit is None else value[:limit]
        for idx, item in enumerate(items):
            lines.append(
                f"{next_indent}[{idx}]: "
                f"{summarize_value(item, max_items, max_depth, depth + 1, stats)}"
            )
        if limit is not None and len(value) > limit:
            lines.append(f"{next_indent}... {len(value) - limit} more item(s)")
        return "\n".join(lines)

    if isinstance(value, (str, int, float, bool, type(None))):
        return f"{type(value).__name__}: {value!r}"

    return f"{type(value).__name__}: {pformat(value)[:200]}"


def main() -> None:
    args = parse_args()
    data = joblib.load(args.path)
    print(f"File: {args.path}")
    print(summarize_value(data, args.max_items, args.max_depth, depth=0, stats=args.stats))


if __name__ == "__main__":
    main()
