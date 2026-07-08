#!/usr/bin/env python3

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch


def _is_tensor(x: Any) -> bool:
    return torch.is_tensor(x)


def _tensor_info(t: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": int(t.numel()),
        "requires_grad": bool(getattr(t, "requires_grad", False)),
    }


def _summarize_nested(obj: Any, prefix: str = "", max_list_items: int = 20) -> Dict[str, Any]:
    """Summarize a nested Python object without printing huge content."""
    if _is_tensor(obj):
        return {"type": "tensor", **_tensor_info(obj)}

    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return {"type": type(obj).__name__, "value": obj}

    if isinstance(obj, (list, tuple)):
        out = {"type": type(obj).__name__, "len": len(obj)}
        preview = []
        for i, v in enumerate(obj[:max_list_items]):
            preview.append(_summarize_nested(v, prefix=f"{prefix}[{i}]"))
        if len(obj) > max_list_items:
            preview.append({"type": "...", "note": f"truncated after {max_list_items} items"})
        out["preview"] = preview
        return out

    if isinstance(obj, dict):
        out = {"type": "dict", "len": len(obj)}
        keys = list(obj.keys())
        out["keys_preview"] = keys[:max_list_items]
        # Add a small preview for common keys
        common_preview = {}
        for k in keys[:max_list_items]:
            try:
                common_preview[str(k)] = _summarize_nested(obj[k], prefix=f"{prefix}.{k}")
            except Exception as e:
                common_preview[str(k)] = {"type": "error", "error": repr(e)}
        if len(keys) > max_list_items:
            common_preview["..."] = {"type": "...", "note": f"truncated after {max_list_items} keys"}
        out["items_preview"] = common_preview
        return out

    return {"type": type(obj).__name__, "repr": repr(obj)[:500]}


def inspect_state_dict(sd: Dict[str, Any], *, title: str, max_keys: int = 200) -> None:
    print("=" * 100)
    print(title)
    print(f"num_keys: {len(sd)}")

    # Shape / dtype stats
    dtype_counter = Counter()
    shape_counter = Counter()
    module_prefix_counter = Counter()

    def module_prefix(k: str) -> str:
        # Heuristic: take first 2 segments as module prefix
        parts = k.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]

    for k, v in sd.items():
        module_prefix_counter[module_prefix(k)] += 1
        if _is_tensor(v):
            dtype_counter[str(v.dtype)] += 1
            shape_counter[str(tuple(v.shape))] += 1

    print("-- module prefix frequency (top 30) --")
    for name, cnt in module_prefix_counter.most_common(30):
        print(f"  {name:40s} {cnt}")

    print("-- dtype frequency --")
    for dt, cnt in dtype_counter.most_common():
        print(f"  {dt:15s} {cnt}")

    print("-- most common shapes (top 30) --")
    for sh, cnt in shape_counter.most_common(30):
        print(f"  {sh:30s} {cnt}")

    print("-- key -> tensor info preview --")
    shown = 0
    for k in list(sd.keys()):
        v = sd[k]
        if _is_tensor(v):
            print(f"  {k}: {_tensor_info(v)}")
            shown += 1
        else:
            print(f"  {k}: type={type(v).__name__}")
            shown += 1
        if shown >= max_keys:
            if len(sd) > max_keys:
                print(f"  ... truncated after {max_keys} keys")
            break


def inspect_optimizer_pth(path: Path) -> None:
    print("=" * 100)
    print(f"Loading optimizer checkpoint: {path}")
    obj = torch.load(path, map_location="cpu")
    print(f"Top-level type: {type(obj).__name__}")

    # Common cases:
    # 1) obj is dict with keys like: 'optimizers', 'schedulers', 'state', 'param_groups' ...
    # 2) obj is directly an optimizer.state_dict()
    if isinstance(obj, dict):
        print(f"Top-level keys ({len(obj)}): {list(obj.keys())[:50]}")

    # Try to detect optimizer state_dict format
    if isinstance(obj, dict) and "state" in obj and "param_groups" in obj:
        opt_sd = obj
    elif isinstance(obj, dict) and "optimizer" in obj and isinstance(obj["optimizer"], dict) and "state" in obj["optimizer"]:
        opt_sd = obj["optimizer"]
    else:
        opt_sd = None

    if opt_sd is None:
        print("Could not confidently parse this as optimizer.state_dict(). Printing a summarized preview:")
        print(json.dumps(_summarize_nested(obj), indent=2)[:20000])
        return

    print("-- optimizer.state_dict() summary --")
    print(f"num_param_groups: {len(opt_sd.get('param_groups', []))}")
    print(f"num_state_entries: {len(opt_sd.get('state', {}))}")

    # Print param group hyperparams
    for i, g in enumerate(opt_sd.get("param_groups", [])):
        g2 = {k: v for k, v in g.items() if k != "params"}
        print(f"param_group[{i}] hyperparams: {g2}")
        if "params" in g:
            print(f"param_group[{i}] num_params: {len(g['params'])}")

    # Print state entry shapes
    state = opt_sd.get("state", {})
    shown = 0
    print("-- state entry preview (show first 50 params) --")
    for pid, st in state.items():
        if not isinstance(st, dict):
            print(f"param_id={pid}: state_type={type(st).__name__}")
            continue
        # Many optimizers store tensors like exp_avg, exp_avg_sq, step, etc.
        tensor_fields = {k: _tensor_info(v) for k, v in st.items() if _is_tensor(v)}
        non_tensor_fields = {k: v for k, v in st.items() if not _is_tensor(v)}
        print(f"param_id={pid}: tensor_fields={list(tensor_fields.keys())}, non_tensor_fields={list(non_tensor_fields.keys())}")
        for k, info in tensor_fields.items():
            print(f"  - {k}: {info}")
        shown += 1
        if shown >= 50:
            if len(state) > 50:
                print("  ... truncated")
            break


def inspect_safetensors(path: Path) -> None:
    print("=" * 100)
    print(f"Loading model weights (safetensors): {path}")

    try:
        from safetensors.torch import safe_open
    except Exception as e:
        print("Failed to import safetensors. Install with: pip install safetensors")
        print(f"Import error: {repr(e)}")
        return

    keys = []
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"num_keys: {len(keys)}")

        # Stats
        dtype_counter = Counter()
        shape_counter = Counter()
        module_prefix_counter = Counter()

        def module_prefix(k: str) -> str:
            parts = k.split(".")
            return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]

        # We avoid reading all tensors (can be huge). Read first N for preview and compute stats on metadata.
        for k in keys:
            module_prefix_counter[module_prefix(k)] += 1
            meta = f.get_tensor(k)
            dtype_counter[str(meta.dtype)] += 1
            shape_counter[str(tuple(meta.shape))] += 1

        print("-- module prefix frequency (top 30) --")
        for name, cnt in module_prefix_counter.most_common(30):
            print(f"  {name:40s} {cnt}")

        print("-- dtype frequency --")
        for dt, cnt in dtype_counter.most_common():
            print(f"  {dt:15s} {cnt}")

        print("-- most common shapes (top 30) --")
        for sh, cnt in shape_counter.most_common(30):
            print(f"  {sh:30s} {cnt}")

        print("-- key -> tensor info preview (first 200 tensors) --")
        for i, k in enumerate(keys[:200]):
            t = f.get_tensor(k)
            print(f"  {k}: {_tensor_info(t)}")
        if len(keys) > 200:
            print("  ... truncated")


def inspect_json(path: Path, title: str) -> None:
    if not path.exists():
        return
    print("=" * 100)
    print(f"{title}: {path}")
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"Failed to read json: {repr(e)}")
        return
    print(json.dumps(_summarize_nested(data), indent=2)[:20000])


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect HumanoidVerse checkpoint artifacts (.pth optimizer + model.safetensors)")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Checkpoint directory (e.g. results/bfmzero-isaac/checkpoint). If provided, the script will inspect known files inside.",
    )
    parser.add_argument(
        "--pth",
        type=str,
        default=None,
        help="Path to a .pth file to inspect (e.g. .../optimizers.pth)",
    )
    parser.add_argument(
        "--safetensors",
        type=str,
        default=None,
        help="Path to a model .safetensors file to inspect (e.g. .../checkpoint/model/model.safetensors)",
    )
    args = parser.parse_args()

    if args.checkpoint_dir is None and args.pth is None and args.safetensors is None:
        raise SystemExit("Provide at least one of --checkpoint_dir / --pth / --safetensors")

    if args.checkpoint_dir is not None:
        ckpt = Path(args.checkpoint_dir)
        if not ckpt.exists():
            raise SystemExit(f"checkpoint_dir not found: {ckpt}")

        # Standard HumanoidVerse layout
        inspect_json(ckpt / "config.json", title="checkpoint/config.json (TrainConfig snapshot)")
        inspect_json(ckpt / "init_kwargs.json", title="checkpoint/init_kwargs.json (agent/model build kwargs)")
        inspect_json(ckpt / "train_status.json", title="checkpoint/train_status.json")

        opt_pth = ckpt / "optimizers.pth"
        if opt_pth.exists():
            inspect_optimizer_pth(opt_pth)

        st = ckpt / "model" / "model.safetensors"
        if st.exists():
            inspect_safetensors(st)

        print("=" * 100)
        print("NOTE")
        print("- optimizers.pth 只包含优化器状态（如 Adam 的 exp_avg/exp_avg_sq），无法完整还原网络结构。")
        print("- model.safetensors 才是模型权重；其参数名/shape 可以反推出模块划分，但精确网络结构仍需要配合源码（agent/model.py）与 init_kwargs/config。")
        return

    if args.pth is not None:
        inspect_optimizer_pth(Path(args.pth))

    if args.safetensors is not None:
        inspect_safetensors(Path(args.safetensors))


if __name__ == "__main__":
    main()
