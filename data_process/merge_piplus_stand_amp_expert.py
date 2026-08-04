"""Append a PiPlus default-pose trajectory to a Stage2 AMP expert dataset."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import joblib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-dataset", type=Path, required=True)
    parser.add_argument("--stand-pose", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_stand_motion(path: Path, reference: dict[str, object]) -> dict[str, object]:
    with path.open("rb") as file:
        stand = pickle.load(file)
    if not isinstance(stand, dict):
        raise ValueError(f"Stand pose must be a mapping, got {type(stand)!r}")

    required = ("framerate", "base_pos_w", "base_quat_w", "joint_pos", "joint_names")
    missing = [key for key in required if key not in stand]
    if missing:
        raise KeyError(f"Stand pose is missing fields: {missing}")
    if list(stand["joint_names"]) != list(reference["joint_names"]):
        raise ValueError("Stand pose joint_names do not match the AMP expert dataset joint order")

    root_pos = np.asarray(stand["base_pos_w"], dtype=np.float32)
    root_quat_wxyz = np.asarray(stand["base_quat_w"], dtype=np.float32)
    dof = np.asarray(stand["joint_pos"], dtype=np.float32)
    frame_count = root_pos.shape[0]
    if root_pos.shape != (frame_count, 3) or root_quat_wxyz.shape != (frame_count, 4):
        raise ValueError("Stand root position/quaternion dimensions are inconsistent")
    if dof.shape != (frame_count, len(reference["joint_names"])):
        raise ValueError("Stand joint positions do not match the expected frame or joint dimensions")
    if not np.isfinite(root_pos).all() or not np.isfinite(root_quat_wxyz).all() or not np.isfinite(dof).all():
        raise ValueError("Stand pose contains non-finite values")
    if not np.allclose(np.linalg.norm(root_quat_wxyz, axis=1), 1.0, atol=1.0e-4):
        raise ValueError("Stand pose contains non-unit root quaternions")

    root_rot_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    pose_aa = np.zeros((frame_count, *np.asarray(reference["pose_aa"]).shape[1:]), dtype=np.float32)
    return {
        "root_trans_offset": root_pos,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_rot_xyzw,
        "smpl_joints": np.zeros((frame_count, *np.asarray(reference["smpl_joints"]).shape[1:]), dtype=np.float32),
        "fps": int(stand["framerate"]),
        "joint_names": list(reference["joint_names"]),
        "motion_name": "default_pose_piplus_s_lse_stand",
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    expert = joblib.load(args.expert_dataset)
    if not isinstance(expert, dict) or not expert:
        raise ValueError("AMP expert dataset must be a non-empty mapping")

    reference = next(iter(expert.values()))
    stand_key = "default_pose_piplus_s_lse_stand"
    if stand_key in expert:
        raise ValueError(f"AMP expert dataset already contains {stand_key!r}")
    merged = dict(expert)
    merged[stand_key] = load_stand_motion(args.stand_pose, reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(merged, args.output)
    print(f"Saved {len(merged)} motions to {args.output}")


if __name__ == "__main__":
    main()
