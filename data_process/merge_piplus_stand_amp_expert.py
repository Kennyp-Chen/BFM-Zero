"""Append raw PiPlus trajectories to a normalized Stage2 AMP expert dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.convert_gmr_lafan import (
    install_numpy_pickle_compat,
    load_dof_metadata,
    make_body_aligned_pose_aa,
    normalize_quat_xyzw,
)

DEFAULT_ROBOT_XML = (
    REPO_ROOT
    / "humanoidverse/data/robots/piplus/PiPlus_S_12L8A0G2H1W_LSE_260611/xml/"
    "PiPlus_S_12L8A0G2H1W_LSE_260611.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-dataset", type=Path, required=True, help="Existing normalized AMP dataset.")
    parser.add_argument("--motion", type=Path, action="append", default=[], help="Raw PiPlus trajectory to append; repeatable.")
    parser.add_argument("--output", type=Path, required=True, help="New dataset path; existing files are never overwritten.")
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--target-fps", type=int, default=30, help="Output frame rate; source rate must divide exactly.")
    return parser.parse_args()


def validate_reference(reference: dict[str, object]) -> tuple[list[str], tuple[int, int]]:
    required = ("root_trans_offset", "root_rot", "dof", "pose_aa", "smpl_joints", "fps", "joint_names")
    missing = [key for key in required if key not in reference]
    if missing:
        raise KeyError(f"AMP reference motion is missing fields: {missing}")

    joint_names = list(reference["joint_names"])
    dof = np.asarray(reference["dof"])
    pose_aa = np.asarray(reference["pose_aa"])
    smpl_joints = np.asarray(reference["smpl_joints"])
    if dof.ndim != 2 or dof.shape[1] != len(joint_names):
        raise ValueError("AMP reference DoF shape does not match its joint names.")
    if pose_aa.ndim != 3 or pose_aa.shape[2] != 3:
        raise ValueError("AMP reference pose_aa must have shape (T, J, 3).")
    if smpl_joints.ndim != 3 or smpl_joints.shape[2] != 3:
        raise ValueError("AMP reference smpl_joints must have shape (T, J, 3).")
    return joint_names, tuple(smpl_joints.shape[1:])


def load_raw_motion(
    path: Path,
    *,
    reference_joint_names: list[str],
    smpl_joint_shape: tuple[int, int],
    dof_axes: np.ndarray,
    body_names: list[str],
    body_to_joint: dict[str, str],
    target_fps: int,
) -> dict[str, object]:
    raw = joblib.load(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: raw trajectory must be a mapping, got {type(raw)!r}.")

    required = ("framerate", "base_pos_w", "base_quat_w", "joint_pos", "joint_names")
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"{path}: raw trajectory is missing fields: {missing}")

    source_fps = int(raw["framerate"])
    if source_fps <= 0 or target_fps <= 0 or source_fps % target_fps:
        raise ValueError(f"{path}: source fps {source_fps} must be a positive multiple of target fps {target_fps}.")
    if list(raw["joint_names"]) != reference_joint_names:
        raise ValueError(f"{path}: joint_names do not match the AMP expert dataset joint order.")

    root_pos = np.asarray(raw["base_pos_w"], dtype=np.float32)
    root_rot = normalize_quat_xyzw(raw["base_quat_w"], "wxyz")
    dof = np.asarray(raw["joint_pos"], dtype=np.float32)
    frame_count = root_pos.shape[0]
    if root_pos.shape != (frame_count, 3) or root_rot.shape != (frame_count, 4):
        raise ValueError(f"{path}: root position/quaternion dimensions are inconsistent.")
    if dof.shape != (frame_count, len(reference_joint_names)):
        raise ValueError(f"{path}: joint positions do not match the expected frame or joint dimensions.")
    if not np.isfinite(root_pos).all() or not np.isfinite(root_rot).all() or not np.isfinite(dof).all():
        raise ValueError(f"{path}: raw trajectory contains non-finite values.")

    frame_indices = np.arange(0, frame_count, source_fps // target_fps)
    root_pos = root_pos[frame_indices]
    root_rot = root_rot[frame_indices]
    dof = dof[frame_indices]
    root_axis_angle = Rotation.from_quat(root_rot).as_rotvec().astype(np.float32)
    pose_aa = make_body_aligned_pose_aa(
        root_axis_angle, dof, reference_joint_names, dof_axes, body_names, body_to_joint
    )

    return {
        "root_trans_offset": root_pos,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_rot,
        "smpl_joints": np.zeros((len(frame_indices), *smpl_joint_shape), dtype=np.float32),
        "fps": target_fps,
        "joint_names": reference_joint_names,
        "motion_name": path.stem,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if not args.motion:
        raise ValueError("At least one --motion is required.")

    install_numpy_pickle_compat()
    expert = joblib.load(args.expert_dataset)
    if not isinstance(expert, dict) or not expert:
        raise ValueError("AMP expert dataset must be a non-empty mapping.")

    reference_joint_names, smpl_joint_shape = validate_reference(next(iter(expert.values())))
    xml_joint_names, dof_axes, body_names, body_to_joint = load_dof_metadata(args.robot_xml)
    if xml_joint_names != reference_joint_names:
        raise ValueError("Robot XML motor order does not match the AMP expert dataset joint order.")

    merged = dict(expert)
    for path in args.motion:
        key = path.stem
        if key in merged:
            raise ValueError(f"AMP expert dataset already contains motion {key!r}.")
        merged[key] = load_raw_motion(
            path,
            reference_joint_names=reference_joint_names,
            smpl_joint_shape=smpl_joint_shape,
            dof_axes=dof_axes,
            body_names=body_names,
            body_to_joint=body_to_joint,
            target_fps=args.target_fps,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(merged, args.output)
    print(f"Saved {len(merged)} motions to {args.output}")


if __name__ == "__main__":
    main()
