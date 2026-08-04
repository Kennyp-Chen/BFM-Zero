"""Add left-right mirrored motions to a PiPlus AMP expert dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mirror_joint_name(name: str) -> str:
    if name.startswith("l_"):
        return "r_" + name[2:]
    if name.startswith("r_"):
        return "l_" + name[2:]
    return name


def _joint_sign(name: str) -> float:
    # This is the PiPlus symmetric_augmentation_joint_reverse_buf expressed
    # by joint name, independent of the dataset's serialized joint order.
    if any(token in name for token in ("yaw", "roll", "thigh", "upper_arm")):
        return -1.0
    return 1.0


def mirror_motion(motion: dict[str, object]) -> dict[str, object]:
    names = list(motion["joint_names"])
    indices = {name: index for index, name in enumerate(names)}
    dof = np.asarray(motion["dof"], dtype=np.float32)
    mirrored_dof = np.empty_like(dof)
    for index, name in enumerate(names):
        partner = _mirror_joint_name(name)
        if partner not in indices:
            raise ValueError(f"Missing mirror joint {partner!r} for {name!r}")
        mirrored_dof[:, index] = _joint_sign(name) * dof[:, indices[partner]]

    mirrored = dict(motion)
    mirrored["dof"] = mirrored_dof
    root_trans = np.asarray(motion["root_trans_offset"], dtype=np.float32).copy()
    root_trans[:, 1] *= -1.0
    mirrored["root_trans_offset"] = root_trans
    root_rot = np.asarray(motion["root_rot"], dtype=np.float32).copy()
    if root_rot.shape[-1] != 4:
        raise ValueError(f"Expected XYZW root quaternions, got {root_rot.shape}")
    # Reflection across the sagittal (x-z) plane in XYZW quaternion form.
    root_rot[:, 0] *= -1.0
    root_rot[:, 2] *= -1.0
    mirrored["root_rot"] = root_rot
    if "smpl_joints" in motion:
        joints = np.asarray(motion["smpl_joints"], dtype=np.float32).copy()
        if joints.ndim >= 3 and joints.shape[-1] == 3:
            joints[..., 1] *= -1.0
            mirrored["smpl_joints"] = joints
    mirrored["motion_name"] = f"{motion.get('motion_name', 'motion')}_mirror"
    return mirrored


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    expert = joblib.load(args.input)
    if not isinstance(expert, dict) or not expert:
        raise ValueError("AMP expert dataset must be a non-empty mapping")
    augmented = dict(expert)
    for key, motion in expert.items():
        mirror_key = f"{key}__mirror"
        if mirror_key in augmented:
            raise ValueError(f"AMP expert dataset already contains {mirror_key!r}")
        augmented[mirror_key] = mirror_motion(motion)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(augmented, args.output)
    print(f"Saved {len(augmented)} motions ({len(expert)} originals + {len(expert)} mirrors) to {args.output}")


if __name__ == "__main__":
    main()
