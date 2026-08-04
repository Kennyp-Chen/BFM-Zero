from __future__ import annotations

from pathlib import Path

import joblib
import mediapy as media
import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from humanoidverse.agents.envs.humanoidverse_isaac import IsaacRendererWithMuJoco
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch


MOTION_ALL_TOKEN = "motion_all"


def _resolve_motion_ids(motion_list, num_motions: int) -> list[int]:
    requested = [motion_list] if isinstance(motion_list, (str, int)) else list(motion_list)
    tokens = [str(motion).strip() for motion in requested]
    if len(tokens) == 1 and tokens[0].lower() == MOTION_ALL_TOKEN:
        return list(range(num_motions))
    if any(token.lower() == MOTION_ALL_TOKEN for token in tokens):
        raise ValueError(f"{MOTION_ALL_TOKEN!r} must be passed by itself.")

    motion_ids = [int(token) for token in tokens]
    invalid_ids = [motion_id for motion_id in motion_ids if motion_id < 0 or motion_id >= num_motions]
    if invalid_ids:
        raise IndexError(f"Motion IDs out of range [0, {num_motions - 1}]: {invalid_ids}")
    return motion_ids


def _resolve_config_path(config_path: Path | None, output_dir: Path) -> Path:
    if config_path is None:
        config_path = output_dir.parent / "config.yaml"
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Motion FK config not found: {config_path}. Pass config_path explicitly or use an output_dir "
            "directly below a training run directory containing config.yaml."
        )
    return config_path


def _motion_to_qpos(motion: dict, model: mujoco.MjModel, motion_batch: Humanoid_Batch) -> np.ndarray:
    required = ("root_trans_offset", "pose_aa", "fps")
    missing = [field for field in required if field not in motion]
    if missing:
        raise KeyError(f"Motion is missing required field(s): {missing}")

    root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_trans_offset must have shape (T, 3), got {root_pos.shape}.")

    pose_aa = torch.as_tensor(motion["pose_aa"], dtype=torch.float32)
    motion_fk = motion_batch.fk_batch(
        pose_aa.unsqueeze(0),
        torch.as_tensor(root_pos, dtype=torch.float32).unsqueeze(0),
        return_full=True,
        dt=1.0 / float(motion["fps"]),
    )
    root_pos = motion_fk.global_translation[0, :, 0].cpu().numpy()
    root_quat_xyzw = motion_fk.global_rotation[0, :, 0].cpu().numpy()
    motion_dof = motion_fk.dof_pos[0].cpu().numpy()
    motion_dof_names = list(motion_batch.cfg.motion_dof_names)
    motion_dof_index = {name: index for index, name in enumerate(motion_dof_names)}

    qpos = np.zeros((root_pos.shape[0], model.nq), dtype=np.float64)
    qpos[:, :3] = root_pos
    qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]
    missing_joints = []
    for joint_id in range(1, model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name not in motion_dof_index:
            missing_joints.append(joint_name)
            continue
        qpos_address = model.jnt_qposadr[joint_id]
        qpos[:, qpos_address] = motion_dof[:, motion_dof_index[joint_name]]
    if missing_joints:
        raise ValueError(f"Reference motion has no values for MuJoCo joints: {missing_joints}")
    return qpos


def _render_motion(qpos: np.ndarray, renderer: IsaacRendererWithMuJoco, output_path: Path, fps: int) -> None:
    frames = renderer.from_qpos(qpos)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(output_path), frames, fps=fps)


def main(
    data_path: Path,
    mujoco_xml_path: Path,
    output_dir: Path,
    config_path: Path | None = None,
    motion_list: list[str] = [MOTION_ALL_TOKEN],
):
    """Render reference motions with the same FK and follow camera as tracking inference."""
    data_path = Path(data_path).expanduser().resolve()
    mujoco_xml_path = Path(mujoco_xml_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    config_path = _resolve_config_path(config_path, output_dir)
    motions = joblib.load(data_path)
    if not isinstance(motions, dict):
        raise TypeError(f"Expected {data_path} to contain a dict, got {type(motions).__name__}.")

    motion_keys = list(motions)
    motion_ids = _resolve_motion_ids(motion_list, len(motion_keys))
    config = OmegaConf.load(config_path)
    motion_batch = Humanoid_Batch(config.env.config.robot.motion)
    renderer = IsaacRendererWithMuJoco(render_size=256, xml_path=mujoco_xml_path)
    try:
        for motion_id in motion_ids:
            motion_key = motion_keys[motion_id]
            motion = motions[motion_key]
            qpos = _motion_to_qpos(motion, renderer.model, motion_batch)
            output_path = output_dir / f"expert_{motion_id:04d}_{motion_key}.mp4"
            print(f"Rendering reference motion {motion_id}: {motion_key}")
            _render_motion(qpos, renderer, output_path, fps=50)
            print(f"Saved reference video: {output_path}")
    finally:
        renderer.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
