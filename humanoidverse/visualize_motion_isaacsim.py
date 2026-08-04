from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humanoidverse.visualize_motion import RobotName, get_robot_spec, _load_motion, _motion_to_qpos


def _axis_angle_to_rotvec(quat_xyzw: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)


def _joint_axis_map(env) -> dict[str, np.ndarray]:
    mesh_parser = env._motion_lib.mesh_parsers
    body_to_joint = mesh_parser.mjcf_data["body_to_joint"]
    joint_names_in_body_order = [body_to_joint[mesh_parser.body_names[body_id]] for body_id in mesh_parser.actuated_joints_idx]
    return {
        joint_name: np.asarray(axis, dtype=np.float32)
        for joint_name, axis in zip(joint_names_in_body_order, mesh_parser.dof_axis.cpu().numpy())
    }


def _body_to_joint_map(env, joint_axes: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        body_name: joint_name
        for body_name, joint_name in env._motion_lib.mesh_parsers.mjcf_data["body_to_joint"].items()
        if joint_name in joint_axes
    }


def _align_motion_pose_aa_for_motion_lib(env, motion_data: dict) -> bool:
    pose_aa = motion_data.get("pose_aa")
    if pose_aa is None:
        return False
    body_names = env._motion_lib.mesh_parsers.body_names
    if pose_aa.shape[1] == len(body_names):
        return False

    joint_names = list(motion_data.get("joint_names", []))
    dof = np.asarray(motion_data["dof"], dtype=np.float32)
    root_rot = np.asarray(motion_data["root_rot"], dtype=np.float32)
    joint_axes = _joint_axis_map(env)
    body_to_joint = _body_to_joint_map(env, joint_axes)
    missing_joint_names = sorted({joint_name for joint_name in body_to_joint.values() if joint_name not in joint_names})
    if missing_joint_names:
        raise ValueError(f"Cannot align pose_aa; motion is missing joint_names entries: {missing_joint_names}")

    aligned_pose_aa = np.zeros((dof.shape[0], len(body_names), 3), dtype=np.float32)
    aligned_pose_aa[:, 0, :] = _axis_angle_to_rotvec(root_rot)
    for body_id, body_name in enumerate(body_names[1:], start=1):
        joint_name = body_to_joint.get(body_name)
        if joint_name is None:
            continue
        dof_id = joint_names.index(joint_name)
        aligned_pose_aa[:, body_id, :] = joint_axes[joint_name][None, :] * dof[:, dof_id, None]

    motion_data["pose_aa"] = aligned_pose_aa
    return True


def _resolve_motion_index(data_path: Path, motion: int | str, motion_key: str) -> int:
    if isinstance(motion, int):
        return motion

    data = joblib.load(data_path)
    return list(data.keys()).index(motion_key)


def _get_reference_marker_positions(env, motion_index: int, num_frames: int, fps: int, motion_data: dict) -> torch.Tensor:
    env.set_is_evaluating(motion_index)
    if _align_motion_pose_aa_for_motion_lib(env, motion_data):
        env._motion_lib._motion_data_list[motion_index] = motion_data
        env._motion_lib.load_motions(random_sample=False, num_motions_to_load=1, start_idx=motion_index)
    motion_times = torch.arange(num_frames, dtype=torch.float32, device=env.device) / fps
    motion_ids = torch.zeros(num_frames, dtype=torch.long, device=env.device)
    motion_res = env._motion_lib.get_motion_state(motion_ids, motion_times)
    marker_pos = motion_res["rg_pos_t"]
    if env.motion_body_ids is not None:
        marker_pos = marker_pos[:, env.motion_body_ids]
        motion_extend_body_ids = getattr(env, "motion_extend_body_ids", None)
        if motion_extend_body_ids is not None:
            marker_pos = torch.cat([marker_pos, motion_res["rg_pos_t"][:, motion_extend_body_ids]], dim=1)
    return marker_pos


def _make_marker_indices(env, num_markers: int) -> torch.Tensor | None:
    num_marker_types = len(getattr(env.simulator, "vis_sphere_marker_names", ["sphere"]))
    if num_marker_types <= 1:
        return None
    marker_body_to_index = getattr(env.simulator, "vis_sphere_marker_body_to_index", {})
    motion_body_names = getattr(env, "motion_body_names_extend", None)
    if motion_body_names is None:
        motion_body_names = getattr(env, "motion_body_names", None)
    if marker_body_to_index and motion_body_names is not None:
        return torch.tensor(
            [marker_body_to_index.get(body_name, body_id % num_marker_types) for body_id, body_name in enumerate(motion_body_names)],
            dtype=torch.long,
            device=env.device,
        )
    return torch.arange(num_markers, dtype=torch.long, device=env.device) % num_marker_types


def main(
    data_path: Path | None = None,
    robot: RobotName = "g1",
    motion: int | str = 0,
    headless: bool = False,
    enable_cameras: bool = False,
    max_frames: int | None = 500,
    stride: int = 1,
    device: str = "cpu",
    realtime: bool = True,
    fps: int | None = None,
) -> None:
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig

    robot_spec = get_robot_spec(robot)
    if data_path is None:
        data_path = robot_spec.default_data_path
    data_path = data_path.resolve()
    key, motion_data, num_motions = _load_motion(data_path, motion)
    motion_index = _resolve_motion_index(data_path, motion, key)
    qpos = _motion_to_qpos(motion_data, robot_spec.dof_size)
    if max_frames is not None:
        qpos = qpos[:max_frames]
    source_fps = int(motion_data.get("fps", 30))
    playback_fps = source_fps if fps is None else int(fps)

    env_cfg = HumanoidVerseIsaacConfig(
        lafan_tail_path=str(data_path.resolve()),
        device=device,
        enable_cameras=enable_cameras,
        include_last_action=False,
        include_history_actor=False,
        include_history_noaction=False,
        disable_domain_randomization=True,
        disable_obs_noise=True,
        hydra_overrides=[
            "simulator=isaacsim",
            f"robot={robot_spec.hydra_robot}",
            "env.config.max_episode_length_s=10000",
            f"env.config.headless={headless}",
        ],
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env
    marker_pos = _get_reference_marker_positions(env, motion_index, len(qpos), source_fps, motion_data)
    marker_scales = torch.ones_like(marker_pos[0])
    marker_indices = _make_marker_indices(env, marker_pos.shape[1])
    root_pos = torch.tensor(qpos[:, :3], dtype=torch.float32)
    root_quat = torch.tensor(qpos[:, 3:7], dtype=torch.float32)
    root_vel = torch.zeros((qpos.shape[0], 6), dtype=torch.float32)
    dof_pos = torch.tensor(qpos[:, 7:], dtype=torch.float32)
    dof_vel = torch.zeros_like(dof_pos)

    target_root = torch.zeros((1, 13), dtype=torch.float32, device=env.device)
    target_dof = torch.zeros((1, env.num_dof, 2), dtype=torch.float32, device=env.device)

    print(f"Loaded {data_path}")
    print(f"Robot: {robot_spec.name} / Hydra robot: {robot_spec.hydra_robot}")
    print(f"Motion {motion!r}: {key} ({num_motions} motions in file)")
    print(f"Frames: {len(qpos)} / fps: {playback_fps} / stride: {stride}")

    env_ids = torch.zeros(1, dtype=torch.long, device=env.device)
    frame_dt = stride / playback_fps

    for i, frame in enumerate(range(0, len(qpos), stride)):
        start_time = time.time()
        target_root[0, :3] = root_pos[frame].to(env.device)
        target_root[0, 3:7] = root_quat[frame].to(env.device)
        target_root[0, 7:13] = root_vel[frame].to(env.device)
        target_dof[0, :, 0] = dof_pos[frame].to(env.device)
        target_dof[0, :, 1] = dof_vel[frame].to(env.device)

        env.simulator.set_actor_root_state_tensor(env_ids, target_root)
        env.simulator.set_dof_state_tensor(env_ids, target_dof)
        env.simulator.draw_spheres_batch(marker_pos[frame], scales=marker_scales, marker_indices=marker_indices)
        env.simulator.scene.write_data_to_sim()
        env.simulator.sim.render()
        env.simulator.scene.update(dt=env.sim_dt)
        if i == 0 and headless:
            print("Isaac Sim headless mode: state applied, no GUI render loop.")
        if realtime:
            sleep_time = frame_dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
