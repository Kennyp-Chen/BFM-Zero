"""Second-stage AMP training for a frozen PiPlus BFM policy.

The trainable policy is a stochastic command encoder:

    velocity command + BFM actor observations -> z -> frozen BFM actor -> action

Only the command encoder and its value head are optimized.  The discriminator
is trained with the PiPlus MSELoss plus gradient-penalty objective on motion features.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import joblib
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from humanoidverse.agents.envs.humanoidverse_isaac import (
    HumanoidVerseIsaacConfig,
    load_expert_trajectories_from_motion_lib,
)
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.utils.asset_paths import resolve_asset_path
from humanoidverse.utils.torch_utils import quat_rotate_inverse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BFM_CHECKPOINT = str(
    PROJECT_ROOT / "huiying" / "bfmzero-piplus-lse-isaac-20260715_143758(1)" / "checkpoint"
)
DEFAULT_EXPERT_DATASET = str(PROJECT_ROOT / "dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run.pkl")
DEFAULT_ROBOT_CONFIG = str(PROJECT_ROOT / "humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H1W_LSE.yaml")
DEFAULT_TEACHER_POLICY = str(PROJECT_ROOT / "0803陈建宏23dof2.zip")
DEFAULT_KEY_BODIES = (
    "l_ankle_roll_link",
    "r_ankle_roll_link",
    # Isaac imports the PiPlus fixed wrist links into the elbow bodies.
    "l_elbow_link",
    "r_elbow_link",
    "head_pitch_link",
)
ENCODER_INPUT_TRANSFORM_VERSION = 2
ENCODER_COMMAND_SCALE = (1.25, 5.0, 1.25)
ENCODER_DOF_VEL_SCALE = 0.05
LINVEL_EXP_ERROR_SCALE = 0.16
MOVING_COMMAND_THRESHOLD = 0.1
TEACHER_POLICY_INPUT_DIM = 78
TEACHER_POLICY_ACTION_DIM = 23
TEACHER_POLICY_ACTION_SCALES = (
    0.09586094426291411,
    0.09586094426291411,
    0.14246510636688653,
    0.09586094426291411,
    0.09586094426291411,
    0.09614231747988017,
    0.15381525329142837,
    0.15381525329142837,
    0.09586094426291411,
    0.09586094426291411,
    0.09614231747988017,
    0.15381525329142837,
    0.15381525329142837,
    0.09586094426291411,
    0.09586094426291411,
    0.15381525329142837,
    0.15381525329142837,
    0.09586094426291411,
    0.09586094426291411,
    0.15381525329142837,
    0.15381525329142837,
    0.09586094426291411,
    0.09586094426291411,
)
TEACHER_POLICY_JOINT_NAMES = (
    "l_hip_pitch_joint",
    "r_hip_pitch_joint",
    "waist_yaw_joint",
    "l_hip_roll_joint",
    "r_hip_roll_joint",
    "head_yaw_joint",
    "l_shoulder_pitch_joint",
    "r_shoulder_pitch_joint",
    "l_thigh_joint",
    "r_thigh_joint",
    "head_pitch_joint",
    "l_shoulder_roll_joint",
    "r_shoulder_roll_joint",
    "l_calf_joint",
    "r_calf_joint",
    "l_upper_arm_joint",
    "r_upper_arm_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_elbow_joint",
    "r_elbow_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
)


def _distributed_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def barrier() -> None:
    if _distributed_ready():
        torch.distributed.barrier()


def average_gradients(parameters) -> None:
    if not _distributed_ready():
        return
    world_size = torch.distributed.get_world_size()
    for parameter in parameters:
        if parameter.grad is not None:
            torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def broadcast_module_state(module: nn.Module) -> None:
    if not _distributed_ready():
        return
    for value in module.state_dict().values():
        torch.distributed.broadcast(value, src=0)


def sync_floating_buffers(module: nn.Module) -> None:
    if not _distributed_ready():
        return
    world_size = torch.distributed.get_world_size()
    for value in module.buffers():
        if torch.is_floating_point(value):
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            value.div_(world_size)


def _quat_rotate_inverse_np(quat_xyzw: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    vec = np.asarray(vectors, dtype=np.float64)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True).clip(min=1.0e-8)
    x, y, z, w = np.moveaxis(quat, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rotation = np.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        axis=-1,
    ).reshape(quat.shape[:-1] + (3, 3))
    return np.einsum("nij,nj->ni", rotation.transpose(0, 2, 1), vec)


def _artifact_member(archive: zipfile.ZipFile, suffix: str) -> bytes:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {suffix!r} in teacher archive, found {matches}")
    return archive.read(matches[0])


class PiPlusAMPTeacherPolicy(nn.Module):
    """Batched PyTorch copy of the supplied 23DoF AMP ONNX teacher."""

    def __init__(self, artifact_path: str | Path, device: torch.device) -> None:
        super().__init__()
        artifact = Path(artifact_path).expanduser().resolve()
        if artifact.is_dir():
            actor_bytes = (artifact / "exported/actor.onnx").read_bytes()
            normalizer_bytes = (artifact / "exported/policy_normalizer.npz").read_bytes()
        elif artifact.suffix.lower() == ".zip":
            with zipfile.ZipFile(artifact) as archive:
                actor_bytes = _artifact_member(archive, "exported/actor.onnx")
                normalizer_bytes = _artifact_member(archive, "exported/policy_normalizer.npz")
        else:
            raise ValueError(f"Teacher policy must be a directory or .zip archive, got {artifact}")

        import onnx
        from onnx import numpy_helper

        onnx_model = onnx.load_model_from_string(actor_bytes)
        initializers = {item.name: numpy_helper.to_array(item) for item in onnx_model.graph.initializer}
        layers: list[nn.Module] = []
        for node in onnx_model.graph.node:
            if node.op_type == "Gemm":
                weight = np.asarray(initializers[node.input[1]], dtype=np.float32).copy()
                bias = np.asarray(initializers[node.input[2]], dtype=np.float32).copy()
                attributes = {item.name: item for item in node.attribute}
                if attributes.get("transB") is None or not int(attributes["transB"].i):
                    weight = weight.T
                linear = nn.Linear(weight.shape[1], weight.shape[0])
                linear.weight.data.copy_(torch.from_numpy(weight))
                linear.bias.data.copy_(torch.from_numpy(bias))
                layers.append(linear)
            elif node.op_type == "Elu":
                alpha = 1.0
                for attribute in node.attribute:
                    if attribute.name == "alpha":
                        alpha = float(attribute.f)
                layers.append(nn.ELU(alpha=alpha))
            else:
                raise ValueError(f"Unsupported operator {node.op_type!r} in teacher actor.onnx")
        if not layers or not isinstance(layers[0], nn.Linear) or layers[0].in_features != TEACHER_POLICY_INPUT_DIM:
            raise ValueError("Teacher actor.onnx does not expose the expected 78-D input")
        if not isinstance(layers[-1], nn.Linear) or layers[-1].out_features != TEACHER_POLICY_ACTION_DIM:
            raise ValueError("Teacher actor.onnx does not expose the expected 23-D action output")
        self.net = nn.Sequential(*layers).to(device=device, dtype=torch.float32).eval()
        with np.load(io.BytesIO(normalizer_bytes), allow_pickle=False) as data:
            mean = np.asarray(data["mean"], dtype=np.float32).reshape(1, -1)
            std = np.asarray(data["std"], dtype=np.float32).reshape(1, -1)
            eps = float(data["eps"]) if "eps" in data.files else 1.0e-2
        if mean.shape != (1, TEACHER_POLICY_INPUT_DIM) or std.shape != mean.shape:
            raise ValueError(f"Teacher normalizer must be [1, {TEACHER_POLICY_INPUT_DIM}], got {mean.shape}/{std.shape}")
        self.register_buffer("normalizer_mean", torch.from_numpy(mean).to(device))
        self.register_buffer("normalizer_std", torch.from_numpy(std).to(device))
        self.normalizer_eps = eps
        self.artifact_path = str(artifact)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def normalize(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[-1] != TEACHER_POLICY_INPUT_DIM:
            raise ValueError(
                f"Teacher observation must have shape [batch, {TEACHER_POLICY_INPUT_DIM}], got {tuple(observation.shape)}"
            )
        return (observation - self.normalizer_mean.to(observation)) / (self.normalizer_std.to(observation) + self.normalizer_eps)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalize(observation))


def _joint_permutation(source_names: tuple[str, ...], target_names: tuple[str, ...]) -> torch.Tensor:
    source_index = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source_index]
    extra = [name for name in source_names if name not in set(target_names)]
    if missing or extra or len(source_names) != len(target_names):
        raise ValueError(f"Joint contract mismatch: missing={missing}, extra={extra}")
    return torch.tensor([source_index[name] for name in target_names], dtype=torch.long)


def _reorder_joint_values(
    values: torch.Tensor,
    source_names: tuple[str, ...],
    target_names: tuple[str, ...],
) -> torch.Tensor:
    permutation = _joint_permutation(source_names, target_names).to(values.device)
    return values.index_select(-1, permutation)


def _joint_control_group(joint_name: str) -> str:
    for group in (
        "hip_pitch",
        "hip_roll",
        "thigh",
        "calf",
        "ankle_pitch",
        "ankle_roll",
        "waist_yaw",
        "head_yaw",
        "head_pitch",
        "shoulder_pitch",
        "shoulder_roll",
        "upper_arm",
        "elbow",
    ):
        if group in joint_name:
            return group
    raise ValueError(f"Cannot infer PiPlus control group from joint {joint_name!r}")


def _bfm_action_position_scales(robot_config: str | Path) -> tuple[float, ...]:
    """Return nominal physical joint-position offsets per unit BFM action."""
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(robot_config).expanduser().resolve())
    robot_cfg = config.robot
    control = robot_cfg.control
    if str(control.control_type).upper() != "P":
        raise ValueError(f"Stage2 action contract requires position control, got {control.control_type!r}")
    action_scale = float(control.action_scale)
    normalize_from = float(control.get("normalize_action_from", 1.0))
    normalize_to = float(control.get("normalize_action_to", 1.0))
    normalize_ratio = normalize_to / normalize_from if bool(control.get("normalize_action", False)) else 1.0
    dof_names = tuple(str(name) for name in robot_cfg.dof_names)
    effort_limits = tuple(float(value) for value in robot_cfg.dof_effort_limit_list)
    if len(dof_names) != len(effort_limits):
        raise ValueError("PiPlus dof_names and dof_effort_limit_list have different lengths")
    stiffness = {str(key): float(value) for key, value in control.stiffness.items()}
    scales = []
    for joint_name, effort_limit in zip(dof_names, effort_limits):
        group = _joint_control_group(joint_name)
        scales.append(normalize_ratio * action_scale * effort_limit / stiffness[group])
    return tuple(scales)


def build_teacher_policy_observation(
    obs: Mapping[str, torch.Tensor],
    commands: torch.Tensor,
    stage2_joint_names: tuple[str, ...],
    *,
    teacher_action_scales: torch.Tensor | None = None,
    bfm_action_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct the teacher's exact 78-D policy observation from Stage2 state."""
    state = obs["state"]
    last_action = obs.get("last_action")
    dof_dim = len(stage2_joint_names)
    if dof_dim != TEACHER_POLICY_ACTION_DIM or state.shape[-1] != 2 * dof_dim + 6:
        raise ValueError(f"Expected PiPlus state [{2 * dof_dim + 6}] and 23 joints, got {tuple(state.shape)}")
    if last_action is None or last_action.shape[-1] != dof_dim:
        raise ValueError("Teacher observation construction requires a 23-D last_action")
    if commands.shape[-1] != 3:
        raise ValueError(f"Expected 3 velocity commands, got {commands.shape[-1]}")

    dof_pos = state[..., :dof_dim]
    dof_vel = state[..., dof_dim : 2 * dof_dim]
    projected_gravity = state[..., 2 * dof_dim : 2 * dof_dim + 3]
    base_ang_vel = state[..., 2 * dof_dim + 3 :]
    teacher_pos = _reorder_joint_values(dof_pos, stage2_joint_names, TEACHER_POLICY_JOINT_NAMES)
    # BFM's raw state stores dof velocity at scale 1.0; the supplied teacher
    # policy's observation contract uses the IsaacLab joint-velocity scale 0.05.
    teacher_vel = _reorder_joint_values(dof_vel, stage2_joint_names, TEACHER_POLICY_JOINT_NAMES) * ENCODER_DOF_VEL_SCALE
    teacher_action = _reorder_joint_values(last_action, stage2_joint_names, TEACHER_POLICY_JOINT_NAMES)
    if (teacher_action_scales is None) != (bfm_action_scales is None):
        raise ValueError("teacher_action_scales and bfm_action_scales must be provided together")
    if teacher_action_scales is not None and bfm_action_scales is not None:
        teacher_scales = teacher_action_scales.to(device=teacher_action.device, dtype=teacher_action.dtype)
        bfm_scales = _reorder_joint_values(
            bfm_action_scales.to(device=last_action.device, dtype=last_action.dtype).reshape(1, -1),
            stage2_joint_names,
            TEACHER_POLICY_JOINT_NAMES,
        ).reshape(-1)
        teacher_action = teacher_action * bfm_scales / teacher_scales.clamp_min(1.0e-8)
    return torch.cat(
        [base_ang_vel, projected_gravity, commands, teacher_pos, teacher_vel, teacher_action], dim=-1
    )


def teacher_encoder_observation(
    obs: Mapping[str, torch.Tensor],
    commands: torch.Tensor,
    stage2_joint_names: tuple[str, ...],
    teacher: PiPlusAMPTeacherPolicy,
    *,
    teacher_action_scales: torch.Tensor | None = None,
    bfm_action_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    return teacher.normalize(
        build_teacher_policy_observation(
            obs,
            commands,
            stage2_joint_names,
            teacher_action_scales=teacher_action_scales,
            bfm_action_scales=bfm_action_scales,
        )
    )


def teacher_action_to_stage2(
    teacher_action: torch.Tensor,
    stage2_joint_names: tuple[str, ...],
) -> torch.Tensor:
    if teacher_action.shape[-1] != TEACHER_POLICY_ACTION_DIM:
        raise ValueError(f"Expected teacher action dimension {TEACHER_POLICY_ACTION_DIM}, got {teacher_action.shape[-1]}")
    return _reorder_joint_values(teacher_action, TEACHER_POLICY_JOINT_NAMES, stage2_joint_names)


def teacher_action_target_to_stage2(
    teacher_action: torch.Tensor,
    stage2_joint_names: tuple[str, ...],
    *,
    teacher_action_scales: torch.Tensor,
) -> torch.Tensor:
    """Convert teacher policy outputs to physical joint-position offsets in Stage2 order."""
    if teacher_action.shape[-1] != TEACHER_POLICY_ACTION_DIM:
        raise ValueError(f"Expected teacher action dimension {TEACHER_POLICY_ACTION_DIM}, got {teacher_action.shape[-1]}")
    scales = teacher_action_scales.to(device=teacher_action.device, dtype=teacher_action.dtype)
    return _reorder_joint_values(teacher_action * scales, TEACHER_POLICY_JOINT_NAMES, stage2_joint_names)


def _policy_dof_from_motion(motion: Mapping[str, Any], policy_joint_names: tuple[str, ...]) -> np.ndarray:
    dof = np.asarray(motion["dof"], dtype=np.float64)
    source_joint_names = [str(name) for name in motion["joint_names"]]
    if dof.ndim != 2 or dof.shape[1] != len(source_joint_names):
        raise ValueError(
            f"AMP expert dof shape {dof.shape} does not match source joint order length {len(source_joint_names)}"
        )
    missing = [name for name in policy_joint_names if name not in source_joint_names]
    if missing:
        raise ValueError(f"AMP expert motion is missing policy joints: {missing}")
    source_index = {name: index for index, name in enumerate(source_joint_names)}
    return dof[:, [source_index[name] for name in policy_joint_names]]


def _motion_qpos(
    motion: Mapping[str, Any],
    policy_joint_names: tuple[str, ...],
    policy_dof: np.ndarray,
    frame_index: int,
    model: mujoco.MjModel,
) -> np.ndarray:
    qpos = np.zeros(model.nq, dtype=np.float64)
    root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float64)[frame_index]
    root_quat_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)[frame_index]
    qpos[:3] = root_pos
    qpos[3:7] = root_quat_xyzw[[3, 0, 1, 2]]
    dof = policy_dof[frame_index]
    if dof.shape != (len(policy_joint_names),):
        raise ValueError(f"Expected policy motion frame shape {(len(policy_joint_names),)}, got {dof.shape}")
    for index, joint_name in enumerate(policy_joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"PiPlus AMP XML is missing joint {joint_name!r}")
        qpos[int(model.jnt_qposadr[joint_id])] = dof[index]
    return qpos


class PiPlusAMPExpertDataset:
    """Precomputed PiPlus AMP features with an eight-frame joint history."""

    def __init__(self, features: np.ndarray, *, history_length: int, feature_dim: int, motion_count: int) -> None:
        if features.ndim != 2 or features.shape[1] != feature_dim:
            raise ValueError(f"AMP expert features must have shape [N, {feature_dim}], got {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError("AMP expert features contain non-finite values")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.history_length = int(history_length)
        self.feature_dim = int(feature_dim)
        self.motion_count = int(motion_count)

    @classmethod
    def from_pkl(
        cls,
        dataset_path: str | Path,
        *,
        robot=None,
        xml_path: str | Path,
        policy_joint_names: tuple[str, ...] | None = None,
        key_bodies: tuple[str, ...] = DEFAULT_KEY_BODIES,
        history_length: int = 8,
    ) -> "PiPlusAMPExpertDataset":
        data = joblib.load(dataset_path)
        if not isinstance(data, Mapping) or not data:
            raise ValueError(f"AMP expert dataset must be a non-empty mapping, got {type(data)!r}")
        if policy_joint_names is None:
            if robot is None or not hasattr(robot, "control_joint_names"):
                raise ValueError("policy_joint_names is required when no robot contract is supplied")
            policy_joint_names = tuple(robot.control_joint_names)
        policy_joint_names = tuple(policy_joint_names)
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        mj_data = mujoco.MjData(model)
        base_body = getattr(robot, "base_body", "base_link")
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
        key_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in key_bodies]
        if base_id < 0 or any(item < 0 for item in key_ids):
            raise ValueError(f"Could not resolve PiPlus AMP key bodies: base={base_body}, keys={key_bodies}")

        feature_rows: list[np.ndarray] = []
        for motion in data.values():
            root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float64)
            root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
            dof = _policy_dof_from_motion(motion, policy_joint_names)
            fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
            frame_count = root_pos.shape[0]
            if root_rot.shape != (frame_count, 4) or dof.shape != (frame_count, len(policy_joint_names)):
                raise ValueError("PiPlus AMP expert motion has inconsistent root_rot/dof dimensions")

            root_vel_world = np.zeros_like(root_pos)
            root_vel_world[:-1] = np.diff(root_pos, axis=0) * fps
            root_vel_world[-1] = root_vel_world[-2] if frame_count > 1 else 0.0
            root_vel_local = _quat_rotate_inverse_np(root_rot, root_vel_world)
            key_positions_local = np.zeros((frame_count, len(key_ids), 3), dtype=np.float64)
            for frame_index in range(frame_count):
                mj_data.qpos[:] = _motion_qpos(motion, policy_joint_names, dof, frame_index, model)
                mujoco.mj_forward(model, mj_data)
                relative = mj_data.xpos[key_ids] - mj_data.xpos[base_id]
                key_positions_local[frame_index] = _quat_rotate_inverse_np(
                    np.repeat(root_rot[frame_index : frame_index + 1], len(key_ids), axis=0), relative
                )

            if frame_count < history_length:
                continue
            for frame_index in range(history_length - 1, frame_count):
                joint_history = dof[frame_index - history_length + 1 : frame_index + 1].reshape(-1)
                row = np.concatenate(
                    [
                        root_vel_local[frame_index],
                        key_positions_local[frame_index].reshape(-1),
                        joint_history,
                    ]
                )
                feature_rows.append(row.astype(np.float32, copy=False))

        feature_dim = 3 + len(key_bodies) * 3 + history_length * len(policy_joint_names)
        features = np.stack(feature_rows, axis=0) if feature_rows else np.zeros((0, feature_dim), dtype=np.float32)
        return cls(features, history_length=history_length, feature_dim=feature_dim, motion_count=len(data))

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.features.shape[0] == 0:
            raise ValueError("AMP expert dataset contains no valid history windows")
        indices = torch.randint(self.features.shape[0], (int(batch_size),))
        return self.features[indices].to(device)


AMP_DISCRIMINATOR_OBJECTIVE = "mse_quad_v1"


class AMPDiscriminator(nn.Module):
    """PiPlus AMP discriminator with MSE logits and quadratic reward mapping."""

    def __init__(self, feature_dim: int, hidden_dims: tuple[int, int] = (1024, 512), grad_penalty_weight: float = 5.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[1], 1),
        )
        self.grad_penalty_weight = float(grad_penalty_weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    def loss(self, expert: torch.Tensor, policy: torch.Tensor) -> dict[str, torch.Tensor]:
        expert_score = self(expert)
        policy_score = self(policy)
        epsilon = torch.rand(expert.shape[0], 1, device=expert.device)
        interpolated = (epsilon * expert + (1.0 - epsilon) * policy).requires_grad_(True)
        interpolated_score = self(interpolated)
        gradients = torch.autograd.grad(
            interpolated_score,
            interpolated,
            grad_outputs=torch.ones_like(interpolated_score),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient_norm = gradients.reshape(gradients.shape[0], -1).norm(2, dim=1)
        gradient_penalty = (gradient_norm - 1.0).square().mean()
        expert_loss = F.mse_loss(expert_score, torch.ones_like(expert_score))
        policy_loss = F.mse_loss(policy_score, torch.zeros_like(policy_score))
        total = expert_loss + policy_loss + self.grad_penalty_weight * gradient_penalty
        return {
            "loss": total,
            "gradient_penalty": gradient_penalty,
            "expert_score": expert_score.mean(),
            "policy_score": policy_score.mean(),
            "gradient_norm": gradient_norm.mean(),
        }

    @torch.no_grad()
    def reward(self, policy_features: torch.Tensor) -> torch.Tensor:
        # Keep this method as the raw score accessor; the caller applies the
        # configured reward mapping exactly once before PPO/GAE.
        return self(policy_features)


class MimicLiteRewardNormalizer(nn.Module):
    """Legacy running scalar normalizer kept for checkpoint compatibility."""

    def __init__(self, decay: float = 0.999, eps: float = 1.0e-5):
        super().__init__()
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        self.decay = float(decay)
        self.eps = float(eps)
        self.register_buffer("sum", torch.zeros(1))
        self.register_buffer("ssq", torch.zeros(1))
        self.register_buffer("count", torch.ones(1))

    @torch.no_grad()
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        flat = values.reshape(-1, 1).to(self.sum.dtype)
        if self.decay < 1.0:
            self.count.mul_(self.decay).add_(flat.shape[0])
            self.sum.mul_(self.decay).add_(flat.sum(0))
            self.ssq.mul_(self.decay).add_(flat.square().sum(0))
        else:
            self.count.add_(flat.shape[0])
            weight = flat.shape[0] / self.count
            self.sum.lerp_(flat.mean(0), weight=weight)
            self.ssq.lerp_(flat.square().mean(0), weight=weight)
        mean = self.sum / self.count
        variance = (self.ssq / self.count - mean.square()).clamp_min(self.eps)
        return (values - mean.to(values.device)) / variance.sqrt().to(values.device)


class CommandEncoderPolicy(nn.Module):
    """Stochastic command-to-latent policy with a PPO value head."""

    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int = 256, hidden_layers: int = 1):
        super().__init__()
        self.z_dim = int(z_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        if self.hidden_layers < 1:
            raise ValueError(f"hidden_layers must be >= 1, got {self.hidden_layers}")
        trunk_layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(self.hidden_layers - 1):
            trunk_layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.trunk = nn.Sequential(*trunk_layers)
        self.latent_mean = nn.Linear(hidden_dim, z_dim)
        self.latent_log_std = nn.Parameter(torch.full((z_dim,), -1.5))
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        mean = self.latent_mean(hidden)
        log_std = self.latent_log_std.clamp(-5.0, 1.0).expand_as(mean)
        value = self.value_head(hidden).squeeze(-1)
        return mean, log_std, value

    def distribution(self, features: torch.Tensor) -> Normal:
        mean, log_std, _ = self(features)
        return Normal(mean, log_std.exp())

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(features)
        distribution = Normal(mean, log_std.exp())
        raw_z = distribution.rsample()
        log_prob = distribution.log_prob(raw_z).sum(dim=-1)
        return raw_z, log_prob, value

    def deterministic_z(self, features: torch.Tensor) -> torch.Tensor:
        mean, _, _ = self(features)
        return mean

    @torch.no_grad()
    def configure_from_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.z_dim,) or std.shape != (self.z_dim,):
            raise ValueError(f"Expected latent stats [{self.z_dim}], got mean={tuple(mean.shape)}, std={tuple(std.shape)}")
        self.latent_mean.bias.copy_(mean.to(self.latent_mean.bias))
        self.latent_log_std.copy_(std.to(self.latent_log_std).clamp_min(1.0e-3).log())


def project_latent(z: torch.Tensor) -> torch.Tensor:
    return math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)


def _latent_stats(values: torch.Tensor) -> dict[str, Any]:
    if not torch.isfinite(values).all():
        raise ValueError("BFM latent statistics contain non-finite values")
    flattened = values.detach().float().reshape(-1, values.shape[-1])
    norms = torch.linalg.vector_norm(flattened, dim=-1)
    return {
        "shape": list(flattened.shape),
        "mean": flattened.mean(dim=0),
        "std": flattened.std(dim=0, unbiased=False),
        "min": flattened.min(dim=0).values,
        "max": flattened.max(dim=0).values,
        "norm_mean": norms.mean(),
        "norm_std": norms.std(unbiased=False),
        "norm_min": norms.min(),
        "norm_max": norms.max(),
    }


def _latent_direction_prior(
    projected_z: torch.Tensor,
    reference_z: torch.Tensor,
    *,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest-reference direction penalty and cosine similarity.

    The BFM checkpoint only constrains the latent radius when ``norm_z`` is
    enabled.  Comparing directions against backward-map latents supplies the
    missing support constraint without changing the checkpoint architecture.
    """
    if projected_z.ndim != 2 or reference_z.ndim != 2 or projected_z.shape[-1] != reference_z.shape[-1]:
        raise ValueError(
            f"Latent direction shapes must be [N, D] and [M, D], got {tuple(projected_z.shape)} and {tuple(reference_z.shape)}"
        )
    if reference_z.shape[0] == 0:
        raise ValueError("Latent direction reference bank is empty")
    projected = F.normalize(projected_z.float(), dim=-1)
    reference = F.normalize(reference_z.float(), dim=-1)
    best_cosine: list[torch.Tensor] = []
    for start in range(0, projected.shape[0], max(int(chunk_size), 1)):
        similarity = projected[start : start + chunk_size] @ reference.T
        best_cosine.append(similarity.max(dim=-1).values)
    cosine = torch.cat(best_cosine, dim=0)
    return 1.0 - cosine, cosine


def _latent_mmd_rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute a small unbiased-ish RBF MMD diagnostic on normalized latents."""
    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return x.new_tensor(float("nan"))
    joined = torch.cat((x, y), dim=0)
    bandwidth = torch.cdist(joined, joined).square().flatten().median().clamp_min(1.0e-4)
    k_xx = torch.exp(-torch.cdist(x, x).square() / bandwidth)
    k_yy = torch.exp(-torch.cdist(y, y).square() / bandwidth)
    k_xy = torch.exp(-torch.cdist(x, y).square() / bandwidth)
    n_x, n_y = x.shape[0], y.shape[0]
    xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / max(n_x * (n_x - 1), 1)
    yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / max(n_y * (n_y - 1), 1)
    return xx + yy - 2.0 * k_xy.mean()


def latent_manifold_metrics(
    projected_z: torch.Tensor,
    commands: torch.Tensor,
    reference_z: torch.Tensor,
    *,
    max_samples: int = 1024,
) -> dict[str, float]:
    """Compare Stage2 projected latents to expert latents globally and by command bin."""
    projected_z = projected_z.reshape(-1, projected_z.shape[-1])
    commands = commands.reshape(-1, commands.shape[-1])
    if projected_z.shape[0] != commands.shape[0]:
        raise ValueError("Projected latent and command sample counts must match")
    generator = torch.Generator(device=projected_z.device)
    generator.manual_seed(17)
    reference = reference_z.to(projected_z.device)
    if reference.shape[0] > max_samples:
        reference = reference[torch.randperm(reference.shape[0], generator=generator, device=reference.device)[:max_samples]]

    def sample_rows(values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] <= max_samples:
            return values
        indices = torch.randperm(values.shape[0], generator=generator, device=values.device)[:max_samples]
        return values[indices]

    masks = {
        "all": torch.ones(projected_z.shape[0], dtype=torch.bool, device=projected_z.device),
        "stand": (commands[:, :2].norm(dim=-1) < 0.1) & (commands[:, 2].abs() < 0.1),
        "turn": (commands[:, :2].norm(dim=-1) < 0.1) & (commands[:, 2].abs() >= 0.1),
        "slow": (commands[:, :2].norm(dim=-1) >= 0.1) & (commands[:, :2].norm(dim=-1) < 0.35),
        "medium": (commands[:, :2].norm(dim=-1) >= 0.35) & (commands[:, :2].norm(dim=-1) < 0.65),
        "fast": commands[:, :2].norm(dim=-1) >= 0.65,
    }
    metrics: dict[str, float] = {}
    for name, mask in masks.items():
        selected = sample_rows(projected_z[mask])
        prefix = f"latent/{name}"
        metrics[f"{prefix}_fraction"] = float(mask.float().mean())
        metrics[f"{prefix}_count"] = float(mask.sum())
        if selected.shape[0] < 2:
            metrics[f"{prefix}_nearest_cosine"] = 0.0
            metrics[f"{prefix}_knn_distance"] = 0.0
            metrics[f"{prefix}_mmd_rbf"] = 0.0
            continue
        penalty, cosine = _latent_direction_prior(selected, reference)
        metrics[f"{prefix}_nearest_cosine"] = float(cosine.mean())
        metrics[f"{prefix}_knn_distance"] = float(penalty.mean())
        metrics[f"{prefix}_mmd_rbf"] = float(_latent_mmd_rbf(selected, reference))
    return metrics


def _latent_stats_to_json(stats: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.tolist() if value.ndim > 0 else float(value)
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    return convert({key: value for key, value in stats.items() if not str(key).startswith("_")})


def _collect_bfm_latent_stats(
    *,
    bfm_agent,
    bfm_model,
    stats_env,
    device: str,
    robot_config: str,
    expert_dataset: str,
    num_envs: int,
    seed: int,
    max_episode_length_s: float,
    max_frames: int,
) -> dict[str, Any]:
    """Collect the same expert backward-latent statistics used by UFO Stage2."""
    del bfm_agent, robot_config, expert_dataset, num_envs, seed, max_episode_length_s
    agent_cfg = SimpleNamespace(model=SimpleNamespace(seq_length=int(bfm_model.cfg.seq_length)))
    expert_buffer = load_expert_trajectories_from_motion_lib(stats_env._env, agent_cfg, device=device)
    observations = {
        key: value
        for key, value in expert_buffer.storage["observation"].items()
        if key in {"state", "last_action", "privileged_state"}
    }
    frame_count = int(observations["state"].shape[0])
    if max_frames > 0:
        frame_count = min(frame_count, max_frames)
        observations = {key: value[:frame_count] for key, value in observations.items()}
    with torch.inference_mode():
        raw_z = bfm_model.backward_map(observations)
        projected_z = bfm_model.project_z(raw_z)
    z_dim = int(bfm_model.cfg.archi.z_dim)
    if raw_z.ndim != 2 or raw_z.shape[-1] != z_dim:
        raise ValueError(f"Backward encoder returned shape {tuple(raw_z.shape)}, expected [N, {z_dim}]")
    return {
        "frame_count": frame_count,
        "raw": _latent_stats(raw_z),
        "projected": _latent_stats(projected_z),
        "_reference_projected": projected_z.detach().float(),
    }


def _validate_and_configure_latent_contract(
    *,
    policy: CommandEncoderPolicy,
    bfm_model,
    latent_stats: dict[str, Any],
) -> dict[str, Any]:
    """Validate checkpoint latent semantics and configure the unprojected branch."""
    z_dim = int(bfm_model.cfg.archi.z_dim)
    norm_z = bool(bfm_model.cfg.archi.norm_z)
    if policy.z_dim != z_dim:
        raise ValueError(f"Command encoder z_dim={policy.z_dim} does not match checkpoint z_dim={z_dim}")
    raw_stats = latent_stats["raw"]
    projected_stats = latent_stats["projected"]
    if norm_z:
        expected_norm = math.sqrt(z_dim)
        projected_norm = float(projected_stats["norm_mean"])
        if not math.isfinite(projected_norm) or abs(projected_norm - expected_norm) > 1.0e-3:
            raise ValueError(
                f"Checkpoint norm_z=True but project_z norm is {projected_norm:.6f}, expected {expected_norm:.6f}"
            )
        policy.configure_from_latent_stats(raw_stats["mean"], raw_stats["std"])
        return {
            "mode": "checkpoint_project_z",
            "expected_norm": expected_norm,
            "projected_norm_mean": projected_norm,
            "raw_latent_norm_mean": float(raw_stats["norm_mean"]),
            "raw_latent_norm_std": float(raw_stats["norm_std"]),
            "direction_prior": "expert_projected_latent_bank",
        }

    policy.configure_from_latent_stats(raw_stats["mean"], raw_stats["std"])
    return {
        "mode": "raw_latent_stats",
        "raw_norm_mean": float(raw_stats["norm_mean"]),
        "raw_norm_std": float(raw_stats["norm_std"]),
        "command_encoder_log_std_mean": float(policy.latent_log_std.mean().detach()),
    }


def encoder_input_scale(obs: Mapping[str, torch.Tensor], commands: torch.Tensor) -> torch.Tensor:
    """Build field-wise scales without changing the Stage2 checkpoint input shape."""
    state = obs["state"]
    last_action = obs.get("last_action", torch.zeros_like(state[..., :0]))
    command_scale = commands.new_tensor(ENCODER_COMMAND_SCALE)
    if commands.shape[-1] != command_scale.numel():
        raise ValueError(f"Expected {command_scale.numel()} commands, got {commands.shape[-1]}")

    state_scale = torch.ones(state.shape[-1], device=state.device, dtype=state.dtype)
    dof_dim = int(last_action.shape[-1])
    if dof_dim > 0:
        expected_state_dim = 2 * dof_dim + 6
        if state.shape[-1] != expected_state_dim:
            raise ValueError(f"Expected PiPlus state dim {expected_state_dim}, got {state.shape[-1]}")
        state_scale[dof_dim : 2 * dof_dim] = ENCODER_DOF_VEL_SCALE

    pieces = [command_scale, state_scale, torch.ones(last_action.shape[-1], device=state.device, dtype=state.dtype)]
    history = obs.get("history_actor")
    if history is not None:
        history_scale = torch.ones(history.shape[-1], device=history.device, dtype=history.dtype)
        per_frame_dim = 3 * dof_dim + 6
        if dof_dim <= 0 or history.shape[-1] % per_frame_dim != 0:
            raise ValueError(
                f"Cannot infer grouped PiPlus history layout from history dim {history.shape[-1]} and dof dim {dof_dim}"
            )
        history_length = history.shape[-1] // per_frame_dim
        # history_actor is grouped by key: actions, base_ang_vel, dof_pos,
        # dof_vel, projected_gravity. Only the raw joint velocity block needs
        # the same 0.05 scale used by the first-stage UFO observations.
        dof_vel_start = history_length * (2 * dof_dim + 3)
        history_scale[dof_vel_start : dof_vel_start + history_length * dof_dim] = ENCODER_DOF_VEL_SCALE
        pieces.append(history_scale)
    return torch.cat(pieces)


def flatten_encoder_observation(obs: Mapping[str, torch.Tensor], commands: torch.Tensor) -> torch.Tensor:
    state = obs["state"]
    last_action = obs.get("last_action", torch.zeros_like(state[..., :0]))
    pieces = [commands, state, last_action]
    history = obs.get("history_actor")
    if history is not None:
        pieces.append(history)
    features = torch.cat(pieces, dim=-1)
    return features * encoder_input_scale(obs, commands)


@torch.no_grad()
def load_command_encoder_policy_state(
    policy: CommandEncoderPolicy,
    checkpoint: Mapping[str, Any],
    input_scale: torch.Tensor,
) -> bool:
    """Load a policy and migrate legacy raw-input weights losslessly.

    Returns True when the first layer was reparameterized. Its optimizer state
    must then be reset because the stored Adam moments use the old coordinates.
    """
    checkpoint_policy = checkpoint["policy"]
    checkpoint_first_layer = checkpoint_policy.get("trunk.0.weight")
    if checkpoint_first_layer is not None and checkpoint_first_layer.shape[1] != policy.trunk[0].in_features:
        raise ValueError(
            "Stage2 checkpoint command encoder input dimension does not match the current layout: "
            f"checkpoint={checkpoint_first_layer.shape[1]}, current={policy.trunk[0].in_features}. "
            "The teacher-distillation layout is new; restart from the frozen BFM checkpoint."
        )
    policy.load_state_dict(checkpoint_policy)
    metadata = checkpoint.get("metadata", {})
    transform = metadata.get("encoder_input_transform", {}) if isinstance(metadata, Mapping) else {}
    version = int(transform.get("version", 0)) if isinstance(transform, Mapping) else 0
    if version == ENCODER_INPUT_TRANSFORM_VERSION:
        return False
    if version != 0:
        raise ValueError(
            f"Unsupported encoder input transform version {version}; expected 0 or {ENCODER_INPUT_TRANSFORM_VERSION}"
        )
    first_layer = policy.trunk[0]
    if not isinstance(first_layer, nn.Linear) or first_layer.in_features != input_scale.numel():
        raise ValueError("Cannot migrate the command encoder first layer for the new input scaling")
    first_layer.weight.div_(input_scale.to(first_layer.weight).unsqueeze(0))
    return True


def restore_policy_optimizer(
    optimizer: torch.optim.Optimizer,
    checkpoint: Mapping[str, Any],
    *,
    learning_rate: float,
    reset_for_input_migration: bool,
) -> str:
    """Restore valid Adam state while always honoring the requested LR."""
    if not reset_for_input_migration and "policy_optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["policy_optimizer"])
        status = "loaded_with_lr_override"
    else:
        status = "reset_for_encoder_input_transform" if reset_for_input_migration else "reset_missing_state"
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
    return status


def restore_discriminator_optimizer(
    optimizer: torch.optim.Optimizer,
    checkpoint: Mapping[str, Any],
    *,
    learning_rate: float,
) -> str:
    """Restore discriminator Adam state while honoring the requested LR."""
    if "discriminator_optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        status = "loaded_with_lr_override"
    else:
        status = "reset_missing_state"
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
    return status


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    timeout_values: torch.Tensor,
    next_value: torch.Tensor,
    discount: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE with terminal and timeout transitions kept distinct.

    A true terminal state does not bootstrap. A time-limit truncation bootstraps
    from the value of its final pre-reset observation, but still cuts the GAE
    recursion so returns never cross episode boundaries.
    """
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for step in range(rewards.shape[0] - 1, -1, -1):
        done = torch.logical_or(terminated[step], truncated[step])
        continuation_value = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        bootstrap = torch.where(truncated[step], timeout_values[step], continuation_value)
        bootstrap = bootstrap * (~terminated[step]).float()
        delta = rewards[step] + discount * bootstrap - values[step]
        last = delta + discount * gae_lambda * (~done).float() * last
        advantages[step] = last
    return advantages, advantages + values


@dataclass
class Stage2Rollout:
    encoder_features: torch.Tensor
    bfm_observations: dict[str, torch.Tensor]
    teacher_actions: torch.Tensor
    teacher_action_targets: torch.Tensor
    commands: torch.Tensor
    raw_z: torch.Tensor
    old_log_prob: torch.Tensor
    values: torch.Tensor
    env_rewards: torch.Tensor
    env_reward_components: dict[str, torch.Tensor]
    locomotion_rewards: torch.Tensor
    locomotion_components: dict[str, torch.Tensor]
    terminated: torch.Tensor
    truncated: torch.Tensor
    crash: torch.Tensor
    fall_over: torch.Tensor
    timeout_values: torch.Tensor
    amp_features: torch.Tensor
    base_lin_vel: torch.Tensor
    base_ang_vel: torch.Tensor


class OnlineAMPHistory:
    def __init__(self, num_envs: int, history_length: int, dof_dim: int, device: torch.device):
        self.history_length = int(history_length)
        self.joint_history = torch.zeros(num_envs, history_length, dof_dim, device=device)

    def reset(self, joint_pos: torch.Tensor) -> None:
        self.joint_history[:] = joint_pos.unsqueeze(1)

    def append(self, joint_pos: torch.Tensor) -> None:
        self.joint_history = torch.cat([self.joint_history[:, 1:], joint_pos.unsqueeze(1)], dim=1)

    def reset_envs(self, env_ids: torch.Tensor, joint_pos: torch.Tensor) -> None:
        if env_ids.numel() > 0:
            self.joint_history[env_ids] = joint_pos[env_ids].unsqueeze(1)

    def feature(self, root_vel: torch.Tensor, key_positions_local: torch.Tensor) -> torch.Tensor:
        return torch.cat([root_vel, key_positions_local.reshape(key_positions_local.shape[0], -1), self.joint_history.reshape(self.joint_history.shape[0], -1)], dim=-1)


def _online_amp_feature(core, history: OnlineAMPHistory, key_body_indices: torch.Tensor) -> torch.Tensor:
    root_pos = core.robot_root_states[:, :3]
    relative = core.body_pos[:, key_body_indices] - root_pos.unsqueeze(1)
    num_keys = int(key_body_indices.numel())
    root_quat = core.base_quat.unsqueeze(1).expand(-1, num_keys, -1).reshape(-1, 4)
    local_positions = quat_rotate_inverse(root_quat, relative.reshape(-1, 3), w_last=True).reshape_as(relative)
    return history.feature(core.base_lin_vel, local_positions)


def _transition_core(live_core, info: Mapping[str, Any]):
    """Expose the target Isaac environment tensors under the UFO Stage2 contract."""
    terminal_state = info.get("terminal_state")
    if terminal_state is not None:
        return SimpleNamespace(
            **terminal_state,
            num_envs=live_core.num_envs,
            device=live_core.device,
        )
    simulator = live_core.simulator
    return SimpleNamespace(
        robot_root_states=simulator.robot_root_states,
        base_quat=live_core.base_quat,
        base_lin_vel=live_core.base_lin_vel,
        base_ang_vel=live_core.base_ang_vel,
        body_pos=simulator._rigid_body_pos,
        body_rot=simulator._rigid_body_rot,
        body_ang_vel=simulator._rigid_body_ang_vel,
        contact_forces=simulator.contact_forces,
        torques=live_core.torques,
        dof_pos=simulator.dof_pos,
        dof_vel=simulator.dof_vel,
        default_dof_pos=live_core.default_dof_pos,
        default_dof_pos_offset=live_core.default_dof_pos_offset,
        num_envs=live_core.num_envs,
        device=live_core.device,
    )


def _stage2_update_reset_buf(core) -> None:
    """Apply UFO's Stage2 crash and fall-over definitions to the Isaac core."""
    core.extras.pop("terminal_state", None)
    core.extras.pop("terminal_observation", None)
    if core.termination_contact_indices.numel() == 0:
        crash = torch.zeros(core.num_envs, dtype=torch.bool, device=core.device)
    else:
        contact_norm = core.simulator.contact_forces[:, core.termination_contact_indices].norm(dim=-1)
        crash = torch.any(contact_norm > 1.0, dim=-1)
    fall_over = core.projected_gravity[:, :2].norm(dim=-1) >= 0.9
    core.reset_buf |= crash
    core.reset_buf |= fall_over
    core.extras["termination_crash"] = crash.detach().clone()
    core.extras["termination_fall_over"] = fall_over.detach().clone()


def _stage2_prepare_pre_reset_transition(core, env_ids: torch.Tensor) -> None:
    """Capture the complete final transition before Isaac auto-resets slots."""
    if env_ids.numel() == 0:
        return
    simulator = core.simulator
    core.extras["terminal_state"] = {
        "robot_root_states": simulator.robot_root_states.detach().clone(),
        "base_quat": core.base_quat.detach().clone(),
        "base_lin_vel": core.base_lin_vel.detach().clone(),
        "base_ang_vel": core.base_ang_vel.detach().clone(),
        "projected_gravity": core.projected_gravity.detach().clone(),
        "body_pos": simulator._rigid_body_pos.detach().clone(),
        "body_rot": simulator._rigid_body_rot.detach().clone(),
        "body_ang_vel": simulator._rigid_body_ang_vel.detach().clone(),
        "contact_forces": simulator.contact_forces.detach().clone(),
        "torques": core.torques.detach().clone(),
        "dof_pos": simulator.dof_pos.detach().clone(),
        "dof_vel": simulator.dof_vel.detach().clone(),
        "default_dof_pos": core.default_dof_pos.detach().clone(),
        "default_dof_pos_offset": core.default_dof_pos_offset.detach().clone(),
    }
    if not torch.any(core.time_out_buf):
        return

    # Build the final observation against the pre-reset physics state. The
    # regular post-step path recomputes the live observation after resetting.
    core._compute_observations()
    raw = core.obs_buf_dict_raw["actor_obs"]
    core.extras["terminal_observation"] = {
        "state": torch.cat(
            [raw["dof_pos"], raw["dof_vel"], raw["projected_gravity"], raw["base_ang_vel"]],
            dim=-1,
        ).detach().clone(),
        "privileged_state": raw["max_local_self"].detach().clone(),
        "last_action": raw["actions"].detach().clone(),
        "history_actor": raw["history_actor"].detach().clone(),
    }


def _sample_commands(
    num_envs: int,
    device: torch.device,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    stand_prob: float = 0.1,
    turn_prob: float = 0.0,
) -> torch.Tensor:
    """Sample velocity targets with MimicLite Twist's stand-command gate."""
    if not 0.0 <= stand_prob <= 1.0:
        raise ValueError(f"stand_prob must be in [0, 1], got {stand_prob}")
    if not 0.0 <= turn_prob <= 1.0:
        raise ValueError(f"turn_prob must be in [0, 1], got {turn_prob}")
    commands = low + torch.rand(num_envs, 3, device=device) * (high - low)
    turn_in_place = torch.rand(num_envs, device=device) < turn_prob
    commands[turn_in_place, :2] = 0.0
    low_speed = commands[:, :2].norm(dim=-1) < 0.1
    stand = (~turn_in_place & low_speed) | (torch.rand(num_envs, device=device) < stand_prob)
    return torch.where(stand.unsqueeze(-1), torch.zeros_like(commands), commands)


def _mimiclite_resample_mask(
    episode_steps: torch.Tensor,
    commands: torch.Tensor,
    *,
    resample_interval: int = 300,
    resample_prob: float = 0.75,
    warmup_steps: int = 20,
) -> torch.Tensor:
    """Match MimicLite Twist's per-environment interval/probability resampling."""
    if resample_interval <= 0:
        raise ValueError(f"resample_interval must be positive, got {resample_interval}")
    if not 0.0 <= resample_prob <= 1.0:
        raise ValueError(f"resample_prob must be in [0, 1], got {resample_prob}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
    interval_reached = (episode_steps - warmup_steps) % resample_interval == 0
    standing = (commands[:, :2].norm(dim=-1) < 0.1) & (commands[:, 2].abs() < 0.1)
    return interval_reached & ((torch.rand(commands.shape[0], device=commands.device) < resample_prob) | standing)


MIMICLITE_LOCOMOTION_WEIGHTS = {
    # Keep direct planar tracking dominant while preserving a usable stability margin.
    "linvel_exp": 3.1,
    # Give signed command alignment a slightly stronger gradient, including for reverse vx.
    # Increase signed direction/yaw gradients after the widened command run
    # plateaued while preserving the baseline-normalized reward semantics.
    "linvel_projection": 1.5,
    "angvel_z_exp": 2.6,
    "single_foot_contact": 0.85,
    "angvel_xy_l2": 0.035,
    "body_upright": 1.1,
    # Match HT_lab_pipeline's PiPlus locomotion stand_still term: its positive
    # L1 pose error with weight -0.8 is represented here as a negative term.
    "stand_still": 0.8,
    "feet_air_time": 2.0,
    # Reward a single swing foot for clearing the ground without encouraging
    # double-support jumps.  The simulator foot contact threshold is 0.07 m,
    # so this target leaves a small but visible clearance margin.
    "feet_clearance": 1.0,
    "energy_l1": 2.0e-4,
    "joint_acc_l2": 1.0e-7,
    "action_rate_l2": 0.005,
    "action_rate2_l2": 0.005,
    "joint_vel_l2": 1.0e-3,
    "joint_deviation_l2": 0.11,
}

FEET_CLEARANCE_TARGET = 0.10
FEET_CLEARANCE_SIGMA = 0.04


def baseline_normalized_linvel_reward(
    base_lin_vel: torch.Tensor,
    commands: torch.Tensor,
    *,
    error_scale: float = LINVEL_EXP_ERROR_SCALE,
) -> torch.Tensor:
    """Return the positive exponential tracking reward used by ``piplus_walk``.

    Stage2 cannot optimize actions directly, so subtracting a zero-speed baseline
    creates a high-variance signed signal for the latent encoder. Keep the helper
    name for checkpoint/test compatibility while restoring the dense positive
    locomotion objective.
    """
    command_xy = commands[..., :2]
    tracking_error = (base_lin_vel[..., :2] - command_xy).square().sum(dim=-1)
    return torch.exp(-tracking_error / error_scale)


def baseline_normalized_angvel_reward(
    base_ang_vel_z: torch.Tensor,
    commands: torch.Tensor,
    *,
    error_scale: float = 0.25,
) -> torch.Tensor:
    """Return the positive exponential yaw tracking reward used by ``piplus_walk``."""
    command_z = commands[..., 2]
    tracking_error = (base_ang_vel_z - command_z).square()
    return torch.exp(-tracking_error / error_scale)


def amp_quadratic_reward(discriminator_score: torch.Tensor) -> torch.Tensor:
    """Match ``piplus_walk``'s positive quadratic discriminator reward."""
    return (1.0 - (discriminator_score - 1.0).square()).clamp_min(0.0)


def _global_tracking_sums(values: torch.Tensor) -> torch.Tensor:
    totals = values.detach().to(dtype=torch.float64)
    if _distributed_ready():
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    return totals


def command_tracking_metrics(
    commands: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
) -> dict[str, float]:
    """Summarize direct command errors and the learned command-to-speed response."""
    command = commands.reshape(-1, 3)
    achieved_xy = base_lin_vel.reshape(-1, base_lin_vel.shape[-1])[:, :2]
    achieved_yaw = base_ang_vel.reshape(-1, base_ang_vel.shape[-1])[:, 2]
    vx_error = (achieved_xy[:, 0] - command[:, 0]).abs()
    vy_error = (achieved_xy[:, 1] - command[:, 1]).abs()
    planar_error = torch.linalg.vector_norm(achieved_xy - command[:, :2], dim=-1)
    yaw_error = (achieved_yaw - command[:, 2]).abs()
    moving = command[:, :2].norm(dim=-1) >= MOVING_COMMAND_THRESHOLD
    turning = command[:, 2].abs() >= MOVING_COMMAND_THRESHOLD

    metrics: dict[str, float] = {}

    def add_mean(name: str, values: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        selected = values if mask is None else values[mask]
        totals = _global_tracking_sums(
            torch.stack((selected.sum(), selected.new_tensor(float(selected.numel()))))
        )
        metrics[name] = float(totals[0] / totals[1].clamp_min(1.0))

    add_mean("tracking/vx_mae", vx_error)
    add_mean("tracking/vy_mae", vy_error)
    add_mean("tracking/planar_l2_mae", planar_error)
    add_mean("tracking/yaw_rate_mae", yaw_error)
    add_mean("tracking/nonzero_yaw_rate_mae", yaw_error, turning)
    add_mean("tracking/nonzero_vx_mae", vx_error, moving)
    add_mean("tracking/nonzero_vy_mae", vy_error, moving)
    add_mean("tracking/nonzero_planar_l2_mae", planar_error, moving)

    bin_masks = {
        "backward": moving & (command[:, 0] < -0.05),
        "near_zero": moving & (command[:, 0].abs() <= 0.05),
        "slow_forward": moving & (command[:, 0] > 0.05) & (command[:, 0] <= 0.4),
        "fast_forward": moving & (command[:, 0] > 0.4),
    }
    global_count = _global_tracking_sums(command.new_tensor(float(command.shape[0]))).clamp_min(1.0)
    for name, mask in bin_masks.items():
        add_mean(f"tracking/vx_bin_{name}_mae", vx_error, mask)
        bin_count = _global_tracking_sums(mask.sum().to(dtype=torch.float64))
        metrics[f"tracking/vx_bin_{name}_fraction"] = float(bin_count / global_count)

    response_mask = moving & (command[:, 0].abs() > 0.05)
    x = command[response_mask, 0]
    y = achieved_xy[response_mask, 0]
    response_sums = _global_tracking_sums(
        torch.stack(
            (
                x.new_tensor(float(x.numel())),
                x.sum(),
                y.sum(),
                x.square().sum(),
                y.square().sum(),
                (x * y).sum(),
            )
        )
    )
    count, sum_x, sum_y, sum_xx, sum_yy, sum_xy = response_sums
    denominator = count.clamp_min(1.0)
    variance_x = (sum_xx - sum_x.square() / denominator).clamp_min(0.0)
    variance_y = (sum_yy - sum_y.square() / denominator).clamp_min(0.0)
    covariance = sum_xy - sum_x * sum_y / denominator
    slope = covariance / variance_x.clamp_min(1.0e-12)
    correlation = covariance / (variance_x * variance_y).sqrt().clamp_min(1.0e-12)
    valid = bool(count >= 2.0 and variance_x > 1.0e-12 and variance_y > 1.0e-12)
    metrics["tracking/nonzero_vx_response_slope"] = float(slope) if valid else 0.0
    metrics["tracking/nonzero_vx_command_correlation"] = float(correlation.clamp(-1.0, 1.0)) if valid else 0.0
    metrics["tracking/nonzero_achieved_vx_std"] = float((variance_y / denominator).sqrt()) if count >= 2.0 else 0.0

    x = command[turning, 2]
    y = achieved_yaw[turning]
    response_sums = _global_tracking_sums(
        torch.stack(
            (
                x.new_tensor(float(x.numel())),
                x.sum(),
                y.sum(),
                x.square().sum(),
                y.square().sum(),
                (x * y).sum(),
            )
        )
    )
    count, sum_x, sum_y, sum_xx, sum_yy, sum_xy = response_sums
    denominator = count.clamp_min(1.0)
    variance_x = (sum_xx - sum_x.square() / denominator).clamp_min(0.0)
    variance_y = (sum_yy - sum_y.square() / denominator).clamp_min(0.0)
    covariance = sum_xy - sum_x * sum_y / denominator
    slope = covariance / variance_x.clamp_min(1.0e-12)
    correlation = covariance / (variance_x * variance_y).sqrt().clamp_min(1.0e-12)
    valid = bool(count >= 2.0 and variance_x > 1.0e-12 and variance_y > 1.0e-12)
    metrics["tracking/nonzero_yaw_response_slope"] = float(slope) if valid else 0.0
    metrics["tracking/nonzero_yaw_command_correlation"] = float(correlation.clamp(-1.0, 1.0)) if valid else 0.0
    metrics["tracking/nonzero_achieved_yaw_std"] = float((variance_y / denominator).sqrt()) if count >= 2.0 else 0.0
    return metrics


class MimicLiteLocomotionRewardState:
    """PiPlus implementation of MimicLite locomotion terms."""

    def __init__(
        self,
        num_envs: int,
        num_dof: int,
        dt: float,
        feet_indices: torch.Tensor,
        torso_index: int,
        joint_vel_indices: torch.Tensor,
        joint_deviation_indices: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.dt = float(dt)
        self.feet_indices = feet_indices.to(device=device, dtype=torch.long)
        self.torso_index = int(torso_index)
        self.joint_vel_indices = joint_vel_indices.to(device=device, dtype=torch.long)
        self.joint_deviation_indices = joint_deviation_indices.to(device=device, dtype=torch.long)
        self.prev_actions = torch.zeros(num_envs, num_dof, device=device)
        self.prev_prev_actions = torch.zeros_like(self.prev_actions)
        self.prev_dof_vel = torch.zeros_like(self.prev_actions)
        self.contact_time = torch.zeros(num_envs, len(self.feet_indices), device=device)
        self.air_time = torch.zeros_like(self.contact_time)
        self.prev_contact = torch.zeros_like(self.contact_time, dtype=torch.bool)

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.prev_actions[env_ids] = 0.0
        self.prev_prev_actions[env_ids] = 0.0
        self.prev_dof_vel[env_ids] = 0.0
        self.contact_time[env_ids] = 0.0
        self.air_time[env_ids] = 0.0
        self.prev_contact[env_ids] = False

    def compute(
        self,
        core,
        commands: torch.Tensor,
        actions: torch.Tensor,
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        contact = core.contact_forces[:, self.feet_indices, 2] > 1.0
        first_contact = contact & ~self.prev_contact
        last_air_time = self.air_time
        contact_time = torch.where(contact, self.contact_time + self.dt, torch.zeros_like(self.contact_time))
        air_time = torch.where(contact, torch.zeros_like(self.air_time), self.air_time + self.dt)

        standing = (commands[:, :2].norm(dim=-1) < 0.1) & (commands[:, 2].abs() < 0.1)
        command_speed = commands[:, :2].norm(dim=-1)
        linvel_exp = baseline_normalized_linvel_reward(core.base_lin_vel, commands)
        linvel_projection = (core.base_lin_vel[:, :2] * commands[:, :2]).sum(dim=-1).clamp_max(command_speed)
        angvel_z_exp = baseline_normalized_angvel_reward(core.base_ang_vel[:, 2], commands)

        down = torch.zeros(core.num_envs, 3, device=core.device)
        down[:, 2] = -1.0
        torso_quat = core.body_rot[:, self.torso_index]
        torso_gravity = quat_rotate_inverse(torso_quat, down, w_last=True)
        body_upright = 1.0 - torso_gravity[:, :2].square().sum(dim=-1)
        torso_angvel = quat_rotate_inverse(core.body_rot[:, self.torso_index], core.body_ang_vel[:, self.torso_index], w_last=True)
        angvel_xy_l2 = -torso_angvel[:, :2].square().sum(dim=-1)

        single_contact = torch.where((contact_time > 0.1).sum(dim=-1) == 1, 0.0, -1.0)
        single_contact = torch.where(standing, torch.zeros_like(single_contact), single_contact)
        feet_air_time = ((last_air_time - 0.5).clamp_max(0.0) * first_contact).sum(dim=-1)
        feet_air_time = torch.where(standing, torch.zeros_like(feet_air_time), feet_air_time)
        feet_height = core.body_pos[:, self.feet_indices, 2]
        swing = ~contact
        swing_count = swing.sum(dim=-1)
        clearance_score = torch.exp(
            -((feet_height - FEET_CLEARANCE_TARGET) / FEET_CLEARANCE_SIGMA).square()
        )
        feet_clearance = (clearance_score * swing).sum(dim=-1) / swing_count.clamp_min(1)
        valid_swing = ((contact_time > 0.1).sum(dim=-1) == 1) & (swing_count == 1) & ~standing
        feet_clearance = torch.where(valid_swing, feet_clearance, torch.zeros_like(feet_clearance))

        energy_l1 = -(core.torques * core.dof_vel).abs().sum(dim=-1)
        joint_acc_l2 = -((core.dof_vel - self.prev_dof_vel) / max(self.dt, 1.0e-6)).square().sum(dim=-1)
        action_delta = actions - self.prev_actions
        action_rate_l2 = -action_delta.square().sum(dim=-1)
        action_rate2_l2 = -(actions - 2.0 * self.prev_actions + self.prev_prev_actions).square().sum(dim=-1)
        joint_vel_l2 = -core.dof_vel[:, self.joint_vel_indices].square().sum(dim=-1)
        default_dof_pos = core.default_dof_pos + core.default_dof_pos_offset
        stand_still = -(core.dof_pos - default_dof_pos).abs().sum(dim=-1)
        stand_still = torch.where(standing, stand_still, torch.zeros_like(stand_still))
        joint_deviation_l2 = -(core.dof_pos[:, self.joint_deviation_indices] - default_dof_pos[:, self.joint_deviation_indices]).square().sum(dim=-1)

        components = {
            "linvel_exp": linvel_exp,
            "linvel_projection": linvel_projection,
            "angvel_z_exp": angvel_z_exp,
            "single_foot_contact": single_contact,
            "angvel_xy_l2": angvel_xy_l2,
            "body_upright": body_upright,
            "stand_still": stand_still,
            "feet_air_time": feet_air_time,
            "feet_clearance": feet_clearance,
            "energy_l1": energy_l1,
            "joint_acc_l2": joint_acc_l2,
            "action_rate_l2": action_rate_l2,
            "action_rate2_l2": action_rate2_l2,
            "joint_vel_l2": joint_vel_l2,
            "joint_deviation_l2": joint_deviation_l2,
        }
        # MimicLite scales the complete task reward by the control timestep.
        # Keep these values in their final, weighted form so logs expose the
        # actual PPO contribution of every copied reward term.
        weighted_components = {
            name: self.dt * MIMICLITE_LOCOMOTION_WEIGHTS[name] * value for name, value in components.items()
        }
        reward = sum(weighted_components.values())

        self.prev_prev_actions = self.prev_actions.clone()
        self.prev_actions = actions.detach().clone()
        self.prev_dof_vel = core.dof_vel.detach().clone()
        self.contact_time = contact_time
        self.air_time = air_time
        self.prev_contact = contact
        if torch.any(done):
            self.reset(done.nonzero(as_tuple=False).squeeze(-1))
        return reward, weighted_components


def _to_torch_obs(obs: Mapping[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in obs.items() if key != "time"}


def _bfm_action(bfm_model, obs: Mapping[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
    normalized_obs = bfm_model._normalize(dict(obs))
    distribution = bfm_model._actor(normalized_obs, z, bfm_model.cfg.actor_std)
    return distribution.mean.float()


def _freeze_bfm(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _load_piplus_robot_contract(robot_config: str | Path) -> SimpleNamespace:
    from omegaconf import OmegaConf

    config_path = Path(robot_config).expanduser().resolve()
    config = OmegaConf.load(config_path)
    robot_cfg = config.robot
    joint_names = tuple(str(name) for name in robot_cfg.dof_names)
    body_names = tuple(str(name) for name in robot_cfg.body_names)
    xml_path = resolve_asset_path(robot_cfg.asset.asset_root, robot_cfg.asset.xml_file)
    if not xml_path.is_file():
        raise FileNotFoundError(f"PiPlus MuJoCo XML does not exist: {xml_path}")
    return SimpleNamespace(
        config_path=config_path,
        policy_joint_names=joint_names,
        bfm_action_position_scales=_bfm_action_position_scales(config_path),
        fixed_joint_names=(),
        robot=SimpleNamespace(
            base_body="base_link",
            body_names=body_names,
            control_joint_names=joint_names,
            xml_path=xml_path.resolve(),
        ),
    )


def build_piplus_locomotion_env(
    *,
    device: str,
    robot_config: str,
    expert_dataset: str,
    num_envs: int,
    seed: int,
    max_episode_length_s: float,
    simulator: str = "isaacsim",
    locomotion_mode: bool = True,
    disable_obs_noise: bool = False,
    disable_domain_randomization: bool = False,
):
    if not locomotion_mode:
        raise ValueError("PiPlus AMP Stage2 requires locomotion_mode=True")
    if simulator not in {"isaacsim", "mujoco"}:
        raise ValueError(f"Unsupported Stage2 playback simulator: {simulator}")
    robot_training = _load_piplus_robot_contract(robot_config)
    hydra_overrides = [
        "robot=piplus/PiPlus_S_12L8A0G2H1W_LSE",
        f"simulator={simulator}",
        # Keep playback/evaluation scenes independent of the training config's
        # recorded env count (for example, a 4096-env run directory).
        f"num_envs={num_envs}",
        f"simulator.config.scene.num_envs={num_envs}",
        "env.config.resample_motion_when_training=False",
        "env.config.termination.terminate_when_motion_end=False",
        "env.config.termination.terminate_when_motion_far=False",
        # HumanoidVerseIsaacConfig requires the generic contact/gravity flags
        # to stay disabled. Stage2 applies UFO's stricter crash/fall rules in
        # _stage2_update_reset_buf instead.
        "env.config.termination.terminate_by_contact=False",
        "env.config.termination.terminate_by_gravity=False",
        "env.config.termination.terminate_by_low_height=False",
        "env.config.lie_down_init=False",
        "+rewards.reward_scales.survival=2.0",
        "rewards.reward_scales.penalty_undesired_contact=-1.0",
        # Keep the wider command curriculum from being paid for with overly
        # abrupt policy changes on the hardware-facing action interface.
        "rewards.reward_scales.penalty_action_rate=-0.55",
    ]
    env_config = HumanoidVerseIsaacConfig(
        name="humanoidverse_isaac",
        device=device,
        lafan_tail_path=expert_dataset,
        max_episode_length_s=max_episode_length_s,
        disable_obs_noise=disable_obs_noise,
        disable_domain_randomization=disable_domain_randomization,
        relative_config_path="exp/bfm_zero_piplus/bfm_zero_piplus",
        include_last_action=True,
        hydra_overrides=hydra_overrides,
        include_history_actor=True,
        root_height_obs=True,
    )
    torch.manual_seed(seed)
    env = env_config.build(num_envs=num_envs)[0]

    # Stage2 is command-conditioned locomotion, not motion tracking. Reuse the
    # target environment while routing resets through the base locomotion logic.
    from types import MethodType

    base_env = env._env
    base_env._reset_tasks_callback = MethodType(LeggedRobotBase._reset_tasks_callback, base_env)
    base_env._reset_dofs = MethodType(LeggedRobotBase._reset_dofs, base_env)
    base_env._reset_root_states = MethodType(LeggedRobotBase._reset_root_states, base_env)
    base_env._update_reset_buf = MethodType(_stage2_update_reset_buf, base_env)
    base_env._prepare_pre_reset_transition = MethodType(_stage2_prepare_pre_reset_transition, base_env)
    return env, robot_training


def _ensure_runtime_cache(work_dir: Path) -> None:
    # Keep multiprocessing socket paths short; long run directories can exceed
    # AF_UNIX's path limit when the motion loader starts a Manager process.
    cache_root = Path(os.environ.get("UFO_STAGE2_CACHE_DIR", "/tmp/ufo_amp_stage2_cache"))
    for variable, subdir in {
        "TMPDIR": "tmp",
        "TEMP": "tmp",
        "TMP": "tmp",
        "WARP_CACHE_PATH": "warp",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "TRITON_CACHE_DIR": "triton",
    }.items():
        path = cache_root / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(path))


def ppo_update(
    policy: CommandEncoderPolicy,
    rollout: Stage2Rollout,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    epochs: int,
    minibatch_size: int,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
    target_kl: float | None = 0.01,
    bfm_model=None,
    action_imitation_coef: float = 0.0,
    bfm_action_position_scales: torch.Tensor | None = None,
) -> dict[str, float]:
    features = rollout.encoder_features.reshape(-1, rollout.encoder_features.shape[-1])
    raw_z = rollout.raw_z.reshape(-1, rollout.raw_z.shape[-1])
    old_log_prob = rollout.old_log_prob.reshape(-1)
    advantages = advantages.reshape(-1)
    returns = returns.reshape(-1)
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1.0e-6)
    if action_imitation_coef < 0.0:
        raise ValueError(f"action_imitation_coef must be non-negative, got {action_imitation_coef}")
    imitation_enabled = action_imitation_coef > 0.0
    if imitation_enabled:
        if bfm_model is None:
            raise ValueError("bfm_model is required when action imitation is enabled")
        bfm_observations = {
            key: value.reshape(-1, value.shape[-1]) for key, value in rollout.bfm_observations.items()
        }
        teacher_action_targets = rollout.teacher_action_targets.reshape(
            -1, rollout.teacher_action_targets.shape[-1]
        )
        if bfm_action_position_scales is None:
            raise ValueError("bfm_action_position_scales is required when action imitation is enabled")
        if not bfm_observations or teacher_action_targets.shape[0] != features.shape[0]:
            raise ValueError("Action imitation rollout tensors do not match encoder feature count")

    metric_sums = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "ratio_mean": 0.0,
        "ratio_max": 0.0,
        "grad_norm": 0.0,
        "raw_z_norm": 0.0,
        "action_imitation_loss": 0.0,
        "action_imitation_mae": 0.0,
    }
    update_count = 0
    early_stop = False
    sample_count = features.shape[0]
    expected_update_count = int(epochs) * math.ceil(sample_count / int(minibatch_size))
    for _ in range(int(epochs)):
        for indices in torch.randperm(sample_count, device=features.device).split(int(minibatch_size)):
            distribution = policy.distribution(features[indices])
            log_prob = distribution.log_prob(raw_z[indices]).sum(dim=-1)
            entropy = distribution.entropy().mean(dim=-1).mean()
            _, _, value = policy(features[indices])
            # Bound the exponent before exp() so one bad minibatch cannot poison
            # all subsequent optimizer steps with inf/nan ratios.
            log_ratio = (log_prob - old_log_prob[indices]).clamp(-20.0, 20.0)
            ratio = log_ratio.exp()
            unclipped = ratio * advantages[indices]
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(value, returns[indices])
            action_imitation_loss = features.new_zeros(())
            action_imitation_mae = features.new_zeros(())
            if imitation_enabled:
                mean_z = policy.deterministic_z(features[indices])
                predicted_action = _bfm_action(
                    bfm_model,
                    {key: value[indices] for key, value in bfm_observations.items()},
                    bfm_model.project_z(mean_z),
                )
                bfm_scales = bfm_action_position_scales.to(device=predicted_action.device, dtype=predicted_action.dtype)
                predicted_offset = predicted_action * bfm_scales
                target_offset = teacher_action_targets[indices].to(predicted_offset)
                action_imitation_loss = F.smooth_l1_loss(predicted_offset, target_offset)
                action_imitation_mae = (predicted_offset - target_offset).abs().mean()
            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
                + action_imitation_coef * action_imitation_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            average_gradients(policy.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            approx_kl = (old_log_prob[indices] - log_prob).mean()
            # Every rank must make the same early-stop decision. Otherwise one
            # rank can leave the optimizer loop earlier and deadlock the next
            # distributed gradient collective.
            synced_kl = approx_kl.detach().clone()
            if _distributed_ready():
                torch.distributed.all_reduce(synced_kl, op=torch.distributed.ReduceOp.SUM)
                synced_kl.div_(torch.distributed.get_world_size())
            metric_sums["policy_loss"] += float(policy_loss.detach())
            metric_sums["value_loss"] += float(value_loss.detach())
            metric_sums["entropy"] += float(entropy.detach())
            metric_sums["approx_kl"] += float(synced_kl)
            metric_sums["clip_fraction"] += float((log_ratio.abs() > clip_ratio).float().mean().detach())
            metric_sums["ratio_mean"] += float(ratio.mean().detach())
            metric_sums["ratio_max"] += float(ratio.max().detach())
            metric_sums["grad_norm"] += float(grad_norm.detach())
            metric_sums["raw_z_norm"] += float(raw_z[indices].norm(dim=-1).mean().detach())
            metric_sums["action_imitation_loss"] += float(action_imitation_loss.detach())
            metric_sums["action_imitation_mae"] += float(action_imitation_mae.detach())
            update_count += 1

            if target_kl is not None and target_kl > 0.0 and float(synced_kl) > 1.5 * target_kl:
                early_stop = True
                break
        if early_stop:
            break

    if update_count == 0:
        return {
            **metric_sums,
            "ppo_updates": 0.0,
            "ppo_expected_updates": float(expected_update_count),
            "ppo_update_fraction": 0.0,
            "ppo_early_stop": 0.0,
        }
    metrics = {name: value / update_count for name, value in metric_sums.items()}
    metrics["ppo_updates"] = float(update_count)
    metrics["ppo_expected_updates"] = float(expected_update_count)
    metrics["ppo_update_fraction"] = float(update_count / max(expected_update_count, 1))
    metrics["ppo_early_stop"] = float(early_stop)
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a command encoder with AMP on top of a frozen PiPlus BFM actor.")
    parser.add_argument(
        "--bfm-checkpoint",
        default=DEFAULT_BFM_CHECKPOINT,
        help="First-stage model checkpoint directory containing config.json and model/.",
    )
    parser.add_argument(
        "--teacher-policy",
        default=DEFAULT_TEACHER_POLICY,
        help="Chen Jianhong's 23DoF AMP teacher directory or ZIP archive.",
    )
    parser.add_argument("--expert-dataset", default=DEFAULT_EXPERT_DATASET)
    parser.add_argument("--robot-config", default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--gpu-ids",
        default="single",
        help="GPU ids for the local torchrun launcher: single, all, or comma-separated ids.",
    )
    parser.add_argument("--work-dir", default="runs/amp_stage2_piplus_lse")
    parser.add_argument("--resume", default=None, help="Stage2 checkpoint (.pt) to resume, including optimizer and normalizer state.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--command-resample-steps", type=int, default=300)
    parser.add_argument("--command-resample-prob", type=float, default=0.75)
    parser.add_argument("--command-stand-prob", type=float, default=0.05)
    parser.add_argument("--command-turn-prob", type=float, default=0.05)
    parser.add_argument("--command-warmup-steps", type=int, default=20)
    parser.add_argument("--command-smoothing", type=float, default=0.02)
    parser.add_argument("--latent-stat-max-frames", type=int, default=4096, help="Expert frames used for automatic BFM latent validation; 0 means all.")
    parser.add_argument(
        "--latent-reference-size",
        type=int,
        default=1024,
        help="Expert projected latent bank size used by the direction prior and manifold diagnostics.",
    )
    parser.add_argument(
        "--latent-prior-weight",
        type=float,
        default=0.02,
        help="Penalty weight for nearest-expert projected latent direction distance.",
    )
    parser.add_argument(
        "--latent-diagnostic-samples",
        type=int,
        default=512,
        help="Maximum generated/reference samples per command bin for latent diagnostics.",
    )
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-kl", type=float, default=0.04, help="Early-stop PPO epochs after KL exceeds 1.5x this value; <=0 disables.")
    parser.add_argument("--discriminator-learning-rate", type=float, default=1e-4)
    parser.add_argument("--amp-weight", type=float, default=0.25)
    parser.add_argument(
        "--action-imitation-coef",
        type=float,
        default=1.0,
        help="Coefficient of the differentiable teacher-action distillation loss.",
    )
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--env-reward-weight", type=float, default=1.0)
    parser.add_argument("--locomotion-reward-weight", type=float, default=1.1)
    parser.add_argument("--max-episode-length-s", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true", help="Run one small CPU/GPU validation iteration.")
    parser.add_argument("--dry-run", action="store_true", help="Validate checkpoint, robot contract, and AMP data without starting Isaac.")
    return parser.parse_args()


def _distributed_context(args: argparse.Namespace) -> tuple[argparse.Namespace, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return args, rank, local_rank, world_size

    visible_devices = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    isolate_worker_gpu = os.environ.get("HT_BFM_ISOLATE_WORKER_GPU", "1") != "0"
    if isolate_worker_gpu and len(visible_devices) > 1:
        if local_rank >= len(visible_devices):
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} cannot select from CUDA_VISIBLE_DEVICES={visible_devices}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices[local_rank]
        os.environ["LOCAL_RANK"] = "0"
        device_rank = 0
    else:
        device_rank = local_rank

    if not torch.cuda.is_available():
        raise RuntimeError("AMP stage2 distributed training requires CUDA")
    from datetime import timedelta

    import torch.distributed as dist

    torch.cuda.set_device(device_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=2),
        )
    args.device = f"cuda:{device_rank}"
    args.seed += rank
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_rank)
    if isolate_worker_gpu:
        rank_cache = Path(os.environ.get("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))) / f"isaac_worker_{rank}"
        rank_cache.mkdir(parents=True, exist_ok=True)
        os.environ["OV_DATA_PATH"] = str(rank_cache / "ov_data")
        os.environ["OMNI_USER_DIR"] = str(rank_cache / "omni_user")
    return args, rank, local_rank, world_size


def _run_dry_validation(args: argparse.Namespace) -> None:
    """Validate all model/data contracts without constructing an Isaac environment."""
    robot_training = _load_piplus_robot_contract(args.robot_config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    teacher = PiPlusAMPTeacherPolicy(args.teacher_policy, device)
    stage2_to_teacher = _joint_permutation(tuple(robot_training.policy_joint_names), TEACHER_POLICY_JOINT_NAMES)
    teacher_to_stage2 = _joint_permutation(TEACHER_POLICY_JOINT_NAMES, tuple(robot_training.policy_joint_names))
    checkpoint = Path(args.bfm_checkpoint).expanduser().resolve()
    bfm_load_device = "cuda" if device.type == "cuda" else "cpu"
    bfm_model = load_model_from_checkpoint_dir(checkpoint, device=bfm_load_device)
    _freeze_bfm(bfm_model)
    expert = PiPlusAMPExpertDataset.from_pkl(
        args.expert_dataset,
        robot=robot_training.robot,
        xml_path=robot_training.robot.xml_path,
        policy_joint_names=robot_training.policy_joint_names,
        key_bodies=DEFAULT_KEY_BODIES,
        history_length=args.history_length,
    )
    action_dim = int(getattr(bfm_model, "action_dim", -1))
    if action_dim != len(robot_training.policy_joint_names):
        raise ValueError(
            f"BFM action dimension {action_dim} does not match PiPlus policy joints "
            f"{len(robot_training.policy_joint_names)}"
        )
    with torch.no_grad():
        projected = bfm_model.project_z(torch.randn(2, int(bfm_model.cfg.archi.z_dim), device=device))
    print(
        json.dumps(
            {
                "bfm_checkpoint": str(checkpoint),
                "model_class": type(bfm_model).__name__,
                "action_dim": action_dim,
                "z_dim": int(bfm_model.cfg.archi.z_dim),
                "norm_z": bool(bfm_model.cfg.archi.norm_z),
                "policy_joint_count": len(robot_training.policy_joint_names),
                "teacher_policy": teacher.artifact_path,
                "teacher_input_dim": TEACHER_POLICY_INPUT_DIM,
                "teacher_action_dim": int(teacher.net[-1].out_features),
                "stage2_to_teacher_joint_permutation": stage2_to_teacher.tolist(),
                "teacher_to_stage2_joint_permutation": teacher_to_stage2.tolist(),
                "expert_feature_dim": expert.feature_dim,
                "expert_motion_count": expert.motion_count,
                "expert_frame_count": int(expert.features.shape[0]),
                "projected_z_norm": float(projected[0].norm()),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main(parsed_args: argparse.Namespace | None = None) -> None:
    args = _parse_args() if parsed_args is None else parsed_args
    if getattr(args, "dry_run", False):
        _run_dry_validation(args)
        return
    args, rank, local_rank, world_size = _distributed_context(args)
    rank0 = rank == 0
    if args.smoke:
        args.num_envs = min(args.num_envs, 2)
        args.iterations = 1
        args.rollout_steps = min(args.rollout_steps, 4)
        args.ppo_epochs = 1
        args.minibatch_size = min(args.minibatch_size, args.num_envs * args.rollout_steps)
    if args.latent_stat_max_frames < 0:
        raise ValueError("latent_stat_max_frames must be >= 0")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    _ensure_runtime_cache(work_dir)
    env, robot_training = build_piplus_locomotion_env(
        device=args.device,
        robot_config=args.robot_config,
        expert_dataset=args.expert_dataset,
        num_envs=args.num_envs,
        seed=args.seed,
        max_episode_length_s=args.max_episode_length_s,
    )
    # Checkpoint configs use the device literal ``cuda``; each torchrun
    # worker has already selected its local CUDA device above.
    bfm_load_device = "cuda" if device.type == "cuda" else "cpu"
    bfm_model = load_model_from_checkpoint_dir(args.bfm_checkpoint, device=bfm_load_device)
    _freeze_bfm(bfm_model)
    teacher = PiPlusAMPTeacherPolicy(args.teacher_policy, device)
    stage2_joint_names = tuple(robot_training.policy_joint_names)
    teacher_action_scales = torch.tensor(TEACHER_POLICY_ACTION_SCALES, device=device)
    bfm_action_scales = torch.tensor(robot_training.bfm_action_position_scales, device=device)
    _joint_permutation(stage2_joint_names, TEACHER_POLICY_JOINT_NAMES)
    _joint_permutation(TEACHER_POLICY_JOINT_NAMES, stage2_joint_names)
    bfm_action_dim = int(getattr(bfm_model, "action_dim", -1))
    env_action_dim = int(env.single_action_space.shape[0])
    if env_action_dim != len(robot_training.policy_joint_names):
        raise ValueError(
            f"PiPlus environment action dimension {env_action_dim} does not match policy joints "
            f"{len(robot_training.policy_joint_names)}"
        )
    if bfm_action_dim != env_action_dim:
        raise ValueError(
            f"BFM action dimension {bfm_action_dim} does not match PiPlus environment action dimension {env_action_dim}"
        )

    obs, _ = env.reset(to_numpy=False, reset_to_default_pose=True)
    obs_t = _to_torch_obs(obs, device)
    commands_low = torch.tensor([-0.5, -0.5, -1.0], device=device)
    commands_high = torch.tensor([1.0, 0.5, 1.0], device=device)
    if not 0.0 <= args.command_smoothing <= 1.0:
        raise ValueError(f"command_smoothing must be in [0, 1], got {args.command_smoothing}")
    if args.latent_reference_size <= 0:
        raise ValueError(f"latent_reference_size must be positive, got {args.latent_reference_size}")
    if args.latent_prior_weight < 0.0:
        raise ValueError(f"latent_prior_weight must be non-negative, got {args.latent_prior_weight}")
    if args.latent_diagnostic_samples <= 0:
        raise ValueError(f"latent_diagnostic_samples must be positive, got {args.latent_diagnostic_samples}")
    commands = torch.zeros(args.num_envs, 3, device=device)
    command_targets = torch.zeros_like(commands)
    command_episode_steps = torch.zeros(args.num_envs, device=device, dtype=torch.long)
    teacher_observation = build_teacher_policy_observation(
        obs_t,
        commands,
        stage2_joint_names,
        teacher_action_scales=teacher_action_scales,
        bfm_action_scales=bfm_action_scales,
    )
    encoder_input = teacher.normalize(teacher_observation)
    input_scale = torch.ones(encoder_input.shape[-1], device=device, dtype=encoder_input.dtype)
    backward_arch = bfm_model.cfg.archi.b
    policy = CommandEncoderPolicy(
        encoder_input.shape[-1],
        int(bfm_model.cfg.archi.z_dim),
        hidden_dim=int(backward_arch.hidden_dim),
        hidden_layers=int(backward_arch.hidden_layers),
    ).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    latent_stats = _collect_bfm_latent_stats(
        bfm_agent=None,
        bfm_model=bfm_model,
        stats_env=env,
        device=args.device,
        robot_config=args.robot_config,
        expert_dataset=args.expert_dataset,
        num_envs=args.num_envs,
        seed=args.seed,
        max_episode_length_s=args.max_episode_length_s,
        max_frames=args.latent_stat_max_frames,
    )
    latent_contract = _validate_and_configure_latent_contract(
        policy=policy,
        bfm_model=bfm_model,
        latent_stats=latent_stats,
    )
    latent_reference = latent_stats["_reference_projected"].to(device=device)
    if latent_reference.shape[0] > args.latent_reference_size:
        reference_generator = torch.Generator(device=device)
        reference_generator.manual_seed(29)
        reference_indices = torch.randperm(
            latent_reference.shape[0], generator=reference_generator, device=device
        )[: args.latent_reference_size]
        latent_reference = latent_reference[reference_indices]
    if rank0:
        print(
            json.dumps(
                {
                    "bfm_latent_check": {
                        "z_dim": int(bfm_model.cfg.archi.z_dim),
                        "norm_z": bool(bfm_model.cfg.archi.norm_z),
                        "frames": latent_stats["frame_count"],
                        **latent_contract,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )

    simulator_body_names = tuple(env._env.simulator._body_list)
    missing_key_bodies = [name for name in DEFAULT_KEY_BODIES if name not in simulator_body_names]
    if missing_key_bodies:
        raise ValueError(f"PiPlus simulator is missing AMP key bodies: {missing_key_bodies}")
    key_body_ids = torch.tensor([simulator_body_names.index(name) for name in DEFAULT_KEY_BODIES], device=device, dtype=torch.long)
    expert = PiPlusAMPExpertDataset.from_pkl(
        args.expert_dataset,
        robot=robot_training.robot,
        xml_path=robot_training.robot.xml_path,
        policy_joint_names=tuple(robot_training.policy_joint_names),
        key_bodies=DEFAULT_KEY_BODIES,
        history_length=args.history_length,
    )
    discriminator = AMPDiscriminator(expert.feature_dim).to(device)
    broadcast_module_state(policy)
    broadcast_module_state(discriminator)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_learning_rate)
    reward_normalizer = MimicLiteRewardNormalizer().to(device)
    start_iteration = 0
    policy_optimizer_resume = "fresh"
    discriminator_optimizer_resume = "fresh"
    legacy_input_migrated = False
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Stage2 resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        legacy_input_migrated = load_command_encoder_policy_state(policy, checkpoint, input_scale)
        checkpoint_discriminator = checkpoint.get("discriminator")
        discriminator_architecture_compatible = True
        if checkpoint_discriminator is not None:
            try:
                discriminator.load_state_dict(checkpoint_discriminator)
            except RuntimeError:
                # PiPlus uses the wider 1024/512 MSE discriminator. The old
                # Stage2 WGAN checkpoint is still valid for the encoder, but
                # its discriminator tensors cannot be loaded into this net.
                discriminator_architecture_compatible = False
        policy_optimizer_resume = restore_policy_optimizer(
            policy_optimizer,
            checkpoint,
            learning_rate=args.learning_rate,
            reset_for_input_migration=legacy_input_migrated,
        )
        checkpoint_objective = checkpoint.get("metadata", {}).get("amp_discriminator_objective")
        if discriminator_architecture_compatible and checkpoint_objective == AMP_DISCRIMINATOR_OBJECTIVE:
            discriminator_optimizer_resume = restore_discriminator_optimizer(
                discriminator_optimizer,
                checkpoint,
                learning_rate=args.discriminator_learning_rate,
            )
        else:
            # Do not carry WGAN-GP optimizer moments into the PiPlus MSE/quad objective.
            discriminator_optimizer_resume = (
                "reset_for_architecture_change" if not discriminator_architecture_compatible else "reset_for_objective_change"
            )
        reward_normalizer.load_state_dict(checkpoint["amp_reward_normalizer"])
        start_iteration = int(checkpoint.get("iteration", 0))
        if start_iteration < 0 or start_iteration > args.iterations:
            raise ValueError(f"Resume iteration {start_iteration} is outside [0, {args.iterations}]")
        if rank0:
            print(
                json.dumps(
                    {
                        "resumed_from": str(resume_path.resolve()),
                        "start_iteration": start_iteration,
                        "legacy_encoder_input_migrated": legacy_input_migrated,
                        "policy_optimizer_resume": policy_optimizer_resume,
                        "policy_learning_rate": args.learning_rate,
                        "discriminator_optimizer_resume": discriminator_optimizer_resume,
                        "discriminator_learning_rate": args.discriminator_learning_rate,
                    }
                ),
                flush=True,
            )
    online_history = OnlineAMPHistory(args.num_envs, args.history_length, len(robot_training.policy_joint_names), device)
    online_history.reset(env._env.simulator.dof_pos)
    control_joint_names = tuple(robot_training.policy_joint_names)
    joint_vel_indices = torch.tensor(
        [index for index, name in enumerate(control_joint_names) if "shoulder" in name], device=device, dtype=torch.long
    )
    joint_deviation_indices = torch.tensor(
        [index for index, name in enumerate(control_joint_names) if any(token in name for token in ("shoulder", "elbow", "hip"))],
        device=device,
        dtype=torch.long,
    )
    torso_name = str(
        env._env.config.robot.get(
            "isaacsim_torso_name",
            env._env.config.robot.get("torso_name", "base_link"),
        )
    )
    if torso_name not in simulator_body_names:
        raise ValueError(f"PiPlus simulator is missing locomotion reward torso body: {torso_name}")
    torso_index = simulator_body_names.index(torso_name)
    locomotion_reward_state = MimicLiteLocomotionRewardState(
        num_envs=args.num_envs,
        num_dof=len(control_joint_names),
        dt=env._env.dt,
        feet_indices=env._env.feet_indices,
        torso_index=torso_index,
        joint_vel_indices=joint_vel_indices,
        joint_deviation_indices=joint_deviation_indices,
        device=device,
    )

    metadata = {
        "robot_config": str(Path(args.robot_config).resolve()),
        "expert_dataset": str(Path(args.expert_dataset).resolve()),
        "bfm_checkpoint": str(Path(args.bfm_checkpoint).resolve()),
        "action_dim": bfm_action_dim,
        "policy_joint_names": list(robot_training.policy_joint_names),
        "fixed_joint_names": list(robot_training.fixed_joint_names),
        "simulator_control_joint_names": list(robot_training.robot.control_joint_names),
        "encoder_observation_layout": [
            {"name": "teacher_policy_observation_normalized", "dim": int(encoder_input.shape[-1])},
        ],
        "encoder_input_transform": {
            "version": ENCODER_INPUT_TRANSFORM_VERSION,
            "source": "chenjianhong_amp_teacher_policy_normalizer",
            "teacher_policy": teacher.artifact_path,
            "legacy_checkpoint_first_layer_migrated": legacy_input_migrated,
        },
        "teacher_policy": {
            "artifact": teacher.artifact_path,
            "input_dim": TEACHER_POLICY_INPUT_DIM,
            "action_dim": TEACHER_POLICY_ACTION_DIM,
            "joint_names": list(TEACHER_POLICY_JOINT_NAMES),
            "stage2_joint_names": list(stage2_joint_names),
            "action_target_order": "stage2_environment_order",
        },
        "z_dim": int(bfm_model.cfg.archi.z_dim),
        "backward_hidden_dim": int(backward_arch.hidden_dim),
        "backward_hidden_layers": int(backward_arch.hidden_layers),
        "command_encoder_hidden_dim": int(policy.hidden_dim),
        "command_encoder_hidden_layers": int(policy.hidden_layers),
        "expert_feature_dim": int(expert.feature_dim),
        "expert_motion_count": int(expert.motion_count),
        "command_range": {"low": commands_low.tolist(), "high": commands_high.tolist()},
        "command_stand_prob": args.command_stand_prob,
        "command_turn_prob": args.command_turn_prob,
        "command_resample_prob": args.command_resample_prob,
        "command_resample_interval": args.command_resample_steps,
        "command_warmup_steps": args.command_warmup_steps,
        "command_smoothing": args.command_smoothing,
        "amp_reward_mapping": "quad(discriminator_score) -> amp_weight",
        "amp_discriminator_objective": AMP_DISCRIMINATOR_OBJECTIVE,
        "amp_reward_weight": args.amp_weight,
        "action_imitation_coef": args.action_imitation_coef,
        "env_reward_weight": args.env_reward_weight,
        "locomotion_reward_weight": args.locomotion_reward_weight,
        "locomotion_reward_terms": MIMICLITE_LOCOMOTION_WEIGHTS,
        "locomotion_reward_scaling": "each weighted term is multiplied by the environment control dt",
        "stand_still_semantics": "HT_lab_pipeline-compatible zero-command negative L1 error from the default joint pose",
        "linvel_exp_error_scale": LINVEL_EXP_ERROR_SCALE,
        "linvel_exp_semantics": "signed improvement over the zero-speed baseline for nonzero planar commands",
        "standalone_tracking_reward": "removed; baseline-normalized linvel_exp and signed baseline-normalized angvel_z_exp provide command tracking",
        "termination": {
            "crash_contacts": list(env._env.config.robot.terminate_after_contacts_on),
            "fall_over_projected_gravity_xy_threshold": 0.9,
        },
        "survival_reward": "provided by the PiPlus environment reward; no extra stage2 alive term",
        "norm_z": bool(bfm_model.cfg.archi.norm_z),
        "latent_contract": latent_contract,
        "latent_stats": _latent_stats_to_json(latent_stats),
        "latent_reference_size": int(latent_reference.shape[0]),
        "latent_prior_weight": float(args.latent_prior_weight),
        "latent_prior_metric": "1 - max_cosine(projected_z, expert_projected_latent_bank)",
        "latent_diagnostic_samples": int(args.latent_diagnostic_samples),
        "distributed_rank": rank,
        "distributed_world_size": world_size,
        "global_num_envs": int(args.num_envs * world_size),
        "resume": str(Path(args.resume).resolve()) if args.resume else None,
        "resume_iteration": start_iteration,
        "policy_optimizer_resume": policy_optimizer_resume,
        "discriminator_optimizer_resume": discriminator_optimizer_resume,
        "discriminator_learning_rate": args.discriminator_learning_rate,
        "ppo": {
            "epochs": args.ppo_epochs,
            "minibatch_size": args.minibatch_size,
            "learning_rate": args.learning_rate,
            "target_kl": args.target_kl,
            "clip_ratio": 0.2,
            "value_coef": 0.5,
            "entropy_coef": args.entropy_coef,
        },
    }
    if rank0:
        (work_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    barrier()

    try:
        for iteration in range(start_iteration, args.iterations):
            encoder_features, commands_store, raw_z_store = [], [], []
            bfm_observations_store = {"state": [], "last_action": [], "history_actor": []}
            teacher_actions_store = []
            teacher_action_targets_store = []
            old_log_probs, values_store = [], []
            env_rewards, locomotion_rewards, terminated_store, truncated_store, crash_store, fall_store, timeout_values, amp_features = (
                [], [], [], [], [], [], [], []
            )
            env_reward_components_store: dict[str, list[torch.Tensor]] = {}
            locomotion_components_store = {name: [] for name in MIMICLITE_LOCOMOTION_WEIGHTS}
            base_lin_vel_store, base_ang_vel_store = [], []
            for step in range(args.rollout_steps):
                with torch.no_grad():
                    obs_t = _to_torch_obs(obs, device)
                    teacher_observation = build_teacher_policy_observation(
                        obs_t,
                        commands,
                        stage2_joint_names,
                        teacher_action_scales=teacher_action_scales,
                        bfm_action_scales=bfm_action_scales,
                    )
                    encoder_input = teacher.normalize(teacher_observation)
                    teacher_action_raw = teacher.net(encoder_input)
                    teacher_action = teacher_action_to_stage2(teacher_action_raw, stage2_joint_names)
                    teacher_action_target = teacher_action_target_to_stage2(
                        teacher_action_raw,
                        stage2_joint_names,
                        teacher_action_scales=teacher_action_scales,
                    )
                    raw_z, old_log_prob, value = policy.sample(encoder_input)
                    # Reuse the first-stage FBModel.project_z(); it reads norm_z from the checkpoint config.
                    z = bfm_model.project_z(raw_z)
                    action = _bfm_action(bfm_model, obs_t, z)
                next_obs, env_reward, terminated, truncated, info = env.step(action, to_numpy=False)
                terminated = terminated.to(device=device, dtype=torch.bool)
                truncated = truncated.to(device=device, dtype=torch.bool)
                done = torch.logical_or(terminated, truncated)
                transition_core = _transition_core(env._env, info)
                locomotion, locomotion_components = locomotion_reward_state.compute(transition_core, commands, action, done)
                base_lin_vel_store.append(transition_core.base_lin_vel.detach())
                base_ang_vel_store.append(transition_core.base_ang_vel.detach())
                online_history.append(transition_core.dof_pos)
                current_amp_feature = _online_amp_feature(transition_core, online_history, key_body_ids).detach()

                timeout_value = torch.zeros(args.num_envs, device=device)
                if torch.any(truncated):
                    terminal_obs = info.get("terminal_observation")
                    if terminal_obs is None:
                        raise RuntimeError("A truncated transition is missing its pre-reset terminal observation")
                    terminal_obs_t = _to_torch_obs(terminal_obs, device)
                    terminal_teacher_observation = build_teacher_policy_observation(
                        terminal_obs_t,
                        commands,
                        stage2_joint_names,
                        teacher_action_scales=teacher_action_scales,
                        bfm_action_scales=bfm_action_scales,
                    )
                    terminal_input = teacher.normalize(terminal_teacher_observation)
                    with torch.no_grad():
                        timeout_value[truncated] = policy(terminal_input)[2][truncated]
                if torch.any(done):
                    online_history.reset_envs(done.nonzero(as_tuple=False).squeeze(-1), env._env.simulator.dof_pos)

                encoder_features.append(encoder_input.detach())
                for key in bfm_observations_store:
                    bfm_observations_store[key].append(obs_t[key].detach())
                teacher_actions_store.append(teacher_action.detach())
                teacher_action_targets_store.append(teacher_action_target.detach())
                commands_store.append(commands.detach())
                raw_z_store.append(raw_z.detach())
                old_log_probs.append(old_log_prob.detach())
                values_store.append(value.detach())
                env_rewards.append(env_reward.to(device=device, dtype=torch.float32))
                for name, contribution in info["reward_components"].items():
                    env_reward_components_store.setdefault(name, []).append(contribution.detach())
                locomotion_rewards.append(locomotion.detach())
                for name, contribution in locomotion_components.items():
                    locomotion_components_store[name].append(contribution.detach())
                terminated_store.append(terminated)
                truncated_store.append(truncated)
                crash_store.append(info["termination_crash"].to(device=device, dtype=torch.bool))
                fall_store.append(info["termination_fall_over"].to(device=device, dtype=torch.bool))
                timeout_values.append(timeout_value)
                amp_features.append(current_amp_feature)
                obs = next_obs
                commands = commands + args.command_smoothing * (command_targets - commands)
                command_episode_steps = torch.where(
                    done,
                    torch.zeros_like(command_episode_steps),
                    command_episode_steps + 1,
                )
                if torch.any(done):
                    commands = commands.clone()
                    command_targets = command_targets.clone()
                    commands[done] = 0.0
                    command_targets[done] = 0.0
                resample = _mimiclite_resample_mask(
                    command_episode_steps,
                    commands,
                    resample_interval=args.command_resample_steps,
                    resample_prob=args.command_resample_prob,
                    warmup_steps=args.command_warmup_steps,
                )
                if torch.any(resample):
                    command_targets = command_targets.clone()
                    command_targets[resample] = _sample_commands(
                        int(resample.sum()),
                        device,
                        commands_low,
                        commands_high,
                        stand_prob=args.command_stand_prob,
                        turn_prob=args.command_turn_prob,
                    )

            rollout = Stage2Rollout(
                encoder_features=torch.stack(encoder_features),
                bfm_observations={key: torch.stack(values) for key, values in bfm_observations_store.items()},
                teacher_actions=torch.stack(teacher_actions_store),
                teacher_action_targets=torch.stack(teacher_action_targets_store),
                commands=torch.stack(commands_store),
                raw_z=torch.stack(raw_z_store),
                old_log_prob=torch.stack(old_log_probs),
                values=torch.stack(values_store),
                env_rewards=torch.stack(env_rewards),
                env_reward_components={name: torch.stack(values) for name, values in env_reward_components_store.items()},
                locomotion_rewards=torch.stack(locomotion_rewards),
                locomotion_components={name: torch.stack(values) for name, values in locomotion_components_store.items()},
                terminated=torch.stack(terminated_store),
                truncated=torch.stack(truncated_store),
                crash=torch.stack(crash_store),
                fall_over=torch.stack(fall_store),
                timeout_values=torch.stack(timeout_values),
                amp_features=torch.stack(amp_features),
                base_lin_vel=torch.stack(base_lin_vel_store),
                base_ang_vel=torch.stack(base_ang_vel_store),
            )
            with torch.no_grad():
                projected_rollout_z = bfm_model.project_z(
                    rollout.raw_z.reshape(-1, int(bfm_model.cfg.archi.z_dim))
                ).reshape(args.rollout_steps, args.num_envs, -1)
                latent_prior_penalty, latent_prior_cosine = _latent_direction_prior(
                    projected_rollout_z.reshape(-1, projected_rollout_z.shape[-1]),
                    latent_reference,
                )
                latent_prior_penalty = latent_prior_penalty.reshape(args.rollout_steps, args.num_envs)
                latent_prior_cosine = latent_prior_cosine.reshape(args.rollout_steps, args.num_envs)
            fake_features = rollout.amp_features.reshape(-1, expert.feature_dim)
            expert_features = expert.sample(fake_features.shape[0], device)
            discriminator_loss = discriminator.loss(expert_features, fake_features.detach())
            discriminator_optimizer.zero_grad(set_to_none=True)
            discriminator_loss["loss"].backward()
            average_gradients(discriminator.parameters())
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10.0)
            discriminator_optimizer.step()

            with torch.no_grad():
                amp_score = discriminator.reward(fake_features)
                amp_reward = amp_quadratic_reward(amp_score).reshape(args.rollout_steps, args.num_envs)
                rewards = (
                    args.env_reward_weight * rollout.env_rewards
                    + args.locomotion_reward_weight * rollout.locomotion_rewards
                    + args.amp_weight * amp_reward
                    - args.latent_prior_weight * latent_prior_penalty
                )
                next_obs_t = _to_torch_obs(obs, device)
                next_teacher_observation = build_teacher_policy_observation(
                    next_obs_t,
                    commands,
                    stage2_joint_names,
                    teacher_action_scales=teacher_action_scales,
                    bfm_action_scales=bfm_action_scales,
                )
                next_input = teacher.normalize(next_teacher_observation)
                next_value = policy(next_input)[2]
                advantages, returns = compute_gae(
                    rewards,
                    rollout.values,
                    rollout.terminated,
                    rollout.truncated,
                    rollout.timeout_values,
                    next_value,
                    discount=0.99,
                    gae_lambda=0.95,
                )
            ppo_metrics = ppo_update(
                policy,
                rollout,
                advantages,
                returns,
                epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                clip_ratio=0.2,
                value_coef=0.5,
                entropy_coef=args.entropy_coef,
                optimizer=policy_optimizer,
                max_grad_norm=1.0,
                target_kl=args.target_kl,
                bfm_model=bfm_model,
                action_imitation_coef=args.action_imitation_coef,
                bfm_action_position_scales=bfm_action_scales,
            )
            tracking_metrics = command_tracking_metrics(rollout.commands, rollout.base_lin_vel, rollout.base_ang_vel)
            latent_metrics = latent_manifold_metrics(
                projected_rollout_z,
                rollout.commands,
                latent_reference,
                max_samples=args.latent_diagnostic_samples,
            )
            metrics = {
                "iteration": iteration,
                "reward": float(rewards.mean()),
                "env_reward_contribution": float((args.env_reward_weight * rollout.env_rewards).mean()),
                "mimiclite_locomotion_reward_contribution": float(
                    (args.locomotion_reward_weight * rollout.locomotion_rewards).mean()
                ),
                "amp_score": float(amp_score.mean()),
                "amp_reward": float(amp_reward.mean()),
                "amp_reward_contribution": float((args.amp_weight * amp_reward).mean()),
                "latent_prior_penalty": float(latent_prior_penalty.mean()),
                "latent_prior_cosine": float(latent_prior_cosine.mean()),
                "latent_prior_contribution": float((-args.latent_prior_weight * latent_prior_penalty).mean()),
                "termination_rate": float(rollout.terminated.float().mean()),
                "termination_crash_rate": float(rollout.crash.float().mean()),
                "termination_fall_over_rate": float(rollout.fall_over.float().mean()),
                "truncation_rate": float(rollout.truncated.float().mean()),
                "discriminator_loss": float(discriminator_loss["loss"].detach()),
                "discriminator_expert_score": float(discriminator_loss["expert_score"].detach()),
                "discriminator_policy_score": float(discriminator_loss["policy_score"].detach()),
                "discriminator_score_gap": float(
                    (discriminator_loss["expert_score"] - discriminator_loss["policy_score"]).detach()
                ),
                "discriminator_gradient_penalty": float(discriminator_loss["gradient_penalty"].detach()),
                "discriminator_gradient_norm": float(discriminator_loss["gradient_norm"].detach()),
                **ppo_metrics,
                **tracking_metrics,
                **latent_metrics,
            }
            metrics.update(
                {
                    f"reward_contribution/mimiclite/{name}": float(
                        (args.locomotion_reward_weight * contribution).mean()
                    )
                    for name, contribution in rollout.locomotion_components.items()
                }
            )
            metrics.update(
                {
                    f"reward_contribution/env/{name}": float((args.env_reward_weight * contribution).mean())
                    for name, contribution in rollout.env_reward_components.items()
                }
            )
            if world_size > 1:
                import torch.distributed as dist

                for key, value in list(metrics.items()):
                    if isinstance(value, (int, float)):
                        metric = torch.tensor(float(value), device=device)
                        dist.all_reduce(metric, op=dist.ReduceOp.SUM)
                        metrics[key] = float((metric / world_size).item())
            if rank0:
                print(json.dumps(metrics, sort_keys=True), flush=True)
            if rank0 and ((iteration + 1) % args.save_every == 0 or iteration + 1 == args.iterations):
                torch.save(
                    {
                        "policy": policy.state_dict(),
                        "policy_optimizer": policy_optimizer.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "discriminator_optimizer": discriminator_optimizer.state_dict(),
                        "amp_reward_normalizer": reward_normalizer.state_dict(),
                        "iteration": iteration + 1,
                        "metadata": metadata,
                    },
                    work_dir / f"checkpoint_{iteration + 1}.pt",
                )
            barrier()
    finally:
        env.close()
        if world_size > 1:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()


def launch() -> None:
    args = _parse_args()
    # A process started by torchrun is already a worker. Do not recursively
    # create another process group from inside that worker.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        main(args)
        return
    if args.gpu_ids in (None, "single"):
        main(args)
        return

    existing_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.gpu_ids == "all":
        num_gpus = torch.cuda.device_count()
        selected_gpus = None
    else:
        requested = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
        if existing_visible:
            visible = [value.strip() for value in existing_visible.split(",") if value.strip()]
            selected_gpus = [visible[index] for index in requested]
        else:
            selected_gpus = [str(index) for index in requested]
        num_gpus = len(selected_gpus)
    if num_gpus <= 1:
        if selected_gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
        main(args)
        return

    if selected_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)

    command = _torchrun_command(num_gpus, sys.argv[1:])
    print(f"Launching PiPlus AMP Stage2 with {num_gpus} torchrun workers", flush=True)
    subprocess.run(command, check=True, env=os.environ.copy())


def _torchrun_command(num_gpus: int, argv: list[str]) -> list[str]:
    """Build the dependency-free local torchrun command used by ``launch``."""
    if int(num_gpus) < 2:
        raise ValueError(f"torchrun requires at least two workers, got {num_gpus}")
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        "--module",
        "humanoidverse.amp_stage2",
        *argv,
    ]


if __name__ == "__main__":
    launch()
