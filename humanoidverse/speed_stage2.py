"""Train a velocity-command encoder on a frozen 22DoF PiPlus BFM decoder.

This is intentionally separate from :mod:`humanoidverse.amp_stage2`: it has
no AMP discriminator, expert-style reward, or 23DoF teacher dependency.  The
only trainable network is a stochastic command encoder with a PPO value head.
It produces BFM latents consumed by a frozen decoder.

The supplied H0W artifact is a bare ``model.safetensors`` state dict.  It
cannot be reconstructed without its matching decoder configuration, so this
module accepts a small decoder factory protocol.  The factory must return an
object exposing ``action_dim``, ``project_z(z)``, and
``act(observation, z, mean=True)``.  ``--validate-assets`` verifies every
local contract without requiring that decoder.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.utils.asset_paths import resolve_asset_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BFM_MODEL = PROJECT_ROOT / "model/piplus_h0w_bfm/model.safetensors"
DEFAULT_EXPERT_DATASET = PROJECT_ROOT / "humanoidverse/data/piplus_h0w_lafan/piplus_h0w_lafan_10s-clipped.pkl"
DEFAULT_ROBOT_CONFIG = PROJECT_ROOT / "humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H0W.yaml"

COMMAND_SCALE = (1.25, 5.0, 1.25)
DOF_VEL_SCALE = 0.05
COMMAND_LOW = (-0.8, -0.5, -0.8)
COMMAND_HIGH = (0.8, 0.5, 0.8)


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


class CommandEncoderPolicy(nn.Module):
    """Stochastic velocity-command-to-latent policy with a PPO critic."""

    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int = 256, hidden_layers: int = 1) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        self.input_dim = int(input_dim)
        self.z_dim = int(z_dim)
        layers: list[nn.Module] = [nn.Linear(self.input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.trunk = nn.Sequential(*layers)
        self.latent_mean = nn.Linear(hidden_dim, self.z_dim)
        self.latent_log_std = nn.Parameter(torch.full((self.z_dim,), -1.5))
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        mean = self.latent_mean(hidden)
        log_std = self.latent_log_std.clamp(-5.0, 1.0).expand_as(mean)
        return mean, log_std, self.value_head(hidden).squeeze(-1)

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(features)
        distribution = Normal(mean, log_std.exp())
        raw_z = distribution.rsample()
        return raw_z, distribution.log_prob(raw_z).sum(dim=-1), value

    def distribution(self, features: torch.Tensor) -> Normal:
        mean, log_std, _ = self(features)
        return Normal(mean, log_std.exp())


@dataclass
class SpeedRollout:
    features: torch.Tensor
    raw_z: torch.Tensor
    old_log_prob: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    commands: torch.Tensor
    base_lin_vel: torch.Tensor
    base_ang_vel: torch.Tensor


def _load_robot_contract(robot_config: str | Path) -> SimpleNamespace:
    from omegaconf import OmegaConf

    config_path = Path(robot_config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Robot config does not exist: {config_path}")
    config = OmegaConf.load(config_path)
    robot = config.robot
    joint_names = tuple(str(name) for name in robot.dof_names)
    xml_path = resolve_asset_path(robot.asset.asset_root, robot.asset.xml_file)
    if len(joint_names) != 22 or int(robot.actions_dim) != 22:
        raise ValueError(f"PiPlus H0W requires 22 DoF, got {len(joint_names)} names and {robot.actions_dim} actions")
    if not xml_path.is_file():
        raise FileNotFoundError(f"PiPlus H0W MJCF does not exist: {xml_path}")
    return SimpleNamespace(config_path=config_path, joint_names=joint_names, xml_path=xml_path.resolve())


def validate_h0w_assets(robot_config: str | Path, bfm_model: str | Path) -> dict[str, Any]:
    """Validate local robot files and the static 22DoF BFM state-dict contract."""
    contract = _load_robot_contract(robot_config)
    model_path = Path(bfm_model).expanduser().absolute()
    if not model_path.is_file():
        raise FileNotFoundError(f"BFM model does not exist: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(contract.xml_path))
    if model.nu != 22:
        raise ValueError(f"H0W MJCF must expose 22 actuators, got {model.nu}")
    actuator_joints = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuator_joints.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
    if tuple(actuator_joints) != contract.joint_names:
        raise ValueError("H0W MJCF actuator order does not match robot.dof_names")

    from safetensors import safe_open

    with safe_open(str(model_path), framework="pt", device="cpu") as artifact:
        keys = list(artifact.keys())
        output_key = "_actor.policy.6.mlp.1.weight"
        if output_key not in keys:
            raise ValueError(f"Expected GCR actor output tensor {output_key!r} in {model_path}")
        output_shape = tuple(artifact.get_tensor(output_key).shape)
        if output_shape[0] != 22:
            raise ValueError(f"BFM actor must produce 22 actions, got tensor shape {output_shape}")
        if not any(key.startswith("_goal_encoder.") for key in keys):
            raise ValueError("BFM state dict does not contain a goal/latent encoder")

    return {
        "robot_config": str(contract.config_path),
        "xml_path": str(contract.xml_path),
        "action_dim": len(contract.joint_names),
        "mjcf_nq": int(model.nq),
        "mjcf_nv": int(model.nv),
        "mjcf_nu": int(model.nu),
        "bfm_model": str(model_path),
        "bfm_format": "gcr_rl_safetensors",
        "bfm_actor_output_shape": list(output_shape),
    }


def _resolve_decoder_factory(spec: str) -> Callable[[Path, torch.device], Any]:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--decoder-factory must use the form package.module:callable")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"Decoder factory {spec!r} is not callable")
    return factory


def _decoder_z_dim(decoder: Any) -> int:
    for candidate in (getattr(decoder, "z_dim", None), getattr(getattr(getattr(decoder, "cfg", None), "archi", None), "z_dim", None)):
        if candidate is not None:
            return int(candidate)
    raise AttributeError("Decoder must expose z_dim or cfg.archi.z_dim")


def _load_decoder(model_path: Path, factory_spec: str | None, device: torch.device) -> Any:
    if factory_spec is None:
        raise RuntimeError(
            "The H0W BFM artifact is a bare model.safetensors file. Supply its matching decoder through "
            "--decoder-factory package.module:callable. Run --validate-assets before the decoder is available."
        )
    decoder = _resolve_decoder_factory(factory_spec)(model_path, device)
    try:
        action_dim = int(decoder.action_dim)
        project_z = decoder.project_z
        act = decoder.act
    except AttributeError as exc:
        raise TypeError("Decoder must expose action_dim, project_z(z), and act(observation, z, mean=True)") from exc
    if action_dim != 22:
        raise ValueError(f"Decoder action_dim must be 22, got {action_dim}")
    if _decoder_z_dim(decoder) <= 0 or not callable(project_z):
        raise ValueError("Decoder must expose a positive latent dimension and project_z(z)")
    if not callable(act):
        raise TypeError("Decoder act must be callable")
    if isinstance(decoder, nn.Module):
        decoder.eval()
        decoder.requires_grad_(False)
    return decoder


def _to_torch_obs(obs: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in obs.items() if key != "time"}


def _encoder_input_scale(obs: Mapping[str, torch.Tensor], commands: torch.Tensor) -> torch.Tensor:
    state = obs["state"]
    last_action = obs.get("last_action", torch.zeros_like(state[..., :0]))
    action_dim = int(last_action.shape[-1])
    if action_dim != 22 or state.shape[-1] != 2 * action_dim + 6:
        raise ValueError(f"Expected H0W state [2 * 22 + 6] and last_action [22], got {state.shape}/{last_action.shape}")
    state_scale = torch.ones(state.shape[-1], device=state.device, dtype=state.dtype)
    state_scale[action_dim : 2 * action_dim] = DOF_VEL_SCALE
    pieces = [commands.new_tensor(COMMAND_SCALE), state_scale, torch.ones_like(last_action[0])]
    history = obs.get("history_actor")
    if history is not None:
        per_frame = 3 * action_dim + 6
        if history.shape[-1] % per_frame:
            raise ValueError(f"Cannot infer H0W history layout from dim {history.shape[-1]}")
        history_length = history.shape[-1] // per_frame
        history_scale = torch.ones(history.shape[-1], device=history.device, dtype=history.dtype)
        velocity_start = history_length * (2 * action_dim + 3)
        history_scale[velocity_start : velocity_start + history_length * action_dim] = DOF_VEL_SCALE
        pieces.append(history_scale)
    return torch.cat(pieces)


def flatten_encoder_observation(obs: Mapping[str, torch.Tensor], commands: torch.Tensor) -> torch.Tensor:
    pieces = [commands, obs["state"], obs.get("last_action", torch.zeros_like(obs["state"][..., :0]))]
    if "history_actor" in obs:
        pieces.append(obs["history_actor"])
    return torch.cat(pieces, dim=-1) * _encoder_input_scale(obs, commands)


def _sample_commands(num_envs: int, device: torch.device, stand_probability: float, turn_probability: float) -> torch.Tensor:
    low = torch.tensor(COMMAND_LOW, device=device)
    high = torch.tensor(COMMAND_HIGH, device=device)
    commands = low + torch.rand(num_envs, 3, device=device) * (high - low)
    turning = torch.rand(num_envs, device=device) < turn_probability
    commands[turning, :2] = 0.0
    standing = (torch.rand(num_envs, device=device) < stand_probability) | ((commands[:, :2].norm(dim=-1) < 0.1) & ~turning)
    commands[standing] = 0.0
    return commands


def speed_tracking_reward(
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    commands: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    linear_error = (base_lin_vel[:, :2] - commands[:, :2]).square().sum(dim=-1)
    yaw_error = (base_ang_vel[:, 2] - commands[:, 2]).square()
    linear = torch.exp(-linear_error / 0.16)
    yaw = torch.exp(-yaw_error / 0.25)
    return linear + 0.5 * yaw, {"linear_velocity": linear, "yaw_velocity": yaw}


def _transition_core(live_core, info: Mapping[str, Any]):
    terminal_state = info.get("terminal_state")
    if isinstance(terminal_state, Mapping):
        return SimpleNamespace(num_envs=live_core.num_envs, device=live_core.device, **terminal_state)
    return live_core


def _stage2_update_reset_buf(core) -> None:
    if core.termination_contact_indices.numel() == 0:
        crash = torch.zeros(core.num_envs, dtype=torch.bool, device=core.device)
    else:
        contact_norm = core.simulator.contact_forces[:, core.termination_contact_indices].norm(dim=-1)
        crash = torch.any(contact_norm > 1.0, dim=-1)
    fall_over = core.projected_gravity[:, :2].norm(dim=-1) >= 0.9
    core.reset_buf |= crash | fall_over


def _stage2_prepare_pre_reset_transition(core, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
        return
    simulator = core.simulator
    core.extras["terminal_state"] = {
        "base_lin_vel": core.base_lin_vel.detach().clone(),
        "base_ang_vel": core.base_ang_vel.detach().clone(),
        "projected_gravity": core.projected_gravity.detach().clone(),
        "contact_forces": simulator.contact_forces.detach().clone(),
    }


def build_h0w_locomotion_env(
    *,
    device: str,
    expert_dataset: str,
    num_envs: int,
    seed: int,
    max_episode_length_s: float,
    simulator: str,
    disable_domain_randomization: bool,
):
    if simulator not in {"isaacsim", "mujoco"}:
        raise ValueError(f"Unsupported simulator {simulator!r}")
    if not Path(expert_dataset).expanduser().is_file():
        raise FileNotFoundError(f"H0W motion dataset does not exist: {expert_dataset}")
    overrides = [
        "robot=piplus/PiPlus_S_12L8A0G2H0W",
        f"simulator={simulator}",
        f"num_envs={num_envs}",
        f"simulator.config.scene.num_envs={num_envs}",
        "env.config.resample_motion_when_training=False",
        "env.config.termination.terminate_when_motion_end=False",
        "env.config.termination.terminate_when_motion_far=False",
        "env.config.termination.terminate_by_contact=False",
        "env.config.termination.terminate_by_gravity=False",
        "env.config.termination.terminate_by_low_height=False",
        "env.config.lie_down_init=False",
    ]
    config = HumanoidVerseIsaacConfig(
        name="humanoidverse_isaac",
        device=device,
        lafan_tail_path=str(Path(expert_dataset).expanduser().resolve()),
        max_episode_length_s=max_episode_length_s,
        disable_obs_noise=False,
        disable_domain_randomization=disable_domain_randomization,
        relative_config_path="exp/bfm_zero_piplus/bfm_zero_piplus",
        include_last_action=True,
        include_history_actor=True,
        root_height_obs=True,
        hydra_overrides=overrides,
    )
    torch.manual_seed(seed)
    env = config.build(num_envs=num_envs)[0]
    core = env._env
    core._reset_tasks_callback = MethodType(LeggedRobotBase._reset_tasks_callback, core)
    core._reset_dofs = MethodType(LeggedRobotBase._reset_dofs, core)
    core._reset_root_states = MethodType(LeggedRobotBase._reset_root_states, core)
    core._update_reset_buf = MethodType(_stage2_update_reset_buf, core)
    core._prepare_pre_reset_transition = MethodType(_stage2_prepare_pre_reset_transition, core)
    return env


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    last_value: torch.Tensor,
    discount: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    running_advantage = torch.zeros_like(last_value)
    next_value = last_value
    for step in reversed(range(rewards.shape[0])):
        done = terminated[step] | truncated[step]
        not_done = (~done).float()
        delta = rewards[step] + discount * next_value * not_done - values[step]
        running_advantage = delta + discount * gae_lambda * not_done * running_advantage
        advantages[step] = running_advantage
        next_value = values[step]
    return advantages, advantages + values


def ppo_update(
    policy: CommandEncoderPolicy,
    rollout: SpeedRollout,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    minibatch_size: int,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> dict[str, float]:
    features = rollout.features.flatten(0, 1)
    raw_z = rollout.raw_z.flatten(0, 1)
    old_log_prob = rollout.old_log_prob.flatten()
    advantages = advantages.flatten()
    returns = returns.flatten()
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1.0e-6)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    updates = 0
    for _ in range(epochs):
        for indices in torch.randperm(features.shape[0], device=features.device).split(minibatch_size):
            distribution = policy.distribution(features[indices])
            log_prob = distribution.log_prob(raw_z[indices]).sum(dim=-1)
            _, _, value = policy(features[indices])
            ratio = (log_prob - old_log_prob[indices]).clamp(-20.0, 20.0).exp()
            policy_loss = -torch.minimum(
                ratio * advantages[indices], ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[indices]
            ).mean()
            value_loss = 0.5 * (value - returns[indices]).square().mean()
            entropy = distribution.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            (policy_loss + value_coef * value_loss - entropy_coef * entropy).backward()
            average_gradients(policy.parameters())
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            totals["policy_loss"] += float(policy_loss.detach())
            totals["value_loss"] += float(value_loss.detach())
            totals["entropy"] += float(entropy.detach())
            totals["approx_kl"] += float((old_log_prob[indices] - log_prob).mean().detach())
            updates += 1
    return {name: value / max(updates, 1) for name, value in totals.items()} | {"ppo_updates": float(updates)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfm-model", default=str(DEFAULT_BFM_MODEL), help="Local H0W model.safetensors path.")
    parser.add_argument("--decoder-factory", default=None, help="package.module:callable that rebuilds and loads the frozen decoder.")
    parser.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--expert-dataset", default=str(DEFAULT_EXPERT_DATASET))
    parser.add_argument("--validate-assets", action="store_true", help="Validate local H0W asset and BFM state-dict contracts then exit.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--simulator", choices=("isaacsim", "mujoco"), default="isaacsim")
    parser.add_argument("--work-dir", default="runs/speed_stage2_piplus_h0w")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--command-stand-prob", type=float, default=0.15)
    parser.add_argument("--command-turn-prob", type=float, default=0.15)
    parser.add_argument("--command-resample-steps", type=int, default=300)
    parser.add_argument("--command-resample-prob", type=float, default=0.75)
    parser.add_argument("--command-smoothing", type=float, default=0.15)
    parser.add_argument("--env-reward-weight", type=float, default=0.0)
    parser.add_argument("--max-episode-length-s", type=float, default=20.0)
    parser.add_argument("--disable-domain-randomization", action="store_true")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _distributed_context(args: argparse.Namespace) -> tuple[argparse.Namespace, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size <= 1:
        return args, rank, world_size
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed Stage2 training requires CUDA")
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    args.device = f"cuda:{local_rank}"
    args.seed += rank
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    return args, rank, world_size


def main(parsed_args: argparse.Namespace | None = None) -> None:
    args = _parse_args() if parsed_args is None else parsed_args
    args, rank, world_size = _distributed_context(args)
    contract = validate_h0w_assets(args.robot_config, args.bfm_model)
    if args.validate_assets:
        print(json.dumps(contract, indent=2, sort_keys=True), flush=True)
        return
    if not 0.0 <= args.command_smoothing <= 1.0:
        raise ValueError("--command-smoothing must be in [0, 1]")
    if args.command_resample_steps <= 0:
        raise ValueError("--command-resample-steps must be positive")
    if not 0.0 <= args.command_resample_prob <= 1.0:
        raise ValueError("--command-resample-prob must be in [0, 1]")

    device = torch.device(args.device)
    decoder = _load_decoder(Path(args.bfm_model).expanduser().absolute(), args.decoder_factory, device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = build_h0w_locomotion_env(
        device=args.device,
        expert_dataset=args.expert_dataset,
        num_envs=args.num_envs,
        seed=args.seed,
        max_episode_length_s=args.max_episode_length_s,
        simulator=args.simulator,
        disable_domain_randomization=args.disable_domain_randomization,
    )
    try:
        obs, _ = env.reset(to_numpy=False)
        obs_t = _to_torch_obs(obs, device)
        commands = _sample_commands(args.num_envs, device, args.command_stand_prob, args.command_turn_prob)
        features = flatten_encoder_observation(obs_t, commands)
        policy = CommandEncoderPolicy(features.shape[-1], _decoder_z_dim(decoder)).to(device)
        broadcast_module_state(policy)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
        work_dir = Path(args.work_dir)
        metadata = {
            "task": "speed_stage2_piplus_h0w",
            "reward": "velocity_command_only_plus_optional_environment_reward",
            "amp": False,
            "bfm_model": contract["bfm_model"],
            "decoder_factory": args.decoder_factory,
            "robot_contract": contract,
            "z_dim": policy.z_dim,
            "encoder_input_dim": policy.input_dim,
        }
        if rank == 0:
            work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        start_iteration = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            policy.load_state_dict(checkpoint["policy"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_iteration = int(checkpoint["iteration"])

        episode_steps = torch.zeros(args.num_envs, dtype=torch.long, device=device)
        for iteration in range(start_iteration, args.iterations):
            rollout_parts: list[dict[str, torch.Tensor]] = []
            for _ in range(args.rollout_steps):
                features = flatten_encoder_observation(obs_t, commands)
                with torch.no_grad():
                    raw_z, log_prob, value = policy.sample(features)
                    action = decoder.act(obs_t, decoder.project_z(raw_z), mean=True)
                next_obs, environment_reward, terminated, truncated, info = env.step(action, to_numpy=False)
                core = _transition_core(env._env, info)
                speed_reward, _ = speed_tracking_reward(core.base_lin_vel, core.base_ang_vel, commands)
                reward = speed_reward + args.env_reward_weight * environment_reward
                rollout_parts.append(
                    {
                        "features": features,
                        "raw_z": raw_z,
                        "old_log_prob": log_prob,
                        "values": value,
                        "rewards": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "commands": commands,
                        "base_lin_vel": core.base_lin_vel.detach().clone(),
                        "base_ang_vel": core.base_ang_vel.detach().clone(),
                    }
                )
                obs_t = _to_torch_obs(next_obs, device)
                done = terminated | truncated
                episode_steps += 1
                episode_steps[done] = 0
                resample = (episode_steps > 0) & (episode_steps % args.command_resample_steps == 0)
                resample &= torch.rand(args.num_envs, device=device) < args.command_resample_prob
                resample |= done
                targets = _sample_commands(args.num_envs, device, args.command_stand_prob, args.command_turn_prob)
                commands = torch.where(
                    resample.unsqueeze(-1),
                    (1.0 - args.command_smoothing) * commands + args.command_smoothing * targets,
                    commands,
                )

            stacked = {name: torch.stack([part[name] for part in rollout_parts]) for name in rollout_parts[0]}
            with torch.no_grad():
                _, _, last_value = policy(flatten_encoder_observation(obs_t, commands))
            rollout = SpeedRollout(**stacked)
            advantages, returns = compute_gae(
                rollout.rewards, rollout.values, rollout.terminated, rollout.truncated, last_value, args.discount, args.gae_lambda
            )
            metrics = ppo_update(
                policy,
                rollout,
                advantages,
                returns,
                optimizer,
                epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                clip_ratio=0.2,
                value_coef=0.5,
                entropy_coef=0.001,
                max_grad_norm=1.0,
            )
            metrics.update(
                {
                    "iteration": float(iteration + 1),
                    "reward_mean": float(rollout.rewards.mean()),
                    "speed/vx_mae": float((rollout.base_lin_vel[..., 0] - rollout.commands[..., 0]).abs().mean()),
                    "speed/vy_mae": float((rollout.base_lin_vel[..., 1] - rollout.commands[..., 1]).abs().mean()),
                    "speed/yaw_rate_mae": float((rollout.base_ang_vel[..., 2] - rollout.commands[..., 2]).abs().mean()),
                    "termination_rate": float(rollout.terminated.float().mean()),
                }
            )
            if world_size > 1:
                for name, value in metrics.items():
                    reduced = torch.tensor(value, device=device)
                    torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
                    metrics[name] = float(reduced / world_size)
            if rank == 0:
                print(json.dumps(metrics, sort_keys=True), flush=True)
                if (iteration + 1) % args.save_every == 0 or iteration + 1 == args.iterations:
                    torch.save(
                        {
                            "policy": policy.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "iteration": iteration + 1,
                            "metadata": metadata,
                        },
                        work_dir / f"checkpoint_{iteration + 1}.pt",
                    )
            barrier()
    finally:
        env.close()
        if _distributed_ready():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
