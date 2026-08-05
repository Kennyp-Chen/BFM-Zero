"""Play a 22DoF PiPlus speed-stage2 command encoder in a MuJoCo viewer."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from humanoidverse.piplus_h0w_onnx_decoder import OnnxPiPlusH0WDecoder
from humanoidverse.speed_stage2 import (
    COMMAND_HIGH,
    COMMAND_LOW,
    DEFAULT_DECODER_PATH,
    DEFAULT_EXPERT_DATASET,
    DEFAULT_ROBOT_CONFIG,
    DEFAULT_WORK_DIR,
    CommandEncoderPolicy,
    _load_robot_contract,
    _to_torch_obs,
    build_h0w_locomotion_env,
    flatten_encoder_observation,
)


def latest_speed_checkpoint(model_folder: Path) -> Path:
    """Return the numerically newest speed-stage2 checkpoint in a directory."""
    candidates: list[tuple[int, Path]] = []
    for path in model_folder.glob("checkpoint_*.pt"):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_<iteration>.pt found in {model_folder}")
    return max(candidates, key=lambda item: item[0])[1]


def policy_shape_from_state(policy_state: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """Recover the command encoder shape from a saved state dict."""
    try:
        input_dim = int(policy_state["trunk.0.weight"].shape[1])
        hidden_dim = int(policy_state["trunk.0.weight"].shape[0])
        z_dim = int(policy_state["latent_mean.weight"].shape[0])
    except KeyError as exc:
        raise ValueError("Checkpoint does not contain a speed-stage2 command encoder") from exc
    return input_dim, hidden_dim, z_dim


class PassivePolicyViewer:
    """A lightweight viewer independent from the training MuJoCo instance."""

    def __init__(self, xml_path: Path, expected_qpos_size: int) -> None:
        import mujoco.viewer

        spec = mujoco.MjSpec.from_file(str(xml_path))
        spec.worldbody.add_geom(
            name="speed_stage2_play_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0.0, 0.0, 0.0],
            size=[20.0, 20.0, 0.02],
            rgba=[0.45, 0.47, 0.50, 1.0],
            contype=0,
            conaffinity=0,
        )
        spec.worldbody.add_light(
            name="speed_stage2_play_light",
            pos=[0.0, -3.0, 4.0],
            dir=[0.2, 0.5, -1.0],
            diffuse=[0.8, 0.8, 0.8],
            ambient=[0.35, 0.35, 0.35],
        )
        self.model = spec.compile()
        if self.model.nq != expected_qpos_size:
            raise ValueError(f"Viewer qpos size {self.model.nq} does not match environment qpos size {expected_qpos_size}")
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 3.0
        self.viewer.cam.azimuth = 135.0
        self.viewer.cam.elevation = -18.0

    def update(self, qpos: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.viewer.cam.lookat[:] = [float(qpos[0]), float(qpos[1]), max(float(qpos[2]), 0.75)]
        self.viewer.sync()

    def is_running(self) -> bool:
        return self.viewer.is_running()

    def close(self) -> None:
        self.viewer.close()


def resolve_checkpoint(model_folder: Path, checkpoint: Path | None) -> Path:
    path = checkpoint.expanduser().resolve() if checkpoint else latest_speed_checkpoint(model_folder)
    if not path.is_file():
        raise FileNotFoundError(f"Missing speed-stage2 checkpoint: {path}")
    return path


def play(args: argparse.Namespace) -> None:
    if args.simulator != "mujoco":
        raise ValueError("speed-stage2 playback currently supports only MuJoCo")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if not 0.0 < args.command_smoothing <= 1.0:
        raise ValueError("--command-smoothing must be in (0, 1]")

    model_folder = args.model_folder.expanduser().resolve()
    checkpoint_path = resolve_checkpoint(model_folder, args.checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("metadata", {}).get("task") not in {None, "speed_stage2_piplus_h0w", "speed_stage2_piplus_22dof"}:
        raise ValueError("Checkpoint is not a PiPlus 22DoF speed-stage2 checkpoint")
    policy_state = checkpoint.get("policy")
    if not isinstance(policy_state, dict):
        raise TypeError("Checkpoint policy state must be a mapping")
    input_dim, hidden_dim, z_dim = policy_shape_from_state(policy_state)

    env = build_h0w_locomotion_env(
        device=str(device),
        expert_dataset=str(args.expert_dataset.expanduser().resolve()),
        num_envs=1,
        seed=args.seed,
        max_episode_length_s=args.max_episode_length_s,
        simulator=args.simulator,
        disable_domain_randomization=True,
    )
    viewer = None
    try:
        observation, _ = env.reset(to_numpy=False, reset_to_default_pose=True)
        observation_t = _to_torch_obs(observation, device)
        expected_input_dim = int(flatten_encoder_observation(observation_t, torch.zeros(1, 3, device=device)).shape[-1])
        if input_dim != expected_input_dim:
            raise ValueError(f"Checkpoint expects encoder input dim {input_dim}, environment provides {expected_input_dim}")
        policy = CommandEncoderPolicy(input_dim, z_dim, hidden_dim=hidden_dim).to(device)
        policy.load_state_dict(policy_state)
        policy.eval()
        decoder = OnnxPiPlusH0WDecoder(args.decoder_path, device)
        if decoder.z_dim != z_dim:
            raise ValueError(f"Checkpoint z_dim={z_dim} does not match decoder z_dim={decoder.z_dim}")

        command = torch.tensor(args.fixed_command, dtype=torch.float32, device=device).unsqueeze(0)
        command_low = torch.tensor(COMMAND_LOW, device=device)
        command_high = torch.tensor(COMMAND_HIGH, device=device)
        if torch.any(command < command_low) or torch.any(command > command_high):
            raise ValueError(f"--fixed-command must be within {COMMAND_LOW} to {COMMAND_HIGH}")
        commands = torch.zeros_like(command)
        qpos, _ = env._get_qpos_qvel(to_numpy=True)
        robot_contract = _load_robot_contract(args.robot_config)
        viewer = PassivePolicyViewer(robot_contract.xml_path, int(qpos.shape[-1]))
        print(
            f"[INFO] checkpoint={checkpoint_path} iteration={checkpoint.get('iteration', 'unknown')} "
            f"command={args.fixed_command} device={device}",
            flush=True,
        )

        step = 0
        while viewer.is_running() and (args.max_steps is None or step < args.max_steps):
            started_at = time.monotonic()
            commands.add_(args.command_smoothing * (command - commands))
            with torch.inference_mode():
                observation_t = _to_torch_obs(observation, device)
                features = flatten_encoder_observation(observation_t, commands)
                raw_z, _log_std, _value = policy(features)
                action = decoder.act(observation_t, decoder.project_z(raw_z))
            observation, _reward, terminated, truncated, _info = env.step(action, to_numpy=False)
            qpos, _ = env._get_qpos_qvel(to_numpy=True)
            viewer.update(qpos[0])
            step += 1
            if step == 1 or (args.log_every_steps > 0 and step % args.log_every_steps == 0):
                velocity = env._env.base_lin_vel[0].detach().cpu().numpy()
                yaw_velocity = float(env._env.base_ang_vel[0, 2].detach().cpu())
                print(
                    f"[INFO] step={step} command={commands[0].detach().cpu().numpy().round(3).tolist()} "
                    f"velocity={[round(float(velocity[0]), 3), round(float(velocity[1]), 3), round(yaw_velocity, 3)]} "
                    f"terminated={bool(terminated.any())} truncated={bool(truncated.any())}",
                    flush=True,
                )
            if args.realtime:
                time.sleep(max(0.0, float(env._env.dt) - (time.monotonic() - started_at)))
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to the newest checkpoint_<iteration>.pt in model-folder.")
    parser.add_argument("--decoder-path", type=Path, default=DEFAULT_DECODER_PATH)
    parser.add_argument("--expert-dataset", type=Path, default=DEFAULT_EXPERT_DATASET)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--simulator", choices=("mujoco",), default="mujoco")
    parser.add_argument("--device", default="cpu", help="Use cpu for portable local playback, or cuda:0 when available.")
    parser.add_argument("--fixed-command", type=float, nargs=3, default=(0.4, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--command-smoothing", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-episode-length-s", type=float, default=20.0)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    play(parse_args())


if __name__ == "__main__":
    main()
