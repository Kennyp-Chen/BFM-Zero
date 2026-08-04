"""Interactive joystick playback for a PiPlus Stage2 AMP command encoder."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mediapy as media
import mujoco
import numpy as np
import torch

try:
    import pygame
except ImportError:  # pragma: no cover - exercised only when --gamepad is requested
    pygame = None

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.amp_stage2 import (
    DEFAULT_TEACHER_POLICY,
    TEACHER_POLICY_ACTION_SCALES,
    CommandEncoderPolicy,
    PiPlusAMPTeacherPolicy,
    _bfm_action,
    _ensure_runtime_cache,
    _to_torch_obs,
    build_piplus_locomotion_env,
    build_teacher_policy_observation,
    encoder_input_scale,
    flatten_encoder_observation,
    load_command_encoder_policy_state,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_FOLDER = (
    PROJECT_ROOT
    / "runs/amp_stage2_piplus_lse"
)
DEFAULT_BFM_RUN = "bfmzero-piplus-lse-isaac-20260715_143758(1)"
JOYSTICK_EVENT = struct.Struct("<IhBB")
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80


def latest_stage2_checkpoint(model_folder: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in model_folder.glob("checkpoint_*.pt"):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_<iteration>.pt found in {model_folder}")
    return max(candidates, key=lambda item: item[0])[1]


def _existing_path(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        resolved = candidate.expanduser()
        try:
            if resolved.exists():
                return resolved.resolve()
        except OSError:
            continue
    attempted = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve {label}. Tried:\n  {attempted}")


def resolve_bfm_checkpoint(metadata: Mapping[str, object], override: Path | None) -> Path:
    if override is not None:
        return _existing_path([override], "first-stage BFM checkpoint")
    recorded = Path(str(metadata.get("bfm_checkpoint", "")))
    run_name = recorded.parent.name if recorded.name == "checkpoint" else DEFAULT_BFM_RUN
    return _existing_path(
        [
            recorded,
            PROJECT_ROOT / "huiying" / run_name / "checkpoint",
            PROJECT_ROOT / "huiying" / DEFAULT_BFM_RUN / "checkpoint",
        ],
        "first-stage BFM checkpoint",
    )


def resolve_robot_config(metadata: Mapping[str, object], override: Path | None) -> Path:
    if override is not None:
        return _existing_path([override], "robot config")
    recorded = Path(str(metadata.get("robot_config", "")))
    return _existing_path([recorded, PROJECT_ROOT / "humanoidverse/config/robot/piplus" / recorded.name], "robot config")


def resolve_expert_dataset(metadata: Mapping[str, object], override: Path | None) -> Path:
    if override is not None:
        return _existing_path([override], "expert dataset")
    recorded = Path(str(metadata.get("expert_dataset", "")))
    return _existing_path([recorded, PROJECT_ROOT / "dataset/pi_LSE_lafan_260706" / recorded.name], "expert dataset")


def resolve_teacher_policy(metadata: Mapping[str, object], override: Path | None) -> Path:
    if override is not None:
        return _existing_path([override], "AMP teacher policy")
    teacher_metadata = metadata.get("teacher_policy", {})
    recorded_value = teacher_metadata.get("artifact") if isinstance(teacher_metadata, Mapping) else None
    candidates = [Path(str(recorded_value))] if recorded_value else []
    candidates.append(Path(DEFAULT_TEACHER_POLICY))
    return _existing_path(candidates, "AMP teacher policy")


def resolve_play_device(requested: str) -> torch.device:
    """Resolve playback device, falling back to CPU when CUDA is unavailable."""
    requested = str(requested).strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[WARN] Requested {device}, but CUDA is unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device


def _apply_deadzone(value: float, deadzone: float) -> float:
    magnitude = abs(float(value))
    if magnitude <= deadzone:
        return 0.0
    return math.copysign((magnitude - deadzone) / (1.0 - deadzone), value)


def _scale_axis(value: float, low: float, high: float) -> float:
    return value * (high if value >= 0.0 else abs(low))


def command_from_axes(
    axes: Mapping[int, float],
    command_low: np.ndarray,
    command_high: np.ndarray,
    *,
    forward_axis: int,
    lateral_axis: int,
    yaw_axis: int,
    forward_sign: float,
    lateral_sign: float,
    yaw_sign: float,
    deadzone: float,
) -> np.ndarray:
    raw = np.asarray(
        [
            forward_sign * axes.get(forward_axis, 0.0),
            lateral_sign * axes.get(lateral_axis, 0.0),
            yaw_sign * axes.get(yaw_axis, 0.0),
        ],
        dtype=np.float32,
    )
    mapped = np.asarray([_apply_deadzone(value, deadzone) for value in raw], dtype=np.float32)
    return np.asarray(
        [_scale_axis(mapped[index], float(command_low[index]), float(command_high[index])) for index in range(3)],
        dtype=np.float32,
    )


class LinuxJoystick:
    """Non-blocking reader for the Linux joystick API."""

    def __init__(self, device: Path) -> None:
        self.device = device
        self.fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
        self.axes: dict[int, float] = {}
        self.buttons: dict[int, bool] = {}

    def poll(self) -> set[int]:
        pressed: set[int] = set()
        while True:
            try:
                payload = os.read(self.fd, JOYSTICK_EVENT.size)
            except BlockingIOError:
                break
            if len(payload) != JOYSTICK_EVENT.size:
                break
            _timestamp, value, event_type, number = JOYSTICK_EVENT.unpack(payload)
            is_initial = bool(event_type & JS_EVENT_INIT)
            event_type &= ~JS_EVENT_INIT
            if event_type == JS_EVENT_AXIS:
                self.axes[int(number)] = float(np.clip(value / 32767.0, -1.0, 1.0))
            elif event_type == JS_EVENT_BUTTON:
                old_value = self.buttons.get(int(number), False)
                new_value = bool(value)
                self.buttons[int(number)] = new_value
                if new_value and not old_value and not is_initial:
                    pressed.add(int(number))
        return pressed

    def close(self) -> None:
        os.close(self.fd)


class PygameGamepad:
    """Read an Xbox-style gamepad using the same mapping as HT_lab_hi/play."""

    def __init__(
        self,
        joystick_id: int,
        *,
        axis_lx: int,
        axis_ly: int,
        axis_rx: int,
        deadzone: float,
        reset_button: int,
        quit_button: int,
        debug: bool = False,
    ) -> None:
        if pygame is None:
            raise RuntimeError("--gamepad requires pygame; install it with `python -m pip install pygame`")
        if not 0.0 <= deadzone < 1.0:
            raise ValueError("--deadzone must be in [0, 1)")
        pygame.init()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError("No pygame joystick detected; check the controller and SDL input permissions")
        if not 0 <= joystick_id < count:
            raise ValueError(f"--gamepad-id must be in [0, {count - 1}], got {joystick_id}")
        self.joystick = pygame.joystick.Joystick(joystick_id)
        self.joystick.init()
        self.axis_lx = int(axis_lx)
        self.axis_ly = int(axis_ly)
        self.axis_rx = int(axis_rx)
        self.deadzone = float(deadzone)
        self.reset_button = int(reset_button)
        self.quit_button = int(quit_button)
        self.debug = bool(debug)
        self._buttons = [False] * self.joystick.get_numbuttons()
        print(
            f"[INFO] Gamepad={self.joystick.get_name()} axes={self.joystick.get_numaxes()} "
            f"buttons={self.joystick.get_numbuttons()} mapping="
            f"(lx={self.axis_lx}, ly={self.axis_ly}, rx={self.axis_rx}) deadzone={self.deadzone:.3f}"
        )
        if self.debug:
            self.print_debug()

    def _axis(self, index: int) -> float:
        if 0 <= index < self.joystick.get_numaxes():
            return float(self.joystick.get_axis(index))
        return 0.0

    def poll(self) -> tuple[dict[int, float], set[int]]:
        pygame.event.pump()
        axes = {
            self.axis_lx: self._axis(self.axis_lx),
            self.axis_ly: self._axis(self.axis_ly),
            self.axis_rx: self._axis(self.axis_rx),
        }
        pressed: set[int] = set()
        current = [bool(self.joystick.get_button(index)) for index in range(len(self._buttons))]
        pressed.update(index for index, (was_down, is_down) in enumerate(zip(self._buttons, current)) if is_down and not was_down)
        self._buttons = current
        if self.debug:
            self.print_debug(axes, current)
        return axes, pressed

    def print_debug(self, axes: Mapping[int, float] | None = None, buttons: list[bool] | None = None) -> None:
        if axes is None:
            axes = {index: self._axis(index) for index in range(self.joystick.get_numaxes())}
        if buttons is None:
            buttons = [bool(self.joystick.get_button(index)) for index in range(self.joystick.get_numbuttons())]
        print(
            f"[GAMEPAD] axes={[f'{axes.get(index, 0.0):+.2f}' for index in sorted(axes)]} "
            f"buttons={buttons}",
            flush=True,
        )

    def close(self) -> None:
        try:
            self.joystick.quit()
        finally:
            pygame.joystick.quit()
            pygame.quit()


class PassivePolicyViewer:
    def __init__(self, xml_path: Path, expected_qpos_size: int) -> None:
        import mujoco.viewer

        spec = mujoco.MjSpec.from_file(str(xml_path))
        spec.worldbody.add_geom(
            name="stage2_play_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0.0, 0.0, 0.0],
            size=[20.0, 20.0, 0.02],
            rgba=[0.45, 0.47, 0.50, 1.0],
            contype=0,
            conaffinity=0,
        )
        spec.worldbody.add_light(
            name="stage2_play_light",
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


class MujocoQposRenderer:
    """Render policy qpos with the target robot's MuJoCo XML."""

    def __init__(
        self,
        xml_path: Path,
        render_size: int = 480,
        *,
        camera_distance: float = 3.0,
        camera_azimuth: float = 135.0,
        camera_elevation: float = -18.0,
        expected_qpos_size: int | None = None,
    ) -> None:
        spec = mujoco.MjSpec.from_file(str(xml_path))
        spec.worldbody.add_geom(
            name="stage2_render_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0.0, 0.0, 0.0],
            size=[20.0, 20.0, 0.02],
            rgba=[0.45, 0.47, 0.50, 1.0],
            contype=0,
            conaffinity=0,
        )
        spec.worldbody.add_light(
            name="stage2_render_light",
            pos=[0.0, -3.0, 4.0],
            dir=[0.2, 0.5, -1.0],
            diffuse=[0.8, 0.8, 0.8],
            ambient=[0.35, 0.35, 0.35],
        )
        self.model = spec.compile()
        if expected_qpos_size is not None and self.model.nq != int(expected_qpos_size):
            raise ValueError(f"Expected renderer nq={expected_qpos_size}, got nq={self.model.nq}")
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), int(render_size))
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), int(render_size))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=render_size, width=render_size)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = float(camera_distance)
        self.camera.azimuth = float(camera_azimuth)
        self.camera.elevation = float(camera_elevation)

    def render_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.size != self.model.nq:
            raise ValueError(f"Expected qpos size {self.model.nq}, got {qpos.size}")
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = [float(qpos[0]), float(qpos[1]), max(float(qpos[2]), 0.75)]
        self.renderer.update_scene(self.data, camera=self.camera)
        return np.ascontiguousarray(self.renderer.render()).astype(np.uint8, copy=False)

    def close(self) -> None:
        self.renderer.close()


@dataclass
class PlaybackPaths:
    model_folder: Path
    checkpoint: Path
    bfm_checkpoint: Path
    robot_config: Path
    expert_dataset: Path


def _resolve_paths(args: argparse.Namespace) -> tuple[PlaybackPaths, dict[str, object]]:
    model_folder = args.model_folder.expanduser().resolve()
    metadata_path = model_folder / "config.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Stage2 metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else latest_stage2_checkpoint(model_folder)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing Stage2 checkpoint: {checkpoint}")
    paths = PlaybackPaths(
        model_folder=model_folder,
        checkpoint=checkpoint,
        bfm_checkpoint=resolve_bfm_checkpoint(metadata, args.bfm_checkpoint),
        robot_config=resolve_robot_config(metadata, args.robot_config),
        expert_dataset=resolve_expert_dataset(metadata, args.expert_dataset),
    )
    return paths, metadata


def play(args: argparse.Namespace) -> None:
    paths, metadata = _resolve_paths(args)
    if not 0.0 < args.command_smoothing <= 1.0:
        raise ValueError("--command-smoothing must be in (0, 1]")
    if not 0.0 <= args.deadzone < 1.0:
        raise ValueError("--deadzone must be in [0, 1)")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.render_size <= 0:
        raise ValueError("--render-size must be positive")

    device = resolve_play_device(args.device)
    policy_device = resolve_play_device(args.policy_device or str(device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if policy_device.type not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported policy device: {policy_device}")
    bfm_load_device = str(policy_device)

    _ensure_runtime_cache(paths.model_folder)
    env, robot_training = build_piplus_locomotion_env(
        device=str(device),
        robot_config=str(paths.robot_config),
        expert_dataset=str(paths.expert_dataset),
        num_envs=1,
        seed=args.seed,
        max_episode_length_s=args.max_episode_length_s,
        simulator=args.simulator,
        disable_obs_noise=True,
        disable_domain_randomization=True,
    )
    # Playback is intentionally single-environment.  Keep this check close to
    # construction so a stale Hydra interpolation can fail before Isaac Sim
    # allocates a training-sized scene on the GPU.
    if int(env.num_envs) != 1:
        env.close()
        raise RuntimeError(
            f"Stage2 playback requires exactly one environment, but the simulator created {env.num_envs}. "
            "Check the playback Hydra overrides instead of starting a training-sized scene."
        )
    bfm_model = load_model_from_checkpoint_dir(str(paths.bfm_checkpoint), device=bfm_load_device)
    bfm_model.eval()
    for parameter in bfm_model.parameters():
        parameter.requires_grad_(False)
    teacher = None
    if "teacher_policy" in metadata:
        teacher = PiPlusAMPTeacherPolicy(resolve_teacher_policy(metadata, args.teacher_policy), policy_device)

    observation, _ = env.reset(to_numpy=False, reset_to_default_pose=True)
    observation_t = _to_torch_obs(observation, policy_device)
    commands = torch.zeros(1, 3, device=policy_device)
    if teacher is not None:
        teacher_action_scales = torch.tensor(TEACHER_POLICY_ACTION_SCALES, device=policy_device)
        bfm_action_scales = torch.tensor(robot_training.bfm_action_position_scales, device=policy_device)
        encoder_input = teacher.normalize(
            build_teacher_policy_observation(
                observation_t,
                commands,
                tuple(robot_training.policy_joint_names),
                teacher_action_scales=teacher_action_scales,
                bfm_action_scales=bfm_action_scales,
            )
        )
        input_scale = torch.ones(encoder_input.shape[-1], device=policy_device, dtype=encoder_input.dtype)
    else:
        encoder_input = flatten_encoder_observation(observation_t, commands)
        input_scale = encoder_input_scale(observation_t, commands)
    policy = CommandEncoderPolicy(
        encoder_input.shape[-1],
        int(metadata["z_dim"]),
        hidden_dim=int(metadata["command_encoder_hidden_dim"]),
        hidden_layers=int(metadata["command_encoder_hidden_layers"]),
    ).to(policy_device)
    checkpoint = torch.load(paths.checkpoint, map_location=policy_device, weights_only=False)
    load_command_encoder_policy_state(policy, checkpoint, input_scale)
    policy.eval()

    command_low = np.asarray(metadata["command_range"]["low"], dtype=np.float32)
    command_high = np.asarray(metadata["command_range"]["high"], dtype=np.float32)
    if args.gamepad and args.fixed_command is not None:
        raise ValueError("--gamepad and --fixed-command are mutually exclusive")
    joystick = None
    gamepad = None
    if args.fixed_command is None:
        if args.gamepad:
            gamepad = PygameGamepad(
                args.gamepad_id,
                axis_lx=args.axis_lx,
                axis_ly=args.axis_ly,
                axis_rx=args.axis_rx,
                deadzone=args.deadzone,
                reset_button=args.reset_button,
                quit_button=args.quit_button,
                debug=args.gamepad_debug,
            )
        else:
            joystick = LinuxJoystick(args.joystick)
    fixed_command = None if args.fixed_command is None else np.asarray(args.fixed_command, dtype=np.float32)
    if fixed_command is not None:
        if (fixed_command < command_low).any() or (fixed_command > command_high).any():
            raise ValueError(f"--fixed-command must be within low={command_low.tolist()} high={command_high.tolist()}")

    initial_qpos, _ = env._get_qpos_qvel(to_numpy=True)
    # MP4 capture already creates an offscreen MuJoCo context.  Keep recording
    # headless by default; an explicit --show-viewer opts into the second GLFW
    # context when interactive display is needed.
    if args.headless and args.show_viewer:
        raise ValueError("--headless and --show-viewer are mutually exclusive")
    show_viewer = args.show_viewer or (not args.headless and not args.save_mp4)
    if args.save_mp4 and not show_viewer:
        print("[INFO] --save-mp4 is running headless; add --show-viewer to open the interactive window.", flush=True)
    if args.save_mp4 and args.show_viewer:
        print("[WARN] --show-viewer + --save-mp4 creates two MuJoCo render contexts and uses more GPU memory.", flush=True)
    viewer = (
        PassivePolicyViewer(Path(robot_training.robot.xml_path), int(initial_qpos.shape[-1]))
        if show_viewer
        else None
    )
    video_renderer = (
        MujocoQposRenderer(
            Path(robot_training.robot.xml_path),
            render_size=args.render_size,
            camera_distance=args.camera_distance,
            camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
            expected_qpos_size=int(initial_qpos.shape[-1]),
        )
        if args.save_mp4
        else None
    )
    video_path = None
    if args.save_mp4:
        video_path = (
            args.output.expanduser().resolve()
            if args.output is not None
            else paths.model_folder
            / "stage2_playback"
            / f"{paths.checkpoint.stem}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
    video_writer: media.VideoWriter | None = None
    print(f"[INFO] Stage2 checkpoint={paths.checkpoint} iteration={checkpoint.get('iteration', 'unknown')}")
    print(f"[INFO] Simulator={args.simulator} env_device={device} policy_device={policy_device}")
    print(f"[INFO] First-stage BFM checkpoint={paths.bfm_checkpoint}")
    if video_path is not None:
        print(f"[INFO] Recording MP4 to {video_path} at {args.fps} FPS")
    if gamepad is not None:
        print(
            "[INFO] Controls: left stick Y=forward, left stick X=lateral, right stick X=yaw; "
            f"button {args.reset_button}=reset, button {args.quit_button}=quit"
        )
    elif joystick is not None:
        print(
            "[INFO] Controls: left stick Y=forward, left stick X=lateral, right stick X=yaw; "
            f"button {args.reset_button}=reset, button {args.quit_button}=quit"
        )

    step = 0
    try:
        while args.max_steps is None or step < args.max_steps:
            if viewer is not None and not viewer.is_running():
                break
            started_at = time.monotonic()
            if gamepad is not None:
                gamepad_axes, pressed = gamepad.poll()
            else:
                gamepad_axes, pressed = None, set() if joystick is None else joystick.poll()
            if args.quit_button in pressed:
                break
            if args.reset_button in pressed:
                observation, _ = env.reset(to_numpy=False, reset_to_default_pose=True)
                commands.zero_()

            if fixed_command is None:
                forward_axis = args.axis_ly if gamepad_axes is not None else args.forward_axis
                lateral_axis = args.axis_lx if gamepad_axes is not None else args.lateral_axis
                yaw_axis = args.axis_rx if gamepad_axes is not None else args.yaw_axis
                target = command_from_axes(
                    gamepad_axes if gamepad_axes is not None else joystick.axes,
                    command_low,
                    command_high,
                    forward_axis=forward_axis,
                    lateral_axis=lateral_axis,
                    yaw_axis=yaw_axis,
                    forward_sign=args.forward_sign,
                    lateral_sign=args.lateral_sign,
                    yaw_sign=args.yaw_sign,
                    deadzone=args.deadzone,
                )
            else:
                target = fixed_command
            target_t = torch.as_tensor(target, device=policy_device).unsqueeze(0)
            commands.add_(args.command_smoothing * (target_t - commands))

            with torch.inference_mode():
                observation_t = _to_torch_obs(observation, policy_device)
                if teacher is not None:
                    encoder_input = teacher.normalize(
                        build_teacher_policy_observation(
                            observation_t,
                            commands,
                            tuple(robot_training.policy_joint_names),
                            teacher_action_scales=teacher_action_scales,
                            bfm_action_scales=bfm_action_scales,
                        )
                    )
                else:
                    encoder_input = flatten_encoder_observation(observation_t, commands)
                raw_z = policy.deterministic_z(encoder_input)
                action = _bfm_action(bfm_model, observation_t, bfm_model.project_z(raw_z)).to(device)
            observation, _reward, terminated, truncated, _info = env.step(action, to_numpy=False)

            if viewer is not None or video_renderer is not None:
                qpos, _ = env._get_qpos_qvel(to_numpy=True)
            if viewer is not None:
                viewer.update(qpos[0])
            if video_renderer is not None:
                frame = video_renderer.render_qpos(qpos[0])
                if video_writer is None:
                    video_writer = media.VideoWriter(video_path, shape=frame.shape[:2], fps=args.fps)
                    video_writer.__enter__()
                video_writer.add_image(frame)
            step += 1
            if step == 1 or (args.log_every_steps > 0 and step % args.log_every_steps == 0):
                base_velocity = env._env.base_lin_vel[0].detach().cpu().numpy()
                yaw_velocity = float(env._env.base_ang_vel[0, 2].detach().cpu())
                print(
                    f"[INFO] step={step} command={commands[0].detach().cpu().numpy().round(3).tolist()} "
                    f"velocity={[round(float(base_velocity[0]), 3), round(float(base_velocity[1]), 3), round(yaw_velocity, 3)]} "
                    f"terminated={bool(terminated.any())} truncated={bool(truncated.any())}",
                    flush=True,
                )
            if args.realtime:
                time.sleep(max(0.0, float(env._env.dt) - (time.monotonic() - started_at)))
    finally:
        if video_writer is not None:
            video_writer.close()
            print(f"[INFO] Saved MP4: {video_path}")
        if video_renderer is not None:
            video_renderer.close()
        if joystick is not None:
            joystick.close()
        if gamepad is not None:
            gamepad.close()
        if viewer is not None:
            viewer.close()
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to the newest checkpoint_<iteration>.pt in model-folder.")
    parser.add_argument("--bfm-checkpoint", type=Path, default=None)
    parser.add_argument("--robot-config", type=Path, default=None)
    parser.add_argument("--expert-dataset", type=Path, default=None)
    parser.add_argument("--teacher-policy", type=Path, default=None)
    parser.add_argument("--simulator", choices=("isaacsim", "mujoco"), default="mujoco")
    parser.add_argument("--device", default="auto", help="auto selects CUDA when available, otherwise CPU.")
    parser.add_argument(
        "--policy-device",
        default=None,
        help="Device for BFM and Stage2 inference; useful as CPU when Isaac Sim exhausts GPU memory.",
    )
    parser.add_argument("--joystick", type=Path, default=Path("/dev/input/js0"))
    parser.add_argument("--gamepad", action="store_true", help="Use pygame gamepad input (HT_lab_hi mapping).")
    parser.add_argument("--gamepad-id", "--gamepad_id", dest="gamepad_id", type=int, default=0, help="pygame joystick index.")
    parser.add_argument(
        "--gamepad-debug",
        "--gamepad_debug",
        dest="gamepad_debug",
        action="store_true",
        help="Print raw pygame axes/buttons each step.",
    )
    parser.add_argument("--axis-lx", "--axis_lx", dest="axis_lx", type=int, default=0, help="Gamepad left stick X axis.")
    parser.add_argument("--axis-ly", "--axis_ly", dest="axis_ly", type=int, default=1, help="Gamepad left stick Y axis.")
    parser.add_argument("--axis-rx", "--axis_rx", dest="axis_rx", type=int, default=3, help="Gamepad right stick X axis.")
    parser.add_argument("--forward-axis", type=int, default=1)
    parser.add_argument("--lateral-axis", type=int, default=0)
    parser.add_argument("--yaw-axis", type=int, default=3)
    parser.add_argument("--forward-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--lateral-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--yaw-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--reset-button", type=int, default=0, help="Xbox A / PlayStation Cross by default.")
    parser.add_argument("--quit-button", type=int, default=1, help="Xbox B / PlayStation Circle by default.")
    parser.add_argument("--fixed-command", type=float, nargs=3, metavar=("VX", "VY", "WZ"), default=None)
    parser.add_argument("--command-smoothing", type=float, default=0.1)
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--show-viewer",
        "--viewer",
        dest="show_viewer",
        action="store_true",
        help="Open the interactive MuJoCo viewer; with --save-mp4 this uses an extra render context.",
    )
    parser.add_argument(
        "--save-mp4",
        action="store_true",
        help="Record the playback to an offscreen MuJoCo-rendered MP4.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="MP4 output path; defaults to model-folder/stage2_playback/<checkpoint>_<timestamp>.mp4.",
    )
    parser.add_argument("--fps", type=float, default=50.0, help="Output video frame rate.")
    parser.add_argument("--render-size", type=int, default=720, help="Output video width and height in pixels.")
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--no-realtime", dest="realtime", action="store_false")
    parser.set_defaults(realtime=True)
    return parser.parse_args()


def main() -> None:
    play(parse_args())


if __name__ == "__main__":
    main()
