from __future__ import annotations

import csv
import json
import os
import warnings
from pathlib import Path
from typing import Any, Sequence

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"

import joblib
import numpy as np
import torch
from torch.utils._pytree import tree_map


DEFAULT_MCAP_PATH = Path("/home/youyou/bfm_rosbag/zs_1_18_40/zs_1_18_40_0.mcap")
PIPLUS_SIM_JOINT_NAMES = [
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
]
STATE_SUBSLICES = {
    "dof_pos": [0, 23],
    "dof_vel": [23, 46],
    "projected_gravity": [46, 49],
    "base_ang_vel": [49, 52],
}
ROBOT_CONFIG_OVERRIDES = {
    "g1": "robot=g1/g1_29dof_hard_waist",
    "PiPlus_S_12L8A0G2H1W_LSE": "robot=piplus/PiPlus_S_12L8A0G2H1W_LSE",
    "piplus_lse": "robot=piplus/PiPlus_S_12L8A0G2H1W_LSE",
}


def _resolve_path(path: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.exists():
        return expanded.resolve()

    import humanoidverse

    humanoidverse_dir = Path(humanoidverse.__file__).parent
    path_text = str(expanded)
    marker = "humanoidverse/"
    if marker in path_text:
        repo_relative = humanoidverse_dir / path_text.split(marker, 1)[1]
        if repo_relative.exists():
            return repo_relative.resolve()

    return expanded


def _append_or_replace_hydra_override(overrides: list[str], override: str) -> None:
    key = override.split("=", 1)[0]
    for index, existing in enumerate(overrides):
        if existing.split("=", 1)[0] == key:
            overrides[index] = override
            return
    overrides.append(override)


def _is_hydra_group_override(override: str) -> bool:
    key = override.split("=", 1)[0].lstrip("+~").split("@", 1)[0]
    return "." not in key


def _load_config_for_inference(
    model_folder: Path,
    data_path: Path | None,
    headless: bool,
    device: str,
    simulator: str,
    disable_dr: bool,
    disable_obs_noise: bool,
    robot: str | None,
    no_training_config: bool,
) -> tuple[dict[str, Any], bool]:
    with (model_folder / "config.json").open("r") as f:
        config = json.load(f)

    resolved_config_path = model_folder / "config.yaml"
    using_training_config = False
    if not no_training_config and resolved_config_path.exists():
        config["env"]["resolved_config_path"] = str(resolved_config_path.resolve())
        using_training_config = True
        print(f"Loading inference YAML config from {resolved_config_path.resolve()}")
    elif no_training_config:
        config["env"].pop("resolved_config_path", None)
        print("Skipping training YAML config; using inference config.json and hydra overrides")

    if data_path is not None:
        resolved_data_path = _resolve_path(data_path)
        if not resolved_data_path.exists():
            raise FileNotFoundError(f"Motion data file not found: {data_path} (resolved to {resolved_data_path})")
        config["env"]["lafan_tail_path"] = str(resolved_data_path)

    if using_training_config:
        saved_hydra_overrides = config["env"].get("hydra_overrides", [])
        stale_group_overrides = [
            override for override in saved_hydra_overrides if _is_hydra_group_override(override)
        ]
        config["env"]["hydra_overrides"] = [
            override for override in saved_hydra_overrides if not _is_hydra_group_override(override)
        ]
        if stale_group_overrides:
            print(
                "Ignoring saved Hydra group overrides because config.yaml is already fully resolved: "
                f"{stale_group_overrides}"
            )

    hydra_overrides = config["env"].setdefault("hydra_overrides", [])
    if robot is not None:
        if robot not in ROBOT_CONFIG_OVERRIDES:
            raise ValueError(f"Unsupported robot {robot!r}. Expected one of: {sorted(ROBOT_CONFIG_OVERRIDES)}")
        _append_or_replace_hydra_override(hydra_overrides, ROBOT_CONFIG_OVERRIDES[robot])

    _append_or_replace_hydra_override(hydra_overrides, "env.config.max_episode_length_s=10000")
    _append_or_replace_hydra_override(hydra_overrides, f"env.config.headless={headless}")
    _append_or_replace_hydra_override(hydra_overrides, f"simulator={simulator}")
    if simulator == "mujoco":
        if robot in (None, "g1"):
            _append_or_replace_hydra_override(
                hydra_overrides,
                "robot.asset.xml_file=g1/scene_29dof_freebase_mujoco.xml",
            )
        elif robot in ("PiPlus_S_12L8A0G2H1W_LSE", "piplus_lse"):
            _append_or_replace_hydra_override(
                hydra_overrides,
                "robot.asset.xml_file=package://ht_urdf/"
                "PiPlus_S_12L8A0G2H1W_LSE_260611/xml/"
                "PiPlus_S_12L8A0G2H1W_LSE_260611_with_armature.xml",
            )

    config["env"]["device"] = "cuda:0" if device == "cuda" else device
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise
    print(f"Inference resolved_config_path: {config['env'].get('resolved_config_path')}")
    print(f"Inference hydra_overrides: {hydra_overrides}")
    return config, using_training_config


def _actor_input_keys(model) -> list[str]:
    input_filter = getattr(model.cfg.archi.actor, "input_filter", None)
    keys = getattr(input_filter, "key", None)
    if keys is None:
        raise ValueError("Cannot find actor input_filter.key in model config")
    if isinstance(keys, str):
        return [keys]
    if isinstance(keys, Sequence):
        return list(keys)
    raise TypeError(f"Unsupported actor input_filter.key type: {type(keys)}")


def _as_2d_tensor(value: torch.Tensor, key: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"Expected observation[{key!r}] to be a torch.Tensor, got {type(value)}")
    if value.ndim == 1:
        return value.unsqueeze(0)
    if value.ndim == 2:
        return value
    return value.reshape(value.shape[0], -1)


def _build_actor_obs(
    observation: dict[str, torch.Tensor],
    z: torch.Tensor,
    actor_keys: list[str],
) -> tuple[torch.Tensor, dict[str, list[int]]]:
    parts = []
    slices: dict[str, list[int]] = {}
    start = 0
    for key in actor_keys:
        if key not in observation:
            raise KeyError(f"Actor observation key {key!r} missing from observation keys {list(observation)}")
        value = _as_2d_tensor(observation[key], key)
        end = start + value.shape[-1]
        slices[key] = [start, end]
        parts.append(value)
        start = end

    z = _as_2d_tensor(z, "z")
    end = start + z.shape[-1]
    slices["z"] = [start, end]
    parts.append(z)
    return torch.cat(parts, dim=-1), slices


def _tracking_inference(model, obs) -> torch.Tensor:
    z = model.backward_map(obs)
    for step in range(z.shape[0]):
        end_idx = min(step + 1, z.shape[0])
        z[step] = z[step:end_idx].mean(dim=0)
    return model.project_z(z)


def _initialize_to_motion(wrapped_env, obs_dict: dict[str, torch.Tensor], simulator: str, num_envs: int) -> dict[str, torch.Tensor]:
    observation, _ = wrapped_env.reset(to_numpy=False)
    del observation

    ref_body_rots = obs_dict["ref_body_rots"][0, 0].clone()
    if simulator == "isaacsim":
        ref_body_rots = ref_body_rots[[3, 0, 1, 2]]
    ref_root_init_state = torch.cat(
        [
            obs_dict["ref_body_pos"][0, 0],
            ref_body_rots,
            obs_dict["ref_body_vels"][0, 0],
            obs_dict["ref_body_angular_vels"][0, 0],
        ]
    )
    dof_init_state = torch.zeros_like(wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0])
    dof_init_state[..., 0] = obs_dict["dof_pos"][0]
    dof_init_state[..., 1] = obs_dict["ref_dof_vel"][0]
    target_states = {
        "dof_states": dof_init_state,
        "root_states": torch.stack([ref_root_init_state.clone() for _ in range(num_envs)]),
    }
    env_ids = torch.arange(num_envs, dtype=torch.long, device=wrapped_env._env.device)
    wrapped_env._env.reset_envs_idx(env_ids, target_states=target_states)

    zero_action = torch.zeros(
        (num_envs, wrapped_env.action_space.shape[-1]),
        dtype=torch.float32,
        device=wrapped_env._env.device,
    )
    wrapped_env.step(zero_action, to_numpy=False)
    return wrapped_env._get_g1env_observation(to_numpy=False)


def _save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"Saved {path}")


def _read_mcap_actor_obs(mcap_path: Path, topic: str, expected_dim: int | None) -> tuple[np.ndarray, np.ndarray]:
    try:
        from mcap_ros2.reader import read_ros2_messages
    except ImportError:
        return _read_mcap_actor_obs_raw_cdr(mcap_path=mcap_path, topic=topic, expected_dim=expected_dim)

    rows = []
    times = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for msg in read_ros2_messages(str(mcap_path), topics=[topic]):
            row = np.asarray(msg.ros_msg.data, dtype=np.float32)
            if expected_dim is not None and row.shape != (expected_dim,):
                raise ValueError(f"{topic} message has shape {row.shape}, expected ({expected_dim},)")
            rows.append(row)
            times.append(msg.log_time_ns)
    if not rows:
        raise ValueError(f"No messages found on {topic} in {mcap_path}")
    return np.stack(rows, axis=0), np.asarray(times, dtype=np.int64)


def _align(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


def _read_u32(data: bytes, offset: int, endian: str) -> tuple[int, int]:
    offset = _align(offset, 4)
    return int.from_bytes(data[offset : offset + 4], endian), offset + 4


def _skip_ros_string(data: bytes, offset: int, endian: str) -> int:
    length, offset = _read_u32(data, offset, endian)
    return offset + length


def _decode_float32_multi_array_cdr(data: bytes) -> np.ndarray:
    if len(data) < 16:
        raise ValueError(f"CDR payload too short for Float32MultiArray: {len(data)} bytes")

    encapsulation = int.from_bytes(data[:2], "big")
    if encapsulation in (0x0000, 0x0002):
        endian = "big"
        dtype = ">f4"
    else:
        endian = "little"
        dtype = "<f4"

    offset = 4

    dim_len, offset = _read_u32(data, offset, endian)
    for _ in range(dim_len):
        offset = _skip_ros_string(data, offset, endian)
        _, offset = _read_u32(data, offset, endian)  # size
        _, offset = _read_u32(data, offset, endian)  # stride

    _, offset = _read_u32(data, offset, endian)  # layout.data_offset
    data_len, offset = _read_u32(data, offset, endian)
    offset = _align(offset, 4)
    byte_len = data_len * 4
    if offset + byte_len > len(data):
        raise ValueError(
            f"Float32MultiArray data overruns CDR payload: offset={offset}, "
            f"data_len={data_len}, payload_len={len(data)}"
        )
    return np.frombuffer(data, dtype=dtype, count=data_len, offset=offset).astype(np.float32, copy=True)


def _read_mcap_actor_obs_raw_cdr(mcap_path: Path, topic: str, expected_dim: int | None) -> tuple[np.ndarray, np.ndarray]:
    try:
        from mcap.reader import make_reader
    except ImportError:
        return _read_mcap_actor_obs_minimal(mcap_path=mcap_path, topic=topic, expected_dim=expected_dim)

    rows = []
    times = []
    with mcap_path.open("rb") as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages(topics=[topic]):
            if channel.message_encoding != "cdr":
                raise ValueError(f"Unsupported {topic} message encoding {channel.message_encoding!r}; expected 'cdr'")
            row = _decode_float32_multi_array_cdr(message.data)
            if expected_dim is not None and row.shape != (expected_dim,):
                raise ValueError(f"{topic} message has shape {row.shape}, expected ({expected_dim},)")
            rows.append(row)
            times.append(message.log_time)

    if not rows:
        raise ValueError(f"No messages found on {topic} in {mcap_path}")
    return np.stack(rows, axis=0), np.asarray(times, dtype=np.int64)


def _mcap_read_u16(data: bytes, offset: int) -> tuple[int, int]:
    return int.from_bytes(data[offset : offset + 2], "little"), offset + 2


def _mcap_read_u32(data: bytes, offset: int) -> tuple[int, int]:
    return int.from_bytes(data[offset : offset + 4], "little"), offset + 4


def _mcap_read_u64(data: bytes, offset: int) -> tuple[int, int]:
    return int.from_bytes(data[offset : offset + 8], "little"), offset + 8


def _mcap_read_string(data: bytes, offset: int) -> tuple[str, int]:
    length, offset = _mcap_read_u32(data, offset)
    value = data[offset : offset + length].decode("utf-8")
    return value, offset + length


def _mcap_skip_bytes(data: bytes, offset: int) -> int:
    length, offset = _mcap_read_u64(data, offset)
    return offset + length


def _mcap_skip_metadata(data: bytes, offset: int) -> int:
    length, offset = _mcap_read_u32(data, offset)
    return offset + length


def _mcap_parse_channel(payload: bytes) -> tuple[int, str, str]:
    offset = 0
    channel_id, offset = _mcap_read_u16(payload, offset)
    _, offset = _mcap_read_u16(payload, offset)  # schema_id
    topic, offset = _mcap_read_string(payload, offset)
    message_encoding, offset = _mcap_read_string(payload, offset)
    _mcap_skip_metadata(payload, offset)
    return channel_id, topic, message_encoding


def _mcap_parse_message(payload: bytes) -> tuple[int, int, bytes]:
    offset = 0
    channel_id, offset = _mcap_read_u16(payload, offset)
    _, offset = _mcap_read_u32(payload, offset)  # sequence
    log_time, offset = _mcap_read_u64(payload, offset)
    _, offset = _mcap_read_u64(payload, offset)  # publish_time
    return channel_id, log_time, payload[offset:]


def _mcap_parse_chunk(payload: bytes) -> bytes:
    offset = 0
    _, offset = _mcap_read_u64(payload, offset)  # message_start_time
    _, offset = _mcap_read_u64(payload, offset)  # message_end_time
    _, offset = _mcap_read_u64(payload, offset)  # uncompressed_size
    _, offset = _mcap_read_u32(payload, offset)  # uncompressed_crc
    compression, offset = _mcap_read_string(payload, offset)
    records_len, offset = _mcap_read_u64(payload, offset)
    records = payload[offset : offset + records_len]

    if compression == "":
        return records
    if compression == "zstd":
        import zstandard

        return zstandard.ZstdDecompressor().decompress(records)
    if compression == "lz4":
        import lz4.frame

        return lz4.frame.decompress(records)
    raise ValueError(f"Unsupported MCAP chunk compression {compression!r}")


def _mcap_iter_records(data: bytes):
    offset = 0
    data_len = len(data)
    while offset + 9 <= data_len:
        opcode = data[offset]
        offset += 1
        length, offset = _mcap_read_u64(data, offset)
        end = offset + length
        if end > data_len:
            break
        yield opcode, data[offset:end]
        offset = end


def _read_mcap_actor_obs_minimal(mcap_path: Path, topic: str, expected_dim: int | None) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    times = []
    channels: dict[int, tuple[str, str]] = {}

    def handle_record(opcode: int, payload: bytes) -> None:
        if opcode == 0x04:  # Channel
            channel_id, channel_topic, message_encoding = _mcap_parse_channel(payload)
            channels[channel_id] = (channel_topic, message_encoding)
            return
        if opcode == 0x05:  # Message
            channel_id, log_time, message_data = _mcap_parse_message(payload)
            channel_topic, message_encoding = channels.get(channel_id, ("", ""))
            if channel_topic != topic:
                return
            if message_encoding != "cdr":
                raise ValueError(f"Unsupported {topic} message encoding {message_encoding!r}; expected 'cdr'")
            row = _decode_float32_multi_array_cdr(message_data)
            if expected_dim is not None and row.shape != (expected_dim,):
                raise ValueError(f"{topic} message has shape {row.shape}, expected ({expected_dim},)")
            rows.append(row)
            times.append(log_time)

    with mcap_path.open("rb") as f:
        data = f.read()
    if not data.startswith(b"\x89MCAP0\r\n"):
        raise ValueError(f"{mcap_path} is not an MCAP file")

    for opcode, payload in _mcap_iter_records(data[8:]):
        if opcode == 0x06:  # Chunk
            for chunk_opcode, chunk_payload in _mcap_iter_records(_mcap_parse_chunk(payload)):
                handle_record(chunk_opcode, chunk_payload)
        else:
            handle_record(opcode, payload)

    if not rows:
        raise ValueError(f"No messages found on {topic} in {mcap_path}")
    return np.stack(rows, axis=0), np.asarray(times, dtype=np.int64)


def _stats(values: np.ndarray) -> dict[str, float]:
    abs_values = np.abs(values)
    return {
        "mean_abs": float(abs_values.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "p95_abs": float(np.percentile(abs_values, 95)),
        "max_abs": float(abs_values.max()),
    }


def _component_for_dim(dim: int, slices: dict[str, list[int]]) -> str:
    for name, (start, end) in slices.items():
        if start <= dim < end:
            return name
    return "unknown"


def _align_ranges(component: str, slices: dict[str, list[int]]) -> list[tuple[int, int]]:
    if component == "all":
        return [(0, max(end for _, end in slices.values()))]
    if component in slices:
        return [tuple(slices[component])]
    if component in STATE_SUBSLICES:
        if "state" not in slices:
            raise ValueError(f"Cannot align on {component!r}: actor_obs has no state slice")
        state_start, state_end = slices["state"]
        if state_end - state_start != 52:
            raise ValueError(f"Cannot align on {component!r}: expected 52-D state, got {state_end - state_start}")
        rel_start, rel_end = STATE_SUBSLICES[component]
        return [(state_start + rel_start, state_start + rel_end)]
    if component == "imu":
        return _align_ranges("projected_gravity", slices) + _align_ranges("base_ang_vel", slices)
    raise ValueError(
        f"Unsupported align_component {component!r}. "
        "Expected one of: all, state, dof_pos, dof_vel, projected_gravity, base_ang_vel, imu, last_action, history_actor, z"
    )


def _slice_ranges(array: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    if len(ranges) == 1:
        start, end = ranges[0]
        return array[:, start:end]
    return np.concatenate([array[:, start:end] for start, end in ranges], axis=-1)


def _compare_actor_obs(
    sim_actor_obs: np.ndarray,
    mcap_actor_obs: np.ndarray,
    slices: dict[str, list[int]],
    sim_start: int,
    mcap_start: int,
    align_window: int,
    align_prefix: int,
    top_k: int,
    align_component: str,
) -> dict[str, Any]:
    if sim_actor_obs.ndim != 2 or mcap_actor_obs.ndim != 2:
        raise ValueError("Expected sim and MCAP actor_obs to be 2D arrays")
    if sim_actor_obs.shape[1] != mcap_actor_obs.shape[1]:
        raise ValueError(f"Actor obs dims differ: sim {sim_actor_obs.shape[1]} vs mcap {mcap_actor_obs.shape[1]}")

    best_mcap_start = mcap_start
    align_scan = []
    if align_window > 0:
        best_score = float("inf")
        align_ranges = _align_ranges(align_component, slices)
        scan_start = max(0, mcap_start - align_window)
        scan_end = min(mcap_start + align_window + 1, len(mcap_actor_obs))
        for candidate in range(scan_start, scan_end):
            length = min(align_prefix, len(sim_actor_obs) - sim_start, len(mcap_actor_obs) - candidate)
            if length <= 0:
                continue
            sim_align = _slice_ranges(sim_actor_obs[sim_start : sim_start + length], align_ranges)
            mcap_align = _slice_ranges(mcap_actor_obs[candidate : candidate + length], align_ranges)
            diff = sim_align - mcap_align
            score = float(np.sqrt(np.mean(np.square(diff))))
            align_scan.append({"mcap_start": candidate, "rmse": score})
            if score < best_score:
                best_score = score
                best_mcap_start = candidate

    compare_len = min(len(sim_actor_obs) - sim_start, len(mcap_actor_obs) - best_mcap_start)
    if compare_len <= 0:
        raise ValueError(
            f"No overlapping samples after starts: sim_start={sim_start}, mcap_start={best_mcap_start}"
        )

    sim_slice = sim_actor_obs[sim_start : sim_start + compare_len]
    mcap_slice = mcap_actor_obs[best_mcap_start : best_mcap_start + compare_len]
    diff = sim_slice - mcap_slice

    component_stats = {}
    for name, (start, end) in slices.items():
        component_stats[name] = _stats(diff[:, start:end])

    dim_rmse = np.sqrt(np.mean(np.square(diff), axis=0))
    top_dims = []
    for dim in np.argsort(-dim_rmse)[:top_k]:
        top_dims.append(
            {
                "dim": int(dim),
                "component": _component_for_dim(int(dim), slices),
                "rmse": float(dim_rmse[dim]),
                "mean_abs": float(np.mean(np.abs(diff[:, dim]))),
                "max_abs": float(np.max(np.abs(diff[:, dim]))),
            }
        )

    return {
        "sim_shape": list(sim_actor_obs.shape),
        "mcap_shape": list(mcap_actor_obs.shape),
        "sim_start": sim_start,
        "mcap_start": best_mcap_start,
        "compare_len": compare_len,
        "slices": slices,
        "align_component": align_component,
        "align_prefix": align_prefix,
        "align_window": align_window,
        "alignment_scan": align_scan,
        "overall": _stats(diff),
        "components": component_stats,
        "top_dims_by_rmse": top_dims,
    }


def _write_component_csv(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for name, stats in report["components"].items():
        rows.append({"component": name, **stats})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "mean_abs", "rmse", "p95_abs", "max_abs"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def _write_alignment_scan(path: Path, report: dict[str, Any]) -> None:
    scan = report.get("alignment_scan") or []
    if not scan:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mcap_start", "rmse"])
        writer.writeheader()
        writer.writerows(scan)
    print(f"Saved {path}")


def _plot_alignment_scan(report: dict[str, Any], output_prefix: Path) -> None:
    scan = report.get("alignment_scan") or []
    if not scan:
        return

    import matplotlib.pyplot as plt

    starts = np.asarray([row["mcap_start"] for row in scan])
    scores = np.asarray([row["rmse"] for row in scan])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(starts, scores, linewidth=1.2)
    ax.axvline(report["mcap_start"], color="tab:red", linestyle="--", linewidth=1.0, label=f"best={report['mcap_start']}")
    ax.set_xlabel("mcap_start")
    ax.set_ylabel("alignment RMSE")
    ax.set_title(f"Alignment scan on {report.get('align_component', 'unknown')}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_prefix.with_name(output_prefix.name + "_alignment_scan.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")


def _set_timestep_axes(ax, sim_start: int, mcap_start: int) -> None:
    ax.set_xlabel("sim timestep")
    offset = mcap_start - sim_start

    def sim_to_mcap(x):
        return x + offset

    def mcap_to_sim(x):
        return x - offset

    secax = ax.secondary_xaxis("top", functions=(sim_to_mcap, mcap_to_sim))
    secax.set_xlabel("mcap timestep")


def _plot_actor_obs_comparison(
    sim_actor_obs: np.ndarray,
    mcap_actor_obs: np.ndarray,
    report: dict[str, Any],
    output_prefix: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_prefix.parent / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_alignment_scan(report, output_prefix)

    sim_start = int(report["sim_start"])
    mcap_start = int(report["mcap_start"])
    compare_len = int(report["compare_len"])
    slices = report["slices"]
    sim = sim_actor_obs[sim_start : sim_start + compare_len]
    mcap = mcap_actor_obs[mcap_start : mcap_start + compare_len]
    diff = sim - mcap
    steps = np.arange(sim_start, sim_start + compare_len)

    component_names = list(slices.keys())
    rmse_values = [report["components"][name]["rmse"] for name in component_names]
    mean_abs_values = [report["components"][name]["mean_abs"] for name in component_names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(component_names))
    width = 0.38
    ax.bar(x - width / 2, rmse_values, width, label="RMSE")
    ax.bar(x + width / 2, mean_abs_values, width, label="Mean abs")
    ax.set_xticks(x, component_names, rotation=20, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("Actor obs component errors")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_prefix.with_name(output_prefix.name + "_component_bar.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, (start, end) in slices.items():
        component_rms = np.sqrt(np.mean(np.square(diff[:, start:end]), axis=1))
        ax.plot(steps, component_rms, label=name, linewidth=1.2)
    _set_timestep_axes(ax, sim_start=sim_start, mcap_start=mcap_start)
    ax.set_ylabel("RMS error")
    ax.set_title("Actor obs component RMS error over time")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    path = output_prefix.with_name(output_prefix.name + "_component_timeseries.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")

    top_dims = [item["dim"] for item in report["top_dims_by_rmse"][: min(8, len(report["top_dims_by_rmse"]))]]
    if top_dims:
        fig, axes = plt.subplots(len(top_dims), 1, figsize=(12, max(2.0 * len(top_dims), 3.5)), sharex=True)
        if len(top_dims) == 1:
            axes = [axes]
        for ax, dim in zip(axes, top_dims):
            ax.plot(steps, sim[:, dim], label="sim", linewidth=1.0)
            ax.plot(steps, mcap[:, dim], label="mcap", linewidth=1.0, alpha=0.75)
            ax.set_ylabel(f"d{dim}")
            ax.grid(alpha=0.25)
            ax.text(
                0.995,
                0.78,
                _component_for_dim(dim, slices),
                ha="right",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
        axes[0].legend(loc="upper left", ncol=2)
        _set_timestep_axes(axes[-1], sim_start=sim_start, mcap_start=mcap_start)
        fig.suptitle("Top RMSE actor_obs dims: sim vs mcap", y=0.995)
        fig.tight_layout()
        path = output_prefix.with_name(output_prefix.name + "_top_dims_overlay.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Saved {path}")

    abs_diff = np.abs(diff)
    max_plot_steps = 1200
    if abs_diff.shape[0] > max_plot_steps:
        idx = np.linspace(0, abs_diff.shape[0] - 1, max_plot_steps).astype(np.int64)
        heatmap = abs_diff[idx]
        x_label = f"sim timestep, downsampled from {abs_diff.shape[0]} samples"
    else:
        heatmap = abs_diff
        x_label = "sim timestep"
    vmax = float(np.percentile(heatmap, 99))
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(
        heatmap.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax,
        extent=[sim_start, sim_start + compare_len - 1, 0, heatmap.shape[1] - 1],
    )
    for name, (start, end) in slices.items():
        ax.axhline(start, color="white", linewidth=0.6, alpha=0.7)
        ax.text(3, (start + end) / 2, name, color="white", va="center", fontsize=8)
    ax.set_xlabel(x_label)
    _set_timestep_axes(ax, sim_start=sim_start, mcap_start=mcap_start)
    ax.set_ylabel("actor_obs dim")
    ax.set_title("|sim - mcap| actor_obs heatmap")
    fig.colorbar(image, ax=ax, label="abs error")
    fig.tight_layout()
    path = output_prefix.with_name(output_prefix.name + "_abs_error_heatmap.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")


def _state_dim_labels() -> list[str]:
    labels = []
    labels.extend([f"dof_pos/{name}" for name in PIPLUS_SIM_JOINT_NAMES])
    labels.extend([f"dof_vel/{name}" for name in PIPLUS_SIM_JOINT_NAMES])
    labels.extend(["projected_gravity/x", "projected_gravity/y", "projected_gravity/z"])
    labels.extend(["base_ang_vel/x", "base_ang_vel/y", "base_ang_vel/z"])
    return labels


def _write_state_dim_csv(path: Path, state_diff: np.ndarray) -> None:
    labels = _state_dim_labels()
    rows = []
    for dim, label in enumerate(labels):
        values = state_diff[:, dim]
        rows.append(
            {
                "state_dim": dim,
                "name": label,
                **_stats(values),
            }
        )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["state_dim", "name", "mean_abs", "rmse", "p95_abs", "max_abs"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def _plot_state_group_overlay(
    sim_state: np.ndarray,
    mcap_state: np.ndarray,
    dims: list[int],
    labels: list[str],
    title: str,
    path: Path,
    sim_start: int,
    mcap_start: int,
) -> None:
    import matplotlib.pyplot as plt

    steps = np.arange(sim_start, sim_start + sim_state.shape[0])
    fig, axes = plt.subplots(len(dims), 1, figsize=(13, max(2.0 * len(dims), 4.0)), sharex=True)
    if len(dims) == 1:
        axes = [axes]
    for ax, dim, label in zip(axes, dims, labels):
        diff = sim_state[:, dim] - mcap_state[:, dim]
        stats = _stats(diff)
        ax.plot(steps, sim_state[:, dim], label="sim", linewidth=0.9)
        ax.plot(steps, mcap_state[:, dim], label="mcap", linewidth=0.9, alpha=0.75)
        ax.text(
            0.99,
            0.92,
            f"{label}\nrmse={stats['rmse']:.4g}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85, "boxstyle": "round,pad=0.2"},
        )
        ax.set_ylabel("value", fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper left", ncol=2)
    _set_timestep_axes(axes[-1], sim_start=sim_start, mcap_start=mcap_start)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def _safe_plot_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)


def _plot_state_individual_overlays(
    sim_state: np.ndarray,
    mcap_state: np.ndarray,
    dims: list[int],
    labels: list[str],
    output_dir: Path,
    sim_start: int,
    mcap_start: int,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    steps = np.arange(sim_start, sim_start + sim_state.shape[0])
    for dim in dims:
        label = labels[dim]
        diff = sim_state[:, dim] - mcap_state[:, dim]
        stats = _stats(diff)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(steps, sim_state[:, dim], label="sim", linewidth=1.0)
        ax.plot(steps, mcap_state[:, dim], label="mcap", linewidth=1.0, alpha=0.8)
        ax.text(
            0.99,
            0.95,
            f"{label}\nrmse={stats['rmse']:.4g}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85, "boxstyle": "round,pad=0.25"},
        )
        _set_timestep_axes(ax, sim_start=sim_start, mcap_start=mcap_start)
        ax.set_ylabel("value")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", ncol=2)
        fig.tight_layout()
        path = output_dir / f"state_dim_{dim:02d}_{_safe_plot_name(label)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
    print(f"Saved {len(dims)} state item plots under {output_dir}")


def _plot_state_detailed_comparison(
    sim_actor_obs: np.ndarray,
    mcap_actor_obs: np.ndarray,
    report: dict[str, Any],
    output_prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    slices = report["slices"]
    if "state" not in slices:
        return

    state_start, state_end = slices["state"]
    if state_end - state_start != 52:
        print(f"Skip state detailed plots: expected 52-D state, got {state_end - state_start}")
        return

    sim_start = int(report["sim_start"])
    mcap_start = int(report["mcap_start"])
    compare_len = int(report["compare_len"])
    sim_state = sim_actor_obs[sim_start : sim_start + compare_len, state_start:state_end]
    mcap_state = mcap_actor_obs[mcap_start : mcap_start + compare_len, state_start:state_end]
    state_diff = sim_state - mcap_state
    labels = _state_dim_labels()

    _write_state_dim_csv(output_prefix.with_name(output_prefix.name + "_state_dim_errors.csv"), state_diff)

    group_rmse = {
        name: np.sqrt(np.mean(np.square(state_diff[:, start:end]), axis=0))
        for name, (start, end) in STATE_SUBSLICES.items()
    }

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    joint_x = np.arange(len(PIPLUS_SIM_JOINT_NAMES))
    width = 0.38
    axes[0].bar(joint_x - width / 2, group_rmse["dof_pos"], width, label="dof_pos")
    axes[0].bar(joint_x + width / 2, group_rmse["dof_vel"], width, label="dof_vel")
    axes[0].set_xticks(joint_x, PIPLUS_SIM_JOINT_NAMES, rotation=70, ha="right", fontsize=8)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("State per-joint RMSE")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    small_names = ["projected_gravity/x", "projected_gravity/y", "projected_gravity/z", "base_ang_vel/x", "base_ang_vel/y", "base_ang_vel/z"]
    small_rmse = np.concatenate([group_rmse["projected_gravity"], group_rmse["base_ang_vel"]])
    axes[1].bar(np.arange(len(small_names)), small_rmse)
    axes[1].set_xticks(np.arange(len(small_names)), small_names, rotation=20, ha="right")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("State IMU RMSE")
    axes[1].grid(axis="y", alpha=0.3)

    path = output_prefix.with_name(output_prefix.name + "_state_rmse_bar.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")

    grouped_pages = [
        ("dof_pos", list(range(0, 23))),
        ("dof_vel", list(range(23, 46))),
        ("projected_gravity", list(range(46, 49))),
        ("base_ang_vel", list(range(49, 52))),
    ]
    for suffix, dims in grouped_pages:
        _plot_state_group_overlay(
            sim_state=sim_state,
            mcap_state=mcap_state,
            dims=dims,
            labels=[labels[dim] for dim in dims],
            title=f"State {suffix}: sim vs mcap",
            path=output_prefix.with_name(output_prefix.name + f"_state_{suffix}.png"),
            sim_start=sim_start,
            mcap_start=mcap_start,
        )

    top_dims = np.argsort(-np.sqrt(np.mean(np.square(state_diff), axis=0)))[:12].tolist()
    _plot_state_group_overlay(
        sim_state=sim_state,
        mcap_state=mcap_state,
        dims=top_dims,
        labels=[labels[dim] for dim in top_dims],
        title="State top RMSE dims: sim vs mcap",
        path=output_prefix.with_name(output_prefix.name + "_state_top_dims.png"),
        sim_start=sim_start,
        mcap_start=mcap_start,
    )

    heatmap = np.abs(state_diff)
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(
        heatmap.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=float(np.percentile(heatmap, 99)),
        extent=[sim_start, sim_start + compare_len - 1, 0, heatmap.shape[1] - 1],
    )
    for name, (start, end) in STATE_SUBSLICES.items():
        ax.axhline(start, color="white", linewidth=0.6, alpha=0.7)
        ax.text(3, (start + end) / 2, name, color="white", va="center", fontsize=8)
    _set_timestep_axes(ax, sim_start=sim_start, mcap_start=mcap_start)
    ax.set_ylabel("state dim")
    ax.set_title("|sim - mcap| state heatmap")
    fig.colorbar(image, ax=ax, label="abs error")
    fig.tight_layout()
    path = output_prefix.with_name(output_prefix.name + "_state_abs_error_heatmap.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_actor_subvector_group(
    sim_actor_obs: np.ndarray,
    mcap_actor_obs: np.ndarray,
    report: dict[str, Any],
    component: str,
    labels: list[str],
    output_prefix: Path,
    page_size: int | None = None,
) -> None:
    slices = report["slices"]
    if component not in slices:
        return

    start, end = slices[component]
    dim = end - start
    if len(labels) != dim:
        labels = [f"{component}/{i:03d}" for i in range(dim)]

    sim_start = int(report["sim_start"])
    mcap_start = int(report["mcap_start"])
    compare_len = int(report["compare_len"])
    sim_values = sim_actor_obs[sim_start : sim_start + compare_len, start:end]
    mcap_values = mcap_actor_obs[mcap_start : mcap_start + compare_len, start:end]

    if page_size is None or dim <= page_size:
        _plot_state_group_overlay(
            sim_state=sim_values,
            mcap_state=mcap_values,
            dims=list(range(dim)),
            labels=labels,
            title=f"{component}: sim vs mcap",
            path=output_prefix.with_name(output_prefix.name + f"_{component}.png"),
            sim_start=sim_start,
            mcap_start=mcap_start,
        )
    else:
        for page_idx, page_start in enumerate(range(0, dim, page_size)):
            page_end = min(page_start + page_size, dim)
            _plot_state_group_overlay(
                sim_state=sim_values,
                mcap_state=mcap_values,
                dims=list(range(page_start, page_end)),
                labels=labels[page_start:page_end],
                title=f"{component} dims {page_start}-{page_end - 1}: sim vs mcap",
                path=output_prefix.with_name(output_prefix.name + f"_{component}_page_{page_idx:02d}.png"),
                sim_start=sim_start,
                mcap_start=mcap_start,
            )

    diff = sim_values - mcap_values
    top_dims = np.argsort(-np.sqrt(np.mean(np.square(diff), axis=0)))[: min(16, dim)].tolist()
    _plot_state_group_overlay(
        sim_state=sim_values,
        mcap_state=mcap_values,
        dims=top_dims,
        labels=[labels[i] for i in top_dims],
        title=f"{component} top RMSE dims: sim vs mcap",
        path=output_prefix.with_name(output_prefix.name + f"_{component}_top_dims.png"),
        sim_start=sim_start,
        mcap_start=mcap_start,
    )


def _plot_last_action_and_z_comparison(
    sim_actor_obs: np.ndarray,
    mcap_actor_obs: np.ndarray,
    report: dict[str, Any],
    output_prefix: Path,
) -> None:
    action_labels = [f"last_action/{name}" for name in PIPLUS_SIM_JOINT_NAMES]
    _plot_actor_subvector_group(
        sim_actor_obs=sim_actor_obs,
        mcap_actor_obs=mcap_actor_obs,
        report=report,
        component="last_action",
        labels=action_labels,
        output_prefix=output_prefix,
        page_size=None,
    )

    z_start, z_end = report["slices"].get("z", [0, 0])
    z_labels = [f"z/{i:03d}" for i in range(z_end - z_start)]
    _plot_actor_subvector_group(
        sim_actor_obs=sim_actor_obs,
        mcap_actor_obs=mcap_actor_obs,
        report=report,
        component="z",
        labels=z_labels,
        output_prefix=output_prefix,
        page_size=16,
    )


def _load_sim_actor_obs_npz(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        actor_obs = np.asarray(data["actor_obs"], dtype=np.float32)
        metadata: dict[str, Any] = {}
        if "metadata_json" in data:
            metadata = json.loads(str(data["metadata_json"].item()))
    if actor_obs.ndim != 2:
        raise ValueError(f"{path} actor_obs must be 2D, got {actor_obs.shape}")
    return actor_obs, metadata


def _fallback_slices_for_dim(dim: int) -> dict[str, list[int]]:
    if dim == 631:
        return {
            "state": [0, 52],
            "last_action": [52, 75],
            "history_actor": [75, 375],
            "z": [375, 631],
        }
    raise ValueError(f"No fallback actor_obs slices for dim {dim}; save metadata_json or pass a known 631-D BFM actor_obs")


def compare_saved_actor_obs(
    sim_actor_obs_path: Path,
    mcap_path: Path = DEFAULT_MCAP_PATH,
    mcap_topic: str = "/actor_obs",
    output_dir: Path | None = None,
    sim_start: int = 0,
    mcap_start: int = 0,
    align_window: int = 0,
    align_prefix: int = 100,
    top_k: int = 20,
    plot: bool = True,
    align_component: str = "all",
):
    """Compare a previously saved sim_actor_obs_motion_*.npz against MCAP /actor_obs."""

    sim_actor_obs_path = _resolve_path(sim_actor_obs_path)
    mcap_path = _resolve_path(mcap_path)
    output_dir = output_dir.expanduser() if output_dir is not None else sim_actor_obs_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_actor_obs, metadata = _load_sim_actor_obs_npz(sim_actor_obs_path)
    slices = metadata.get("slices") or _fallback_slices_for_dim(sim_actor_obs.shape[1])
    mcap_actor_obs, mcap_times_ns = _read_mcap_actor_obs(
        mcap_path=mcap_path,
        topic=mcap_topic,
        expected_dim=sim_actor_obs.shape[1],
    )
    _save_npz(
        output_dir / "mcap_actor_obs.npz",
        actor_obs=mcap_actor_obs,
        log_time_ns=mcap_times_ns,
        source=np.asarray(str(mcap_path)),
        topic=np.asarray(mcap_topic),
    )

    report = _compare_actor_obs(
        sim_actor_obs=sim_actor_obs,
        mcap_actor_obs=mcap_actor_obs,
        slices=slices,
        sim_start=sim_start,
        mcap_start=mcap_start,
        align_window=align_window,
        align_prefix=align_prefix,
        top_k=top_k,
        align_component=align_component,
    )
    stem = sim_actor_obs_path.stem.replace("sim_actor_obs_", "actor_obs_compare_")
    stem = f"{stem}_align_{align_component}"
    report_path = output_dir / f"{stem}.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {report_path}")
    _write_component_csv(output_dir / f"{stem}_components.csv", report)
    _write_alignment_scan(output_dir / f"{stem}_alignment_scan.csv", report)
    if plot:
        _plot_actor_obs_comparison(
            sim_actor_obs=sim_actor_obs,
            mcap_actor_obs=mcap_actor_obs,
            report=report,
            output_prefix=output_dir / stem,
        )
        _plot_state_detailed_comparison(
            sim_actor_obs=sim_actor_obs,
            mcap_actor_obs=mcap_actor_obs,
            report=report,
            output_prefix=output_dir / stem,
        )
        _plot_last_action_and_z_comparison(
            sim_actor_obs=sim_actor_obs,
            mcap_actor_obs=mcap_actor_obs,
            report=report,
            output_prefix=output_dir / stem,
        )

    overall = report["overall"]
    print(
        f"Comparison: len={report['compare_len']}, sim_start={report['sim_start']}, "
        f"mcap_start={report['mcap_start']}, rmse={overall['rmse']:.6g}, "
        f"mean_abs={overall['mean_abs']:.6g}, p95_abs={overall['p95_abs']:.6g}, "
        f"max_abs={overall['max_abs']:.6g}"
    )
    for name, stats in report["components"].items():
        print(
            f"  {name}: rmse={stats['rmse']:.6g}, mean_abs={stats['mean_abs']:.6g}, "
            f"p95_abs={stats['p95_abs']:.6g}, max_abs={stats['max_abs']:.6g}"
        )


def main(
    model_folder: Path | None = None,
    data_path: Path | None = None,
    headless: bool = True,
    device: str = "cuda",
    simulator: str = "isaacsim",
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    motion_list: list[int] = [25],
    robot: str | None = None,
    episode_len: int | None = 2000,
    no_training_config: bool = False,
    mcap_path: Path | None = DEFAULT_MCAP_PATH,
    mcap_topic: str = "/actor_obs",
    output_dir: Path | None = None,
    sim_start: int = 0,
    mcap_start: int = 0,
    align_window: int = 0,
    align_prefix: int = 100,
    top_k: int = 20,
    sim_actor_obs_path: Path | None = None,
    plot: bool = True,
    align_component: str = "all",
):
    """Record simulation actor_obs and compare it with a ROS2 MCAP /actor_obs topic."""

    if sim_actor_obs_path is not None:
        compare_saved_actor_obs(
            sim_actor_obs_path=sim_actor_obs_path,
            mcap_path=DEFAULT_MCAP_PATH if mcap_path is None else mcap_path,
            mcap_topic=mcap_topic,
            output_dir=output_dir,
            sim_start=sim_start,
            mcap_start=mcap_start,
            align_window=align_window,
            align_prefix=align_prefix,
            top_k=top_k,
            plot=plot,
            align_component=align_component,
        )
        return

    if model_folder is None:
        raise ValueError("model_folder is required unless sim_actor_obs_path is provided")

    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
    from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
    from humanoidverse.utils.helpers import get_backward_observation

    model_folder = _resolve_path(model_folder)
    simulator = simulator.lower()
    output_dir = output_dir.expanduser() if output_dir is not None else model_folder / "actor_obs_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    actor_keys = _actor_input_keys(model)
    z_dim = int(model.cfg.archi.z_dim)
    print(f"Actor input keys: {actor_keys}; z_dim={z_dim}")

    config, _ = _load_config_for_inference(
        model_folder=model_folder,
        data_path=data_path,
        headless=headless,
        device=device,
        simulator=simulator,
        disable_dr=disable_dr,
        disable_obs_noise=disable_obs_noise,
        robot=robot,
        no_training_config=no_training_config,
    )
    use_root_height_obs = config["env"].get("root_height_obs", False)

    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env

    mcap_actor_obs = None
    mcap_times_ns = None
    if mcap_path is not None:
        mcap_path = _resolve_path(mcap_path)
        if not mcap_path.exists():
            raise FileNotFoundError(f"MCAP file not found: {mcap_path}")

    for motion_id in motion_list:
        env.set_is_evaluating(motion_id)
        loaded_motion_ids = env._motion_lib._curr_motion_ids.detach().cpu().tolist()
        loaded_motion_keys = env._motion_lib.curr_motion_keys
        print(f"Requested motion id {motion_id}; loaded ids {loaded_motion_ids}; loaded keys {loaded_motion_keys}")

        obs, obs_dict = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs)
        z = _tracking_inference(model, tree_map(lambda x: x[1:], obs))
        joblib.dump(z.cpu().numpy(), output_dir / f"zs_{motion_id}.pkl")
        print(f"Saved {output_dir / f'zs_{motion_id}.pkl'}")

        observation = _initialize_to_motion(wrapped_env, obs_dict, simulator=simulator, num_envs=num_envs)
        current_episode_len = episode_len if episode_len is not None else z.shape[0]
        if current_episode_len > z.shape[0]:
            print(f"Requested {current_episode_len} steps; cycling {z.shape[0]} inferred latent steps")

        records = []
        z_records = []
        slices = None
        print(f"Recording simulation actor_obs for motion {motion_id} for {current_episode_len} steps")
        for i in range(current_episode_len):
            z_batch = z[i % len(z)].repeat(num_envs, 1)
            actor_obs, slices = _build_actor_obs(observation, z_batch, actor_keys)
            records.append(actor_obs[0].detach().cpu().numpy().astype(np.float32))
            z_records.append(z_batch[0].detach().cpu().numpy().astype(np.float32))

            action = model.act(observation, z_batch, mean=True)
            observation, _, _, _, _ = wrapped_env.step(action, to_numpy=False)

        assert slices is not None
        sim_actor_obs = np.stack(records, axis=0)
        sim_z = np.stack(z_records, axis=0)
        metadata = {
            "motion_id": motion_id,
            "actor_keys": actor_keys,
            "slices": slices,
            "model_folder": str(model_folder),
            "data_path": str(data_path) if data_path is not None else None,
            "simulator": simulator,
            "robot": robot,
        }
        sim_path = output_dir / f"sim_actor_obs_motion_{motion_id}.npz"
        _save_npz(
            sim_path,
            actor_obs=sim_actor_obs,
            z=sim_z,
            metadata_json=np.asarray(json.dumps(metadata, indent=2)),
        )

        if mcap_path is None:
            continue

        if mcap_actor_obs is None:
            mcap_actor_obs, mcap_times_ns = _read_mcap_actor_obs(
                mcap_path=mcap_path,
                topic=mcap_topic,
                expected_dim=sim_actor_obs.shape[1],
            )
            _save_npz(
                output_dir / "mcap_actor_obs.npz",
                actor_obs=mcap_actor_obs,
                log_time_ns=mcap_times_ns,
                source=np.asarray(str(mcap_path)),
                topic=np.asarray(mcap_topic),
            )

        report = _compare_actor_obs(
            sim_actor_obs=sim_actor_obs,
            mcap_actor_obs=mcap_actor_obs,
            slices=slices,
            sim_start=sim_start,
            mcap_start=mcap_start,
            align_window=align_window,
            align_prefix=align_prefix,
            top_k=top_k,
            align_component=align_component,
        )
        stem = f"actor_obs_compare_motion_{motion_id}_align_{align_component}"
        report_path = output_dir / f"{stem}.json"
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved {report_path}")
        _write_component_csv(output_dir / f"{stem}_components.csv", report)
        _write_alignment_scan(output_dir / f"{stem}_alignment_scan.csv", report)
        if plot:
            _plot_actor_obs_comparison(
                sim_actor_obs=sim_actor_obs,
                mcap_actor_obs=mcap_actor_obs,
                report=report,
                output_prefix=output_dir / stem,
            )
            _plot_state_detailed_comparison(
                sim_actor_obs=sim_actor_obs,
                mcap_actor_obs=mcap_actor_obs,
                report=report,
                output_prefix=output_dir / stem,
            )
            _plot_last_action_and_z_comparison(
                sim_actor_obs=sim_actor_obs,
                mcap_actor_obs=mcap_actor_obs,
                report=report,
                output_prefix=output_dir / stem,
            )

        overall = report["overall"]
        print(
            f"Comparison motion {motion_id}: len={report['compare_len']}, "
            f"sim_start={report['sim_start']}, mcap_start={report['mcap_start']}, "
            f"rmse={overall['rmse']:.6g}, mean_abs={overall['mean_abs']:.6g}, "
            f"p95_abs={overall['p95_abs']:.6g}, max_abs={overall['max_abs']:.6g}"
        )
        for name, stats in report["components"].items():
            print(
                f"  {name}: rmse={stats['rmse']:.6g}, mean_abs={stats['mean_abs']:.6g}, "
                f"p95_abs={stats['p95_abs']:.6g}, max_abs={stats['max_abs']:.6g}"
            )


if __name__ == "__main__":
    try:
        import tyro

        tyro.cli(main)
    except ModuleNotFoundError as exc:
        if exc.name != "tyro":
            raise

        import argparse

        parser = argparse.ArgumentParser(description=main.__doc__)
        parser.add_argument("--sim-actor-obs-path", type=Path, default=None)
        parser.add_argument("--model_folder", "--model-folder", type=Path, default=None)
        parser.add_argument("--data_path", "--data-path", type=Path, default=None)
        parser.add_argument("--headless", dest="headless", action="store_true", default=True)
        parser.add_argument("--no-headless", dest="headless", action="store_false")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--simulator", default="isaacsim")
        parser.add_argument("--disable-dr", action="store_true")
        parser.add_argument("--disable-obs-noise", action="store_true")
        parser.add_argument("--motion-list", type=int, nargs="+", default=[25])
        parser.add_argument("--robot", default=None)
        parser.add_argument("--episode-len", type=int, default=2000)
        parser.add_argument("--no-training-config", action="store_true")
        parser.add_argument("--mcap-path", type=Path, default=DEFAULT_MCAP_PATH)
        parser.add_argument("--mcap-topic", default="/actor_obs")
        parser.add_argument("--output-dir", type=Path, default=None)
        parser.add_argument("--sim-start", type=int, default=0)
        parser.add_argument("--mcap-start", type=int, default=0)
        parser.add_argument("--align-window", type=int, default=0)
        parser.add_argument("--align-prefix", type=int, default=100)
        parser.add_argument("--top-k", type=int, default=20)
        parser.add_argument("--plot", dest="plot", action="store_true", default=True)
        parser.add_argument("--no-plot", dest="plot", action="store_false")
        parser.add_argument("--align-component", default="all")
        args = parser.parse_args()
        if args.sim_actor_obs_path is not None:
            compare_saved_actor_obs(
                sim_actor_obs_path=args.sim_actor_obs_path,
                mcap_path=args.mcap_path,
                mcap_topic=args.mcap_topic,
                output_dir=args.output_dir,
                sim_start=args.sim_start,
                mcap_start=args.mcap_start,
                align_window=args.align_window,
                align_prefix=args.align_prefix,
                top_k=args.top_k,
                plot=args.plot,
                align_component=args.align_component,
            )
        else:
            main(
                model_folder=args.model_folder,
                data_path=args.data_path,
                headless=args.headless,
                device=args.device,
                simulator=args.simulator,
                disable_dr=args.disable_dr,
                disable_obs_noise=args.disable_obs_noise,
                motion_list=args.motion_list,
                robot=args.robot,
                episode_len=args.episode_len,
                no_training_config=args.no_training_config,
                mcap_path=args.mcap_path,
                mcap_topic=args.mcap_topic,
                output_dir=args.output_dir,
                sim_start=args.sim_start,
                mcap_start=args.mcap_start,
                align_window=args.align_window,
                align_prefix=args.align_prefix,
                top_k=args.top_k,
                plot=args.plot,
                align_component=args.align_component,
            )
