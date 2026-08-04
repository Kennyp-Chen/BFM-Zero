import os

os.environ.setdefault("MUJOCO_GL", "egl")  # Default to EGL for offscreen rendering.
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import json
import torch
import joblib
import numpy as np
from torch.utils._pytree import tree_map
from tqdm import tqdm

import humanoidverse

# Resolve humanoidverse root directory
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).parent.parent.parent


ROBOT_CONFIG_OVERRIDES = {
    "g1": "robot=g1/g1_29dof_hard_waist",
    "PiPlus_S_12L8A0G2H1W_LSE": "robot=piplus/PiPlus_S_12L8A0G2H1W_LSE",
    "piplus_lse": "robot=piplus/PiPlus_S_12L8A0G2H1W_LSE",
    "PiPlus_S_12L8A0G2H0W": "robot=piplus/PiPlus_S_12L8A0G2H0W",
    "piplus_h0w": "robot=piplus/PiPlus_S_12L8A0G2H0W",
    "Hi_P_12L10A0G2H1W_260402": "robot=Hi/Hi_P_12L10A0G2H1W_260402",
    "h1_260402": "robot=Hi/Hi_P_12L10A0G2H1W_260402",
}

PIPLUS_LSE_ROBOTS = {"PiPlus_S_12L8A0G2H1W_LSE", "piplus_lse"}
PIPLUS_H0W_ROBOTS = {"PiPlus_S_12L8A0G2H0W", "piplus_h0w"}
PIPLUS_ROBOTS = PIPLUS_LSE_ROBOTS | PIPLUS_H0W_ROBOTS
H1_260402_ROBOTS = {"Hi_P_12L10A0G2H1W_260402", "h1_260402"}
H1_260402_SIM_XML = "package://ht_urdf/Hi_P_12L10A0G2H1W_260402/xml/Hi_P_12L10A0G2H1W_Simplify_260402_with_armature.xml"


def _resolve_path(path: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.exists():
        return expanded.resolve()

    path_text = str(expanded)
    marker = "humanoidverse/"
    if marker in path_text:
        repo_relative = HUMANOIDVERSE_DIR / path_text.split(marker, 1)[1]
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


def _default_goal_json_paths(robot: str | None) -> list[Path]:
    if robot in PIPLUS_ROBOTS | H1_260402_ROBOTS:
        return [
            HUMANOIDVERSE_DIR / "data" / "robots" / "piplus" / "goal_frames_piplus_lse_lafan.json",
            HUMANOIDVERSE_DIR / "data" / "goal_frames_piplus_lse_lafan.json",
        ]
    return [
        HUMANOIDVERSE_DIR / "data" / "robots" / "g1" / "goal_frames_lafan29dof.json",
        HUMANOIDVERSE_DIR / "data" / "goal_frames_lafan29dof.json",
    ]


def _load_goal_frames(goal_json: Path | None, robot: str | None) -> tuple[list[dict] | None, str | None]:
    if goal_json is not None:
        resolved_goal_json = _resolve_path(goal_json)
        if not resolved_goal_json.exists():
            raise FileNotFoundError(f"Goal JSON not found: {goal_json} (resolved to {resolved_goal_json})")
        with open(resolved_goal_json, "r") as f:
            return json.load(f), str(resolved_goal_json)

    searched_paths = _default_goal_json_paths(robot)
    for path in searched_paths:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f), str(path)

    return None, None


def _motion_name_for_goal(env, motion_id: int) -> str:
    motion_keys = getattr(env._motion_lib, "_motion_data_keys", [])
    if motion_id < len(motion_keys):
        return Path(str(motion_keys[motion_id])).stem
    return f"motion_{motion_id}"


def _goals_from_motion_lib(env) -> list[dict]:
    return [
        {
            "motion_id": motion_id,
            "frames": None,
            "motion_name": _motion_name_for_goal(env, motion_id),
        }
        for motion_id in range(env._motion_lib._num_unique_motions)
    ]


def _default_goal_frames(num_frames: int) -> list[int]:
    if num_frames <= 0:
        return []
    candidates = [num_frames // 4, num_frames // 2, (3 * num_frames) // 4]
    return sorted({min(max(frame, 0), num_frames - 1) for frame in candidates})


def main(
    model_folder: Path,
    data_path: Path | None = None,
    headless: bool = True,
    device="cuda",
    simulator: str = "isaacsim",
    save_mp4: bool = False,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    episode_len: int = 1000,
    video_folder: str | None = None,
    robot: str | None = None,
    goal_json: Path | None = None,
    no_training_config: bool = False,
):

    model_folder = _resolve_path(model_folder)
    video_folder = Path(video_folder) if video_folder is not None else model_folder / "goal_inference" / "videos"
    video_folder.mkdir(parents=True, exist_ok=True)
    env_device = "cuda:0" if device == "cuda" else device
    simulator = simulator.lower()

    from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
    from humanoidverse.utils.helpers import export_meta_policy_as_onnx, export_z_encoder_as_onnx
    from humanoidverse.utils.helpers import get_backward_observation

    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    model_name = "model"
    model_name = model.__class__.__name__
    with open(model_folder / "config.json", "r") as f:
        config = json.load(f)

    use_root_height_obs = config["env"].get("root_height_obs", False)
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
    elif not Path(config["env"].get("lafan_tail_path", "")).exists():
        default_path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof.pkl"
        if default_path.exists():
            config["env"]["lafan_tail_path"] = str(default_path)
        else:
            config["env"]["lafan_tail_path"] = "data/lafan_29dof.pkl"
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
    # import ipdb; ipdb.set_trace()
    _append_or_replace_hydra_override(hydra_overrides, "env.config.max_episode_length_s=10000")
    _append_or_replace_hydra_override(hydra_overrides, f"env.config.headless={headless}")
    _append_or_replace_hydra_override(hydra_overrides, f"simulator={simulator}")
    if simulator == "mujoco":
        if robot in (None, "g1"):
            _append_or_replace_hydra_override(hydra_overrides, "robot.asset.xml_file=g1/scene_29dof_freebase_mujoco.xml")
        elif robot in PIPLUS_LSE_ROBOTS:
            _append_or_replace_hydra_override(
                hydra_overrides,
                "robot.asset.xml_file=package://ht_urdf/PiPlus_S_12L8A0G2H1W_LSE_260611/xml/PiPlus_S_12L8A0G2H1W_LSE_260611_with_armature.xml",
            )
        elif robot in PIPLUS_H0W_ROBOTS:
            _append_or_replace_hydra_override(
                hydra_overrides,
                "robot.asset.xml_file=package://ht_urdf/PiPlus_S_12L8A0G2H0W/xml/PiPlus_S_12L8A0G2H0W.xml",
            )
        elif robot in H1_260402_ROBOTS:
            _append_or_replace_hydra_override(hydra_overrides, f"robot.asset.xml_file={H1_260402_SIM_XML}")
    config["env"]["device"] = env_device
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise
    print(f"Inference resolved_config_path: {config['env'].get('resolved_config_path')}")
    print(f"Inference hydra_overrides: {hydra_overrides}")
    print(f"Inference model device: {device}")
    print(f"Inference env device: {env_device}")

    output_dir = model_folder / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = export_meta_policy_as_onnx(
        model,
        output_dir,
        f"{model_name}.onnx",
        {"actor_obs": torch.randn(1, model._actor.input_filter.output_space.shape[0] + model.cfg.archi.z_dim)},
        z_dim=model.cfg.archi.z_dim,
        history=('history_actor' in model.cfg.archi.actor.input_filter.key),
    )
    z_encoder_path = export_z_encoder_as_onnx(
        model,
        output_dir,
        f"{model_name}_z_encoder.onnx",
    )
    print(f"Exported model to {policy_path}")
    print(f"Exported z encoder to {z_encoder_path}")
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env
    print("="*80)
    print(env.config.simulator)
    print("-"*80)

    goals_to_evaluate, goal_source = _load_goal_frames(goal_json, robot)
    if goals_to_evaluate is None:
        goals_to_evaluate = _goals_from_motion_lib(env)
        print(
            "No goal JSON found; generating goal frames from the loaded motion library "
            f"({len(goals_to_evaluate)} motions)."
        )
    else:
        print(f"Loading goal frames from {goal_source}")

    pbar = tqdm(goals_to_evaluate, leave=False, disable=False)
    z_dict = {}
    with torch.no_grad():
        for goal in pbar:
            env.set_is_evaluating(goal["motion_id"])
                # we visulize the first env
            gobs, gobs_dict = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs, velocity_multiplier=0)
            num_frames = next(iter(gobs.values())).shape[0]
            goal_frames = goal["frames"] if goal["frames"] is not None else _default_goal_frames(num_frames)
            frame_pbar = tqdm(goal_frames, leave=False, disable=False, desc="frames")
            for frame_idx in frame_pbar:
                if frame_idx >= num_frames:
                    pbar.write(f"  Skipping frame_idx {frame_idx} (motion has {num_frames} frames)")
                    continue
                goal_name = f"{goal['motion_name']}_{frame_idx}"
                goal_observation = {k: v[frame_idx][None,...] for k,v in gobs.items()}
                goal_observation = tree_map(lambda x: torch.tensor(x, device=model.device, dtype=torch.float32), goal_observation)

                z_dict[goal_name] = model.goal_inference(goal_observation).cpu().numpy()
    path = model_folder / "goal_inference"
    path.mkdir(exist_ok=True)
    with open(os.path.join(path, "goal_reaching.pkl"), "wb") as f:
        joblib.dump(z_dict, f)

    if not z_dict:
        raise RuntimeError("No goal latents were produced. Check goal frames and motion data.")

    if save_mp4:
        import mediapy as media

        from humanoidverse.agents.envs.humanoidverse_isaac import IsaacRendererWithMuJoco

        if robot in PIPLUS_ROBOTS | H1_260402_ROBOTS:
            raise ValueError("save_mp4 currently uses the g1 MuJoCo renderer and is not supported for this robot.")
        rgb_renderer = IsaacRendererWithMuJoco(render_size=256)

    observation, info = wrapped_env.reset(to_numpy=False)
    observation, info = wrapped_env.reset(to_numpy=False)
    observation, info = wrapped_env.reset(to_numpy=False)
    observation, info = wrapped_env.reset(to_numpy=False)

    frames = []
    counter = 0
    # episode_len: number of sim steps for the optional goal-reaching video (~12 it/s → 5000 ≈ 7 min)
    _pbar = tqdm(desc="steps", disable=False, leave=False, total=episode_len)
    goal_idx = -1
    goal_names = list(z_dict.keys())

    while counter < episode_len:
        if counter % 100 == 0:
            goal_idx = (goal_idx + 1) % len(goal_names)
            print(f"Switching to goal {goal_names[goal_idx]} at step {counter}")
            z = z_dict[goal_names[goal_idx]].copy()
            z = torch.tensor(z, device=model.device, dtype=torch.float32)

        action = model.act(observation, z.repeat(num_envs, 1), mean=True)
        observation, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)
        if save_mp4:
            frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])
        counter += 1
        _pbar.update(1)
    _pbar.close()
    if save_mp4:
        media.write_video(video_folder / "goal.mp4", frames, fps=50)
        print("Saved video for goal")

if __name__ == "__main__":
    import tyro

    tyro.cli(main)
