import os

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
import json
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco
import torch
from humanoidverse.utils.helpers import export_meta_policy_as_onnx, export_z_encoder_as_onnx
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.utils.asset_paths import resolve_asset_path
import joblib
import mediapy as media
import numpy as np
from torch.utils._pytree import tree_map

import humanoidverse
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent


ROBOT_CONFIG_OVERRIDES = {
    "g1": "robot=g1/g1_29dof_hard_waist",
    "PiPlus_S_12L8A0G2H1W_LSE": "robot=piplus/PiPlus_S_12L8A0G2H1W_LSE",
    "piplus_lse": "robot=piplus/PiPlus_S_12L8A0G2H1W_LSE",
    "PiPlus_S_12L8A0G2H1W_LSE_40V": "robot=piplus/PiPlus_S_12L8A0G2H1W_LSE_40V",
    "piplus_lse_40v": "robot=piplus/PiPlus_S_12L8A0G2H1W_LSE_40V",
    "PiPlus_S_12L8A0G2H0W": "robot=piplus/PiPlus_S_12L8A0G2H0W",
    "piplus_h0w": "robot=piplus/PiPlus_S_12L8A0G2H0W",
    "Hi_P_12L10A0G2H1W_260402": "robot=Hi/Hi_P_12L10A0G2H1W_260402",
    "h1_260402": "robot=Hi/Hi_P_12L10A0G2H1W_260402",
}
PIPLUS_LSE_ROBOTS = {
    "PiPlus_S_12L8A0G2H1W_LSE",
    "piplus_lse",
    "PiPlus_S_12L8A0G2H1W_LSE_40V",
    "piplus_lse_40v",
}
PIPLUS_H0W_ROBOTS = {"PiPlus_S_12L8A0G2H0W", "piplus_h0w"}
PIPLUS_ROBOTS = PIPLUS_LSE_ROBOTS | PIPLUS_H0W_ROBOTS
H1_260402_ROBOTS = {"Hi_P_12L10A0G2H1W_260402", "h1_260402"}
G1_RENDER_XML = HUMANOIDVERSE_DIR / "data" / "robots" / "g1" / "scene_29dof_freebase_mujoco.xml"
PIPLUS_LSE_RENDER_XML = (
    HUMANOIDVERSE_DIR
    / "data"
    / "robots"
    / "piplus"
    / "PiPlus_S_12L8A0G2H1W_LSE_260611"
    / "xml"
    / "PiPlus_S_12L8A0G2H1W_LSE_260611_with_armature.xml"
)
PIPLUS_H0W_RENDER_XML = Path(
    "package://ht_urdf/PiPlus_S_12L8A0G2H0W/xml/"
    "PiPlus_S_12L8A0G2H0W.xml"
)
H1_260402_RENDER_XML = "package://ht_urdf/Hi_P_12L10A0G2H1W_260402/xml/Hi_P_12L10A0G2H1W_Simplify_260402_with_armature.xml"
MOTION_ALL_TOKEN = "motion_all"


def _render_xml_for_robot(robot: str | None, qpos_dim: int) -> Path:
    if robot in PIPLUS_LSE_ROBOTS or qpos_dim == 30:
        return PIPLUS_LSE_RENDER_XML
    if robot in PIPLUS_H0W_ROBOTS or qpos_dim == 29:
        return PIPLUS_H0W_RENDER_XML
    if robot in H1_260402_ROBOTS or qpos_dim == 32:
        return H1_260402_RENDER_XML
    if robot in (None, "g1") and qpos_dim == 36:
        return G1_RENDER_XML
    raise ValueError(f"No MP4 renderer configured for robot={robot!r} with {qpos_dim}-D qpos.")


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


def _resolve_motion_list(motion_list, num_motions: int) -> list[int]:
    if isinstance(motion_list, (str, int)):
        requested = [motion_list]
    else:
        requested = list(motion_list)

    requested_tokens = [str(motion).strip() for motion in requested]

    if len(requested_tokens) == 1 and requested_tokens[0].lower() == MOTION_ALL_TOKEN:
        return list(range(num_motions))

    if any(token.lower() == MOTION_ALL_TOKEN for token in requested_tokens):
        raise ValueError(f"{MOTION_ALL_TOKEN!r} must be passed by itself, e.g. --motion-list {MOTION_ALL_TOKEN}")

    try:
        return [int(token) for token in requested_tokens]
    except ValueError as exc:
        raise ValueError(
            f"Invalid --motion-list value {requested!r}. Use integer motion ids or {MOTION_ALL_TOKEN!r}."
        ) from exc


def main(
    model_folder: Path,
    data_path: Path | None = None,
    headless: bool = True,
    device="cuda",
    simulator: str = "isaacsim",
    save_mp4: bool = False,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    motion_list: list[str] = ["25"],
    robot: str | None = None,
    episode_len: int | None = None,
    no_training_config: bool = False,
):
    # motion_list: motion ids to evaluate, or ["motion_all"] for every motion.
    
    model_folder = _resolve_path(model_folder)
    env_device = "cuda:0" if device == "cuda" else device
    simulator = simulator.lower()

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
                "robot.asset.xml_file=xml/PiPlus_S_12L8A0G2H1W_LSE_260611_with_armature.xml",
            )
        elif robot in PIPLUS_H0W_ROBOTS:
            _append_or_replace_hydra_override(
                hydra_overrides,
                "robot.asset.xml_file=package://ht_urdf/PiPlus_S_12L8A0G2H0W/xml/PiPlus_S_12L8A0G2H0W.xml",
            )
        elif robot in H1_260402_ROBOTS:
            _append_or_replace_hydra_override(
                hydra_overrides,
                f"robot.asset.xml_file={H1_260402_RENDER_XML}",
            )
    config["env"]["device"] = env_device
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise
    print(f"Inference resolved_config_path: {config['env'].get('resolved_config_path')}")
    print(f"Inference hydra_overrides: {hydra_overrides}")
    print(f"Inference model device: {device}")
    print(f"Inference env device: {env_device}")

    # Outputs under model_folder/tracking_inference (sibling of exported/)
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

    def tracking_inference(obs) -> torch.Tensor:
        z = model.backward_map(obs)
        horizon = model.cfg.seq_length
        print(f"Horizon: {horizon}")
        for step in range(z.shape[0]):
            end_idx = min(step + horizon, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return model.project_z(z)

    # rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env
    print("="*80)
    print(env.config.simulator)
    print("-"*80)

    motion_list = _resolve_motion_list(motion_list, env._motion_lib._num_unique_motions)
    if motion_list == list(range(env._motion_lib._num_unique_motions)):
        print(f"Resolved motion list to all {len(motion_list)} motions")
    else:
        print(f"Resolved motion list: {motion_list}")
    
    output_dir = model_folder / "tracking_inference"

    for MOTION_ID in motion_list:
        env.set_is_evaluating(MOTION_ID)
        loaded_motion_ids = env._motion_lib._curr_motion_ids.detach().cpu().tolist()
        loaded_motion_keys = env._motion_lib.curr_motion_keys
        local_motion_id = 0
        print(f"Requested motion id {MOTION_ID}; loaded ids {loaded_motion_ids}; loaded keys {loaded_motion_keys}")
        obs, obs_dict = get_backward_observation(env, local_motion_id, use_root_height_obs=use_root_height_obs)

        expert_qpos = np.concatenate([
            obs_dict["ref_body_pos"][:,0].cpu().numpy(),
            np.roll(obs_dict["ref_body_rots"][:,0].cpu().numpy(),1,axis=-1),
            obs_dict["dof_pos"].cpu().numpy()
        ], axis=-1)

        # import ipdb; ipdb.set_trace()

        z = tracking_inference(tree_map(lambda x: x[1:], obs))
        output_dir.mkdir(parents=True, exist_ok=True)
        loaded_motion_key_name = (
            "_".join(str(motion_key) for motion_key in loaded_motion_keys)
            if isinstance(loaded_motion_keys, (list, tuple))
            else str(loaded_motion_keys)
        )
        output_filename = f"zs_{MOTION_ID}_{loaded_motion_key_name}.pkl"
        joblib.dump(z.cpu().numpy(), output_dir / output_filename)
        print(f"Saved {output_filename}")

        observation, info = wrapped_env.reset(to_numpy=False)

        # Root state: pos(3) + quat(4) + lin_vel(3) + ang_vel(3). Isaac expects quat as wxyz; motion lib uses xyzw.
        ref_body_rots = obs_dict["ref_body_rots"][0, 0].clone()
        if simulator == "isaacsim":
            ref_body_rots = ref_body_rots[[3, 0, 1, 2]]  # xyzw -> wxyz for correct humanoid facing in Isaac
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
            "root_states": torch.stack([ref_root_init_state.clone() for i in range(num_envs)])
        }
        env_ids = torch.arange(num_envs, dtype=torch.long, device=wrapped_env._env.device)
        observation, info = wrapped_env._env.reset_envs_idx(env_ids, target_states=target_states)
        # refresh_env_ids = wrapped_env._env.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        # wrapped_env._env.simulator.set_actor_root_state_tensor(refresh_env_ids, wrapped_env._env.target_robot_root_states)
        # wrapped_env._env.simulator.set_dof_state_tensor(refresh_env_ids, wrapped_env._env.target_robot_dof_state)
        # wrapped_env._env.need_to_refresh_envs[refresh_env_ids] = False
        zero_action = torch.zeros((num_envs, wrapped_env.action_space.shape[-1]), dtype=torch.float32, device=wrapped_env._env.device)
        observation_new, reward, terminated, truncated, info = wrapped_env.step(zero_action, to_numpy=False)
        observation = wrapped_env._get_g1env_observation(to_numpy=False)
        qpos, qvel = wrapped_env._get_qpos_qvel(to_numpy=True)
        assert np.allclose(wrapped_env._env.simulator.dof_pos.clone().cpu(), expert_qpos[0, 7:])
        joint_pos = [wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy()]
        episode_len=2000
        current_episode_len = episode_len if episode_len is not None else z.shape[0]
        if current_episode_len > z.shape[0]:
            print(f"Requested {current_episode_len} steps; cycling {z.shape[0]} inferred latent steps")
        print(f"Saving video for tracking ({current_episode_len} steps)")
        if save_mp4:
            render_xml = resolve_asset_path("", _render_xml_for_robot(robot, expert_qpos.shape[-1]))
            if not render_xml.exists():
                raise FileNotFoundError(f"MuJoCo render XML not found: {render_xml}")
            print(f"Rendering MP4 with MuJoCo XML: {render_xml}")
            rgb_renderer = IsaacRendererWithMuJoco(render_size=256, xml_path=render_xml)
            # Only render 1 + episode_len frames (same as frames list), not the full motion
            expert_video = rgb_renderer.from_qpos(expert_qpos[: 1 + current_episode_len])
            frames = [rgb_renderer.render(wrapped_env._env, 0)[0]]

        print(f"Running tracking inference for motion {MOTION_ID} for {current_episode_len} steps")
        for i in range(current_episode_len):
            print(f"Step {i} of {current_episode_len}")
            action = model.act(observation, z[i % len(z)].repeat(num_envs, 1), mean=True)
            observation, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)
            joint_pos.append(wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy())
            if save_mp4:
                frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])

        joint_pos = np.stack(joint_pos, axis=0).squeeze(1)
        stats = {}
        
        # breakpoint()  # use PYTHONBREAKPOINT=0 to disable, or install ipdb for a nicer debugger

        if save_mp4:
            new_frames = []
            for a, b in zip(expert_video, frames):
                new_frames.append(np.concatenate([a, b], axis=1))
            video_path = output_dir / f"tracking_{MOTION_ID}.mp4"
            media.write_video(str(video_path), new_frames, fps=50)
            print(f"Saved video for tracking: {video_path}")
            rgb_renderer.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
