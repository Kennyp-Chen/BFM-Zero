import os

os.environ.setdefault("MUJOCO_GL", "egl")  # Default to EGL for offscreen rendering.
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import json
import torch
import joblib
import rich
import time
import numpy as np

import humanoidverse
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.buffers.transition import DictBuffer
from humanoidverse.envs.g1_env_helper.bench import RewardWrapperHV
from humanoidverse.envs.piplus_env_helper.bench import PiPlusRewardWrapperHV
from humanoidverse.utils.asset_paths import resolve_asset_path
from humanoidverse.utils.helpers import export_meta_policy_as_onnx, export_z_encoder_as_onnx

# Resolve humanoidverse root directory
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).parent.parent.parent


ROBOT_G1 = "g1"
ROBOT_PIPLUS_LSE = "PiPlus_S_12L8A0G2H1W_LSE"
ROBOT_PIPLUS_ALIAS = "piplus_lse"
ROBOT_PIPLUS_H0W = "PiPlus_S_12L8A0G2H0W"
ROBOT_PIPLUS_H0W_ALIAS = "piplus_h0w"
ROBOT_H1_260402 = "Hi_P_12L10A0G2H1W_260402"
ROBOT_H1_260402_ALIAS = "h1_260402"
PIPLUS_LSE_ROBOTS = {ROBOT_PIPLUS_LSE, ROBOT_PIPLUS_ALIAS}
PIPLUS_H0W_ROBOTS = {ROBOT_PIPLUS_H0W, ROBOT_PIPLUS_H0W_ALIAS}
PIPLUS_ROBOTS = PIPLUS_LSE_ROBOTS | PIPLUS_H0W_ROBOTS
H1_260402_ROBOTS = {ROBOT_H1_260402, ROBOT_H1_260402_ALIAS}
PI_STYLE_REWARD_ROBOTS = PIPLUS_ROBOTS | H1_260402_ROBOTS
SUPPORTED_ROBOTS = {ROBOT_G1, *PI_STYLE_REWARD_ROBOTS}
ROBOT_CONFIG_OVERRIDES = {
    ROBOT_G1: "robot=g1/g1_29dof_hard_waist",
    ROBOT_PIPLUS_LSE: "robot=piplus/PiPlus_S_12L8A0G2H1W_LSE",
    ROBOT_PIPLUS_ALIAS: "robot=piplus/PiPlus_S_12L8A0G2H1W_LSE",
    ROBOT_PIPLUS_H0W: "robot=piplus/PiPlus_S_12L8A0G2H0W",
    ROBOT_PIPLUS_H0W_ALIAS: "robot=piplus/PiPlus_S_12L8A0G2H0W",
    ROBOT_H1_260402: "robot=Hi/Hi_P_12L10A0G2H1W_260402",
    ROBOT_H1_260402_ALIAS: "robot=Hi/Hi_P_12L10A0G2H1W_260402",
}
G1_REWARD_XML = HUMANOIDVERSE_DIR / "data" / "robots" / "g1" / "scene_29dof_freebase_noadditional_actuators.xml"
PIPLUS_REWARD_XML = Path(
    "package://ht_urdf/PiPlus_S_12L8A0G2H1W_LSE_260611/xml/"
    "PiPlus_S_12L8A0G2H1W_LSE_260611_with_armature.xml"
)
PIPLUS_SIM_XML = Path(
    "package://ht_urdf/PiPlus_S_12L8A0G2H1W_LSE_260611/xml/"
    "PiPlus_S_12L8A0G2H1W_LSE_260611_with_armature.xml"
)
PIPLUS_H0W_SIM_XML = Path(
    "package://ht_urdf/PiPlus_S_12L8A0G2H0W/xml/"
    "PiPlus_S_12L8A0G2H0W.xml"
)
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


class MuJoCoStateRenderer:
    def __init__(self, xml_path: Path, render_size: int = 256):
        import mujoco

        xml_path = resolve_asset_path("", xml_path)
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=render_size, height=render_size)

    def render(self, wrapped_env, env_idx: int = 0):
        qpos, qvel = wrapped_env._get_qpos_qvel(to_numpy=True)
        qpos = np.asarray(qpos[env_idx]).ravel()
        qvel = np.asarray(qvel[env_idx]).ravel()
        if qpos.shape[0] != self.model.nq:
            raise ValueError(f"Renderer qpos mismatch: env has {qpos.shape[0]}, MuJoCo model expects {self.model.nq}")
        if qvel.shape[0] != self.model.nv:
            raise ValueError(f"Renderer qvel mismatch: env has {qvel.shape[0]}, MuJoCo model expects {self.model.nv}")
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()


def main(
    model_folder: Path,
    data_path: Path | None = None,
    buffer_folder: Path | None = None,
    headless: bool = True,
    device="cuda",
    simulator: str = "isaacsim",
    save_mp4: bool = False,
    episode_length: int = 500,
    video_folder: str | None = None,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    num_samples: int = 150_000,
    n_inferences: int = 1,
    skip_rollouts: bool = False,
    robot: str = ROBOT_G1,
    no_training_config: bool = False,
):
    if robot not in SUPPORTED_ROBOTS:
        raise ValueError(f"Unsupported robot {robot!r}. Expected one of: {sorted(SUPPORTED_ROBOTS)}")

    model_folder = _resolve_path(model_folder)
    video_folder = Path(video_folder) if video_folder is not None else model_folder / "reward_inference" / "videos"
    video_folder.mkdir(parents=True, exist_ok=True)
    env_device = "cuda:0" if device == "cuda" else device
    simulator = simulator.lower()

    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    model_name = model.__class__.__name__
    with open(model_folder / "config.json", "r") as f:
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

    if not Path(config["env"]["lafan_tail_path"]).exists():
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
    _append_or_replace_hydra_override(hydra_overrides, ROBOT_CONFIG_OVERRIDES[robot])
    _append_or_replace_hydra_override(hydra_overrides, "env.config.max_episode_length_s=10000")
    _append_or_replace_hydra_override(hydra_overrides, f"env.config.headless={headless}")
    # config["env"]["hydra_overrides"].append("env.config.lie_down_init=True")
    # config["env"]["hydra_overrides"].append("env.config.lie_down_init_prob=1")
    _append_or_replace_hydra_override(hydra_overrides, f"simulator={simulator}")
    if simulator == "mujoco":
        if robot == ROBOT_G1:
            _append_or_replace_hydra_override(hydra_overrides, "robot.asset.xml_file=g1/scene_29dof_freebase_mujoco.xml")
        elif robot in PIPLUS_LSE_ROBOTS:
            _append_or_replace_hydra_override(hydra_overrides, f"robot.asset.xml_file={PIPLUS_SIM_XML}")
        elif robot in PIPLUS_H0W_ROBOTS:
            _append_or_replace_hydra_override(hydra_overrides, f"robot.asset.xml_file={PIPLUS_H0W_SIM_XML}")
        elif robot in H1_260402_ROBOTS:
            _append_or_replace_hydra_override(hydra_overrides, f"robot.asset.xml_file={H1_260402_SIM_XML}")
    config["env"]["device"] = env_device
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise

    print(f"Inference resolved_config_path: {config['env'].get('resolved_config_path')}")
    print(f"Inference hydra_overrides: {hydra_overrides}")
    print(f"Inference model device: {device}")
    print(f"Inference env device: {env_device}")
    rich.print(config["env"])
    num_envs = 1
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    wrapped_env, _ = env_cfg.build(num_envs)

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
    tasks = [
        # stand
        "move-ego-0-0",
        "move-ego-low0.5-0-0",

        # locomotion medium
        "move-ego-0-0.7",
        # "move-ego-90-0.7",
        # "move-ego-180-0.7",
        # "move-ego--90-0.7",

        # "move-ego-low0.6-0-0.7",

        # locomotion slow
        "move-ego-0-0.3",
        # "move-ego-90-0.3",
        # "move-ego-180-0.3",
        # "move-ego--90-0.3",
        
        # locomotion fast
        # "move-ego-0-1",
        # "move-ego-90-1",
        # "move-ego-180-1",
        # "move-ego--90-1",

        # spin
        "rotate-z-5-0.5",
        "rotate-z--5-0.5",
    
        # # raise arms
        # "raisearms-l-l",
        # "raisearms-l-m",
        # "raisearms-m-l",
        # "raisearms-m-m",


        # # move + arms
        # "move-arms-0-0.7-m-m",
        # "move-arms-90-0.7-m-m",
        # "move-arms-180-0.4-m-m",
        # "move-arms--90-0.7-m-m",
        # "move-arms-0-0.7-l-m",
        # "move-arms-90-0.7-l-m",
        # "move-arms-180-0.4-l-m",
        # "move-arms--90-0.7-l-m",
        # "move-arms-0-0.7-m-l",
        # "move-arms-90-0.7-m-l",
        # "move-arms-180-0.4-m-l",
        # "move-arms--90-0.7-m-l",
        # "move-arms-0-0.7-l-l",
        # "move-arms-90-0.7-l-l",
        # "move-arms-180-0.4-l-l",
        # "move-arms--90-0.7-l-l",

        # # spin + arms
        # "spin-arms-5-l-l",
        # "spin-arms--5-l-l",
        # "spin-arms-5-l-m",
        # "spin-arms--5-l-m",
        # # "spin-arms-5-m-m",
        # # "spin-arms--5-m-m",
        # "spin-arms-5-m-l",
        # "spin-arms--5-m-l",

        # # sit
        # "crouch-0",
        # "crouch-0.25",
        # "sitonground",
    ]

    print("Loading the replay buffer...", end=" ", flush=True)
    start_t = time.time()
    if buffer_folder is not None:
        buffer_path = _resolve_path(buffer_folder)
        if not (buffer_path / "config.json").exists() or not (buffer_path / "buffer.hdf5").exists():
            raise FileNotFoundError(
                f"Replay buffer folder is incomplete: {buffer_path}. "
                "Expected config.json and buffer.hdf5."
            )
        with open(buffer_path / "config.json", "r") as f:
            buffer_config = json.load(f)
        if buffer_config.get("__target__", "").endswith("DictBuffer"):
            dataset = DictBuffer.load(buffer_path, device="cpu")
        else:
            dataset = TrajectoryDictBufferMultiDim.load(buffer_path, device="cpu")
        print(f"Loaded replay buffer from {buffer_path}")
    else:
        reduced_buffer_path = model_folder / "checkpoint/buffers/train_reduced"
        full_buffer_path = model_folder / "checkpoint/buffers/train"
        if (reduced_buffer_path / "buffer.hdf5").exists():
            dataset = DictBuffer.load(reduced_buffer_path, device="cpu")
            print("Loaded reduced buffer")
        elif (full_buffer_path / "buffer.hdf5").exists():
            dataset = TrajectoryDictBufferMultiDim.load(full_buffer_path, device="cpu")
            print("Loaded original buffer")
        else:
            raise FileNotFoundError(
                "No replay buffer HDF5 found. Reward inference needs replay buffer samples "
                "for reward relabeling. Searched:\n"
                f"  - {reduced_buffer_path / 'buffer.hdf5'}\n"
                f"  - {full_buffer_path / 'buffer.hdf5'}\n"
                "Pass --buffer-folder PATH_TO_BUFFER_DIR if the buffer is stored elsewhere."
            )
    # dataset = fast_load_buffer(model_folder / "checkpoint/buffers/train", device="cpu")
    print(f"done in {time.time()-start_t}s")
    inference_function = "reward_wr_inference"
    if robot in PI_STYLE_REWARD_ROBOTS:
        reward_xml = H1_260402_SIM_XML if robot in H1_260402_ROBOTS else PIPLUS_REWARD_XML
        reward_xml = resolve_asset_path("", reward_xml)
        reward_eval_agent = PiPlusRewardWrapperHV(
            model=model,
            inference_dataset=dataset,
            num_samples_per_inference=num_samples,
            inference_function=inference_function,
            max_workers=24,
            env_model=str(reward_xml),
        )
    else:
        reward_eval_agent = RewardWrapperHV(
            model=model,
            inference_dataset=dataset,
            num_samples_per_inference=num_samples,
            inference_function=inference_function,
            max_workers=24,
            process_executor=True,
            env_model=str(G1_REWARD_XML),
        )
    z_dict = {}
    for r in range(n_inferences):
        for task in tasks:
            print(f"Started inference for {task}...", end=" ", flush=True)
            start_t = time.time()
            z = reward_eval_agent.reward_inference(task=task)
            z_dict[task] = z_dict.get(task, []) + [z.cpu()]
            print(f"done in {time.time()-start_t}s")

            path = model_folder / "reward_inference"
            path.mkdir(exist_ok=True)
            with open(os.path.join(path, "reward_locomotion.pkl"), "wb") as f:
                joblib.dump(z_dict, f)
            print(f"Saved file at {path}/reward_locomotion.pkl")

    # z_dict = joblib.load(model_folder / "reward_inference/reward_locomotion.pkl")

    if not skip_rollouts:
        print("Generating videos...")
        if save_mp4:
            import mediapy as media

            if robot in PIPLUS_LSE_ROBOTS:
                render_xml = PIPLUS_SIM_XML
            elif robot in PIPLUS_H0W_ROBOTS:
                render_xml = PIPLUS_H0W_SIM_XML
            elif robot in H1_260402_ROBOTS:
                render_xml = H1_260402_SIM_XML
            else:
                render_xml = G1_REWARD_XML
            rgb_renderer = MuJoCoStateRenderer(render_xml, render_size=256)
        for task in tasks:
            frames = []
            for z in z_dict[task]:
                z = z.repeat(num_envs, 1).to(device)
                
                observation, info = wrapped_env.reset(to_numpy=False, reset_to_default_pose=True)
                if save_mp4:
                    frames.append(rgb_renderer.render(wrapped_env, 0))
                for i in range(episode_length):
                    action = model.act(observation, z, mean=True)
                    observation, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)

                    if save_mp4:
                        frames.append(rgb_renderer.render(wrapped_env, 0))
            if save_mp4:
                file = video_folder / f"{task}.mp4"
                media.write_video(file, frames, fps=50)
                print(f"Saved video for {task}: {file}")
        if save_mp4:
            rgb_renderer.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
