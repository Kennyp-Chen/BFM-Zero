#!/usr/bin/env python3
"""
Comprehensive script to export 23DOF models and generate three pickle files:
1. goal_reaching.pkl - goal inference latent variables
2. zs_{motion_id}.pkl - tracking inference latent variables
3. reward_inference.pkl - reward inference data

This script combines functionality from goal_inference.py, tracking_inference.py,
and adds reward inference support for 23DOF models. No simulation rendering or
video saving is included.
"""

import os
import time
from pathlib import Path
import json
import pickle
import torch
import joblib
import numpy as np
from torch.utils._pytree import tree_map
from tqdm import tqdm

import humanoidverse
from humanoidverse.agents.load_utils import (
    MODEL_NAME_TO_CLASS,
    load_model_from_checkpoint_dir,
)
from humanoidverse.agents.envs.humanoidverse_isaac import (
    HumanoidVerseIsaacConfig
)
from humanoidverse.agents.buffers.trajectory import (
    TrajectoryDictBufferMultiDim
)
from humanoidverse.agents.buffers.transition import DictBuffer
from humanoidverse.envs.g1_env_helper.bench import RewardWrapperHV
from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
from humanoidverse.utils.helpers import (
    export_meta_policy_as_onnx,
    get_backward_observation
)

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering
os.environ["OMP_NUM_THREADS"] = "1"


# Resolve humanoidverse root directory
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).parent.parent.parent


def convert_29dof_to_23dof(
    joint_names_29: list[str], params_29: np.ndarray
) -> np.ndarray:
    """
    Convert 29DoF parameters to 23DoF by removing specific joints.
    """
    # Joints to remove
    joints_to_remove = {
        'waist_roll_joint',
        'waist_pitch_joint',
        'left_wrist_pitch_joint',
        'left_wrist_yaw_joint',
        'right_wrist_pitch_joint',
        'right_wrist_yaw_joint'
    }

    # Filter out the joints to remove
    indices_to_keep = [
        i for i, joint_name in enumerate(joint_names_29)
        if joint_name not in joints_to_remove
    ]

    params_23 = params_29[..., indices_to_keep]

    return params_23


def build_model_from_checkpoint_model_dir(
    model_dir: Path,
    device: str,
    model_config_path: Path | None = None,
):
    cfg_path = (
        model_config_path
        if model_config_path is not None
        else (model_dir / "config.json")
    )
    with cfg_path.open("r") as f:
        config = json.load(f)

    # Some checkpoints store the *agent* config under model/config.json.
    # In that case, the actual model config lives under config["model"].
    if config.get("name") not in MODEL_NAME_TO_CLASS and isinstance(
        config.get("model"), dict
    ):
        config = config["model"]

    model_name = config.get("name")
    if model_name not in MODEL_NAME_TO_CLASS:
        available_models = list(MODEL_NAME_TO_CLASS.keys())
        raise ValueError(
            f"Unknown model name: {model_name}. Available: {available_models}"
        )

    config["device"] = device
    model_class = MODEL_NAME_TO_CLASS[model_name]
    config_class = model_class.config_class

    build_kwargs = None
    if (model_dir / "init_kwargs.pkl").exists():
        with (model_dir / "init_kwargs.pkl").open("rb") as f:
            build_kwargs = pickle.load(f)
    elif (model_dir / "init_kwargs.json").exists():
        with (model_dir / "init_kwargs.json").open("r") as f:
            build_kwargs = json.load(f)
        if "obs_space" in build_kwargs:
            build_kwargs["obs_space"] = json_to_space(
                build_kwargs["obs_space"]
            )

    if build_kwargs is None:
        raise ValueError(
            f"Missing init kwargs in {model_dir}. Expected init_kwargs.pkl or "
            "init_kwargs.json."
        )

    loaded_cfg = config_class(**config)
    return loaded_cfg.build(**build_kwargs)


def export_goal_inference(
    model, model_folder, checkpoint_path, config, device, robot_dof=23
):
    """Export goal inference model and generate goal_reaching.pkl"""
    print("=" * 60)
    print("EXPORTING GOAL INFERENCE")
    print("=" * 60)

    # Setup environment
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env

    # Find goal frames JSON file
    goal_json_paths = [
        HUMANOIDVERSE_DIR / "data" / "robots" / "g1" /
        "goal_frames_lafan29dof.json",
        HUMANOIDVERSE_DIR / "data" / "goal_frames_lafan29dof.json",
    ]
    goal_json = None
    for path in goal_json_paths:
        if path.exists():
            goal_json = str(path)
            break

    if goal_json is None:
        print("Warning: Could not find goal_frames_lafan29dof.json, "
              "skipping goal inference")
        return

    with open(goal_json, "r") as f:
        goals_to_evaluate = json.load(f)

    use_root_height_obs = config["env"].get("root_height_obs", False)

    # Generate goal inference data
    z_dict = {}
    with torch.no_grad():
        for goal in tqdm(goals_to_evaluate, desc="Processing goals"):
            env.set_is_evaluating(goal["motion_id"])
            gobs, gobs_dict = get_backward_observation(
                env, 0, use_root_height_obs=use_root_height_obs,
                velocity_multiplier=0
            )
            num_frames = next(iter(gobs.values())).shape[0]

            for frame_idx in goal["frames"]:
                if frame_idx >= num_frames:
                    continue
                goal_name = f"{goal['motion_name']}_{frame_idx}"
                goal_observation = {
                    k: v[frame_idx][None, ...] for k, v in gobs.items()
                }
                goal_observation = tree_map(
                    lambda x: torch.tensor(
                        x, device=model.device, dtype=torch.float32
                    ), goal_observation
                )

                z_dict[goal_name] = model.goal_inference(
                    goal_observation
                ).cpu().numpy()
    
    # Save goal inference data
    path = model_folder / "goal_inference"
    path.mkdir(exist_ok=True)
    with open(path / "goal_reaching.pkl", "wb") as f:
        joblib.dump(z_dict, f)
    print(f"Saved goal inference data to {path / 'goal_reaching.pkl'}")


def export_tracking_inference(
    model, model_folder, checkpoint_path, config, device, robot_dof=23
):
    """Export tracking inference model and generate zs_{motion_id}.pkl files"""
    print("=" * 60)
    print("EXPORTING TRACKING INFERENCE")
    print("=" * 60)

    # Setup environment
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env

    def tracking_inference(obs) -> torch.Tensor:
        z = model.backward_map(obs)
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return model.project_z(z)
    
    use_root_height_obs = config["env"].get("root_height_obs", False)

    # Generate tracking inference data for all motions
    motion_list = list(range(40))  # Process all 40 motions
    output_dir = model_folder / "tracking_inference"
    output_dir.mkdir(exist_ok=True)

    for motion_id in tqdm(motion_list, desc="Processing motions"):
        env.set_is_evaluating(motion_id)
        obs, obs_dict = get_backward_observation(
            env, 0, use_root_height_obs=use_root_height_obs
        )

        z = tracking_inference(tree_map(lambda x: x[1:], obs))
        joblib.dump(z.cpu().numpy(), output_dir / f"zs_{motion_id}.pkl")
    
    print(f"Saved tracking inference data to {output_dir}/zs_*.pkl")


def export_reward_inference(
    model, model_folder, checkpoint_path, config, device, robot_dof=23
):
    """Export reward inference data using RewardWrapperHV"""
    print("=" * 60)
    print("EXPORTING REWARD INFERENCE")
    print("=" * 60)

    tasks = [
        # stand
        "move-ego-0-0",
        "move-ego-low0.5-0-0",

        # locomotion medium
        "move-ego-0-0.7",
        "move-ego-90-0.7",
        "move-ego-180-0.7",
        "move-ego--90-0.7",

        "move-ego-low0.6-0-0.7",

        # locomotion slow
        "move-ego-0-0.3",
        "move-ego-90-0.3",
        "move-ego-180-0.3",
        "move-ego--90-0.3",
        
        # locomotion fast
        "move-ego-0-1",
        "move-ego-90-1",
        "move-ego-180-1",
        "move-ego--90-1",

        # spin
        "rotate-z-5-0.5",
        "rotate-z--5-0.5",
    
        # raise arms
        "raisearms-l-l",
        "raisearms-l-m",
        "raisearms-m-l",
        "raisearms-m-m",

        # move + arms
        "move-arms-0-0.7-m-m",
        "move-arms-90-0.7-m-m",
        "move-arms-180-0.4-m-m",
        "move-arms--90-0.7-m-m",
        "move-arms-0-0.7-l-m",
        "move-arms-90-0.7-l-m",
        "move-arms-180-0.4-l-m",
        "move-arms--90-0.7-l-m",
        "move-arms-0-0.7-m-l",
        "move-arms-90-0.7-m-l",
        "move-arms-180-0.4-m-l",
        "move-arms--90-0.7-m-l",
        "move-arms-0-0.7-l-l",
        "move-arms-90-0.7-l-l",
        "move-arms-180-0.4-l-l",
        "move-arms--90-0.7-l-l",

        # spin + arms
        "spin-arms-5-l-l",
        "spin-arms--5-l-l",
        "spin-arms-5-l-m",
        "spin-arms--5-l-m",
        "spin-arms-5-m-m",
        "spin-arms--5-m-m",
        "spin-arms-5-m-l",
        "spin-arms--5-m-l",

        # sit
        "crouch-0",
        "crouch-0.25",
        "sitonground",
    ]

    print("Loading the replay buffer...", end=" ", flush=True)
    start_t = time.time()
    buffer_path = checkpoint_path / "buffers/train"
    if buffer_path.is_dir() and (buffer_path / "config.json").exists():
        dataset = TrajectoryDictBufferMultiDim.load(buffer_path,device)
    print(f"Loaded original buffer from {buffer_path} ")
    print(f"done in {time.time()-start_t}s")

    inference_function = "reward_wr_inference"
    xml_path = (
        HUMANOIDVERSE_DIR / "data" / "robots" / "g1" /
        f"scene_{robot_dof}dof_freebase_noadditional_actuators.xml"
    )

    reward_eval_agent = RewardWrapperHV(
        model=model,
        inference_dataset=dataset,
        num_samples_per_inference=150_000,
        inference_function=inference_function,
        max_workers=24,
        process_executor=True,
        env_model=str(xml_path),
    )

    z_dict = {}
    n_inferences = 1
    for r in range(n_inferences):
        for task in tqdm(tasks, desc="Processing tasks"):
            print(f"Started inference for {task}...", end=" ", flush=True)
            start_t = time.time()
            z = reward_eval_agent.reward_inference(task=task)
            z_dict[task] = z_dict.get(task, []) + [z.cpu()]
            print(f"done in {time.time()-start_t}s")

    # Save reward inference data
    path = model_folder / "reward_inference"
    path.mkdir(exist_ok=True)
    with open(path / "reward_locomotion.pkl", "wb") as f:
        joblib.dump(z_dict, f)
    print(f"Saved reward inference data to {path / 'reward_locomotion.pkl'}")


def main(
    model_folder: Path = Path(
        "results/23dof-bfmzero-isaac-low/20260407_182514"
    ),
    # checkpoint_name: str = "checkpoint_mode_144000000l",
    checkpoint_name: str = "checkpoint",
    checkpoint_file: Path | None = None,
    checkpoint_model_config: Path | None = None,

    data_path: Path | None = Path("humanoidverse/data/lafan_23dof.pkl"),# humanoidverse/data/lafan_23dof.pkl
    device: str = "cuda",
    robot_dof: int = 23,
    headless: bool = True,
    simulator: str = "isaacsim",
    disable_dr: bool = True,
    disable_obs_noise: bool = True,
    export_goal: bool = True,
    export_tracking: bool = True,
    export_reward: bool = True,
    export_onnx: bool = True,
):
    """
    Export 23DOF models and generate three pickle files:
    1. goal_reaching.pkl - goal inference latent variables
    2. zs_{motion_id}.pkl - tracking inference latent variables
    3. reward_locomotion.pkl - reward inference data
    """
    model_folder = Path(model_folder)
    checkpoint_path = model_folder / checkpoint_name

    if checkpoint_file is not None:
        # If user passes a direct checkpoint file, infer the matching
        # checkpoint directory and workdir from its location.
        # Expected layout:
        #   .../<workdir>/checkpoint/model/model_XXXX.safetensors
        ckpt_file_path = Path(checkpoint_file)
        inferred_ckpt_dir = None
        for parent in ckpt_file_path.parents:
            if parent.name == "checkpoint":
                inferred_ckpt_dir = parent
                break
        if inferred_ckpt_dir is not None:
            checkpoint_path = inferred_ckpt_dir
            model_folder = inferred_ckpt_dir.parent

    print(f"Starting export for 23DOF model from: {model_folder}")
    if checkpoint_file is not None:
        print(f"Using specific checkpoint file: {checkpoint_file}")
    else:
        print(f"Checkpoint folder: {checkpoint_path}")
    print(f"Robot DOF: {robot_dof}")
    print(f"Device: {device}")

    # Load model
    if checkpoint_file is not None:
        # If we have a direct file, we still need the model config + init kwargs
        # to build the correct network structure.
        model = build_model_from_checkpoint_model_dir(
            checkpoint_path / "model",
            device=device,
            model_config_path=checkpoint_model_config,
        )
        print(f"Loading weights from {checkpoint_file}")

        state_dict = None
        if str(checkpoint_file).endswith('.pth'):
            state_dict = torch.load(checkpoint_file, map_location=device)
        elif str(checkpoint_file).endswith('.safetensors'):
            import safetensors.torch
            state_dict = safetensors.torch.load_file(
                checkpoint_file, device=device
            )

        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(
                f"Could not load checkpoint from {checkpoint_file}. "
                "Supported formats: .pth, .safetensors"
            )
    else:
        model = load_model_from_checkpoint_dir(checkpoint_path, device=device)

    model.to(device)
    model.eval()

    # Load config
    with open(model_folder / "config.json", "r") as f:
        config = json.load(f)

    # Set data path
    if data_path is not None:
        config["env"]["lafan_tail_path"] = str(Path(data_path).resolve())
    elif not Path(config["env"].get("lafan_tail_path", "")).exists():
        default_path = (
            HUMANOIDVERSE_DIR / "data" / f"lafan_{robot_dof}dof.pkl"
        )
        if default_path.exists():
            config["env"]["lafan_tail_path"] = str(default_path)
        else:
            config["env"]["lafan_tail_path"] = f"data/lafan_{robot_dof}dof.pkl"

    # Configure environment
    config["env"]["hydra_overrides"].append(
        "env.config.max_episode_length_s=10000"
    )
    config["env"]["hydra_overrides"].append(f"env.config.headless={headless}")
    config["env"]["hydra_overrides"].append(f"simulator={simulator}")
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise

    print(f"Using data path: {config['env']['lafan_tail_path']}")
    # Export ONNX model once
    if export_onnx:
        print("=" * 60)
        print("EXPORTING ONNX MODEL")
        print("=" * 60)
        output_dir = model_folder / "exported"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = model.__class__.__name__

        export_meta_policy_as_onnx(
            model,
            output_dir,
            f"{model_name}.onnx",
            {"actor_obs": torch.randn(
                1, model._actor.input_filter.output_space.shape[0] +
                model.cfg.archi.z_dim
            )},
            z_dim=model.cfg.archi.z_dim,
            history=('history_actor' in model.cfg.archi.actor.input_filter.key),
            use_29dof=(robot_dof == 29),
        )
        print(f"Exported model to {output_dir}/{model_name}.onnx")

    # Export selected components
    if export_goal:
        try:
            export_goal_inference(
                model, model_folder, checkpoint_path, config, device, robot_dof
            )
            print("Goal inference export completed successfully")
        except Exception as e:
            print(f"Goal inference export failed: {e}")

    if export_tracking:
        try:
            export_tracking_inference(
                model, model_folder, checkpoint_path, config, device, robot_dof
            )
            print("Tracking inference export completed successfully")
        except Exception as e:
            print(f"Tracking inference export failed: {e}")

    if export_reward:
        
        export_reward_inference(
            model, model_folder, checkpoint_path, config, device, robot_dof
        )
        

    print("=" * 60)
    print("EXPORT PROCESS COMPLETED")
    print("="*60)
    print("Generated files:")
    print(f"1. {model_folder}/goal_inference/goal_reaching.pkl")
    print(f"2. {model_folder}/tracking_inference/zs_*.pkl (40 files)")
    print(f"3. {model_folder}/reward_inference/reward_locomotion.pkl")
    print(f"4. {model_folder}/exported/*.onnx (model exports)")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
