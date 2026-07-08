import os

os.environ["MUJOCO_GL"] = "glfw"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
import mujoco
import imageio
import json
import torch
from torch.utils._pytree import tree_map
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import (
    HumanoidVerseIsaacConfig,
    IsaacRendererWithMuJoco,
)
from humanoidverse.utils.helpers import get_backward_observation

import humanoidverse

if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent


def main(model_folder: Path = Path("results/23dof-bfmzero-isaac-low/20260407_182514/"),
         data_path: Path | None = None,
         headless: bool = True,  # Back to True to avoid GUI issues
         device: str = "cuda",
         robot_dof: int = 23,
         episode_len: int = 1000):

    model_folder = Path(model_folder)

    # Load model
    model = load_model_from_checkpoint_dir(
        model_folder / "checkpoint_mode_115200000l", device=device)
    model.to(device)
    model.eval()

    # Load config
    with open(model_folder / "config.json", "r") as f:
        config = json.load(f)

    use_root_height_obs = config["env"].get("root_height_obs", False)

    # Set data path
    if data_path is not None:
        config["env"]["lafan_tail_path"] = str(Path(data_path).resolve())
    elif not Path(config["env"].get("lafan_tail_path", "")).exists():
        default_path = HUMANOIDVERSE_DIR / "data" / f"lafan_{robot_dof}dof.pkl"
        if default_path.exists():
            config["env"]["lafan_tail_path"] = str(default_path)
        else:
            config["env"]["lafan_tail_path"] = f"data/lafan_{robot_dof}dof.pkl"

    # Configure environment
    config["env"]["hydra_overrides"].append(
        "env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append(
        f"env.config.headless={headless}")
    config["env"]["hydra_overrides"].append(f"simulator=isaacsim")

    # Create environment
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env

    # Initialize with motion ID 25
    MOTION_ID = 36
    env.set_is_evaluating(MOTION_ID)

    # Get initial observation
    obs, obs_dict = get_backward_observation(
        env, 0, use_root_height_obs=use_root_height_obs)

    # Compute latent variables
    def tracking_inference(obs) -> torch.Tensor:
        z = model.backward_map(obs)
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return model.project_z(z)

    z = tracking_inference(tree_map(lambda x: x[1:], obs))

    # Reset environment
    observation, info = wrapped_env.reset(to_numpy=False)

    # Set initial states
    ref_body_rots = obs_dict["ref_body_rots"][0, 0].clone()
    ref_body_rots = ref_body_rots[[3, 0, 1, 2]]

    ref_root_init_state = torch.cat([
        obs_dict["ref_body_pos"][0, 0],
        ref_body_rots,
        obs_dict["ref_body_vels"][0, 0],
        obs_dict["ref_body_angular_vels"][0, 0],
    ])

    dof_init_state = torch.zeros_like(
        wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0])
    dof_init_state[..., 0] = obs_dict["dof_pos"][0]
    dof_init_state[..., 1] = obs_dict["ref_dof_vel"][0]

    target_states = {
        "dof_states": dof_init_state,
        "root_states": torch.stack(
            [ref_root_init_state.clone() for i in range(num_envs)])
    }

    env_ids = torch.arange(num_envs, dtype=torch.long)
    observation, info = wrapped_env._env.reset_envs_idx(
        env_ids, target_states=target_states)

    # Step once to sync
    observation_new, reward, terminated, truncated, info = wrapped_env.step(
        torch.zeros((num_envs, wrapped_env.action_space.shape[-1]),
                   dtype=torch.float32),
        to_numpy=False)
    observation = wrapped_env._get_g1env_observation(to_numpy=False)

    # Load MuJoCo model directly for rendering
    xml_path = HUMANOIDVERSE_DIR / f"data/robots/g1/scene_{robot_dof}dof_freebase_mujoco.xml"


    # Load MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)
    
    print(f"Loaded MuJoCo model: {xml_path}")
    print(f"Model has {mj_model.nq} DOF positions and {mj_model.nv} DOF velocities")
    print(f"Running model inference simulation for {episode_len} steps")
    print("Press Ctrl+C to stop")

    # Setup MuJoCo renderer and camera
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    frames = []
    
    # Create camera for fixed view
    camera = mujoco.MjvCamera()
    camera.distance = 8.0  # Pull camera back further
    camera.azimuth = 90    # Side view angle
    camera.elevation = -15 # Slightly downward angle
    
    # Setup matplotlib for real-time display
    if not headless:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(np.zeros((480, 640, 3)))
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    try:
        for i in range(episode_len):
            print(f"Step {i} of {episode_len}")

            # Get action from model
            action = model.act(observation,
                            z[i % len(z)].repeat(num_envs, 1),
                            mean=True)

            # Step environment
            observation, reward, terminated, truncated, info = wrapped_env.step(
                action, to_numpy=False)

            # Get current robot state from environment
            qpos, qvel = wrapped_env._get_qpos_qvel(to_numpy=True)
            
            # Update MuJoCo data with current robot state
            mj_data.qpos[:] = qpos[0]  # Use first environment's state
            mj_data.qvel[:] = qvel[0]
            
            # Step MuJoCo physics to synchronize
            for _ in range(5):  # Multiple sub-steps for stability
                mujoco.mj_step(mj_model, mj_data)
            
            # Capture frame with fixed camera
            renderer.update_scene(mj_data, camera=camera)
            pixels = renderer.render()
            frames.append(pixels.copy())
            
            # Update matplotlib display for real-time viewing
            if not headless:
                im.set_data(pixels)
                plt.pause(0.01)  # Small pause to allow display update
                plt.draw()


    except KeyboardInterrupt:
        print("\nSimulation stopped by user")

    # Save video
    if frames:
        print(f"Saving video with {len(frames)} frames...")
        imageio.mimsave(f"model_inference_tracked_{MOTION_ID}.mp4", frames, fps=30)
        print(f"Video saved as model_inference_tracked_{MOTION_ID}.mp4")
    
    # Cleanup matplotlib if needed
    if not headless:
        plt.ioff()  # Turn off interactive mode
        plt.close('all')
        print("Real-time simulation completed")
    
    print("Simulation completed")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
