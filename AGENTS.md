# BFM-Zero

**Promptable Behavioral Foundation Model for Humanoid Control via Unsupervised RL.**

## Quick Start

```bash
# Install
uv sync

# Launch training (tyro-based, uses FBcprAux agent)
uv run python -m humanoidverse.train

# Inference
uv run python -m humanoidverse.tracking_inference --model_folder /path/to/model
uv run python -m humanoidverse.goal_inference --model_folder /path/to/model
uv run python -m humanoidverse.reward_inference --model_folder /path/to/model
```

See `humanoidverse/train.py` for the `train_bfm_zero()` function that configures everything.

## Motion Data

- Training: `humanoidverse/data/lafan_29dof_10s-clipped.pkl` (10s clips, 862 sequences)
- Evaluation: `humanoidverse/data/lafan_29dof.pkl` (40 full sequences)
- Stored via Git LFS — run `git lfs pull` after clone
- 23-DOF variant available: `lafan_23dof*.pkl` (removes 6 wrist/waist DOFs)

## Project Structure

```
humanoidverse/
  agents/
    fb/              # Core Forward-Backward (FB) algorithm
      model.py       # FBModel: F(s,z,a), B(s'), actor, z latent space
      agent.py       # FBAgent: FB training loop, z-buffer, goal/tracking inference
    fb_cpr/          # FB + Contrastive Preference Recovery
      model.py       # FBcprModel: adds discriminator + critic
      agent.py       # FBcprAgent: discriminator training, mixed-z sampling, Q-learning
    fb_cpr_aux/      # FB + CPR + Auxiliary Critic (what train_bfm_zero() uses)
      model.py       # FBcprAuxModel: adds aux_critic (reward-shaped value), aux_reward_normalizer
      agent.py       # FBcprAuxAgent: actor loss combines Q_fb + Q_discriminator + Q_aux
    nn_models.py     # Residual/MLP architecture builders
    nn_filters.py    # DictInputFilter: selects sub-keys from obs dict for each network
    normalizers.py   # BatchNorm / ObsNormalizer
    buffers/         # DictBuffer, TrajectoryDictBufferMultiDim, expert data loading
  envs/
    legged_robot_motions/    # LeggedRobotMotions: main training env with motion tracking
    legged_base_task/        # LeggedRobotBase: termination, rewards, physics step
    g1_env_helper/           # G1 robot config, rewards (MuJoCo standalone), bench eval
  config/
    exp/bfm_zero/bfm_zero.yaml   # Hydra experiment config (env, robot, domain_rand, rewards, obs)
    env/legged_motions.yaml       # Env config (termination, init, noise)
    robot/g1/g1_29dof_hard_waist.yaml  # Unitree G1 29-DOF robot spec
    rewards/reward_bfm_zero.yaml       # Reward scales
    obs/bfm_zero_obs.yaml              # Observation config
    domain_rand/domain_rand.yaml       # Domain randomization
```

## Algorithm Architecture: FBcprAux

Agent hierarchy: `FBAgent → FBcprAgent → FBcprAuxAgent`

| Component | Role | Architecture |
|-----------|------|-------------|
| `_backward_map` B(s') | Encodes observations → latent z | MLP, 6×2048 residual, z_dim=256 |
| `_forward_map` F(s,z,a) | Predicts future given state, latent, action | Dual ensemble, 6×2048 residual |
| `_actor` π(s,z) | Policy conditioned on latent z | Residual MLP, 6×2048 |
| `_discriminator` D(s,z) | Distinguishes expert vs agent trajectories | MLP, 3×1024 |
| `_critic` Q(s,z,a) | Value function for discriminator reward | Dual ensemble, 6×2048 |
| `_aux_critic` Q_aux(s,z,a) | Value function for auxiliary reward shaping | Dual ensemble, 6×2048 |

### Key Mechanisms

- **z sampling**: mixture of goal-encoding (20%), expert-encoding (0%), and uniform (80%). The expert_asm_ratio=0 means expert z's are not mixed — they only come through discriminator training.
- **z relabeling**: 80% of transitions are relabeled with newly sampled z during training
- **Discriminator**: WGAN-GP style, computes reward = log D(s,z) - log(1-D(s,z))
- **Actor objective**: maximize Q_fb + λ·Q_discriminator + λ_aux·Q_aux (with uncertainty penalty for pessimism)
- **State input**: `[base_ang_vel, projected_gravity, dof_pos, dof_vel, actions, history_actor, max_local_self]`

## Exact Commands

```bash
# Train (from train.py's train_bfm_zero())
uv run python -m humanoidverse.train

# Evaluate
uv run python -m humanoidverse.tracking_inference \
    --model_folder results/bfmzero-isaac/20260407_174258 \
    --data_path humanoidverse/data/lafan_29dof.pkl \
    --save_mp4

# Lint
uv run ruff check humanoidverse/
uv run ruff format --check humanoidverse/
```

## Important Codebase Conventions

- **Formatter**: `ruff`, line-length=140
- **Python**: 3.10 only (strict pin in pyproject.toml)
- **Config stacking**: Hydra YAML for sim/env config → Pydantic dataclasses for training config → tyro CLI for inference
- **Obs filtering**: Each network (actor, critic, F, B, discriminator) uses `DictInputFilter` to select only relevant observation keys. Adding new obs must update filter configs.
- **z_latent projection**: normalized by `sqrt(z_dim)` on unit sphere for stable training
- **Checkpoint format**: safetensors + JSON metadata + train_status.json
- **Buffer**: Two replay buffers — `train` (agent rollouts) and `expert_slicer` (reference motion data)
- **Termination**: Many termination checks disabled by default — the env is designed to let the robot explore freely. Only `timeout` terminates normally.
- **Evaluation**: Eval uses the same env instance (for Isaac Sim), so env reset is forced after eval.

## Fall Recovery Training

Configured through env and domain randomization:
- `env.config.lie_down_init=True` + `lie_down_init_prob=0.3` — 30% of resets start robot lying on back
- `domain_rand.push_robots=True` — random pushes every 1-3s (up to 0.5 m/s) to destabilize
- `push_robot_recovery_time=2.0s` — motion-far termination disabled for 2s after push
- `_check_termination()` in `LeggedRobotMotions` suspends motion-far check during recovery counter

## Pretrained Checkpoints

Available on the `deploy` branch and HuggingFace. Loading:
```python
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
model = load_model_from_checkpoint_dir("checkpoint/", device="cuda")
```

## 23-DOF vs 29-DOF

29-DOF is the full robot with wrist pitch/yaw and waist roll/pitch. 23-DOF removes 6 joints for reduced complexity. See `23DOF_CONVERSION_SUMMARY.md` for the joint mapping.

## Per-Environment Auxiliary Reward Scaling

In `FBcprAuxAgentConfig.aux_rewards_scaling`, penalties are negated and scaled. The same aux rewards are also computed in the env reward function, but the aux_critic operates on these separately:
```python
aux_rewards_scaling={
    'penalty_action_rate': -0.1, 'penalty_feet_ori': -0.4,
    'penalty_ankle_roll': -4.0, 'limits_dof_pos': -10.0,
    'penalty_slippage': -2.0, 'penalty_undesired_contact': -1.0,
    'penalty_torques': 0.0, 'limits_torque': 0.0
}
```
