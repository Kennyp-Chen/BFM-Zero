# PiPlus 22DoF AMP Stage2 Design

## Goal

Add an independent PiPlus H0W 22DoF AMP Stage2 training and playback path while
preserving the existing no-AMP `speed_stage2` implementation and checkpoints.

## Architecture

The new trainer reuses the current AMP Stage2 algorithm for Command Encoder,
AMP discriminator, MimicLite locomotion reward, latent direction prior,
timeout-aware GAE, PPO, metrics, and distributed gradient synchronization.
Robot-specific integration is isolated in a new entrypoint:

```text
command + H0W actor observation
        -> CommandEncoderPolicy
        -> frozen 22DoF ONNX BFM decoder
        -> H0W locomotion environment
        -> AMP/loco/environment/latent rewards
        -> PPO update of CommandEncoderPolicy
```

The 22DoF `model.safetensors` is retained as the base-model contract artifact.
The exported `FBcprAuxModel.onnx` is the executable frozen decoder, loaded by
`OnnxPiPlusH0WDecoder`. The existing speed-stage2 entrypoint remains untouched.

## Robot Contract

- Robot config: `PiPlus_S_12L8A0G2H0W.yaml`.
- XML/MJCF: `humanoidverse/data/robots/piplus_h0w/xml/piplus_h0w_bfm.xml`.
- URDF override: `humanoidverse/data/robots/piplus_h0w/urdf/PiPlus_S_12L8A0G2H0W.urdf`.
- Action and policy joint dimension: 22.
- Decoder input: state 50 + last action 22 + actor history 288 + latent 256 = 616.
- AMP expert feature: root local velocity 3 + five key-body positions 15 + eight
  frames of 22DoF joint positions 176 = 194.
- Key bodies: left/right ankle roll, left/right elbow, and head pitch.
- Latent reference bank: the decoder's matching `tracking_inference/zs_8.pkl`,
  validated to contain 256D vectors with norm 16.

## Training Semantics

The decoder is always in evaluation/frozen mode. The policy samples a Gaussian
raw latent and calls `decoder.project_z()` before `decoder.act()`. The AMP
discriminator trains on 194D online/expert features. The total reward keeps the
current AMP implementation, including the latent direction penalty:

```text
r = env_weight * env_reward
  + locomotion_weight * mimiclite_reward
  + amp_weight * amp_quadratic_reward(score)
  - latent_prior_weight * latent_direction_penalty
```

The new checkpoint metadata uses a distinct task name so it cannot be mistaken
for a no-AMP speed-stage2 checkpoint. Playback uses deterministic latent means,
the same H0W decoder, and MuJoCo viewer configuration, but accepts only the new
AMP task metadata.

## Verification

Verification has three layers:

1. Static contract validation: 22DoF action dimension, URDF/MJCF body names,
   decoder input/output shapes, expert feature shape, and latent bank shape.
2. CPU/MuJoCo smoke: one environment, one rollout, one discriminator update,
   one PPO update, and a loadable AMP checkpoint.
3. Playback smoke: load the generated checkpoint and execute a bounded
   deterministic rollout with the H0W decoder. Existing speed-stage2 tests and
   files remain part of the regression surface.
