# UFO AMP Stage2 -> HT_BFM PiPlus

## Migration brief

```yaml
source:
  repo_or_path: /home/sunteng/Project/UFO
  framework: custom BFM + MuJoCo/MJLab
  training_entry: humanoidverse.amp_stage2
  play_or_eval_entry: humanoidverse.amp_stage2_play
  task_name: H1 AMP command-conditioned locomotion
  robot_name: H1 260402
  algorithm_family: AMP + PPO command encoder

target:
  ht_lab_root: /home/sunteng/Project/HT_BFM
  task_name: PiPlus LSE AMP Stage2
  robot_name: PiPlus_S_12L8A0G2H1W_LSE
  training_entry: humanoidverse.amp_stage2
  training_backend: target HumanoidVerse Isaac wrapper
  integration_style: hybrid
```

## Contract mapping

| UFO Stage2 contract | HT_BFM implementation |
| --- | --- |
| Frozen BFM actor | `FBcprAuxModel` loaded from `checkpoint/model` |
| BFM actor observation | `state + last_action + history_actor` from `HumanoidVerseIsaacVectorEnv` |
| Latent | checkpoint `cfg.archi.z_dim` (the supplied model is 256) |
| Latent projection | checkpoint `FBModel.project_z()`; no second normalization |
| Action | checkpoint action dimension; supplied PiPlus model and environment both use 23 |
| Online environment | existing PiPlus `HumanoidVerseIsaacConfig` / IsaacLab backend |
| AMP expert data | `dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run_with_stand.pkl` |
| AMP feature | local root velocity (3), five local key-body positions (15), eight-frame joint history (184), total 202; key bodies are ankles, elbows, and head pitch because Isaac merges fixed wrist links into elbows |
| Trainable modules | command encoder, PPO value head, WGAN-GP discriminator, AMP reward normalizer |
| Frozen modules | BFM actor, backward/forward maps, critics, observation normalizers |

## Deliberate target adaptations

UFO's MJLab-only `HumanoidVerseMjlab`, robot-spec, and distributed helper modules are
not copied. The target uses the existing Isaac vector wrapper and resolves the PiPlus
asset/config from this repository. Motion-tracking reset callbacks are replaced at
runtime by the base locomotion reset callbacks so command-conditioned rollouts are not
forced back onto a reference motion after a termination. The motion library is still
loaded once by the target environment and the supplied run dataset is independently
converted into AMP expert features.

The supplied checkpoint is model-only. It contains no BFM optimizer or replay buffer;
this is sufficient for Stage2 because every BFM parameter is frozen.

## Validation gates

1. Import and unit-test the pure command encoder, discriminator, GAE, and feature builder.
2. Load the supplied model checkpoint and verify action/latent dimensions.
3. Run `--dry-run` without starting Isaac Sim.
4. Run `--smoke` with a small Isaac environment when IsaacLab/GPU is available.

## Multi-GPU status

The repository's generic `config/base/fabric.yaml` is a dormant Lightning
Fabric template (`base.yaml` keeps `multi_gpu: False`) and is not wired into
the existing training entrypoint. Stage2 therefore owns a small, standard
PyTorch distributed launcher in `humanoidverse/amp_stage2.py`: it uses one
Isaac environment per rank, averages trainable gradients, broadcasts initial
trainable state, reduces scalar metrics, and writes checkpoints only from
rank 0. It requires only PyTorch's built-in `torch.distributed.run`; no
`torchrunx` package is needed.

## Server training

Run from the HT_BFM repository on the training server after copying the project,
the model-only checkpoint directory, and the AMP dataset. Use `--gpu-ids all`
for the built-in standard PyTorch torchrun launcher; use `--gpu-ids single`
for one GPU. The Stage2 launcher performs `torch.distributed` process-group
initialization, synchronous gradient averaging, rank-0 checkpoint writes, and
scalar metric reduction.

```bash
python -m humanoidverse.amp_stage2 \
  --bfm-checkpoint huiying/bfmzero-piplus-lse-isaac-20260715_143758\(1\)/checkpoint \
  --expert-dataset dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run_with_stand.pkl \
  --robot-config humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H1W_LSE.yaml \
  --device cuda --gpu-ids all \
  --num-envs 1024 --iterations 10000 --rollout-steps 32 \
  --work-dir logs/amp_stage2_piplus_lse
```

Before a long run, execute `python -m humanoidverse.amp_stage2 --dry-run --device cpu`.
The model loader requires the device literal `cuda` in the checkpoint config;
the script handles local `cuda:0` selection internally.
