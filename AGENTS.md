# Repository Guidelines

## What This Is

BFM-Zero (LeCAR-Lab) fork for humanoid RL training and AMP stage-2 fine-tuning. This repo is under git on branch **`HT`**: its root commit is the exact `bfm-command` tree from `HighTorque-Locomotion/HT_BFM`, and `HEAD` carries only local tweaks (`AGENTS.md`, `.gitignore`). Remotes: `origin` = `Kennyp-Chen/BFM-Zero` (your fork), `upstream` = `HighTorque-Locomotion/HT_BFM` (has `bfm-command`; `origin`'s `main` is unrelated — it tracks `LeCAR-Lab/BFM-Zero`). Motion-data fallback: `https://huggingface.co/LeCAR-Lab/BFM-Zero`. New reusable procedures belong in `docs/` (runbooks already exist for AMP stage-2, task migration, checkpoint playback).

## Environment

- Canonical env is Conda env **`HT_BFM`** (python 3.11, torch 2.7+cu128, mujoco 3.8.1, isaacsim 5.1.0.0, IsaacLab editable from `../IsaacLab2.3`) defined in `environment.yml`. **It is not yet created on this machine** (only `base` exists).
- `pyproject.toml`/uv is stale for this fork: it pins python `==3.10.*`, isaaclab 2.0.2 / isaacsim 4.5.0, so `uv sync` produces a different env than the Conda one. Prefer Conda; only use uv if the task's deps match pyproject.
- Run entrypoints from the repo root with the env active.

## Layout

- `humanoidverse/` — the package. `train.py` is the tyro/Hydra training entry (wandb, torchrun multi-GPU). `agents/` = FB / FB-CPR / FB-CPR-AUX agents, buffers, wrappers; `envs/` = `legged_base_task`, `legged_robot_motions`, `piplus_env_helper`, `g1_env_helper`; `simulator/` = `base_simulator`, `isaacgym`, `isaacsim`, `mujoco`, `genesis`; `utils/` = `asset_paths.py`, `motion_lib/`, `torch_utils.py`.
- `humanoidverse/config/` — Hydra YAML. `base.yaml` composes `base/hydra`, `base/structure`, `callbacks/model_save`, `callbacks/autoresume`. Groups: `exp/` (`bfm_zero`, `bfm_zero_h1`, `bfm_zero_piplus`), `robot/` (`g1`, `Hi`, `piplus`), `env/`, `simulator/` (`isaacsim.yaml`, `mujoco.yaml`), `obs/`, `rewards/`, `terrain/`, `callbacks/`, `domain_rand/`. Valid robot identifiers are the `SUPPORTED_ROBOTS` tuple in `train.py` (e.g. `PiPlus_S_12L8A0G2H1W_LSE`, `Hi_P_12L10A0G2H1W_260402`).
- `data_process/` — motion-conversion / dataset-augmentation scripts (piplus/GMR/AMP); generated datasets go to `data_process/dataset` (gitignored).
- `tests/` — unittest smoke tests for `amp_stage2` / `amp_stage2_play`. They import `humanoidverse.amp_stage2`, so they need the full env (not just MuJoCo).
- `tuning-log.md` — chronological record of distributed-training tuning on remote machines; append to it when changing training hyperparameters or launching a new run.
- `static/`, `model/`, `logs/` (default Hydra `base_dir`), `humanoidverse/data`, `wandb/` are generated artifacts — data, not source.

## Commands

```bash
conda activate HT_BFM
python -m humanoidverse.train --help                # training CLI (tyro)
python -m humanoidverse.tracking_inference --help   # tracking inference + ONNX export
python -m humanoidverse.goal_inference --help
python -m humanoidverse.reward_inference --help
python -m humanoidverse.amp_stage2 --help           # stage-2 AMP fine-tune (frozen PiPlus BFM)
python -m humanoidverse.amp_stage2_play --help      # rollout/playback of stage-2 policy
python -m unittest discover -s tests                # smoke tests
ruff check humanoidverse data_process               # lint (140-col, import sorting; E402/E731 ignored)
```

- Inference scripts: `--simulator mujoco` runs without Isaac Sim/Isaac Lab; `--no-headless` shows the viewer; `--save_mp4` renders videos. Isaac Sim paths need GPU + Linux.
- Distributed training runs via `torchrun --nproc_per_node=...`; consult `tuning-log.md` before touching distributed sync code (flat-gradient buckets, checksums, SyncBatchNorm were added there and are load-bearing).

## Git & Network

- GitHub bulk traffic is throttled from this machine (HTTPS pack transfers crawl at ~20-30KB/s; a full 1GB clone dies mid-transfer). `~/.ssh/config` routes `github.com` through `ssh.github.com:443`, which handles small/medium packs fast (a ~114MB push took 30s). Never attempt full fetches of `upstream` (~1GB); if only refs/trees are needed use `git fetch --filter=blob:none upstream bfm-command` (completed in ~10s).
- Keep commits small with short imperative lowercase summaries. Work happens on branch `HT`; push with plain `git push origin HT` (force-push only deliberately).

## Gotchas

- The LaFan `.pkl` motion files are **absent** here: they are LFS-tracked (`.gitattributes`) and gitignored. Fetch via `git lfs pull` or the HF mirror. Robot assets under `humanoidverse/data/robots/` and `model/` ARE tracked on this branch (added with `git add -f`), but `.gitignore`'s `humanoidverse/data` rule still hides the former from plain `git add` — use `git add -f humanoidverse/data/robots` to update them. `config/env/` is tracked normally since `.gitignore`'s `env/` was anchored to `/env/`.
- `amp_stage2.py` hardcodes default paths (`huiying/.../checkpoint`, `dataset/.../run.pkl`, `0803陈建宏23dof2.zip`) that do **not** exist in this working copy — always pass explicit `--model_folder`, dataset, and robot paths.
- Do not commit secrets, private machine paths (e.g. `/data/laihuiying/...`), checkpoints, or regenerated logs. Keep robot asset, motion-data, and config names in sync so paths resolve in both Isaac Sim and MuJoCo.
- Style: python 3.10/3.11-compatible, `snake_case`, four-space indent, ruff 140-col limit; match existing Hydra naming for new YAML/robot identifiers.
