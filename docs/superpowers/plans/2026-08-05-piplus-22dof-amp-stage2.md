# PiPlus 22DoF AMP Stage2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate H0W 22DoF AMP Stage2 trainer and playback script without modifying the existing no-AMP speed Stage2 path.

**Architecture:** Reuse the current AMP core classes and PPO/reward logic through imports, and provide a 22DoF adapter around the existing H0W ONNX decoder, H0W environment, expert motion data, and latent bank. Save checkpoints with a new task identifier and use a separate playback entrypoint.

**Tech Stack:** Python 3.11, PyTorch, MuJoCo, IsaacLab/Isaac Sim, ONNX Runtime, existing HumanoidVerse utilities.

## Global Constraints

- Do not modify `humanoidverse/speed_stage2.py` or `humanoidverse/speed_stage2_play.py`.
- Do not overwrite or reuse no-AMP speed-stage2 checkpoint directories.
- Preserve the current AMP algorithm unless a change is required by the 22DoF contract.
- Use the tracked H0W URDF/MJCF and the existing local ONNX decoder.
- Keep headless Isaac support and MuJoCo playback support.

### Task 1: Add the 22DoF AMP trainer

**Files:**
- Create: `humanoidverse/amp_stage2_piplus_22dof.py`
- Test: `tests/test_amp_stage2_piplus_22dof.py`

**Interfaces:**
- Consumes the decoder factory from `humanoidverse.piplus_h0w_onnx_decoder` and H0W environment builder from `humanoidverse.speed_stage2`.
- Reuses `CommandEncoderPolicy`, `AMPDiscriminator`, `PiPlusAMPExpertDataset`, `MimicLiteLocomotionRewardState`, `ppo_update`, `compute_gae`, and AMP helper functions from `humanoidverse.amp_stage2`.
- Produces `--validate-assets`, `--smoke`, `--dry-run`, `--resume`, and normal training CLI modes with task metadata `amp_stage2_piplus_22dof`.

- [ ] Write tests for the 194D H0W expert feature contract, 22DoF decoder contract, and task-specific checkpoint metadata.
- [ ] Implement H0W asset/decoder/expert/latent-bank validation.
- [ ] Implement rollout, discriminator update, reward synthesis, timeout-aware GAE, PPO update, DDP synchronization, and checkpoint save/resume.
- [ ] Run the focused unit tests and CPU dry-run.

### Task 2: Add the 22DoF AMP playback entrypoint

**Files:**
- Create: `humanoidverse/amp_stage2_piplus_22dof_play.py`
- Test: `tests/test_amp_stage2_piplus_22dof_play.py`

**Interfaces:**
- Consumes checkpoints emitted by `amp_stage2_piplus_22dof.py`.
- Uses the existing H0W MuJoCo environment, ONNX decoder, robot config, and fixed command interface.
- Produces deterministic viewer playback and bounded `--max-steps` smoke mode.

- [ ] Implement checkpoint discovery and task validation.
- [ ] Implement deterministic mean-latent playback with H0W assets.
- [ ] Run playback contract tests and bounded headless/import smoke.

### Task 3: Document the comparison and verification

**Files:**
- Create: `docs/piplus_22dof_amp_vs_ppo.md`
- Modify: `docs/piplus_22dof_speed_stage2.md` only if a cross-reference is needed; do not alter historical results.

**Interfaces:**
- Documents the exact differences between no-AMP speed Stage2 and 22DoF AMP Stage2, commands, checkpoint formats, and smoke results.

- [ ] Record the two training graphs as separate pipelines and list shared versus different components.
- [ ] Add train/play commands and generated artifact locations.
- [ ] Append smoke-test results only after commands complete successfully.
