## 2026-07-21 09:17:11 CST - Restart 4-GPU H1 BFM with synchronized gradients and half-width networks

- Context: remote run bfmzero-h1-p12l10-isaac-20260720_205512; tmux h1_bfm; remote torchrun --nproc_per_node=4
- Phenomenon: 4-GPU run had manual gradient all-reduce with grad=None collective risk, local BatchNorm/EMA statistics, no parameter consistency check, clip_grad_norm=0, and actor/Q scale drift (actor_loss about 17381, Q1 about -212, Q_fb about 1736 at about 13.7M timesteps).
- Analysis: The main gradient all-reduce covers the trainable modules, but local normalization statistics make per-rank gradients differ; dynamic Q_fb scaling and missing clipping can amplify critic/actor drift.
- Adjustment: All-reduce zero-filled gradients for a fixed parameter order; use SyncBatchNorm and distributed EMA batch statistics; add periodic cross-rank parameter checksum validation; clip_grad_norm 0.0 -> 1.0; clamp Q_fb-based actor regularizer weight to [1,10]; hidden_dim f/actor/critic/aux 2048 -> 1024, backward 256 -> 128, discriminator 1024 -> 512; restart from a fresh workdir.
- Rationale: Make each rank compute comparable normalized gradients, detect silent parameter divergence, reduce actor gradient spikes, and test a lower-capacity model.
- Expected effect: No collective deadlock from unused parameters; synchronized normalization; bounded actor updates; lower memory and potentially more stable early learning. Watch critic/Q drift, mean_aux_reward, disc reward, termination/time_out, and checksum warnings.
- Result: Patched files passed local and remote py_compile; old run was stopped after preserving its code/results; fresh run `bfmzero-h1-p12l10-isaac-20260721_092347` is alive in tmux `h1_bfm_half_20260721_092347`. Isaac initialization completed and reached `Starting training`/initial evaluation at timestep 0; first update metrics are pending.
- Files/commands: remote humanoidverse/train.py; humanoidverse/agents/fb/agent.py; humanoidverse/agents/fb_cpr/agent.py; humanoidverse/agents/fb_cpr_aux/agent.py; humanoidverse/agents/normalizers.py; humanoidverse/agents/nn_models.py

## 2026-07-21 12:15:22 CST - Restart H1 BFM with explicit NCCL devices, flat gradient buckets, and Qfb diagnostics

- Context: remote run bfmzero-h1-p12l10-isaac-20260721_092347; tmux h1_bfm_half_20260721_092347; 4 GPUs
- Phenomenon: Training reached about 8.3M timesteps without collective deadlock, but Q_fb grew from about 11.5 to 1710, Q1 from about -1 to -279, critic_loss to about 11.9, and mean_disc_reward worsened to about -5.66; NCCL reported device mapping warnings.
- Analysis: Gradient averaging is operating, but initialization relies on implicit NCCL device selection, manual per-parameter collectives remain fragile, and the direct actor -Q_fb term is not bounded by the regularizer clamp. Missing gradient/action/term metrics make the source of drift difficult to distinguish.
- Adjustment: Add init_process_group(device_id) and barrier(device_ids); replace per-parameter all_reduce with fixed 64MB flat gradient buckets and zero-fill missing grads; replace sum/square checksum with exact rank0-broadcast max-absolute parameter comparison every 100 updates; log gradient norms, action std, and actor Qfb term decomposition; start a fresh run.
- Rationale: Remove implicit rank-device behavior, make collective order independent of grad presence, detect true parameter divergence, and expose whether actor drift comes from Qfb, discriminator, auxiliary, or clipping.
- Expected effect: No NCCL mapping warnings or collective hangs; exact replica consistency checks; actionable Qfb/gradient diagnostics; watch direct actor_fb_term and Qfb growth.
- Result: Pending fresh remote restart
- Files/commands: remote humanoidverse/train.py; agents/fb/agent.py; agents/fb_cpr/agent.py; agents/fb_cpr_aux/agent.py; local logs/remote_launch/h1_bfm_4096_4gpu_half_20260721_092347.log

## 2026-07-21 12:34:25 CST - Flat-sync restart smoke validation

- Context: fresh remote run bfmzero-h1-p12l10-isaac-flatb-20260721_122851; tmux h1_bfm_flatb_20260721_122851
- Phenomenon: The first flat-gradient version remained in initialization before Starting training; model-state broadcast was still per-parameter.
- Analysis: The initialization path needed the same fixed-bucket treatment as gradient synchronization. After replacing per-parameter model parameter/buffer broadcast with 64MB buckets, startup completed and the first training interval ran.
- Adjustment: Added fixed 64MB model parameter/buffer broadcast buckets; retained explicit NCCL device_id/barrier device_ids, flat gradient buckets, exact parameter comparison, and diagnostic metrics.
- Rationale: Reduce startup collective count and keep all distributed state transfers on deterministic fixed buckets.
- Expected effect: Fast reliable startup, no implicit NCCL device warnings, no collective hangs; exact parameter checks and Qfb decomposition available during training.
- Result: Passed smoke validation: fresh run reached 102400 timesteps; actor_grad_norm=2.986, actor_action_std=0.853, actor_fb_term=-11.622, actor_disc_term=0.636, actor_aux_term=1.121, forward_grad_norm=168.116, backward_grad_norm=64984.9; no NCCL/checksum/runtime errors observed.
- Files/commands: remote humanoidverse/train.py; agents/fb/agent.py; agents/fb_cpr/agent.py; agents/fb_cpr_aux/agent.py; tmux h1_bfm_flatb_20260721_122851

## 2026-07-21 15:04:17 CST - Resume with reduced forward-map learning rate

- Context: remote run bfmzero-h1-p12l10-isaac-flatb-20260721_122851; tmux h1_bfm_flatb_resume_lrf_20260721_1458; resume checkpoint time 8671232
- Phenomenon: Q_fb increased from about 1620 at 7.17M to about 1768 at 8.40M, actor_fb_term remained dominant, forward_grad_norm about 1200, and critic_loss about 12.2; multi-GPU communication showed no errors.
- Analysis: The actor learning rate was already reduced to 5.24288e-5, so repeated actor-rate reduction is unlikely to address the remaining Qfb scale drift. The forward map directly produces Q_fb and still used lr_f=3e-4.
- Adjustment: lr_f: 0.0003 -> 0.00024 in humanoidverse/train.py and the resume checkpoint config.json; checkpoint and source backups created under backups/codex_tune_lr_f_20260721_145524
- Rationale: Reduce forward-map update magnitude by 20 percent while preserving model architecture, task interfaces, replay buffer, and checkpoint compatibility.
- Expected effect: Slow Qfb representation-scale growth and reduce forward-gradient-driven critic/actor drift; watch Q_fb, fb_loss, actor_fb_term, critic_loss, and imitation reward.
- Result: Resume started from checkpoint time 8671232; first post-resume metric window pending.
- Files/commands: remote tmux h1_bfm_flatb_resume_lrf_20260721_1458; console logs/remote_launch/h1_bfm_4096_4gpu_flatb_resume_lrf_20260721_1458.log

## 2026-07-21 15:15:44 CST - Make resumed forward-map learning rate effective

- Context: remote run bfmzero-h1-p12l10-isaac-flatb-20260721_122851; tmux h1_bfm_flatb_resume_lrf2_20260721_1509; checkpoint time 8671232
- Phenomenon: The first resume loaded checkpoint config lr_f=0.00024 but optimizers.pth still contained forward_optimizer.lr=0.0003, so the intended parameter change was not active.
- Analysis: FBAgent.load constructs the optimizer from checkpoint config and then loads optimizer state, whose param-group learning rate overrides the constructor value.
- Adjustment: Backed up optimizers.pth and changed only forward_optimizer.param_groups[0].lr: 0.0003 -> 0.00024; restarted from the same checkpoint.
- Rationale: Ensure the selected learning-rate change is applied without changing optimizer moments, model weights, architecture, or other optimizer learning rates.
- Expected effect: The resumed forward-map updates should be 20 percent smaller; evaluate Q_fb slope, forward_grad_norm, fb_loss, critic_loss, and imitation reward over multiple logging windows.
- Result: Second resume is active; first post-resume window at 102400 additional steps has Q_fb=1801.58, actor_fb_term=-1801.58, forward_grad_norm=1105.29, critic_loss=15.45, and no distributed/runtime errors. Long-term effect pending.
- Files/commands: remote backup /home/zhuzejian/Project/HT_BFM/backups/codex_tune_lr_f_20260721_145524; logs/remote_launch/h1_bfm_4096_4gpu_flatb_resume_lrf2_20260721_1509.log

## 2026-07-21 15:53:05 CST - Evaluate reduced forward-map learning rate

- Context: remote run bfmzero-h1-p12l10-isaac-flatb-20260721_122851; tmux h1_bfm_flatb_resume_lrf2_20260721_1509; latest saved checkpoint time 10166272; console through timestep 10412032
- Phenomenon: After lr_f=0.00024 became active, Q_fb rose from about 1801.6 at the first 102400-step window to 1909.5 by timestep 10412032. mean_disc_reward remained about -5.86 and actor_action_std declined to 0.808.
- Analysis: The Qfb growth slope appears lower than the pre-change slope (roughly 65-80 per million steps versus about 100 per million), so the change slows the drift but does not stop it. Critic_loss remains about 12-15 and actor_fb_term still dominates actor_loss.
- Adjustment: No new parameter change; keep lr_f=0.00024 for continued observation.
- Rationale: The parameter shows partial stabilization evidence but needs a longer window before another change is selected.
- Expected effect: Qfb slope should flatten further or stabilize; watch absolute Qfb, critic_loss, actor_grad_norm, action std, and mean_disc_reward.
- Result: Partial improvement in drift rate, but overall value/actor scale is still regressing; training remains active and distributed execution is stable.
- Files/commands: logs/remote_launch/h1_bfm_4096_4gpu_flatb_resume_lrf2_20260721_1509.log; logs/remote_runs/bfmzero-h1-p12l10-isaac-flatb-resume-lrf2-20260721_1509/checkpoint/model/model.safetensors

## 2026-07-22 15:30:47 CST - Prepare second forward-map learning-rate reduction

- Task: H1 BFM whole-body motion tracking over 862 reference motions; optimize tracking accuracy while preserving upright posture, smooth actions, low torque/slippage, and physically plausible contacts.
- Context: pulled run `bfmzero-h1-p12l10-isaac-flatb-resume-lrf2-20260721_1509`; latest local console timestep 55263232; latest complete local checkpoint 55017472; latest tracking evaluation timestep 47075328.
- Primary metrics: tracking MPJPE improved 1377.52 -> 1333.55 from 37.47M to 47.08M (-3.19%), and proximity improved 0.96743 -> 0.96945. Guardrails also improved over training: mean_aux_reward about -1.39 -> -0.61, action-rate penalty about 5807 -> 800, torque penalty about 17214 -> 8809, and undesired-contact penalty about 0.905 -> 0.687.
- Classification: task performance is still improving but is near a plateau; distributed execution is stable. Learning health remains regressing: Q_fb about 1924 -> 2809, Q1 about -291 -> -365, critic_loss about 12.4 -> 22.9, mean_disc_reward about -5.81 -> -6.70, and actor_action_std about 0.804 -> 0.516.
- Bottleneck: forward-map value scale continues to grow and dominate the actor objective, although the Q_fb growth rate slowed materially after the prior lr_f reduction.
- Selected checkpoint: 55017472 from the active lineage, because it is the latest fully synchronized checkpoint after the best available 47.08M tracking evaluation and has no NaN, NCCL, checksum, OOM, or runtime errors.
- Single parameter change: forward-map learning rate `lr_f` 0.00024 -> 0.000192 (-20%). Rollback value: 0.00024.
- Changed file: `humanoidverse/train.py`; added `--resume_lr_f` so the selected value is applied after checkpoint optimizer state loading and is recorded in the new run config. Local validation passed with `py_compile`, CLI help, and a checkpoint-load unit smoke test.
- Hypothesis: a second conservative 20% reduction will further flatten Q_fb growth and lower critic/actor gradient pressure without changing the actor, discriminator, auxiliary critic, architecture, replay buffer, task interface, or tracking reward.
- Prepared patch: `logs/tuning/h1_bfm_resume_lr_f_20260722_1530.patch`.
- Intended remote backup: `/home/zhuzejian/Project/HT_BFM/backups/codex_tune_lr_f_20260722_1530/humanoidverse/train.py`; intended remote patch: `/tmp/h1_bfm_resume_lr_f_20260722_1530.patch`.
- Intended resume: 4-GPU torchrun from checkpoint 55017472 with `--resume_lr_f 0.000192`, preserving replay buffer and existing H1/4096-env settings.
- Result: remote deployment and restart are blocked because `zhuzejian@192.168.21.99` rejected the current SSH agent, Ed25519 key, and RSA key. No remote file or training process was changed. After authentication is restored, verify Q_fb slope, critic_loss, actor_grad_norm, forward_grad_norm, mean_disc_reward, actor_action_std, and the next tracking evaluation over at least 3-5 logging windows.

## 2026-07-22 16:16 CST - Deploy lr_f override and resume 4-GPU H1 BFM

- Authentication: user supplied the remote password in the active conversation; it was used only for live SSH/rsync calls and was not written to disk.
- Remote state: `/home/zhuzejian/Project/HT_BFM` is not a Git worktree, so deployment used SHA-256 source verification, timestamped file backups, a unified patch, and `py_compile` instead of Git status/apply checks.
- Latest task metrics before restart: tracking MPJPE improved 1333.55 -> 1279.84 -> 1223.57 at 47.08M, 56.68M, and 66.28M (-4.03% and -4.40%); proximity improved 0.96945 -> 0.97529 -> 0.98203. This supported resuming from the newest stable checkpoint instead of rolling back.
- Selected checkpoint: `70266880`, verified stable before shutdown; old tmux `h1_bfm_flatb_resume_lrf2_20260721_1509` was stopped with Ctrl-C and all torchrun workers exited before restart.
- Single parameter change: forward-map learning rate `lr_f` 0.00024 -> 0.000192 (-20%). No actor, critic, discriminator, reward, environment, architecture, replay-buffer, or task-interface parameter changed.
- Implementation: added `--resume-lr-f`, applied it after optimizer-state loading, and used Pydantic `model_copy(update=...)` so frozen config objects and the serialized run config both contain the effective value. The first launch exposed a frozen-config assignment error before training updates began; it exited without changing the checkpoint, and the corrected implementation was deployed and revalidated.
- Remote backup: `/home/zhuzejian/Project/HT_BFM/backups/codex_tune_lr_f_20260722_1600/humanoidverse/train.py`; failed intermediate version: `train.failed_frozen.py`; patch: `/tmp/h1_bfm_resume_lr_f_20260722_1604_fix.patch`.
- Active run: tmux `h1_bfm_flatb_resume_lrf3fix_20260722_1605`; console `logs/h1_bfm_4096_4gpu_flatb_resume_lrf3fix_20260722_1605.log`; exact launch adds `--resume-lr-f 0.000192` to the prior 4-GPU 4096-env resume command.
- Verification: all four ranks loaded checkpoint 70266880, printed `Applied resume override after optimizer load: lr_f=0.000192`, restored the replay buffer, and entered `Starting training`. The run config records both `resume_lr_f=0.000192` and `agent.train.lr_f=0.000192`; no NCCL, checksum, OOM, or runtime errors after the corrected restart.
- Initial result over three 102400-step windows: Q_fb = 2937.07, 2946.44, 2952.89 (mean 2945.46; approximately flat versus 2942.49 immediately before restart); forward_grad_norm = 2388.34, 2456.92, 2415.52 (mean 2420.26, about -12.9%); actor_grad_norm = 82.87, 83.94, 86.37 (mean 84.39, about -6.3%); actor_action_std mean 0.518 versus 0.503 before restart; critic_loss mean 21.86 versus 20.30 before restart, so critic improvement is not yet established.
- Latest complete post-change checkpoint: `70565888`; local artifacts: `logs/remote_runs/bfmzero-h1-p12l10-isaac-flatb-resume-lrf3-20260722_1605/`, `logs/remote_launch/h1_bfm_4096_4gpu_flatb_resume_lrf3fix_20260722_1605.log`, and `logs/tuning/h1_bfm_resume_lr_f_20260722_1604_fix.patch`.
- Follow-up: keep this single change running and compare Q_fb slope and critic_loss over at least 10-20 windows plus the next 9.6M-step tracking evaluation. Roll back to 0.00024 only if tracking or safety metrics regress materially while critic/Q health does not improve.

## 2026-08-01 10:34:39 CST - Stage2 AMP synchronized KL early-stop restart

- Context: fresh remote run `amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_klfix_sync_scratch` on GPU10 physical GPUs 5-8, four workers, 1024 environments per GPU.
- Phenomenon: the first KL-guarded run reached iteration 1, then stopped on an NCCL collective timeout because each rank made an independent early-stop decision.
- Adjustment: synchronize minibatch KL with an all-reduce before the early-stop decision; commit `3f9e4c6`.
- Result: the restarted run reached iteration 37 with seven live Stage2 processes and no Traceback/NCCL/OOM/disk errors. Approx KL fell from first-10 mean `0.01125` to last-10 `0.00756`; clip fraction fell `0.16024 -> 0.11385`; termination rate stayed near zero (`0.000116 -> 0.000003`). AMP score improved `0.18890 -> 0.51799`, while velocity tracking reward `linvel_exp` remained approximately flat (`0.01586 -> 0.01600`).
- Assessment: stability and AMP discriminator alignment are improving; velocity tracking has not yet shown a clear upward trend and needs more iterations before reward tuning.

## 2026-08-01 11:42:27 CST - Resume Stage2 AMP from checkpoint 300 with reduced AMP weight

- Context: amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_klfix_sync_scratch; GPU10 physical GPUs 5-8; 4 workers; 1024 envs/GPU; resume checkpoint_300.pt
- Phenomenon: At iteration 1416 the run remained physically stable with termination/fall rates near zero, but velocity tracking reward linvel_exp stayed around 0.0151-0.0152 and amp_score degraded to about -2.1; amp reward contribution remained negative.
- Analysis: Stability is not the bottleneck. The negative discriminator-derived AMP reward is competing with the command-conditioned locomotion objective and can suppress the speed-tracking gradient.
- Adjustment: amp_weight: 0.04 -> 0.02; resume from checkpoint_300.pt. No other parameter or architecture change.
- Rationale: Reduce the harmful negative AMP contribution by 50 percent while retaining AMP as a shaping signal and preserving the frozen BFM, discriminator state, optimizer states, environment, and PPO settings.
- Expected effect: linvel_exp and locomotion reward should improve without increasing termination/fall rates; monitor amp_score, amp_reward, speed tracking, termination rates, KL, and value loss.
- Result: Pending remote restart and follow-up window.
- Files/commands: humanoidverse/amp_stage2.py CLI parameters; remote run logs/amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_ampw002_resume300; checkpoint_300.pt

## 2026-08-01 11:45:26 CST - Stage2 AMP weight 0.02 resume startup validation

- Context: remote run amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_ampw002_resume300; launcher PID 658031; resume checkpoint_300.pt
- Phenomenon: The new run loaded iteration 300 and reached iteration 319 with all four workers alive. After the first reset-heavy batch, termination/crash/fall rates returned to zero.
- Analysis: The single parameter change is active and the restart is operationally stable. AMP contribution is approximately halved as intended; the window is too short to classify velocity-tracking improvement.
- Adjustment: No additional change; keep amp_weight=0.02 and continue training.
- Rationale: Preserve single-change attribution and collect a comparable 100-iteration window before another intervention.
- Expected effect: Compare iterations 300-399 against the prior run window for linvel_exp, angvel_z_exp, amp_score, AMP contribution, total reward, and termination rates.
- Result: Startup passed through iteration 319; no Traceback, NCCL, CUDA OOM, or runtime error observed.
- Files/commands: remote config logs/amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_ampw002_resume300/config.json; launcher.log

## 2026-08-01 12:34:24 CST - Assess latest Stage2 AMP improvement evidence

- Context: PiPlus AMP Stage2; synchronized-KL run through iteration 1416; amp_weight=0.02 resume verified through iteration 319; latest locally available status 2026-08-01 11:45 CST
- Phenomenon: The synchronized-KL run removed the rank-divergent early-stop deadlock and remained physically stable, but linvel_exp stayed near 0.0151-0.0152 and amp_score later degraded to about -2.1. The amp_weight=0.02 resume started cleanly and reached iteration 319 with zero post-reset termination/crash/fall rates.
- Analysis: Operational stability clearly improved after synchronizing KL. Halving AMP weight reduced the harmful AMP contribution as designed, but the available post-change window is too short to establish improvement in velocity tracking or total task performance.
- Adjustment: No parameter or process change in this assessment.
- Rationale: Preserve single-change attribution until a comparable 100-iteration or longer post-resume window is available.
- Expected effect: If the change helps, iterations 300-399 and later should show higher linvel_exp and locomotion reward without increased termination, while AMP contribution remains less negative.
- Result: Partial improvement: distributed/runtime and physical stability improved; velocity-tracking improvement is unproven. Live remote verification was unavailable because SSH authentication to zhuzejian@192.168.21.99 was rejected.
- Files/commands: tuning-log.md; local checkpoint logs/amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_klfix_sync_scratch/checkpoint_300.pt; attempted read-only SSH status check

## 2026-08-01 12:41:33 CST - Evaluate latest GPU10 Stage2 log via SFTP

- Context: GPU10 via j4.tx.ppp.clicki.cn; latest visible artifact ufo_pull_stage2_v7_latest_5100_20260731; H1 25DoF UFO Stage2, resumed iteration 2600
- Phenomenon: Latest accessible log reaches iteration 5116. Mean reward rose from 0.0245 at iterations 2600-2699 to 0.0866 at 3000-3099, then fell to 0.0696 at 4000-4099 and 0.0561 at 5000-5116. linvel_exp stayed nearly flat at 0.0204 -> 0.0208 -> 0.0209 -> 0.0203. AMP score improved from -0.824 to about 1.3, then regressed to 0.359; AMP reward contribution became negative again (-0.0117). Termination rate increased from near zero to about 0.000414.
- Analysis: There was a transient mid-run improvement, but the latest window shows regression rather than continued improvement. The locomotion speed term has not established a positive trend, while AMP alignment and total reward have weakened. KL remains around 0.03 and value loss rose, indicating ongoing update pressure.
- Adjustment: No parameter or process change; read-only remote inspection.
- Rationale: Separate the latest visible H1 run from the local PiPlus amp_weight=0.02 lineage and avoid attributing its metrics to PiPlus.
- Expected effect: A genuinely improved run should sustain or increase linvel_exp and total reward over successive 100-iteration windows while keeping AMP contribution non-negative and termination low.
- Result: No continued improvement in the latest accessible GPU10 artifact; transient gain followed by regression. The current PiPlus amp_weight=0.02 run could not be verified because this restricted SFTP view did not expose its directory.
- Files/commands: SFTP read-only: ufo_pull_stage2_v7_latest_5100_20260731/{train.log,config.json}; no remote writes or restarts

## 2026-08-01 12:45:56 CST - Prepare GPU10 Stage2 stability and velocity-tracking resume

- Context: GPU10 via j4.tx.ppp.clicki.cn; visible H1 25DoF UFO Stage2 lineage ufo_pull_stage2_v7_latest_3000_20260730 and ufo_pull_stage2_v7_latest_5100_20260731
- Phenomenon: Within iterations 2600-5116, reward and linvel_exp peaked around iterations 3300-3600 (linvel_exp about 0.02193-0.02197; reward about 0.099) and then regressed by iterations 5000-5116 (linvel_exp about 0.02028; reward about 0.056; amp_reward_contribution about -0.0117; termination_rate about 0.0004).
- Analysis: The residual bottleneck is late-training AMP incentive drift: AMP score and AMP contribution deteriorate together with total reward while directed velocity tracking remains flat. Lowering AMP reward weight should reduce the harmful shaping pressure without changing observations, action space, robot, or checkpoint architecture.
- Adjustment: Proposed single change: amp_reward_weight 0.04 -> 0.02. Proposed resume checkpoint: checkpoint_3000.pt, same lineage/config as checkpoint_5100 and closest visible checkpoint before the best 3300-3600 performance window; checkpoint_3400 was not exposed by the SFTP gateway.
- Rationale: Restore a stable/improving part of the run and halve the late-stage AMP contribution while preserving locomotion reward and safety terms.
- Expected effect: Over the next 100-300 iterations, linvel_exp and total reward should stop declining or rise, AMP contribution should be less negative, and termination_rate should remain near zero. Roll back to checkpoint_5100/weight 0.04 if task performance worsens materially.
- Result: Blocked before application: j4.tx.ppp.clicki.cn authenticates only to a forced SFTP subsystem; SSH exec, interactive shell, process inspection, patch application, and tmux restart are unavailable. No remote file, process, or local training code was changed.
- Files/commands: SFTP read-only artifacts: ufo_pull_stage2_v7_latest_3000_20260730/checkpoint_3000.pt and config.json; ufo_pull_stage2_v7_latest_5100_20260731/train.log; required remote patch/restart pending a shell-capable endpoint

## 2026-08-01 13:08:12 CST - Resume H1 BFM from best stable checkpoint with half learning rate

- Context: remote z-autodl-h20-GPU10; UFO run bfm_command_24V_hi_4gpu_4096env_scratch_20260801; checkpoint time 48046080; 4 GPUs
- Phenomenon: Tracking improved sharply by 3.2M steps, then plateaued through 48.0M: all-motion MPJPE about 2321.4 -> 2313.9 mm, velocity error 3113.9 -> 3101.0, proximity 0.788 -> 0.791; mean_disc_reward worsened and critic/aux-critic loss had intermittent spikes.
- Analysis: The run is operationally stable, but the FB/critic update pressure is high for the saturated off-policy regime. The checkpoint loader restores optimizer state and checkpoint config, so a CLI lr-scale alone would not take effect.
- Adjustment: Uniform learning-rate scale 0.5: lr_f/lr_actor/lr_critic/lr_aux_critic 0.0003 -> 0.00015; lr_b/lr_discriminator 0.00001 -> 0.000005. Applied to checkpoint config.json and all optimizer param groups while preserving optimizer moments.
- Rationale: Reduce Q/critic/actor update amplitude without changing reward, observations, randomization, architecture, replay data, or task objective; preserve single-change attribution.
- Expected effect: Flatten critic/Q drift, reduce intermittent loss spikes, and allow incremental velocity-tracking improvement; watch MPJPE, vel_dist, proximity, mean_disc_reward, critic_loss, aux_critic_loss, torso contact fraction, and FPS.
- Result: Checkpoint prepared; process stopped cleanly; resume launch pending.
- Files/commands: remote baseline snapshot runs/bfm_command_24V_hi_4gpu_4096env_scratch_20260801_baseline_before_lrscale_20260801_1300; remote checkpoint config and optimizers.pth

## 2026-08-01 13:20:12 CST - First post-resume stability signal

- Context: remote z-autodl-h20-GPU10; same UFO run resumed from checkpoint time 48046080 with uniform 0.5x learning rates.
- Phenomenon: First post-resume training row reached timestep 48431104. Q_fb decreased about 952.5 -> 934.4, actor_loss about 25567.6 -> 24589.3, critic_loss about 1.619 -> 1.491, and mean_disc_reward improved about -9.425 -> -8.996. FPS was 759 in this startup/evaluation-adjacent window while GPU utilization remained 91-98%.
- Analysis: Initial value/critic pressure is lower and imitation reward is less negative, supporting the selected stability intervention. One row is insufficient to claim velocity-tracking improvement; wait for multiple steady-state rows and the next tracking evaluation.
- Adjustment: No additional parameter change.
- Rationale: Preserve single-change attribution for stability and speed-tracking assessment.
- Expected effect: Over 10-20 normal update rows, Q_fb/critic loss should remain flatter than the pre-resume run, torso-contact variability should reduce, and MPJPE/vel_dist should not regress.
- Result: Partial positive signal; velocity-tracking result pending.
- Files/commands: remote `resume_lrscale05_launch.log`, `train_log.txt`, and `humanoidverse_tracking_eval.csv`.

## 2026-08-01 14:34:45 CST - Resume Stage2 with stronger locomotion objective

- Context: PiPlus AMP Stage2 on GPU10; prior `ampw002_resume300` stopped at iteration 1094 after reaching checkpoint_1000, while `klfix_sync` reached iteration 1485 with stable synchronized PPO KL and checkpoint_1400.
- Primary metrics: `linvel_exp`, `linvel_projection`, locomotion reward; guardrails: termination/crash/fall rates, PPO `approx_kl`, value loss, AMP discriminator score.
- Diagnosis: AMP-weight reduction did not improve velocity tracking and degraded discriminator alignment (expert score about 0.55 near iterations 1092-1094). The synchronized-KL checkpoint_1400 is the best stable compatible resume point. Shift one reward-family weight toward command-conditioned locomotion while retaining AMP and all interfaces.
- Adjustment: `locomotion_reward_weight: 1.0 -> 1.1`; no other training parameter changed.
- Rationale: Increase the relative optimization pressure on velocity tracking, body uprightness, contact timing, and smoothness by 10% without changing observations, actions, robot, command range, AMP data, PPO, or checkpoint architecture.
- Selected checkpoint: remote `logs/amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_klfix_sync_scratch/checkpoint_1400.pt`; it had approx KL about `0.006-0.010`, high discriminator expert score, and no persistent termination failures.
- Remote backup: `humanoidverse/amp_stage2.py.bak.20260801142542`; uploaded patch: `/tmp/amp_stage2_loco_weight.patch`.
- Restart: `logs/amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_loco11_resume1400_v2/launcher.log`, 4 GPUs (IDs 4-7), 1024 envs/GPU, resume iteration 1400. Initial startup passed; iterations 1400-1404 show `linvel_exp` about `0.0164-0.0182`, approx KL `0.0061-0.0087`, and termination rate returning to zero after reset-heavy startup.
- Follow-up: evaluate the first 100-200 iterations and checkpoint_1500/1600; retain 1.1 only if linvel_exp rises without termination or KL regression.

## 2026-08-01 14:50:48 CST - Pull Stage2 checkpoint and refresh play docs

- Context: GPU10 run `amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_loco11_resume1400_v2` remained active after the locomotion-weight adjustment.
- Pulled artifacts: latest complete checkpoint visible at pull time, `checkpoint_1700.pt`, together with `config.json` and `launcher.log`.
- Verification: checkpoint metadata reports iteration `1700` and `locomotion_reward_weight=1.1`; the launcher log contains the resumed training window through iteration `1717`.
- Documentation: updated `docs/play_checkpoint_bfmzero-piplus-lse.md` Stage2 playback and command-encoder ONNX export commands to use the local `checkpoint_1700.pt`.
- Smoke test: two-step headless MuJoCo playback completed successfully on CPU without termination or runtime error.

## 2026-08-01 15:10:20 CST - Fix Stage2 playback initialization and simulator control

- Phenomenon: checkpoint playback initialized all 23 joints to `-0.25` because the default pose target used the wrong `[env, dof, state]` shape; the PiPlus MuJoCo XML also had no collidable ground and playback used direct generalized torques instead of Isaac-style position targets.
- Changes: added a tested `[num_envs, num_dofs, 2]` default-pose builder; added a real collidable MuJoCo plane; added explicit MuJoCo position-target PD application for P control; normalized `cuda`/`auto` device strings for IsaacLab; added `--policy-device` to avoid Isaac Sim OOM on low-memory GPUs.
- Verification: unit tests passed (5/5); MuJoCo zero-command and `0.4 m/s` playback each ran 100 steps without process failure; Isaac Sim ran 100 steps with `--device cuda:0 --policy-device cpu`, no OOM or termination, and reached about `0.336 m/s` under the `0.4 m/s` command.
- Limitation: MuJoCo remains an approximate dynamics backend and showed velocity oscillation; final stability judgment should use Isaac Sim with the training robot asset.

## 2026-08-01 15:24:58 CST - Add HT_lab_hi-style gamepad velocity control

- Context: PiPlus AMP Stage2 playback; checkpoint_1700; humanoidverse.amp_stage2_play
- Phenomenon: Stage2 playback previously supported only the Linux joystick API, while the reference HT_lab_hi play path uses pygame and Xbox-style axis semantics.
- Analysis: The controller input path should use the known HT_lab_hi mapping: left stick Y to forward, left stick X to lateral, right stick X to yaw, with centered-stick deadzone and the Stage2 metadata command range applied after normalization.
- Adjustment: Added optional pygame gamepad path with default axes lx=0, ly=1, rx=3, deadzone 0.08, edge-triggered reset/quit buttons, debug output, and CLI axis overrides; retained Linux joystick compatibility.
- Rationale: Use the same input semantics as the existing PiPlus play workflow while preserving asymmetric Stage2 command limits and avoiding random commands at startup.
- Expected effect: Centered sticks produce a zero target; deliberate stick motion gives bounded velocity commands; reset and quit are deterministic. Hardware-specific axis differences can be diagnosed with --gamepad-debug.
- Result: Pending hardware rollout; pygame is installed locally but no joystick is connected in this session. Pure mapping tests pass.
- Files/commands: humanoidverse/amp_stage2_play.py; tests/test_amp_stage2_play.py; docs/play_checkpoint_bfmzero-piplus-lse.md

## 2026-08-01 15:55:46 CST - Increase direct linear-velocity tracking reward

- Context: Remote GPU10 PiPlus AMP Stage2; run amp_stage2_piplus_lse_4gpu_4096env_1m_linvel18_resume2800_20260801_retry2; checkpoint_2800; 4 GPUs 4-7
- Phenomenon: The prior locomotion-weight resume plateaued: linvel_exp contribution was about 0.0167 near iteration 1400 and about 0.0160 near iterations 2600-2800, while fall/crash rates stayed near zero.
- Analysis: The aggregate locomotion weight also amplified upright/contact/smoothness terms but did not improve direct command tracking. The direct linvel_exp term is the narrowest reward bottleneck; a moderate 20% increase preserves the observation, action, checkpoint, AMP, PPO, and stability contracts.
- Adjustment: MIMICLITE_LOCOMOTION_WEIGHTS linvel_exp: 1.5 -> 1.8; no other training parameter changed
- Rationale: Increase gradient pressure on command-conditioned base linear velocity without changing unrelated reward families or safety terms.
- Expected effect: linvel_exp and actual velocity tracking should rise over 100-300 steady-state iterations while termination/crash/fall rates remain near zero; watch KL, value loss, AMP score, and action-rate penalties for regressions.
- Result: Restart verified: config reports linvel_exp=1.8 and locomotion_reward_weight=1.1; iterations 2805-2809 show linvel_exp contribution about 0.0193-0.0199, termination/crash/fall rates 0.0. This is an initial signal, not yet a conclusive evaluation.
- Files/commands: humanoidverse/amp_stage2.py; remote backup humanoidverse/amp_stage2.py.bak.20260801154604; command: python -u -m humanoidverse.amp_stage2 --gpu-ids 4,5,6,7 --resume .../checkpoint_2800.pt --work-dir logs/amp_stage2_piplus_lse_4gpu_4096env_1m_linvel18_resume2800_20260801_retry2

## 2026-08-01 16:03:04 CST - Evaluate first 168 linvel-weight resume iterations

- Context: Remote GPU10 PiPlus AMP Stage2; linvel18 resume from checkpoint_2800; iterations 2800-2967
- Phenomenon: Weighted linvel_exp contribution averaged 0.0194405 versus about 0.01614 before restart; reward averaged 0.08151 and fall-over rate remained 0.
- Analysis: Because the logged contribution includes the reward weight, the apparent 20% linvel_exp increase is mostly the intentional 1.5 -> 1.8 scaling. Weight-normalized signal changed only from about 0.01076 to 0.01080, so actual speed-tracking improvement is not yet established. Training health is stable: approx KL 0.00766, termination/crash about 5.9e-6, no falls.
- Adjustment: No additional parameter change; preserve single-change attribution.
- Rationale: Wait for a 100-300 iteration response window and direct velocity-error evaluation before another reward change.
- Expected effect: The weight-normalized linvel_exp signal or direct velocity error should improve while termination and PPO health remain stable.
- Result: Operationally successful and stable through iteration 2967; true speed-tracking improvement pending.
- Files/commands: remote launcher.log and config.json under logs/amp_stage2_piplus_lse_4gpu_4096env_1m_linvel18_resume2800_20260801_retry2; local tuning-log.md

## 2026-08-01 16:55:00 CST - Narrow Stage2 x-velocity command range and resume

- Context: Remote GPU10 PiPlus AMP Stage2; prior linvel18 run regressed after about iteration 3745, while checkpoint_3700 was the latest complete checkpoint before that decline.
- Phenomenon: The prior command sampler exposed forward x velocity `[-0.5, 1.2]`; speed tracking remained weak and the late window had weighted `linvel_exp` around `0.018-0.020` with no persistent termination failures.
- Analysis: Reduce the command distribution to the requested forward/reverse operating range so the policy receives denser supervision over the intended speeds; preserve existing reward, PPO, AMP, observation, action, and simulator settings for attribution.
- Adjustment: `commands_low[0]: -0.5 -> -0.2` and `commands_high[0]: 1.2 -> 0.8`; no other training parameter changed. Resume from `checkpoint_3700.pt`.
- Rationale: Make the training distribution match the target x-speed envelope and avoid spending samples on the poorly learned high-speed tail.
- Expected effect: Over the next 100-300 iterations, command-conditioned velocity error should decrease and normalized `linvel_exp` should rise or remain stable, with termination rate near zero and PPO KL in the prior stable band.
- Result: Remote restart succeeded after setting a writable `TMPDIR` for Isaac Lab logs. Config records command range `[-0.2, -0.2, -0.8]` to `[0.8, 0.2, 0.8]` and resume iteration 3700. Initial iterations 3701-3712 show `linvel_exp` contribution about `0.0243-0.0264`, KL `0.0067-0.0098`, and termination rate returning to zero; evaluation is still in progress.
- Files/commands: `humanoidverse/amp_stage2.py`, `docs/第二阶段AMP训练流程.md`, remote backup `humanoidverse/amp_stage2.py.bak.command_range_20260801_164914`, local artifact `logs/instinct_rl/amp_stage2/amp_stage2_piplus_lse_4gpu_4096env_1m_cmdx_m02_p08_resume3700_20260801`.

## 2026-08-01 18:56:51 CST - Repair Stage2 PPO updates and speed-tracking contract

- Context: PiPlus AMP Stage2 on GPU10, four GPUs 4-7, 1024 environments/GPU; stable source checkpoint `checkpoint_5600.pt` from the narrowed command-range run.
- Phenomenon: iterations 3700-5698 remained physically stable, but PPO performed only about 4-5 minibatch updates per iteration because KL early stopping fired every round. Weighted `linvel_exp` stayed near `0.0255`, which is approximately the zero-velocity baseline under the narrowed command distribution, so it did not demonstrate tracking.
- Analysis: the command encoder was update-starved, its 378-D input mixed unscaled joint velocities/history with small command fields, and the positive exponential reward paid a large reward for standing still under moving commands. Direct velocity errors and command-response gain were not logged.
- Adjustment: set PPO to 4 epochs, disable KL early stopping, and enforce the requested LR after checkpoint load. A controlled `1e-4` probe produced excessive KL/clip pressure, so the final LR is `5e-5`. Replace linear-velocity reward scale `0.25` with a signed, zero-speed-baseline-normalized reward using error scale `0.16`; keep `linvel_exp` weight `1.8`. Normalize command inputs by `[1.25, 5.0, 1.25]` and current/history joint velocity by `0.05` without changing input dimension. Legacy checkpoints receive an equivalent first-layer weight migration and reset policy Adam state; new-format resumes preserve valid state but override LR.
- Rationale: expose a learnable signed speed signal, give PPO the full requested update budget, and improve encoder conditioning while retaining the frozen BFM, discriminator, AMP normalizer, action interface, and checkpoint policy output at the resume boundary.
- Expected effect: `ppo_updates=ppo_expected_updates=128`, zero early stops, low termination rate, decreasing nonzero `vx/vy` and speed-bin MAE, and increasing positive command-response slope/correlation. Treat the baseline-normalized `linvel_exp` as improvement over zero speed, not as a maximum-reward ratio.
- Result: final run `amp_stage2_piplus_lse_4gpu_4096env_1m_speed_contract_lr5e5_resume5600_20260801` resumed successfully with `policy_optimizer_resume=reset_for_encoder_input_transform` and LR `5e-5`. Over iterations 5627-5646, PPO completed `128/128` updates, mean KL was `0.03316`, clip fraction `0.43477`, and termination rate `0`. Nonzero vx MAE averaged `0.27705`, fast-forward MAE `0.38091`, slow-forward MAE `0.14398`, and response slope was positive but small at `0.01854`; this is an early optimization signal, not yet proof of a fully controllable speed latent.
- Files/commands: `humanoidverse/amp_stage2.py`, `humanoidverse/amp_stage2_play.py`, `tests/test_amp_stage2.py`; remote backup `backups/stage2_speed_contract_20260801_before_resume5600`; local artifacts under `logs/instinct_rl/amp_stage2/amp_stage2_piplus_lse_4gpu_4096env_1m_speed_contract_lr5e5_resume5600_20260801/`.

## 2026-08-01 19:15:00 CST - Continue speed-contract resume through checkpoint 5900

- Context: Same four-GPU GPU10 run, resumed from compatible `checkpoint_5600.pt`; current remote process remains active and has produced `checkpoint_5900.pt`.
- Result: In iterations 5918-5937, PPO completed `128/128` updates every iteration, mean `approx_kl=0.02805`, clip fraction `0.39146`, and termination rate (3.8e-7). Nonzero vx MAE averaged `0.24005`, fast-forward vx MAE `0.30417`, response slope `0.19825`, and command correlation `0.31446`. Over the wider 5838-5937 window, the corresponding values were `0.26504`, `0.37468`, `0.13795`, and `0.23058`.
- Interpretation: Speed tracking is improving relative to the immediate post-resume window (nonzero vx MAE `0.27705 -> 0.24005`; fast-forward MAE `0.38091 -> 0.30417`; response slope `0.01854 -> 0.19825`), but the command-response latent is not yet fully reliable. A single transient KL spike occurred near iteration 5912 and recovered without early stopping; continue monitoring the next 100-200 iterations.
- Artifact: Local `logs/instinct_rl/amp_stage2/amp_stage2_piplus_lse_4gpu_4096env_1m_speed_contract_lr5e5_resume5600_20260801/checkpoint_5900.pt`; SHA256 `464c7ef4d2a935839cdbdcb5b22d81ab44336a94c8f169b6b70ac5e3e807e119`.

## 2026-08-01 19:20:00 CST - Continue Stage2 run through checkpoint 6000

- Context: The resumed four-GPU run remains active on GPU10 and advanced beyond iteration 6100. The latest fully synchronized local artifact is `checkpoint_6000.pt`.
- Result: Iterations 6002-6021 kept `ppo_updates=128`, `ppo_update_fraction=1.0`, and termination rate `2.67e-6`. Mean `approx_kl=0.04392`; nonzero vx MAE was `0.24272`, fast-forward MAE `0.28757`, response slope `0.19061`, and command correlation `0.28075`. A later live window, iterations 6097-6116, recovered to mean KL `0.03003`, nonzero vx MAE `0.23797`, response slope `0.26680`, and correlation `0.38040`, with no traceback.
- Interpretation: Compared with iterations 5627-5646, nonzero vx MAE improved from `0.27705` to `0.23797`, and response slope from `0.01854` to `0.26680`. The speed latent is becoming controllable, although fast-forward error remains noisy and should be monitored before changing LR or reward coefficients again.
- Artifact: Local `logs/instinct_rl/amp_stage2/amp_stage2_piplus_lse_4gpu_4096env_1m_speed_contract_lr5e5_resume5600_20260801/checkpoint_6000.pt`; SHA256 `fba833da00d61d5433abf3c8cfae30c5d1cca7f70b8edaede72f0475fa573bcf`.

## 2026-08-01 20:56:06 CST - Increase signed vx tracking weight and resume checkpoint 7100

- Context: GPU10 PiPlus Stage2; old run amp_stage2_piplus_lse_4gpu_4096env_1m_speed_contract_lr5e5_resume5600_20260801; new run amp_stage2_piplus_lse_4gpu_4096env_1m_speed_contract_linvel21_lr5e5_resume7100_20260801; detached PID 726224
- Phenomenon: Relative to checkpoint 6800, x tracking plateaued: checkpoint 7300 window had fast-forward MAE 0.24976, backward MAE 0.14804, response slope 0.49680; hardware test can move forward but does not visibly turn or reverse.
- Analysis: Checkpoint 7100 is the best compatible x-tracking candidate: vx MAE 0.15601, nonzero vx MAE 0.16864, fast-forward MAE 0.22744, response slope 0.54848, correlation 0.71395, termination near zero. The checkpoint7300 ONNX produces distinct forward/backward and left/right yaw latents, while the onboard default deadband maps half backward stick to only -0.057 m/s and full backward to the trained -0.2 m/s limit; yaw requires a separate controlled reward/deployment diagnosis.
- Adjustment: MIMICLITE_LOCOMOTION_WEIGHTS[linvel_exp]: 1.8 -> 2.1; no PPO, command range, sampler, angular reward, architecture, or deployment behavior change.
- Rationale: Increase the gradient of the already signed, zero-speed-baseline-normalized planar tracking reward by 16.7% from the strongest historical x-tracking checkpoint.
- Expected effect: Within 100-300 iterations, reduce fast-forward/backward vx MAE and sustain response slope at or above 0.55; watch KL, clip fraction, value loss, termination, and AMP score. Roll back to weight 1.8 and checkpoint 7100 if x-tail MAE worsens.
- Result: Resumed successfully from checkpoint 7100. Through iteration 7115: 128/128 PPO updates, KL 0.0254, termination 0, latest nonzero vx MAE 0.1683, fast-forward MAE 0.2176, response slope 0.5447. Too early for attribution.
- Files/commands: local/remote humanoidverse/amp_stage2.py; remote backup /root/autodl-tmp/zhuzejian/Project/HT_BFM/humanoidverse/amp_stage2.py.bak.linvel21.20260801204850; log /tmp/amp_stage2_speed_contract_linvel21_resume7100.log; launch: python -u -m humanoidverse.amp_stage2 --device cuda --gpu-ids 4,5,6,7 --num-envs 1024 --iterations 1000000 --rollout-steps 32 --save-every 100 --amp-weight 0.04 --locomotion-reward-weight 1.1 --ppo-epochs 4 --learning-rate 5e-5 --target-kl 0 --resume .../checkpoint_7100.pt --work-dir logs/amp_stage2_piplus_lse_4gpu_4096env_1m_speed_contract_linvel21_lr5e5_resume7100_20260801

## 2026-08-01 21:06:43 CST - Evaluate linvel 2.1 matched resume window

- Context: GPU10 PiPlus Stage2; compare old linvel_exp=1.8 and new linvel_exp=2.1 over the same iterations 7101-7184 after resuming checkpoint_7100.
- Phenomenon: The old run plateaued after checkpoint_7100; hardware moves forward but reverse and yaw are not visibly effective.
- Analysis: The controlled comparison isolates the 16.7% linear tracking-weight increase. Reverse remains the weakest x-direction despite improved aggregate, nonzero, slow-forward, and fast-forward errors. Yaw was intentionally unchanged this cycle.
- Adjustment: No additional parameter change. Keep the linvel_exp=2.1 run active and preserve single-variable attribution.
- Rationale: The matched window shows a modest x-tracking gain without PPO or stability regression, so stopping or stacking another change now would discard useful evidence.
- Expected effect: Continue monitoring checkpoint windows for backward vx MAE, fast-forward MAE, response slope, yaw MAE, KL, termination, and full 128/128 PPO updates.
- Result: New versus old: vx MAE 0.15997 vs 0.16496; nonzero vx MAE 0.17651 vs 0.18282; slow-forward 0.13149 vs 0.13711; fast-forward 0.24105 vs 0.25246; response slope 0.52710 vs 0.52264; correlation 0.71027 vs 0.70163. Backward MAE slightly regressed 0.14024 vs 0.13915 and yaw MAE was effectively flat/slightly worse 0.38626 vs 0.38458. Termination stayed zero, value loss improved 0.14759 vs 0.15782, and checkpoint_7200 was saved and pulled.
- Files/commands: humanoidverse/amp_stage2.py; logs/instinct_rl/amp_stage2/amp_stage2_piplus_lse_4gpu_4096env_1m_speed_contract_linvel21_lr5e5_resume7100_20260801/checkpoint_7200.pt; docs/play_checkpoint_bfmzero-piplus-lse.md; remote log /tmp/amp_stage2_speed_contract_linvel21_resume7100.log

## 2026-08-01 21:09:15 CST - Record mixed late response after checkpoint 7200

- Context: GPU10 PiPlus Stage2 linvel_exp=2.1 run; matched old/new windows through local iteration 7259 and live remote window 7315-7334.
- Phenomenon: The early 7101-7200 advantage did not remain uniformly positive in the later window.
- Analysis: Over matched iterations 7160-7259, the new run improved backward MAE but slightly regressed aggregate nonzero and fast-forward errors; the latest live 20 rounds also fell below the desired response-slope threshold. This is optimization noise/plateau, not evidence that reverse and yaw are solved.
- Adjustment: No parameter change. Retain checkpoint_7200 as the pulled candidate and keep the remote run active for observation.
- Rationale: A second change would violate single-variable attribution; checkpoint_7200 preserves the best validated snapshot from this cycle while the live run may still recover.
- Expected effect: Evaluate the next complete checkpoint window before accepting linvel_exp=2.1. Treat yaw as a separate controlled reward experiment after verifying nonzero onboard yaw commands.
- Result: Matched 7160-7259 new vs old: backward MAE 0.13913 vs 0.14521 improved, but nonzero vx MAE 0.18675 vs 0.18339 and fast-forward MAE 0.26218 vs 0.25216 regressed; response slope 0.50269 vs 0.50517. Live 7315-7334: nonzero vx MAE 0.19571, backward 0.13385, fast-forward 0.28230, slope 0.48476, yaw MAE 0.35397, termination 0, PPO 128/128, no traceback.
- Files/commands: tuning-log.md; remote /tmp/amp_stage2_speed_contract_linvel21_resume7100.log; local checkpoint_7200.pt SHA256 74f444ab5af154cbdf568688519f0a77d42e2cb301c570e0b264e99fcb6869d8

## 2026-08-02 10:23:04 CST - Widen x/y command domain and tighten action-rate stability

- Context: GPU10 PiPlus AMP Stage2; previous linvel_exp=2.1 run reached checkpoint_19800 with strong vx tracking but AMP score about -5.0, termination about 0.0012, and environment action-rate contribution about -0.0415.
- Phenomenon: Real robot still only moves forward; reverse command range was too narrow, yaw remained ineffective, and late training became dynamically aggressive.
- Analysis: The active source sampled x [-0.2,0.8], y [-0.2,0.2], yaw [-0.8,0.8]. The environment penalty_action_rate=-0.5 dominated the smoothness cost; MimicLite action-rate terms contributed only about 1e-6. Checkpoint_14800 was the best stable resume point with termination 0, vx MAE about 0.100, vy MAE about 0.091, yaw MAE about 0.253, and AMP score about 0.636.
- Adjustment: Set commands_low/high to x [-0.8,0.8], y [-0.5,0.5], yaw [-0.8,0.8]; override environment penalty_action_rate -0.5 -> -0.55. Keep linvel_exp=2.1, PPO, LR, architecture, and deployment interface unchanged.
- Rationale: Expose enough negative and lateral command supervision for the requested hardware behaviors while applying a conservative 10% stronger action-change penalty to counter the instability seen after the late high-tracking checkpoint.
- Expected effect: Reverse and lateral command bins should receive meaningful training; action-rate contribution should remain controlled while termination stays near zero. Yaw remains a separately monitored bottleneck because its reward implementation was not changed in this cycle.
- Result: Remote backup created at /root/autodl-tmp/zhuzejian/Project/HT_BFM/humanoidverse/amp_stage2.py.bak.cmdx_m08_y05_act055.20260802000000. New run resumed checkpoint_14800 successfully with command_range [-0.8,-0.5,-0.8] to [0.8,0.5,0.8], penalty_action_rate=-0.55, LR 5e-5, and 128/128 PPO updates. checkpoint_14900 was saved and pulled. Initial live window through 14979 has termination 5.7e-6, action-rate contribution -0.00749, nonzero vx MAE 0.1634, backward MAE 0.1987, fast-forward MAE 0.1655, nonzero vy MAE 0.1146, yaw MAE 0.3052, and no traceback; this is an early post-resume result.
- Files/commands: humanoidverse/amp_stage2.py; docs/第二阶段AMP训练流程.md; docs/play_checkpoint_bfmzero-piplus-lse.md; local checkpoint logs/instinct_rl/amp_stage2/amp_stage2_piplus_lse_4gpu_4096env_1m_cmdx_m08_y05_act055_resume14800_20260802/checkpoint_14900.pt; remote log /tmp/amp_stage2_cmdx_m08_y05_act055_resume14800.log; launch --resume .../checkpoint_14800.pt --work-dir logs/amp_stage2_piplus_lse_4gpu_4096env_1m_cmdx_m08_y05_act055_resume14800_20260802

## 2026-08-02 12:07:00 CST - Coordinated Stage2 direction/yaw and PPO guard resume

- Task: PiPlus 23DoF AMP Stage2 command-conditioned locomotion; optimize direct planar/yaw velocity tracking while preserving upright motion, contacts, smooth actions, and four-GPU operational health.
- Active lineage: amp_stage2_piplus_lse_4gpu_4096env_1m_resume14900_20260802_retry; physical GPUs 4-7, four ranks, 1024 environments/rank, rollout 32, target 1,000,000 iterations.
- Metric windows: iterations 14901-15000 versus 15001-15100. planar_l2_mae 0.20989 -> 0.20467; vx_mae 0.15575 -> 0.14973; nonzero_vx_mae 0.17185 -> 0.16508; yaw_rate_mae 0.29824 -> 0.28875; response slope 0.65259 -> 0.67496; command correlation 0.89003 -> 0.89920. The last 50 iterations slightly regressed versus the preceding 50, so classification was plateaued with a direction/yaw bottleneck.
- Guardrails: termination/crash/fall rates stayed about 2e-6/2e-6/0, truncation below 1e-5, approx_kl about 0.035-0.037, clip fraction about 0.40, and full 128/128 PPO updates except isolated KL/value spikes. AMP contribution remained strongly negative (about -0.0716) versus locomotion contribution about +0.0537.
- Coordinated changes: linvel_projection 0.6 -> 0.72 (+20%); angvel_z_exp 1.4 -> 1.6 (+14%); default amp_weight 0.04 -> 0.03; default target_kl 0.0 -> 0.04. Resume explicitly used amp-weight 0.03 and target-kl 0.04; linvel_exp 2.1, locomotion weight 1.1, PPO epochs 4, LR 5e-5, command range, robot, interface, and architecture were preserved.
- Selected checkpoint: checkpoint_15100.pt, complete and size/mtime-stable across two checks; it loads policy, policy optimizer, discriminator, discriminator optimizer, AMP reward normalizer, and metadata, and aligned with the best stable pre-change tracking window.
- Patch/rollback: local logs/tuning/stage2_amp_finetune_20260802_1158.patch; remote /tmp/stage2_amp_finetune_20260802_1158.patch; remote backup humanoidverse/amp_stage2.py.bak.stage2_amp_finetune_20260802_1158.
- Restart: no tmux available. Parent 816235, torchrun 816367, workers 816434-816437; detached output appends to logs/amp_stage2_piplus_lse_4gpu_4096env_1m_resume14900_20260802_retry/launcher.log and launcher.pid was atomically updated. TMPDIR=/tmp/zhuzejian_isaaclab was required for writable IsaacLab logs. launcher.log confirms resumed_from checkpoint_15100, start_iteration 15100, and live progress through iteration 15155 with all four ranks on GPUs 4-7. The post-resume 15101-15155 window averages planar MAE 0.20111, vx MAE 0.14680, yaw MAE 0.28281, nonzero-vx MAE 0.16287, response slope 0.68472, correlation 0.90812, AMP contribution -0.04536, termination 1.94e-6, approx KL 0.02898, and value loss 0.27252.
- Follow-up: compare at least 50-100 post-resume iterations against the 15001-15100 baseline, prioritizing planar/vx/vy/yaw MAE, bin MAEs, response slope/correlation, achieved-vx std, termination/safety, KL, clip fraction, value loss, discriminator, AMP contribution, and PPO update fraction.

## 2026-08-02 13:45:07 CST - Audit failed klfix scratch lineage; resume blocked

- Task: PiPlus 23DoF AMP Stage2 command-conditioned locomotion; the actual objective is direct planar and yaw-rate command tracking, with upright/contact/smooth-action safety guardrails and four-rank GPU health.
- Active lineage requested: `amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_klfix_scratch`; expected physical GPUs 4-7, 1024 environments/rank, rollout 32, target 1,000,000 iterations.
- Pulled artifacts: local `logs/amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_klfix_scratch/config.json` and `launcher.log`; remote/local SHA256 match (`config.json` `e4da147e...3fe91e`, `launcher.log` `c371a9a7...f3d35`). No checkpoint artifact was present locally or remotely.
- Operational health: dead/unstable. `launcher.pid` 612163 is absent; the launcher log ends at 2026-08-01 03:38:54 with a 7,200,000 ms NCCL `ALLREDUCE` timeout (sequence 198), rank 0/2 watchdog failures, SIGABRT for ranks 1/3, and `ChildFailedError`. The launch command confirms four torchrun ranks, `--gpu-ids 4,5,6,7`, 1024 envs/rank, rollout 32, save interval 100, and 1,000,000 target iterations. Current GPU mapping shows the unrelated UFO job on physical GPUs 0-3 and a separate `resume14900_20260802_retry` lineage on 4-7; neither was touched.
- Learning health: `insufficient-data`, not improving/plateaued/regressing. Only iterations 0 and 1 emitted metrics; direct `tracking/planar_l2_mae`, `tracking/vx_mae`, `tracking/vy_mae`, `tracking/yaw_rate_mae`, nonzero/bin MAEs, response slope, command correlation, and achieved-vx std are absent, so command tracking cannot be inferred from reward terms. Supporting signals changed reward `0.06662 -> 0.08514`, `mimiclite/linvel_exp` `0.02025 -> 0.01655`, `mimiclite/linvel_projection` `-0.0000188 -> 0.0000542`, `mimiclite/angvel_z_exp` `0.00538 -> 0.01329`, AMP contribution `0.000002 -> 0.02109`, discriminator expert score `0.05569 -> 0.08781`, approximate KL `0.01888 -> 0.01210`, value loss `1.17666 -> 0.76093`, and termination/crash rate `0.000938 -> 0.000221`; this is too short and ends in a collective failure.
- Stability/safety metrics available: fall-over and truncation rates were 0 in both records; body-upright, undesired-contact, torque, joint-limit, foot-slippage, episode length, and NaN/Inf metrics were not emitted. PPO early-stop was active on both records (`ppo_updates=2`), so the independent early-stop/collective interaction is the leading operational diagnosis, but there is no usable task window.
- Required finetune decision: blocked no-op. A coordinated tuning patch could not be safely applied because the exact work directory contains no complete `checkpoint_<iteration>.pt` after two stable checks; the workflow explicitly forbids starting from scratch or borrowing a checkpoint from `klfix_sync_scratch` or any newer retry lineage. No remote file, process, or unrelated run was modified; no backup or patch path exists; no restart command was issued.
- Docs: scanned `docs/` for Stage2 playback commands. Unchanged because the exact lineage has no complete/playable checkpoint; existing commands continue to reference the separate complete `resume14900_20260802_retry/checkpoint_15100.pt` lineage.
- Follow-up condition: rerun this cycle only after a complete compatible checkpoint is produced in this exact work directory, or provide an explicit new lineage scope. Until then the exact lineage remains non-resumable.

## 2026-08-02 15:20:29 CST - Scheduler cycle 88b6974daecc re-audit; resume still blocked

- Task: PiPlus 23DoF AMP Stage2 command-conditioned locomotion on GPU10; primary objective is direct planar/yaw-rate command tracking, with upright/contact/action smoothness and four-rank GPU health as guardrails.
- Active lineage: `amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_klfix_scratch`; required GPUs 4,5,6,7, 1024 environments/rank, rollout 32, save interval 100, target 1,000,000 iterations.
- Pull/revalidation: exact remote `config.json` and `launcher.log` hashes match local copies (`e4da147e...3fe91e`, `c371a9a7...f3d35`). Remote `find` again returned zero `checkpoint_<iteration>.pt` files; no selected checkpoint exists.
- Operational health: dead/unstable. `launcher.pid=612163` is absent; no matching process remains. The last log records only iterations 0 and 1, then a 7,200,000 ms NCCL `ALLREDUCE` timeout (sequence 198), rank watchdog failures, SIGABRT, and `ChildFailedError`. GPUs 0-3 remain unrelated UFO workers; GPUs 4-7 remain the separate `resume14900_20260802_retry` lineage and were not touched.
- Learning health: `insufficient-data`. Reward changed `0.06662 -> 0.08514`; `mimiclite/linvel_exp` `0.02025 -> 0.01655`; `linvel_projection` `-1.88e-05 -> 5.42e-05`; `angvel_z_exp` `0.00538 -> 0.01329`; AMP contribution `1.65e-06 -> 0.02109`; discriminator expert score `0.05569 -> 0.08781`; approximate KL `0.01888 -> 0.01210`; value loss `1.17666 -> 0.76093`; termination/crash rate `9.38e-04 -> 2.21e-04`; fall-over and truncation stayed zero. Direct planar/vx/vy/yaw MAE, nonzero/bin MAEs, response slope, command correlation, achieved-vx std, episode length, body-upright, contact, torque, joint-limit, slippage, entropy, NaN/Inf metrics were absent, so reward cannot establish command-tracking improvement.
- Finetune decision: blocked no-op. The exact lineage has no complete compatible checkpoint after stable rechecks, so the workflow forbids starting from scratch or borrowing another lineage. No local source/config parameter was changed, no remote backup or patch was created, no process was stopped, and no restart was attempted.
- Docs: Stage2 playback docs were scanned; unchanged because this lineage has no complete/playable checkpoint. Existing commands continue to reference the separate complete `resume14900_20260802_retry/checkpoint_15100.pt` lineage.
- Follow-up condition: resume this exact lineage only after a complete compatible checkpoint appears in its own work directory, or after the user explicitly changes the lineage scope.

## 2026-08-02 16:58:46 CST - Scheduler 29a0d6005eff re-audit; exact scratch lineage remains non-resumable

- Task: PiPlus 23DoF AMP Stage2 command-conditioned locomotion on GPU10; optimize direct planar and yaw-rate command tracking while preserving upright/contact/action-smoothness safety and four-rank GPU health.
- Scope: `amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_klfix_scratch`, physical GPUs 4-7, 1024 environments/rank, rollout 32, save interval 100, target 1,000,000 iterations.
- Pulled artifacts: local `logs/amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_klfix_scratch/config.json` and `launcher.log`; SHA256 matches remote (`config.json` `e4da147eb5931ae22d79dc21cdd0b7af207cc86c3484c6db559e08e43e3fe91e`, `launcher.log` `c371a9a7f6ac64708b55a426ea40ae4bc1f9c79f80d8bf2ffbb215849e3f3d35`). Two stable remote checks found zero `checkpoint_*.pt` files; `launcher.pid=612163` is dead.
- Operational health: `dead/unstable`. The launcher emitted only iterations 0 and 1, then all four ranks failed a 7,200,000 ms NCCL `ALLREDUCE` timeout (sequence 198), with watchdog errors, SIGABRT, and `ChildFailedError`. Current GPUs 0-3 run unrelated UFO workers; GPUs 4-7 run the separate `resume14900_20260802_retry` lineage, which was not touched.
- Learning health: `insufficient-data`, not improving/plateaued/regressing. Supporting changes from iteration 0 -> 1 were reward `0.06662 -> 0.08514`, weighted `linvel_exp` `0.020247 -> 0.016553`, `linvel_projection` `-1.88e-05 -> 5.42e-05`, `angvel_z_exp` `0.005377 -> 0.013290`, AMP contribution `1.65e-06 -> 0.021085`, discriminator expert score `0.05569 -> 0.08781`, approx KL `0.01888 -> 0.01210`, value loss `1.17666 -> 0.76093`, and termination/crash `9.38e-04 -> 2.21e-04`; fall-over and truncation were zero. Direct planar/vx/vy/yaw MAE, nonzero/bin MAEs, response slope, command correlation, achieved-vx std, episode length, body-upright, contact, torque, joint-limit, slippage, entropy, and explicit NaN/Inf metrics were absent, so reward terms cannot establish command-tracking success.
- Finetune decision: required coordinated patch and restart blocked by the concrete absence of a complete compatible checkpoint in this exact work directory. No source/config parameter was changed, no remote backup or patch path exists, no process was stopped, and no restart was attempted; starting from scratch or borrowing `resume14900_20260802_retry` is prohibited.
- Docs: Stage2 playback docs scanned and unchanged because this lineage has no complete/playable checkpoint; existing commands remain pointed at the separate complete retry lineage.
- Follow-up condition: rerun only after this exact directory contains a stable, loadable `checkpoint_<iteration>.pt`, or after the lineage scope is explicitly changed.

## 2026-08-02 18:32:57 CST - Scheduler 134c0c487964; exact KL-fix scratch lineage still blocked

- Task: PiPlus 23DoF AMP Stage2 command-conditioned locomotion; optimize direct planar and yaw-rate command tracking, with upright/contact/action-smoothness and numerical stability as guardrails.
- Scope: `amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_klfix_scratch`; required physical GPUs 4-7, four ranks, 1024 environments/rank, rollout 32, save interval 100, target 1,000,000 iterations. Remote project was reached through the interactive JumpServer because non-interactive exec returned no output.
- Pulled/verified artifacts: local `logs/amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_klfix_scratch/config.json` and `launcher.log`; remote and local SHA256 are `e4da147eb5931ae22d79dc21cdd0b7af207cc86c3484c6db559e08e43e3fe91e` and `c371a9a7f6ac64708b55a426ea40ae4bc1f9c79f80d8bf2ffbb215849e3f3d35`. Two stable checks one second apart found no `checkpoint_<iteration>.pt` files and no exact-pattern process.
- Operational health: `dead/unstable`. `launcher.pid=612163` is absent; the log contains only iterations 0 and 1, then a 7,200,000 ms NCCL `ALLREDUCE` timeout at sequence 198, watchdog failures on ranks 0-3, SIGABRT, and `ChildFailedError`. The current GPU map has unrelated UFO workers on physical GPUs 0-3 and a separate `resume14900_20260802_retry` lineage on 4-7; all were preserved.
- Learning health: `insufficient-data`, not improving/plateaued/regressing. Available iteration 0 -> 1 support signals: reward `0.066622 -> 0.085136`; weighted `linvel_exp` `0.020247 -> 0.016553`; `linvel_projection` `-1.88e-05 -> 5.42e-05`; `angvel_z_exp` `0.005377 -> 0.013290`; AMP contribution `1.65e-06 -> 0.021085`; discriminator expert score `0.05569 -> 0.08781`; approx KL `0.01888 -> 0.01210`; clip fraction `0.21570 -> 0.19995`; value loss `1.17666 -> 0.76093`; termination/crash `9.384e-04 -> 2.213e-04`; fall-over and truncation were zero. Direct planar/vx/vy/yaw MAE, nonzero/bin MAEs, response slope, command correlation, achieved-vx std, episode length, body-upright, contact, torque, joint-limit, slippage, entropy, and NaN/Inf metrics were absent, so reward terms cannot establish command-tracking success.
- Diagnosis: the leading failure is the distributed collective hang, with no usable learning window and no evidence to support a diagnosis-driven parameter edit for this exact lineage. The config records `amp_reward_weight=0.04`, `locomotion_reward_weight=1.0`, PPO learning rate `3e-5`, target KL `0.01`, clip ratio `0.2`, and entropy coefficient `0.001`, but changing them without a compatible resume checkpoint would violate lineage safety.
- Finetune decision: required coordinated patch and one resume restart are blocked by the concrete absence of a complete compatible checkpoint after stable rechecks. No local source/config parameter was changed, no remote backup or patch was created, no process was stopped, and no restart was attempted. Starting from scratch or borrowing `resume14900_20260802_retry` is prohibited.
- Docs: Stage2 playback docs were scanned; unchanged because this exact lineage has no complete/playable checkpoint. Existing commands continue to reference the separate complete retry lineage.
- Follow-up condition: resume this exact lineage only after its own directory contains a stable, loadable `checkpoint_<iteration>.pt`, or after the user explicitly changes the lineage scope.

## 2026-08-02 20:08:14 CST - Scheduler 56a866d728f3; exact KL-fix scratch cycle blocked

- Task: PiPlus 23DoF AMP Stage2 command-conditioned locomotion on GPU10; optimize direct planar/vx/vy/yaw-rate tracking, with upright/contact/action-rate, PPO, numerical, and four-rank GPU health as guardrails.
- Scope and launch contract: `amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_klfix_scratch`, physical GPUs 4-7, four ranks, 1024 environments/rank (4096 total), rollout 32, save interval 100, target 1,000,000 iterations. The exact launcher command was verified in `launcher.log`; `config.json` records `amp_reward_weight=0.04`, `locomotion_reward_weight=1.0`, `linvel_exp=1.5`, PPO learning rate `3e-5`, target KL `0.01`, clip ratio `0.2`, and entropy coefficient `0.001`.
- Pull/revalidation: entered GPU10 through the interactive JumpServer because direct remote-command execution returned no payload. Remote and local `config.json`, `launcher.log`, and `launcher.pid` hashes match (`e4da147eb5931ae22d79dc21cdd0b7af207cc86c3484c6db559e08e43e3fe91e`, `c371a9a7f6ac64708b55a426ea40ae4bc1f9c79f80d8bf2ffbb215849e3f3d35`, `9469c8603016be3765f9c189a4502cecd2f7cb7f9a01ab6a0c7c3721fb476cb7`). Two stable checks two seconds apart found zero `checkpoint_<iteration>.pt` files; `launcher.pid=612163` is dead.
- Operational health: `dead/unstable`. The launcher produced only iterations 0 and 1, then all ranks failed a 7,200,000 ms NCCL `ALLREDUCE` timeout at sequence 198, followed by watchdog errors, SIGABRT, and `ChildFailedError`. No exact-lineage parent or workers remain. Unrelated UFO workers on physical GPUs 0-3 and the separate `resume14900_20260802_retry` lineage on GPUs 4-7 were observed and left untouched.
- Learning health: `insufficient-data`. Only two metric rows exist, so no equal-length learning window is defensible. Supporting iteration 0 -> 1 changes were reward `0.066622 -> 0.085136`, weighted `mimiclite/linvel_exp` `0.020247 -> 0.016553`, `linvel_projection` `-1.88e-05 -> 5.42e-05`, `angvel_z_exp` `0.005377 -> 0.013290`, AMP contribution `1.65e-06 -> 0.021085`, discriminator expert score `0.05569 -> 0.08781`, approximate KL `0.01888 -> 0.01210`, clip fraction `0.21570 -> 0.19995`, value loss `1.17666 -> 0.76093`, and termination/crash rate `9.384e-04 -> 2.213e-04`; fall-over and truncation were zero. Direct planar/vx/vy/yaw MAE, nonzero and speed-bin MAEs, response slope, command correlation, achieved-vx standard deviation, episode length, body-upright, contact, torque, joint-limit, slippage, entropy, and explicit NaN/Inf metrics were absent. Reward terms therefore cannot establish command-tracking improvement.
- Diagnosis: the leading failure is the distributed collective hang; there is no usable learning window and no complete checkpoint from which a diagnosis-driven parameter change could be resumed safely.
- Finetune decision: required coordinated tuning/upload and one resume restart are blocked by the concrete absence of a complete, stable, loadable checkpoint in this exact work directory. No local source/config parameter was changed, no remote backup or patch was created, no process was stopped, and no restart was attempted. Starting from scratch or borrowing another lineage is prohibited.
- Docs: scanned Stage2 playback docs, including `docs/play_checkpoint_bfmzero-piplus-lse.md`, `docs/第二阶段AMP训练流程.md`, and `docs/task_migrate_amp_stage2_piplus.md`; unchanged because this lineage has no complete/playable checkpoint. Existing commands continue to reference the separate complete `resume14900_20260802_retry/checkpoint_15100.pt` lineage.
- Follow-up condition: resume this exact lineage only after its own directory contains a stable, compatible `checkpoint_<iteration>.pt`, or after the user explicitly changes the lineage scope.

## 2026-08-02 21:50:46 CST - Rebalance Stage2 tracking and stability rewards

- Context: GPU10 PiPlus AMP Stage2; active lineage resume14900_20260802_retry; select checkpoint_18200.pt
- Phenomenon: Tracking improved near iterations 17800-18400 but regressed by 25000: nonzero planar MAE about 0.222 -> 0.309, vx MAE 0.156 -> 0.236, command correlation 0.910 -> 0.780; fall-over remained near zero.
- Analysis: The late run raised total reward and AMP score while direct speed metrics degraded. Tracking contributions were too weak relative to upright/action/torque terms, and continued 5e-5 updates drifted away from the best stable command response.
- Adjustment: MIMICLITE weights: linvel_exp 2.1 -> 2.5; linvel_projection 0.72 -> 0.95; angvel_z_exp 1.6 -> 1.9; body_upright 1.0 -> 1.2; single_foot_contact 0.75 -> 0.95; angvel_xy_l2 0.02 -> 0.04; joint_deviation_l2 0.10 -> 0.12. Planned resume LR 5e-5 -> 3e-5 and command_smoothing 0.10 -> 0.15.
- Rationale: Increase direct planar/yaw gradients while adding modest posture, contact, and torso-angular guardrails; the lower learning rate and smoother commands should reduce late checkpoint drift.
- Expected effect: Recover vx/vy/yaw tracking around the checkpoint_18200 window, keep response correlation near or above 0.90, and keep crash/fall termination near zero. Watch action-rate, torque-limit, KL, clip fraction, and AMP score.
- Result: Pending remote sync, restart, and post-resume evaluation.
- Files/commands: humanoidverse/amp_stage2.py; tuning-log.md; uv run ruff check; uv run python -m unittest tests/test_amp_stage2.py

## 2026-08-02 22:00:10 CST - Evaluate reward rebalance after resume18200

- Context: GPU10 physical GPUs 4-7, 4096 envs; new run reward_track_stability_resume18200_20260802
- Phenomenon: After 55 logged iterations (18200-18254), early resumed metrics are planar MAE 0.221, vx MAE 0.155, vy MAE 0.129, yaw MAE 0.230, vx correlation 0.909, response slope 0.690, termination 7e-6; old late window at 24980-25079 was planar 0.309, vx 0.236, vy 0.157, yaw 0.256, correlation 0.780.
- Analysis: The selected checkpoint and lower learning rate preserve the strongest observed tracking regime while the revised reward gives direct planar and yaw tracking more influence and modestly raises upright/contact stability terms.
- Adjustment: Applied linvel_exp 2.5, linvel_projection 0.95, angvel_z_exp 1.9, single_foot_contact 0.95, angvel_xy_l2 0.04, body_upright 1.2, joint_deviation_l2 0.12; resumed checkpoint_18200 with learning_rate 3e-5 and command_smoothing 0.15.
- Rationale: Avoid the regression seen after iteration 18k-25k and reduce policy update overshoot while retaining AMP prior.
- Expected effect: Maintain planar/vx/yaw errors near the 18.2k checkpoint regime, improve response correlation and reduce drift/fall events.
- Result: Pending longer window; process is alive at iteration 18254 with all four GPU workers.
- Files/commands: humanoidverse/amp_stage2.py; checkpoint_18200.pt; 100-300 iteration rolling metric recheck required.

## 2026-08-03 09:33:40 CST - Evaluate Stage2 reward rebalance through iteration 20327

- Context: GPU10 physical GPUs 4-7; run amp_stage2_piplus_lse_4gpu_4096env_1m_reward_track_stability_resume18200_20260802; resumed checkpoint_18200
- Phenomenon: Training reached iteration 20327 and stopped after torchrun received SIGHUP. First 100 vs last 100: planar MAE 0.2246 -> 0.2605, vx MAE 0.1586 -> 0.1969, vy MAE 0.1290 -> 0.1359, yaw MAE 0.2345 -> 0.2288, vx correlation 0.8982 -> 0.8748, response slope 0.6721 -> 0.5500, termination 6.94e-6 -> 1.45e-6, fall 0.99e-6 -> 0; AMP score 0.166 -> -1.091.
- Analysis: Stability improved and yaw tracking improved slightly, but linear velocity response progressively regressed. The strongest composite tracking windows were 18300-18500. AMP discriminator contribution also collapsed, so the run is not a clean overall improvement and checkpoint_20300 is not the preferred policy.
- Adjustment: No new parameter change or restart in this diagnostic turn.
- Rationale: User requested status only; avoid modifying external training state without explicit authorization.
- Expected effect: N/A
- Result: Mixed: stability improved, yaw slightly improved, planar/vx tracking worsened over the run; process is stopped.
- Files/commands: Pulled logs/.../checkpoint_20300.pt, config.json, and launcher.log; playback docs scanned but not changed because latest checkpoint is measurably worse than the 18300-18500 peak.

## 2026-08-03 09:34:31 CST - Correct exact terminal-rate averages for Stage2 evaluation

- Context: Same 18200-20327 reward-track-stability run; exact local JSON parse
- Phenomenon: The prior entry rounded terminal rates from a compact remote summary.
- Analysis: Exact first/last 100 means are termination 6.94275e-6 -> 1.296997e-6, crash 5.95093e-6 -> 1.144409e-6, and fall-over 9.91821e-7 -> 1.525879e-7.
- Adjustment: No training adjustment; record precision correction only.
- Rationale: Keep the append-only tuning record numerically accurate.
- Expected effect: N/A
- Result: Stability conclusion remains unchanged: terminal events decreased substantially.
- Files/commands: Parsed local launcher.log with JSON.

## 2026-08-03 10:08:06 CST - Prefer hardware-validated cmdx checkpoint 15200 as next finetune base

- Context: PiPlus AMP Stage2; compare three distinct checkpoint_15200.pt files, current 18400 resume run, and real-robot joystick feedback
- Phenomenon: User reports the deployed 15200 policy responds correctly to joystick commands but has unattractive posture. The matching deployment copy hashes to cmdx_m08_y05_act055_resume14800_20260802/checkpoint_15200.pt.
- Analysis: The cmdx 15151-15250 window has nonzero planar MAE 0.2215, vx MAE 0.1590, vx command correlation 0.8976, and response slope 0.6988, matching the hardware response. Its discriminator expert/policy scores are 0.049/-2.725, normalized AMP contribution is -0.0611, and yaw MAE is 0.3025, supporting weak imitation/posture. Hardware-validated command response should outrank later simulation-only checkpoint scores.
- Adjustment: Recommendation only: use the exact cmdx checkpoint_15200.pt as the next finetune policy base; preserve policy/command encoder, use stronger but controlled AMP/posture training, and do not treat the other two checkpoint_15200.pt files as interchangeable. No process change in this analysis turn.
- Rationale: This checkpoint preserves behavior already proven on hardware while isolating the remaining failure to imitation/posture. Starting from 18400 risks losing the known-good joystick response and has not been hardware validated.
- Expected effect: A finetune from cmdx 15200 should retain command responsiveness while improving gait appearance; guard against degradation in vx correlation and response slope.
- Result: Pending a dedicated resume cycle and real-robot playback validation.
- Files/commands: Compared /tmp/cmdx_m08_y05_act055_train_15243.log, retry launcher.log, current yaw_amp log; pulled current checkpoint_18500/config/launcher.log; docs scanned and left unchanged because checkpoint_18500 is not hardware validated.

## 2026-08-03 10:50:22 CST - Align stand_still with HT_lab_pipeline and resume cmdx 15200

- Context: PiPlus AMP Stage2 on GPU10 physical GPUs 4-7, 1024 environments per GPU (4096 global), target 1,000,000 iterations.
- Phenomenon: Hardware-validated checkpoint_15200 responds to joystick commands but stands with asymmetric/unnatural posture; yaw tracking and AMP imitation remain weak.
- Source comparison: `/home/sunteng/Project/HT_lab_pipeline/source/HT_lab/HT_lab/tasks/piplus_locomotion/mdp/rewards.py` defines `stand_still` as `sum(abs(joint_pos-default_joint_pos))`, gated by planar command norm `<0.1` and absolute yaw command `<0.1`; the PiPlus config uses weight `-0.8`.
- Adjustment: Replaced the prior custom exponential pose/motion term with the mathematically equivalent Stage2 term `-sum(abs(dof_pos-default_dof_pos))`, weighted by `0.8` and multiplied by the existing control timestep. Increased `angvel_z_exp` to `2.6`; increased stand sampling to `0.20` and pure-turn sampling to `0.20`; increased AMP weight to `0.06`; kept discriminator LR `1e-4`, policy LR `2e-5`, target KL `0.025`, and command smoothing `0.15`.
- Rationale: Preserve the HT_lab_pipeline reward semantics exactly while keeping body-upright and torso angular-velocity terms independent. The stand term is zero for moving/turning commands, so it cannot reward standing instead of tracking.
- Validation: `uv run ruff check humanoidverse/amp_stage2.py tests/test_amp_stage2.py`, `uv run python -m unittest tests/test_amp_stage2.py` (20 passed), and `uv run python -m py_compile humanoidverse/amp_stage2.py` all passed. Local source SHA256 `958e50390078099d0358c99b017d646fa9d85247d5d67d97b89b447e56ef22da` matches remote after patch application; old remote source is backed up as `humanoidverse/amp_stage2.py.bak.stand_yaw_amp006_resume15200_20260803`.
- Restart: Launched `logs/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_yaw_resume15200_20260803` from the exact cmdx checkpoint_15200.pt (SHA256 `1c8a97916b41e91ffdbbcb9560504e8d9606cdbeee06050e15b890eec61bb16e`). Remote config confirms resume iteration `15200`, global environments `4096`, stand/turn probabilities `0.2/0.2`, AMP `0.06`; workers 919663-919666 occupy physical GPUs 4-7.
- Early result: iterations 15201-15224 average planar MAE `0.2132`, vx MAE `0.1528`, nonzero yaw MAE `0.3159`, yaw correlation `0.6804`, yaw response slope `0.7159`, stand_still contribution `-0.0180`, AMP contribution `-0.0698`, termination rate `5.1e-6`, and approx KL `0.0133`. This is an initialization window, not a final improvement claim; reassess after at least 100-300 iterations and then playback the best stable checkpoint.
- Artifacts: `humanoidverse/amp_stage2.py`, `docs/amp_stage2_ht_bfm.md`, `logs/tuning/amp_stage2_stand_yaw_amp006_resume15200_20260803.patch`, and remote `/tmp/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_yaw_resume15200_20260803.log`.

## 2026-08-03 10:51:00 CST - Early health recheck of stand/yaw/AMP resume

- Observation: Run remains alive through iteration `15264` with four workers on GPUs 4-7. The `15225-15264` window averages planar MAE `0.2270`, nonzero yaw MAE `0.3157`, yaw correlation `0.6701`, yaw response slope `0.7328`, stand_still contribution `-0.0168`, AMP contribution `-0.0540`, termination `4e-6`, and approx KL `0.0137`.
- Assessment: Direct tracking is still noisy in this startup window and has not yet demonstrated improvement over the hardware-validated baseline; yaw response slope is moving in the desired direction, while AMP contribution and expert score are improving. Keep the run alive for a 100-300 iteration comparison before selecting a checkpoint or changing rewards.

## 2026-08-03 11:14:51 CST - Evaluate yaw/AMP resume through iteration 18813

- Context: PiPlus AMP Stage2; yaw_amp_resume18400_20260803; checkpoint_18700; local latest verified log
- Phenomenon: After the previous reward-rebalance run, which improved stability but degraded late linear tracking, the yaw/AMP run changed amp_weight 0.03 -> 0.04, linvel_exp 2.5 -> 2.8, linvel_projection 0.95 -> 1.1, angvel_z_exp 1.9 -> 2.25, and enabled command_turn_prob 0.15.
- Analysis: Compared with the previous run first 100 iterations, current first 100 reduced planar MAE 0.2056 -> 0.1904 and vx MAE 0.1440 -> 0.1331 while keeping vx correlation about 0.899; current last 100 versus previous last 100 is better on planar MAE 0.2398 -> 0.1996, vx MAE 0.1799 -> 0.1434, vx correlation 0.8748 -> 0.9007, and response slope 0.5500 -> 0.6265. Current yaw MAE is 0.2270 in the last 100, slightly below previous 0.2288, while termination remains near zero. Within the current run, vx tracking drifted mildly from the first to last 100, while yaw slope/correlation improved and AMP score rose.
- Adjustment: No new parameter change; retain current checkpoint lineage for observation.
- Rationale: The current reward/AMP/yaw balance recovers the previous late tracking regression and improves yaw signals without a safety regression, but the mild within-run vx drift requires a longer stable window before another intervention.
- Expected effect: Maintain termination near 1e-6 to 1e-5, keep vx correlation near 0.90 or higher, and prevent response slope from falling below about 0.60 while yaw MAE and yaw correlation continue improving.
- Result: Mixed positive: stability remains good, late linear tracking is materially better than the previous run, yaw tracking is slightly better, but current vx metrics are not monotonically improving. Latest verified iteration is 18813 with checkpoint_18700; no local active process is present.
- Files/commands: logs/amp_stage2_piplus_lse_4gpu_4096env_1m_yaw_amp_resume18400_20260803/launcher.log; logs/amp_stage2_piplus_lse_4gpu_4096env_1m_yaw_amp_resume18400_20260803/checkpoint_18700.pt; tuning-log.md

## 2026-08-03 11:15:27 CST - Correct lineage for latest Stage2 run

- Context: User clarification; latest intended run is amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_yaw_resume15200_20260803
- Phenomenon: The prior status message treated the stale local yaw_amp_resume18400 artifact as the current training run.
- Analysis: The tuning journal explicitly records that the active stand/yaw/AMP finetune resumed from the exact hardware-validated cmdx checkpoint_15200, with stand/turn probabilities 0.20/0.20 and AMP weight 0.06. The local 18400 artifact is a separate older lineage and cannot be used to report the current run.
- Adjustment: No training parameter change; correct the lineage attribution only.
- Rationale: Keep status comparisons tied to the active checkpoint lineage and avoid mixing incompatible runs.
- Expected effect: Future status reports should use amp006_stand_yaw_resume15200 logs/checkpoints and compare against its checkpoint_15200 baseline.
- Result: Correction accepted. Latest verified active point from the available record is iteration 15264; the amp006 run directory/log has not been pulled into the local logs tree, so later metrics are currently unavailable locally.
- Files/commands: tuning-log.md; logs/tuning/amp_stage2_stand_yaw_amp006_resume15200_20260803.patch

## 2026-08-03 11:30:11 CST - Evaluate remote amp006 stand/yaw resume through iteration 15907

- Context: GPU10 via JumpServer; physical GPUs 4-7; run amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_yaw_resume15200_20260803; baseline cmdx_m08_y05_act055 checkpoint_15200 window 15200-15299
- Phenomenon: The active run is confirmed alive from checkpoint_15200 with amp_weight 0.06, learning_rate 2e-5, target_kl 0.025, stand/turn probabilities 0.20/0.20. Latest checkpoint is 15900 and metrics reach iteration 15907.
- Analysis: Compared with the baseline 15200-15299 window, the current first 100 had vx_mae 0.1434 -> 0.1200 and yaw_mae 0.3008 -> 0.2845, with termination unchanged near 4e-6. By the latest 100 (15808-15907), AMP improved strongly: amp_score -2.7903 -> -0.3603, discriminator expert score 0.0489 -> 2.0894, policy score -2.7814 -> -0.3514, and AMP contribution -0.0569 -> +0.0087. Stability remains good: termination 3e-6, crash 3e-6, fall 0. However linear tracking regressed late: nonzero vx MAE 0.1560 -> 0.1993, vx correlation 0.9006 -> 0.8676, response slope 0.7017 -> 0.5721. Yaw improved late to yaw_mae 0.2612, correlation 0.7528, slope 0.8879. The run is stable but speed tracking is drifting while AMP and yaw improve.
- Adjustment: No new parameter change; keep the remote run alive and preserve checkpoint_15900 for the next controlled comparison.
- Rationale: AMP imitation and yaw response show clear gains without fall/termination regression, but another reward change now would confound whether the late vx drift recovers.
- Expected effect: Over the next 100-300 iterations, recover vx correlation toward 0.90 and response slope above 0.65 while retaining amp_score near -0.4 or better, yaw_mae near 0.26, and termination below 1e-5.
- Result: Operationally healthy and still training. Composite result versus checkpoint_15200 is mixed-positive: stability maintained, AMP imitation substantially improved, yaw tracking improved, but late speed tracking is worse than baseline.
- Files/commands: Remote /tmp/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_yaw_resume15200_20260803.log; remote checkpoint_15900.pt; tuning-log.md

## 2026-08-03 11:48:58 CST - Pull and evaluate amp006 resume checkpoint 16200

- Context: GPU10 physical GPUs 4-7; amp006_stand_yaw_resume15200_20260803; remote run from cmdx checkpoint_15200; checkpoint_16200 pulled locally and SHA256 verified
- Phenomenon: Training advanced from the prior observation near iteration 15907 to 16201. The checkpoint_16200 artifact, config, and console log were pulled through the JumpServer SFTP channel.
- Analysis: Relative to the run first 100 iterations (15200-15299), the latest 100 (16102-16201) retains low but slightly higher termination 4e-6 -> 7e-6, AMP score improves -1.757 -> +0.268, and AMP contribution -0.0523 -> +0.0327. Yaw improves: yaw MAE 0.2845 -> 0.2488, yaw correlation 0.6758 -> 0.7694, and yaw response slope 0.7323 -> 0.8901. Linear velocity continues regressing: nonzero vx MAE 0.1603 -> 0.2177, correlation 0.8992 -> 0.8497, and response slope 0.6878 -> 0.5235. Value loss rose 0.3505 -> 2.1816 despite KL and clip fraction remaining controlled.
- Adjustment: No training change. Pulled checkpoint_16200.pt, config.json, and launcher.log into the matching local logs directory. Updated clearly related Stage2 playback and ONNX export commands to checkpoint_16200.
- Rationale: The checkpoint is complete and hash-verified, so it can be documented for playback; however it is not a composite policy improvement because current speed tracking is materially weaker than the checkpoint_15200 baseline.
- Expected effect: Continue observing for a vx response recovery before treating later checkpoints as deployment candidates. Preserve checkpoint_15200 as the speed-response baseline and checkpoint_16200 as the current AMP/yaw candidate.
- Result: Mixed and not continuing to improve overall: AMP imitation and yaw tracking continue improving, stability remains acceptable, but linear-speed tracking continues to regress. Training process was active during inspection.
- Files/commands: Remote /tmp/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_yaw_resume15200_20260803.log; local logs/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_yaw_resume15200_20260803/checkpoint_16200.pt; docs/play_checkpoint_bfmzero-piplus-lse.md

## 2026-08-03 12:24:55 CST - Merge standing AMP expert and resume 16200 with velocity recovery tuning

- Context: PiPlus AMP Stage2 on GPU10 physical GPUs 4-7; merged expert dataset and controlled resume from checkpoint_16200
- Phenomenon: The 16102-16201 window had strong AMP/yaw gains but degraded nonzero vx tracking and elevated value loss: AMP score +0.268, AMP contribution +0.0327, yaw MAE 0.2488, nonzero vx MAE 0.2177, vx correlation 0.8497, vx slope 0.5235, value loss 2.1816. The user required AMP weight to remain 0.06.
- Analysis: Added the validated standing pose as a 149th motion to the existing 148-motion expert set. To reduce velocity/yaw competition and critic drift while preserving AMP strength, lowered angvel_z_exp 2.6->2.2, stand/turn sampling 0.20->0.15, PPO epochs 4->3, policy learning rate 2e-5->1.5e-5, and target KL 0.025->0.02; retained AMP weight 0.06, locomotion reward weight 1.1, discriminator LR 1e-4, and command smoothing 0.15.
- Adjustment: Generated dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run_with_stand.pkl from the original expert set plus default_pose_piplus_s_lse.pkl. Uploaded and hash-verified it remotely. Patched remote amp_stage2.py with a reversible backup, stopped the prior run, and resumed from the complete checkpoint_16200.pt.
- Rationale: The stand pose broadens AMP expert support for stable initialization; lower yaw/stand pressure and gentler PPO updates target the observed velocity regression and value-loss spike without changing the user-approved AMP coefficient.
- Expected effect: Recover vx MAE toward 0.16 or lower, vx correlation toward 0.90 and response slope above 0.65, while retaining low termination/fall rates, positive AMP contribution, and improved yaw tracking.
- Result: A first live window through iteration 16221 is operationally healthy: four workers are active, termination remains approximately 0-5e-5 with no sustained falls, AMP score is roughly 0.21-0.55 and AMP contribution 0.027-0.044, and value loss is mostly 0.1-0.2. Tracking remains noisy in this short startup window; continue monitoring before judging final improvement.
- Files/commands: data_process/merge_piplus_stand_amp_expert.py; dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run_with_stand.pkl; remote /tmp/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_velrecover_resume16200_20260803_retry2.log; remote backup humanoidverse/amp_stage2.py.bak.velrecover_20260803_115100; logs/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_velrecover_resume16200_20260803_retry2

## 2026-08-03 12:32:04 CST - Recheck 16200 velocity-recovery resume through iteration 16358

- Context: Remote run amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_velrecover_resume16200_20260803_retry2 on GPU10 cards 5-8, merged 149-motion AMP expert set
- Phenomenon: The run is healthy and writes checkpoint_16300, but speed tracking has not recovered during the first 158 post-resume iterations.
- Analysis: The 16250-16310 window averages AMP score 0.3297, AMP contribution +0.0326, vx MAE 0.2261, vx correlation 0.8436, vx slope 0.5028, yaw MAE 0.2737, yaw correlation 0.7579, yaw slope 0.8642, termination 9e-6, and value loss 0.3779. The 16300-16358 window raises AMP score to 0.5509 and AMP contribution +0.0417 while value loss stays 0.3904 and fall-over rate is 0; vx remains the dominant unresolved regression.
- Adjustment: No second parameter change yet; keep the controlled 16200 lineage alive to avoid confounding the first intervention. AMP weight remains exactly 0.06.
- Rationale: The current change achieved its critic/value-loss and AMP objectives with low termination risk, but the speed response has not shown a recovery signal. A second reward or command-distribution change should be selected only after a longer stable window or a targeted comparison.
- Expected effect: Next decision gate: recover vx MAE toward 0.16, correlation toward 0.90, and slope above 0.65 without giving back positive AMP contribution or low fall rate.
- Result: Mixed: operationally successful and improving AMP/critic stability; not yet a composite improvement because speed tracking remains below the 15200 baseline. Latest verified iteration 16358; checkpoint_16300 exists remotely.
- Files/commands: Remote /tmp/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_velrecover_resume16200_20260803_retry2.log; remote logs/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_velrecover_resume16200_20260803_retry2/checkpoint_16300.pt; tuning-log.md

## 2026-08-03 18:22:02 CST - Resume Stage2 from best tracking checkpoint with stronger velocity reward

- Context: PiPlus AMP Stage2; remote GPU10 physical GPUs 4,5,6,7; run amp_stage2_piplus_lse_4gpu_4096env_1m_rewardtrack_loco31_proj125_amp004_resume19000_20260803
- Phenomenon: The 19000-19099 window was the best stable tracking window in the active lineage: nonzero vx MAE 0.18146, planar MAE 0.25228, vx response slope 0.63527, while the latest 21900-21999 window regressed to 0.20911, 0.27597, and 0.58966. Termination remained low and PPO completed updates.
- Analysis: The residual bottleneck is direct planar velocity tracking rather than safety or PPO instability. The AMP contribution became noisy and was not a reliable primary task signal, so reduce its weight during the next resume while increasing the two existing direct planar tracking terms.
- Adjustment: MIMICLITE_LOCOMOTION_WEIGHTS linvel_exp: 2.8 -> 3.1; linvel_projection: 1.1 -> 1.25; launch amp-weight: 0.06 -> 0.04; resume checkpoint: checkpoint_19000.pt
- Rationale: Strengthen signed direct velocity and direction gradients while preserving observation/action/checkpoint compatibility and reducing fluctuating AMP shaping pressure.
- Expected effect: Improve nonzero vx and planar MAE and restore response slope toward or above the 19000 window without increasing crash/fall or PPO KL; watch value loss and AMP contribution.
- Result: Patch applied and backup created remotely, but both restart attempts failed before iteration 1 with Isaac Sim Vulkan ERROR_DEVICE_LOST on GPUs 4-7. No new checkpoint was produced; training processes were stopped. Requires GPU/container reset before retry.
- Files/commands: Remote backup humanoidverse/amp_stage2.py.bak.reward_track_20260803181145; /tmp/stage2_reward_track.patch; failed logs /tmp/amp_stage2_rewardtrack_loco31_proj125_amp004_resume19000_20260803.log; local humanoidverse/amp_stage2.py

## 2026-08-03 18:43:06 CST - Reset GPU10 and resume Stage2 reward-track run

- Context: PiPlus AMP Stage2; GPU10 physical GPUs 4,5,6,7; run amp_stage2_piplus_lse_4gpu_4096env_1m_rewardtrack_loco31_proj125_amp004_resume19000_20260803
- Phenomenon: The previous resume attempts hit Isaac Vulkan ERROR_DEVICE_LOST; user requested reset and resume.
- Analysis: User-level nvidia-smi reset was unavailable due insufficient permissions. After stopping stale workers and clearing the user-owned Isaac temporary directory, an explicit CUDA_VISIBLE_DEVICES=4,5,6,7 launch avoided the prior device initialization failure.
- Adjustment: Reset stale worker processes and /tmp/zhuzejian_isaaclab; resume checkpoint_19000 with linvel_exp=3.1, linvel_projection=1.25, amp_weight=0.04 and gpu_ids=all under CUDA_VISIBLE_DEVICES=4,5,6,7.
- Rationale: Restore the best tracking checkpoint with the diagnosed reward changes while keeping the physical GPU mapping explicit.
- Expected effect: Sustain vx/planar tracking improvement without renewed device failure, rising termination, or PPO instability.
- Result: Successful: four workers running; config records resume_iteration=19000, amp_reward_weight=0.04, linvel_exp=3.1, linvel_projection=1.25. Reached iteration 19020 with vx MAE about 0.155, planar MAE about 0.212, yaw MAE about 0.254, approx KL about 0.011; GPUs 4-7 use about 11.7 GiB each. First checkpoint_19100 was not yet written at verification time.
- Files/commands: Remote /tmp/zhuzejian_isaaclab; launch log /tmp/amp_stage2_rewardtrack_loco31_proj125_amp004_resume19000_20260803_retry4.log; remote work directory logs/amp_stage2_piplus_lse_4gpu_4096env_1m_rewardtrack_loco31_proj125_amp004_resume19000_20260803

## 2026-08-03 19:10:28 CST - Evaluate reward-track resume after GPU reset

- Context: PiPlus AMP Stage2; new run rewardtrack_loco31_proj125_amp004_resume19000_20260803; iterations 19000-19539
- Phenomenon: The resumed run is healthy and has produced checkpoint_19500, but the latest 100 logged iterations have worse direct tracking than the original checkpoint_19000 baseline window.
- Analysis: Compared with original 19000-19099: vx MAE 0.18146 -> 0.2205, planar MAE 0.25228 -> 0.28432, vx response slope 0.63527 -> 0.49282. Safety and PPO remain controlled, so this is not a crash or optimizer failure, but the reward-weight change has not yet demonstrated the intended tracking gain.
- Adjustment: No new parameter change; continue observation only.
- Rationale: Avoid compounding a tuning change before a longer comparable window confirms whether the early regression is transient.
- Expected effect: A longer window should either recover direct tracking or confirm that linvel_exp=3.1 and linvel_projection=1.25 are too aggressive for this resume.
- Result: Pending longer evaluation; latest checkpoint_19500 exists and four workers remain active.
- Files/commands: Remote log /tmp/amp_stage2_rewardtrack_loco31_proj125_amp004_resume19000_20260803_retry4.log; remote run logs/amp_stage2_piplus_lse_4gpu_4096env_1m_rewardtrack_loco31_proj125_amp004_resume19000_20260803

## 2026-08-03 20:11:01 CST - Resume Stage2 with projection and yaw reward emphasis

- Context: PiPlus AMP Stage2; GPU10 physical GPUs 4,5,6,7; run yaw26_proj15_amp004_smooth010_resume19000_20260803
- Phenomenon: Previous reward-track run plateaued with vx MAE about 0.220, planar MAE about 0.287, vx response slope about 0.498, despite stable PPO and low termination.
- Analysis: Increasing linvel_exp did not restore direct tracking. The remaining controllable signals are signed direction projection, yaw tracking weight, and command target smoothing, which can improve directional response without changing the checkpoint interface.
- Adjustment: linvel_projection: 1.25 -> 1.5; angvel_z_exp: 2.2 -> 2.6; command-smoothing: 0.15 -> 0.10; resume checkpoint_19000; amp-weight remains 0.04.
- Rationale: Increase directional and yaw gradients while reducing command filtering lag; keep linvel_exp fixed at 3.1 and preserve safety/PPO settings.
- Expected effect: Lower vx/planar/yaw tracking MAE and raise response slopes toward 0.65/1.0 without increased crash/fall or KL instability.
- Result: Restart succeeded; reached iteration 19015. Early vx MAE 0.161-0.188, vx response slope 0.636-0.644, yaw response slope about 0.993-1.000, approx KL about 0.010-0.011, termination rate 1.5e-5 to 3.1e-5. Longer window pending.
- Files/commands: Remote backup humanoidverse/amp_stage2.py.bak.yaw_projection_20260803200831; /tmp/stage2_yaw_projection.patch; run logs/amp_stage2_piplus_lse_4gpu_4096env_1m_yaw26_proj15_amp004_smooth010_resume19000_20260803

## 2026-08-03 20:38:42 CST - Evaluate projection and yaw reward resume

- Context: PiPlus AMP Stage2; run yaw26_proj15_amp004_smooth010_resume19000_20260803; latest checkpoint_19500, iteration about 19528
- Phenomenon: The new projection/yaw/smoothing reward run is stable but direct tracking did not continue improving after the early 19015 signal.
- Analysis: Compared with its own 19000-19099 window, latest 19429-19528 worsened vx MAE 0.18725 -> 0.21234, planar MAE 0.25669 -> 0.27725, and vx response slope 0.59941 -> 0.51697. Yaw MAE stayed roughly flat at 0.280 -> 0.265 and yaw response slope stayed near 0.97. Safety and PPO remain healthy; AMP contribution is negative and fluctuating.
- Adjustment: No new parameter change in this evaluation.
- Rationale: The tested projection/yaw weight increase and command smoothing reduction did not produce a sustained direct tracking gain; avoid stacking another reward change without a new diagnosis.
- Expected effect: Pending decision on whether to roll back to the best stable checkpoint/config or inspect command distribution and reward semantics further.
- Result: No sustained improvement; latest checkpoint_19500 exists and training remains active.
- Files/commands: Remote log /tmp/amp_stage2_yaw26_proj15_smooth010_resume19000_20260803.log; remote run logs/amp_stage2_piplus_lse_4gpu_4096env_1m_yaw26_proj15_amp004_smooth010_resume19000_20260803

## 2026-08-03 21:35:56 CST - Constrain Stage2 latent to BFM manifold

- Context: HT_BFM Stage2 AMP+BFM; prior Stage2 checkpoint lineage
- Phenomenon: Velocity/yaw tracking did not improve consistently; norm_z=True only enforced projected latent radius and did not measure direction support.
- Analysis: The frozen BFM actor receives projected command-encoder latents, while the Stage2 code validated only ||z||=sqrt(z_dim). The piplus_walk_amp reference uses a separate AMP contract, so latent direction mismatch must be diagnosed independently.
- Adjustment: Added expert projected-latent reference bank, nearest-expert cosine prior penalty (default weight 0.02), raw latent-stat initialization for norm_z=True, and global/stand/turn/slow/medium/fast cosine-kNN-MMD diagnostics.
- Rationale: Keep generated latent directions near backward-map expert support while preserving old policy checkpoint shapes and resume compatibility.
- Expected effect: Higher nearest-expert cosine and lower kNN/MMD distance, with less latent-induced action drift; monitor whether vx/yaw tracking improves without increasing falls.
- Result: Pending remote rollout.
- Files/commands: humanoidverse/amp_stage2.py; tests/test_amp_stage2.py; uv run python -m unittest tests.test_amp_stage2

## 2026-08-04 22:34:00 CST - Resume Stage2 with PiPlus reward/AMP alignment

- Context: HT_BFM Stage2 on GPU10 physical GPUs 4,5,6,7; resumed from checkpoint_21200.pt.
- Adjustment: Positive exponential velocity/yaw tracking, PiPlus MSE + gradient-penalty discriminator, quadratic AMP reward, command stand/turn probabilities 0.05, smoothing 0.02, PPO lr 1e-4, 5 epochs, entropy 0.003.
- Rationale: Remove signed baseline reward variance and align AMP/task reward contracts with the known-good piplus_walk configuration.
- Compatibility: Encoder/policy optimizer resumed; old 512/256 WGAN discriminator was incompatible with the new 1024/512 MSE discriminator, so discriminator weights and optimizer were reset deliberately.
- Result: Upload and resume succeeded; bfm_latent_check passed, start_iteration=21200, four workers active. The 21200-21299 window improved AMP contribution to 0.1190 and lowered latent MMD to 0.3694, but primary tracking remained mixed (vx MAE 0.1755, planar MAE 0.2462, yaw MAE 0.2584). By 21500-21585, AMP contribution stayed positive at 0.0773 and MMD fell to 0.2969, while tracking drifted to vx 0.1826, planar 0.2604, yaw 0.2704; classify as stable but not a sustained tracking improvement.
- Files/commands: remote backup humanoidverse/amp_stage2.py.bak_20260804_piplusquad; remote backup humanoidverse/amp_stage2.py.bak_20260804_precompat; /tmp/amp_stage2_piplusquad_resume21200_20260804.log; local commit ee1b687.

- Follow-up result through checkpoint_21700: the 21586-21685 window has lower latent MMD (0.2822 vs 0.2969) and better yaw MAE (0.2508 vs 0.2704), but vx MAE worsened (0.1937 vs 0.1826), planar MAE worsened (0.2711 vs 0.2604), and vx response slope fell (0.4495 vs 0.4737). Overall classification remains mixed/stable, not a sustained locomotion improvement.

- Latest result through checkpoint_22200: speed tracking recovered in the 22086-22200 window (vx MAE 0.1639, planar MAE 0.2332, vx slope 0.6436 versus 0.178/0.236/0.4717 before the resume), while yaw remained weaker (yaw MAE 0.2622, yaw slope 0.8372 versus 0.241/0.9359). Termination stayed zero, but value loss rose to 2.38, so classify as partial speed improvement with critic-health risk, not an across-the-board improvement.
