# HT_BFM PiPlus 第二阶段 AMP 训练流程

本文档说明 HT_BFM 当前 PiPlus 第二阶段 AMP 训练的实际实现。第二阶段不重新训练
第一阶段 BFM，也不使用动作级 imitation loss，而是在冻结 BFM 的 latent 空间上训练
一个由速度指令控制的 Command Encoder，并通过 AMP discriminator 约束生成动作的
运动风格。

## 1. 目标链路

```text
速度指令 command = [vx, vy, wz]
                 + BFM actor observation
                 |
                 v
         CommandEncoderPolicy
                 |
              raw z
                 |
       z = BFM.project_z(raw z)
                 |
          冻结 BFM Actor
                 |
              action
                 |
      PiPlus Isaac Lab 环境
```

Command Encoder 的输入为：

```text
[vx, vy, wz] + obs["state"] + obs["last_action"] + obs["history_actor"]
```

当前输入会做字段级缩放，但不改变 checkpoint 的输入维度：

- command 使用 `[1.25, 5.0, 1.25]` 缩放；
- 当前帧和历史中的关节速度使用 `0.05` 缩放；
- 旧格式 checkpoint 在加载时会对第一层权重做等价迁移；
- 发生输入迁移时重置 policy Adam 状态，避免继续使用旧坐标系下的动量。

送入冻结 BFM actor 前，`raw_z` 必须调用第一阶段 checkpoint 自带的
`FBModel.project_z()`。当前 `z_dim=256`、`norm_z=true`，投影后的 latent L2 范数
约为 `sqrt(256)=16`，不是逐维 `[0, 1]` 归一化。

## 2. 冻结范围与训练范围

冻结模块包括：

- 第一阶段 BFM actor；
- backward/forward map、critic 和 auxiliary critic；
- 第一阶段 observation normalizer；
- 第一阶段机器人、动作空间和 latent contract。

第二阶段更新：

- `CommandEncoderPolicy.trunk`；
- `latent_mean` 和 `latent_log_std`；
- PPO value head；
- AMP discriminator。

当前 Command Encoder 继承第一阶段 backward map 的结构参数：

```text
input -> Linear(256) -> LayerNorm -> Tanh
      -> latent mean (256)
      -> latent log std (256)
      -> value (1)
```

## 3. PiPlus 机器人与专家数据

默认第一阶段 checkpoint：

```text
huiying/bfmzero-piplus-lse-isaac-20260715_143758(1)/checkpoint
```

默认机器人配置：

```text
humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H1W_LSE.yaml
```

默认专家数据：

```text
dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run_with_stand.pkl
```

当前训练 contract：

```text
PiPlus policy DoF       23
BFM action dimension   23
latent z dimension     256
expert motions         153
AMP feature dimension  202
history length         8
```

每个 AMP feature 由以下字段组成：

```text
root local linear velocity       3
5 个 key body 的 local position  5 * 3
8 帧 23DoF joint position        8 * 23
                                 ------
                                    202
```

key bodies 为：

```text
l_ankle_roll_link
r_ankle_roll_link
l_elbow_link
r_elbow_link
head_pitch_link
```

PiPlus 的 wrist 是 fixed link，导入 Isaac 后并入 elbow，因此在线环境和 MuJoCo
专家特征统一使用 elbow body，不能直接照搬 H1 的 wrist body 名称。

## 4. 环境初始化与终止

第二阶段创建 PiPlus locomotion 环境，而不是 motion-reference tracking reset 环境：

1. 使用 `PiPlus_S_12L8A0G2H1W_LSE` 的 23DoF action、PD 和 actuator 配置；
2. `locomotion_mode=True`，每个环境从默认姿态开始；
3. 关闭 motion-end 和 motion-far termination；
4. 环境 observation noise 默认关闭，第一阶段 domain randomization 保持开启；
5. 专家数据只用于 AMP feature，不用于每步 reference tracking；
6. survival reward 由 PiPlus 环境提供，不额外叠加 Stage2 alive reward。

Stage2 自己维护两类真实终止：

- crash：`waist_yaw_link`、`head`、`shoulder` 或 `hip` 等终止接触体受力超过阈值；
- fall-over：projected gravity 的 xy 范数达到 `0.9`。

time-limit truncation 与真实 terminal 分开处理。truncation 会使用 reset 前最终 observation
计算 bootstrap value，但会截断 GAE，避免 return 跨 episode 串联。

## 5. Rollout 与速度指令

每个 rollout step 的顺序为：

1. 拼接 observation 和当前 `[vx, vy, wz]`；
2. Command Encoder 采样 `raw_z`、log probability 和 value；
3. 使用冻结 BFM 的 `project_z()` 得到 actor latent；
4. 冻结 BFM actor 输出 23 维 action；
5. 推进 Isaac 环境并计算环境/locomotion reward；
6. 更新 8 帧 joint history，构造 policy AMP feature；
7. 保存 PPO rollout、terminal snapshot 和跟踪诊断量；
8. rollout 结束后更新 discriminator，再计算 AMP reward 和 PPO returns。

当前 command 范围：

```text
vx in [-0.8, 0.8] m/s
vy in [-0.5, 0.5] m/s
wz in [-0.8, 0.8] rad/s
```

command 以 `0.15` 的平滑系数逼近目标值。默认每 300 step 检查重采样，重采样概率
为 `0.75`，warmup 为 20 step，stand 概率为 `0.2`，与 HT_lab_pipeline 的 PiPlus
locomotion 配置一致。为避免偏航 MAE 被零指令稀释，
当前额外使用 `command_turn_prob=0.20`，即 20% 新指令为纯原地偏航指令。

## 6. Locomotion 奖励

线速度和偏航奖励使用“相对零响应基线”的有符号形式。非零指令下，如果机器人完全
不响应，奖励约为 0；正确跟踪为正，反向响应可以为负，从而避免策略靠站立获得较高
指数奖励。

当前 MimicLite locomotion 权重：

```text
linvel_exp             2.8
linvel_projection      1.1
angvel_z_exp           2.6
backward_velocity_progress 0.9
turn_rate_progress      0.65
single_foot_contact    0.85
angvel_xy_l2           0.035
body_upright           1.1
stand_still            0.8
feet_air_time          2.0
energy_l1              2.0e-4
joint_acc_l2           1.0e-7
action_rate_l2         0.005
action_rate2_l2        0.005
joint_vel_l2           1.0e-3
joint_deviation_l2     0.11
```

每个 term 先乘环境 control `dt`，再乘 `--locomotion-reward-weight`。正式 15200
finetune 使用 `locomotion_reward_weight=1.1`。

`stand_still` 与 `/home/sunteng/Project/HT_lab_pipeline` 的 PiPlus locomotion 实现保持
数学一致：只在平面速度与偏航指令均小于 `0.1` 时，对全关节相对默认姿态的 L1 偏差
施加 `-0.8` 权重。躯干竖直和角速度分别由 `body_upright`、`angvel_xy_l2` 处理；存在
移动或转向指令时 `stand_still` 严格为零，因此不会用“站着不动”覆盖跟踪目标。

## 7. AMP discriminator 与总奖励

AMP discriminator 是 `202 -> 512 -> 256 -> 1` 的 LeakyReLU WGAN-GP：

```text
L_D = mean(D(policy)) - mean(D(expert)) + 10 * gradient_penalty
```

不使用 sigmoid。原始 score 经 decay `0.999` 的 running normalizer 标准化：

```text
r_amp_score = D(policy_feature)
r_amp_normalized = VecNorm(r_amp_score)
r_amp = amp_weight * r_amp_normalized
```

本轮从真机验证的 `checkpoint_15200.pt` resume 时，将 AMP 权重从 `0.04` 提高到
`0.06`，同时将 discriminator learning rate 从 `2e-4` 降为 `1e-4`。resume 会加载
discriminator optimizer 状态，但强制覆盖其学习率，避免 checkpoint 中旧学习率继续生效。

最终用于 GAE/PPO 的 reward 是直接相加：

```text
r_total = 1.0 * r_env
        + 1.1 * r_mimiclite_locomotion
        + 0.06 * normalized_amp_reward
```

`0.06` 是归一化 AMP reward 的缩放系数，不表示 AMP 固定占总奖励 6%。PPO 是优化
算法，不是另一个占比 `0.95` 的奖励项。实际贡献应查看日志中的：

```text
env_reward_contribution
mimiclite_locomotion_reward_contribution
amp_reward_contribution
```

## 8. PPO 与多卡同步

本轮正式参数：

```text
rollout_steps                  32
ppo_epochs                     4
minibatch_size                 1024
policy learning rate           2e-5
target KL                      0.025
clip ratio                     0.2
value coefficient              0.5
entropy coefficient            0.001
discriminator learning rate    1e-4
discount                       0.99
GAE lambda                     0.95
max policy grad norm           1.0
```

Stage2 使用 `torch.distributed.run`。每个 rank 创建独立 Isaac 环境；policy 与
discriminator 梯度做 all-reduce 平均，reward normalizer buffer 同步，只有 rank 0
保存 checkpoint。

`--num-envs` 是每张 GPU 的环境数。GPU10 物理卡 5–8 对应 CUDA index `4,5,6,7`；
每卡 1024 环境时，全局环境数为 4096，每轮 rollout 的全局 transition 数为：

```text
4 * 1024 * 32 = 131072
```

## 9. 本轮 15200 Resume

远端有三个同名 `checkpoint_15200.pt`，本轮必须使用已经真机验证过的 `cmdx` lineage：

```text
logs/amp_stage2_piplus_lse_4gpu_4096env_1m_cmdx_m08_y05_act055_resume14800_20260802/checkpoint_15200.pt
```

SHA256：

```text
1c8a97916b41e91ffdbbcb9560504e8d9606cdbeee06050e15b890eec61bb16e
```

正式 4 卡 resume 命令：

```bash
python -u -m humanoidverse.amp_stage2 \
  --device cuda \
  --gpu-ids 4,5,6,7 \
  --num-envs 1024 \
  --iterations 1000000 \
  --rollout-steps 32 \
  --save-every 100 \
  --amp-weight 0.06 \
  --locomotion-reward-weight 1.1 \
  --ppo-epochs 4 \
  --learning-rate 2e-5 \
  --target-kl 0.025 \
  --discriminator-learning-rate 1e-4 \
  --command-smoothing 0.15 \
  --command-stand-prob 0.20 \
  --command-turn-prob 0.20 \
  --resume logs/amp_stage2_piplus_lse_4gpu_4096env_1m_cmdx_m08_y05_act055_resume14800_20260802/checkpoint_15200.pt \
  --work-dir logs/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_yaw_resume15200_20260803
```

`--iterations 1000000` 是最终 iteration 上限，不是再训练一百万轮。resume 后从
checkpoint 内记录的 15200 继续递增。

服务器没有 `tmux` 时，应使用 `setsid` 创建无控制终端的新会话，并重定向 stdin、
stdout 和 stderr，避免 SSH 断开向 `torchrun` 传播 `SIGHUP`。

## 10. 输出文件与恢复语义

每个工作目录包含：

```text
config.json
checkpoint_<iteration>.pt
```

checkpoint 保存：

- Command Encoder 和 value head；
- policy optimizer；
- discriminator 和 discriminator optimizer；
- AMP reward normalizer；
- iteration 与 metadata。

resume 时会：

- 恢复 policy/discriminator/normalizer；
- 恢复兼容的 optimizer 状态；
- 强制使用命令行给出的 policy/discriminator learning rate；
- 在旧 checkpoint 输入格式下执行等价第一层迁移，并在必要时重置 policy optimizer。

历史目录必须保留。新的 reward、optimizer 或 checkpoint 选择使用新 `work-dir`，不要
向旧目录覆盖写入。

## 11. 训练结果检查

主任务指标：

```text
tracking/nonzero_planar_l2_mae
tracking/nonzero_vx_mae
tracking/nonzero_vy_mae
tracking/nonzero_vx_command_correlation
tracking/nonzero_vx_response_slope
tracking/nonzero_yaw_rate_mae
tracking/nonzero_yaw_command_correlation
tracking/nonzero_yaw_response_slope
```

稳定性护栏：

```text
termination_rate
termination_crash_rate
termination_fall_over_rate
reward_contribution/mimiclite/body_upright
reward_contribution/mimiclite/stand_still
reward_contribution/mimiclite/single_foot_contact
reward_contribution/env/penalty_action_rate
```

AMP 学习健康度：

```text
amp_score
amp_reward_contribution
discriminator_expert_score
discriminator_policy_score
discriminator_score_gap
discriminator_gradient_penalty
discriminator_gradient_norm
```

PPO 学习健康度：

```text
approx_kl
clip_fraction
ppo_update_fraction
value_loss
grad_norm
entropy
```

不要只看总 reward。速度 MAE 上升、response slope/command correlation 下降时，即使
AMP reward 或总 reward 上升，也应判定为指令跟踪退化。偏航必须看非零指令相关性和
响应斜率，单独的全样本 yaw MAE 会被零指令稀释。

当前日志没有 mean episode length 字段，稳定性使用真实 termination/crash/fall-over
rate 判断。

## 12. 验证与播放

本地静态验证：

```bash
uv run ruff check humanoidverse/amp_stage2.py tests/test_amp_stage2.py
uv run python -m unittest tests/test_amp_stage2.py
uv run python -m py_compile humanoidverse/amp_stage2.py
```

快速播放固定指令：

```bash
python -m humanoidverse.amp_stage2_play \
  --model-folder logs/<stage2_run> \
  --checkpoint logs/<stage2_run>/checkpoint_<iteration>.pt \
  --simulator mujoco \
  --device auto \
  --fixed-command 0.4 0.0 0.0
```

Isaac Sim 手柄验证：

```bash
python -m humanoidverse.amp_stage2_play \
  --model-folder logs/<stage2_run> \
  --checkpoint logs/<stage2_run>/checkpoint_<iteration>.pt \
  --simulator isaacsim \
  --device cuda:0 \
  --policy-device cpu \
  --gamepad \
  --show-viewer
```

默认映射为左摇杆控制 `vx/vy`、右摇杆 X 控制 `wz`，A 复位、B 退出。真机/手柄
响应、姿态观感和接触自然度是 checkpoint 选择的重要依据，优先级高于单独的仿真总
reward。

## 13. 当前实现边界

第二阶段只优化 command-conditioned latent policy，不会修改第一阶段 BFM actor。
AMP 能在已有 latent 行为空间中偏向更像专家的动作，但无法凭空生成第一阶段 BFM
完全没有覆盖的步态。

本轮从 `cmdx checkpoint_15200.pt` 开始，是因为该 checkpoint 已在真机证明手柄指令
响应正常；其主要剩余问题是姿态和 AMP imitation。后续 checkpoint 的验收条件是：

1. 保持或提高 `vx`/偏航 command correlation 与 response slope；
2. 降低非零速度和偏航 MAE；
3. AMP contribution 不再长期显著为负，expert-policy score gap 不持续扩大；
4. crash/fall-over 不恶化；
5. 通过 Isaac 播放和真机手柄测试确认姿态更自然。
