# 第二阶段 AMP 训练流程

## 1. 对齐结论

本项目 `/home/sunteng/Project/HT_BFM` 的 PiPlus Stage2 AMP 已按
`/home/sunteng/Project/UFO/humanoidverse/amp_stage2.py` 逐项对齐。两边的
训练目标、latent 采样、冻结 BFM actor、AMP 判别器、MimicLite 奖励、PPO/GAE
和多卡同步逻辑一致；Isaac 与 MJLab、PiPlus 与 H1 的物理接口差异保留在适配层。

本次补齐的会改变训练目标的差异：

- 环境奖励恢复 UFO 的 `survival=2.0`，并将 Stage2 的 `penalty_undesired_contact` 覆盖为 `-1.0`。
- Isaac Stage2 使用 UFO 相同的 crash 判定：配置的非期望接触体接触力范数大于 `1.0`。
- 使用相同的 fall-over 判定：`||projected_gravity_xy|| >= 0.9`。
- Isaac 自动 reset 前保存 terminal state 和 terminal observation，timeout 能正确使用终止前 value bootstrap。
- AMP 在线特征、MimicLite reward 和终止分类都读取终止前物理帧，不读取 reset 后的初始姿态。
- PiPlus 躯干奖励体使用 `isaacsim_torso_name=waist_yaw_link`，不再错误回退到 `base_link`。
- model-only checkpoint 也会通过当前 motion library 重建专家 observation，运行冻结 backward map 得到真实 latent 统计。
- 训练入口检查环境 action space、BFM action dimension 和 PiPlus 23 个 policy joints 三者一致。

仍然保留的机器人相关差异是有意的：PiPlus 为 23 DoF，Isaac 固定 wrist link
合并到 elbow body，因此 AMP key body 使用 ankles、elbows、head pitch；这些是
PiPlus XML/Isaac body contract 的必要映射，不是算法差异。

## 2. 输入与目录

默认输入如下：

```text
BFM Stage1 checkpoint:
  huiying/bfmzero-piplus-lse-isaac-20260715_143758(1)/checkpoint

AMP expert dataset:
  dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run_with_stand.pkl

PiPlus robot config:
  humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H1W_LSE.yaml

训练入口:
  python -m humanoidverse.amp_stage2
```

checkpoint 中的 `FBcprAuxModel` 被完整加载后冻结。Stage2 不更新 BFM actor、
forward/backward map、critic、observation normalizer 或第一阶段 replay buffer。
只训练 command encoder/value head、AMP WGAN-GP discriminator 和 AMP reward normalizer。

`--dry-run` 已验证当前 checkpoint、PiPlus action dimension、AMP 数据和 latent 投影：

- policy joints/action dimension: `23`
- latent dimension: `256`
- `norm_z`: `true`
- AMP feature dimension: `202`
- motion count: `153`
- valid AMP windows: `44635`
- projected latent norm: `sqrt(256)=16`

## 3. 单步训练链路

每个环境步执行以下顺序：

1. 读取 BFM observation，拼接 `[command, state, last_action, history_actor]`。
2. command encoder 用 Gaussian policy 采样 raw `z`，记录 log-prob 和 value。
3. 调用 checkpoint 的 `project_z(raw_z)`，再送入冻结 BFM actor 得到 23 维动作。
4. Isaac 环境执行动作，计算 UFO 对齐后的环境 reward，并在自动 reset 前保存 terminal state。
5. 使用本地 root velocity、五个 key-body local position 和 8 帧 23DoF joint history 构造 202 维 online AMP feature。
6. 计算 MimicLite command-conditioned locomotion reward。
7. 用专家 AMP feature 和 online feature 更新 WGAN-GP 判别器。
8. 判别器 raw score 经过 running normalizer 后乘 `amp_weight`，与环境 reward、MimicLite reward 相加。
9. 按 terminal/truncated 分离的 GAE 计算 PPO advantage/return；真实终止不 bootstrap，timeout 使用 terminal observation 的 value bootstrap。
10. PPO 更新 command encoder/value head；按保存周期写 Stage2 checkpoint。

当前调参 run 的 command 范围为 `[-0.8, -0.5, -0.8]` 到 `[0.8, 0.5, 0.8]`，每 `300` 步按 `0.75`
概率重采样，前 `20` 步为 warmup，command smoothing 为 `0.1`，低速或 stand gate
会生成零速度站立指令。

## 4. 奖励与终止

Stage2 总奖励为：

```text
reward = env_reward_weight * env_reward
       + locomotion_reward_weight * mimiclite_reward
       + amp_weight * normalized_discriminator_reward
```

默认三项权重分别为 `1.0`、`1.0`、`0.04`。

MimicLite reward 的最终每步贡献为 `dt * weight * raw_term`：

| term | weight |
| --- | ---: |
| linvel_exp | 2.1 |
| linvel_projection | 0.6 |
| angvel_z_exp | 1.4 |
| backward_velocity_progress | 0.9 |
| turn_rate_progress | 0.65 |
| single_foot_contact | 0.75 |
| angvel_xy_l2 | 0.02 |
| body_upright | 1.0 |
| feet_air_time | 2.0 |
| energy_l1 | 0.0002 |
| joint_acc_l2 | 0.0000001 |
| action_rate_l2 | 0.005 |
| action_rate2_l2 | 0.005 |
| joint_vel_l2 | 0.001 |
| joint_deviation_l2 | 0.1 |

偏航跟踪项 `angvel_z_exp` 使用相对零偏航基线的有符号改进奖励：非零偏航指令下原地不转的基线得分为 0，实际转向不足会产生负向信号，零偏航指令仍保留普通稳定性奖励。

环境 reward 使用 `reward_bfm_zero` 的原有项，并在 Stage2 入口覆盖：

```text
survival = +2.0
penalty_undesired_contact = -1.0
```

Stage2 不再使用 motion-end 或 motion-far termination。碰撞终止体来自 PiPlus
配置中的 `terminate_after_contacts_on`，当前为 `waist_yaw/head/shoulder/hip`。

为配合扩大的速度指令域，Stage2 入口将环境 `penalty_action_rate` 从 `-0.5`
收紧到 `-0.55`，用于抑制硬件侧过快的动作跳变。

## 5. Noise、Delay 与 Domain Randomization

第二阶段使用与本项目 PiPlus Stage1 完全相同的 `exp/bfm_zero_piplus/bfm_zero_piplus`
环境配置，`disable_obs_noise=False`、`disable_domain_randomization=False`。当前数值为：

| 类别 | 配置 |
| --- | --- |
| observation noise | `base_ang_vel=0.2`、`projected_gravity=0.05`、`dof_pos=0.01`、`dof_vel=0.5`；其余 `0` |
| noise curriculum | 关闭（`add_noise_currculum=False`），所以 noise multiplier 恒为 `1.0` |
| control delay | 开启，按环境 slot 随机 `0..2` 个 policy step，episode reset 时重新采样 |
| push | 开启；间隔 `[1,3] s`，线速度 `xy=0.5 m/s`、`z=0.1 m/s`，角速度 `0.5 rad/s`，恢复 `2 s` |
| base COM | 开启；x/y/z 均为 `[-0.02,0.02] m` |
| link mass | 开启；比例 `[0.8,1.2]` |
| PD gain | 开启；Kp `[0.8,1.2]`，Kd `[0.9,1.1]` |
| friction | 开启；摩擦 `[0.3,2.0]`，restitution `[0.05,0.5]`，512 buckets，一致化开启 |
| default pose | 开启；每个 DoF offset `[-0.02,0.02] rad` |
| reset DoF state | 开启；位置 `[-0.15,0.15] rad`，速度 `[0,0]` |

这里的 `control delay` 是 HumanoidVerse 的 action queue 随机延迟；PiPlus
actuator 配置中各组 `HTMotorCfg` 的 `min_delay=0,max_delay=4` 是另一层 actuator
延迟，也和第一阶段使用同一份机器人配置。Stage2 不额外关闭或修改这两层延迟。

通用 `terminate_by_contact/gravity` 保持为 `False`，这是 Isaac 环境构建的契约；
Stage2 在 reset hook 中按 `terminate_after_contacts_on` 和 projected-gravity 阈值
执行自己的 crash/fall 判定，训练语义不变。

## 6. PPO、AMP 和保存

默认 PPO 参数：

```text
rollout_steps = 32
ppo_epochs = 4
minibatch_size = 1024
learning_rate = 3e-4
clip_ratio = 0.2
value_coef = 0.5
entropy_coef = 0.001
discount = 0.99
gae_lambda = 0.95
```

判别器为 `202 -> 512 -> 256 -> 1` 的 LeakyReLU WGAN-GP，gradient penalty 权重为
`10.0`。默认 discriminator learning rate 为 `2e-4`。Stage2 checkpoint 保存
command encoder、optimizer、discriminator、discriminator optimizer、AMP normalizer、
iteration 和 metadata，可使用 `--resume` 继续训练。

## 7. 多卡训练

本项目通用 Fabric 配置没有接入该训练入口，Stage2 使用 PyTorch 内置
`torch.distributed.run`。每个 rank 创建独立 Isaac 环境；trainable module 初始状态
广播，梯度 all-reduce 平均，reward normalizer buffer 同步，标量 metric 平均，只有
rank 0 保存 checkpoint。每卡的 `--num-envs` 是本卡环境数，全局 batch 为
`world_size * num_envs * rollout_steps`。

服务器上使用全部可见 GPU：

```bash
python -m humanoidverse.amp_stage2 \
  --bfm-checkpoint huiying/bfmzero-piplus-lse-isaac-20260715_143758\(1\)/checkpoint \
  --expert-dataset dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run_with_stand.pkl \
  --robot-config humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H1W_LSE.yaml \
  --device cuda --gpu-ids all \
  --num-envs 1024 --iterations 10000 --rollout-steps 32 \
  --work-dir logs/amp_stage2_piplus_lse
```

只使用指定 GPU 时，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m humanoidverse.amp_stage2 \
  --device cuda --gpu-ids all --work-dir logs/amp_stage2_piplus_lse
```

单卡调试使用 `--gpu-ids single`。服务器正式运行前先执行：

```bash
python -m humanoidverse.amp_stage2 --dry-run --device cpu
```

有 IsaacLab/GPU 后再执行小规模验证：

```bash
python -m humanoidverse.amp_stage2 --smoke --gpu-ids single --num-envs 2
```

## 8. 验证记录

已完成：

- `python -m unittest tests/test_amp_stage2.py`：12 tests passed。
- `uv run ruff check humanoidverse/amp_stage2.py tests/test_amp_stage2.py`：passed。
- `python -m humanoidverse.amp_stage2 --dry-run`：checkpoint、23DoF、202 维 AMP 数据和 latent contract passed。
- `python -m py_compile humanoidverse/amp_stage2.py humanoidverse/envs/legged_base_task/legged_robot_base.py`：passed。

尚未在本机启动 Isaac GPU 长跑；正式 smoke、显存配置和长时间收敛验证放到训练服务器执行。
`humanoidverse/envs/legged_base_task/legged_robot_base.py` 本身存在历史 Ruff 报告，
本次只新增了通用 survival reward、奖励分量记录和 Stage2 terminal snapshot hook，未做无关格式化。
