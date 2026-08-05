# PiPlus 22DoF PPO Command Encoder 实验记录

> 本文档是 `HT` 分支 PiPlus 22DoF 速度控制实验的主记录。每次修改训练语义、启动训练、调整配置、恢复 checkpoint 或完成评估后，必须在文末的实验日志追加：日期、代码改动、完整命令、配置、结果和结论。

## 1. 实验目的

目标机器人是 `PiPlus_S_12L8A0G2H0W`，即 HT PiPlus 22DoF 版本。实验不训练 BFM 本体，而是冻结已导出的 BFM decoder，仅训练一个 PPO command encoder：

```text
速度指令 [vx, vy, wz] + 当前状态/动作历史
                 |
                 v
      PPO CommandEncoderPolicy (唯一可训练网络)
                 |
            raw latent z (256D)
                 |
       sqrt(256) * normalize(z)
                 |
       冻结 FBcprAux ONNX BFM decoder
                 |
             22DoF action
                 |
          PiPlus H0W 物理环境
                 |
   线速度/横向速度/偏航速度跟踪奖励
```

这与仓库已有的 `humanoidverse.amp_stage2` 明确不同：

- 不使用 23DoF AMP teacher；
- 不使用 AMP discriminator、专家风格 reward、动作蒸馏或 AMP 数据特征；
- 不把 motion 数据作为每步 reference tracking 目标；motion 数据只用于底层环境的 reset/motion library 初始化契约；
- PPO 的主奖励只由速度指令跟踪构成。默认 `env_reward_weight=0.0`，环境原生 reward 不参与优化。

这里的“无动力学”指奖励不依赖动力学模型或专家动力学项；策略仍在 MuJoCo/Isaac Sim 物理仿真中采样并优化。

## 2. 资产与维度契约

| 项目 | 值 |
| --- | --- |
| Robot | `PiPlus_S_12L8A0G2H0W` |
| Action DoF | 22 |
| H0W MJCF | `humanoidverse/data/robots/piplus_h0w/xml/piplus_h0w_bfm.xml` |
| MJCF state | `nq=29`, `nv=28`, `nu=22` |
| Frozen decoder | `model/piplus_h0w_bfm/decoder/bfmzero-piplus-h0w-isaac-20260629_214205/exported/FBcprAuxModel.onnx` |
| Decoder input/output | 616 / 22 |
| Latent | 256D，投影后 L2 norm 为 `sqrt(256)=16` |
| Motion data | `humanoidverse/data/piplus_h0w_lafan/piplus_h0w_lafan_10s-clipped.pkl`，869 motions |

decoder 输入严格为：

```text
state         50 = dof_pos(22), dof_vel(22), projected_gravity(3), base_ang_vel(3)
last_action   22
history_actor 288 = 4 * [actions(22), base_ang_vel(3), dof_pos(22), dof_vel(22), projected_gravity(3)]
z             256
---------------------------------
total         616
```

`model/piplus_h0w_bfm/model.safetensors` 是另一个 GCR-RL 格式的 22DoF 状态字典。训练入口只用它验证动作维度/robot contract；实际执行动作的冻结基座是上述 ONNX decoder，不能将两者当作同一 PyTorch checkpoint 混合加载。

decoder、motion pkl 和训练日志均为本地生成物，已忽略，不随 Git 提交。仅 clone 仓库和复制 PPO checkpoint 不足以播放，必须同时取得 decoder ONNX 与 motion pkl。

## 3. 代码入口

| 文件 | 责任 |
| --- | --- |
| `humanoidverse/speed_stage2.py` | PPO 训练入口、速度奖励、H0W 环境构建、DDP 同步、checkpoint 保存 |
| `humanoidverse/piplus_h0w_onnx_decoder.py` | 冻结 ONNX decoder；内存中将 batch=1 标注改为动态 batch，不改磁盘模型 |
| `humanoidverse/speed_stage2_play.py` | 本地单环境 MuJoCo GUI playback，加载本实验 PPO checkpoint + ONNX decoder |
| `tests/test_speed_stage2_play.py` | playback checkpoint 选择和 policy shape 纯单元测试 |
| `tuning-log.md` | 全仓库调参流水；本实验的精确运行记录以本文为准 |

训练时 command encoder 输入为 `command(3) + state(50) + last_action(22) + history_actor(288)`，即 363 维。网络为 `Linear(363,256) -> LayerNorm -> Tanh`，随后产生 256D 高斯 latent mean、共享的 latent log-std 和 value。采样的 raw latent 经 decoder 投影后产生动作；PPO 对 raw latent 的 log probability 进行更新。

每个 rollout step：采样指令，编码 latent，decoder 输出 22DoF action，推进物理仿真，读取 reset 前 base velocity，计算速度 reward，保存 rollout。每轮结束后使用 GAE 与 clipped PPO 更新 command encoder/value head。多卡时每个 rank 独立创建环境，trainable 参数梯度 all-reduce 平均，仅 rank 0 保存 checkpoint。

## 4. 速度奖励与默认配置

指令范围：

```text
vx: [-0.8, 0.8] m/s
vy: [-0.5, 0.5] m/s
wz: [-0.8, 0.8] rad/s
```

奖励：

```text
linear_error = ||base_lin_vel_xy - command_xy||^2
yaw_error    = (base_ang_vel_z - command_wz)^2
reward       = exp(-linear_error / 0.16) + 0.5 * exp(-yaw_error / 0.25)
```

默认 stand 概率为 `0.15`，纯转向概率为 `0.15`，每 300 步以 `0.75` 概率重采样指令，并以 `0.15` 的平滑系数过渡。motion-end/reference/contact/gravity 等配置终止均关闭；训练入口用 contact force 和 projected gravity 自己判定 crash/fall。

默认 PPO：`rollout_steps=32`、`ppo_epochs=5`、`learning_rate=3e-4`、`discount=0.98`、`gae_lambda=0.95`、`clip_ratio=0.2`、`value_coef=0.5`、`entropy_coef=0.001`。

### 4.1 奖励、GAE 与 PPO 损失（ASCII 公式版）

本节刻意使用纯文本公式，避免 Markdown/MathJax 环境不可用时公式显示为空。令实际基座速度为
`v = [vx, vy]`、实际偏航角速度为 `wz`，速度指令为 `c = [cx, cy, cwz]`。

```text
linear_error_t = (vx_t - cx_t)^2 + (vy_t - cy_t)^2
yaw_error_t    = (wz_t - cwz_t)^2

r_linear_t = exp(-linear_error_t / 0.16)
r_yaw_t    = exp(-yaw_error_t / 0.25)
r_t        = r_linear_t + 0.5 * r_yaw_t
```

因此 `r_t` 的最大值为 `1.5`，且仅在 `[vx, vy, wz] == [cx, cy, cwz]` 时达到。两个分母是误差尺度：
线速度平方误差增加 `0.16` 时，其对应项缩小到 `exp(-1)`；偏航平方误差增加 `0.25` 时同理。
当前正式命令没有传 `--env-reward-weight`，其默认值为 `0.0`，所以实际优化的即时奖励就是上式，
没有 AMP、teacher、expert motion 或环境原生奖励项。

GAE 与 return：

```text
delta_t = r_t + gamma * V(s_{t+1}) * not_done_t - V(s_t)
A_t     = delta_t + gamma * lambda * not_done_t * A_{t+1}
R_t     = A_t + V(s_t)

gamma  = 0.98
lambda = 0.95
```

对存入 rollout 的 raw latent `z_t`，PPO 使用：

```text
ratio_t = exp(log_pi_new(z_t | s_t) - log_pi_old(z_t | s_t))

L_policy = -mean(min(ratio_t * A_t,
                     clip(ratio_t, 0.8, 1.2) * A_t))

L_value  = 0.5 * mean((V(s_t) - R_t)^2)
L_total  = L_policy + 0.5 * L_value - 0.001 * mean(entropy(pi_new))
```

训练首先对 `A_t` 做零均值、单位方差归一化。`L_policy` 的 clip 把一次 update 的重要性比率限制在
`[0.8, 1.2]`；`L_value` 让 critic 预测 GAE return；熵项避免 256D latent 高斯策略过早退化。
优化器对 command encoder、latent log-std、value head 求梯度；decoder 处于 `eval()` 且
`requires_grad_(False)`，不会接收或更新梯度。源码分别见 `speed_tracking_reward()`、`compute_gae()`、
`ppo_update()`：`humanoidverse/speed_stage2.py:270`、`:364`、`:386`。

## 5. 运行前检查与命令

在仓库根目录、`HT_BFM` Conda 环境中执行：

```bash
conda activate HT_BFM
python -m humanoidverse.speed_stage2 --validate-assets
```

MuJoCo 每个 rank 只支持一个环境，可用于单环境 smoke/playback：

```bash
python -m humanoidverse.speed_stage2 \
  --simulator mujoco --device cuda:0 --num-envs 1 \
  --iterations 1 --rollout-steps 1 --ppo-epochs 1 --minibatch-size 1 \
  --disable-domain-randomization --save-every 1 \
  --work-dir logs/speed_stage2_piplus_22dof/smoke_manual
```

四卡 DDP 集成 smoke 可在当前机器使用 MuJoCo 完成。它验证四个 GPU 上的 command encoder、frozen decoder、rollout、PPO、NCCL all-reduce 和 rank-0 checkpoint，但不代表 Isaac Sim 的向量化采样吞吐：

```bash
cd /root/autodl-tmp/chenyupeng/HT_BFM
conda activate HT_BFM

TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun --standalone --nproc_per_node=4 -m humanoidverse.speed_stage2 \
  --simulator mujoco --num-envs 1 --iterations 1 --rollout-steps 1 \
  --ppo-epochs 1 --minibatch-size 1 --disable-domain-randomization \
  --save-every 1 \
  --work-dir /root/autodl-tmp/chenyupeng/HT_BFM/logs/speed_stage2_piplus_22dof/smoke_4gpu_mujoco
```

正式向量化训练使用 Isaac Sim。服务器为无头环境：配置会强制 `headless=True`。当前 H20 驱动上的 Isaac GPU PhysX/Vulkan 路径会触发 `ERROR_DEVICE_LOST`，因此已实现并验证 `--sim-device cpu`：Isaac Sim 仍创建真实向量化环境并执行物理 rollout，PPO command encoder/decoder/NCCL 仍在 `--device cuda` 上运行，动作进入仿真前转 CPU，观测/奖励回传 CUDA。为避免 Kit 启动 GPU Vulkan，运行时设置软件 Vulkan ICD `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`，并传 `--kit_args='--renderer/enabled=pxr --app/vulkan=false'`。多卡时仍保留原始 `LOCAL_RANK=0..3`，每个 rank 的 CUDA policy 和 Isaac cache 独立；不可将 `LOCAL_RANK` 重写为零。

```bash
cd /root/autodl-tmp/chenyupeng/HT_BFM
conda activate HT_BFM
mkdir -p logs/speed_stage2_piplus_22dof/preflight_4gpu

TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun --standalone --nproc_per_node=4 -m humanoidverse.speed_stage2 \
  --simulator isaacsim --num-envs 16 --iterations 5 --rollout-steps 8 \
  --ppo-epochs 1 --minibatch-size 128 --disable-domain-randomization \
  --save-every 5 \
  --work-dir /root/autodl-tmp/chenyupeng/HT_BFM/logs/speed_stage2_piplus_22dof/preflight_4gpu
```

只有预检写出 `config.json` 和 `checkpoint_5.pt` 后，才运行长训练。起步建议每卡 256 环境，总计 1024 环境，每轮 32768 samples：

```bash
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun --standalone --nproc_per_node=4 -m humanoidverse.speed_stage2 \
  --simulator isaacsim --num-envs 256 --iterations 1000000 --rollout-steps 32 \
  --ppo-epochs 5 --minibatch-size 2048 --learning-rate 3e-4 --save-every 100 --seed 4728 \
  --work-dir /root/autodl-tmp/chenyupeng/HT_BFM/logs/speed_stage2_piplus_22dof/isaacsim_4gpu_1024env_seed4728
```

当前 `onnxruntime` 仅检测到 CPU provider。decoder 已支持动态 batch，单进程 batch 32/128/256 的 CPU forward 分别约为 24/41/99 ms；Isaac Sim 与 PPO 在 GPU 上执行，decoder 的 GPU provider 是后续性能优化项。

## 6. 本地 GUI Playback

将以下内容复制到本地机器：

1. 本分支代码；
2. 一个 `checkpoint_<iteration>.pt`；
3. decoder `FBcprAuxModel.onnx`；
4. H0W motion pkl；
5. 本仓库已跟踪的 `humanoidverse/data/robots/piplus_h0w/` 资产。

在有桌面/图形会话的本地机器中运行 MuJoCo viewer。以下使用 CPU，避免 decoder/physics 的 CUDA 依赖；本地有 CUDA 时可改为 `--device cuda:0`：

```bash
conda activate HT_BFM
python -m humanoidverse.speed_stage2_play \
  --checkpoint /absolute/path/checkpoint_100.pt \
  --decoder-path /absolute/path/FBcprAuxModel.onnx \
  --expert-dataset /absolute/path/piplus_h0w_lafan_10s-clipped.pkl \
  --simulator mujoco --device cpu \
  --fixed-command 0.4 0.0 0.0 \
  --log-every-steps 100
```

关闭 MuJoCo 窗口即可退出。可用 `--max-steps 1000` 自动结束，或把 `--fixed-command` 改为任意契约范围内的 `[vx, vy, wz]`。当前 playback 是固定指令版，不含手柄输入、Isaac Sim viewer 或 MP4 导出；这些是后续功能，不应通过 AMP playback 脚本替代。

## 6.1 TensorBoard 曲线

`speed_stage2` 原生输出 JSONL，不直接创建 TensorBoard event。当前 run 已启动日志转换器和 TensorBoard 服务：

```bash
RUN_DIR=/root/autodl-tmp/chenyupeng/HT_BFM/logs/speed_stage2_piplus_22dof/full_4gpu_isaac_cpu_lvp_20260805_1505
python tools/speed_stage2_log_to_tensorboard.py "$RUN_DIR/torchrun.log" \
  --out "$RUN_DIR/tensorboard" --follow
tensorboard --logdir "$RUN_DIR/tensorboard" --host 0.0.0.0 --port 6006
```

无头服务器上从本地执行 `ssh -L 6006:127.0.0.1:6006 <server>`，然后打开 `http://127.0.0.1:6006`。曲线包括 `reward_mean`、`speed/vx_mae`、`speed/vy_mae`、`speed/yaw_rate_mae`、`termination_rate`、`approx_kl`、`value_loss`、`policy_loss` 和 `entropy`。

## 7. 当前阶段与 TODO

当前阶段：**四卡 Isaac Sim 向量化 DDP smoke 已通过；正式全量训练后台运行中**。

已完成：22DoF 资产和静态 contract 验证；ONNX decoder 616->22 前向；动态 batch 输出与原始逐条 ONNX 对齐（最大误差 `1.56e-7`）；MuJoCo 单环境一轮 PPO smoke；MuJoCo 四 rank、四 GPU、一次 rollout/一次 PPO update/NCCL 同步 smoke，并写出可加载 checkpoint；数据盘默认输出；错误的 Isaac Sim 四卡预检进程已清理。

已解决：默认 GPU PhysX/Vulkan 路径在 H20 上的 `ERROR_DEVICE_LOST`。单卡 CPU PhysX + 软件 Vulkan 完成 2-env PPO，并写出 `checkpoint_1.pt`；四卡、每卡 2 env 的 Isaac Sim 向量化 DDP smoke 完成一次 PPO update，rank0 写出 `checkpoint_1.pt`，无 `DEVICE_LOST`。当前正式运行使用四卡、每卡 16 env、32 步 rollout、5 个 PPO epoch、10000 iterations，输出目录见实验日志。退出时偶发 Kit cleanup warning 不影响 checkpoint；训练期间无 Vulkan crash。

后续 TODO：

1. 继续监控正式运行的 reward、vx/vy/yaw MAE、termination、KL、steps/s，并确认 `checkpoint_100.pt`、`checkpoint_1000.pt` 等周期性产物。
2. 定期复制 checkpoint 到本地，使用第 6 节 GUI playback 对 stand、前进、横移和纯 yaw 指令做视觉检查。
3. 根据 CPU PhysX 吞吐决定是否迁移到已修复驱动的 GPU PhysX，或安装支持 CUDA 的 ONNX Runtime；当前先保持稳定配置。
4. 在速度 tracking 可稳定收敛后，再评估 AMP 对照实验，不混入本实验主线。

### 7.1 Isaac Sim GPU PhysX/Vulkan 平台问题

当前正式 run 不会再触发 `ERROR_DEVICE_LOST`，因为它使用 CPU PhysX + llvmpipe 软件 Vulkan；但这不是
GPU PhysX 性能配置，采样吞吐会明显低于原生 GPU PhysX。PPO policy/value、NCCL all-reduce 仍在 GPU，
但 PhysX 和 ONNX decoder 都不在 GPU：本环境 `onnxruntime==1.26.0` 只有 CPU provider。

截至 2026-08-05 的主机诊断：

```text
GPU:              NVIDIA H20 x8
driver:           590.48.01
Isaac Sim package: 5.1.0.0
Isaac Sim core:    5.1.0-rc.19+release.26219.9c81211b
native Vulkan:     vulkaninfo 能枚举全部 H20，NVIDIA proprietary driver 590.48.01
RT capability:      可见 VK_KHR_acceleration_structure 与 VK_KHR_ray_tracing_pipeline
native test:       官方 Isaac Lab 空 SimulationContext 和本项目单卡场景均复现 ERROR_DEVICE_LOST
CPU fallback:      4 rank x 16 env 的 Isaac Sim rollout/PPO/NCCL 持续运行，无 DEVICE_LOST
```

这说明问题不由 PiPlus 资产、22DoF decoder、奖励、PPO 或 DDP 引起，而位于 Isaac Sim RC build、H20 driver、
Vulkan/RTX 或无头 Kit 组合。BFM-Zero 官方仓库已有同类报告：
`LeCAR-Lab/BFM-Zero#13, "got errors while training on h20"`，其环境是 H20、driver 570.158.01、
CUDA 12.8，报错为 GPU device 创建失败。该 issue 已关闭但没有维护者修复结论。

Isaac Sim 官方也记录了容器 headless Vulkan `ERROR_DEVICE_LOST`（IsaacSim#431）；维护者要求提供完整 Kit log、
命令和 `nvidia-smi` 以进一步定位。该讨论中的用户通过 `vulkan=false` 规避崩溃，但 NVIDIA 维护者明确指出
Linux 上 Vulkan 是 Isaac Sim 唯一图形后端，因此它是 workaround，不是 GPU PhysX 修复。

处理顺序（不得在当前 run 中途切换）：

1. 保持当前 CPU fallback run 完成或到达选定 checkpoint，不能热切换 `sim_device`，因为物理后端状态不兼容。
2. 在独立环境安装正式、非 RC 的 Isaac Sim build，并将 NVIDIA driver 对齐该发行版官方要求；当前最新官方要求页列出 Linux driver `595.58.03`，本机为 `590.48.01`。必须同时核验所选 Isaac Sim 5.1 build 的发行说明，不能仅因当前网页指向更高版本而盲目混装。
3. 在不运行训练时，先用官方 Isaac Lab `create_empty.py` 在单张 H20 上做原生 Vulkan smoke，再以本入口
   `--sim-device cuda:0` 做 1 env/1 iteration smoke；二者均成功、无 GPU dump 后再做 4 rank 向量化 smoke。
4. 仅在上述 smoke 通过后，启动新的 GPU PhysX run；它与当前 CPU fallback run 是不同实验，不能从当前 checkpoint
   声称等价续训。届时再安装 CUDA-enabled ONNX Runtime，使 decoder inference 也留在每个 rank 的 GPU。

不要用降低 `num_envs`、改奖励权重或修改 PPO loss 来处理该错误，它们无法修复 Vulkan device loss。

## 8. 实验日志

| 时间 (UTC) | 改动/配置 | 命令与结果 | 结论 |
| --- | --- | --- | --- |
| 2026-08-05 10:48 | 解压 FBcprAux ONNX，接入 22DoF decoder | ONNX 输入 `[1,616]`、输出 `[1,22]`；H0W MJCF `nu=22` | decoder 与 H0W 动作维度匹配 |
| 2026-08-05 10:50 | MuJoCo，1 env，1 iter，1 rollout step | `reward_mean=0.2184`，无终止，写出 checkpoint | 首次 PPO/action 链路通过 |
| 2026-08-05 11:09 | dynamic ONNX batch | batch-16 与逐条模型最大误差 `1.56e-7` | 可批量推理，磁盘 checkpoint 未改 |
| 2026-08-05 11:09 | MuJoCo，2 env | IMU tensor shape 错误 | 确认 MuJoCo wrapper 只支持 1 env，已在入口显式拒绝 |
| 2026-08-05 11:19 | MuJoCo，1 env，dynamic decoder | PPO smoke 再次成功 | 改造后单环境链路通过 |
| 2026-08-05 | Isaac Sim，2 env headless | 初始化超过 6 分钟，无训练输出，手动停止 | Isaac 多环境未验证 |
| 2026-08-05 | Isaac Sim，四卡 preflight | 首次命令 `RUN_DIR` 为空，四个 worker 没有实际输出路径且成为孤儿；已停止 | 命令无效，需用本文件的绝对路径版本重试 |
| 2026-08-05 | Isaac Sim，四卡 preflight | 修正输出路径后报告 Vulkan `ERROR_DEVICE_LOST`；无 checkpoint，已停止 | 初始版本的 worker GPU 处理不正确 |
| 2026-08-05 | Isaac Sim，四卡 GPU 隔离修复尝试 | 日志显示四个 AppLauncher 都是 `cuda:0`，说明将 `LOCAL_RANK` 重写为 0 反而使四个 Kit 争抢 GPU 0；无 checkpoint，已停止 | 保留原始 `LOCAL_RANK`，移除首四卡命令的 `CUDA_VISIBLE_DEVICES`，待后台重测 |
| 2026-08-05 12:19 UTC | Isaac Sim，单卡、1 env、1 step 受控 smoke | AppLauncher 正确使用 `cuda:0`，完成环境构建并加载 869 motions；约 26 秒后出现 `VkResult: ERROR_DEVICE_LOST`，生成 `kit_20260805_121935-0.nv-gpudmp`，无 checkpoint；`SIGTERM` 无效，已 `SIGKILL` | 单卡可复现，阻塞在 Isaac Sim/Vulkan 运行环境，禁止继续四卡训练，先修复/重启容器 |
| 2026-08-05 12:23 UTC | Isaac Sim，单卡 renderer multi-GPU 禁用 smoke | 启动器加入 `renderer/multiGpu/enabled=false`、`autoEnable=false`、`maxGpuCount=1`；Kit 仅标记 GPU 0 Active，但约 19 秒后仍报同一错误，生成 `kit_20260805_122313-0.nv-gpudmp`，无 checkpoint | 启动器已保留该防护；multi-GPU renderer 不是根因，仍须修复 Isaac Sim/Vulkan 环境 |
| 2026-08-05 12:31 UTC | 官方 Isaac Lab `create_empty.py` | 不加载本仓库资产，无法完成 `SimulationContext` 初始化；手动停止 | 平台问题独立于 PiPlus 场景 |
| 2026-08-05 12:34 UTC | 官方空场景，仅 NVIDIA ICD + XDG runtime + reset user | 同样无法完成 `SimulationContext` 初始化；手动停止 | ICD、runtime 和 Kit 用户配置不是根因 |
| 2026-08-05 12:36 UTC | MuJoCo，4 rank x 1 env、1 iter、1 rollout step、1 PPO epoch | 4 个 rank 分别在 `cuda:0..3` 载入 motion；NCCL 同步完成，`reward_mean=0.47984`，`termination_rate=0`，写出并验证 `checkpoint_1.pt` | 四卡 command encoder/decoder/PPO/DDP 链路通过；仅可作集成 smoke，非 Isaac Sim 向量化训练 |

后续记录模板：

```text
时间：
Git commit：
目的：
代码/配置改动：
完整命令：
硬件：GPU、每卡 num_envs、总环境数：
输出目录：
结果：iteration、reward、vx/vy/yaw MAE、termination、KL、steps/s、显存：
本地 playback 观察：
结论与下一步：
```

## 2026-08-05 13:00-13:02 UTC - Isaac Sim 向量化 DDP smoke 与正式训练

- 单卡 CPU PhysX + 软件 Vulkan (`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`) 完成 2-env、1 iteration PPO，写出 `checkpoint_1.pt`，无 `DEVICE_LOST`。
- 四卡 smoke 使用 `--device cuda --sim-device cpu`，每卡 2 env，四个 rank 均完成 Isaac Sim motion 加载、rollout、PPO 和 NCCL 同步；rank0 写出 `checkpoint_1.pt`。指标：`reward_mean=0.47226`，`vx/vy/yaw MAE=0.30936/0.44813/0.70866`，`termination=0`。
- 正式后台训练已启动，PID `689988`，输出目录 `logs/speed_stage2_piplus_22dof/full_4gpu_isaac_cpu_lvp_20260805_1505`。命令为四卡、每卡 16 env、`rollout_steps=32`、`ppo_epochs=5`、`iterations=10000`、`save_every=100`。已完成 `checkpoint_100.pt` 并继续运行到 iteration 102；iteration 100 的 `reward_mean=0.9888`、`vx/vy/yaw MAE=0.1479/0.1532/0.3750`、`termination=0`，无 Vulkan 错误。

## 2026-08-05 14:38 UTC - H20 ERROR_DEVICE_LOST 根因复核

- 正式 CPU fallback run 仍健康，已超过 iteration 1435，已写出 `checkpoint_1400.pt`；最近一组采样指标为
  `reward_mean=1.3252`、`vx/vy/yaw MAE=0.1127/0.0526/0.1301`、`termination_rate=0`。当前未出现新的
  Vulkan crash。
- GPU 0--3 功率约 114--122 W、显存约 1.9 GiB、瞬时 util 为 0%；这是 CPU PhysX、CPU ONNX decoder 和很小的
  PPO 网络共同导致，并非训练进程停滞。约每 100 iteration 需要 6 分 45 秒，即约 4.05 秒/iteration。
- 版本与平台检查：`isaacsim` package `5.1.0.0`，内核 build `5.1.0-rc.19`，driver `590.48.01`；`vulkaninfo`
  可枚举所有 H20 且报告 `VK_KHR_acceleration_structure`、`VK_KHR_ray_tracing_pipeline`，没有温度、ECC、
  retired-page 或硬件 slowdown 迹象。因此“无 RT 硬件能力”不是原因；此前 native GPU Vulkan 在官方空场景和项目
  场景均重现 device loss，暂定为 Isaac Sim RC build、driver 与无头 Vulkan 组合的兼容性问题，不修改当前训练代码或超参数。
- 联网核查：BFM-Zero#13 为同类 H20 GPU 创建失败报告；IsaacSim#431 为 5.1.0 container headless
  `ERROR_DEVICE_LOST`，官方要求完整 Kit log/命令/`nvidia-smi` 继续调查，未给出通用代码修复。详细结论和后续
  验证顺序已记录在第 7.1 节。

## 2026-08-05 - 合并 upstream bfm-command-amponly

- 训练在用户许可下停止；`full_4gpu_isaac_cpu_lvp_20260805_1505/checkpoint_1400.pt` 已用 CPU 成功加载，
  包含 `policy`、`optimizer`、`iteration=1400` 和 metadata，可作为后续同一训练语义的恢复点。
- 将 `upstream/bfm-command-amponly` 合并为本仓库 AMP 主线。两个历史没有共同 Git ancestor，因此采用
  `--allow-unrelated-histories -X theirs`：同名的基础训练/环境/AMP 文件以该分支为准。
- 保留本分支仅有的 22DoF 实验实现与资产：`speed_stage2.py`、`speed_stage2_play.py`、
  `piplus_h0w_onnx_decoder.py`、H0W robot assets、speed playback test、TensorBoard converter 和本文档。
  保留本分支 `AGENTS.md`，以维持本实验进度记录约定。
- 目标分支覆盖了 CPU PhysX/software Vulkan fallback 的 Isaac 启动器改动。故恢复训练前必须由原生 GPU
  Isaac Sim smoke 验证环境；不能使用原 CPU fallback 命令假设其仍受支持。该平台问题由另一 session 继续处理。

## 2026-08-05 - 本地 zhuzejian/UFO 代码与环境勘察

- `/root/autodl-tmp/zhuzejian` 确实存在，但当前账号只能看到目录元数据：权限为 `0700`，所有者为 `zhuzejian:zhuzejian`。
  当前账号不是 root，也不能通过 `runuser`/`su` 读取，因此尚未能检查其原始仓库、shell 启动脚本或 Conda 环境。
- 可读的相关副本为 `/root/autodl-tmp/chenyupeng/UFO_HT`、`/root/autodl-tmp/chenyupeng/UFO` 和
  `/root/autodl-tmp/hejunfu/projects/UFO-main`。这些副本的训练入口已经是 MuJoCo/MJLab；未发现
  `AppLauncher`、`SimulationContext`、Isaac Sim 训练入口或 `ht_urdf` Python/ROS 包，因此不能据此推断
  zhuzejian 原始 Isaac Sim 启动方式。
- 可复用的实现事实：H0W 资产被内化到 `humanoidverse/data/robots/piplus_h0w/`，MuJoCo XML 为
  `xml/piplus_h0w_bfm.xml`，训练配置使用仓库相对路径；`UFO-main` 的 H20 脚本额外固定了 CUDA 12.8
  PyTorch、显式 GPU 列表、独立缓存目录和 `NCCL_NVLS_ENABLE=0`。这解释了当前 HT_BFM 配置的差异：它
  仍把 Isaac/URDF 资产写成 `package://ht_urdf/...`，但本机环境没有可导入的 `ht_urdf` 包。
- 下一步门禁：请目录所有者对 `/root/autodl-tmp/zhuzejian` 开放至少 `r-x` 权限后，再核对其 Isaac Sim
  版本、`isaaclab.sh`/启动器参数、资产路径和环境变量；在此之前不把 UFO/MJLab 的 MuJoCo 结果当成
  Isaac Sim 兼容性证据，也不恢复 GPU PhysX 训练。

## 2026-08-05 - 本地资产路径修复与 GPU 状态复核

- 修正 `speed_stage2.py`：`--robot-config` 现在同时驱动 Hydra 的 `robot.asset` 和 `robot.motion.asset`，不再
  只用于契约校验；`--isaac-urdf` 仍可覆盖 Isaac Sim 使用的 URDF。使用仓库内置 XML 和 no-visual URDF 的
  MuJoCo 单步 smoke 成功，`reward_mean=0.01768`，说明该路径修复没有破坏 PPO 链路。
- Isaac Sim GPU no-visual smoke 已通过资产校验并加载 `869 motions`，但约 22 秒后仍生成
  `kit_20260805_152847-0.nv-gpudmp`，随后报 `VkResult: ERROR_DEVICE_LOST`。因此视觉 STL 不是根因。
- 随后的官方 Cartpole articulation probe 在设置 `CUDA_VISIBLE_DEVICES=0` 后反复报告
  `Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA H20`；执行 `nvidia-smi --gpu-reset -i 0` 返回
  `Insufficient Permissions`。需要管理员执行 GPU reset 或重启容器/节点后，才能进行下一次原生 GPU smoke。
- 当前不再启动 GPU PhysX 训练；CPU PhysX + software Vulkan 的 checkpoint `1400` 保留为可恢复实验结果。

## 2026-08-05 - GPU reset 能力确认

- 用户执行 `sudo nvidia-smi --gpu-reset -i 0,1,2,3`，四张 H20 均返回 `Not Supported`。这不是 sudo 权限问题，
  而是当前容器的 H20 passthrough/虚拟化模式不提供 GPU reset 接口。
- GPU 0-3 当前无计算进程、显存仅约 4 MiB；GPU 4-7 正由其他 UFO 训练进程使用，禁止重置或重启这些卡。
- 下一步必须从 AutoDL 控制面板重启当前容器/实例后再做 GPU smoke；若容器重启后仍复现，需要平台管理员重启
  宿主机驱动或更换 Isaac Sim/driver 组合。仅重新运行 Python 命令不会清除当前 CUDA bad state。

## 2026-08-05 - 原版 BFM-Zero 版本基线

- 原版 `LeCAR-Lab/BFM-Zero` 的 `pyproject.toml` 固定：Python `3.10.*`、`isaaclab[all,isaacsim]==2.0.2`、
  `isaacsim[all,extscache]==4.5.0`；README 只要求 Linux Isaac Sim 或 MuJoCo，没有声明 H20 专用版本。
- 当前 HT_BFM 实际环境为 Python 3.11、PyTorch 2.7.0+cu128、Isaac Sim `5.1.0-rc.19`、IsaacLab 2.3.0
  （安装包内部版本 `0.47.2`）。IsaacLab v2.3 官方兼容表允许 Isaac Sim 4.5/5.0/5.1，因此当前组合在
  版本表上可用，但不是原版 BFM-Zero 的基线组合。
- 版本切换有可能规避当前 5.1 RC + H20 Vulkan/PhysX 崩溃，但会引入 IsaacLab API、Python 和资产导入差异；
  不能直接在现有 `HT_BFM` 环境中降级覆盖。应单独建立 Python 3.10 + IsaacLab 2.0.2 + Isaac Sim 4.5
  的隔离环境，先跑官方 Cartpole，再跑 PiPlus no-visual smoke，与当前 5.1 环境做 A/B 对照。
