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

## 5. 运行前检查与命令

在仓库根目录、`HT_BFM` Conda 环境中执行：

```bash
conda activate HT_BFM
python -m humanoidverse.speed_stage2 --validate-assets
```

MuJoCo 只可用于单环境 smoke/playback：

```bash
python -m humanoidverse.speed_stage2 \
  --simulator mujoco --device cuda:0 --num-envs 1 \
  --iterations 1 --rollout-steps 1 --ppo-epochs 1 --minibatch-size 1 \
  --disable-domain-randomization --save-every 1 \
  --work-dir logs/speed_stage2_piplus_22dof/smoke_manual
```

正式向量化训练必须使用 Isaac Sim。服务器为无头环境：配置会强制 `headless=True`；多卡时 IsaacLab 会用 worker-local GPU 模式。先用固定绝对输出路径做四卡预检，避免未定义 shell 变量导致空 `--work-dir`：

```bash
cd /root/autodl-tmp/chenyupeng/HT_BFM
conda activate HT_BFM
mkdir -p logs/speed_stage2_piplus_22dof/preflight_4gpu

CUDA_VISIBLE_DEVICES=0,1,2,3 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun --standalone --nproc_per_node=4 -m humanoidverse.speed_stage2 \
  --simulator isaacsim --num-envs 16 --iterations 5 --rollout-steps 8 \
  --ppo-epochs 1 --minibatch-size 128 --disable-domain-randomization \
  --save-every 5 \
  --work-dir /root/autodl-tmp/chenyupeng/HT_BFM/logs/speed_stage2_piplus_22dof/preflight_4gpu
```

只有预检写出 `config.json` 和 `checkpoint_5.pt` 后，才运行长训练。起步建议每卡 256 环境，总计 1024 环境，每轮 32768 samples：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
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

## 7. 当前阶段与 TODO

当前阶段：**冒烟测试中**。

已完成：22DoF 资产和静态 contract 验证；ONNX decoder 616->22 前向；动态 batch 输出与原始逐条 ONNX 对齐（最大误差 `1.56e-7`）；MuJoCo 单环境一轮 PPO smoke；数据盘默认输出；错误的四卡预检进程已清理。

尚未完成：headless Isaac Sim 多环境 smoke 与正确四卡 DDP preflight。此前一次本机 Isaac Sim 单进程初始化超过六分钟未完成；随后四卡预检命令因空 `RUN_DIR` 形成孤儿 worker，未进入训练且已强制停止。两者均不能视为训练成功。

后续 TODO：

1. 用第 5 节的绝对路径命令完成四卡 `checkpoint_5.pt` preflight，记录每个 rank 的 Isaac 初始化、吞吐、NCCL 和 GPU 内存。
2. 若 Isaac 初始化持续卡住，优先抓取每个 rank 的 stdout/stderr、确认 IsaacLab Kit 首次缓存/驱动/`CUDA_VISIBLE_DEVICES` 行为，再修复启动路径；不要绕过为 MuJoCo 多环境。
3. 预检通过后运行短窗口（例如 100-500 iterations），观察 reward、vx/vy/yaw MAE、termination rate、KL 和实际 steps/s。
4. 定期复制 checkpoint 到本地，使用第 6 节 GUI playback 对 stand、前进、横移和纯 yaw 指令做视觉检查。
5. 根据吞吐决定是否安装支持 CUDA 的 ONNX Runtime；在此之前保持动态 batch CPU decoder 并记录其成本。
6. 在速度 tracking 可稳定收敛后，再评估是否需要把 AMP 作为对照实验，而不是混入本实验主线。

## 8. 实验日志

| 时间 (UTC) | 改动/配置 | 命令与结果 | 结论 |
| --- | --- | --- | --- |
| 2026-08-05 10:48 | 解压 FBcprAux ONNX，接入 22DoF decoder | ONNX 输入 `[1,616]`、输出 `[1,22]`；H0W MJCF `nu=22` | decoder 与 H0W 动作维度匹配 |
| 2026-08-05 10:50 | MuJoCo，1 env，1 iter，1 rollout step | `reward_mean=0.2184`，无终止，写出 checkpoint | 首次 PPO/action 链路通过 |
| 2026-08-05 11:09 | dynamic ONNX batch | batch-16 与逐条模型最大误差 `1.56e-7` | 可批量推理，磁盘 checkpoint 未改 |
| 2026-08-05 11:09 | MuJoCo，2 env | IMU tensor shape 错误 | 确认 MuJoCo wrapper 只支持 1 env，已在入口显式拒绝 |
| 2026-08-05 11:19 | MuJoCo，1 env，dynamic decoder | PPO smoke 再次成功 | 改造后单环境链路通过 |
| 2026-08-05 | Isaac Sim，2 env headless | 初始化超过 6 分钟，无训练输出，手动停止 | Isaac 多环境未验证 |
| 2026-08-05 | Isaac Sim，四卡 preflight | `RUN_DIR` 为空，四个 worker 没有实际输出路径且成为孤儿；已停止 | 命令无效，需用本文件的绝对路径版本重试 |

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
