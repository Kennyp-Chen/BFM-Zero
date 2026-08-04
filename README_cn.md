

## Requirements

- Python 3.11
- CUDA-capable GPU
- Isaac Sim (Linux) or MuJoCo for simulation

## Installation

### 1. Clone and fetch large files (Git LFS)

This repo uses [Git LFS](https://git-lfs.github.com/) for motion data and model. After cloning, install LFS and pull the large files:

```bash
git clone https://github.com/LeCAR-Lab/BFM-Zero.git
cd BFM-Zero
git lfs install
git lfs pull
```
Note: If the repository exceeded its LFS budget,  you can access the data here: https://huggingface.co/LeCAR-Lab/BFM-Zero/tree/main/data

### 2. Create the Python environment

You can install the project with either Conda or uv. Conda users do not need to
run `uv`.

#### Option A: Conda

From this directory (BFM-Zero):

```bash
(暂时的conda环境创建方法，后续再规范化)
conda create -n HT_BFM --clone HT_lab的环境

  conda activate HT_BFM
  cd /data/laihuiying/BFM-Zero
  # for 5090/H20
  python -m pip install \
    "easydict>=1.13" \
    "exca==0.4.5" \
    "humenv @ git+https://github.com/facebookresearch/humenv.git" \
    "loguru>=0.7.3" \
    "mediapy>=1.2.3" \
    "ml-collections>=1.1.0" \
    "mujoco==3.8.1" \
    "notebook>=7.4.2" \
    "numpy-stl>=3.2.0" \
    "onnxruntime==1.26.0" \
    "open3d>=0.19.0" \
    "pot>=0.9.5" \
    "tensordict>=0.8.3" \
    "termcolor>=3.0.1" \
    "tyro>=0.9.18"\
    "wandb"\
  # for 4090
  python -m pip install \
    "numpy==1.26.0" \
    "packaging==23.0" \
    "easydict>=1.13" \
    "exca==0.4.5" \
    "humenv @ git+https://github.com/facebookresearch/humenv.git" \
    "loguru>=0.7.3" \
    "mediapy>=1.2.3" \
    "ml-collections>=1.1.0" \
    "mujoco==3.8.1" \
    "notebook>=7.4.2" \
    "numpy-stl>=3.2.0" \
    "onnxruntime==1.26.0" \
    "open3d>=0.19.0" \
    "pot>=0.9.5" \
    "tensordict>=0.8.3" \
    "termcolor>=3.0.1" \
    "tyro>=0.9.18" \
    "wandb"
 conda install -c conda-forge libglu mesalib


#### Option B: uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or with pip: `pip install uv`

From this directory (BFM-Zero):

```bash
uv sync
```

## Data数据处理

- **Motion data**: Included via Git LFS in `humanoidverse/data/` after `git lfs pull`. `lafan_29dof.pkl` is for evaluation; `lafan_29dof_10s-clipped.pkl` is for training.
- If you are unsure about the data format, please check the discussion in
  [Issue #12](https://github.com/LeCAR-Lab/BFM-Zero/issues/12).
gmr生成的数据转换

```bash
#g1
python data_process/convert_gmr_lafan.py --robot g1 --output-dir ~/HT_BFM/humanoidverse/data/g1_0618 --overwrite
# pi_plus
python data_process/convert_gmr_lafan.py --robot piplus_lse --output-dir ~/HT_BFM/humanoidverse/data/pi_LSE_lafan_dataset_20260617 --overwrite

``` 
### 数据可视化
```bash
#mujoco
 python -m humanoidverse.visualize_motion \
    --robot piplus_lse \
    --data-path humanoidverse/data/pi_LSE_lafan_dataset_20260617/piplus_lse_lafan_10s-clipped.pkl \
    --max-frames 300 \
    --output humanoidverse/data/piplus_lse/pi_LSE_lafan_dataset_20260617.mp4
# isaacsim
python -m humanoidverse.visualize_motion_isaacsim \
    --data-path humanoidverse/data/pi_LSE_lafan_dataset_20260617/piplus_lse_lafan.pkl \
    --motion 0 \
    --max-frames 1600\
    --robot piplus_lse

``` 


## Training
```bash

CUDA_VISIBLE_DEVICES=2 python -m humanoidverse.train --robot PiPlus_S_12L8A0G2H1W_LSE

``` 
## Play
```bash
# isaaclab 使用model dir下的yaml文件config
python -m humanoidverse.tracking_inference \
    --model_folder ~/HT_BFM/results/bfmzero-piplus-lse-isaac-20260617_224007 \
    --data_path ~/BFM-Zero/humanoidverse/data/pi_LSE_lafan_dataset_20260617/piplus_lse_lafan.pkl \
    --no-headless \
    --motion-list 0 --robot PiPlus_S_12L8A0G2H1W_LSE
# mujoco
python -m humanoidverse.tracking_inference \
    --model_folder ~/HT_BFM/results/bfmzero-piplus-lse-isaac-20260617_224007 \
    --data_path ~/BFM-Zero/humanoidverse/data/pi_LSE_lafan_dataset_20260617/piplus_lse_lafan.pkl \
    --simulator mujoco \
    --no-headless \
    --motion-list 0 --robot PiPlus_S_12L8A0G2H1W_LSE
``` 

