#!/usr/bin/env python3
"""
将29-DOF的LAFAN pickle数据转换为23-DOF格式
基于dofs_convert.py中的关节映射规则
"""

import joblib
import numpy as np
import os
from pathlib import Path

# === 要删除的关节索引映射 ===
# 基于29-DOF的dof_names顺序
JOINTS_TO_REMOVE = {
    'waist_roll_joint',      # 腰部滚转
    'waist_pitch_joint',      # 腰部俯仰  
    'left_wrist_pitch_joint',  # 左手腕俯仰
    'left_wrist_yaw_joint',   # 左手腕偏航
    'right_wrist_pitch_joint', # 右手腕俯仰
    'right_wrist_yaw_joint'   # 右手腕偏航
}

# 29-DOF关节名称顺序 (从g1_29dof_hard_waist.yaml获取)
DOF_NAMES_29 = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
    'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 
    'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
]

# 23-DOF关节名称顺序 (从g1_23dof_hard_waist.yaml获取)
DOF_NAMES_23 = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint'
]

def get_dof_mapping():
    """获取29-DOF到23-DOF的索引映射"""
    remove_indices = {i for i, name in enumerate(DOF_NAMES_29) if name in JOINTS_TO_REMOVE}
    
    old_to_new = {}
    new_idx = 0
    for old_idx in range(len(DOF_NAMES_29)):
        if old_idx not in remove_indices:
            old_to_new[old_idx] = new_idx
            new_idx += 1
    
    return old_to_new, remove_indices

def convert_dof_data(dof_data, mapping):
    """转换DOF数据从29-DOF到23-DOF"""
    if dof_data.ndim == 2:  # (frames, 29)
        new_data = np.zeros((dof_data.shape[0], 23), dtype=dof_data.dtype)
        for old_idx, new_idx in mapping.items():
            new_data[:, new_idx] = dof_data[:, old_idx]
        return new_data
    elif dof_data.ndim == 1:  # (29,)
        new_data = np.zeros(23, dtype=dof_data.dtype)
        for old_idx, new_idx in mapping.items():
            new_data[new_idx] = dof_data[old_idx]
        return new_data
    else:
        raise ValueError(f"Unsupported DOF data shape: {dof_data.shape}")

def convert_pose_aa_data(pose_aa_data, mapping):
    """转换pose_aa数据从29-DOF到23-DOF"""
    # pose_aa shape: (frames, 30, 3) - 30个轴角，最后一个是手腕的额外自由度
    new_data = np.zeros((pose_aa_data.shape[0], 24, 3), dtype=pose_aa_data.dtype)
    
    # 前29个轴角对应29个DOF
    for old_idx, new_idx in mapping.items():
        if old_idx < 29:  # 确保不超出范围
            new_data[:, new_idx, :] = pose_aa_data[:, old_idx, :]
    
    # 第30个轴角(索引29)通常用于手腕，这里我们保留它作为第24个轴角
    # 或者可以设为零，取决于具体需求
    new_data[:, 23, :] = pose_aa_data[:, 29, :]  # 将最后一个轴角映射到第24个位置
    
    return new_data

def convert_motion_data(motion_data, mapping):
    """转换单个动作数据"""
    converted = {}
    
    for key, value in motion_data.items():
        if key == 'dof':
            converted[key] = convert_dof_data(value, mapping)
        elif key == 'pose_aa':
            converted[key] = convert_pose_aa_data(value, mapping)
        else:
            # 其他字段保持不变
            converted[key] = value
    
    return converted

def convert_pkl_file(input_path, output_path):
    """转换整个pkl文件"""
    print(f"Converting {input_path} to {output_path}")
    
    # 加载原始数据
    data = joblib.load(input_path)
    print(f"Loaded {len(data)} motions")
    
    # 获取映射
    mapping, remove_indices = get_dof_mapping()
    print(f"Removing indices: {sorted(remove_indices)}")
    print(f"Mapping {len(mapping)} DOFs from 29 to 23")
    
    # 转换每个动作
    converted_data = {}
    for motion_name, motion_data in data.items():
        converted_data[motion_name] = convert_motion_data(motion_data, mapping)
    
    # 保存转换后的数据
    joblib.dump(converted_data, output_path)
    print(f"Saved {len(converted_data)} motions to {output_path}")
    
    # 验证转换结果
    verify_conversion(converted_data)

def verify_conversion(data):
    """验证转换结果"""
    print("\n=== Verification ===")
    first_motion = list(data.values())[0]
    
    print(f"DOF shape: {first_motion['dof'].shape}")
    print(f"pose_aa shape: {first_motion['pose_aa'].shape}")
    
    # 检查DOF数据
    dof_sample = first_motion['dof'][0]
    print(f"DOF sample (first frame): {dof_sample}")
    
    # 检查是否有NaN或异常值
    if np.any(np.isnan(dof_sample)):
        print("⚠️  Warning: NaN values found in DOF data")
    
    if np.any(np.isinf(dof_sample)):
        print("⚠️  Warning: Infinite values found in DOF data")
    
    print("✅ Conversion completed successfully")

def main():
    """主函数"""
    base_dir = Path("humanoidverse/data")
    
    # 转换两个文件
    files_to_convert = [
        ("lafan_29dof.pkl", "lafan_23dof.pkl"),
        ("lafan_29dof_10s-clipped.pkl", "lafan_23dof_10s-clipped.pkl")
    ]
    
    for input_file, output_file in files_to_convert:
        input_path = base_dir / input_file
        output_path = base_dir / output_file
        
        if input_path.exists():
            convert_pkl_file(input_path, output_path)
        else:
            print(f"⚠️  Input file not found: {input_path}")
    
    print("\n=== Summary ===")
    print("Conversion completed!")
    print("Generated files:")
    for _, output_file in files_to_convert:
        output_path = base_dir / output_file
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  - {output_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
