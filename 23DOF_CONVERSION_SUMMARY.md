# 23-DOF vs 29-DOF Data Conversion Summary

## Overview
This document explains the relationship between 23-DOF and 29-DOF configurations and the data conversion process.

## Joint Mapping

### Removed Joints (6 joints)
The following joints are removed when converting from 29-DOF to 23-DOF:

1. `waist_roll_joint` - Waist roll
2. `waist_pitch_joint` - Waist pitch  
3. `left_wrist_pitch_joint` - Left wrist pitch
4. `left_wrist_yaw_joint` - Left wrist yaw
5. `right_wrist_pitch_joint` - Right wrist pitch
6. `right_wrist_yaw_joint` - Right wrist yaw

### Preserved Joints (23 joints)
The remaining joints are organized as follows:

- **Lower body (12 joints)**: Both legs with 6 DOF each
  - Left: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
  - Right: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll

- **Waist (1 joint)**: Only waist_yaw is preserved
  - waist_yaw_joint

- **Upper body (10 joints)**: Shoulders, elbows, and wrist rolls
  - Left: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
  - Right: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll

## Data Structure Comparison

### Original 29-DOF Data
- **dof**: (frames, 29) - Joint angles
- **pose_aa**: (frames, 30, 3) - Axis-angle representation (30 includes extra wrist DOF)

### Converted 23-DOF Data  
- **dof**: (frames, 23) - Joint angles (removed 6 joints)
- **pose_aa**: (frames, 24, 3) - Axis-angle representation (reduced from 30 to 24)

### Unchanged Fields
- `root_trans_offset`: (frames, 3) - Root position
- `root_rot`: (frames, 4) - Root rotation quaternion
- `smpl_joints`: (frames, 24, 3) - SMPL joint positions
- `fps`: 30 - Frame rate
- `motion_name`: String identifier (for clipped data)

## Generated Files

### Conversion Results
Successfully generated two 23-DOF pickle files:

1. **lafan_23dof.pkl** (175.7 MB)
   - 40 complete motion sequences
   - Average length: ~5047 frames (~168 seconds)
   - Suitable for evaluation and full-sequence training

2. **lafan_23dof_10s-clipped.pkl** (171.9 MB)
   - 862 motion clips (10 seconds each)
   - Fixed length: 300 frames per clip
   - Suitable for standard training with uniform sequence lengths

### Index Mapping
The conversion removes the following indices from the original 29-DOF data:
- Index 13: waist_roll_joint
- Index 14: waist_pitch_joint  
- Index 20: left_wrist_pitch_joint
- Index 21: left_wrist_yaw_joint
- Index 27: right_wrist_pitch_joint
- Index 28: right_wrist_yaw_joint

## Usage

### Training Configuration
To use the 23-DOF data, update your training configuration:

```python
# In train_low.py or train_low_23dof.py
env=HumanoidVerseIsaacConfig(
    lafan_tail_path='humanoidverse/data/lafan_23dof_10s-clipped.pkl',  # Use 23-DOF data
    hydra_overrides=[
        'robot=g1/g1_23dof_hard_waist',  # Use 23-DOF robot config
        # ... other overrides
    ]
)
```

### Benefits of 23-DOF
1. **Reduced Complexity**: 20% fewer DOFs (29 vs 23)
2. **Lower Memory Usage**: Smaller state and action spaces
3. **Faster Training**: Reduced computational requirements
4. **Simplified Control**: Focus on core locomotion and basic arm movements

### Trade-offs
1. **Limited Wrist Control**: No wrist pitch/yaw, only roll
2. **Reduced Waist Flexibility**: No waist roll/pitch, only yaw
3. **Simpler Upper Body**: Less expressive arm movements

## Verification

The converted data has been verified to:
- Maintain correct dimensional shapes (23 DOFs, 24 axis-angles)
- Preserve data integrity (no NaN or infinite values)
- Load successfully with the motion library
- Work with 23-DOF robot configurations

## Files Created

1. `convert_pkl_23dof.py` - Conversion script
2. `humanoidverse/data/lafan_23dof.pkl` - 23-DOF full sequences
3. `humanoidverse/data/lafan_23dof_10s-clipped.pkl` - 23-DOF 10s clips
4. `test_23dof_data.py` - Verification script

The conversion maintains full compatibility with the existing training pipeline while providing the benefits of reduced DOF complexity.
