## 关节配置

### 29-DOF → 23-DOF 转换
**移除的关节 (6个):**
- `waist_roll_joint` - 腰部滚转
- `waist_pitch_joint` - 腰部俯仰  
- `left_wrist_pitch_joint` - 左手腕俯仰
- `left_wrist_yaw_joint` - 左手腕偏航
- `right_wrist_pitch_joint` - 右手腕俯仰
- `right_wrist_yaw_joint` - 右手腕偏航

**保留的关节 (23个):**
- 下肢关节 (12个): 双腿各6个自由度
- 腰部关节 (1个): `waist_yaw_joint` 仅保留偏航
- 上肢关节 (10个): 双肩、双肘、双手腕roll

## 文件说明

### 机器人模型
- `g1_23dof.xml` - MuJoCo模型文件
- `g1_23dof.urdf` - URDF模型文件

### 训练脚本
- `train_low_23dof.py` - 23-DOF训练配置

