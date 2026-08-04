from pathlib import Path

import isaaclab.sim as sim_utils
from HT_lab.actuators.HT_motor import HTMotorCfg_4438, HTMotorCfg_5031, HTMotorCfg_5036
from HT_lab.actuators.HT_motor_cfg import (
    ARMATURE_3536,
    ARMATURE_4438,
    ARMATURE_5031,
    ARMATURE_5036,
    DAMPING_3536,
    DAMPING_4438,
    DAMPING_5031,
    DAMPING_5036,
    STIFFNESS_3536,
    STIFFNESS_4438,
    STIFFNESS_5031,
    STIFFNESS_5036,
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ASSET_DIR = Path(__file__).resolve().parents[3] / "data" / "robots" / "piplus"

"""
joint name order (from isaacsim):
[
0: l_hip_pitch_joint
1: r_hip_pitch_joint
2: waist_yaw_joint
3: l_hip_roll_joint
4: r_hip_roll_joint
5: head_yaw_joint
6: l_shoulder_pitch_joint
7: r_shoulder_pitch_joint
8: l_thigh_joint
9: r_thigh_joint
10: head_pitch_joint
11: l_shoulder_roll_joint
12: r_shoulder_roll_joint
13: l_calf_joint
14: r_calf_joint
15: l_upper_arm_joint
16: r_upper_arm_joint
17: l_ankle_pitch_joint
18: r_ankle_pitch_joint
19: l_elbow_joint
20: r_elbow_joint
21: l_ankle_roll_joint
22: r_ankle_roll_joint
]
"""


PiPlus_S_12L8A0G2H1W_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=str(
            ASSET_DIR
            / "PiPlus_S_12L8A0G2H1W_LSE_260611"
            / "urdf"
            / "PiPlus_S_12L8A0G2H1W_LSE_260611.urdf"
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.351),
        joint_pos={
            ".*_hip_pitch_joint": -0.25,
            ".*_calf_joint": 0.65,
            ".*_ankle_pitch_joint": -0.4,
            ".*_elbow_joint": 0.0,
            "l_shoulder_roll_joint": 0.0,
            "l_shoulder_pitch_joint": 0.0,
            "r_shoulder_roll_joint": 0.0,
            "r_shoulder_pitch_joint": 0.0,
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": HTMotorCfg_5036(
            joint_names_expr=[
                ".*_thigh_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_calf_joint",
            ],
            effort_limit_sim={
                ".*_thigh_joint": 20.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 20.0,
                ".*_calf_joint": 20.0,
            },
            velocity_limit_sim={
                ".*_thigh_joint": 60.0,
                ".*_hip_roll_joint": 60.0,
                ".*_hip_pitch_joint": 60.0,
                ".*_calf_joint": 60.0,
            },
            velocity_limit={
                ".*_thigh_joint": 8.0,
                ".*_hip_roll_joint": 8.0,
                ".*_hip_pitch_joint": 8.0,
                ".*_calf_joint": 8.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_5036,
                ".*_hip_roll_joint": STIFFNESS_5036,
                ".*_thigh_joint": STIFFNESS_5036,
                ".*_calf_joint": STIFFNESS_5036,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_5036,
                ".*_hip_roll_joint": DAMPING_5036,
                ".*_thigh_joint": DAMPING_5036,
                ".*_calf_joint": DAMPING_5036,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_5036,
                ".*_hip_roll_joint": ARMATURE_5036,
                ".*_thigh_joint": ARMATURE_5036,
                ".*_calf_joint": ARMATURE_5036,
            },
            min_delay=0,
            max_delay=4,
        ),
        "feet": HTMotorCfg_5036(
            effort_limit_sim=20.0,
            velocity_limit_sim=60.0,
            velocity_limit=8.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=STIFFNESS_5036,
            damping=DAMPING_5036,
            armature=ARMATURE_5036,
            min_delay=0,
            max_delay=4,
        ),
        "waist_yaw": HTMotorCfg_5031(
            effort_limit_sim=20,
            velocity_limit_sim=60.0,
            velocity_limit=8.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_5031,
            damping=DAMPING_5031,
            armature=ARMATURE_5031,
            min_delay=0,
            max_delay=4,
        ),
        "head": ImplicitActuatorCfg(
            effort_limit_sim=3.0,
            velocity_limit_sim=60.0,
            velocity_limit=60.0,
            joint_names_expr=[".*head_yaw_joint", ".*head_pitch_joint"],
            stiffness=STIFFNESS_3536,
            damping=DAMPING_3536,
            armature=ARMATURE_3536,
        ),
        "arms": HTMotorCfg_4438(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_upper_arm_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 20.0,
                ".*_shoulder_roll_joint": 20.0,
                ".*_upper_arm_joint": 20.0,
                ".*_elbow_joint": 20.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 60.0,
                ".*_shoulder_roll_joint": 60.0,
                ".*_upper_arm_joint": 60.0,
                ".*_elbow_joint": 60.0,
            },
            velocity_limit={
                ".*_shoulder_pitch_joint": 20.0,
                ".*_shoulder_roll_joint": 20.0,
                ".*_upper_arm_joint": 20.0,
                ".*_elbow_joint": 20.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_4438,
                ".*_shoulder_roll_joint": STIFFNESS_4438,
                ".*_upper_arm_joint": STIFFNESS_4438,
                ".*_elbow_joint": STIFFNESS_4438,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_4438,
                ".*_shoulder_roll_joint": DAMPING_4438,
                ".*_upper_arm_joint": DAMPING_4438,
                ".*_elbow_joint": DAMPING_4438,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_4438,
                ".*_shoulder_roll_joint": ARMATURE_4438,
                ".*_upper_arm_joint": ARMATURE_4438,
                ".*_elbow_joint": ARMATURE_4438,
            },
            min_delay=0,
            max_delay=4,
        ),
    },
)

PiPlus_S_12L8A0G2H1W_ACTION_SCALE = {}
for a in PiPlus_S_12L8A0G2H1W_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            PiPlus_S_12L8A0G2H1W_ACTION_SCALE[n] = 0.25 * e[n] / s[n]


# Symmetric augmentation joint mapping for PI Plus 80
# Maps left-right joint pairs for data augmentation
PiPlus_S_12L8A0G2H1W_symmetric_augmentation_joint_mapping = [
    1,  # 0: l_hip_pitch_joint -> r_hip_pitch_joint
    0,  # 1: r_hip_pitch_joint -> l_hip_pitch_joint
    2,  # 2: waist_yaw_joint (stays same)
    4,  # 3: l_hip_roll_joint -> r_hip_roll_joint
    3,  # 4: r_hip_roll_joint -> l_hip_roll_joint
    5,  # 5: head_yaw_joint (stays same)
    7,  # 6: l_shoulder_pitch_joint -> r_shoulder_pitch_joint
    6,  # 7: r_shoulder_pitch_joint -> l_shoulder_pitch_joint
    9,  # 8: l_thigh_joint -> r_thigh_joint
    8,  # 9: r_thigh_joint -> l_thigh_joint
    10,  # 10: head_pitch_joint (stays same)
    12,  # 11: l_shoulder_roll_joint -> r_shoulder_roll_joint
    11,  # 12: r_shoulder_roll_joint -> l_shoulder_roll_joint
    14,  # 13: l_calf_joint -> r_calf_joint
    13,  # 14: r_calf_joint -> l_calf_joint
    16,  # 15: l_upper_arm_joint -> r_upper_arm_joint
    15,  # 16: r_upper_arm_joint -> l_upper_arm_joint
    18,  # 17: l_ankle_pitch_joint -> r_ankle_pitch_joint
    17,  # 18: r_ankle_pitch_joint -> l_ankle_pitch_joint
    20,  # 19: l_elbow_joint -> r_elbow_joint
    19,  # 20: r_elbow_joint -> l_elbow_joint
    22,  # 21: l_ankle_roll_joint -> r_ankle_roll_joint
    21,  # 22: r_ankle_roll_joint -> l_ankle_roll_joint
]

# Joint reverse buffer for symmetric augmentation
# 1 means keep sign, -1 means flip sign when mirroring
PiPlus_S_12L8A0G2H1W_symmetric_augmentation_joint_reverse_buf = [
    1,  # 0: l_hip_pitch_joint (keep)
    1,  # 1: r_hip_pitch_joint (keep)
    -1,  # 2: waist_yaw_joint (flip)
    -1,  # 3: l_hip_roll_joint (flip)
    -1,  # 4: r_hip_roll_joint (flip)
    -1,  # 5: head_yaw_joint (flip)
    1,  # 6: l_shoulder_pitch_joint (keep)
    1,  # 7: r_shoulder_pitch_joint (keep)
    -1,  # 8: l_thigh_joint (flip)
    -1,  # 9: r_thigh_joint (flip)
    1,  # 10: head_pitch_joint (keep)
    -1,  # 11: l_shoulder_roll_joint (flip)
    -1,  # 12: r_shoulder_roll_joint (flip)
    1,  # 13: l_calf_joint (keep)
    1,  # 14: r_calf_joint (keep)
    -1,  # 15: l_upper_arm_joint (flip)
    -1,  # 16: r_upper_arm_joint (flip)
    1,  # 17: l_ankle_pitch_joint (keep)
    1,  # 18: r_ankle_pitch_joint (keep)
    1,  # 19: l_elbow_joint (keep)
    1,  # 20: r_elbow_joint (keep)
    -1,  # 21: l_ankle_roll_joint (flip)
    -1,  # 22: r_ankle_roll_joint (flip)
]

PiPlus_S_12L8A0G2H1W_LINKS = [  # Order not guaranteed.
    "base_link",
    "waist_yaw_link",
    "head_yaw_link",
    "head_pitch_link",
    "l_hip_pitch_link",
    "l_hip_roll_link",
    "l_thigh_link",
    "l_calf_link",
    "l_ankle_pitch_link",
    "l_ankle_roll_link",
    "l_shoulder_pitch_link",
    "l_shoulder_roll_link",
    "l_upper_arm_link",
    "l_elbow_link",
    "r_hip_pitch_link",
    "r_hip_roll_link",
    "r_thigh_link",
    "r_calf_link",
    "r_ankle_pitch_link",
    "r_ankle_roll_link",
    "r_shoulder_pitch_link",
    "r_shoulder_roll_link",
    "r_upper_arm_link",
    "r_elbow_link",
]
