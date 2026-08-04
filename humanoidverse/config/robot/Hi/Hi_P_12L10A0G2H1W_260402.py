import isaaclab.sim as sim_utils
from HT_lab.actuators.HT_motor import (
    HTMotorCfg_4438,
    HTMotorCfg_5036,
    HTMotorCfg_6036,
)
from HT_lab.actuators.HT_motor_cfg import (
    ARMATURE_3536,
    ARMATURE_4438,
    ARMATURE_5036,
    ARMATURE_6036,
    DAMPING_3536,
    DAMPING_4438,
    DAMPING_5036,
    DAMPING_6036,
    STIFFNESS_3536,
    STIFFNESS_4438,
    STIFFNESS_5036,
    STIFFNESS_6036,
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from humanoidverse.utils.asset_paths import resolve_asset_path


HI_260402_URDF_PATH = resolve_asset_path(
    "package://ht_urdf/Hi_P_12L10A0G2H1W_260402",
    "urdf/Hi_P_12L10A0G2H1W_Simplify_260402.urdf",
)


HI_260402_JOINT_NAMES = [
    "waist_yaw_joint",
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_upper_arm_joint",
    "r_elbow_joint",
    "r_wrist_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_upper_arm_joint",
    "l_elbow_joint",
    "l_wrist_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_thigh_joint",
    "r_calf_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_thigh_joint",
    "l_calf_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
]

HI_260402_DEPLOY_JOINT_NAMES = [
    "r_hip_pitch_joint",
    "l_hip_pitch_joint",
    "r_hip_roll_joint",
    "l_hip_roll_joint",
    "r_thigh_joint",
    "l_thigh_joint",
    "r_calf_joint",
    "l_calf_joint",
    "r_ankle_pitch_joint",
    "l_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "l_ankle_roll_joint",
    "waist_yaw_joint",
    "r_shoulder_pitch_joint",
    "l_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "l_shoulder_roll_joint",
    "r_upper_arm_joint",
    "l_upper_arm_joint",
    "r_elbow_joint",
    "l_elbow_joint",
    "r_wrist_joint",
    "l_wrist_joint",
]

HI_260402_INIT_JOINT_POS = {joint_name: 0.0 for joint_name in HI_260402_JOINT_NAMES}
HI_260402_INIT_JOINT_POS.update(
    {
        "r_hip_pitch_joint": -0.25,
        "l_hip_pitch_joint": -0.25,
        "r_calf_joint": 0.65,
        "l_calf_joint": 0.65,
        "r_ankle_pitch_joint": -0.40,
        "l_ankle_pitch_joint": -0.40,
    }
)

Hi_P_12L10A0G2H1W_260402_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=str(HI_260402_URDF_PATH),
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
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4315),
        joint_pos=HI_260402_INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim=2.0,
            velocity_limit_sim=300.0,
            stiffness=STIFFNESS_3536,
            damping=DAMPING_3536,
            armature=ARMATURE_3536,
        ),
        "arms": HTMotorCfg_5036(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_upper_arm_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 21.0,
                ".*_shoulder_roll_joint": 21.0,
                ".*_upper_arm_joint": 21.0,
                ".*_elbow_joint": 21.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 300.0,
                ".*_shoulder_roll_joint": 300.0,
                ".*_upper_arm_joint": 300.0,
                ".*_elbow_joint": 300.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5036,
                ".*_shoulder_roll_joint": STIFFNESS_5036,
                ".*_upper_arm_joint": STIFFNESS_5036,
                ".*_elbow_joint": STIFFNESS_5036,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5036,
                ".*_shoulder_roll_joint": DAMPING_5036,
                ".*_upper_arm_joint": DAMPING_5036,
                ".*_elbow_joint": DAMPING_5036,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5036,
                ".*_shoulder_roll_joint": ARMATURE_5036,
                ".*_upper_arm_joint": ARMATURE_5036,
                ".*_elbow_joint": ARMATURE_5036,
            },
            min_delay=0,
            max_delay=4,
        ),
        "wrists": HTMotorCfg_4438(
            joint_names_expr=[".*_wrist_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=300.0,
            stiffness=STIFFNESS_4438,
            damping=DAMPING_4438,
            armature=ARMATURE_4438,
            min_delay=0,
            max_delay=4,
        ),
        "waist": HTMotorCfg_6036(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=21.0,
            velocity_limit_sim=300.0,
            stiffness=STIFFNESS_6036,
            damping=DAMPING_6036,
            armature=ARMATURE_6036,
            min_delay=0,
            max_delay=4,
        ),
        "legs": HTMotorCfg_6036(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_thigh_joint",
                ".*_calf_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 36.0,
                ".*_hip_roll_joint": 36.0,
                ".*_thigh_joint": 36.0,
                ".*_calf_joint": 36.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 300.0,
                ".*_hip_roll_joint": 300.0,
                ".*_thigh_joint": 300.0,
                ".*_calf_joint": 300.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_6036,
                ".*_hip_roll_joint": STIFFNESS_6036,
                ".*_thigh_joint": STIFFNESS_6036,
                ".*_calf_joint": STIFFNESS_6036,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_6036,
                ".*_hip_roll_joint": DAMPING_6036,
                ".*_thigh_joint": DAMPING_6036,
                ".*_calf_joint": DAMPING_6036,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_6036,
                ".*_hip_roll_joint": ARMATURE_6036,
                ".*_thigh_joint": ARMATURE_6036,
                ".*_calf_joint": ARMATURE_6036,
            },
            min_delay=0,
            max_delay=4,
        ),
        "feet": HTMotorCfg_5036(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=36.0,
            velocity_limit_sim=300.0,
            stiffness={
                ".*_ankle_pitch_joint": STIFFNESS_5036,
                ".*_ankle_roll_joint": STIFFNESS_5036,
            },
            damping={
                ".*_ankle_pitch_joint": DAMPING_5036,
                ".*_ankle_roll_joint": DAMPING_5036,
            },
            armature=ARMATURE_5036,
            min_delay=0,
            max_delay=4,
        ),
    },
)

Hi_P_12L10A0G2H1W_260402_ACTION_SCALE = {}
for actuator_cfg in Hi_P_12L10A0G2H1W_260402_CFG.actuators.values():
    effort = actuator_cfg.effort_limit_sim
    stiffness = actuator_cfg.stiffness
    joint_names = actuator_cfg.joint_names_expr
    if not isinstance(effort, dict):
        effort = {joint_name: effort for joint_name in joint_names}
    if not isinstance(stiffness, dict):
        stiffness = {joint_name: stiffness for joint_name in joint_names}
    for joint_name in joint_names:
        if joint_name in effort and joint_name in stiffness and stiffness[joint_name]:
            Hi_P_12L10A0G2H1W_260402_ACTION_SCALE[joint_name] = 0.25 * effort[joint_name] / stiffness[joint_name]

Hi_P_12L10A0G2H1W_260402_symmetric_augmentation_joint_mapping = [
    0,
    6,
    7,
    8,
    9,
    10,
    1,
    2,
    3,
    4,
    5,
    11,
    12,
    19,
    20,
    21,
    22,
    23,
    24,
    13,
    14,
    15,
    16,
    17,
    18,
]

Hi_P_12L10A0G2H1W_260402_symmetric_augmentation_joint_reverse_buf = [
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
]

Hi_P_12L10A0G2H1W_260402_LINKS = [
    "base_link",
    "waist_yaw_link",
    "torso_link",
    "head_yaw_link",
    "head_pitch_link",
    "l_shoulder_pitch_link",
    "l_shoulder_roll_link",
    "l_upper_arm_link",
    "l_elbow_link",
    "l_wrist_link",
    "r_shoulder_pitch_link",
    "r_shoulder_roll_link",
    "r_upper_arm_link",
    "r_elbow_link",
    "r_wrist_link",
    "l_hip_pitch_link",
    "l_hip_roll_link",
    "l_thigh_link",
    "l_calf_link",
    "l_ankle_pitch_link",
    "l_ankle_roll_link",
    "r_hip_pitch_link",
    "r_hip_roll_link",
    "r_thigh_link",
    "r_calf_link",
    "r_ankle_pitch_link",
    "r_ankle_roll_link",
]

HI_260402_CFG = Hi_P_12L10A0G2H1W_260402_CFG
HI_260402_ACTION_SCALE = Hi_P_12L10A0G2H1W_260402_ACTION_SCALE
Hi_P_12L10A0G2H1W_260402_JOINT_NAMES = HI_260402_JOINT_NAMES
Hi_P_12L10A0G2H1W_260402_DEPLOY_JOINT_NAMES = HI_260402_DEPLOY_JOINT_NAMES
HI_260402_symmetric_augmentation_joint_mapping = Hi_P_12L10A0G2H1W_260402_symmetric_augmentation_joint_mapping
HI_260402_symmetric_augmentation_joint_reverse_buf = Hi_P_12L10A0G2H1W_260402_symmetric_augmentation_joint_reverse_buf
HI_260402_LINKS = Hi_P_12L10A0G2H1W_260402_LINKS
