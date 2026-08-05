import sys
import os
from loguru import logger
import torch
from humanoidverse.utils.torch_utils import to_torch, torch_rand_float
import numpy as np
from typing import Optional
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
# from humanoidverse.simulator.isaaclab_cfg import IsaacLabCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils.timer import Timer

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, Imu, RayCaster
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
from isaaclab.assets import ArticulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen

from isaaclab_assets import H1_CFG
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.envs import ViewerCfg

import isaaclab.sim as sim_utils

from humanoidverse.simulator.isaacsim.isaaclab_viewpoint_camera_controller import ViewportCameraController
import builtins
import inspect
import copy
from dataclasses import MISSING
from humanoidverse.simulator.isaacsim.isaacsim_articulation_cfg import ARTICULATION_CFG
from humanoidverse.utils.asset_paths import resolve_asset_path

from humanoidverse.simulator.isaacsim.event_cfg import EventCfg

from isaaclab.managers import EventManager

from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from humanoidverse.simulator.isaacsim.events import randomize_body_com
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.sensors import TiledCamera, TiledCameraCfg, Camera, CameraCfg, FrameTransformerCfg, FrameTransformer
from humanoidverse.simulator.isaacsim.actuators import (
    ARMATURE_3536,
    ARMATURE_4438,
    ARMATURE_5031,
    ARMATURE_5036,
    ARMATURE_6036,
    DAMPING_3536,
    DAMPING_4438,
    DAMPING_5031,
    DAMPING_5036,
    DAMPING_6036,
    HTMotorCfg_4438,
    HTMotorCfg_5031,
    HTMotorCfg_5036,
    HTMotorCfg_6036,
    HTMotor40VCfg_3536,
    HTMotor40VCfg_4438,
    HTMotor40VCfg_5036,
    STIFFNESS_3536,
    STIFFNESS_4438,
    STIFFNESS_5031,
    STIFFNESS_5036,
    STIFFNESS_6036,
)

class IsaacSim(BaseSimulator):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.simulator_config = config.simulator.config
        self.robot_config = config.robot
        self.env_config = config
        self.terrain_config = config.terrain
        self.domain_rand_config = config.domain_rand
        self.isaacsim_torso_name = self._resolve_isaacsim_torso_name()
        
        sim_config: SimulationCfg = SimulationCfg(dt=1./self.simulator_config.sim.fps, 
                                           render_interval=self.simulator_config.sim.render_interval, 
                                           device=self.sim_device,
                                           physx=PhysxCfg(bounce_threshold_velocity=self.simulator_config.sim.physx.bounce_threshold_velocity,
                                                          solver_type=self.simulator_config.sim.physx.solver_type,
                                                          max_position_iteration_count=self.simulator_config.sim.physx.num_position_iterations,
                                                          max_velocity_iteration_count=self.simulator_config.sim.physx.num_velocity_iterations))
        
        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            self.sim: SimulationContext = SimulationContext(sim_config)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")

        self.sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
        
        logger.info("IsaacSim initialized.")
        # Log useful information
        logger.info("[INFO]: Base environment:")
        logger.info(f"\tEnvironment device    : {self.sim_device}")
        logger.info(f"\tPhysics step-size     : {1./self.simulator_config.sim.fps}")
        logger.info(f"\tRendering step-size   : {1./self.simulator_config.sim.fps * self.simulator_config.sim.substeps}")


        if self.simulator_config.sim.render_interval < self.simulator_config.sim.control_decimation:
            msg = (
                f"The render interval ({self.simulator_config.sim.render_interval}) is smaller than the decimation "
                f"({self.simulator_config.sim.control_decimation}). Multiple render calls will happen for each environment step."
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            logger.warning(msg)
        
        
        scene_config: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=self.simulator_config.scene.num_envs, env_spacing=self.simulator_config.scene.env_spacing, replicate_physics=self.simulator_config.scene.replicate_physics)
        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene = InteractiveScene(scene_config)
            self._setup_scene()
        print("[INFO]: Scene manager: ", self.scene)
    
        
        viewer_config: ViewerCfg = ViewerCfg()
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, viewer_config)
        else:
            self.viewport_camera_controller = None

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            logger.info("Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()
        
        self.default_coms = self._robot.root_physx_view.get_coms().clone()
        self.base_com_bias = torch.zeros((self.simulator_config.scene.num_envs, 3), dtype=torch.float, device="cpu")

        self.event_types = set()
        self.events_cfg = EventCfg()
        if self.domain_rand_config.get("randomize_link_mass", False):
            self.events_cfg.scale_body_mass = EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=self.robot_config.get("randomize_link_body_names", ".*"),
                    ),
                    "mass_distribution_params": tuple(self.domain_rand_config["link_mass_range"]),
                    "operation": "scale",
                    "distribution": "uniform",
                },
            )
            self.event_types.add("startup")

        # Randomize joint friction
        if self.domain_rand_config.get("randomize_friction", False):
            self.events_cfg.random_joint_friction = EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": self.domain_rand_config["friction_range"],
                    "dynamic_friction_range": self.domain_rand_config["friction_range"],
                    "restitution_range": self.domain_rand_config.get("restitution_range", (0.0, 0.0)),
                    "num_buckets": self.domain_rand_config.get("friction_num_buckets", 1024),
                    "make_consistent": self.domain_rand_config.get("friction_make_consistent", False),
                },
            )
            
            
            self.event_types.add("startup")

        if self.domain_rand_config.get("randomize_base_com", False):
            
            self.events_cfg.random_base_com = EventTerm(
                func=randomize_body_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=[
                            self.isaacsim_torso_name,
                        ],
                    ),
                    "distribution_params": (
                        torch.tensor([float(self.domain_rand_config["base_com_range"]["x"][0]), float(self.domain_rand_config["base_com_range"]["y"][0]), float(self.domain_rand_config["base_com_range"]["z"][0])]),
                        torch.tensor([float(self.domain_rand_config["base_com_range"]["x"][1]), float(self.domain_rand_config["base_com_range"]["y"][1]), float(self.domain_rand_config["base_com_range"]["z"][1])])
                    ),
                    "operation": "add",
                    "distribution": "uniform",
                    "num_envs": self.simulator_config.scene.num_envs,
                },
            )
            self.event_types.add("startup")
            
        if self.domain_rand_config.get("push_robots", False):
            self.events_cfg.push_robots = EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="on_push_by_setting_velocity",
                params={"velocity_range": {"x": (-self.domain_rand_config["max_push_vel_xy"],
                                                 self.domain_rand_config["max_push_vel_xy"]),
                                           "y": (-self.domain_rand_config["max_push_vel_xy"], 
                                                 self.domain_rand_config["max_push_vel_xy"]),
                                           "z": (-self.domain_rand_config.get("max_push_vel_z", 0.0),
                                                 self.domain_rand_config.get("max_push_vel_z", 0.0)),
                                           "roll": (-self.domain_rand_config["max_push_ang_vel"], 
                                                 self.domain_rand_config["max_push_ang_vel"]),
                                           "pitch": (-self.domain_rand_config["max_push_ang_vel"], 
                                                 self.domain_rand_config["max_push_ang_vel"]),
                                           "yaw": (-self.domain_rand_config["max_push_ang_vel"], 
                                                 self.domain_rand_config["max_push_ang_vel"]),
                                           },
                        },
            )
            self.event_types.add("on_push_by_setting_velocity")    
        
        self.event_manager = EventManager(self.events_cfg, self)
        print("[INFO] Event Manager: ", self.event_manager)
        
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
                
        # -- event manager used for randomization
        # if self.cfg.events:
        #     self.event_manager = EventManager(self.cfg.events, self)
        #     print("[INFO] Event Manager: ", self.event_manager)

        if "cuda" in self.sim_device:
            torch.cuda.set_device(self.sim_device)
        
        # # extend UI elements
        # # we need to do this here after all the managers are initialized
        # # this is because they dictate the sensors and commands right now
        # if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
        #     self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        # else:
        #     # if no window, then we don't need to store the window
        #     self._window = None


        # perform events at the start of the simulation
        # if self.cfg.events:
        #     if "startup" in self.event_manager.available_modes:
        #         self.event_manager.apply(mode="startup")

        # # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        # self.metadata["render_fps"] = 1. / self.config.sim.fps * self.config.sim.control_decimation


        self._sim_step_counter = 0

        # debug visualization
        # self.draw = _debug_draw.acquire_debug_draw_interface()
        
        # print the environment information
        logger.info("Completed setting up the environment...")
        
    def _resolve_isaacsim_torso_name(self) -> str:
        return self.robot_config.get(
            "isaacsim_torso_name",
            self.robot_config.get("torso_name", "pelvis"),
        )

    def _resolve_height_scanner_body_name(self) -> str:
        motion_config = self.robot_config.get("motion", {})
        return motion_config.get(
            "pelvis_link",
            self._resolve_isaacsim_torso_name(),
        )

    def _build_implicit_actuators(
        self,
        dof_names_list,
        dof_effort_limit_list,
        dof_vel_limit_list,
        dof_armature_list,
        dof_joint_friction_list,
    ):
        stiffness_dict = {
            ".*" + key + ".*": value
            for key, value in self.robot_config.control.stiffness.items()
        }
        damping_dict = {
            ".*" + key + ".*": value
            for key, value in self.robot_config.control.damping.items()
        }
        return {
            "all": ImplicitActuatorCfg(
                joint_names_expr=[dof_names_list[i] for i in range(len(dof_names_list))],
                effort_limit_sim={
                    dof_names_list[i]: dof_effort_limit_list[i] for i in range(len(dof_names_list))
                },
                velocity_limit_sim={
                    dof_names_list[i]: dof_vel_limit_list[i] for i in range(len(dof_names_list))
                },
                stiffness=stiffness_dict,
                damping=damping_dict,
                armature={
                    dof_names_list[i]: dof_armature_list[i] for i in range(len(dof_names_list))
                },
                friction={
                    dof_names_list[i]: dof_joint_friction_list[i] for i in range(len(dof_names_list))
                },
            )
        }

    def _build_piplus_ht_motor_actuators(self):
        def motor_cfg_value(motor_cfg_cls, field_name):
            config_fields = getattr(motor_cfg_cls, "__dataclass_fields__", {})
            if field_name in config_fields:
                default = config_fields[field_name].default
                if default is not MISSING:
                    return default
            if hasattr(motor_cfg_cls, field_name):
                return getattr(motor_cfg_cls, field_name)
            return getattr(motor_cfg_cls(), field_name)

        def motor_constants(motor_cfg_name, fallback_model):
            model_constants = {
                "3536": (STIFFNESS_3536, DAMPING_3536, ARMATURE_3536),
                "4438": (STIFFNESS_4438, DAMPING_4438, ARMATURE_4438),
                "5031": (STIFFNESS_5031, DAMPING_5031, ARMATURE_5031),
                "5036": (STIFFNESS_5036, DAMPING_5036, ARMATURE_5036),
                "6036": (STIFFNESS_6036, DAMPING_6036, ARMATURE_6036),
            }
            for model_name in ("6036", "5036", "5031", "4438", "3536"):
                if motor_cfg_name.endswith(model_name):
                    return model_constants[model_name]
            return model_constants[fallback_model]

        motor_actuators = self.robot_config.control.get("motor_actuators", {})
        use_motor_limits = bool(motor_actuators)
        legs_motor_cfg = globals()[motor_actuators.get("legs", "HTMotorCfg_5036")]
        feet_motor_cfg = globals()[motor_actuators.get("feet", "HTMotorCfg_5036")]
        arms_motor_cfg = globals()[motor_actuators.get("arms", "HTMotorCfg_4438")]
        wrists_motor_cfg_name = motor_actuators.get("wrists")
        wrists_motor_cfg = globals()[wrists_motor_cfg_name] if wrists_motor_cfg_name else None
        head_motor_cfg_name = motor_actuators.get("head")
        head_motor_cfg = globals()[head_motor_cfg_name] if head_motor_cfg_name else ImplicitActuatorCfg
        head_motor_delay = {"min_delay": 0, "max_delay": 3} if head_motor_cfg_name else {}
        waist_yaw_motor_cfg_name = motor_actuators.get("waist_yaw", "HTMotorCfg_5031")
        waist_yaw_motor_cfg = globals()[waist_yaw_motor_cfg_name]
        legs_stiffness, legs_damping, legs_armature = motor_constants(motor_actuators.get("legs", "HTMotorCfg_5036"), "5036")
        feet_stiffness, feet_damping, feet_armature = motor_constants(motor_actuators.get("feet", "HTMotorCfg_5036"), "5036")
        arms_stiffness, arms_damping, arms_armature = motor_constants(motor_actuators.get("arms", "HTMotorCfg_4438"), "4438")
        wrists_stiffness, wrists_damping, wrists_armature = motor_constants(wrists_motor_cfg_name or "HTMotorCfg_4438", "4438")
        waist_yaw_stiffness, waist_yaw_damping, waist_yaw_armature = motor_constants(waist_yaw_motor_cfg_name, "5031")
        legs_effort_limit = motor_cfg_value(legs_motor_cfg, "max_torque") if use_motor_limits else 20.0
        legs_velocity_limit = motor_cfg_value(legs_motor_cfg, "max_velocity") if use_motor_limits else 60.0
        feet_effort_limit = motor_cfg_value(feet_motor_cfg, "max_torque") if use_motor_limits else 20.0
        feet_velocity_limit = motor_cfg_value(feet_motor_cfg, "max_velocity") if use_motor_limits else 60.0
        arms_effort_limit = motor_cfg_value(arms_motor_cfg, "max_torque") if use_motor_limits else 20.0
        arms_velocity_limit = motor_cfg_value(arms_motor_cfg, "max_velocity") if use_motor_limits else 60.0
        wrists_effort_limit = motor_cfg_value(wrists_motor_cfg, "max_torque") if wrists_motor_cfg else 10.0
        wrists_velocity_limit = motor_cfg_value(wrists_motor_cfg, "max_velocity") if wrists_motor_cfg else 60.0
        head_effort_limit = motor_cfg_value(head_motor_cfg, "max_torque") if head_motor_cfg_name else 3.0
        head_velocity_limit = motor_cfg_value(head_motor_cfg, "max_velocity") if head_motor_cfg_name else 60.0
        waist_yaw_effort_limit = motor_cfg_value(waist_yaw_motor_cfg, "max_torque") if use_motor_limits else 20.0
        waist_yaw_velocity_limit = motor_cfg_value(waist_yaw_motor_cfg, "max_velocity") if use_motor_limits else 60.0

        actuators = {
            "legs": legs_motor_cfg(
                joint_names_expr=[
                    ".*_thigh_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_calf_joint",
                ],
                effort_limit_sim={
                    ".*_thigh_joint": legs_effort_limit,
                    ".*_hip_roll_joint": legs_effort_limit,
                    ".*_hip_pitch_joint": legs_effort_limit,
                    ".*_calf_joint": legs_effort_limit,
                },
                velocity_limit_sim={
                    ".*_thigh_joint": legs_velocity_limit,
                    ".*_hip_roll_joint": legs_velocity_limit,
                    ".*_hip_pitch_joint": legs_velocity_limit,
                    ".*_calf_joint": legs_velocity_limit,
                },
                stiffness={
                    ".*_hip_pitch_joint": legs_stiffness,
                    ".*_hip_roll_joint": legs_stiffness,
                    ".*_thigh_joint": legs_stiffness,
                    ".*_calf_joint": legs_stiffness,
                },
                damping={
                    ".*_hip_pitch_joint": legs_damping,
                    ".*_hip_roll_joint": legs_damping,
                    ".*_thigh_joint": legs_damping,
                    ".*_calf_joint": legs_damping,
                },
                armature={
                    ".*_hip_pitch_joint": legs_armature,
                    ".*_hip_roll_joint": legs_armature,
                    ".*_thigh_joint": legs_armature,
                    ".*_calf_joint": legs_armature,
                },
                min_delay=0,
                max_delay=3,
            ),
            "feet": feet_motor_cfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit_sim=feet_effort_limit,
                velocity_limit_sim=feet_velocity_limit,
                stiffness=feet_stiffness,
                damping=feet_damping,
                armature=feet_armature,
                min_delay=0,
                max_delay=3,
            ),
            "head": head_motor_cfg(
                joint_names_expr=[".*head_yaw_joint", ".*head_pitch_joint"],
                effort_limit_sim=head_effort_limit,
                velocity_limit_sim=head_velocity_limit,
                stiffness=STIFFNESS_3536,
                damping=DAMPING_3536,
                armature=ARMATURE_3536,
                **head_motor_delay,
            ),
            "arms": arms_motor_cfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_upper_arm_joint",
                    ".*_elbow_joint",
                ],
                effort_limit_sim={
                    ".*_shoulder_pitch_joint": arms_effort_limit,
                    ".*_shoulder_roll_joint": arms_effort_limit,
                    ".*_upper_arm_joint": arms_effort_limit,
                    ".*_elbow_joint": arms_effort_limit,
                },
                velocity_limit_sim={
                    ".*_shoulder_pitch_joint": arms_velocity_limit,
                    ".*_shoulder_roll_joint": arms_velocity_limit,
                    ".*_upper_arm_joint": arms_velocity_limit,
                    ".*_elbow_joint": arms_velocity_limit,
                },
                stiffness={
                    ".*_shoulder_pitch_joint": arms_stiffness,
                    ".*_shoulder_roll_joint": arms_stiffness,
                    ".*_upper_arm_joint": arms_stiffness,
                    ".*_elbow_joint": arms_stiffness,
                },
                damping={
                    ".*_shoulder_pitch_joint": arms_damping,
                    ".*_shoulder_roll_joint": arms_damping,
                    ".*_upper_arm_joint": arms_damping,
                    ".*_elbow_joint": arms_damping,
                },
                armature={
                    ".*_shoulder_pitch_joint": arms_armature,
                    ".*_shoulder_roll_joint": arms_armature,
                    ".*_upper_arm_joint": arms_armature,
                    ".*_elbow_joint": arms_armature,
                },
                min_delay=0,
                max_delay=3,
            ),
        }
        if wrists_motor_cfg is not None and any("wrist_joint" in dof_name for dof_name in self.robot_config.dof_names):
            actuators["wrists"] = wrists_motor_cfg(
                joint_names_expr=[".*_wrist_joint"],
                effort_limit_sim=wrists_effort_limit,
                velocity_limit_sim=wrists_velocity_limit,
                stiffness=wrists_stiffness,
                damping=wrists_damping,
                armature=wrists_armature,
                min_delay=0,
                max_delay=3,
            )
        if self.robot_config.get("waist_dof_names", []):
            actuators["waist_yaw"] = waist_yaw_motor_cfg(
                joint_names_expr=["waist_yaw_joint"],
                effort_limit_sim=waist_yaw_effort_limit,
                velocity_limit_sim=waist_yaw_velocity_limit,
                stiffness=waist_yaw_stiffness,
                damping=waist_yaw_damping,
                armature=waist_yaw_armature,
                min_delay=0,
                max_delay=3,
            )
        return actuators

    def _build_actuators(
        self,
        dof_names_list,
        dof_effort_limit_list,
        dof_vel_limit_list,
        dof_armature_list,
        dof_joint_friction_list,
    ):
        if self.robot_config.control.get("use_ht_motor", False):
            print("Using HTMotorCfg")
            return self._build_piplus_ht_motor_actuators()
        return self._build_implicit_actuators(
            dof_names_list,
            dof_effort_limit_list,
            dof_vel_limit_list,
            dof_armature_list,
            dof_joint_friction_list,
        )

    def _setup_scene(self):
        asset_root = self.robot_config.asset.asset_root
        asset_path = self.robot_config.asset.usd_file or self.robot_config.asset.urdf_file
        # prepare to override the spawn configuration in HumanoidVerse/humanoidverse/simulator/isaacsim_articulation_cfg.py
        asset_abs_path = str(resolve_asset_path(asset_root, asset_path).resolve())

        assert(os.path.isfile(asset_abs_path))

        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        )
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=not bool(self.env_config.robot.asset.self_collisions),
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        )
        if self.robot_config.asset.usd_file:
            spawn = sim_utils.UsdFileCfg(
                usd_path=asset_abs_path,
                activate_contact_sensors=True,
                rigid_props=rigid_props,
                articulation_props=articulation_props,
            )
        else:
            spawn = sim_utils.UrdfFileCfg(
                asset_path=asset_abs_path,
                fix_base=bool(self.robot_config.asset.fix_base_link),
                merge_fixed_joints=bool(self.robot_config.asset.collapse_fixed_joints),
                replace_cylinders_with_capsules=bool(self.robot_config.asset.replace_cylinder_with_capsule),
                activate_contact_sensors=True,
                rigid_props=rigid_props,
                articulation_props=articulation_props,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
                ),
            )

        # prepare to override the articulation configuration in HumanoidVerse/humanoidverse/simulator/isaacsim_articulation_cfg.py
        default_joint_angles = copy.deepcopy(self.robot_config.init_state.default_joint_angles)
        # import ipdb; ipdb.set_trace()
        init_state = ArticulationCfg.InitialStateCfg(
            pos=tuple(self.robot_config.init_state.pos),
            joint_pos={
                joint_name: joint_angle for joint_name, joint_angle in default_joint_angles.items()
            },
            joint_vel={".*": 0.0},
        )
       
        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        # for i, name in enumerate(dof_names_list):
        #     dof_names_list[i] = name.replace("_joint", "")    
        dof_effort_limit_list = self.robot_config.dof_effort_limit_list
        dof_vel_limit_list = self.robot_config.dof_vel_limit_list
        dof_armature_list = self.robot_config.dof_armature_list
        dof_joint_friction_list = self.robot_config.dof_joint_friction_list

        # get kp and kd from config
        kp_list = []
        kd_list = []
        stiffness_dict = self.robot_config.control.stiffness
        damping_dict = self.robot_config.control.damping
        
        for i in range(len(dof_names_list)):
            dof_names_i_without_joint = dof_names_list[i].replace("_joint", "")
            for key in stiffness_dict.keys():
                if key in dof_names_i_without_joint:
                    kp_list.append(stiffness_dict[key])
                    kd_list.append(damping_dict[key])
                    print(f"key: {key}, kp: {stiffness_dict[key]}, kd: {damping_dict[key]}")
                    
                    
        actuators = self._build_actuators(
            dof_names_list,
            dof_effort_limit_list,
            dof_vel_limit_list,
            dof_armature_list,
            dof_joint_friction_list,
        )
 
        robot_articulation_config: ArticulationCfg = ARTICULATION_CFG.replace(prim_path="/World/envs/env_.*/Robot", spawn=spawn, init_state=init_state, actuators=actuators)
        
        contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
        )
        contact_pair_sensor_configs = self._build_contact_pair_sensor_configs()

        imu_body_name = self.robot_config.get("imu_body_name", None)
        imu_body_config = None
        if imu_body_name is not None:
            imu_body_config = ImuCfg(
                prim_path=f"/World/envs/env_.*/Robot/{imu_body_name}",
                offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )

        height_scanner_body = self._resolve_height_scanner_body_name()
        # Add a height scanner to the robot root body to detect the height of the terrain mesh.
        height_scanner_config = RayCasterCfg(
            prim_path=f"/World/envs/env_.*/Robot/{height_scanner_body}",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            # Apply a grid pattern that is smaller than the resolution to only return one height value.
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        
        if (self.terrain_config.mesh_type == "heightfield") or (self.terrain_config.mesh_type == "trimesh"):
            sub_terrains = {}
            terrain_types = self.terrain_config.terrain_types
            terrain_proportions = self.terrain_config.terrain_proportions
            for terrain_type, proportion in zip(terrain_types, terrain_proportions):
                if proportion > 0:
                    if terrain_type == "flat":
                        sub_terrains[terrain_type] = terrain_gen.MeshPlaneTerrainCfg(
                            proportion=proportion
                        )
                    elif terrain_type == "rough":
                        sub_terrains[terrain_type] = terrain_gen.HfRandomUniformTerrainCfg(
                            proportion=proportion, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
                        )
                    elif terrain_type == "low_obst":
                        sub_terrains[terrain_type] = terrain_gen.MeshRandomGridTerrainCfg(
                            proportion=proportion, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
                        )

            terrain_generator_config = TerrainGeneratorCfg(
                curriculum=self.terrain_config.curriculum,
                size=(self.terrain_config.terrain_length, self.terrain_config.terrain_width),
                border_width=self.terrain_config.border_size,
                num_rows=self.terrain_config.num_rows,
                num_cols=self.terrain_config.num_cols,
                horizontal_scale=self.terrain_config.horizontal_scale,
                vertical_scale=self.terrain_config.vertical_scale,
                slope_threshold=self.terrain_config.slope_treshold,
                use_cache=False,
                sub_terrains=sub_terrains,
            )

            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=terrain_generator_config,
                max_init_terrain_level=9,
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                ),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                    project_uvw=True,
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            # terrain_config.env_spacing = self.scene.cfg.env_spacing

        else:
            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                    restitution=0.0,
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            terrain_config.env_spacing = self.scene.cfg.env_spacing
        
        self._robot = Articulation(robot_articulation_config)
        self.scene.articulations["robot"] = self._robot
        self.contact_sensor = ContactSensor(contact_sensor_config)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.contact_pair_sensors = {}
        for pair_name, pair_sensor_config in contact_pair_sensor_configs.items():
            sensor_name = f"contact_pair_{pair_name}"
            pair_sensor = ContactSensor(pair_sensor_config)
            self.contact_pair_sensors[pair_name] = pair_sensor
            self.scene.sensors[sensor_name] = pair_sensor
        self.imu_body = None
        if imu_body_config is not None:
            self.imu_body = Imu(imu_body_config)
            self.scene.sensors["imu_body"] = self.imu_body
        self._height_scanner = RayCaster(height_scanner_config)
        self.scene.sensors["height_scanner"] = self._height_scanner

        
        self.terrain = terrain_config.class_type(terrain_config)
        self.terrain.env_origins = self.terrain.terrain_origins
        
        # import ipdb; ipdb.set_trace()

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[terrain_config.prim_path])

        # add lights
        # light_config = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.98, 0.95, 0.88))
        # light_config.func("/World/Light", light_config)

        light_config1 = sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(0.98, 0.95, 0.88),
        )
        light_config1.func("/World/DomeLight", light_config1, translation=(1, 0, 10))
        
        self.vis_sphere_marker_names = ["sphere"]
        self.vis_sphere_marker_body_to_index = {}
        sphere_markers = {
            "sphere": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ),
        }
        marker_colors = self.robot_config.motion.get("visualization", {}).get("marker_joint_colors", [])
        customize_marker_colors = self.robot_config.motion.get("visualization", {}).get("customize_color", False)
        if customize_marker_colors and marker_colors:
            sphere_markers = {}
            self.vis_sphere_marker_names = []
            marker_body_names = list(self.robot_config.get("body_names", []))
            for color_id, color in enumerate(marker_colors):
                marker_name = f"sphere_{color_id}"
                sphere_markers[marker_name] = sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
                )
                self.vis_sphere_marker_names.append(marker_name)
                if color_id < len(marker_body_names):
                    self.vis_sphere_marker_body_to_index[marker_body_names[color_id]] = color_id

        self.vis_spheres = VisualizationMarkers(VisualizationMarkersCfg( prim_path="/Visuals/goal_marker",
            markers=sphere_markers))
        if self.config.simulator.get('enable_cameras', False):
            self.setup_rendering_cameras()

    def _build_contact_pair_sensor_configs(self):
        body_names = set(self.robot_config.get("isaacsim_body_names", self.robot_config.body_names))
        pair_specs = {
            "r_elbow_r_thigh": ("r_elbow_link", "r_thigh_link"),
            "l_elbow_l_thigh": ("l_elbow_link", "l_thigh_link"),
            "r_thigh_base": ("r_thigh_link", "base_link"),
            "l_thigh_base": ("l_thigh_link", "base_link"),
            "l_ankle_pitch_r_ankle_pitch": ("l_ankle_pitch_link", "r_ankle_pitch_link"),
            "l_ankle_pitch_r_ankle_roll": ("l_ankle_pitch_link", "r_ankle_roll_link"),
            "l_ankle_roll_r_ankle_pitch": ("l_ankle_roll_link", "r_ankle_pitch_link"),
            "l_ankle_roll_r_ankle_roll": ("l_ankle_roll_link", "r_ankle_roll_link"),
            "r_elbow_waist_yaw": ("r_elbow_link", "waist_yaw_link"),
            "l_elbow_waist_yaw": ("l_elbow_link", "waist_yaw_link"),
        }
        pair_sensor_configs = {}
        for pair_name, (sensor_body_name, filter_body_name) in pair_specs.items():
            if sensor_body_name not in body_names or filter_body_name not in body_names:
                continue
            pair_sensor_configs[pair_name] = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{sensor_body_name}",
                history_length=1,
                update_period=0.005,
                filter_prim_paths_expr=[f"/World/envs/env_.*/Robot/{filter_body_name}"],
            )
        return pair_sensor_configs
        
    def setup_keyboard(self):
        # TODO: add back
        from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard, Se2KeyboardCfg
        self.keyboard_interface = Se2Keyboard(Se2KeyboardCfg(sim_device=self.sim_device))
        
    def add_keyboard_callback(self, key, callback):
        # TODO: add back
        self.keyboard_interface.add_callback(key, callback)
        
        
    def set_headless(self, headless):
        # call super
        super().set_headless(headless)
        if not self.headless:
            from isaacsim.util.debug_draw import _debug_draw
            self.draw = _debug_draw.acquire_debug_draw_interface()
        else:
            self.draw = None

    def setup(self):
        self.sim_dt = 1. / self.simulator_config.sim.fps
        if not self.headless:
            self.setup_keyboard()
        
        
    def setup_terrain(self, mesh_type):
        pass
    
    def trigger_on_push_by_setting_velocity_events(self, env_ids: torch.Tensor | None = None, global_env_step_count: Optional[int] = None):
        self.event_manager.apply(env_ids=env_ids, mode="on_push_by_setting_velocity", global_env_step_count=global_env_step_count)
    

    def load_assets(self):
        '''
        save self.num_dofs, self.num_bodies, self.dof_names, self.body_names in simulator class
        '''

        dof_names_list = copy.deepcopy(self.robot_config.dof_names)

        self.dof_ids, self.dof_names = self._robot.find_joints(dof_names_list, preserve_order=True) 
        body_names_list = self.robot_config.get("isaacsim_body_names", self.robot_config.body_names)
        self.body_ids, self.body_names = self._robot.find_bodies(body_names_list, preserve_order=True)
        
        self.contact_to_body_idx = [self.contact_sensor.body_names.index(body_name) for body_name in self.body_names]

        self._body_list = self.body_names.copy()
        # dof_ids and body_ids is convert dfs order (isaacsim) to dfs order (isaacgym, humanoidverse config)
            # i.e., bfs_order_tensor = dfs_order_tensor[dof_ids]

    
        # add joint names with "joint" postfix
        # for i, name in enumerate(self.dof_names):
        #     self.dof_names[i] = name + "_joint"
        '''
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=True)
        ([0, 1, 4, 8, 12, 16, 2, 5, 9, 13, 17, 3, 6, 10, 14, 18, 7, 11, 15, 19], ['pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'])
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=False)
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], ['pelvis', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'left_hip_roll_link', 'right_hip_roll_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_hip_pitch_link', 'right_hip_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_knee_link', 'right_knee_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_ankle_link', 'right_ankle_link', 'left_elbow_link', 'right_elbow_link'])
        '''
        
        self.num_dof = len(self.dof_ids)
        self.num_bodies = len(self.body_ids)

        # warning if the dof_ids order does not match the joint_names order in robot_config
        if self.dof_ids != list(range(self.num_dof)):
            logger.warning("The order of the joint_names in the robot_config does not match the order of the joint_ids in IsaacSim.")
        
        # assert if  aligns with config
        assert self.num_dof == len(self.robot_config.dof_names), "Number of DOFs must be equal to number of actions"
        assert self.num_bodies == len(body_names_list), "Number of bodies must be equal to number of body names"
        # import ipdb; ipdb.set_trace()
        assert self.dof_names == self.robot_config.dof_names, "DOF names must match the config"
        assert self.body_names == list(body_names_list), "Body names must match the config"
       
        
        # return self.num_dof, self.num_bodies, self.dof_names, self.body_names
        

    def create_envs(self, num_envs, env_origins, base_init_state):
        
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state
        
        return self.scene, self._robot
    
    def get_dof_limits_properties(self):
        self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_config.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_config.dof_effort_limit_list[i]
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name):
        '''
        ipdb> self.simulator._robot.find_bodies("left_ankle_link")
        ([16], ['left_ankle_link'])
        ipdb> self.simulator.contact_sensor.find_bodies("left_ankle_link")
        ([4], ['left_ankle_link'])

        this function returns the indice of the body in BFS order
        '''
        indices, names = self._robot.find_bodies(body_name)
        indices = [self.body_ids.index(i) for i in indices]
        if len(indices) == 0:
            logger.warning(f"Body {body_name} not found in the contact sensor.")
            return None
        elif len(indices) == 1:
            return indices[0]
        else: # multiple bodies found
            logger.warning(f"Multiple bodies found for {body_name}.")
            return indices
                
    def prepare_sim(self):
        self.refresh_sim_tensors() # initialize tensors

    @property
    def dof_state(self):
        # This will always use the latest dof_pos and dof_vel
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    def refresh_sim_tensors(self):
        ############################################################################################
        # TODO: currently, we only consider the robot root state, ignore other objects's root states
        ############################################################################################
        pass
        
        
    @property
    def contact_forces(self):
        return self.contact_sensor.data.net_forces_w[:, self.contact_to_body_idx, :]  # (num_envs, num_bodies, 3)

    @property
    def contact_pair_forces(self):
        pair_forces = {}
        for pair_name, pair_sensor in getattr(self, "contact_pair_sensors", {}).items():
            force_matrix_w = pair_sensor.data.force_matrix_w
            if force_matrix_w is None or force_matrix_w.shape[1] == 0 or force_matrix_w.shape[2] == 0:
                continue
            pair_forces[pair_name] = force_matrix_w[:, 0, 0, :]
        return pair_forces
    
    @property
    def base_quat(self):
        return self._robot.data.root_state_w[:, [4, 5, 6, 3]] # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency
    
    @property
    def all_root_states(self):
        return self._robot.data.root_state_w[:, [0, 1, 2, 4, 5, 6, 3, 7, 8, 9, 10, 11, 12]]
    
    @property
    def robot_root_states(self):
        return self._robot.data.root_state_w[:, [0, 1, 2, 4, 5, 6, 3, 7, 8, 9, 10, 11, 12]]
    
    @property
    def dof_pos(self):
        return self._robot.data.joint_pos[:, self.dof_ids]
    
    @property
    def dof_vel(self):
        return self._robot.data.joint_vel[:, self.dof_ids]
    
    @property
    def _rigid_body_pos(self):
        return self._robot.data.body_pos_w[:, self.body_ids, :]
    
    @property
    def _rigid_body_rot(self):
        return self._robot.data.body_quat_w[:, self.body_ids][:, :, [1, 2, 3, 0]] # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency
    
    @property
    def _rigid_body_vel(self):
        return self._robot.data.body_lin_vel_w[:, self.body_ids, :]
    
    @property
    def _rigid_body_ang_vel(self):
        return self._robot.data.body_ang_vel_w[:, self.body_ids, :]

    @property
    def _rigid_body_dr_masses(self):
        masses = self._robot.root_physx_view.get_masses().to(self.sim_device)
        return masses[:, self.body_ids]

    @property
    def _rigid_body_dr_friction(self):
        frictions = self._robot.root_physx_view.get_material_properties()[:, self.body_ids, :].to(self.sim_device)
        # Tensor is [n_envs, n_bodies, 3] where 3 is [static_friction, dynamic_friction, restitution]
        return frictions.reshape(frictions.shape[0], frictions.shape[1] * 3)

    @property
    def _rigid_body_dr_coms(self):
        coms = self._robot.root_physx_view.get_coms()[:, self.body_ids, :].to(self.sim_device)
        return coms.reshape(coms.shape[0], coms.shape[1] * coms.shape[2])

    def apply_torques_at_dof(self, torques):
        self._robot.set_joint_effort_target(torques, joint_ids=self.dof_ids)
    
    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        self._robot.write_root_pose_to_sim(root_states[set_env_ids, :7], set_env_ids)
        self._robot.write_root_velocity_to_sim(root_states[set_env_ids, 7:], set_env_ids)

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        dof_pos, dof_vel = dof_states[set_env_ids, :, 0], dof_states[set_env_ids, :, 1]
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, self.dof_ids, set_env_ids)
    
    def simulate_at_each_physics_step(self):
        self._sim_step_counter += 1
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        
        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        if self._sim_step_counter % self.simulator_config.sim.render_interval == 0 and is_rendering:
            self.sim.render()
        # update buffers at sim 
        self.scene.update(dt=1./self.simulator_config.sim.fps)
    
    def setup_viewer(self):
        self.viewer = self.viewport_camera_controller


    def render(self, sync_frame_time=True):
        pass

     # debug visualization
    def clear_lines(self):
        self.draw.clear_lines()
        self.draw.clear_points()

    def draw_sphere(self, pos, radius, color, env_id, pos_id):
        # draw a big sphere
        point_list = [(pos[0].item(), pos[1].item(), pos[2].item())]
        color_list = [(color[0], color[1], color[2], 1.0)]
        sizes = [20]
        self.draw.draw_points(point_list, color_list, sizes)
        
    def draw_spheres_batch(self, pos, rot = None, scales = None, marker_indices = None):
        if marker_indices is None and len(self.vis_sphere_marker_names) > 1:
            marker_body_to_index = getattr(self, "vis_sphere_marker_body_to_index", {})
            if marker_body_to_index and pos.shape[0] == len(self.body_names):
                marker_indices = torch.tensor(
                    [
                        marker_body_to_index.get(body_name, body_id % len(self.vis_sphere_marker_names))
                        for body_id, body_name in enumerate(self.body_names)
                    ],
                    dtype=torch.long,
                    device=pos.device,
                )
            elif marker_body_to_index and pos.shape[0] == len(self._body_list):
                marker_indices = torch.tensor(
                    [
                        marker_body_to_index.get(body_name, body_id % len(self.vis_sphere_marker_names))
                        for body_id, body_name in enumerate(self._body_list)
                    ],
                    dtype=torch.long,
                    device=pos.device,
                )
            else:
                marker_indices = torch.arange(pos.shape[0], device=pos.device) % len(self.vis_sphere_marker_names)
        self.vis_spheres.visualize(pos, rot, scales, marker_indices)

    def draw_line(self, start_point, end_point, color, env_id):
        # import ipdb; ipdb.set_trace()
        start_point_list = [(   start_point.x.item(), start_point.y.item(), start_point.z.item())]
        end_point_list = [(end_point.x.item(), end_point.y.item(), end_point.z.item())]
        color_list = [(color.x, color.y, color.z, 1.0)]
        sizes = [1]
        self.draw.draw_lines(start_point_list, end_point_list, color_list, sizes)
        
    def setup_rendering_cameras(self):
        self.rendering_cam = TiledCamera(TiledCameraCfg(
                prim_path="/World/envs/env_.*/Camera_ego",
                offset=TiledCameraCfg.OffsetCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), convention="world"),
                data_types=['rgb'],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=5.0, focus_distance=50.0, horizontal_aperture=5, clipping_range=(0.1, 20.0)
                ),
                colorize_semantic_segmentation=False,
                width=512,
                height=512,
            ))
        self.scene.sensors["rendering_cam"] = self.rendering_cam

    
