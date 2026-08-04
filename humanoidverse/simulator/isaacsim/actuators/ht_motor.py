from __future__ import annotations

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class HTMotor(DelayedPDActuator):
    """HT motor actuator with an identified torque-speed curve."""

    cfg: HTMotorCfg

    def __init__(self, cfg: HTMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._curve_a = cfg.curve_param_a
        self._curve_b = cfg.curve_param_b
        self._curve_c = cfg.curve_param_c
        self._max_torque = cfg.max_torque
        self._max_velocity = cfg.max_velocity
        self._saturation_effort = self._max_torque

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        self.computed_effort = (
            self.stiffness * error_pos
            + self.damping * error_vel
            + control_action.joint_efforts
        )
        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        abs_vel = torch.abs(self._joint_vel)
        abs_effort = torch.abs(effort)
        is_motoring = (effort * self._joint_vel) >= 0.0

        a = -self._curve_a
        b = -self._curve_b
        c = abs_vel - self._curve_c
        discriminant = torch.clamp(b * b - 4 * a * c, min=0.0)
        max_torque_motoring = (-b + torch.sqrt(discriminant)) / (2 * a)
        max_torque_motoring = torch.clamp(max_torque_motoring, min=0.0, max=self._max_torque)
        max_torque_motoring = torch.where(
            abs_vel > self._max_velocity,
            torch.zeros_like(max_torque_motoring),
            max_torque_motoring,
        )

        max_torque_braking = torch.full_like(abs_vel, self._max_torque)
        max_torque = torch.where(is_motoring, max_torque_motoring, max_torque_braking)
        return torch.sign(effort) * torch.clamp(abs_effort, max=max_torque)


@configclass
class HTMotorCfg(DelayedPDActuatorCfg):
    class_type: type = HTMotor

    curve_param_a: float = -0.0141
    curve_param_b: float = -0.0709
    curve_param_c: float = 6.2756
    max_torque: float = 20.0
    max_velocity: float = 6.0
    saturation_effort: float = 20.0


@configclass
class HTMotorCfg_5047(HTMotorCfg):
    curve_param_a: float = -0.0141
    curve_param_b: float = -0.0709
    curve_param_c: float = 6.2756
    max_torque: float = 18.7
    max_velocity: float = 6.28
    saturation_effort: float = 18.7


@configclass
class HTMotorCfg_5031(HTMotorCfg):
    curve_param_a: float = -0.0141
    curve_param_b: float = -0.0709
    curve_param_c: float = 6.2756
    max_torque: float = 20.0
    max_velocity: float = 6.0
    saturation_effort: float = 20.0


@configclass
class HTMotorCfg_5036(HTMotorCfg):
    curve_param_a: float = -0.006667
    curve_param_b: float = -0.113990
    curve_param_c: float = 7.732552
    max_torque: float = 23.7
    max_velocity: float = 7.95
    saturation_effort: float = 23.7


@configclass
class HTMotorCfg_4438(HTMotorCfg):
    curve_param_a: float = -0.128416
    curve_param_b: float = -0.699618
    curve_param_c: float = 19.833274
    max_torque: float = 10.0
    max_velocity: float = 20.0
    saturation_effort: float = 10.0

@configclass
class HTMotorCfg_6036(HTMotorCfg):
    curve_param_a: float = -0.004215367
    curve_param_b: float = -0.045892325
    curve_param_c: float = 7.209663332
    max_torque: float = 36.5
    max_velocity: float = 7.435
    saturation_effort: float = 36.5


@configclass
class HTMotor40VCfg_3536(HTMotorCfg):
    curve_param_a: float = 2.860840068
    curve_param_b: float = -22.221680648
    curve_param_c: float = 41.753409187
    max_torque: float = 3.3
    max_velocity: float = 37.18
    saturation_effort: float = 3.3


@configclass
class HTMotor40VCfg_4438(HTMotorCfg):
    curve_param_a: float = -0.184120895
    curve_param_b: float = -0.637858724
    curve_param_c: float = 24.600869510
    max_torque: float = 10.2
    max_velocity: float = 24.6
    saturation_effort: float = 10.2


@configclass
class HTMotor40VCfg_5036(HTMotorCfg):
    curve_param_a: float = -0.021735007
    curve_param_b: float = -0.030980508
    curve_param_c: float = 14.063931256
    max_torque: float = 22.3
    max_velocity: float = 14.45
    saturation_effort: float = 22.3
