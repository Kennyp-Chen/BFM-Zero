import inspect
import sys

import numpy as np

from humanoidverse.envs.g1_env_helper import rewards as g1_rewards


BODY_NAME_ALIASES = {
    "pelvis": "base_link",
    "left_wrist_roll_link": "l_wrist_link",
    "right_wrist_roll_link": "r_wrist_link",
    "left_knee_link": "l_calf_link",
    "right_knee_link": "r_calf_link",
}

SENSOR_NAME_ALIASES = {
    "imu-angular-velocity": "angular-velocity",
}


def body_name(name: str) -> str:
    return BODY_NAME_ALIASES.get(name, name)


def sensor_name(name: str) -> str:
    return SENSOR_NAME_ALIASES.get(name, name)


def get_xpos(model, data, name: str):
    return g1_rewards.get_xpos(model, data, body_name(name))


def get_xmat(model, data, name: str):
    return g1_rewards.get_xmat(model, data, body_name(name))


def get_sensor_data(model, data, name: str):
    if name == "upvector_torso":
        return data.body("torso_link").xmat.reshape(3, 3)[:, 2].copy()
    return g1_rewards.get_sensor_data(model, data, sensor_name(name))


def get_center_of_mass_linvel(model, data):
    return data.body("torso_link").cvel[3:6].copy()


def upright_reward(data) -> float:
    upvector_torso = get_sensor_data(None, data, "upvector_torso")
    return g1_rewards.rewards.tolerance(
        np.sum(np.square(upvector_torso - np.array([0.0, 0.0, 1.0]))),
        bounds=(0, 0.1),
        margin=3,
        value_at_margin=0,
        sigmoid="linear",
    )


def arm_height_reward(height: float, pose: str) -> float:
    limits = g1_rewards.REWARD_LIMITS[pose]
    reward = g1_rewards.rewards.tolerance(
        height,
        bounds=(limits[0], limits[1]),
        margin=limits[2],
        value_at_margin=0,
        sigmoid="linear",
    )
    return (4 * reward + 1) / 5


def directional_move_reward(velocity_xy: np.ndarray, move_speed: float, move_angle: float | None) -> float:
    com_velocity = np.linalg.norm(velocity_xy)
    move = g1_rewards.rewards.tolerance(
        com_velocity,
        bounds=(move_speed - 0.1 * move_speed, move_speed + 0.1 * move_speed),
        margin=move_speed / 2,
        value_at_margin=0.5,
        sigmoid="gaussian",
    )
    move = (5 * move + 1) / 6
    if np.isclose(com_velocity, 0.0) or move_angle is None:
        return move
    direction = velocity_xy / (com_velocity + 1e-6)
    target_direction = np.array([np.cos(move_angle), np.sin(move_angle)])
    angle_reward = (target_direction.dot(direction) + 1.0) / 2.0
    return move * angle_reward


class PiPlusLocomotionReward(g1_rewards.LocomotionReward):
    def compute(self, model, data) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        move_angle = np.deg2rad(self.move_angle) if self.move_angle is not None else None
        if self.egocentric_target and move_angle is not None:
            move_angle += g1_rewards.rot2eul(get_xmat(model, data, name="pelvis"))[-1]

        if self.stay_low:
            standing = g1_rewards.rewards.tolerance(
                root_height,
                bounds=(self.stand_height * 0.95, self.stand_height * 1.05),
                margin=self.stand_height / 2,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        else:
            standing = g1_rewards.rewards.tolerance(
                root_height,
                bounds=(self.stand_height, float("inf")),
                margin=self.stand_height,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        stand_reward = standing * upright_reward(data)
        if 0 <= self.move_speed <= 0.01:
            dont_move = g1_rewards.rewards.tolerance(center_of_mass_velocity[[0, 1]], margin=0.2).mean()
            dont_rotate = g1_rewards.rewards.tolerance(get_sensor_data(model, data, "imu-angular-velocity"), margin=0.1).mean()
            return stand_reward * dont_move * dont_rotate
        return stand_reward * directional_move_reward(center_of_mass_velocity[[0, 1]], self.move_speed, move_angle)

    @staticmethod
    def reward_from_name(name: str):
        reward = g1_rewards.LocomotionReward.reward_from_name(name)
        return None if reward is None else PiPlusLocomotionReward(**reward.__dict__)


class PiPlusRotationReward(g1_rewards.RotationReward):
    def compute(self, model, data) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        torso_rotation = get_xmat(model, data, name="pelvis")[2, :].ravel()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
        height_reward = g1_rewards.rewards.tolerance(
            pelvis_height,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        direction = np.sign(self.target_ang_velocity)
        targ_av = np.abs(self.target_ang_velocity)
        move = g1_rewards.rewards.tolerance(
            direction * angular_velocity[g1_rewards.COORD_TO_INDEX[self.axis]],
            bounds=(targ_av, targ_av + 5),
            margin=targ_av / 2,
            value_at_margin=0,
            sigmoid="linear",
        )
        aligned = g1_rewards.rewards.tolerance(
            torso_rotation[g1_rewards.COORD_TO_INDEX[self.axis]],
            bounds=g1_rewards.ALIGNMENT_BOUNDS[self.axis],
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )
        return move * height_reward * aligned

    @staticmethod
    def reward_from_name(name: str):
        reward = g1_rewards.RotationReward.reward_from_name(name)
        return None if reward is None else PiPlusRotationReward(**reward.__dict__)


class PiPlusArmsReward(g1_rewards.ArmsReward):
    def compute(self, model, data) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        left_height = data.body(body_name("left_wrist_roll_link")).xpos[-1]
        right_height = data.body(body_name("right_wrist_roll_link")).xpos[-1]
        standing = g1_rewards.rewards.tolerance(
            root_height,
            bounds=(self.stand_height, float("inf")),
            margin=self.stand_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        dont_move = g1_rewards.rewards.tolerance(center_of_mass_velocity, margin=0.2).mean()
        dont_rotate = g1_rewards.rewards.tolerance(get_sensor_data(model, data, "imu-angular-velocity"), margin=0.1).mean()
        return (
            standing
            * upright_reward(data)
            * dont_move
            * dont_rotate
            * arm_height_reward(left_height, self.left_pose)
            * arm_height_reward(right_height, self.right_pose)
        )

    @staticmethod
    def reward_from_name(name: str):
        reward = g1_rewards.ArmsReward.reward_from_name(name)
        return None if reward is None else PiPlusArmsReward(**reward.__dict__)


class PiPlusMoveArmsReward(g1_rewards.MoveArmsReward):
    def compute(self, model, data) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        move_angle = np.deg2rad(self.move_angle) if self.move_angle is not None else None
        if self.egocentric_target and move_angle is not None:
            move_angle += g1_rewards.rot2eul(get_xmat(model, data, name="pelvis"))[-1]

        if self.stay_low:
            standing = g1_rewards.rewards.tolerance(
                root_height,
                bounds=(self.low_height / 2, self.low_height),
                margin=self.low_height / 2,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        else:
            standing = g1_rewards.rewards.tolerance(
                root_height,
                bounds=(self.stand_height, float("inf")),
                margin=self.stand_height,
                value_at_margin=0.01,
                sigmoid="linear",
            )

        left_height = data.body(body_name("left_wrist_roll_link")).xpos[-1]
        right_height = data.body(body_name("right_wrist_roll_link")).xpos[-1]
        arms = arm_height_reward(left_height, self.left_pose) * arm_height_reward(right_height, self.right_pose)
        stand_reward = standing * upright_reward(data)
        if self.move_speed == 0:
            dont_move = g1_rewards.rewards.tolerance(center_of_mass_velocity[[0, 1]], margin=0.2).mean()
            dont_rotate = g1_rewards.rewards.tolerance(get_sensor_data(model, data, "imu-angular-velocity"), margin=0.1).mean()
            return stand_reward * dont_move * dont_rotate * arms
        return stand_reward * directional_move_reward(center_of_mass_velocity[[0, 1]], self.move_speed, move_angle) * arms

    @staticmethod
    def reward_from_name(name: str):
        reward = g1_rewards.MoveArmsReward.reward_from_name(name)
        return None if reward is None else PiPlusMoveArmsReward(**reward.__dict__)


class PiPlusSpinArmsReward(g1_rewards.SpinArmsReward):
    def compute(self, model, data) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        torso_rotation = get_xmat(model, data, name="pelvis")[2, :].ravel()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
        height_reward = g1_rewards.rewards.tolerance(
            pelvis_height,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        direction = np.sign(self.target_ang_velocity)
        targ_av = np.abs(self.target_ang_velocity)
        move = g1_rewards.rewards.tolerance(
            direction * angular_velocity[g1_rewards.COORD_TO_INDEX[self.axis]],
            bounds=(targ_av, targ_av + 5),
            margin=targ_av / 2,
            value_at_margin=0,
            sigmoid="linear",
        )
        aligned = g1_rewards.rewards.tolerance(
            torso_rotation[g1_rewards.COORD_TO_INDEX[self.axis]],
            bounds=g1_rewards.ALIGNMENT_BOUNDS[self.axis],
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )
        left_height = data.body(body_name("left_wrist_roll_link")).xpos[-1]
        right_height = data.body(body_name("right_wrist_roll_link")).xpos[-1]
        arms = arm_height_reward(left_height, self.left_pose) * arm_height_reward(right_height, self.right_pose)
        return move * height_reward * aligned * arms

    @staticmethod
    def reward_from_name(name: str):
        reward = g1_rewards.SpinArmsReward.reward_from_name(name)
        return None if reward is None else PiPlusSpinArmsReward(**reward.__dict__)


class PiPlusSitOnGroundReward(g1_rewards.SitOnGroundReward):
    def compute(self, model, data) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        left_knee_pos = get_xpos(model, data, name="left_knee_link")[-1]
        right_knee_pos = get_xpos(model, data, name="right_knee_link")[-1]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        dont_move = g1_rewards.rewards.tolerance(center_of_mass_velocity, margin=0.5).mean()
        dont_rotate = g1_rewards.rewards.tolerance(get_sensor_data(model, data, "imu-angular-velocity"), margin=0.1).mean()
        pelvis_reward = g1_rewards.rewards.tolerance(
            pelvis_height,
            bounds=(self.pelvis_height_th, self.pelvis_height_th + 0.1),
            sigmoid="linear",
            margin=0.7,
            value_at_margin=0,
        )
        knee_reward = 1
        if self.constrained_knees:
            knee_reward *= g1_rewards.rewards.tolerance(left_knee_pos, bounds=(0, 0.1), sigmoid="linear", margin=0.7, value_at_margin=0)
            knee_reward *= g1_rewards.rewards.tolerance(right_knee_pos, bounds=(0, 0.1), sigmoid="linear", margin=0.7, value_at_margin=0)
        if self.knees_not_on_ground:
            knee_reward *= g1_rewards.rewards.tolerance(left_knee_pos, bounds=(0.2, 1), sigmoid="linear", margin=0.1, value_at_margin=0)
            knee_reward *= g1_rewards.rewards.tolerance(right_knee_pos, bounds=(0.2, 1), sigmoid="linear", margin=0.1, value_at_margin=0)
        return upright_reward(data) * dont_move * dont_rotate * pelvis_reward * (2 * knee_reward + 1) / 3

    @staticmethod
    def reward_from_name(name: str):
        reward = g1_rewards.SitOnGroundReward.reward_from_name(name)
        return None if reward is None else PiPlusSitOnGroundReward(**reward.__dict__)


def make_from_name(name: str | None = None):
    all_rewards = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    for reward_class_name, reward_cls in all_rewards:
        if reward_class_name.startswith("PiPlus") and hasattr(reward_cls, "reward_from_name"):
            reward_obj = reward_cls.reward_from_name(name)
            if reward_obj is not None:
                return reward_obj
    return g1_rewards.make_from_name(name)
