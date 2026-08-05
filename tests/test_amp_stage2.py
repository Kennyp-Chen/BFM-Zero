import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from humanoidverse.amp_stage2 import (
    MIMICLITE_LOCOMOTION_WEIGHTS,
    AMPDiscriminator,
    CommandEncoderPolicy,
    MimicLiteLocomotionRewardState,
    _latent_direction_prior,
    _load_piplus_robot_contract,
    _motion_qpos,
    _policy_dof_from_motion,
    _sample_commands,
    _stage2_prepare_pre_reset_transition,
    _stage2_update_reset_buf,
    _torchrun_command,
    _transition_core,
    amp_quadratic_reward,
    baseline_normalized_linvel_reward,
    command_tracking_metrics,
    compute_gae,
    encoder_input_scale,
    flatten_encoder_observation,
    latent_manifold_metrics,
    load_command_encoder_policy_state,
    ppo_update,
    project_latent,
    restore_discriminator_optimizer,
    restore_policy_optimizer,
)
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase


class AmpStage2Test(unittest.TestCase):
    def test_latent_direction_prior_prefers_matching_expert_direction(self):
        reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        projected = torch.tensor([[2.0, 0.0], [-1.0, 0.0]])
        penalty, cosine = _latent_direction_prior(projected, reference)
        self.assertAlmostEqual(float(cosine[0]), 1.0, places=6)
        self.assertAlmostEqual(float(penalty[0]), 0.0, places=6)
        self.assertAlmostEqual(float(cosine[1]), 0.0, places=6)
        self.assertAlmostEqual(float(penalty[1]), 1.0, places=6)

    def test_latent_manifold_metrics_include_command_bins(self):
        projected = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        )
        commands = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.7, 0.0]]
        )
        metrics = latent_manifold_metrics(projected, commands, projected, max_samples=8)
        for name in ("all", "stand", "turn", "slow", "medium", "fast"):
            self.assertIn(f"latent/{name}_nearest_cosine", metrics)
            self.assertIn(f"latent/{name}_mmd_rbf", metrics)
        self.assertEqual(metrics["latent/stand_count"], 1.0)
        self.assertEqual(metrics["latent/turn_count"], 1.0)
        self.assertEqual(metrics["latent/slow_count"], 1.0)
        self.assertEqual(metrics["latent/medium_count"], 1.0)
        self.assertEqual(metrics["latent/fast_count"], 2.0)

    def test_piplus_robot_and_motion_joint_contract(self):
        contract = _load_piplus_robot_contract(
            "humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H1W_LSE.yaml"
        )
        joint_names = contract.policy_joint_names
        frame = np.arange(len(joint_names), dtype=np.float64)
        motion = {
            "root_trans_offset": np.zeros((1, 3), dtype=np.float64),
            "root_rot": np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            "dof": frame[None],
            "joint_names": list(reversed(joint_names)),
        }
        motion["dof"] = motion["dof"][:, ::-1]
        policy_dof = _policy_dof_from_motion(motion, joint_names)
        model = mujoco.MjModel.from_xml_path(str(contract.robot.xml_path))
        qpos = _motion_qpos(motion, joint_names, policy_dof, 0, model)

        self.assertEqual(len(joint_names), 23)
        self.assertEqual(policy_dof.shape, (1, 23))
        for index, joint_name in enumerate(joint_names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.assertEqual(qpos[int(model.jnt_qposadr[joint_id])], float(index))

    def test_command_encoder_shapes_and_latent_projection(self):
        policy = CommandEncoderPolicy(input_dim=19, z_dim=8, hidden_dim=32)
        raw_z, log_prob, value = policy.sample(torch.randn(5, 19))

        self.assertEqual(raw_z.shape, (5, 8))
        self.assertEqual(log_prob.shape, (5,))
        self.assertEqual(value.shape, (5,))
        projected = project_latent(raw_z)
        self.assertTrue(torch.allclose(projected.norm(dim=-1), torch.full((5,), 8.0**0.5), atol=1.0e-5))

    def test_ppo_update_early_stops_and_reports_averaged_metrics(self):
        torch.manual_seed(3)
        policy = CommandEncoderPolicy(input_dim=5, z_dim=4, hidden_dim=16)
        features = torch.randn(8, 5)
        raw_z, log_prob, values = policy.sample(features)
        rollout = SimpleNamespace(
            encoder_features=features.reshape(2, 4, 5),
            raw_z=raw_z.reshape(2, 4, 4),
            old_log_prob=(log_prob + 1.0).reshape(2, 4),
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-4)
        metrics = ppo_update(
            policy,
            rollout,
            torch.ones(2, 4),
            values.detach().reshape(2, 4),
            epochs=4,
            minibatch_size=8,
            clip_ratio=0.2,
            value_coef=0.5,
            entropy_coef=0.001,
            optimizer=optimizer,
            max_grad_norm=1.0,
            target_kl=0.01,
        )
        self.assertEqual(metrics["ppo_updates"], 1.0)
        self.assertEqual(metrics["ppo_expected_updates"], 4.0)
        self.assertEqual(metrics["ppo_update_fraction"], 0.25)
        self.assertEqual(metrics["ppo_early_stop"], 1.0)
        self.assertTrue(np.isfinite(metrics["ratio_max"]))
        self.assertGreaterEqual(metrics["clip_fraction"], 0.0)

    def test_encoder_input_scaling_and_legacy_policy_migration_are_equivalent(self):
        torch.manual_seed(4)
        obs = {
            "state": torch.randn(3, 10),
            "last_action": torch.randn(3, 2),
            "history_actor": torch.randn(3, 48),
        }
        commands = torch.randn(3, 3)
        raw_features = torch.cat((commands, obs["state"], obs["last_action"], obs["history_actor"]), dim=-1)
        scaled_features = flatten_encoder_observation(obs, commands)
        scale = encoder_input_scale(obs, commands)
        self.assertTrue(torch.allclose(scaled_features, raw_features * scale))
        self.assertTrue(torch.equal(scale[5:7], torch.full((2,), 0.05)))
        self.assertEqual(int((scale == 0.05).sum()), 10)

        legacy_policy = CommandEncoderPolicy(raw_features.shape[-1], z_dim=4, hidden_dim=16)
        expected = legacy_policy(raw_features)
        migrated_policy = CommandEncoderPolicy(raw_features.shape[-1], z_dim=4, hidden_dim=16)
        checkpoint = {"policy": legacy_policy.state_dict(), "metadata": {}}
        migrated = load_command_encoder_policy_state(migrated_policy, checkpoint, scale)
        actual = migrated_policy(scaled_features)
        self.assertTrue(migrated)
        for expected_value, actual_value in zip(expected, actual, strict=True):
            self.assertTrue(torch.allclose(expected_value, actual_value, atol=1.0e-6))

    def test_optimizer_resume_overrides_checkpoint_learning_rate(self):
        policy = CommandEncoderPolicy(input_dim=5, z_dim=4, hidden_dim=16)
        old_optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-5)
        checkpoint = {"policy_optimizer": old_optimizer.state_dict()}
        optimizer = torch.optim.Adam(policy.parameters(), lr=9.0e-4)
        status = restore_policy_optimizer(
            optimizer,
            checkpoint,
            learning_rate=1.0e-4,
            reset_for_input_migration=False,
        )
        self.assertEqual(status, "loaded_with_lr_override")
        self.assertTrue(all(group["lr"] == 1.0e-4 for group in optimizer.param_groups))

    def test_discriminator_optimizer_resume_overrides_checkpoint_learning_rate(self):
        discriminator = AMPDiscriminator(feature_dim=5, hidden_dims=(16, 8))
        old_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2.0e-4)
        checkpoint = {"discriminator_optimizer": old_optimizer.state_dict()}
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=9.0e-4)
        status = restore_discriminator_optimizer(optimizer, checkpoint, learning_rate=1.0e-4)
        self.assertEqual(status, "loaded_with_lr_override")
        self.assertTrue(all(group["lr"] == 1.0e-4 for group in optimizer.param_groups))

    def test_tracking_rewards_are_positive_and_dense(self):
        commands = torch.tensor([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
        achieved = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [-0.4, 0.0, 0.0]])
        reward = baseline_normalized_linvel_reward(achieved, commands)
        self.assertGreater(float(reward[0]), 0.0)
        self.assertAlmostEqual(float(reward[1]), 1.0, places=6)
        self.assertGreater(float(reward[2]), 0.0)

    def test_amp_quadratic_reward_is_positive_and_bounded(self):
        reward = amp_quadratic_reward(torch.tensor([-1.0, 1.0, 3.0]))
        self.assertTrue(torch.all(reward >= 0.0))
        self.assertTrue(torch.all(reward <= 1.0))
        self.assertAlmostEqual(float(reward[1]), 1.0, places=6)
        self.assertAlmostEqual(float(reward[0]), 0.0, places=6)
        self.assertAlmostEqual(float(reward[2]), 0.0, places=6)

    def test_tracking_metrics_report_nonzero_response_slope(self):
        commands = torch.tensor([[[0.2, 0.0, -0.4], [0.4, 0.0, 0.2], [0.6, 0.0, 0.8]]])
        base_lin_vel = torch.tensor([[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]])
        base_ang_vel = torch.tensor([[[0.0, 0.0, -0.2], [0.0, 0.0, 0.1], [0.0, 0.0, 0.4]]])
        metrics = command_tracking_metrics(commands, base_lin_vel, base_ang_vel)
        self.assertAlmostEqual(metrics["tracking/nonzero_vx_response_slope"], 0.5, places=6)
        self.assertAlmostEqual(metrics["tracking/nonzero_vx_command_correlation"], 1.0, places=6)
        self.assertAlmostEqual(metrics["tracking/nonzero_vx_mae"], 0.2, places=6)
        self.assertAlmostEqual(metrics["tracking/nonzero_yaw_response_slope"], 0.5, places=6)
        self.assertAlmostEqual(metrics["tracking/nonzero_yaw_command_correlation"], 1.0, places=6)
        self.assertAlmostEqual(metrics["tracking/nonzero_yaw_rate_mae"], 0.2333333, places=6)

    def test_amp_discriminator_wgan_gp_is_finite(self):
        discriminator = AMPDiscriminator(feature_dim=12, hidden_dims=(32, 16))
        losses = discriminator.loss(torch.randn(7, 12), torch.randn(7, 12))
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in discriminator.parameters()))

    def test_gae_distinguishes_terminal_and_timeout(self):
        rewards = torch.tensor([[1.0], [2.0], [3.0]])
        values = torch.zeros_like(rewards)
        terminated = torch.tensor([[False], [True], [False]])
        truncated = torch.zeros_like(terminated)
        advantages, returns = compute_gae(
            rewards,
            values,
            terminated,
            truncated,
            torch.zeros_like(rewards),
            torch.tensor([4.0]),
            discount=0.9,
            gae_lambda=1.0,
        )
        expected = torch.tensor([[2.8], [2.0], [6.6]])
        self.assertTrue(torch.allclose(advantages, expected))
        self.assertTrue(torch.allclose(returns, expected))

    def test_transition_core_maps_target_isaac_tensors(self):
        simulator = SimpleNamespace(
            robot_root_states=torch.zeros(2, 13),
            _rigid_body_pos=torch.zeros(2, 4, 3),
            _rigid_body_rot=torch.zeros(2, 4, 4),
            _rigid_body_ang_vel=torch.zeros(2, 4, 3),
            contact_forces=torch.zeros(2, 4, 3),
            dof_pos=torch.zeros(2, 23),
            dof_vel=torch.zeros(2, 23),
        )
        live = SimpleNamespace(
            simulator=simulator,
            base_quat=torch.zeros(2, 4),
            base_lin_vel=torch.zeros(2, 3),
            base_ang_vel=torch.zeros(2, 3),
            torques=torch.zeros(2, 23),
            default_dof_pos=torch.zeros(23),
            default_dof_pos_offset=torch.zeros(2, 23),
            num_envs=2,
            device=torch.device("cpu"),
        )
        core = _transition_core(live, {})
        self.assertIs(core.dof_pos, simulator.dof_pos)
        self.assertIs(core.body_pos, simulator._rigid_body_pos)

    def test_transition_core_uses_pre_reset_snapshot(self):
        live = SimpleNamespace(num_envs=2, device="cpu", dof_pos=torch.full((2, 1), 9.0))
        snapshot = {"dof_pos": torch.tensor([[1.0], [2.0]])}
        transition = _transition_core(live, {"terminal_state": snapshot})
        self.assertTrue(torch.equal(transition.dof_pos, snapshot["dof_pos"]))

    def test_stage2_termination_matches_ufo(self):
        core = SimpleNamespace(
            num_envs=2,
            device=torch.device("cpu"),
            reset_buf=torch.zeros(2, dtype=torch.bool),
            extras={},
            termination_contact_indices=torch.tensor([1]),
            simulator=SimpleNamespace(
                contact_forces=torch.tensor(
                    [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
                )
            ),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0], [0.91, 0.0, -0.4]]),
        )
        _stage2_update_reset_buf(core)
        self.assertTrue(torch.equal(core.reset_buf, torch.tensor([True, True])))
        self.assertTrue(torch.equal(core.extras["termination_crash"], torch.tensor([True, False])))
        self.assertTrue(torch.equal(core.extras["termination_fall_over"], torch.tensor([False, True])))

    def test_stage2_captures_terminal_state_and_timeout_observation(self):
        simulator = SimpleNamespace(
            robot_root_states=torch.ones(2, 13),
            _rigid_body_pos=torch.ones(2, 3, 3),
            _rigid_body_rot=torch.ones(2, 3, 4),
            _rigid_body_ang_vel=torch.ones(2, 3, 3),
            contact_forces=torch.ones(2, 3, 3),
            dof_pos=torch.ones(2, 2),
            dof_vel=torch.ones(2, 2),
        )
        core = SimpleNamespace(
            simulator=simulator,
            base_quat=torch.ones(2, 4),
            base_lin_vel=torch.ones(2, 3),
            base_ang_vel=torch.ones(2, 3),
            projected_gravity=torch.ones(2, 3),
            torques=torch.ones(2, 2),
            default_dof_pos=torch.ones(2, 2),
            default_dof_pos_offset=torch.ones(2, 2),
            time_out_buf=torch.tensor([True, False]),
            extras={},
        )
        core._compute_observations = lambda: setattr(
            core,
            "obs_buf_dict_raw",
            {
                "actor_obs": {
                    "dof_pos": torch.ones(2, 2),
                    "dof_vel": torch.ones(2, 2),
                    "projected_gravity": torch.ones(2, 3),
                    "base_ang_vel": torch.ones(2, 3),
                    "actions": torch.ones(2, 2),
                    "max_local_self": torch.ones(2, 5),
                    "history_actor": torch.ones(2, 7),
                }
            },
        )
        _stage2_prepare_pre_reset_transition(core, torch.tensor([0]))
        self.assertIn("terminal_state", core.extras)
        self.assertIn("terminal_observation", core.extras)
        self.assertEqual(core.extras["terminal_observation"]["state"].shape, (2, 10))

    def test_survival_reward_is_available_for_stage2_override(self):
        core = SimpleNamespace(num_envs=3, device=torch.device("cpu"))
        survival = LeggedRobotBase._reward_survival(core)
        self.assertTrue(torch.equal(survival, torch.ones(3)))

    def test_mimiclite_reward_terms_are_stable(self):
        state = MimicLiteLocomotionRewardState(
            num_envs=1,
            num_dof=2,
            dt=0.02,
            feet_indices=torch.tensor([0, 1]),
            torso_index=0,
            joint_vel_indices=torch.tensor([0]),
            joint_deviation_indices=torch.tensor([0]),
            device=torch.device("cpu"),
        )
        core = SimpleNamespace(
            num_envs=1,
            device=torch.device("cpu"),
            base_lin_vel=torch.zeros(1, 3),
            base_ang_vel=torch.zeros(1, 3),
            body_ang_vel=torch.zeros(1, 1, 3),
            contact_forces=torch.zeros(1, 2, 3),
            body_pos=torch.tensor([[[0.0, 0.0, 0.35], [0.0, 0.0, 0.10]]]),
            body_rot=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
            torques=torch.zeros(1, 2),
            dof_vel=torch.zeros(1, 2),
            dof_pos=torch.zeros(1, 2),
            default_dof_pos=torch.zeros(1, 2),
            default_dof_pos_offset=torch.zeros(1, 2),
        )
        reward, components = state.compute(core, torch.zeros(1, 3), torch.zeros(1, 2), torch.zeros(1, dtype=torch.bool))
        self.assertEqual(set(components), set(MIMICLITE_LOCOMOTION_WEIGHTS))
        self.assertTrue(torch.isfinite(reward).all())

    def test_stand_still_reward_prefers_default_pose_and_is_command_gated(self):
        state = MimicLiteLocomotionRewardState(
            num_envs=1,
            num_dof=2,
            dt=0.02,
            feet_indices=torch.tensor([0, 1]),
            torso_index=0,
            joint_vel_indices=torch.tensor([0]),
            joint_deviation_indices=torch.tensor([0]),
            device=torch.device("cpu"),
        )
        core = SimpleNamespace(
            num_envs=1,
            device=torch.device("cpu"),
            base_lin_vel=torch.zeros(1, 3),
            base_ang_vel=torch.zeros(1, 3),
            body_ang_vel=torch.zeros(1, 1, 3),
            contact_forces=torch.zeros(1, 2, 3),
            body_pos=torch.tensor([[[0.0, 0.0, 0.35], [0.0, 0.0, 0.10]]]),
            body_rot=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
            torques=torch.zeros(1, 2),
            dof_vel=torch.zeros(1, 2),
            dof_pos=torch.zeros(1, 2),
            default_dof_pos=torch.zeros(1, 2),
            default_dof_pos_offset=torch.zeros(1, 2),
        )
        done = torch.zeros(1, dtype=torch.bool)
        actions = torch.zeros(1, 2)
        _, default_components = state.compute(core, torch.zeros(1, 3), actions, done)
        core.dof_pos.fill_(0.4)
        _, crooked_components = state.compute(core, torch.zeros(1, 3), actions, done)
        _, moving_components = state.compute(core, torch.tensor([[0.4, 0.0, 0.0]]), actions, done)

        self.assertEqual(default_components["stand_still"].item(), 0.0)
        self.assertAlmostEqual(crooked_components["stand_still"].item(), -0.02 * 0.8 * 0.8)
        self.assertEqual(moving_components["stand_still"].item(), 0.0)

    def test_command_sampler_supports_standing(self):
        low = torch.tensor([-0.5, -0.2, -0.8])
        high = torch.tensor([1.2, 0.2, 0.8])
        commands = _sample_commands(32, torch.device("cpu"), low, high, stand_prob=1.0)
        self.assertTrue(torch.equal(commands, torch.zeros_like(commands)))

    def test_command_sampler_supports_turning_in_place(self):
        low = torch.tensor([-0.8, -0.5, 0.5])
        high = torch.tensor([0.8, 0.5, 0.5])
        commands = _sample_commands(32, torch.device("cpu"), low, high, stand_prob=0.0, turn_prob=1.0)
        self.assertTrue(torch.equal(commands[:, :2], torch.zeros_like(commands[:, :2])))
        self.assertTrue(torch.equal(commands[:, 2], torch.full((32,), 0.5)))

    def test_torchrun_command_uses_standard_pytorch_launcher(self):
        command = _torchrun_command(4, ["--smoke", "--gpu-ids", "all"])
        self.assertEqual(command[1:4], ["-m", "torch.distributed.run", "--standalone"])
        self.assertIn("--nproc_per_node=4", command)
        self.assertEqual(command[-3:], ["--smoke", "--gpu-ids", "all"])


if __name__ == "__main__":
    unittest.main()
