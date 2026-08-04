import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from humanoidverse.agents.envs.humanoidverse_isaac import build_default_pose_target
from humanoidverse.amp_stage2_play import command_from_axes, latest_stage2_checkpoint, resolve_play_device


class AmpStage2PlayTest(unittest.TestCase):
    def test_latest_checkpoint_uses_numeric_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "checkpoint_9.pt").touch()
            (folder / "checkpoint_100.pt").touch()
            self.assertEqual(latest_stage2_checkpoint(folder).name, "checkpoint_100.pt")

    def test_command_axes_apply_asymmetric_limits(self):
        command = command_from_axes(
            {0: 0.5, 1: -1.0, 3: -0.5},
            np.asarray([-0.5, -0.2, -0.8], dtype=np.float32),
            np.asarray([1.2, 0.2, 0.8], dtype=np.float32),
            forward_axis=1,
            lateral_axis=0,
            yaw_axis=3,
            forward_sign=-1.0,
            lateral_sign=-1.0,
            yaw_sign=-1.0,
            deadzone=0.1,
        )
        self.assertAlmostEqual(float(command[0]), 1.2, places=5)
        self.assertAlmostEqual(float(command[1]), -0.2 * (0.4 / 0.9), places=5)
        self.assertAlmostEqual(float(command[2]), 0.8 * (0.4 / 0.9), places=5)

    def test_gamepad_mapping_matches_ht_lab_hi(self):
        command = command_from_axes(
            {0: -0.5, 1: -1.0, 3: 0.5},
            np.asarray([-0.5, -0.2, -0.8], dtype=np.float32),
            np.asarray([1.2, 0.2, 0.8], dtype=np.float32),
            forward_axis=1,
            lateral_axis=0,
            yaw_axis=3,
            forward_sign=-1.0,
            lateral_sign=-1.0,
            yaw_sign=-1.0,
            deadzone=0.08,
        )
        # HT_lab_hi: push left stick up => +vx, left/right => +/-vy, right stick left => +wz.
        self.assertAlmostEqual(float(command[0]), 1.2, places=5)
        self.assertGreater(float(command[1]), 0.0)
        self.assertLess(float(command[2]), 0.0)

    def test_gamepad_custom_axis_indices_are_supported(self):
        command = command_from_axes(
            {4: -1.0, 5: 0.5, 6: -0.5},
            np.asarray([-0.5, -0.2, -0.8], dtype=np.float32),
            np.asarray([1.2, 0.2, 0.8], dtype=np.float32),
            forward_axis=5,
            lateral_axis=4,
            yaw_axis=6,
            forward_sign=-1.0,
            lateral_sign=-1.0,
            yaw_sign=-1.0,
            deadzone=0.08,
        )
        self.assertAlmostEqual(float(command[0]), -0.5 * (0.42 / 0.92), places=5)
        self.assertAlmostEqual(float(command[1]), 0.2, places=5)
        self.assertAlmostEqual(float(command[2]), 0.8 * (0.42 / 0.92), places=5)

    def test_play_device_cpu_is_explicit(self):
        self.assertEqual(resolve_play_device("cpu"), torch.device("cpu"))

    def test_play_device_cuda_gets_default_index_when_available(self):
        if torch.cuda.is_available():
            self.assertEqual(resolve_play_device("cuda"), torch.device("cuda:0"))

    def test_default_pose_target_preserves_joint_order_and_velocity_zero(self):
        default = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32)
        target = build_default_pose_target(default, num_envs=2)
        self.assertEqual(tuple(target.shape), (2, 3, 2))
        torch.testing.assert_close(target[:, :, 0], default.expand(2, -1))
        torch.testing.assert_close(target[:, :, 1], torch.zeros(2, 3))


if __name__ == "__main__":
    unittest.main()
