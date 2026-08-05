import tempfile
import unittest
from pathlib import Path

import torch

from humanoidverse.speed_stage2_play import latest_speed_checkpoint, policy_shape_from_state


class SpeedStage2PlayTest(unittest.TestCase):
    def test_latest_checkpoint_uses_numeric_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "checkpoint_9.pt").touch()
            (folder / "checkpoint_100.pt").touch()
            self.assertEqual(latest_speed_checkpoint(folder).name, "checkpoint_100.pt")

    def test_policy_shape_comes_from_checkpoint(self):
        policy_state = {
            "trunk.0.weight": torch.zeros(256, 363),
            "latent_mean.weight": torch.zeros(256, 256),
        }
        self.assertEqual(policy_shape_from_state(policy_state), (363, 256, 256))


if __name__ == "__main__":
    unittest.main()
