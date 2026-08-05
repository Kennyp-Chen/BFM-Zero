"""Frozen ONNX decoder adapter for the 22DoF PiPlus H0W BFM export."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch.nn import functional


class OnnxPiPlusH0WDecoder:
    """Run the exported FBcprAux model with raw HumanoidVerse observations.

    The supplied export fixes its batch dimension at one even though all of
    its operations support batching. The adapter changes only the input and
    output batch annotations in memory, leaving the checkpoint on disk intact.
    """

    action_dim = 22
    z_dim = 256
    state_dim = 50
    history_dim = 288
    actor_observation_dim = state_dim + action_dim + history_dim + z_dim

    def __init__(self, decoder_path: str | Path, device: torch.device) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX decoder requires onnxruntime in the active environment") from exc

        self.decoder_path = Path(decoder_path).expanduser().resolve()
        if not self.decoder_path.is_file():
            raise FileNotFoundError(f"ONNX decoder does not exist: {self.decoder_path}")
        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if device.type == "cuda" and "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError("ONNX decoder requires the onnx package to enable batched inference") from exc

        model = onnx.load(str(self.decoder_path))
        for value_info in (*model.graph.input, *model.graph.output):
            batch_dim = value_info.type.tensor_type.shape.dim[0]
            batch_dim.ClearField("dim_value")
            batch_dim.dim_param = "batch"
        onnx.checker.check_model(model)
        self.session = ort.InferenceSession(model.SerializeToString(), providers=providers)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("PiPlus H0W ONNX decoder must have one input and one output")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        input_shape = inputs[0].shape
        output_shape = outputs[0].shape
        if input_shape[-1] != self.actor_observation_dim:
            raise ValueError(
                f"ONNX decoder expects {input_shape[-1]} actor-observation values, "
                f"expected {self.actor_observation_dim}"
            )
        if output_shape[-1] != self.action_dim:
            raise ValueError(f"ONNX decoder produces {output_shape[-1]} actions, expected {self.action_dim}")

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        return functional.normalize(z, dim=-1).mul(float(self.z_dim) ** 0.5)

    def act(self, observation: Mapping[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        del mean
        state = observation["state"]
        last_action = observation["last_action"]
        history_actor = observation["history_actor"]
        expected = (self.state_dim, self.action_dim, self.history_dim, self.z_dim)
        actual = (state.shape[-1], last_action.shape[-1], history_actor.shape[-1], z.shape[-1])
        if actual != expected:
            raise ValueError(f"Unexpected PiPlus H0W decoder input dimensions: got {actual}, expected {expected}")
        if not (state.shape[0] == last_action.shape[0] == history_actor.shape[0] == z.shape[0]):
            raise ValueError("PiPlus H0W decoder inputs must share a batch dimension")

        actor_obs = torch.cat((state, last_action, history_actor, z), dim=-1)
        actor_obs_np = actor_obs.detach().to(device="cpu", dtype=torch.float32).numpy()
        actions = self.session.run([self.output_name], {self.input_name: actor_obs_np})[0]
        return torch.as_tensor(actions, device=z.device, dtype=torch.float32)


def load_decoder(_bfm_model_path: Path, decoder_path: Path, device: torch.device) -> OnnxPiPlusH0WDecoder:
    """Factory consumed by :mod:`humanoidverse.speed_stage2`."""
    return OnnxPiPlusH0WDecoder(decoder_path, device)
