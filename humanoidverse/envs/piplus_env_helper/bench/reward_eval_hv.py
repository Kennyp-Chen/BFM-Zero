import dataclasses
from typing import Any

import mujoco
import numpy as np
import torch

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.wrappers.humenvbench import BaseHumEnvBenchWrapper, get_next
from humanoidverse.envs.piplus_env_helper.rewards import make_from_name


def _set_ctrl(data: mujoco.MjData, action: np.ndarray) -> None:
    data.ctrl[:] = 0.0
    if action.shape[0] == data.ctrl.shape[0]:
        data.ctrl[:] = action
    elif action.shape[0] < data.ctrl.shape[0]:
        data.ctrl[-action.shape[0] :] = action
    else:
        data.ctrl[:] = action[-data.ctrl.shape[0] :]


def _relabel_worker(x, model: mujoco.MjModel, reward_fn):
    qpos, qvel, action = x
    rewards = np.zeros((qpos.shape[0], 1))
    data = mujoco.MjData(model)
    for i in range(qpos.shape[0]):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        _set_ctrl(data, action[i])
        mujoco.mj_forward(model, data)
        rewards[i] = reward_fn.compute(model, data)
    return rewards


def relabel(model, qpos: np.ndarray, qvel: np.ndarray, action: np.ndarray, reward_fn, max_workers: int = 5):
    from concurrent.futures import ThreadPoolExecutor
    import functools

    chunk_size = int(np.ceil(qpos.shape[0] / max_workers))
    args = [(qpos[i : i + chunk_size], qvel[i : i + chunk_size], action[i : i + chunk_size]) for i in range(0, qpos.shape[0], chunk_size)]
    if max_workers == 1:
        result = [_relabel_worker(args[0], model=model, reward_fn=reward_fn)]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            f = functools.partial(_relabel_worker, model=model, reward_fn=reward_fn)
            result = exe.map(f, args)
    return np.concatenate([r for r in result])


@dataclasses.dataclass(kw_only=True)
class PiPlusRewardWrapperHV(BaseHumEnvBenchWrapper):
    inference_dataset: Any
    num_samples_per_inference: int
    inference_function: str
    max_workers: int
    env_model: str

    def reward_inference(self, task: str) -> torch.Tensor:
        if isinstance(self.env_model, str):
            self.env_model = mujoco.MjModel.from_xml_path(self.env_model)

        if isinstance(self.inference_dataset, TrajectoryDictBufferMultiDim):
            if "qpos" not in self.inference_dataset.output_key_tp1:
                self.inference_dataset.output_key_tp1.append("qpos")
            if "qvel" not in self.inference_dataset.output_key_tp1:
                self.inference_dataset.output_key_tp1.append("qvel")

        if self.num_samples_per_inference >= self.inference_dataset.size() and hasattr(self.inference_dataset, "get_full_buffer"):
            data = self.inference_dataset.get_full_buffer()
        else:
            data = self.inference_dataset.sample(self.num_samples_per_inference)

        qpos = get_next("qpos", data)
        qvel = get_next("qvel", data)
        action = data["action"]
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.cpu().detach().numpy()
            qvel = qvel.cpu().detach().numpy()
            action = action.cpu().detach().numpy()

        rewards = relabel(
            self.env_model,
            qpos,
            qvel,
            action,
            make_from_name(task),
            max_workers=self.max_workers,
        )

        td = {"reward": torch.tensor(rewards, dtype=torch.float32, device=self.device)}
        if "B" in data:
            td["B_vect"] = data["B"]
        else:
            td["next_obs"] = get_next("observation", data)
        inference_fn = getattr(self.model, self.inference_function, None)
        return inference_fn(**td).reshape(1, -1)

