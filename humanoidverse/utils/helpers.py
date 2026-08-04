import os
import copy
import torch
from torch import nn
import numpy as np
import random

from collections.abc import Sequence
from typing import Any, List, Dict
from termcolor import colored
from loguru import logger
from omegaconf import DictConfig, OmegaConf

def select_motion_body_data(env, motion_state: dict[str, torch.Tensor], velocity_multiplier: float = 1.0):
    ref_body_pos = motion_state["rg_pos_t"]
    ref_body_rots = motion_state["rg_rot_t"]
    ref_body_vels = motion_state["body_vel_t"] * velocity_multiplier
    ref_body_angular_vels = motion_state["body_ang_vel_t"] * velocity_multiplier

    motion_body_ids = getattr(env, "motion_body_ids", None)
    if motion_body_ids is not None:
        ref_body_pos = ref_body_pos[:, motion_body_ids]
        ref_body_rots = ref_body_rots[:, motion_body_ids]
        ref_body_vels = ref_body_vels[:, motion_body_ids]
        ref_body_angular_vels = ref_body_angular_vels[:, motion_body_ids]

        motion_extend_body_ids = getattr(env, "motion_extend_body_ids", None)
        if motion_extend_body_ids is not None:
            ref_body_pos = torch.cat([ref_body_pos, motion_state["rg_pos_t"][:, motion_extend_body_ids]], dim=1)
            ref_body_rots = torch.cat([ref_body_rots, motion_state["rg_rot_t"][:, motion_extend_body_ids]], dim=1)
            ref_body_vels = torch.cat(
                [ref_body_vels, motion_state["body_vel_t"][:, motion_extend_body_ids] * velocity_multiplier],
                dim=1,
            )
            ref_body_angular_vels = torch.cat(
                [ref_body_angular_vels, motion_state["body_ang_vel_t"][:, motion_extend_body_ids] * velocity_multiplier],
                dim=1,
            )

    return ref_body_pos, ref_body_rots, ref_body_vels, ref_body_angular_vels

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def pre_process_config(config) -> None:
    
    # compute observation_dim
    # config.robot.policy_obs_dim = -1
    # config.robot.critic_obs_dim = -1
    
    obs_dim_dict = dict()
    _obs_key_list = config.env.config.obs.obs_dict
    _aux_obs_key_list = config.env.config.obs.obs_auxiliary
    
    assert set(config.env.config.obs.noise_scales.keys()) == set(config.env.config.obs.obs_scales.keys())

    # convert obs_dims to list of dicts
    # import ipdb; ipdb.set_trace()
    import omegaconf
    from omegaconf import ListConfig
    if isinstance(config.env.config.obs.obs_dims, ListConfig):
        each_dict_obs_dims = {k: v for d in config.env.config.obs.obs_dims for k, v in d.items()}
        config.env.config.obs.obs_dims = each_dict_obs_dims
    logger.info(f"obs_dims: {config.env.config.obs.obs_dims}")
    auxiliary_obs_dims = {}
    for aux_obs_key, aux_config in _aux_obs_key_list.items():
        auxiliary_obs_dims[aux_obs_key] = 0
        for _key, _num in aux_config.items():
            try:
                assert _key in config.env.config.obs.obs_dims.keys()
                auxiliary_obs_dims[aux_obs_key] += config.env.config.obs.obs_dims[_key] * _num
            except:
                # import ipdb; ipdb.set_trace()
                logger.warning(f"aux_obs_key: {aux_obs_key} not found in obs_dims")
    logger.info(f"auxiliary_obs_dims: {auxiliary_obs_dims}")
    for obs_key, obs_config in _obs_key_list.items():
        obs_dim_dict[obs_key] = 0
        for key in obs_config:
            if key.endswith("_raw"): key = key[:-4]
            if key in config.env.config.obs.obs_dims.keys(): 
                obs_dim_dict[obs_key] += config.env.config.obs.obs_dims[key]
                logger.info(f"{obs_key}: {key} has dim: {config.env.config.obs.obs_dims[key]}")
            else:
                obs_dim_dict[obs_key] += auxiliary_obs_dims[key]
                logger.info(f"{obs_key}: {key} has dim: {auxiliary_obs_dims[key]}")
    config.robot.algo_obs_dim_dict = obs_dim_dict
    
    OmegaConf.set_struct(config.env.config.obs.obs_dims, False)
    config.env.config.obs.obs_dims.update(auxiliary_obs_dims) # ZL: adding auxiliary obs dims to obs_dims
    
    logger.info(f"algo_obs_dim_dict: {config.robot.algo_obs_dim_dict}")

    # compute action_dim for ppo
    # for agent in config.algo.config.network_dict.keys():
    #     for network in config.algo.config.network_dict[agent].keys():
    #         output_dim = config.algo.config.network_dict[agent][network].output_dim
    #         if output_dim == "action_dim":
    #             config.algo.config.network_dict[agent][network].output_dim = config.env.config.robot.actions_dim
                
    # print the config
    # logger.debug(f"PPO CONFIG")
    # logger.debug(f"{config.algo.config.module_dict}")
    # logger.debug(f"{config.algo.config.network_dict}")

def parse_observation(cls: Any, 
                      key_list: List, 
                      buf_dict: Dict, 
                      obs_scales: Dict, 
                      noise_scales: Dict,
                      current_noise_curriculum_value: Any,
                      use_noise: bool = True) -> None:
    """ Parse observations for the legged_robot_base class
    """
    # TOOD: Parse observations for manipulation tasks

    for obs_key in key_list:
        if use_noise:
            obs_noise = noise_scales[obs_key] * current_noise_curriculum_value
        else:
            obs_noise = 0.
        
        # print(f"obs_key: {obs_key}, obs_noise: {obs_noise}")
        
        actor_obs = getattr(cls, f"_get_obs_{obs_key}")().clone()
        obs_scale = obs_scales[obs_key]
        
       
        buf_dict[obs_key] = (actor_obs + (torch.rand_like(actor_obs)* 2. - 1.) * obs_noise) * obs_scale


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


def get_backward_observation(env, motion_id, use_root_height_obs: bool = False, velocity_multiplier: float = 1.0) -> torch.Tensor:
    from humanoidverse.utils.torch_utils import quat_rotate_inverse
    from humanoidverse.envs.legged_robot_motions.legged_robot_motions import compute_humanoid_observations_max, compute_humanoid_observations_max_with_contact
    
    motion_times = torch.arange(int(np.ceil((env._motion_lib._motion_lengths[motion_id]/env.dt).cpu()))).to(env.device) * env.dt
    
    # get blend motion state
    motion_state = env._motion_lib.get_motion_state(motion_id, motion_times)

    ref_dof_pos = motion_state["dof_pos"] - env.default_dof_pos[0]
    ref_dof_vel = motion_state["dof_vel"] * velocity_multiplier

    ref_body_pos, ref_body_rots, ref_body_vels, ref_body_angular_vels = select_motion_body_data(
        env,
        motion_state,
        velocity_multiplier=velocity_multiplier,
    )

    # construct observation
    if env.use_contact_in_obs_max:
        contact_binary = env.foot_contact_detect(ref_body_pos, ref_body_vels)
        obs_dict = compute_humanoid_observations_max_with_contact(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=use_root_height_obs,
            contact_binary=contact_binary
        )
    else:
        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=use_root_height_obs,
        )
    max_local_self_obs = torch.cat([v for v in obs_dict.values()], dim=-1)

    if env.config.obs.use_obs_filter:
        imu_body_idx = 0
        imu_body_name = env.config.robot.get("imu_body_name", None)
        if imu_body_name is not None:
            motion_body_names = getattr(env, "motion_body_names_extend", None)
            if motion_body_names is None:
                motion_body_names = getattr(env, "motion_body_names", None)
            if motion_body_names is None:
                motion_body_names = env._motion_lib.mesh_parsers.body_names
            imu_body_idx = motion_body_names.index(imu_body_name)

        imu_quat = ref_body_rots[:, imu_body_idx]
        ref_ang_vel = ref_body_angular_vels[:, imu_body_idx]
        projected_gravity = quat_rotate_inverse(
            imu_quat,
            env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1),
            w_last=True
        )
        bogus_actions = ref_dof_pos

        bogus_history_actor = torch.cat([bogus_actions, ref_ang_vel, ref_dof_pos, ref_dof_vel, projected_gravity], dim=-1).repeat(1, 4)
        ref_dict = {
            "actions": bogus_actions,
            "ref_ang_vel": ref_ang_vel,
            "ref_dof_pos": ref_dof_pos,
            "ref_dof_vel": ref_dof_vel,
            "dof_pos": motion_state["dof_pos"],
            "fake_history": bogus_history_actor,
            "max_local_self_obs": max_local_self_obs,
            "projected_gravity": projected_gravity,
            "ref_body_pos": ref_body_pos,
            "ref_body_rots": ref_body_rots,
            "ref_body_vels": ref_body_vels,
            "ref_body_angular_vels": ref_body_angular_vels
        }
        state = torch.cat([ref_dof_pos,
                    ref_dof_vel,
                    projected_gravity,
                    ref_ang_vel], dim=-1)
        last_action = bogus_actions
        # TODO get obs from raw obs instead of the obs
        bfmzero_obs = {
            "state": state,
            "last_action": last_action,
            "privileged_state": max_local_self_obs
        }
        return bfmzero_obs, ref_dict
    else:
        ref_dict = {
            "max_local_self_obs": max_local_self_obs,
        }
        return max_local_self_obs, ref_dict


def _get_space_dim(space) -> int:
    if not hasattr(space, "shape") or len(space.shape) != 1:
        raise ValueError(f"Expected a 1D observation space, got {space}")
    return int(space.shape[0])


def _filter_keys(input_filter) -> list[str] | None:
    input_keys = getattr(input_filter, "key", None)
    if input_keys is None:
        return None
    if isinstance(input_keys, str):
        return [input_keys]
    if isinstance(input_keys, Sequence):
        return list(input_keys)
    raise TypeError(f"Unsupported input_filter.key type: {type(input_keys)}")


def _infer_obs_layout_from_filter(
    inference_model,
    input_filter,
    *,
    filter_name: str,
    skip_keys: set[str] | None = None,
) -> list[tuple[str | None, int]]:
    obs_space = getattr(inference_model, "obs_space", None)
    input_keys = _filter_keys(input_filter)
    skip_keys = skip_keys or set()

    if input_keys is None:
        if hasattr(obs_space, "spaces") and len(obs_space.spaces) == 1:
            key = next(iter(obs_space.spaces.keys()))
            return [(key, _get_space_dim(obs_space.spaces[key]))]

        output_space = getattr(getattr(inference_model, f"_{filter_name}", None), "input_filter", None)
        output_space = getattr(output_space, "output_space", obs_space)
        if output_space is None:
            raise ValueError(f"Cannot infer ONNX {filter_name} observation layout from model.")
        return [(None, _get_space_dim(output_space))]

    if not hasattr(obs_space, "spaces"):
        raise ValueError(
            f"Cannot infer ONNX {filter_name} observation layout from model. "
            f"Expected model.obs_space from checkpoint/model/init_kwargs.json and {filter_name}.input_filter.key, "
            "or pass obs_layout explicitly."
        )

    layout = []
    for key in input_keys:
        if key in skip_keys:
            continue
        if key not in obs_space.spaces:
            raise KeyError(f"{filter_name} input key {key!r} not found in obs_space keys {list(obs_space.spaces.keys())}")
        layout.append((key, _get_space_dim(obs_space.spaces[key])))
    return layout


def _infer_actor_obs_layout(inference_model, history: bool) -> list[tuple[str | None, int]]:
    input_filter = getattr(inference_model.cfg.archi.actor, "input_filter", None)
    skip_keys = set() if history else {"history_actor"}
    return _infer_obs_layout_from_filter(inference_model, input_filter, filter_name="actor", skip_keys=skip_keys)


def _infer_backward_obs_layout(inference_model) -> list[tuple[str | None, int]]:
    input_filter = getattr(inference_model.cfg.archi.b, "input_filter", None)
    layout = _infer_obs_layout_from_filter(inference_model, input_filter, filter_name="backward_map")

    obs_space = getattr(inference_model, "obs_space", None)
    normalizer_cfg = getattr(getattr(inference_model, "cfg", None), "obs_normalizer", None)
    normalizers = getattr(normalizer_cfg, "normalizers", None)
    allow_mismatching = getattr(normalizer_cfg, "allow_mismatching_keys", False)
    if (
        layout
        and layout[0][0] is not None
        and hasattr(obs_space, "spaces")
        and isinstance(normalizers, dict)
        and not allow_mismatching
    ):
        known_keys = {key for key, _ in layout}
        for key in normalizers.keys():
            if key not in known_keys:
                if key not in obs_space.spaces:
                    raise KeyError(f"Normalizer key {key!r} not found in obs_space keys {list(obs_space.spaces.keys())}")
                layout.append((key, _get_space_dim(obs_space.spaces[key])))
    return layout


def _split_flat_obs(flat_obs: torch.Tensor, obs_layout: list[tuple[str | None, int]]):
    if len(obs_layout) == 1 and obs_layout[0][0] is None:
        return flat_obs

    obs = {}
    start = 0
    for key, dim in obs_layout:
        if key is None:
            raise ValueError("Tensor obs layout cannot be mixed with keyed obs layout.")
        end = start + dim
        obs[key] = flat_obs[:, start:end]
        start = end
    return obs


def export_meta_policy_as_onnx(
    inference_model,
    path,
    exported_policy_name,
    example_obs_dict,
    z_dim,
    history: bool = False,
    obs_layout: list[tuple[str, int]] | None = None,
):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)
    inference_model = inference_model.eval()
    actor = copy.deepcopy(inference_model).to("cpu")
    actor_obs_layout = obs_layout or _infer_actor_obs_layout(inference_model, history=history)

    class PPOWrapper(nn.Module):
        def __init__(self, actor, actor_obs_layout):
            """
            model: The original PyTorch model.
            input_keys: List of input names as keys for the input dictionary.
            """
            super(PPOWrapper, self).__init__()
            self.actor = actor
            self.actor_obs_layout = actor_obs_layout

        def forward(self, actor_obs):
            """
            Dynamically creates a dictionary from the input keys and args.
            """
            actor_obs, ctx = actor_obs[:, :-z_dim], actor_obs[:, -z_dim:]
            actor_input = _split_flat_obs(actor_obs, self.actor_obs_layout)
            return self.actor.act(actor_input, ctx)

    wrapper = PPOWrapper(actor, actor_obs_layout=actor_obs_layout)
    example_input_list = example_obs_dict["actor_obs"]
    torch.onnx.export(
        wrapper,
        example_input_list,  # Pass x1 and x2 as separate inputs
        path,
        verbose=True,
        input_names=["actor_obs"],  # Specify the input names
        output_names=["action"],  # Name the output
        opset_version=13,  # Specify the opset version, if needed
    )
    return path


def export_z_encoder_as_onnx(
    inference_model,
    path,
    exported_encoder_name,
    example_obs_dict: dict[str, torch.Tensor] | None = None,
    obs_layout: list[tuple[str | None, int]] | None = None,
    project_z: bool = True,
):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_encoder_name)
    inference_model = inference_model.eval()
    encoder = copy.deepcopy(inference_model).to("cpu")
    encoder_obs_layout = obs_layout or _infer_backward_obs_layout(inference_model)

    class ZEncoderWrapper(nn.Module):
        def __init__(self, encoder, encoder_obs_layout, project_z: bool):
            super().__init__()
            self.encoder = encoder
            self.encoder_obs_layout = encoder_obs_layout
            self.project_z = project_z

        def forward(self, encoder_obs):
            obs = _split_flat_obs(encoder_obs, self.encoder_obs_layout)
            z = self.encoder.backward_map(obs)
            if self.project_z:
                z = self.encoder.project_z(z)
            return z.float()

    wrapper = ZEncoderWrapper(encoder, encoder_obs_layout=encoder_obs_layout, project_z=project_z)
    if example_obs_dict is not None and "encoder_obs" in example_obs_dict:
        example_input = example_obs_dict["encoder_obs"]
    else:
        example_dim = sum(dim for _, dim in encoder_obs_layout)
        example_input = torch.randn(1, example_dim)

    torch.onnx.export(
        wrapper,
        example_input,
        path,
        verbose=False,
        input_names=["encoder_obs"],
        output_names=["z"],
        opset_version=13,
    )
    return path
