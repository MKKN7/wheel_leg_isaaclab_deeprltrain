# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================


import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
import numpy as np
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import IdealPDActuator, ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from typing import  Literal
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import quat_apply


def _sample_event_distribution(
    distribution_params: tuple[float, float],
    shape: tuple[int, ...],
    device: torch.device | str,
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    if distribution == "uniform":
        return math_utils.sample_uniform(*distribution_params, shape, device=device)
    if distribution == "log_uniform":
        return math_utils.sample_log_uniform(*distribution_params, shape, device=device)
    if distribution == "gaussian":
        return math_utils.sample_gaussian(*distribution_params, shape, device=device)
    raise ValueError(f"Unsupported distribution: {distribution}")


def _resolve_actuator_joint_indices(asset_cfg: SceneEntityCfg, actuator, device: torch.device) -> tuple[slice | torch.Tensor, slice | torch.Tensor] | None:
    if isinstance(asset_cfg.joint_ids, slice):
        actuator_indices = slice(None)
        if isinstance(actuator.joint_indices, slice):
            global_indices = slice(None)
        elif isinstance(actuator.joint_indices, torch.Tensor):
            global_indices = actuator.joint_indices.to(device=device)
        else:
            global_indices = torch.as_tensor(actuator.joint_indices, device=device, dtype=torch.long)
        return actuator_indices, global_indices

    asset_joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=device, dtype=torch.long)
    if isinstance(actuator.joint_indices, slice):
        return asset_joint_ids, asset_joint_ids

    actuator_joint_indices = (
        actuator.joint_indices.to(device=device)
        if isinstance(actuator.joint_indices, torch.Tensor)
        else torch.as_tensor(actuator.joint_indices, device=device, dtype=torch.long)
    )
    actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
    if actuator_indices.numel() == 0:
        return None
    return actuator_indices, actuator_joint_indices[actuator_indices]


def _install_effort_output_randomizer(actuator) -> None:
    if getattr(actuator, "_wb_effort_output_randomizer_installed", False):
        return

    actuator._wb_effort_output_scale = torch.ones_like(actuator.computed_effort)
    actuator._wb_effort_output_bias = torch.zeros_like(actuator.computed_effort)
    actuator._wb_effort_output_noise_std = torch.zeros_like(actuator.computed_effort)
    actuator._wb_effort_output_original_compute = actuator.compute

    def compute_with_effort_output_randomization(control_action, joint_pos, joint_vel):
        result = actuator._wb_effort_output_original_compute(control_action, joint_pos, joint_vel)
        base_effort = actuator.computed_effort
        if base_effort is None or base_effort.shape != actuator._wb_effort_output_scale.shape:
            base_effort = result.joint_efforts
        disturbed_effort = base_effort * actuator._wb_effort_output_scale + actuator._wb_effort_output_bias
        noise_std = actuator._wb_effort_output_noise_std
        if torch.any(noise_std > 0.0):
            disturbed_effort = disturbed_effort + torch.randn_like(disturbed_effort) * noise_std
        actuator.computed_effort = disturbed_effort
        actuator.applied_effort = actuator._clip_effort(disturbed_effort)
        result.joint_efforts = actuator.applied_effort
        result.joint_positions = None
        result.joint_velocities = None
        return result

    actuator.compute = compute_with_effort_output_randomization
    actuator._wb_effort_output_randomizer_installed = True


def randomize_gimbal_heading_pd_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    kp_distribution_params: tuple[float, float] | None = None,
    kd_distribution_params: tuple[float, float] | None = None,
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> None:
    """Randomize the env-side gimbal heading PD gains used by Wheelbipe V14."""

    num_envs = int(getattr(env, "num_envs", getattr(getattr(env, "scene", None), "num_envs", 0)))
    device = getattr(env, "device", "cpu")
    if num_envs <= 0:
        return
    if env_ids is None:
        env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    cfg = getattr(env, "cfg", None)
    heading_cfg = getattr(cfg, "gimbal_heading_control_cfg", {}) if cfg is not None else {}
    if not isinstance(heading_cfg, dict):
        heading_cfg = {}
    base_kp = float(heading_cfg.get("kp", 2.0))
    base_kd = float(heading_cfg.get("kd", 0.15))

    if not hasattr(env, "_gimbal_heading_kp") or env._gimbal_heading_kp.shape[0] != num_envs:
        env._gimbal_heading_kp = torch.full((num_envs,), base_kp, dtype=torch.float, device=device)
    if not hasattr(env, "_gimbal_heading_kd") or env._gimbal_heading_kd.shape[0] != num_envs:
        env._gimbal_heading_kd = torch.full((num_envs,), base_kd, dtype=torch.float, device=device)

    if kp_distribution_params is None:
        env._gimbal_heading_kp[env_ids] = base_kp
    else:
        env._gimbal_heading_kp[env_ids] = torch.clamp(
            _sample_event_distribution(kp_distribution_params, (int(env_ids.numel()),), device, distribution),
            min=0.0,
        )
    if kd_distribution_params is None:
        env._gimbal_heading_kd[env_ids] = base_kd
    else:
        env._gimbal_heading_kd[env_ids] = torch.clamp(
            _sample_event_distribution(kd_distribution_params, (int(env_ids.numel()),), device, distribution),
            min=0.0,
        )


class randomize_actuator_effort_output(ManagerTermBase):
    """Randomize explicit IdealPD-family actuator output torque.

    This event installs a light runtime wrapper on every selected ``IdealPDActuator``
    or subclass. The wrapper perturbs the torque after the actuator computes its
    nominal effort and before the actuator-specific effort clipping is applied:

    ``tau = clip(tau_nominal * scale + bias + N(0, noise_std))``.

    It does not affect implicit PhysX actuators.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, Articulation):
            raise TypeError("randomize_actuator_effort_output only supports Articulation assets.")
        if not hasattr(self.asset, "actuators"):
            raise TypeError("Asset does not expose explicit actuator models.")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        effort_scale_distribution_params: tuple[float, float] | None = None,
        effort_bias_distribution_params: tuple[float, float] | None = None,
        effort_noise_std_distribution_params: tuple[float, float] | None = None,
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        reset_to_default: bool = True,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)
        else:
            env_ids = env_ids.to(device=self.asset.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        for actuator in self.asset.actuators.values():
            if not isinstance(actuator, IdealPDActuator):
                continue

            resolved_indices = _resolve_actuator_joint_indices(self.asset_cfg, actuator, self.asset.device)
            if resolved_indices is None:
                continue
            actuator_indices, _ = resolved_indices
            _install_effort_output_randomizer(actuator)

            if reset_to_default:
                actuator._wb_effort_output_scale[env_ids[:, None], actuator_indices] = 1.0
                actuator._wb_effort_output_bias[env_ids[:, None], actuator_indices] = 0.0
                actuator._wb_effort_output_noise_std[env_ids[:, None], actuator_indices] = 0.0

            if isinstance(actuator_indices, slice):
                num_actuator_joints = actuator.num_joints
            else:
                num_actuator_joints = int(actuator_indices.numel())
            shape = (int(env_ids.numel()), num_actuator_joints)
            env_index = env_ids[:, None]

            if effort_scale_distribution_params is not None:
                scale = _sample_event_distribution(
                    effort_scale_distribution_params, shape, self.asset.device, distribution
                )
                actuator._wb_effort_output_scale[env_index, actuator_indices] = torch.clamp(scale, min=0.0)
            if effort_bias_distribution_params is not None:
                bias = _sample_event_distribution(
                    effort_bias_distribution_params, shape, self.asset.device, distribution
                )
                actuator._wb_effort_output_bias[env_index, actuator_indices] = bias
            if effort_noise_std_distribution_params is not None:
                noise_std = _sample_event_distribution(
                    effort_noise_std_distribution_params, shape, self.asset.device, distribution
                )
                actuator._wb_effort_output_noise_std[env_index, actuator_indices] = torch.clamp(noise_std, min=0.0)


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "scale",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the inertia of rigid bodies by scaling or adding values.
    
    This function randomizes the inertia tensor of the specified rigid bodies.
    It supports three operations:
    - "scale": Multiply the inertia by a random value sampled from the given range
    - "add": Add a random value sampled from the given range to the inertia
    - "abs": Set the inertia to a random absolute value sampled from the given range
    
    Args:
        env: The environment instance
        env_ids: Environment indices to randomize. If None, all environments are randomized.
        inertia_distribution_params: Tuple of (min, max) for the random distribution
        operation: Operation to perform on the inertia tensor ("add", "scale", or "abs")
        asset_cfg: Scene entity configuration for the asset
        
    Note:
        This function uses CPU tensors to assign the inertia. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    
    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    
    # 获取当前的转动惯量 (num_assets, num_bodies, 9)
    # PhysX stores inertia as: [Ixx, Iyy, Izz, Ixy, Ixz, Iyz, 0, 0, 0]
    inertias = asset.root_physx_view.get_inertias().clone()
    
    # 生成随机采样值 - 每个 body 使用统一的采样因子以保持物理合法性 (正定性)
    rand_samples = math_utils.sample_uniform(
        inertia_distribution_params[0],
        inertia_distribution_params[1],
        (len(env_ids), len(body_ids), 1),
        device="cpu"
    )
    
    # 应用操作到前 6 个分量 (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
    # 统一缩放可以保持惯量张量的结构完整，防止 PhysX 报非负/非正定错误
    if operation == "scale":
        inertias[env_ids.unsqueeze(1), body_ids.unsqueeze(0), :6] *= rand_samples
    elif operation == "add":
        # 对于 add，通常只建议增加量级，或者确保 rand_samples 为正
        inertias[env_ids.unsqueeze(1), body_ids.unsqueeze(0), :3] += rand_samples
    elif operation == "abs":
        # 对于 abs，直接赋值对角项，非对角项清零以保证安全
        inertias[env_ids.unsqueeze(1), body_ids.unsqueeze(0), :3] = rand_samples
        inertias[env_ids.unsqueeze(1), body_ids.unsqueeze(0), 3:6] = 0.0
    else:
        raise ValueError(f"Invalid operation: {operation}. Must be 'add', 'scale', or 'abs'.")
    
    # 设置新的转动惯量
    asset.root_physx_view.set_inertias(inertias, env_ids)
    
def reset_root_state_uniform_test(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    # print(rand_samples)
    # Prefer terrain origins (for non-flat/shifted terrains); fallback to scene origins.
    # env_origins = (
    #     env.terrain.env_origins[env_ids]
    #     if hasattr(env, "terrain") and hasattr(env.terrain, "env_origins")
    #     else env.scene.env_origins[env_ids]
    # )
    env_origins = env.terrain.env_origins[env_ids]
    positions = root_states[:, 0:3] + env_origins + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def reset_root_state_uniform_vel_b(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    # env_origins = (
    #     env.terrain.env_origins[env_ids]
    #     if hasattr(env, "terrain") and hasattr(env.terrain, "env_origins")
    #     else env.scene.env_origins[env_ids]
    # )
    env_origins = env.terrain.env_origins[env_ids]
    # positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    positions = root_states[:, 0:3] + env_origins + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # get velocities frame
    rand_samples_yaw = rand_samples[:,3:6].clone()
    rand_samples_yaw[:,0] = 0
    rand_samples_yaw[:,1] = 0
    orientations_vel_delta = math_utils.quat_from_euler_xyz(rand_samples_yaw[:, 0], rand_samples_yaw[:, 1], rand_samples_yaw[:, 2])
    orientations_vel = math_utils.quat_mul(root_states[:, 3:7], orientations_vel_delta)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    rand_samples_b = math_utils.quat_apply(orientations_vel,rand_samples[:,:3])
    rand_samples[:,:3] = rand_samples_b.clone()

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def additional_root_z_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    pose_w = asset.data.root_pose_w[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], len(env_ids), device=asset.device)
    pose_w[:,2] += rand_samples

    # velocities
    vel_w = asset.data.root_vel_w[env_ids].clone()
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], len(env_ids), device=asset.device)
    vel_w[:, 2] += rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(pose_w, env_ids=env_ids)
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

def apply_external_force_torque_xyz(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # 修改力范围参数，支持每个轴单独设定范围
    force_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    # 修改力矩范围参数，支持每个轴单独设定范围
    torque_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges for each axis. 
    The number of forces and torques is equal to the number of bodies times the number of environments. 
    The forces and torques are applied to the bodies by calling ``asset.set_external_force_and_torque``. 
    The forces and torques are only applied when ``asset.write_data_to_sim()`` is called in the environment.
    
    Args:
        force_range: 三个轴的力范围，格式为 ((fx_min, fx_max), (fy_min, fy_max), (fz_min, fz_max))
        torque_range: 三个轴的力矩范围，格式为 ((tx_min, tx_max), (ty_min, ty_max), (tz_min, tz_max))
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # 为每个轴单独采样力
    forces_x = math_utils.sample_uniform(force_range[0][0], force_range[0][1], 
                                        (len(env_ids), num_bodies, 1), asset.device)
    forces_y = math_utils.sample_uniform(force_range[1][0], force_range[1][1], 
                                        (len(env_ids), num_bodies, 1), asset.device)
    forces_z = math_utils.sample_uniform(force_range[2][0], force_range[2][1], 
                                        (len(env_ids), num_bodies, 1), asset.device)
    # 合并三个轴的力
    forces = torch.cat([forces_x, forces_y, forces_z], dim=2)
    
    # 为每个轴单独采样力矩
    torques_x = math_utils.sample_uniform(torque_range[0][0], torque_range[0][1], 
                                         (len(env_ids), num_bodies, 1), asset.device)
    torques_y = math_utils.sample_uniform(torque_range[1][0], torque_range[1][1], 
                                         (len(env_ids), num_bodies, 1), asset.device)
    torques_z = math_utils.sample_uniform(torque_range[2][0], torque_range[2][1], 
                                         (len(env_ids), num_bodies, 1), asset.device)
    # 合并三个轴的力矩
    torques = torch.cat([torques_x, torques_y, torques_z], dim=2)
    
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


class randomize_joint_parameters_v1(ManagerTermBase):
    """Randomize the simulated joint parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the joint parameters of the asset. These correspond to the physics engine
    joint properties that affect the joint behavior. The properties include the joint friction coefficient, armature,
    and joint position limits.

    The function samples random values from the given distribution parameters and applies the operation to the
    joint properties. It then sets the values into the physics simulation. If the distribution parameters are
    not provided for a particular property, the function does not modify the property.

    .. tip::
        This function uses CPU tensors to assign the joint properties. It is recommended to use this function
        only during the initialization of the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "static_friction_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["static_friction_distribution_params"], "static_friction_distribution_params")
            if "dynamic_friction_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["dynamic_friction_distribution_params"], "dynamic_friction_distribution_params")
            if "viscous_friction_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["viscous_friction_distribution_params"], "viscous_friction_distribution_params")
            if "armature_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["armature_distribution_params"], "armature_distribution_params")
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_fixed_tendon_parameters' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        static_friction_distribution_params: tuple[float, float] | None = None,
        dynamic_friction_distribution_params: tuple[float, float] | None = None,
        viscous_friction_distribution_params: tuple[float, float] | None = None,
        armature_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        # resolve joint indices
        if self.asset_cfg.joint_ids == slice(None):
            joint_ids = slice(None)  # for optimization purposes
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, dtype=torch.int, device=self.asset.device)

        # sample joint properties from the given ranges and set into the physics simulation
        # joint friction coefficient
        if static_friction_distribution_params is not None:
            friction_coeff = _randomize_prop_by_op(
                self.asset.data.default_joint_friction_coeff.clone(),
                static_friction_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )

            # ensure the friction coefficient is non-negative
            friction_coeff = torch.clamp(friction_coeff, min=0.0)

            # Always set static friction (indexed once)
            static_friction_coeff = friction_coeff[env_ids[:, None], joint_ids]
        else:
            static_friction_coeff = None

        # if isaacsim version is lower than 5.0.0 we can set only the static friction coefficient
        major_version = int(env.sim.get_version()[0])
        if major_version >= 5:
            # Randomize raw tensors
            if dynamic_friction_distribution_params is not None:
                dynamic_friction_coeff = _randomize_prop_by_op(
                    self.asset.data.default_joint_dynamic_friction_coeff.clone(),
                    dynamic_friction_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
            elif static_friction_distribution_params is not None:
                dynamic_friction_coeff = _randomize_prop_by_op(
                    self.asset.data.default_joint_dynamic_friction_coeff.clone(),
                    static_friction_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
            else:
                dynamic_friction_coeff = None
            
            if dynamic_friction_coeff is not None and static_friction_coeff is not None:
                dynamic_friction_coeff = torch.clamp(dynamic_friction_coeff, min=0.0)
                dynamic_friction_coeff = torch.minimum(dynamic_friction_coeff, friction_coeff)
                dynamic_friction_coeff = dynamic_friction_coeff[env_ids[:, None], joint_ids]
            else:
                dynamic_friction_coeff = None
            
            if viscous_friction_distribution_params is not None:
                viscous_friction_coeff = _randomize_prop_by_op(
                    self.asset.data.default_joint_viscous_friction_coeff.clone(),
                    viscous_friction_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
                viscous_friction_coeff = torch.clamp(viscous_friction_coeff, min=0.0)
                viscous_friction_coeff = viscous_friction_coeff[env_ids[:, None], joint_ids]
            else:
                viscous_friction_coeff = None
        else:
            # For versions < 5.0.0, we do not set these values
            dynamic_friction_coeff = None
            viscous_friction_coeff = None

        # Single write call for all versions
        self.asset.write_joint_friction_coefficient_to_sim(
            joint_friction_coeff=static_friction_coeff,
            joint_dynamic_friction_coeff=dynamic_friction_coeff,
            joint_viscous_friction_coeff=viscous_friction_coeff,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )

        # joint armature
        if armature_distribution_params is not None:
            armature = _randomize_prop_by_op(
                self.asset.data.default_joint_armature.clone(),
                armature_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_joint_armature_to_sim(
                armature[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids
            )

        # joint position limits
        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            joint_pos_limits = self.asset.data.default_joint_pos_limits.clone()
            # -- randomize the lower limits
            if lower_limit_distribution_params is not None:
                joint_pos_limits[..., 0] = _randomize_prop_by_op(
                    joint_pos_limits[..., 0],
                    lower_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
            # -- randomize the upper limits
            if upper_limit_distribution_params is not None:
                joint_pos_limits[..., 1] = _randomize_prop_by_op(
                    joint_pos_limits[..., 1],
                    upper_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )

            # extract the position limits for the concerned joints
            joint_pos_limits = joint_pos_limits[env_ids[:, None], joint_ids]
            if (joint_pos_limits[..., 0] > joint_pos_limits[..., 1]).any():
                raise ValueError(
                    "Randomization term 'randomize_joint_parameters' is setting lower joint limits that are greater"
                    " than upper joint limits. Please check the distribution parameters for the joint position limits."
                )
            # set the position limits into the physics simulation
            self.asset.write_joint_position_limit_to_sim(
                joint_pos_limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
            )

def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data
