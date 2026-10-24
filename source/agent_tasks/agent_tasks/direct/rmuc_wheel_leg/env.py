"""V14-style control environment for the bundled RMUC closed-loop USD."""
from __future__ import annotations

from collections.abc import Sequence
import math
import random

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_inv, quat_mul


_LOOP_POINTS = (
    ("left_node_link", "left_upper_leg", (0.005250, 0.088675, 0.069231), (0.009000, -0.064325, 0.069228)),
    ("left_parallel_link", "left_lower_leg", (0.006625, -0.078620, 0.084618), (-0.001500, 0.059840, 0.025380)),
    ("right_node_link", "right_upper_leg", (-0.005250, 0.088675, 0.069231), (-0.009000, -0.064325, 0.069228)),
    ("right_parallel_link", "right_lower_leg", (-0.006625, -0.078620, 0.084618), (0.001500, 0.059840, 0.025380)),
)

from .bank_geometry import bank_reference
from .env_cfg import (
    ACTION_DIM,
    DOF_NAMES,
    LEG_POLICY_DOF_NAMES,
    OBSERVATION_DIM,
    POLICY_DOF_NAMES,
    STANDING_JOINT_POS,
    WHEEL_POLICY_DOF_NAMES,
    RmucWheelLegCourseEnvCfg,
)
from .observation_noise import add_command_consistent_noise, sensor_noise, delayed_sensor_observation
from .command_contract import interpolate_leg_scale, validate_leg_scale_profile
from .command_sequence import validate_sequence, sequence_commands, dynamic_tracking_cost, dynamic_stop_posture_cost, matched_filtered_target, tracking_cohort_mask
from .command_sampling import sample_opened_commands, fixed_turn_sign_cohorts, ensure_priority_sign_coverage
from .low_speed_tracking import tracking_targets, rebase_aligned_history
from .motor_response import install_effort_response
from .wheel_passive import (configure_wheel_torque_contract, read_wheel_joint_geometry,
                            mechanical_torque, feedback_damping_torque, body_torque_pair,
                            install_explicit_pd_sample_hold, sample_separated_wheel_parameters)
from .gas_spring import (spring_body_torques, validate_reference_spring_joints,
                         radial_spring_force_pair, radial_attachment_points, sample_equivalent_torques)
from .gas_spring import firmware_spring_torque_from_joint_positions


def validate_rehearsal_priority_config(cfg):
    """Validate final overrides before allocating simulation resources.

    The config class body runs before Hydra overrides; zip must never silently
    pair a new stage list with a truncated subset of old weights.
    """
    stages = tuple(cfg.command_rehearsal_priority_stage_ids)
    weights = tuple(cfg.command_rehearsal_priority_stage_weights)
    if len(stages) != len(weights):
        raise ValueError(
            "command_rehearsal_priority_stage_ids and "
            "command_rehearsal_priority_stage_weights must have matching lengths "
            "after overrides; override both lists together"
        )
    if len(set(stages)) != len(stages):
        raise ValueError("command_rehearsal_priority_stage_ids must be unique")
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("command_rehearsal_priority_stage_weights must be finite and positive")
    if (getattr(cfg, 'command_rehearsal_stratified_coverage', False)
            and not getattr(cfg, 'command_turn_fixed_sign_cohorts', False)):
        raise ValueError('Stratified rehearsal coverage requires fixed turn sign cohorts')


class RmucWheelLegEnv(DirectRLEnv):
    """Closed-loop RMUC task with V14-style leg-position and wheel-torque control.

    The closed-loop leg uses the same four active command channels as V14, with
    the left/right sign convention represented by the reset profile. Passive
    linkage and guide-wheel joints are explicitly configured with V14-like
    zero-stiffness drives in the asset wrapper.
    """

    cfg: RmucWheelLegCourseEnvCfg

    def __init__(self, cfg: RmucWheelLegCourseEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.real2sim_end_to_end_delay:
            from .real2sim_wheel import load_profile, configure_total_delay
            if not cfg.real2sim_wheel_profile:
                raise ValueError('End-to-end timing requires explicit real2sim wheel profile')
            timing_profile = load_profile(cfg.real2sim_wheel_profile)
            # Identified wheel transport owns the full observed-to-execution
            # delay; do not add the historical whole-policy 8ms buffers again.
            configure_total_delay(cfg, timing_profile)
        self._separated_wheel_passive = configure_wheel_torque_contract(cfg)
        validate_rehearsal_priority_config(cfg)
        reverse_alignment_scale = cfg.command_reverse_alignment_scale_deg
        if not math.isfinite(reverse_alignment_scale) or reverse_alignment_scale < 0:
            raise ValueError('Reverse alignment scale must be finite and nonnegative')
        sequence = getattr(cfg, 'command_training_sequence', ())
        random_range = getattr(cfg, 'command_random_speed_range', ())
        dynamic_schedule = bool(sequence or random_range)
        if sequence and random_range:
            raise ValueError('Fixed and random training sequences are mutually exclusive')
        self._random_speed_sequence = None
        self._training_sequence_table = None
        dynamic_weight = cfg.command_dynamic_tracking_weight
        crossing_weight = cfg.command_sequence_overshoot_weight
        if (not math.isfinite(crossing_weight) or not 0 <= crossing_weight <= 20
                or (crossing_weight and (not sequence or random_range))):
            raise ValueError('Sequence overshoot weight requires fixed sequence and must be within [0,20]')
        stop_weight = cfg.command_dynamic_stop_posture_weight
        if not math.isfinite(stop_weight) or stop_weight < 0 or (stop_weight and not dynamic_schedule):
            raise ValueError('Stop posture weight must be finite, nonnegative and require training sequence')
        if not math.isfinite(dynamic_weight) or dynamic_weight < 0:
            raise ValueError('Dynamic tracking weight must be finite and nonnegative')
        if dynamic_weight and not dynamic_schedule:
            raise ValueError('Dynamic tracking reward requires explicit training sequence')
        if dynamic_weight:
            dynamic_tracking_cost(torch.zeros(1), torch.zeros(1), torch.zeros(1), cfg.command_dynamic_tracking_scale_mps)
        if dynamic_schedule:
            if (getattr(cfg, 'play', False) or cfg.curriculum_stage != 1
                    or not 0 < cfg.command_continuous_sampling_fraction <= .25
                    or not cfg.feedback_contract_by_command
                    or not cfg.command_reward_contract_by_command
                    or cfg.command_reset_acquisition_ramp_s > 0
                    or cfg.command_speed_extension_sampling_box):
                raise ValueError('Training sequence requires command-consistent continuous cohort without play, acquisition ramp or turn box')
            if sequence:
                self._training_sequence_table = validate_sequence(
                    [list(row) for row in sequence], cfg.sim.dt * cfg.decimation, cfg.episode_length_s)
            if random_range:
                from .random_speed_sequence import RandomSpeedSequence
                # Validate on CPU before allocating the simulation.
                RandomSpeedSequence(0, 'cpu', cfg.sim.dt * cfg.decimation, cfg.episode_length_s,
                    random_range, cfg.command_random_hold_s, cfg.command_random_ramp_s,
                    cfg.command_random_step_fraction, cfg.command_random_max_delta_mps,
                    isolated_rng=cfg.command_random_isolated_rng,
                    stop_probability=cfg.command_random_stop_probability)
        entry_times = (cfg.command_turn_entry_hold_s, cfg.command_turn_entry_ramp_s)
        if any(not math.isfinite(t) or not 0 <= t <= 4 for t in entry_times):
            raise ValueError('Turn entry durations must be finite within0..4s')
        if not 0 <= cfg.command_turn_entry_abrupt_fraction <= 1:
            raise ValueError('Turn entry abrupt fraction must be within0..1')
        if sum(entry_times) > 0 and not (cfg.feedback_contract_by_command and cfg.command_reward_contract_by_command):
            raise ValueError('Turn entry requires command-consistent reward and feedback')
        super().__init__(cfg, render_mode, **kwargs)
        if self.cfg.command_reset_acquisition_ramp_s > 0 and not (
                self.cfg.feedback_contract_by_command and self.cfg.command_reward_contract_by_command):
            raise ValueError('Acquisition ramp requires command-consistent feedback and reward')
        if len(POLICY_DOF_NAMES) != ACTION_DIM or ACTION_DIM != 6:
            raise RuntimeError(f"RMUC V14 action layout must contain six joints, got {POLICY_DOF_NAMES!r}")
        if len(LEG_POLICY_DOF_NAMES) != 4 or len(WHEEL_POLICY_DOF_NAMES) != 2:
            raise RuntimeError("RMUC V14 action layout requires four leg motors and two wheel motors")

        self._dof_ids = self._find_joint_ids(DOF_NAMES)
        self._control_dof_ids = self._find_joint_ids(POLICY_DOF_NAMES)
        self._leg_control_ids = self._find_joint_ids(LEG_POLICY_DOF_NAMES)
        self._nominal_leg_velocity_limits = self.robot.data.joint_vel_limits.torch[:, self._leg_control_ids].clone()
        self._nominal_leg_armature = self.robot.data.joint_armature.torch[:, self._leg_control_ids].clone()
        self._wheel_control_ids = self._find_joint_ids(WHEEL_POLICY_DOF_NAMES)
        self._wheel_body_ids = self._find_body_ids(("left_wheel", "right_wheel"))
        self._root_body_id = self._find_body_ids(("robot",))[0]
        self._wheel_joint_ids_by_body = self._find_joint_ids(
            ("left_wheel_spin", "right_wheel_spin")
        )
        if self._separated_wheel_passive:
            geometry = read_wheel_joint_geometry(self.cfg.robot_cfg.spawn.usd_path, WHEEL_POLICY_DOF_NAMES)
            self._wheel_passive_body_ids = [torch.tensor(
                [self.robot.body_names.index(parent), self.robot.body_names.index(child)],
                device=self.device, dtype=torch.long) for parent, child, _ in geometry]
            self._wheel_passive_axes_b = torch.tensor([axis for _, _, axis in geometry], device=self.device)
            # Policy order [right, left]. Mechanical quantities are not motor output.
            self._last_wheel_mechanical_torque_nm = torch.zeros((self.num_envs, 2), device=self.device)
            self._last_wheel_feedback_torque_nm = torch.zeros((self.num_envs, 2), device=self.device)
            self._last_wheel_total_passive_torque_nm = torch.zeros((self.num_envs, 2), device=self.device)
            self._last_wheel_mechanical_body_torques_w = torch.zeros((self.num_envs, 2, 2, 3), device=self.device)
            (self._wheel_mechanical_viscous_per_env, self._wheel_coulomb_per_env,
             self._wheel_feedback_damping_per_env) = sample_separated_wheel_parameters(
                cfg, self.num_envs, 0.0, self.device)
        # V14-compatible bookkeeping names used by diagnostics and playback.
        self._legs_act_idx = self._leg_control_ids
        self._wheel_idx = self._wheel_control_ids
        self._spring_idx: list[int] = []
        gas_model = getattr(self.cfg, "gas_spring_model", "equivalent_assist_joint_torque")
        if gas_model not in ("equivalent_assist_joint_torque", "firmware_motor_equivalent"):
            raise ValueError(f"Unsupported RMUC gas spring model: {gas_model}")
        if getattr(self.cfg, "firmware_spring_enabled", False) and gas_model != "firmware_motor_equivalent":
            raise ValueError("firmware_spring_enabled requires gas_spring_model=firmware_motor_equivalent")
        spring_magnitude = float(self.cfg.gas_spring_assist_torque_nm)
        if not math.isfinite(spring_magnitude) or not 0.0 <= spring_magnitude <= 10.775:
            raise ValueError("Gas spring assistance must be in [0,10.775] Nm for the reference approximation")
        self._gas_spring_torque_nm = sample_equivalent_torques(self.num_envs,spring_magnitude,0.,
            self.cfg.gas_spring_torque_common_dr_fraction,self.cfg.gas_spring_torque_independent_dr_fraction,device=self.device)
        self._gas_spring_body_ids = None
        if spring_magnitude > 0:
            validate_reference_spring_joints(self.cfg.robot_cfg.spawn.usd_path)
            self._gas_spring_body_ids = torch.tensor(
                [self.robot.body_names.index(name) for name in ("robot", "left_active_link", "right_active_link")],
                device=self.device, dtype=torch.long)

        radial_force = float(self.cfg.gas_spring_radial_force_n)
        radial_dr = float(self.cfg.gas_spring_radial_dr_fraction)
        if not math.isfinite(radial_force) or radial_force < 0 or not math.isfinite(radial_dr) or not 0 <= radial_dr <= 1:
            raise ValueError("Radial spring force must be nonnegative and DR fraction in [0,1]")
        if radial_force > 0 and spring_magnitude > 0:
            raise ValueError("Do not double-count constant joint torque and radial gas-spring assistance")
        self._radial_spring_force_n = torch.full((self.num_envs, 2), radial_force, device=self.device)
        self._radial_spring_body_ids = None
        if radial_force > 0:
            self._radial_spring_points_b = torch.tensor(
                radial_attachment_points(self.cfg.robot_cfg.spawn.usd_path), device=self.device)
            self._radial_spring_body_ids = [torch.tensor(
                [self.robot.body_names.index("robot"), self.robot.body_names.index(f"{side}_wheel")],
                device=self.device, dtype=torch.long) for side in ("left", "right")]

        self._use_gimbal = False
        self._joint_pos_limits = self.robot.data.joint_pos_limits.torch[:, self._dof_ids, :]
        self._default_control_pos = self.robot.data.default_joint_pos.torch[:, self._control_dof_ids].clone()
        standing = torch.tensor(
            [STANDING_JOINT_POS.get(name, 0.0) for name in POLICY_DOF_NAMES],
            device=self.device,
            dtype=self._default_control_pos.dtype,
        )
        self._standing_control_pos = standing.expand(self.num_envs, -1)
        standing_all = torch.tensor(
            [STANDING_JOINT_POS.get(name, 0.0) for name in DOF_NAMES],
            device=self.device,
            dtype=self.robot.data.default_joint_pos.torch.dtype,
        )
        self._standing_leg_pos = standing_all[[i for i, name in enumerate(DOF_NAMES) if "wheel" not in name]].expand(
            self.num_envs, -1
        )
        self._leg_dof_ids = [i for i, name in enumerate(DOF_NAMES) if "wheel" not in name]
        self._leg_local_index = {name: self._leg_dof_ids.index(i) for i, name in enumerate(DOF_NAMES) if "wheel" not in name}
        self._policy_actions = torch.zeros((self.num_envs, ACTION_DIM), device=self.device)
        # When the trainer asks for an action-aware residual, the transformed
        # action is cached here and consumed by ``_pre_physics_step``.  This
        # keeps the action stored by PPO identical to the action sent to PhysX.
        self._prepared_policy_action: torch.Tensor | None = None
        self._previous_actions = torch.zeros_like(self._policy_actions)
        self._last_speed_feedback_action = torch.zeros(self.num_envs, device=self.device)
        self._last_highspeed_propulsion_action = torch.zeros(
            self.num_envs, device=self.device
        )
        self._last_wheel_damping_compensation_nm = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self._last_wheel_damping_anneal_fraction = torch.zeros(
            self.num_envs, device=self.device
        )
        self._last_effective_wheel_damping = torch.zeros(
            self.num_envs, device=self.device
        )
        self._last_effective_wheel_torque_scale = torch.zeros(
            self.num_envs, device=self.device
        )
        self._last_effective_wheel_effort_limit = torch.zeros(
            self.num_envs, device=self.device
        )
        self._last_wheel_damping_anneal_advancing = torch.zeros(
            self.num_envs, device=self.device
        )
        self._last_requested_wheel_effort_nm = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self._last_wheel_effort_target_nm = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self._last_action_feedback_delta = torch.zeros_like(self._policy_actions)
        self._estop = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        asset_rpy = torch.deg2rad(torch.tensor(self.cfg.asset_frame_rpy_deg, device=self.device))
        self._asset_frame_quat = quat_from_euler_xyz(asset_rpy[0], asset_rpy[1], asset_rpy[2]).expand(self.num_envs, -1)
        self._asset_frame_inv_quat = quat_inv(self._asset_frame_quat)
        self._gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        self._command = torch.tensor(
            [
                self.cfg.command_linear_x_mps if self.cfg.command_enabled else 0.0,
                0.0,
                self.cfg.command_yaw_rate_rad_s if self.cfg.command_enabled else 0.0,
            ],
            device=self.device,
        ).expand(self.num_envs, -1)
        if len(self.cfg.command_observation_scale) != 3:
            raise RuntimeError("RMUC command observation scale must contain vx, vy and yaw entries")
        self._command_scale = torch.tensor(self.cfg.command_observation_scale, device=self.device)
        self._wheel_effort_limit = torch.tensor(
            [
                self.cfg.robot_cfg.actuators["wheel_motors"].effort_limit,
                self.cfg.robot_cfg.actuators["wheel_motors"].effort_limit,
            ],
            device=self.device,
        )
        self._wheel_actuator = self.robot.actuators["wheel_motors"]
        if self._separated_wheel_passive:
            self._leg_pd_sample_hold = install_explicit_pd_sample_hold(
                self.robot.actuators['leg_motors'], self.cfg.sim.dt,
                self.cfg.leg_pd_update_period_s)
        from .leg_transport_delay import install_leg_transport
        self._leg_transport = install_leg_transport(
            self.robot.actuators['leg_motors'], self.cfg.sim.dt,
            self.cfg.leg_transport_delay_s, self.cfg.leg_transport_delay_probability,
            self.cfg.leg_transport_delay_seed)
        self._motor_responses = {}
        if self.cfg.motor_response_enabled:
            for group in ('leg_motors', 'wheel_motors'):
                self._motor_responses[group] = install_effort_response(
                    self.robot.actuators[group], self.cfg.sim.dt,
                    self.cfg.motor_response_time_constant_min_s,
                    self.cfg.motor_response_time_constant_max_s)
        self._real2sim_wheel = None
        if self.cfg.real2sim_wheel_profile:
            if not self._separated_wheel_passive or self.cfg.wheel_feedback_damping_mode != 'active_motor_v1':
                raise ValueError('real2sim wheels require separated active_motor_v1')
            if self.cfg.policy_outer_feedback_enabled:
                raise ValueError('real2sim diagnostic forbids task outer feedback')
            from .real2sim_wheel import Real2SimWheel, load_profile
            self._real2sim_wheel = Real2SimWheel(
                self._last_wheel_feedback_torque_nm, self.cfg.sim.dt,
                load_profile(self.cfg.real2sim_wheel_profile))
            # Actuator internal order is not guaranteed to follow policy order.
            motor_to_policy = [self._wheel_actuator.joint_names.index(n) for n in WHEEL_POLICY_DOF_NAMES]
            policy_to_motor = [list(WHEEL_POLICY_DOF_NAMES).index(n) for n in self._wheel_actuator.joint_names]
            self._real2sim_wheel.last_motor_effort_limit.copy_(self._wheel_actuator.effort_limit[:, motor_to_policy])
            wheel_compute = self._wheel_actuator.compute
            wheel_reset = self._wheel_actuator.reset
            def real2sim_compute(control_action, joint_pos, joint_vel):
                action = wheel_compute(control_action, joint_pos, joint_vel)
                effort = self._real2sim_wheel.motor_effort(action.joint_efforts[:, motor_to_policy],
                    self._wheel_actuator.effort_limit[:, motor_to_policy])[:, policy_to_motor]
                self._wheel_actuator.applied_effort = effort
                action.joint_efforts = effort
                return action
            def real2sim_reset(env_ids):
                wheel_reset(env_ids)
                self._real2sim_wheel.reset(env_ids)
            self._wheel_actuator.compute = real2sim_compute
            self._wheel_actuator.reset = real2sim_reset
        self._wheel_authored_effort_limit = torch.full_like(
            self._wheel_effort_limit,
            float(self.cfg.wheel_authored_effort_limit_nm),
        )
        self._wheel_actuator_damping = float(
            self.cfg.robot_cfg.actuators["wheel_motors"].damping
        )
        self._domain_randomization_strength = torch.zeros(
            self.num_envs, device=self.device
        )
        self._domain_base_mass_scale = torch.ones(self.num_envs, device=self.device)
        self._domain_base_inertia_shape_scale = torch.ones(self.num_envs, device=self.device)
        self._domain_base_inertia_scale = torch.ones(self.num_envs, device=self.device)
        self._domain_friction_scale = torch.ones(self.num_envs, device=self.device)
        self._domain_wheel_damping_scale = torch.ones(self.num_envs, device=self.device)
        self._domain_wheel_torque_gain = torch.ones(self.num_envs, device=self.device)
        self._domain_observation_noise_std = torch.zeros(self.num_envs, device=self.device)
        self._domain_observation_delay_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._domain_nominal_root_mass = self.robot.data.body_mass.torch[
            :, self._root_body_id
        ].clone()
        self._domain_nominal_root_inertia = self.robot.data.body_inertia.torch[
            :, self._root_body_id, :
        ].clone()
        from .com_randomization import RootComRandomizer
        self._base_com_randomizer = RootComRandomizer(
            self.robot, self._root_body_id,
            enabled=self.cfg.base_com_randomization_enabled,
            max_offset_flu_m=self.cfg.base_com_max_offset_flu_m)
        self._domain_nominal_material_properties: torch.Tensor | None = None
        self._domain_material_properties: torch.Tensor | None = None
        if self.cfg.domain_randomization_enabled:
            material_properties = wp.to_torch(
                self.robot.root_view.get_material_properties()
            ).clone()
            self._domain_nominal_material_properties = material_properties
            self._domain_material_properties = material_properties.clone()
        self._arena_ground_randomizer = None
        if self.cfg.arena_ground_migration_progress >= 0:
            from .arena_ground import ArenaGroundRandomizer
            self._arena_ground_randomizer = ArenaGroundRandomizer(self)
        self._last_effective_wheel_damping.fill_(self._wheel_actuator_damping)
        self._last_effective_wheel_torque_scale.fill_(self.cfg.wheel_torque_scale_nm)
        self._last_effective_wheel_effort_limit.fill_(
            float(self._wheel_effort_limit.mean())
        )
        if (
            self.cfg.wheel_damping_anneal_target >= 0.0
            and self.cfg.wheel_damping_compensation_ratio > 0.0
        ):
            raise RuntimeError(
                "wheel_damping_anneal_target and wheel_damping_compensation_ratio "
                "cannot both be enabled; choose direct damping migration or feed-forward compensation"
            )
        if (
            self.cfg.wheel_damping_anneal_target >= 0.0
            and self.cfg.wheel_damping_anneal_target > self._wheel_actuator_damping
        ):
            raise ValueError(
                "wheel_damping_anneal_target must not exceed the configured actuator damping"
            )
        if (
            self.cfg.run_mode == "real_hardware"
            and (
                float(self.cfg.wheel_damping_compensation_ratio) > 0.0
                or float(self.cfg.wheel_damping_anneal_target) >= 0.0
            )
        ):
            raise RuntimeError(
                "wheel damping compensation and annealing are simulation-only. "
                "Use the final validated actuator contract for real_hardware "
                "instead of applying simulation-only damping modifications."
            )
        if len(self.cfg.wheel_action_sign) != 2:
            raise RuntimeError("RMUC wheel_action_sign must contain right and left signs")
        self._wheel_action_sign_tensor = torch.tensor(
            self.cfg.wheel_action_sign, device=self.device, dtype=self._wheel_effort_limit.dtype
        )
        self.contact_sensor = self.scene.sensors.get("contact_forces")
        if self.contact_sensor is None:
            raise RuntimeError("RMUC contact sensor was not created for Robot/.*")
        self._contact_sensor_ids = [
            index for index, name in enumerate(self.contact_sensor.body_names)
            if name in {"robot", "left_wheel", "right_wheel"}
        ]
        self._wheel_contact_sensor_ids = [
            index for index, name in enumerate(self.contact_sensor.body_names)
            if name in {"left_wheel", "right_wheel"}
        ]
        if len(self._wheel_contact_sensor_ids) != 2:
            raise RuntimeError(
                "RMUC contact sensor must include left_wheel and right_wheel; "
                f"got {self.contact_sensor.body_names!r}"
            )
        self._curriculum_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._obs_delay_steps = max(int(self.cfg.observation_delay_steps), 0)
        domain_delay_cap = (
            int(self.cfg.domain_randomization_max_observation_delay_steps)
            if (
                self.cfg.domain_randomization_enabled
                and self.cfg.curriculum_stage == 1
                and (
                    not getattr(self.cfg, "play", False)
                    or self.cfg.domain_randomization_apply_in_play
                )
            )
            else 0
        )
        from .gravity_direction_dr import GravityDirectionDR
        self._gravity_direction_dr = GravityDirectionDR(
            self.num_envs, self.device, strength=self.cfg.gravity_direction_dr_strength,
            probability=self.cfg.gravity_direction_dr_probability,
            noise_deg=self.cfg.gravity_direction_dr_noise_deg,
            bias_deg=self.cfg.gravity_direction_dr_bias_deg,
            walk_deg_sqrt_s=self.cfg.gravity_direction_dr_walk_deg_sqrt_s,
            drift_bound_deg=self.cfg.gravity_direction_dr_drift_bound_deg,
            seed=self.cfg.gravity_direction_dr_seed)
        self._sensor_cfg = dict(self.cfg.domain_randomization_sensor_model)
        self._sensor_model_enabled = bool(self._sensor_cfg.get("enabled", False))
        self._sensor_delay_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._communication_delay_steps = torch.zeros_like(self._sensor_delay_steps)
        self._command_action_history = torch.zeros(self.num_envs, int(self._sensor_cfg.get("delay_max_steps", 3))+1, ACTION_DIM, device=self.device)
        self._sensor_gyro_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._sensor_accel_bias = torch.zeros_like(self._sensor_gyro_bias)
        if self._sensor_model_enabled:
            if self.cfg.command_observation_mode != "tracking_error_with_target":
                raise ValueError("Physical sensor DR requires tracking_error_with_target")
            domain_delay_cap = max(domain_delay_cap, int(self._sensor_cfg['delay_max_steps']))
        self._obs_history_depth = max(self._obs_delay_steps, domain_delay_cap)
        self._obs_history = torch.zeros(
            (self.num_envs, self._obs_history_depth + 1, OBSERVATION_DIM), device=self.device
        )
        self._obs_history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._training_iteration = 0
        self._global_step_counter = 0
        self._command_level = int(self.cfg.command_start_level)
        self._domain_randomization_reported_level: int | None = None
        self._manual_command = None
        initial_transition_steps = int(
            self.cfg.command_stage_transition_steps[self._command_level]
        )
        self._command_level_start_step = (
            -initial_transition_steps if self.cfg.command_start_at_target else 0
        )
        # Direct entry above the bridge stage means the loaded checkpoint has
        # already completed the migration. Direct entry at the bridge stage
        # starts from the checkpoint-compatible damping.
        self._wheel_damping_anneal_start_step = (
            -int(self.cfg.wheel_damping_anneal_steps)
            if (
                self.cfg.wheel_damping_anneal_target >= 0.0
                and self._command_level > self.cfg.wheel_damping_anneal_start_level
            )
            else self._global_step_counter
        )
        self._wheel_damping_anneal_transition_count = (
            int(self.cfg.wheel_damping_anneal_steps)
            if (
                self.cfg.wheel_damping_anneal_target >= 0.0
                and self._command_level > self.cfg.wheel_damping_anneal_start_level
            )
            else 0
        )
        self._command_transition_gate_reset = bool(
            self.cfg.command_start_at_target or initial_transition_steps <= 0
        )
        self._command_stable_fraction = torch.zeros((), device=self.device)
        self._command_stable_fraction_ema = torch.zeros((), device=self.device)
        self._command_safe_fraction = torch.zeros((), device=self.device)
        self._command_safe_fraction_ema = torch.zeros((), device=self.device)
        self._command_wheel_contact_fraction = torch.zeros((), device=self.device)
        self._command_wheel_contact_fraction_ema = torch.zeros((), device=self.device)
        self._command_linear_error_ema = torch.zeros((), device=self.device)
        self._command_yaw_error_ema = torch.zeros((), device=self.device)
        self._command_survival_fraction = torch.zeros((), device=self.device)
        self._command_survival_fraction_ema = torch.zeros((), device=self.device)
        self._command_ready_fraction = torch.zeros((), device=self.device)
        self._command_survival_sample_count = 0
        self._command_survival_success_count = 0
        self._command_completed_linear_error_mean = torch.zeros((), device=self.device)
        self._command_completed_yaw_error_mean = torch.zeros((), device=self.device)
        self._command_completed_error_sample_count = 0
        self._command_episode_started = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._command_episode_linear_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._command_episode_yaw_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._command_episode_tracking_step_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._command_gate_sample_count = 0
        self._command_stable_sample_count = 0
        self._command_gate_confirmation_steps = 0
        self._command_velocity_window = torch.zeros(
            (self.num_envs, self.cfg.command_gate_steps), device=self.device
        )
        self._command_target_velocity_window = torch.zeros_like(self._command_velocity_window)
        self._command_yaw_error_window = torch.zeros_like(self._command_velocity_window)
        self._command_velocity_window_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._command_velocity_window_index = 0
        self._command_turn_gate_sample_count = torch.zeros(
            2, dtype=torch.long, device=self.device
        )
        self._command_turn_gate_linear_error_ema = torch.zeros(2, device=self.device)
        self._command_turn_gate_yaw_error_ema = torch.zeros(2, device=self.device)
        self._command_turn_gate_stable_fraction_ema = torch.zeros(2, device=self.device)
        self._command_turn_gate_bank_error_deg_ema = torch.zeros(2, device=self.device)
        self._command_turn_survival_episode_count = torch.zeros(
            2, dtype=torch.long, device=self.device
        )
        self._command_turn_survival_fraction_ema = torch.zeros(2, device=self.device)
        self._rehearsal_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._continuous_command_mask = torch.zeros_like(self._rehearsal_mask)
        self._continuous_commands = torch.zeros(self.num_envs, 3, device=self.device)
        if random_range:
            self._random_speed_sequence = RandomSpeedSequence(
                self.num_envs, self.device, self.step_dt, cfg.episode_length_s,
                random_range, cfg.command_random_hold_s, cfg.command_random_ramp_s,
                cfg.command_random_step_fraction, cfg.command_random_max_delta_mps,
                isolated_rng=cfg.command_random_isolated_rng,
                stop_probability=cfg.command_random_stop_probability)
        self._rehearsal_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._rehearsal_gate_stage_ids = torch.as_tensor(
            self.cfg.command_rehearsal_gate_stage_ids,
            device=self.device,
            dtype=torch.long,
        )
        self._rehearsal_gate_linear_limits = torch.as_tensor(
            [
                self.cfg.command_stage_advance_linear_error_mps[stage]
                for stage in self.cfg.command_rehearsal_gate_stage_ids
            ],
            device=self.device,
        )
        self._rehearsal_gate_turning = torch.as_tensor(
            [
                abs(self.cfg.command_stage_yaw_rate_rad_s[stage]) > 1.0e-4
                for stage in self.cfg.command_rehearsal_gate_stage_ids
            ],
            device=self.device,
            dtype=torch.bool,
        )
        retention_shape = (self._rehearsal_gate_stage_ids.numel(), 2)
        self._rehearsal_gate_sample_count = torch.zeros(
            retention_shape, device=self.device, dtype=torch.long
        )
        self._rehearsal_gate_linear_error_ema = torch.zeros(
            retention_shape, device=self.device
        )
        self._rehearsal_gate_yaw_error_ema = torch.zeros(
            retention_shape, device=self.device
        )
        self._rehearsal_gate_bank_error_deg_ema = torch.zeros_like(self._rehearsal_gate_yaw_error_ema)
        self._rehearsal_gate_stable_fraction_ema = torch.zeros(
            retention_shape, device=self.device
        )
        self._rehearsal_gate_survival_episode_count = torch.zeros(
            retention_shape, device=self.device, dtype=torch.long
        )
        self._rehearsal_gate_survival_fraction_ema = torch.zeros(
            retention_shape, device=self.device
        )
        # Observational diagnostics for the Level-23 zero-sample anomaly.
        # These counters do not affect reward, routing, or gate decisions.
        self._level23_rehearsal_episode_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._level23_rehearsal_episode_length_sum = torch.zeros((), device=self.device)
        self._level23_rehearsal_window_ready_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._level23_rehearsal_steady_ready_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._level23_rehearsal_termination_counts = {
            name: torch.zeros((), dtype=torch.long, device=self.device)
            for name in ("nonfinite", "joint_limit", "tilt", "root_high", "root_low", "xy_distance", "linear_speed")
        }
        self._level23_rehearsal_window_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._level23_rehearsal_steady_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_speed_bootstrap_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._reset_speed_bootstrap_exact_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._reset_speed_bootstrap_speed = torch.zeros(self.num_envs, device=self.device)
        self._reset_speed_bootstrap_pitch_deg = torch.zeros(
            self.num_envs, device=self.device
        )
        self._previous_command_aligned_speed = torch.zeros(self.num_envs, device=self.device)
        self._previous_filtered_command_target = torch.zeros(self.num_envs, device=self.device)
        self._previous_reward_direction = torch.ones(self.num_envs, device=self.device)
        self._previous_command_aligned_speed_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._last_pitch_feedback_action = torch.zeros(self.num_envs, device=self.device)
        self._last_pitch_emergency_feedback_action = torch.zeros(
            self.num_envs, device=self.device
        )
        self._last_yaw_feedback_action = torch.zeros(self.num_envs, device=self.device)
        self._last_raw_policy_actions = torch.zeros(
            (self.num_envs, ACTION_DIM), device=self.device
        )
        stage_lengths = {
            len(self.cfg.command_stage_names),
            len(self.cfg.command_stage_linear_x_mps),
            len(self.cfg.command_stage_yaw_rate_rad_s),
            len(self.cfg.command_stage_leg_scale_rad),
            len(self.cfg.command_stage_transition_steps),
            len(self.cfg.command_stage_survival_gate_steps),
        }
        if len(stage_lengths) != 1 or not self.cfg.command_stage_names:
            raise RuntimeError("RMUC command stages must define equally sized non-empty name/vx/yaw/leg-scale arrays")
        if not 0 <= self._command_level < len(self.cfg.command_stage_names):
            raise RuntimeError(
                f"RMUC command start level {self._command_level} is outside "
                f"[0, {len(self.cfg.command_stage_names) - 1}]"
            )
        self._command_turn_sign = torch.where(
            torch.arange(self.num_envs, device=self.device) % 2 == 0,
            torch.ones(self.num_envs, device=self.device),
            -torch.ones(self.num_envs, device=self.device),
        )
        cohort_fraction = float(self.cfg.command_turn_positive_fraction_override)
        if cohort_fraction == -1.0:
            cohort_fraction = 0.5
        self._turn_fixed_sign_mask = (fixed_turn_sign_cohorts(
            self.num_envs, cohort_fraction, self.cfg.seed + 6119, self.device)
            if self.cfg.command_turn_fixed_sign_cohorts else None)
        self._turn_episode_assignment_count = torch.zeros(
            (len(self.cfg.command_stage_names), 2),
            dtype=torch.long,
            device=self.device,
        )
        self._turn_positive_assignment_credit = torch.zeros(
            len(self.cfg.command_stage_names), dtype=torch.float64, device=self.device
        )
        self._refresh_rehearsal_assignments()
        self._rebalance_turn_assignments(torch.arange(self.num_envs, device=self.device))
        leg_knots = {0.0: self.cfg.stand_leg_position_scale_rad}
        leg_knots.update(zip(self.cfg.command_stage_linear_x_mps, self.cfg.command_stage_leg_scale_rad))
        ordered_knots = sorted(leg_knots)
        self._manual_leg_speed_knots = torch.tensor(ordered_knots, device=self.device)
        self._manual_leg_scale_knots = torch.tensor([leg_knots[k] for k in ordered_knots], device=self.device)
        if cfg.command_leg_scale_profile is not None:
            speeds, scales = validate_leg_scale_profile(cfg.command_leg_scale_profile)
            cfg.command_leg_scale_profile = dict(schema_version=1, speed_abs_mps=speeds, scale_rad=scales)
            self._manual_leg_speed_knots = torch.tensor(speeds, device=self.device)
            self._manual_leg_scale_knots = torch.tensor(scales, device=self.device)
        self._termination_debug_printed = False

    def set_training_progress(self, iteration: int) -> None:
        self._training_iteration = max(int(iteration), 0)

    def ignore_current_episodes_for_survival_gate(self) -> None:
        """Exclude initially phase-randomized partial episodes from survival."""

        self._command_episode_started.zero_()

    def _reset_active_command_gate_measurements(self) -> None:
        """Start a clean gate measurement period without changing simulation state."""

        self._command_stable_fraction.zero_()
        self._command_stable_fraction_ema.zero_()
        self._command_safe_fraction.zero_()
        self._command_safe_fraction_ema.zero_()
        self._command_wheel_contact_fraction.zero_()
        self._command_wheel_contact_fraction_ema.zero_()
        self._command_linear_error_ema.zero_()
        self._command_yaw_error_ema.zero_()
        self._command_survival_fraction.zero_()
        self._command_survival_fraction_ema.zero_()
        self._command_ready_fraction.zero_()
        self._command_survival_sample_count = 0
        self._command_survival_success_count = 0
        self._command_completed_linear_error_mean.zero_()
        self._command_completed_yaw_error_mean.zero_()
        self._command_completed_error_sample_count = 0
        self._command_episode_started.zero_()
        self._command_episode_linear_error_sum.zero_()
        self._command_episode_yaw_error_sum.zero_()
        self._command_episode_tracking_step_count.zero_()
        self._command_gate_sample_count = 0
        self._command_stable_sample_count = 0
        self._command_gate_confirmation_steps = 0
        self._command_velocity_window.zero_()
        self._command_target_velocity_window.zero_()
        self._command_yaw_error_window.zero_()
        self._command_velocity_window_count.zero_()
        self._command_velocity_window_index = 0
        self._command_turn_gate_sample_count.zero_()
        self._command_turn_gate_linear_error_ema.zero_()
        self._command_turn_gate_yaw_error_ema.zero_()
        self._command_turn_gate_stable_fraction_ema.zero_()
        self._command_turn_gate_bank_error_deg_ema.zero_()
        self._command_turn_survival_episode_count.zero_()
        self._command_turn_survival_fraction_ema.zero_()
        self._turn_episode_assignment_count.zero_()
        self._turn_positive_assignment_credit.zero_()

    def _rebalance_turn_assignments(self, env_ids: torch.Tensor) -> None:
        """Accumulate stage-local direction credit at each batch's current focus.

        Carry only fractional rounding credit. Changing focus must not
        reinterpret old assignments and create a long one-direction catch-up.
        Episode counts remain separate diagnostic totals.
        """

        if self.cfg.curriculum_stage != 1:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        assigned_stage = torch.where(
            self._rehearsal_mask[env_ids],
            self._rehearsal_stage[env_ids],
            torch.full_like(env_ids, self._command_level),
        )
        for stage_tensor in torch.unique(assigned_stage):
            stage = int(stage_tensor)
            stage_ids = env_ids[assigned_stage == stage]
            if stage_ids.numel() == 0:
                continue
            if getattr(self, '_turn_fixed_sign_mask', None) is not None:
                positive = self._turn_fixed_sign_mask[stage_ids]
                self._command_turn_sign[stage_ids] = torch.where(
                    positive, torch.ones_like(self._command_turn_sign[stage_ids]),
                    -torch.ones_like(self._command_turn_sign[stage_ids]))
                self._turn_episode_assignment_count[stage] += torch.bincount(
                    (~positive).long(), minlength=2)
                continue
            positive_fraction = self._turn_positive_fraction_for_stage(stage)
            credit = (
                self._turn_positive_assignment_credit[stage]
                + stage_ids.numel() * positive_fraction
            )
            positive_count = torch.clamp(
                torch.round(credit).to(torch.long),
                min=0,
                max=stage_ids.numel(),
            )
            self._turn_positive_assignment_credit[stage] = credit - positive_count
            direction_indices = (
                torch.arange(stage_ids.numel(), device=self.device) >= positive_count
            ).to(torch.long)
            direction_indices = direction_indices[
                torch.randperm(stage_ids.numel(), device=self.device)
            ]
            self._command_turn_sign[stage_ids] = torch.where(
                direction_indices == 0,
                torch.ones_like(direction_indices, dtype=self._command_turn_sign.dtype),
                -torch.ones_like(direction_indices, dtype=self._command_turn_sign.dtype),
            )
            self._turn_episode_assignment_count[stage] += torch.bincount(
                direction_indices, minlength=2
            )

    def _turn_positive_fraction_for_stage(self, stage: int) -> float:
        """Focus bounded episode coverage on one stage's weaker yaw sign."""

        override = float(getattr(self.cfg, 'command_turn_positive_fraction_override', -1.0))
        if override != -1.0:
            if not 0.0 <= override <= 1.0:
                raise ValueError('Direction sampling override must be -1 or within [0,1]')
            return override

        if (
            self._command_level < self.cfg.command_turn_adaptive_focus_start_level
            or abs(self.cfg.command_stage_yaw_rate_rad_s[stage]) <= 1.0e-4
        ):
            return 0.5
        if stage == self._command_level:
            samples = self._command_turn_gate_sample_count
            linear_error = self._command_turn_gate_linear_error_ema
            yaw_error = self._command_turn_gate_yaw_error_ema
            bank_error = self._command_turn_gate_bank_error_deg_ema
            stable = self._command_turn_gate_stable_fraction_ema
            survival_episodes = self._command_turn_survival_episode_count
            survival_fraction = self._command_turn_survival_fraction_ema
        else:
            matches = (self._rehearsal_gate_stage_ids == stage).nonzero(
                as_tuple=False
            ).flatten()
            if matches.numel() != 1:
                return 0.5
            index = int(matches[0])
            samples = self._rehearsal_gate_sample_count[index]
            linear_error = self._rehearsal_gate_linear_error_ema[index]
            yaw_error = self._rehearsal_gate_yaw_error_ema[index]
            bank_error = self._rehearsal_gate_bank_error_deg_ema[index]
            stable = self._rehearsal_gate_stable_fraction_ema[index]
            survival_episodes = self._rehearsal_gate_survival_episode_count[index]
            survival_fraction = self._rehearsal_gate_survival_fraction_ema[index]
        # Window samples can be abundant while one direction has too few
        # complete episodes. Obtain balanced certification evidence first;
        # an 80/20 deficit focus was starving already-good directions.
        if torch.any(survival_episodes < self.cfg.command_rehearsal_gate_min_episodes):
            return 0.5
        min_samples = min(
            int(self.cfg.command_turn_gate_min_samples_per_direction),
            int(self.cfg.command_rehearsal_gate_min_samples),
        )
        if torch.any(samples < min_samples):
            # Build comparable statistics first; direct resets to the direction
            # with fewer samples without permanently changing the 50/50 quota.
            if samples[0] < samples[1]:
                return self.cfg.command_turn_adaptive_focus_fraction
            if samples[1] < samples[0]:
                return 1.0 - self.cfg.command_turn_adaptive_focus_fraction
            return 0.5
        linear_limit = max(self._command_linear_error_limit(stage), 1.0e-6)
        yaw_limit = max(float(self.cfg.command_advance_yaw_error_rad_s), 1.0e-6)
        stable_limit = max(float(self.cfg.command_min_stable_env_fraction), 1.0e-6)
        bank_limit = max(float(self.cfg.command_turn_gate_max_bank_error_deg), 1.0e-6)
        both_pass = (
            (linear_error <= linear_limit) & (yaw_error <= yaw_limit)
            & (stable >= stable_limit)
            & ((bank_error <= bank_limit) | (not self.cfg.command_require_analytic_bank))
            & (survival_fraction >= self.cfg.command_min_survival_env_fraction)
        )
        if torch.all(both_pass):
            return 0.5
        deficit = (
            linear_error / linear_limit
            + yaw_error / yaw_limit
            + torch.clamp(bank_error / bank_limit - 1.0, min=0.0) * self.cfg.command_require_analytic_bank
            + torch.clamp(stable_limit - stable, min=0.0) / stable_limit
        )
        survival_ready = survival_episodes > 0
        survival_deficit = torch.clamp(
            self.cfg.command_min_survival_env_fraction
            - survival_fraction,
            min=0.0,
        ) / max(float(self.cfg.command_min_survival_env_fraction), 1.0e-6)
        deficit = deficit + torch.where(
            survival_ready, survival_deficit, torch.zeros_like(survival_deficit)
        )
        if torch.abs(deficit[0] - deficit[1]) < 0.05:
            return 0.5
        if deficit[0] > deficit[1]:
            return self.cfg.command_turn_adaptive_focus_fraction
        return 1.0 - self.cfg.command_turn_adaptive_focus_fraction

    def _update_active_turn_gate_stats(
        self,
        *,
        stable: torch.Tensor,
        window_linear_error: torch.Tensor,
        window_yaw_error: torch.Tensor,
        bank_error_deg: torch.Tensor,
        command: torch.Tensor,
        window_ready: torch.Tensor,
    ) -> None:
        """Track current-stage left/right turning quality independently."""

        stage_yaw = abs(self.cfg.command_stage_yaw_rate_rad_s[self._command_level])
        if stage_yaw <= 1.0e-4:
            return
        direction_mask = torch.stack(
            (command[:, 2] > 1.0e-4, command[:, 2] < -1.0e-4), dim=-1
        )
        sample_mask = window_ready.unsqueeze(-1) & direction_mask
        sample_count = sample_mask.sum(dim=0)
        valid = sample_count > 0
        denominator = sample_count.clamp_min(1).to(window_linear_error.dtype)
        linear_mean = (
            window_linear_error.unsqueeze(-1) * sample_mask
        ).sum(dim=0) / denominator
        yaw_mean = (
            window_yaw_error.unsqueeze(-1) * sample_mask
        ).sum(dim=0) / denominator
        stable_mean = (
            stable.unsqueeze(-1).to(window_linear_error.dtype) * sample_mask
        ).sum(dim=0) / denominator
        bank_error_mean = (
            bank_error_deg.unsqueeze(-1) * sample_mask
        ).sum(dim=0) / denominator
        previous_count = self._command_turn_gate_sample_count
        updated_count = previous_count + sample_count
        ema_denominator = torch.minimum(
            updated_count,
            torch.full_like(
                updated_count,
                self.cfg.command_turn_gate_min_samples_per_direction,
            ),
        ).clamp_min(1)
        alpha = sample_count.to(window_linear_error.dtype) / ema_denominator.to(
            window_linear_error.dtype
        )
        first = previous_count == 0

        def update(target: torch.Tensor, value: torch.Tensor) -> None:
            blended = target + alpha * (value - target)
            candidate = torch.where(first, value, blended)
            target.copy_(torch.where(valid, candidate, target))

        update(self._command_turn_gate_linear_error_ema, linear_mean)
        update(self._command_turn_gate_yaw_error_ema, yaw_mean)
        update(self._command_turn_gate_stable_fraction_ema, stable_mean)
        update(self._command_turn_gate_bank_error_deg_ema, bank_error_mean)
        self._command_turn_gate_sample_count.copy_(updated_count)

    def _active_turn_gate_status(self) -> dict[str, torch.Tensor]:
        """Require both yaw signs to pass at a current turning stage."""

        true = torch.ones((), device=self.device, dtype=torch.bool)
        zero = torch.zeros((), device=self.device)
        one = torch.ones((), device=self.device)
        stage_yaw = abs(self.cfg.command_stage_yaw_rate_rad_s[self._command_level])
        if stage_yaw <= 1.0e-4:
            return {
                "bank": true,
                "ready": true,
                "linear": true,
                "yaw": true,
                "stable": true,
                "survival": true,
                "ready_fraction": one,
                "max_linear_error": zero,
                "max_yaw_error": zero,
                "min_stable_fraction": one,
                "min_sample_count": zero,
                "max_bank_error_deg": zero,
                "min_survival_fraction": one,
                "min_survival_episode_count": zero,
            }
        ready_mask = self._command_turn_gate_sample_count >= int(
            self.cfg.command_turn_gate_min_samples_per_direction
        )
        ready = torch.all(ready_mask)
        survival_ready = torch.all(
            self._command_turn_survival_episode_count
            >= int(self.cfg.command_rehearsal_gate_min_episodes)
        )
        linear_limit = self._command_linear_error_limit()
        return {
            "bank": ready & (torch.all(self._command_turn_gate_bank_error_deg_ema
                                       <= self.cfg.command_turn_gate_max_bank_error_deg)
                              | (not self.cfg.command_require_analytic_bank)),
            "ready": ready,
            "linear": ready
            & torch.all(self._command_turn_gate_linear_error_ema <= linear_limit),
            "yaw": ready
            & torch.all(
                self._command_turn_gate_yaw_error_ema
                <= self.cfg.command_advance_yaw_error_rad_s
            ),
            "stable": ready
            & torch.all(
                self._command_turn_gate_stable_fraction_ema
                >= self.cfg.command_min_stable_env_fraction
            ),
            "survival": survival_ready
            & torch.all(
                self._command_turn_survival_fraction_ema
                >= self.cfg.command_min_survival_env_fraction
            ),
            "ready_fraction": ready_mask.float().mean(),
            "max_linear_error": self._command_turn_gate_linear_error_ema.max(),
            "max_yaw_error": self._command_turn_gate_yaw_error_ema.max(),
            "min_stable_fraction": self._command_turn_gate_stable_fraction_ema.min(),
            "min_sample_count": self._command_turn_gate_sample_count.min().float(),
            "max_bank_error_deg": self._command_turn_gate_bank_error_deg_ema.max(),
            "min_survival_fraction": self._command_turn_survival_fraction_ema.min(),
            "min_survival_episode_count": self._command_turn_survival_episode_count.min().float(),
        }

    def _find_joint_ids(self, names: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for name in names:
            found, _ = self.robot.find_joints(name)
            if len(found) != 1:
                raise RuntimeError(f"Expected exactly one RMUC joint named {name!r}, found {len(found)}")
            ids.append(int(found[0]))
        return ids

    def _find_body_ids(self, names: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for name in names:
            found, _ = self.robot.find_bodies(name)
            if len(found) != 1:
                raise RuntimeError(f"Expected exactly one RMUC body named {name!r}, found {len(found)}")
            ids.append(int(found[0]))
        return ids

    def _refresh_rehearsal_assignments(self, env_ids: torch.Tensor | None = None) -> None:
        """Assign fixed previous-stage commands for on-policy rehearsal."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if (
            not self.cfg.command_rehearsal_enabled
            or self._command_level < self.cfg.command_rehearsal_start_level
        ):
            self._rehearsal_mask[env_ids] = False
            self._rehearsal_stage[env_ids] = 0
            return
        self._rehearsal_mask[env_ids] = False
        # Reset batches are often one or two environments. Maintain a global
        # quota so asynchronous failures cannot drive the rehearsal population
        # toward either zero or a survivor-biased majority.
        speed_focus = (
            self.cfg.command_rehearsal_speed_focus_start_level
            <= self._command_level
            < self.cfg.command_rehearsal_speed_focus_end_level
        )
        rehearsal_fraction = (
            self.cfg.command_rehearsal_speed_focus_fraction
            if speed_focus
            else self.cfg.command_rehearsal_fraction
        )
        # Maintain a population quota rather than drawing Bernoulli on every
        # reset. High-speed active episodes terminate more often than low-speed
        # rehearsal episodes; independent reset-time draws therefore create a
        # survivor bias that made a configured 15% cohort grow beyond 35%.
        # Clear the resetting environments, then fill only the missing global
        # slots. Commands remain fixed for each episode.
        target_rehearsal_count = int(round(self.num_envs * rehearsal_fraction))
        current_rehearsal_count = int(self._rehearsal_mask.sum().item())
        missing_rehearsal_count = max(
            target_rehearsal_count - current_rehearsal_count, 0
        )
        replay_count = min(missing_rehearsal_count, int(env_ids.numel()))
        if replay_count > 0:
            replay_ids = env_ids[
                torch.randperm(env_ids.numel(), device=self.device)[:replay_count]
            ]
        else:
            replay_ids = env_ids[:0]
        rehearsal_count = int(replay_ids.numel())
        if rehearsal_count > 0:
            self._rehearsal_mask[replay_ids] = True
            # Keep the immediately previous stage and a fixed bank of
            # low/medium turning anchors.  A single previous-stage replay
            # cohort lets early turning skills disappear once speed increases.
            # The retention gate may nominate a certified stage (for example
            # stage 13 at the first 2.0 m/s turn) in the priority list even if
            # it is not part of the older generic anchor list.  Build the
            # candidate pool from both lists. A completed stage may have a
            # higher speed than the current stage (for example 3.2 m/s
            # straight retained while training a 3.0 m/s turn), so filtering
            # by the current target speed would make its mandatory gate bucket
            # stay at zero samples forever.
            candidate_stage_ids = tuple(dict.fromkeys(
                (*self.cfg.command_rehearsal_anchor_stage_ids,
                 *self.cfg.command_rehearsal_priority_stage_ids)
            ))
            candidates = [
                stage
                for stage in candidate_stage_ids
                if 0 <= stage < self._command_level
            ]
            previous_stage = max(self._command_level - 1, 0)
            if previous_stage not in candidates:
                candidates.append(previous_stage)
            if not candidates:
                candidates = [0]
            priority_candidates = [
                stage
                for stage in self.cfg.command_rehearsal_priority_stage_ids
                if stage in candidates
            ]
            priority_count = (
                min(
                    rehearsal_count,
                    int(round(rehearsal_count * self.cfg.command_rehearsal_priority_fraction)),
                )
                if priority_candidates
                else 0
            )
            if priority_candidates and priority_count > 0:
                priority_tensor = torch.as_tensor(
                    priority_candidates, device=self.device, dtype=torch.long
                )
                priority_weight_by_stage = dict(
                    zip(
                        self.cfg.command_rehearsal_priority_stage_ids,
                        self.cfg.command_rehearsal_priority_stage_weights,
                    )
                )
                priority_weights = torch.as_tensor(
                    [priority_weight_by_stage[stage] for stage in priority_candidates],
                    device=self.device,
                    dtype=torch.float32,
                )
                # Reset batches are commonly a single environment. Starting a
                # deterministic round-robin at bucket zero on every reset made
                # all long-running rehearsal assignments collapse to the first
                # priority stage (straight), eliminating the yaw teacher data.
                priority_bucket = torch.multinomial(
                    priority_weights,
                    priority_count,
                    replacement=True,
                )
                sampled_stages = priority_tensor[priority_bucket]
                if getattr(self.cfg, 'command_rehearsal_stratified_coverage', False):
                    sampled_stages = ensure_priority_sign_coverage(
                        sampled_stages,
                        self._turn_fixed_sign_mask[replay_ids[:priority_count]],
                        priority_tensor,
                    )
                self._rehearsal_stage[replay_ids[:priority_count]] = sampled_stages
            general_ids = replay_ids[priority_count:]
            general_candidates = [stage for stage in candidates if stage not in priority_candidates]
            if general_ids.numel() > 0:
                if not general_candidates:
                    general_candidates = candidates
                candidate_tensor = torch.as_tensor(
                    general_candidates, device=self.device, dtype=torch.long
                )
                bucket = torch.randint(
                    candidate_tensor.numel(), (general_ids.numel(),), device=self.device
                )
                self._rehearsal_stage[general_ids] = candidate_tensor[bucket]
        non_replay_ids = env_ids[~self._rehearsal_mask[env_ids]]
        self._rehearsal_stage[non_replay_ids] = max(self._command_level - 1, 0)

    def _reset_rehearsal_gate_stats(self) -> None:
        self._rehearsal_gate_sample_count.zero_()
        self._rehearsal_gate_linear_error_ema.zero_()
        self._rehearsal_gate_yaw_error_ema.zero_()
        self._rehearsal_gate_bank_error_deg_ema.zero_()
        self._rehearsal_gate_stable_fraction_ema.zero_()
        self._rehearsal_gate_survival_episode_count.zero_()
        self._rehearsal_gate_survival_fraction_ema.zero_()

    def _command_linear_error_limit(self, level: int | None = None) -> float:
        """Return the configured absolute speed gate for one command level."""

        stage = self._command_level if level is None else int(level)
        stage = min(max(stage, 0), len(self.cfg.command_stage_advance_linear_error_mps) - 1)
        return float(self.cfg.command_stage_advance_linear_error_mps[stage])

    def _rehearsal_gate_status(self) -> dict[str, torch.Tensor]:
        """Return retention checks for completed left/right turn rehearsal windows."""

        active = self._rehearsal_gate_stage_ids < self._command_level
        true = torch.ones((), device=self.device, dtype=torch.bool)
        zero = torch.zeros((), device=self.device)
        one = torch.ones((), device=self.device)
        enforce = self._command_level >= int(
            self.cfg.command_rehearsal_gate_enforce_start_level
        )
        enforced = torch.as_tensor(
            enforce,
            device=self.device,
            dtype=torch.bool,
        )
        if not self.cfg.command_rehearsal_enabled or not torch.any(active):
            return {
                "ready": true,
                "linear": true,
                "yaw": true,
                "stable": true,
                "survival": true,
                "bank": true,
                "max_bank_error_deg": zero,
                "ready_fraction": one,
                "max_linear_error": zero,
                "max_yaw_error": zero,
                "min_stable_fraction": one,
                "min_sample_count": zero,
                "min_survival_fraction": one,
                "min_survival_episode_count": zero,
                "enforced": enforced,
            }
        active_samples = self._rehearsal_gate_sample_count[active]
        ready_mask = active_samples >= int(self.cfg.command_rehearsal_gate_min_samples)
        active_linear_error = self._rehearsal_gate_linear_error_ema[active]
        active_yaw_error = self._rehearsal_gate_yaw_error_ema[active]
        active_bank = self._rehearsal_gate_bank_error_deg_ema[active]
        bank_ok = (~self._rehearsal_gate_turning[active, None]) | (active_bank <= self.cfg.command_turn_gate_max_bank_error_deg)
        bank_ok = bank_ok | (not self.cfg.command_require_analytic_bank)
        active_stable = self._rehearsal_gate_stable_fraction_ema[active]
        active_survival_samples = self._rehearsal_gate_survival_episode_count[active]
        active_survival = self._rehearsal_gate_survival_fraction_ema[active]
        linear_limits = self._rehearsal_gate_linear_limits[active].unsqueeze(-1)
        ready = torch.all(ready_mask)
        survival_ready = torch.all(
            active_survival_samples
            >= int(self.cfg.command_rehearsal_gate_min_episodes)
        )
        if not enforce:
            # Straight speed acquisition still rehearses and reports every
            # retained bucket. Certification becomes mandatory at the first
            # high-speed yaw stage, where both turn directions are relevant.
            passed = torch.ones((), device=self.device, dtype=torch.bool)
            return {
                "ready": passed,
                "linear": passed,
                "yaw": passed,
                "stable": passed,
                "survival": passed,
                "bank": passed,
                "max_bank_error_deg": active_bank.max(),
                "ready_fraction": ready_mask.float().mean(),
                "max_linear_error": active_linear_error.max(),
                "max_yaw_error": active_yaw_error.max(),
                "min_stable_fraction": active_stable.min(),
                "min_sample_count": active_samples.min().float(),
                "min_survival_fraction": active_survival.min(),
                "min_survival_episode_count": active_survival_samples.min().float(),
                "enforced": enforced,
            }
        return {
            "ready": ready & survival_ready,
            "bank": ready & torch.all(bank_ok),
            "max_bank_error_deg": active_bank.max(),
            "linear": ready & torch.all(active_linear_error <= linear_limits),
            "yaw": ready
            & torch.all(active_yaw_error <= self.cfg.command_advance_yaw_error_rad_s),
            "stable": ready
            & torch.all(active_stable >= self.cfg.command_min_stable_env_fraction),
            "survival": survival_ready
            & torch.all(
                active_survival >= self.cfg.command_min_survival_env_fraction
            ),
            "ready_fraction": ready_mask.float().mean(),
            "max_linear_error": active_linear_error.max(),
            "max_yaw_error": active_yaw_error.max(),
            "min_stable_fraction": active_stable.min(),
            "min_sample_count": active_samples.min().float(),
            "min_survival_fraction": active_survival.min(),
            "min_survival_episode_count": active_survival_samples.min().float(),
            "enforced": enforced,
        }

    def _update_rehearsal_survival_gate(self, env_ids: torch.Tensor) -> None:
        """Count complete retained episodes, including pre-window failures."""

        if not self.cfg.command_rehearsal_enabled or env_ids.numel() == 0:
            return
        completed = (
            self._command_episode_started[env_ids]
            & self._rehearsal_mask[env_ids]
            & (self._curriculum_step_counter[env_ids] > 0)
        )
        completed_ids = env_ids[completed]
        if completed_ids.numel() == 0:
            return
        level23_completed = self._rehearsal_stage[completed_ids] == 23
        if torch.any(level23_completed):
            level23_ids = completed_ids[level23_completed]
            self._level23_rehearsal_episode_count += int(level23_ids.numel())
            self._level23_rehearsal_episode_length_sum += self._curriculum_step_counter[level23_ids].float().sum()
            self._level23_rehearsal_window_ready_count += self._level23_rehearsal_window_seen[level23_ids].sum()
            self._level23_rehearsal_steady_ready_count += self._level23_rehearsal_steady_seen[level23_ids].sum()
            for name, mask in self._last_termination_reasons.items():
                self._level23_rehearsal_termination_counts[name] += mask[level23_ids].sum()
        completed_stage = self._rehearsal_stage[completed_ids]
        completed_direction = (self._command_turn_sign[completed_ids] < 0.0).long()
        stage_survival_steps = torch.as_tensor(
            self.cfg.command_stage_survival_gate_steps,
            device=self.device,
            dtype=self._curriculum_step_counter.dtype,
        )
        unsafe_completion = torch.stack(
            tuple(self._last_termination_reasons.values()), dim=0
        ).any(dim=0)[completed_ids]
        survived = (
            self._curriculum_step_counter[completed_ids]
            >= stage_survival_steps[completed_stage]
        ) & ~unsafe_completion
        survived = survived.float()
        for gate_index, stage_id in enumerate(self.cfg.command_rehearsal_gate_stage_ids):
            stage_mask = completed_stage == int(stage_id)
            if not torch.any(stage_mask):
                continue
            for direction_index in range(2):
                bucket_mask = stage_mask & (completed_direction == direction_index)
                bucket_count = int(bucket_mask.sum().item())
                if bucket_count == 0:
                    continue
                value = survived[bucket_mask].mean()
                previous_count = int(
                    self._rehearsal_gate_survival_episode_count[
                        gate_index, direction_index
                    ].item()
                )
                updated_count = previous_count + bucket_count
                denominator = min(
                    updated_count,
                    int(self.cfg.command_rehearsal_gate_min_episodes),
                )
                alpha = min(bucket_count / max(float(denominator), 1.0), 1.0)
                target = self._rehearsal_gate_survival_fraction_ema[
                    gate_index, direction_index
                ]
                if previous_count == 0:
                    target.copy_(value)
                else:
                    target.lerp_(value, alpha)
                self._rehearsal_gate_survival_episode_count[
                    gate_index, direction_index
                ] = updated_count

    def _update_rehearsal_gate_stats(
        self,
        *,
        safe: torch.Tensor,
        window_linear_error: torch.Tensor,
        window_yaw_error: torch.Tensor,
        bank_error_deg: torch.Tensor,
        command: torch.Tensor,
        window_ready: torch.Tensor,
    ) -> None:
        """Accumulate balanced stage/direction retention metrics on the GPU."""

        if not self.cfg.command_rehearsal_enabled or self._rehearsal_gate_stage_ids.numel() == 0:
            return
        stage_match = (
            self._rehearsal_mask.unsqueeze(-1)
            & window_ready.unsqueeze(-1)
            & (
                self._rehearsal_stage.unsqueeze(-1)
                == self._rehearsal_gate_stage_ids.unsqueeze(0)
            )
        )
        turn_direction_match = torch.stack(
            (command[:, 2] > 1.0e-4, command[:, 2] < -1.0e-4), dim=-1
        ).unsqueeze(1)
        direction_match = turn_direction_match | (~self._rehearsal_gate_turning)[
            None, :, None
        ]
        sample_mask = stage_match.unsqueeze(-1) & direction_match
        sample_count = sample_mask.sum(dim=0)
        valid = sample_count > 0
        denominator = sample_count.clamp_min(1).to(window_linear_error.dtype)
        linear_mean = (
            window_linear_error[:, None, None] * sample_mask
        ).sum(dim=0) / denominator
        yaw_mean = (window_yaw_error[:, None, None] * sample_mask).sum(dim=0) / denominator
        bank_mean = (bank_error_deg[:, None, None] * sample_mask).sum(dim=0) / denominator
        stage_stable = (
            safe.unsqueeze(-1)
            & (
                window_linear_error.unsqueeze(-1)
                < self._rehearsal_gate_linear_limits.unsqueeze(0)
            )
            & (
                window_yaw_error.unsqueeze(-1)
                < self.cfg.command_advance_yaw_error_rad_s
            )
        )
        stable_mean = (
            stage_stable.unsqueeze(-1).to(window_linear_error.dtype) * sample_mask
        ).sum(dim=0) / denominator
        previous_count = self._rehearsal_gate_sample_count
        updated_count = previous_count + sample_count
        ema_denominator = torch.minimum(
            updated_count,
            torch.full_like(updated_count, self.cfg.command_rehearsal_gate_min_samples),
        ).clamp_min(1)
        alpha = sample_count.to(window_linear_error.dtype) / ema_denominator.to(
            window_linear_error.dtype
        )
        first = previous_count == 0

        def update(target: torch.Tensor, value: torch.Tensor) -> None:
            blended = target + alpha * (value - target)
            candidate = torch.where(first, value, blended)
            target.copy_(torch.where(valid, candidate, target))

        update(self._rehearsal_gate_linear_error_ema, linear_mean)
        update(self._rehearsal_gate_yaw_error_ema, yaw_mean)
        update(self._rehearsal_gate_bank_error_deg_ema, bank_mean)
        update(self._rehearsal_gate_stable_fraction_ema, stable_mean)
        self._rehearsal_gate_sample_count.copy_(updated_count)

    def _setup_scene(self) -> None:
        self.arena_ground_audit = None
        if self.cfg.arena_ground_severity >= 0:
            from .arena_ground import prepare_ground
            self.arena_ground_audit = dict(severity=self.cfg.arena_ground_severity,
                                           targets=prepare_ground(self.cfg),behavior_certified=False)
        if self.cfg.arena_ground_migration_progress >= 0:
            from .arena_ground import prepare_curriculum_ground
            self.arena_ground_audit = prepare_curriculum_ground(self.cfg)
        self.robot = Articulation(self.cfg.robot_cfg)
        if self.cfg.arena_ground_migration_progress >= 0:
            from .arena_ground import bind_curriculum_wheels
            self.arena_ground_audit = bind_curriculum_wheels(sim_utils.get_current_stage(), self.cfg)
        elif self.arena_ground_audit is not None:
            from .arena_ground import bind_wheels
            self.arena_ground_audit.update(bind_wheels(sim_utils.get_current_stage()))
        if self.cfg.friction_randomization_enabled:
            self._randomize_terrain_friction()
        self.cfg.terrain.num_envs = self.cfg.scene.num_envs
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.arena_ground_migration_progress >= 0:
            from .arena_ground import bind_curriculum_clones
            bind_curriculum_clones(sim_utils.get_current_stage(), self.cfg, self.arena_ground_audit)
        self.scene.articulations["robot"] = self.robot
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.78, 0.82))
        light_cfg.func("/World/Light", light_cfg)

    def _randomize_terrain_friction(self) -> None:
        """Sample one bounded terrain friction pair for the robustness stage."""
        material = getattr(self.cfg.terrain, "physics_material", None)
        if material is None:
            return
        rng = random.Random(int(self.cfg.seed) + 7919)
        static_low, static_high = self.cfg.friction_static_range
        dynamic_low, dynamic_high = self.cfg.friction_dynamic_range
        material.static_friction = rng.uniform(float(static_low), float(static_high))
        material.dynamic_friction = rng.uniform(float(dynamic_low), float(dynamic_high))

    def _progressive_domain_randomization_strength(self) -> float:
        """Return flat-domain randomization strength backed by command progress."""

        if (
            not self.cfg.domain_randomization_enabled
            or self.cfg.curriculum_stage != 1
            or (
                getattr(self.cfg, "play", False)
                and not self.cfg.domain_randomization_apply_in_play
            )
        ):
            return 0.0
        if self.cfg.domain_randomization_strength_override >= 0.0:
            return float(self.cfg.domain_randomization_strength_override)
        start_level = self.cfg.domain_randomization_start_level
        full_level = self.cfg.domain_randomization_full_level
        extended_start = getattr(self.cfg, 'domain_randomization_extended_start_level', -1)
        if extended_start >= 0 and self._command_level >= extended_start:
            start_level = extended_start
            full_level = self.cfg.domain_randomization_extended_full_level
        if self._command_level < start_level:
            return 0.0
        transition_steps = max(self._current_command_transition_steps(), 1)
        transition_fraction = min(
            max(
                float(self._global_step_counter - self._command_level_start_step)
                / float(transition_steps),
                0.0,
            ),
            1.0,
        )
        span = max(
            full_level
            - start_level
            + 2,
            1,
        )
        completed = (
            self._command_level
            - start_level
            + 1
            + transition_fraction
        )
        return min(max(completed / float(span), 0.0), 1.0)

    def _apply_progressive_domain_randomization(self, env_ids: torch.Tensor) -> None:
        """Resample mild flat-domain parameters for newly reset environments."""

        if env_ids.numel() == 0:
            return
        strength = self._progressive_domain_randomization_strength()
        self._domain_randomization_strength[env_ids] = strength
        if self._separated_wheel_passive:
            sampled = sample_separated_wheel_parameters(self.cfg, env_ids.numel(), strength, self.device)
            for target, values in zip((self._wheel_mechanical_viscous_per_env,
                                       self._wheel_coulomb_per_env,
                                       self._wheel_feedback_damping_per_env), sampled):
                target[env_ids] = values
            if getattr(self, '_real2sim_wheel', None) is not None:
                # Fixed reported estimates; never recalculate perfect compensation
                # for each random draw or allow reset to restore the old priors.
                self._wheel_mechanical_viscous_per_env[env_ids] = self._real2sim_wheel.mechanical_viscous
                self._wheel_coulomb_per_env[env_ids] = self._real2sim_wheel.coulomb
                self._wheel_feedback_damping_per_env[env_ids] = self._real2sim_wheel.damping_compensation
        for response in self._motor_responses.values():
            response.sample(env_ids, strength)
        self._gravity_direction_dr.reset(env_ids)
        if self._sensor_model_enabled:
            c = self._sensor_cfg
            self._sensor_delay_steps[env_ids] = 0
            self._communication_delay_steps[env_ids] = 0
            self._sensor_gyro_bias[env_ids] = 0
            self._sensor_accel_bias[env_ids] = 0
            if strength > 0:
                n = env_ids.numel()
                from .recovery_delay import communication_delay_probability
                observation_probability=communication_delay_probability(strength,
                    float(c.get('observation_delay_probability',-1.0)))
                self._sensor_delay_steps[env_ids] = torch.where(torch.rand(n, device=self.device) < observation_probability,
                    torch.randint(int(c['delay_min_steps']), int(c['delay_max_steps'])+1, (n,), device=self.device), 0)
                if c.get('action_delay_enabled', True):
                    from .recovery_delay import communication_delay_probability
                    probability=communication_delay_probability(strength,
                        float(c.get('communication_delay_probability',
                            getattr(self.cfg,'recovery_communication_delay_probability_override',-1.0))))
                    self._communication_delay_steps[env_ids] = torch.where(torch.rand(n, device=self.device) < probability,
                        torch.randint(int(c['delay_min_steps']), int(c['delay_max_steps'])+1, (n,), device=self.device), 0)
                self._sensor_gyro_bias[env_ids] = torch.empty(n,3,device=self.device).uniform_(-1,1) * c['gyro_bias_max_rad_s'] * strength
                self._sensor_accel_bias[env_ids] = torch.empty(n,3,device=self.device).uniform_(-1,1) * c['accel_bias_max_mps2'] * strength

        if getattr(self.cfg, 'leg_actuator_dr_enabled', False):
            from .leg_actuator_dr import sample_leg_parameters
            leg_velocity, leg_armature = sample_leg_parameters(
                self._nominal_leg_velocity_limits[env_ids], self._nominal_leg_armature[env_ids],
                strength, self.cfg.leg_actuator_velocity_dr_range, self.cfg.leg_actuator_armature_dr_range)
            self.robot.write_joint_velocity_limit_to_sim_index(
                limits=leg_velocity, joint_ids=self._leg_control_ids, env_ids=env_ids)
            self.robot.write_joint_armature_to_sim_index(
                armature=leg_armature, joint_ids=self._leg_control_ids, env_ids=env_ids)
        if strength <= 0.0:
            return

        count = env_ids.numel()

        def symmetric_scale(max_fraction: float) -> torch.Tensor:
            extent = float(max_fraction) * strength
            return torch.empty(count, device=self.device).uniform_(1.0 - extent, 1.0 + extent)

        mass_scale = symmetric_scale(self.cfg.domain_randomization_max_base_mass_fraction)
        friction_scale = symmetric_scale(self.cfg.domain_randomization_max_friction_fraction)
        damping_scale = (torch.ones_like(mass_scale) if self._separated_wheel_passive
                         else symmetric_scale(self.cfg.domain_randomization_max_wheel_damping_fraction))
        torque_gain = symmetric_scale(self.cfg.domain_randomization_max_wheel_torque_gain_fraction)
        inertia_fraction = float(self.cfg.domain_randomization_max_base_inertia_fraction)
        if not math.isfinite(inertia_fraction) or not 0.0 <= inertia_fraction < 1.0:
            raise ValueError("Independent base inertia fraction must be finite and in [0,1)")
        # Do not consume random draws when disabled: old experiments stay paired.
        inertia_shape_scale = (symmetric_scale(inertia_fraction) if inertia_fraction > 0.0
                               else torch.ones_like(mass_scale))
        self._domain_base_mass_scale[env_ids] = mass_scale
        self._domain_base_inertia_shape_scale[env_ids] = inertia_shape_scale
        self._domain_base_inertia_scale[env_ids] = mass_scale * inertia_shape_scale
        self._domain_friction_scale[env_ids] = friction_scale
        self._domain_wheel_damping_scale[env_ids] = damping_scale
        self._domain_wheel_torque_gain[env_ids] = torque_gain
        self._domain_observation_noise_std[env_ids] = (
            strength * self.cfg.domain_randomization_max_observation_noise_std
        )

        root_mass = self._domain_nominal_root_mass[env_ids] * mass_scale
        root_inertia = (
            self._domain_nominal_root_inertia[env_ids]
            * mass_scale.view(-1, 1)
            * inertia_shape_scale.view(-1, 1)
        )
        self.robot.set_masses_index(
            masses=root_mass.view(-1, 1),
            body_ids=[self._root_body_id],
            env_ids=env_ids,
        )
        self.robot.set_inertias_index(
            inertias=root_inertia.view(-1, 1, 9),
            body_ids=[self._root_body_id],
            env_ids=env_ids,
        )

        self._wheel_actuator.damping[env_ids] = (
            self._wheel_actuator_damping * damping_scale.unsqueeze(-1)
        )

        if (
            self._domain_nominal_material_properties is not None
            and self._domain_material_properties is not None
        ):
            cpu_ids = env_ids.detach().to(device="cpu", dtype=torch.long)
            cpu_scale = friction_scale.detach().to(device="cpu").view(-1, 1)
            self._domain_material_properties[cpu_ids] = (
                self._domain_nominal_material_properties[cpu_ids]
            )
            self._domain_material_properties[cpu_ids, :, 0] *= cpu_scale
            self._domain_material_properties[cpu_ids, :, 1] *= cpu_scale
            self._domain_material_properties[cpu_ids, :, 1] = torch.minimum(
                self._domain_material_properties[cpu_ids, :, 1],
                self._domain_material_properties[cpu_ids, :, 0],
            )
            self.robot.root_view.set_material_properties(
                wp.from_torch(self._domain_material_properties, dtype=wp.float32),
                wp.from_torch(cpu_ids.to(dtype=torch.int32), dtype=wp.int32),
            )

        max_delay = int(self.cfg.domain_randomization_max_observation_delay_steps)
        delay_probability = (
            strength
            * self.cfg.domain_randomization_max_observation_delay_probability
        )
        if max_delay > 0 and delay_probability > 0.0:
            delayed = torch.rand(count, device=self.device) < delay_probability
            sampled_delay = torch.randint(
                1, max_delay + 1, (count,), device=self.device
            )
            self._domain_observation_delay_steps[env_ids] = torch.where(
                delayed, sampled_delay, torch.zeros_like(sampled_delay)
            )
        else:
            self._domain_observation_delay_steps[env_ids] = 0

        if self._domain_randomization_reported_level != self._command_level:
            self._domain_randomization_reported_level = self._command_level
            print(
                "[RMUC_DOMAIN_RANDOMIZATION] "
                f"level={self._command_level} strength={strength:.3f} flat_only=true "
                f"mass=+/-{100.0 * strength * self.cfg.domain_randomization_max_base_mass_fraction:.2f}% "
                f"friction=+/-{100.0 * strength * self.cfg.domain_randomization_max_friction_fraction:.2f}% "
                f"wheel_damping=+/-{0.0 if self._separated_wheel_passive else 100.0 * strength * self.cfg.domain_randomization_max_wheel_damping_fraction:.2f}% "
                f"wheel_gain=+/-{100.0 * strength * self.cfg.domain_randomization_max_wheel_torque_gain_fraction:.2f}%",
                flush=True,
            )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._command_in_physics_step = True
        self._curriculum_step_counter += 1
        self._global_step_counter += self.num_envs
        if self._global_step_counter == self.num_envs:
            print(
                "[RMUC_FIRST_ACTION] "
                f"mean={actions.mean(dim=0).detach().cpu().tolist()} "
                f"abs_max={float(actions.abs().max()):.4f}",
                flush=True,
            )
        if self.cfg.curriculum_stage == 1:
            gravity, linear_velocity, angular_velocity = self._state()
            command = self._current_command()
            wheel_force = torch.linalg.vector_norm(self.contact_sensor.data.net_forces_w[:, self._wheel_contact_sensor_ids], dim=-1)
            wheel_contact = wheel_force > self.cfg.contact_force_threshold_N
            instantaneous_wheel_contact_ok = (
                wheel_contact.float().mean(dim=-1)
                >= self.cfg.command_min_instantaneous_wheel_contact_fraction
            )
            safe = ((-gravity[:, 2] > 0.94)
                    & (torch.abs(linear_velocity[:, 1]) < 0.35)
                    & (self.robot.data.root_pos_w.torch[:, 2] >= self.cfg.command_min_operating_root_height_m))
            heading_velocity = self._heading_forward_velocity(self.robot.data.root_lin_vel_w.torch)
            linear_error = torch.abs(heading_velocity - command[:, 0])
            yaw_error = torch.abs(angular_velocity[:, 2] - command[:, 2])
            window_index = self._command_velocity_window_index
            self._command_velocity_window[:, window_index] = heading_velocity
            self._command_target_velocity_window[:, window_index] = command[:, 0]
            self._command_yaw_error_window[:, window_index] = yaw_error
            self._command_velocity_window_count.clamp_max_(self.cfg.command_gate_steps - 1).add_(1)
            self._command_velocity_window_index = (window_index + 1) % self.cfg.command_gate_steps
            window_count = self._command_velocity_window_count.clamp_min(1).float()
            window_linear_error = (
                torch.abs(
                    self._command_velocity_window
                    - self._command_target_velocity_window
                ).sum(dim=-1)
                / window_count
            )
            mean_yaw_error = self._command_yaw_error_window.sum(dim=-1) / window_count
            command_roll_rad, desired_bank_rad = bank_reference(gravity, command)
            command_bank_error_deg = torch.rad2deg(
                torch.abs(command_roll_rad - desired_bank_rad)
            )
            window_ready = self._command_velocity_window_count >= self.cfg.command_gate_steps
            # Use the same temporal metric for live and complete-episode gates.
            # Instantaneous absolute error made the unavoidable launch transient
            # dominate otherwise stable full-episode tracking.
            # Do not fold the reset acceleration transient into the completed
            # episode tracking metric. The old mask started accumulating as
            # soon as the 300-step rolling window became ready, so a policy
            # already holding 1.75 m/s could remain stuck above the 0.20 m/s
            # gate because its first launch window was permanently averaged
            # into every episode result.
            gate_warmup_steps = max(
                int(self.cfg.command_gate_steps),
                # Tracking keeps the original steady-state horizon even when
                # a stage uses a shorter survival prerequisite. Otherwise a
                # survival-only curriculum change also redefines speed error.
                int(self.cfg.command_survival_gate_steps),
            )
            # ``episode_length_buf`` is intentionally randomized by RSL-RL at
            # rollout start. It is a logging age, not physical time since the
            # last reset, and using it here contaminated v49 completed-error
            # statistics with launch transients. Use the env-owned counter.
            steady_state_episode = self._curriculum_step_counter >= gate_warmup_steps
            level23_rehearsal = self._rehearsal_mask & (self._rehearsal_stage == 23)
            self._level23_rehearsal_window_seen |= level23_rehearsal & window_ready
            self._level23_rehearsal_steady_seen |= (
                level23_rehearsal & window_ready & steady_state_episode
            )
            episode_tracking_mask = (
                ~self._rehearsal_mask
                & ~self._continuous_command_mask
                & ~self._reset_speed_bootstrap_mask
                & self._command_episode_started
                & window_ready
                & steady_state_episode
                & self._acquisition_ramp_tracking_ready()
            )
            self._command_episode_linear_error_sum[episode_tracking_mask] += window_linear_error[
                episode_tracking_mask
            ]
            self._command_episode_yaw_error_sum[episode_tracking_mask] += mean_yaw_error[
                episode_tracking_mask
            ]
            self._command_episode_tracking_step_count[episode_tracking_mask] += 1
            linear_error_limit = self._command_linear_error_limit()
            stable = (safe
                      & (window_linear_error < linear_error_limit)
                      & (mean_yaw_error < self.cfg.command_advance_yaw_error_rad_s))
            self._update_rehearsal_gate_stats(
                safe=(
                    safe
                    & instantaneous_wheel_contact_ok
                ),
                window_linear_error=window_linear_error,
                window_yaw_error=mean_yaw_error,
                bank_error_deg=command_bank_error_deg,
                command=command,
                # Retention is a skill-preservation gate, not an acquisition
                # gate.  Exclude the same launch transient used by the active
                # gate; otherwise every 1200-step episode permanently lowers
                # the retained turn stable fraction.
                window_ready=window_ready & steady_state_episode,
            )
            # Assisted high-speed resets provide on-policy coverage, but may
            # not satisfy the curriculum gate. Only zero-speed-start active
            # episodes can prove that the actor acquired the full transition.
            gate_mask = ~self._rehearsal_mask & ~self._reset_speed_bootstrap_mask & ~self._continuous_command_mask
            gate_window_ready = gate_mask & window_ready & steady_state_episode & self._acquisition_ramp_tracking_ready()
            readiness_mask = gate_mask & window_ready & self._acquisition_ramp_tracking_ready()
            self._update_active_turn_gate_stats(
                stable=(
                    stable
                    & instantaneous_wheel_contact_ok
                ),
                window_linear_error=window_linear_error,
                window_yaw_error=mean_yaw_error,
                bank_error_deg=command_bank_error_deg,
                command=command,
                window_ready=gate_window_ready,
            )
            self._command_ready_fraction = (
                readiness_mask.float().sum() / gate_mask.float().sum().clamp_min(1.0)
            )
            # Readiness is an eligibility condition, not a failed sample.
            # Counting episode ages below gate_steps as unstable capped this
            # metric at roughly (episode_length-gate_steps)/episode_length;
            # with 999-step episodes and a 300-step gate, the configured 0.70
            # threshold was practically unreachable even for a perfect policy.
            self._command_stable_fraction = (
                stable[gate_window_ready].float().mean()
                if torch.any(gate_window_ready)
                else torch.zeros((), device=self.device)
            )
            self._command_safe_fraction = safe[gate_mask].float().mean() if torch.any(gate_mask) else safe.float().mean()
            self._command_wheel_contact_fraction = wheel_contact[gate_mask].float().mean() if torch.any(gate_mask) else wheel_contact.float().mean()
            self._command_gate_sample_count += 1
            ema_alpha = 1.0 / max(
                float(min(self._command_gate_sample_count, self.cfg.command_gate_steps)), 1.0
            )
            if torch.any(gate_window_ready):
                self._command_stable_sample_count += 1
                stable_alpha = 1.0 / max(
                    float(min(self._command_stable_sample_count, self.cfg.command_gate_steps)),
                    1.0,
                )
                self._command_stable_fraction_ema.lerp_(
                    self._command_stable_fraction, stable_alpha
                )
                self._command_linear_error_ema.lerp_(
                    window_linear_error[gate_window_ready].mean(), stable_alpha
                )
            self._command_safe_fraction_ema.lerp_(self._command_safe_fraction, ema_alpha)
            self._command_wheel_contact_fraction_ema.lerp_(
                self._command_wheel_contact_fraction, ema_alpha
            )
            self._command_yaw_error_ema.lerp_(
                mean_yaw_error[gate_window_ready].mean()
                if torch.any(gate_window_ready)
                else (yaw_error[gate_mask].mean() if torch.any(gate_mask) else yaw_error.mean()),
                ema_alpha,
            )
            stage_elapsed_steps = self._global_step_counter - self._command_level_start_step
            transition_steps = self._current_command_transition_steps()
            hold_steps = int(self.cfg.command_stage_hold_steps)
            # A level must first reach its final command and then spend the
            # full hold window there.  Previously the transition itself was
            # counted as the hold, allowing Level 5 to advance immediately
            # after the 1.0 -> 1.5 m/s ramp with almost no 1.5 m/s training.
            transition_complete = stage_elapsed_steps >= transition_steps
            hold_reached = transition_complete and stage_elapsed_steps >= transition_steps + hold_steps
            if transition_complete and not self._command_transition_gate_reset:
                self._reset_active_command_gate_measurements()
                self._command_transition_gate_reset = True
                print(
                    f"[RMUC_CURRICULUM] level={self._command_level} transition complete; "
                    "cleared ramp-era active gate metrics and started hold-only validation.",
                    flush=True,
                )
            last_level = len(self.cfg.command_stage_names) - 1
            rehearsal_gate = self._rehearsal_gate_status()
            active_turn_gate = self._active_turn_gate_status()
            gate_conditions_met = (
                hold_reached
                and self._wheel_damping_anneal_gate_complete()
                and self._command_safe_fraction_ema >= self.cfg.command_min_posture_env_fraction
                and self._command_wheel_contact_fraction_ema >= self.cfg.command_min_wheel_contact_fraction
                and self._command_ready_fraction >= self.cfg.command_min_ready_env_fraction
                and self._command_stable_fraction_ema >= self.cfg.command_min_stable_env_fraction
                and self._command_linear_error_ema <= linear_error_limit
                and self._command_yaw_error_ema <= self.cfg.command_advance_yaw_error_rad_s
                and self._command_survival_sample_count >= max(int(gate_mask.sum()), 1)
                and self._command_survival_fraction_ema >= self.cfg.command_min_survival_env_fraction
                and self._command_completed_error_sample_count >= max(int(gate_mask.sum()), 1)
                and self._command_completed_linear_error_mean <= linear_error_limit
                and self._command_completed_yaw_error_mean <= self.cfg.command_advance_yaw_error_rad_s
                and active_turn_gate["ready"]
                and active_turn_gate["linear"]
                and active_turn_gate["yaw"]
                and active_turn_gate["stable"]
                and active_turn_gate["survival"]
                and active_turn_gate["bank"]
                and rehearsal_gate["ready"]
                and rehearsal_gate["linear"]
                and rehearsal_gate["yaw"]
                and rehearsal_gate["stable"]
                and rehearsal_gate["survival"]
                and rehearsal_gate["bank"]
            )
            if gate_conditions_met:
                self._command_gate_confirmation_steps += 1
            else:
                self._command_gate_confirmation_steps = 0
            if (
                self._command_level < last_level
                and self._command_gate_confirmation_steps
                >= self.cfg.command_advance_confirmation_steps
            ):
                safe_ema = float(self._command_safe_fraction_ema)
                wheel_contact_ema = float(self._command_wheel_contact_fraction_ema)
                linear_error_ema = float(self._command_linear_error_ema)
                yaw_error_ema = float(self._command_yaw_error_ema)
                survival_fraction = float(self._command_survival_fraction_ema)
                stable_fraction = float(self._command_stable_fraction_ema)
                survival_samples = int(self._command_survival_sample_count)
                self._command_level += 1
                self._command_level_start_step = self._global_step_counter
                if self._command_level == self.cfg.wheel_damping_anneal_start_level:
                    self._wheel_damping_anneal_start_step = self._global_step_counter
                    self._wheel_damping_anneal_transition_count = 0
                self._command_transition_gate_reset = (
                    self._current_command_transition_steps() <= 0
                )
                self._refresh_rehearsal_assignments()
                self._reset_rehearsal_gate_stats()
                # Ignore episodes which straddle the level boundary. The next
                # completed episode for each environment is then a clean
                # sample of the new command level.
                self._reset_active_command_gate_measurements()
                self._rebalance_turn_assignments(
                    torch.arange(self.num_envs, device=self.device)
                )
                print(
                    f"[RMUC_CURRICULUM] advanced to level={self._command_level} "
                    f"name={self.cfg.command_stage_names[self._command_level]} "
                    f"posture_ema={safe_ema:.3f} "
                    f"wheel_contact_ema={wheel_contact_ema:.3f} "
                    f"damping_anneal={self._wheel_damping_anneal_progress():.3f} "
                    f"linear_error_ema={linear_error_ema:.3f} "
                    f"yaw_error_ema={yaw_error_ema:.3f} "
                    f"stable_ema={stable_fraction:.3f} "
                    f"survival={survival_fraction:.3f} "
                    f"survival_samples={survival_samples} "
                    f"turn_linear_max={float(active_turn_gate['max_linear_error']):.3f} "
                    f"turn_yaw_max={float(active_turn_gate['max_yaw_error']):.3f} "
                    f"turn_stable_min={float(active_turn_gate['min_stable_fraction']):.3f} "
                    f"turn_bank_error_max_deg={float(active_turn_gate['max_bank_error_deg']):.3f} "
                    f"turn_survival_min={float(active_turn_gate['min_survival_fraction']):.3f} "
                    f"retention_linear_max={float(rehearsal_gate['max_linear_error']):.3f} "
                    f"retention_yaw_max={float(rehearsal_gate['max_yaw_error']):.3f} "
                    f"retention_stable_min={float(rehearsal_gate['min_stable_fraction']):.3f} "
                    f"retention_survival_min={float(rehearsal_gate['min_survival_fraction']):.3f}",
                    flush=True,
                )
        self._update_wheel_damping_anneal_progress()
        self._estop |= ~torch.isfinite(actions).all(dim=-1)
        if self._prepared_policy_action is not None:
            # The trainer already prepared and cached the deterministic
            # outer-loop residual. PPO still stores the raw actor sample and
            # its original log-probability; PhysX consumes this cached action
            # exactly once here.
            self._policy_actions = self._prepared_policy_action
            self._prepared_policy_action = None
        else:
            self._policy_actions = self._compute_policy_action(actions)
        if self._sensor_model_enabled:
            self._command_action_history = torch.roll(self._command_action_history, shifts=1, dims=1)
            self._command_action_history[:, 0] = self._policy_actions
            self._policy_actions = self._command_action_history[
                torch.arange(self.num_envs, device=self.device), self._communication_delay_steps].clone()
        self._apply_curriculum_push()

    def prepare_action_for_policy(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply the same bounded residual that PhysX will execute.

        The training runner calls this immediately after sampling. Caching the
        result lets ``_pre_physics_step`` consume it without a second residual.
        The raw sample remains in the PPO transition because this deterministic
        controller output is not drawn from the actor distribution.
        """

        prepared = self._compute_policy_action(actions)
        self._prepared_policy_action = prepared.detach().clone()
        return prepared

    def _compute_policy_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Return the clipped policy action plus the configured residuals."""

        policy_actions = torch.clamp(
            torch.nan_to_num(
                actions,
                nan=0.0,
                posinf=self.cfg.action_clip,
                neginf=-self.cfg.action_clip,
            ),
            -self.cfg.action_clip,
            self.cfg.action_clip,
        )
        if getattr(self.cfg, 'position_authority_enabled', False):
            if self.cfg.policy_outer_feedback_enabled or self.cfg.action_clip != 1.0:
                raise RuntimeError('Position authority requires E2E and legacy base clip1')
            # Use the exact delivered actor observation, including sensor delay,
            # not a newly sampled command or privileged current state.
            observation = self._position_authority_observation
            from rmuc_position_authority import execute_authority_action
            clean = torch.nan_to_num(actions, nan=0., posinf=1., neginf=-1.)
            policy_actions = execute_authority_action(
                observation, clean,
                getattr(self.cfg, 'position_authority_speed_plateau_end', 1.0),
            )
        base_policy_actions = policy_actions.clone()
        self._last_raw_policy_actions.copy_(base_policy_actions)
        if not getattr(self.cfg, "policy_outer_feedback_enabled", True):
            self._last_speed_feedback_action.zero_()
            self._last_highspeed_propulsion_action.zero_()
            self._last_pitch_feedback_action.zero_()
            self._last_pitch_emergency_feedback_action.zero_()
            self._last_yaw_feedback_action.zero_()
            self._last_action_feedback_delta.zero_()
            return policy_actions
        if self.cfg.curriculum_stage != 1:
            self._last_speed_feedback_action.zero_()
            self._last_highspeed_propulsion_action.zero_()
            self._last_pitch_feedback_action.zero_()
            self._last_action_feedback_delta.zero_()
            return policy_actions

        # Diagnostic opt-in: retained commands use their certified stage's
        # outer-loop controller, rather than losing it solely due to cohort.
        feedback_allowed = (
            torch.ones_like(self._rehearsal_mask)
            if self.cfg.rehearsal_feedback_parity else ~self._rehearsal_mask
        )
        feedback_stage = torch.where(
            self._rehearsal_mask, self._rehearsal_stage,
            torch.full_like(self._rehearsal_stage, self._command_level),
        )
        gravity, _, angular_velocity = self._state()
        command = self._current_command()
        if self.cfg.rehearsal_feedback_parity and self.cfg.rehearsal_feedback_straight_only:
            # Straight experts were certified with propulsion/pitch control.
            # Retained turn experts used the older rehearsal contract; applying
            # a global parity fix regresses their negative-yaw tracking.
            feedback_allowed = (~self._rehearsal_mask) | (
                (command[:, 2].abs() <= 1e-4)
                & (command[:, 0].abs() > self.cfg.wheel_highspeed_propulsion_start_mps)
            )
        if self.cfg.feedback_contract_by_command:
            feedback_allowed = torch.ones_like(self._rehearsal_mask)
        if (self.cfg.feedback_contract_by_command
                or (getattr(self.cfg, "play", False) and self._manual_command is not None)):
            # A controller's global level is fixed while W/S changes speed.
            # Do not enable high-speed pitch control for a low-speed command.
            feedback_stage = torch.where(command[:, 0].abs() >= 2.0,
                                         torch.full_like(feedback_stage, 15),
                                         torch.zeros_like(feedback_stage))
        heading_velocity = self._heading_forward_velocity(
            self.robot.data.root_lin_vel_w.torch
        )
        speed_error = command[:, 0] - heading_velocity
        # Use the same filtered aligned speed that drives the reward for the
        # outer-loop wheel feedback.  This is deliberately shared by training
        # and evaluation; the runner records the resulting action as executed.
        command_direction = torch.sign(command[:, 0])
        target_speed = torch.abs(command[:, 0])
        aligned_speed = heading_velocity * command_direction
        filtered_speed_error = (
            target_speed - self._previous_command_aligned_speed
        ) * command_direction
        feedback_speed_error = torch.where(
            self._previous_command_aligned_speed_valid,
            filtered_speed_error,
            speed_error,
        )
        speed_feedback_gain = torch.where(
            torch.abs(command[:, 0])
            >= self.cfg.wheel_speed_feedback_high_speed_threshold_mps,
            torch.full_like(
                speed_error,
                self.cfg.wheel_speed_feedback_high_speed_action_per_mps,
            ),
            torch.full_like(speed_error, self.cfg.wheel_speed_feedback_action_per_mps),
        )
        if (
            self.cfg.wheel_damping_anneal_target >= 0.0
            and self._command_level >= self.cfg.wheel_damping_anneal_start_level
        ):
            anneal_fraction = self._wheel_damping_anneal_progress()
            target_scale = self.cfg.wheel_speed_feedback_damping_anneal_target_scale
            gain_scale = 1.0 - anneal_fraction * (1.0 - target_scale)
            speed_feedback_gain = speed_feedback_gain * gain_scale
        speed_feedback = torch.clamp(
            feedback_speed_error * speed_feedback_gain,
            -self.cfg.wheel_speed_feedback_action_clip,
            self.cfg.wheel_speed_feedback_action_clip,
        )
        negative_turn_active = (
            (feedback_stage >= 15)
            & (command[:, 2] < -1.0e-4)
            & feedback_allowed
            & (feedback_speed_error > 0.0)
        )
        speed_feedback = torch.where(
            negative_turn_active,
            torch.clamp(
                speed_feedback * self.cfg.wheel_speed_feedback_negative_turn_scale,
                -self.cfg.wheel_speed_feedback_action_clip,
                self.cfg.wheel_speed_feedback_action_clip,
            ),
            speed_feedback,
        )
        highspeed_acquisition = (
            (target_speed > self.cfg.wheel_highspeed_propulsion_start_mps)
            & feedback_allowed
        )
        aligned_speed_feedback = speed_feedback * command_direction
        aligned_speed_feedback = torch.where(
            highspeed_acquisition,
            torch.minimum(
                aligned_speed_feedback,
                torch.full_like(
                    aligned_speed_feedback,
                    self.cfg.wheel_highspeed_feedback_overspeed_action_clip,
                ),
            ),
            aligned_speed_feedback,
        )
        speed_feedback = aligned_speed_feedback * command_direction
        self._last_speed_feedback_action.copy_(speed_feedback)
        policy_actions[:, 2] = torch.clamp(
            policy_actions[:, 2] + speed_feedback,
            -self.cfg.action_clip,
            self.cfg.action_clip,
        )
        policy_actions[:, 5] = torch.clamp(
            policy_actions[:, 5] + speed_feedback,
            -self.cfg.action_clip,
            self.cfg.action_clip,
        )
        propulsion_action = torch.clamp(
            (target_speed - self.cfg.wheel_highspeed_propulsion_start_mps)
            * self.cfg.wheel_highspeed_propulsion_action_per_mps,
            0.0,
            self.cfg.wheel_highspeed_propulsion_action_clip,
        ) * command_direction
        overspeed_fade = torch.clamp(
            (
                target_speed
                + self.cfg.wheel_highspeed_propulsion_overspeed_fade_mps
                - aligned_speed
            )
            / self.cfg.wheel_highspeed_propulsion_overspeed_fade_mps,
            0.0,
            1.0,
        )
        propulsion_action = propulsion_action * overspeed_fade * (
            (feedback_allowed) & (-gravity[:, 2] > 0.85)
        ).to(propulsion_action.dtype)
        self._last_highspeed_propulsion_action.copy_(propulsion_action)
        policy_actions[:, 2] = torch.clamp(
            policy_actions[:, 2] + propulsion_action,
            -self.cfg.action_clip,
            self.cfg.action_clip,
        )
        policy_actions[:, 5] = torch.clamp(
            policy_actions[:, 5] + propulsion_action,
            -self.cfg.action_clip,
            self.cfg.action_clip,
        )
        pitch_feedback = torch.zeros_like(speed_error)
        self._last_pitch_emergency_feedback_action.zero_()
        if self.cfg.feedback_contract_by_command or self._command_level >= self.cfg.wheel_pitch_feedback_start_level:
            pitch_rad = torch.atan2(gravity[:, 0], -gravity[:, 2])
            target_speed = torch.abs(command[:, 0]).clamp_min(0.10)
            underspeed_deficit = torch.clamp(
                target_speed - heading_velocity * torch.sign(command[:, 0]),
                min=0.0,
            )
            desired_pitch_deg = torch.clamp(
                target_speed * self.cfg.command_pitch_target_deg_per_mps
                + underspeed_deficit
                * self.cfg.command_pitch_target_underspeed_gain_deg_per_mps,
                0.0,
                self.cfg.command_pitch_target_max_deg,
            ) * torch.sign(command[:, 0])
            pitch_error_rad = pitch_rad - torch.deg2rad(desired_pitch_deg)
            pitch_rate = angular_velocity[:, 1]
            pitch_emergency_feedback = (
                torch.sign(pitch_rate)
                * torch.clamp(
                    torch.abs(pitch_rate)
                    - self.cfg.wheel_pitch_rate_emergency_threshold_rad_s,
                    min=0.0,
                )
                * self.cfg.wheel_pitch_rate_emergency_action_per_rad_s
            )
            pitch_feedback = torch.clamp(
                pitch_error_rad * self.cfg.wheel_pitch_feedback_action_per_rad
                + pitch_rate * self.cfg.wheel_pitch_rate_feedback_action_per_rad_s
                + pitch_emergency_feedback,
                -self.cfg.wheel_pitch_feedback_action_clip,
                self.cfg.wheel_pitch_feedback_action_clip,
            )
            pitch_feedback = pitch_feedback * (
                (feedback_allowed)
                & (feedback_stage >= self.cfg.wheel_pitch_feedback_start_level)
                & (
                    -gravity[:, 2]
                    > self.cfg.wheel_pitch_feedback_min_upright_cos
                )
            ).to(pitch_feedback.dtype)
            pitch_emergency_feedback = pitch_emergency_feedback * (
                (feedback_allowed)
                & (feedback_stage >= self.cfg.wheel_pitch_feedback_start_level)
                & (
                    -gravity[:, 2]
                    > self.cfg.wheel_pitch_feedback_min_upright_cos
                )
            ).to(pitch_emergency_feedback.dtype)
            self._last_pitch_emergency_feedback_action.copy_(
                pitch_emergency_feedback
            )
            policy_actions[:, 2] = torch.clamp(
                policy_actions[:, 2] + pitch_feedback,
                -self.cfg.action_clip,
                self.cfg.action_clip,
            )
            policy_actions[:, 5] = torch.clamp(
                policy_actions[:, 5] + pitch_feedback,
                -self.cfg.action_clip,
                self.cfg.action_clip,
            )
        self._last_pitch_feedback_action.copy_(pitch_feedback)
        yaw_error = command[:, 2] - angular_velocity[:, 2]
        # Preserve low-speed teacher steering without restoring the unsafe
        # high-speed differential clip that consumed pitch-control headroom.
        turn_feedback_limit = torch.where(
            command[:, 0].abs() < self.cfg.wheel_speed_feedback_high_speed_threshold_mps,
            torch.full_like(yaw_error, self.cfg.wheel_yaw_feedback_low_speed_action_clip),
            torch.full_like(yaw_error, self.cfg.wheel_yaw_feedback_action_clip),
        )
        yaw_feedback_limit = torch.where(
            torch.abs(command[:, 2]) > 1.0e-4,
            turn_feedback_limit,
            torch.full_like(
                yaw_error,
                self.cfg.wheel_yaw_feedback_straight_action_clip,
            ),
        )
        yaw_feedback_raw = yaw_error * self.cfg.wheel_yaw_feedback_action_per_rad_s
        yaw_feedback = torch.maximum(
            torch.minimum(yaw_feedback_raw, yaw_feedback_limit),
            -yaw_feedback_limit,
        )
        # Allocate the differential yaw residual only inside the wheel-action
        # headroom.  Clipping each wheel after applying yaw changes the wheel
        # common mode whenever one side is already saturated, which removes
        # the fore-aft authority needed to catch pitch during fast turns.
        # Keeping right + left invariant makes yaw yield to balance at the
        # actuator boundary; the policy can then learn to create headroom.
        right_wheel_action = policy_actions[:, 2]
        left_wheel_action = policy_actions[:, 5]
        yaw_feedback_min = torch.maximum(
            right_wheel_action - self.cfg.action_clip,
            -self.cfg.action_clip - left_wheel_action,
        )
        yaw_feedback_max = torch.minimum(
            right_wheel_action + self.cfg.action_clip,
            self.cfg.action_clip - left_wheel_action,
        )
        yaw_feedback = torch.maximum(
            torch.minimum(yaw_feedback, yaw_feedback_max),
            yaw_feedback_min,
        )
        self._last_yaw_feedback_action.copy_(yaw_feedback)
        policy_actions[:, 2] = torch.clamp(
            policy_actions[:, 2] - yaw_feedback,
            -self.cfg.action_clip,
            self.cfg.action_clip,
        )
        policy_actions[:, 5] = torch.clamp(
            policy_actions[:, 5] + yaw_feedback,
            -self.cfg.action_clip,
            self.cfg.action_clip,
        )
        self._last_action_feedback_delta.copy_(policy_actions - base_policy_actions)
        return policy_actions

    def _apply_curriculum_push(self) -> None:
        if not self.cfg.push_enabled or self.cfg.push_interval_steps <= 0:
            return
        push_ids = (
            (self._curriculum_step_counter > 0)
            & (self._curriculum_step_counter % int(self.cfg.push_interval_steps) == 0)
        ).nonzero(as_tuple=False).flatten()
        if push_ids.numel() == 0:
            return
        root_lin = self.robot.data.root_lin_vel_w.torch[push_ids].clone()
        root_ang = self.robot.data.root_ang_vel_w.torch[push_ids].clone()
        delta = torch.empty((push_ids.numel(), 2), device=self.device).uniform_(
            -float(self.cfg.push_velocity_xy_mps), float(self.cfg.push_velocity_xy_mps)
        )
        root_lin[:, :2] += delta
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=torch.cat((root_lin, root_ang), dim=-1), env_ids=push_ids
        )

    def _apply_action(self) -> None:
        if getattr(self, "_separated_wheel_passive", False):
            omega = self.robot.data.joint_vel.torch[:, self._wheel_control_ids]
            mechanical = mechanical_torque(omega, self._wheel_mechanical_viscous_per_env,
                                           self._wheel_coulomb_per_env,
                                           (self._real2sim_wheel.profile['friction_epsilon_rad_s']
                                            if getattr(self, '_real2sim_wheel', None) is not None
                                            else self.cfg.wheel_coulomb_omega_eps_rad_s))
            feedback = feedback_damping_torque(omega, self._wheel_feedback_damping_per_env)
            if getattr(self, '_real2sim_wheel', None) is not None:
                feedback = self._real2sim_wheel.feedback(omega)
            feedback_is_active_motor = (
                getattr(self.cfg, "wheel_feedback_damping_mode", "postclip_loss_v1")
                == "active_motor_v1"
            )
            # The active deployment path adds feedback D to the electromagnetic
            # request below, before its sole 5.5 Nm motor clip. Keep mechanical
            # b/C outside the motor in both modes.
            passive = mechanical if feedback_is_active_motor else mechanical + feedback
            self._last_wheel_mechanical_torque_nm.copy_(mechanical)
            self._last_wheel_feedback_torque_nm.copy_(feedback)
            self._last_wheel_total_passive_torque_nm.copy_(passive)
            for index, ids in enumerate(self._wheel_passive_body_ids):
                torques = body_torque_pair(self.robot.data.body_quat_w.torch[:, ids[0]],
                                          self._wheel_passive_axes_b[index], passive[:, index])
                self._last_wheel_mechanical_body_torques_w[:, index].copy_(torques)
                # Composer clears each write_data_to_sim; reapply every physics step.
                # Adds to spring wrenches and bypasses clipping and motor response.
                self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
                    torques=torques, body_ids=ids, is_global=True)
        if self._radial_spring_body_ids is not None:
            for side, ids in enumerate(self._radial_spring_body_ids):
                quat = self.robot.data.body_quat_w.torch[:, ids]
                local_points = self._radial_spring_points_b[side].expand(self.num_envs, -1, -1)
                points = self.robot.data.body_pos_w.torch[:, ids] + quat_apply(
                    quat.reshape(-1, 4), local_points.reshape(-1, 3)).reshape(self.num_envs, 2, 3)
                forces, positions = radial_spring_force_pair(
                    points[:, 0], points[:, 1], self._radial_spring_force_n[:, side])
                self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
                    forces=forces, positions=positions, body_ids=ids, is_global=True)
        if self._gas_spring_body_ids is not None:
            parent_quat = self.robot.data.body_quat_w.torch[:, self._gas_spring_body_ids[0]]
            spring_torques = spring_body_torques(parent_quat,
                self._gas_spring_torque_nm if (self.cfg.gas_spring_torque_common_dr_fraction+self.cfg.gas_spring_torque_independent_dr_fraction)>0
                else self.cfg.gas_spring_assist_torque_nm,
                quaternion_order=self.cfg.gas_spring_quaternion_order)
            # Reapplied each physics substep; composer clears after write_data_to_sim.
            # Accumulate rather than reset other external wrenches. Passive spring
            # forces persist through motor estop and do not consume motor torque limits.
            self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
                torques=spring_torques, body_ids=self._gas_spring_body_ids, is_global=True)
        leg_actions = self._policy_actions[:, [0, 1, 3, 4]]
        if self.cfg.leg_pair_mapping == "common_mode":
            right = 0.5 * (leg_actions[:, 0] + leg_actions[:, 1])
            left = 0.5 * (leg_actions[:, 2] + leg_actions[:, 3])
            leg_actions = torch.stack((right, right, left, left), dim=-1)
        elif self.cfg.leg_pair_mapping != "independent":
            raise RuntimeError(f"Unsupported RMUC leg_pair_mapping: {self.cfg.leg_pair_mapping!r}")

        default_leg_pos = self._default_control_pos[:, [0, 1, 3, 4]]
        if self.cfg.curriculum_stage == 0:
            leg_scale = torch.full(
                (self.num_envs,), self.cfg.stand_leg_position_scale_rad, device=self.device
            )
        else:
            leg_scale = self._current_command_stage_values_tensor()[2]
        target_pos = default_leg_pos + leg_scale.unsqueeze(-1) * leg_actions
        self._last_firmware_spring_torque_nm = torch.zeros_like(target_pos)
        self._last_firmware_spring_l0_m = torch.zeros((self.num_envs, 2), device=self.device)
        if getattr(self.cfg, "firmware_spring_enabled", False):
            if float(self.cfg.firmware_spring_target_kp) <= 0.0:
                raise ValueError("firmware_spring_target_kp must be positive")
            firmware_torque, firmware_l0 = firmware_spring_torque_from_joint_positions(
                self.robot.data.joint_pos.torch[:, self._leg_control_ids]
            )
            self._last_firmware_spring_torque_nm.copy_(firmware_torque)
            self._last_firmware_spring_l0_m.copy_(firmware_l0)
            # V14-style DC-motor equivalent: feed the measured torque through
            # the existing position PD as a bounded target offset.
            spring_scale = float(self.cfg.firmware_spring_scale)
            if not 0.0 <= spring_scale <= 1.0:
                raise ValueError("firmware_spring_scale must be in [0,1]")
            target_pos = target_pos + spring_scale * firmware_torque / float(self.cfg.firmware_spring_target_kp)
        limits = self.robot.data.joint_pos_limits.torch[:, self._leg_control_ids, :]
        lower = limits[..., 0] + self.cfg.leg_joint_limit_margin_rad
        upper = limits[..., 1] - self.cfg.leg_joint_limit_margin_rad
        target_pos = torch.maximum(torch.minimum(target_pos, upper), lower)
        self.robot.set_joint_position_target(target_pos, joint_ids=self._leg_control_ids)

        anneal_fraction = (
            self._wheel_damping_anneal_progress()
            if self.cfg.wheel_damping_anneal_target >= 0.0
            else 0.0
        )
        # The checkpoint used a 5.5 Nm normalized action contract, while the
        # authored wheel drive is 3.0 Nm.  Migrate action scale and explicit
        # actuator clipping with the same cosine bridge as damping so the
        # final policy cannot rely on torque unavailable to hardware.
        base_effective_torque_scale = (
            self.cfg.wheel_torque_scale_nm
            + anneal_fraction
            * (self.cfg.wheel_authored_effort_limit_nm - self.cfg.wheel_torque_scale_nm)
        )
        effective_torque_scale = (
            base_effective_torque_scale * self._domain_wheel_torque_gain
        )
        effective_effort_limit = (
            self._wheel_effort_limit
            + anneal_fraction
            * (self._wheel_authored_effort_limit - self._wheel_effort_limit)
        )
        self._wheel_actuator.effort_limit.copy_(
            effective_effort_limit.unsqueeze(0).expand_as(
                self._wheel_actuator.effort_limit
            )
        )
        self._last_effective_wheel_torque_scale.copy_(effective_torque_scale)
        self._last_effective_wheel_effort_limit.fill_(
            float(effective_effort_limit.mean())
        )
        requested_wheel = (
            self._policy_actions[:, [2, 5]]
            * effective_torque_scale.unsqueeze(-1)
            * self._wheel_action_sign_tensor
        )
        if (
            getattr(self, "_separated_wheel_passive", False)
            and getattr(self.cfg, "wheel_feedback_damping_mode", "postclip_loss_v1") == "active_motor_v1"
        ):
            requested_wheel = requested_wheel + self._last_wheel_feedback_torque_nm
        # IdealPD computes effort_target - damping * joint_velocity. Add the
        # same signed velocity term to the target to cancel only the configured
        # fraction of that artificial damping. The previous subtraction had the
        # wrong sign and would have doubled damping if compensation was enabled.
        damping_ratio = float(self.cfg.wheel_damping_compensation_ratio)
        damping_compensation = torch.zeros_like(requested_wheel)
        if (
            getattr(self.cfg, "policy_outer_feedback_enabled", True)
            and damping_ratio > 0.0
            and self._wheel_actuator_damping > 0.0
            and self.cfg.run_mode != "real_hardware"
            and self.cfg.curriculum_stage == 1
            and self._command_level >= self.cfg.wheel_damping_compensation_start_level
        ):
            gravity, _, _ = self._state()
            command = self._current_command()
            heading_velocity = self._heading_forward_velocity(
                self.robot.data.root_lin_vel_w.torch
            )
            command_direction = torch.sign(command[:, 0])
            aligned_speed = heading_velocity * command_direction
            target_speed = torch.abs(command[:, 0])
            support_progress = torch.clamp(
                (
                    target_speed
                    + self.cfg.wheel_damping_compensation_margin_mps
                    - aligned_speed
                )
                / self.cfg.wheel_damping_compensation_scale_mps,
                0.0,
                1.0,
            )
            support_factor = support_progress.square() * (
                3.0 - 2.0 * support_progress
            )
            upright_factor = torch.clamp((-gravity[:, 2] - 0.85) / 0.10, 0.0, 1.0)
            support_factor = (
                support_factor
                * upright_factor
                * (torch.ones_like(self._rehearsal_mask) if self.cfg.feedback_contract_by_command
                   else ~self._rehearsal_mask).to(requested_wheel.dtype)
            )
            wheel_velocity = self.robot.data.joint_vel.torch[:, self._wheel_control_ids]
            damping_compensation = (
                damping_ratio
                * support_factor.unsqueeze(-1)
                * (
                    self._wheel_actuator_damping
                    * self._domain_wheel_damping_scale
                ).unsqueeze(-1)
                * wheel_velocity
            )
            requested_wheel = requested_wheel + damping_compensation
        # Explicit IdealPD damping is a mutable compute tensor, already changed
        # by reset-time DR. Write it directly: adding velocity feed-forward and
        # then clipping the effort target loses the compensation at saturation.
        anneal_target = float(self.cfg.wheel_damping_anneal_target)
        if (
            anneal_target >= 0.0
            and self.cfg.curriculum_stage == 1
            and self._command_level >= self.cfg.wheel_damping_anneal_start_level
            and self.cfg.run_mode != "real_hardware"
        ):
            anneal_fraction = self._wheel_damping_anneal_progress()
            domain_base_damping = (
                self._wheel_actuator_damping * self._domain_wheel_damping_scale
            )
            domain_target_damping = anneal_target * self._domain_wheel_damping_scale
            effective_damping = (
                domain_base_damping
                - (domain_base_damping - domain_target_damping) * anneal_fraction
            )
            self._wheel_actuator.damping.copy_(
                effective_damping.unsqueeze(-1).expand_as(self._wheel_actuator.damping)
            )
            self._last_wheel_damping_anneal_fraction.fill_(anneal_fraction)
            self._last_effective_wheel_damping.copy_(
                effective_damping
            )
        else:
            self._last_wheel_damping_anneal_fraction.zero_()
            self._last_effective_wheel_damping.copy_(
                self._wheel_actuator_damping * self._domain_wheel_damping_scale
            )
        self._last_wheel_damping_compensation_nm.copy_(damping_compensation)
        self._last_requested_wheel_effort_nm.copy_(requested_wheel)
        # Stand is a static support stage: wheel torque is disabled so the
        # policy cannot exploit rolling/driving to inflate reward or lose both
        # contacts before the legs establish support.
        # Stand is the 2-D inverted-pendulum stage: wheel torque is required
        # to balance the elevated chassis, so do not disable it here.
        rated_limit = effective_effort_limit
        ratio = self.cfg.torque_limit_ratio_real if self.cfg.run_mode == "real_hardware" else self.cfg.torque_limit_ratio_sim
        limit = rated_limit * ratio
        if self.cfg.run_mode == "real_hardware":
            self._estop |= torch.any(torch.abs(requested_wheel) > limit, dim=-1)
        wheel_effort = torch.clamp(requested_wheel, -limit, limit)
        wheel_effort = torch.where(self._estop.unsqueeze(-1), torch.zeros_like(wheel_effort), wheel_effort)
        self._last_wheel_effort_target_nm.copy_(wheel_effort)
        self.robot.set_joint_effort_target(wheel_effort, joint_ids=self._wheel_control_ids)

    def _state(self):
        root_quat_w = self.robot.data.root_quat_w.torch

        def canonical(vector_w: torch.Tensor) -> torch.Tensor:
            # USD world geometry is Z-up, while its rigid-body frames carry
            # +90 deg about X. Map body-frame values back to the canonical
            # forward/lateral/up convention used by the policy.
            source = quat_apply_inverse(root_quat_w, vector_w)
            return torch.stack((-source[:, 2], source[:, 0], source[:, 1]), dim=-1)

        gravity = canonical(self._gravity_w)
        linear_velocity = canonical(self.robot.data.root_lin_vel_w.torch)
        angular_velocity = canonical(self.robot.data.root_ang_vel_w.torch)
        return gravity, linear_velocity, angular_velocity

    def _heading_forward_velocity(self, velocity_w: torch.Tensor) -> torch.Tensor:
        """Project a world velocity onto the robot's horizontal forward heading."""
        authored_forward = torch.zeros_like(velocity_w)
        authored_forward[..., 2] = -1.0
        forward_w = quat_apply(self.robot.data.root_quat_w.torch, authored_forward)
        forward_xy = forward_w[..., :2]
        forward_xy = forward_xy / torch.linalg.vector_norm(
            forward_xy, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        return torch.sum(velocity_w[..., :2] * forward_xy, dim=-1)

    def _locomotion_diagnostics(
        self,
        gravity: torch.Tensor,
        command: torch.Tensor,
        actual_vx: torch.Tensor,
        leg_length_diff_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return bounded/physical diagnostics for command-stage bottlenecks."""

        target_speed = torch.abs(command[:, 0]).clamp_min(0.10)
        command_denominator = torch.where(
            torch.abs(command[:, 0]) >= 0.10,
            command[:, 0],
            torch.where(
                command[:, 0] < 0.0,
                torch.full_like(command[:, 0], -0.10),
                torch.full_like(command[:, 0], 0.10),
            ),
        )
        speed_ratio = torch.clamp(actual_vx / command_denominator, -3.0, 3.0)
        # Canonical x is forward, y is lateral and z is up. Roll is rotation
        # about forward x, so it is encoded by the lateral gravity component.
        roll_rad = torch.atan2(-gravity[:, 1], -gravity[:, 2])
        pitch_rad = torch.atan2(gravity[:, 0], -gravity[:, 2])
        turning = torch.abs(command[:, 2]) > 1.0e-4

        applied_wheel_torque = self.robot.data.applied_torque.torch[
            :, self._wheel_joint_ids_by_body
        ]
        torque_ratio = torch.abs(applied_wheel_torque) / self._wheel_authored_effort_limit.clamp_min(1.0e-6)

        authored_forward = torch.zeros_like(self.robot.data.root_lin_vel_w.torch)
        authored_forward[:, 2] = -1.0
        forward_w = quat_apply(self.robot.data.root_quat_w.torch, authored_forward)
        forward_xy = forward_w[:, :2] / torch.linalg.vector_norm(
            forward_w[:, :2], dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        wheel_velocity_w = self.robot.data.body_lin_vel_w.torch[:, self._wheel_body_ids]
        wheel_center_speed = torch.sum(
            wheel_velocity_w[..., :2] * forward_xy.unsqueeze(1), dim=-1
        )
        wheel_angular_speed = self.robot.data.joint_vel.torch[
            :, self._wheel_joint_ids_by_body
        ]
        wheel_surface_speed = torch.abs(wheel_angular_speed) * self.cfg.wheel_radius_m
        slip_speed = torch.abs(wheel_surface_speed - torch.abs(wheel_center_speed))
        slip_ratio = slip_speed / torch.maximum(
            torch.maximum(wheel_surface_speed, torch.abs(wheel_center_speed)),
            torch.full_like(slip_speed, 0.10),
        )
        wheel_force = torch.linalg.vector_norm(
            self.contact_sensor.data.net_forces_w[:, self._wheel_contact_sensor_ids], dim=-1
        )
        contact = (wheel_force > self.cfg.contact_force_threshold_N).float()
        contact_count = contact.sum(dim=-1).clamp_min(1.0)

        return {
            "speed_ratio": speed_ratio,
            "roll_deg": torch.rad2deg(roll_rad),
            "roll_abs_deg": torch.rad2deg(torch.abs(roll_rad)),
            "pitch_deg": torch.rad2deg(pitch_rad),
            "turn_fraction": turning.float(),
            "turn_roll_abs_deg": torch.rad2deg(torch.abs(roll_rad)) * turning.float(),
            "turn_roll_aligned_deg": (
                torch.rad2deg(roll_rad) * torch.sign(command[:, 2]) * turning.float()
            ),
            "leg_length_diff_m": leg_length_diff_m,
            "turn_leg_length_diff_m": leg_length_diff_m * turning.float(),
            "wheel_torque_abs_mean_nm": torch.abs(applied_wheel_torque).mean(dim=-1),
            "wheel_torque_ratio_mean": torque_ratio.mean(dim=-1),
            "wheel_torque_saturation_fraction": (torque_ratio >= 0.95).float().mean(dim=-1),
            "wheel_slip_speed_mps": (slip_speed * contact).sum(dim=-1) / contact_count,
            "wheel_slip_ratio": (slip_ratio * contact).sum(dim=-1) / contact_count,
        }

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._command_in_physics_step = False
        command = self._current_command()
        gravity, linear_velocity, angular_velocity = self._state()
        control_pos = self.robot.data.joint_pos.torch[:, self._control_dof_ids]
        control_vel = self.robot.data.joint_vel.torch[:, self._control_dof_ids]
        leg_pos = control_pos[:, [0, 1, 3, 4]] - self._default_control_pos[:, [0, 1, 3, 4]]
        if self.cfg.command_observation_mode == "tracking_error":
            heading_velocity = self._heading_forward_velocity(
                self.robot.data.root_lin_vel_w.torch
            )
            command_observation = torch.stack(
                (
                    command[:, 0] - heading_velocity,
                    command[:, 1] - linear_velocity[:, 1],
                    command[:, 2] - angular_velocity[:, 2],
                ),
                dim=-1,
            )
        elif self.cfg.command_observation_mode == "tracking_error_with_target":
            heading_velocity = self._heading_forward_velocity(
                self.robot.data.root_lin_vel_w.torch
            )
            command_observation = torch.stack(
                (
                    command[:, 0] - heading_velocity,
                    command[:, 0],
                    command[:, 2] - angular_velocity[:, 2],
                ),
                dim=-1,
            )
        elif self.cfg.command_observation_mode == "target_with_forward_velocity":
            heading_velocity = self._heading_forward_velocity(self.robot.data.root_lin_vel_w.torch)
            command_observation = torch.stack(
                (command[:, 0], heading_velocity, command[:, 2]),
                dim=-1,
            )
        elif self.cfg.command_observation_mode == "target":
            command_observation = command
        else:
            raise RuntimeError(
                f"Unsupported RMUC command observation mode: {self.cfg.command_observation_mode!r}"
            )
        yaw_observation_unclipped = command_observation[:, 2] * self._command_scale[2]
        command_observation = torch.clamp(
            command_observation * self._command_scale,
            -self.cfg.command_observation_error_clip,
            self.cfg.command_observation_error_clip,
        )
        command_observation[:, 2] = yaw_observation_unclipped.clamp(
            -self.cfg.command_yaw_observation_error_clip, self.cfg.command_yaw_observation_error_clip)
        obs = torch.cat(
            (
                angular_velocity * 0.25,
                gravity,
                command_observation,
                leg_pos,
                control_vel * 0.05,
                self._policy_actions,
            ),
            dim=-1,
        )
        obs = self._gravity_direction_dr.apply(obs, self.step_dt)
        if obs.shape[-1] != OBSERVATION_DIM:
            raise RuntimeError(f"RMUC observation dimension mismatch: {obs.shape[-1]} != {OBSERVATION_DIM}")
        self._estop |= ~torch.isfinite(obs).all(dim=-1)
        if self._sensor_model_enabled and torch.any(self._domain_randomization_strength > 0):
            c = self._sensor_cfg
            scale = self._domain_randomization_strength[:, None]
            self._sensor_gyro_bias.add_(torch.randn_like(self._sensor_gyro_bias) * c['gyro_bias_walk_rad_s_sqrt_s'] * math.sqrt(self.step_dt) * scale)
            bound = c['gyro_bias_max_rad_s'] * scale
            self._sensor_gyro_bias.copy_(torch.maximum(torch.minimum(self._sensor_gyro_bias, bound), -bound))
            obs = sensor_noise(obs, self._domain_randomization_strength, self._sensor_gyro_bias,
                               self._sensor_accel_bias, command[:,2], c,
                               yaw_error_clip=self.cfg.command_yaw_observation_error_clip)
        elif not self._sensor_model_enabled and torch.any(self._domain_observation_noise_std > 0.0):
            if (self.cfg.domain_randomization_command_consistent_noise
                    and self.cfg.command_observation_mode == "tracking_error_with_target"):
                obs = add_command_consistent_noise(
                    obs, self._domain_observation_noise_std, command[:, 2],
                    command_error_clip=self.cfg.command_yaw_observation_error_clip,
                )
            else:
                obs = obs + torch.randn_like(obs) * self._domain_observation_noise_std.unsqueeze(-1)
        if self._obs_history_depth > 0:
            invalid = ~self._obs_history_valid
            self._obs_history = torch.roll(self._obs_history, shifts=1, dims=1)
            self._obs_history[:, 0] = obs
            if torch.any(invalid):
                self._obs_history[invalid] = obs[invalid].unsqueeze(1)
            self._obs_history_valid[:] = True
            fixed_delay = torch.full_like(
                self._domain_observation_delay_steps, self._obs_delay_steps
            )
            effective_delay = torch.maximum(
                fixed_delay, self._domain_observation_delay_steps
            ).clamp_max(self._obs_history_depth)
            if self._sensor_model_enabled:
                obs = delayed_sensor_observation(self._obs_history,
                    torch.maximum(self._sensor_delay_steps, fixed_delay), obs, command[:,2],
                    yaw_error_clip=self.cfg.command_yaw_observation_error_clip)
            else:
                obs = self._obs_history[
                    torch.arange(self.num_envs, device=self.device), effective_delay]

        observation = torch.nan_to_num(obs, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
        if getattr(self.cfg, 'position_authority_enabled', False):
            self._position_authority_observation = observation.detach().clone()
        return {"policy": observation}

    def _additional_reward_before_clip(self, reward: torch.Tensor) -> torch.Tensor:
        """Subclass extension; legacy25D returns the same tensor unchanged."""
        return reward

    def _get_rewards(self) -> torch.Tensor:
        gravity, linear_velocity, angular_velocity = self._state()
        command = self._current_command()
        upright = torch.clamp(-gravity[:, 2], -1.0, 1.0)
        velocity_tolerance = (
            self.cfg.command_velocity_tolerance_mps
            if self.cfg.curriculum_stage == 1
            else self.cfg.velocity_tolerance_mps
        )
        # Use horizontal heading speed for command tracking. Body-frame x
        # velocity contains pitch/roll-induced vertical coupling in turns and
        # was inconsistent with the curriculum gate's heading-speed metric.
        command_linear_velocity = self._heading_forward_velocity(
            self.robot.data.root_lin_vel_w.torch
        )
        velocity_error = command_linear_velocity - command[:, 0]
        yaw_tolerance = (
            self.cfg.command_yaw_tolerance_rad_s
            if self.cfg.curriculum_stage == 1
            else self.cfg.yaw_tolerance_rad_s
        )
        yaw_error_signed = angular_velocity[:, 2] - command[:, 2]
        spin_mask = ((command[:, 0].abs() < 1.0e-4) & (command[:, 1].abs() < 1.0e-4)
                     & (command[:, 2].abs() > 1.0e-4)
                     & self.cfg.spin_asymmetric_reward_enabled & (self.cfg.curriculum_stage == 1))
        spin_aligned_rate = angular_velocity[:, 2] * torch.sign(command[:, 2])
        spin_deficit = (command[:, 2].abs() - spin_aligned_rate).clamp_min(0.0)
        spin_excess = (spin_aligned_rate - command[:, 2].abs()
                       - self.cfg.spin_overspeed_deadband_rad_s).clamp_min(0.0)
        yaw_error_signed = torch.where(spin_mask, spin_excess - spin_deficit, yaw_error_signed)

        if self.cfg.curriculum_stage == 1:
            velocity_reward = 1.0 / (1.0 + (velocity_error / velocity_tolerance).square())
            yaw_reward = 1.0 / (1.0 + (yaw_error_signed / yaw_tolerance).square())
        else:
            velocity_reward = torch.exp(-(velocity_error / velocity_tolerance).square())
            yaw_reward = torch.exp(-(yaw_error_signed / yaw_tolerance).square())
        stillness = torch.exp(-torch.linalg.vector_norm(linear_velocity, dim=-1) - torch.linalg.vector_norm(angular_velocity, dim=-1))
        spin_stillness = torch.exp(-torch.linalg.vector_norm(linear_velocity, dim=-1)
                                  - torch.linalg.vector_norm(angular_velocity[:, :2], dim=-1))
        stillness = torch.where(spin_mask, spin_stillness, stillness)
        tilt_cost = 1.0 - upright
        stand_metrics = self._stand_metrics(gravity)
        action_rate = torch.mean((self._policy_actions - self._previous_actions).square(), dim=-1)
        applied = self.robot.data.applied_torque.torch[:, self._control_dof_ids]
        normalized_effort = torch.clamp(applied / max(self.cfg.effort_limit_nm, 1.0e-3), -4.0, 4.0)
        energy = torch.mean(normalized_effort.square(), dim=-1)
        reward = (
            self.cfg.upright_weight * upright
            + self.cfg.velocity_weight * velocity_reward
            + self.cfg.yaw_weight * yaw_reward
            + self.cfg.stillness_weight * stillness
            - self.cfg.tilt_weight * tilt_cost
            - self.cfg.action_rate_weight * action_rate
            - self.cfg.energy_weight * energy
        )
        settled_wheel_smoothness_cost = torch.zeros_like(reward)
        if self.cfg.settled_wheel_smoothness_weight > 0:
            from .settled_wheel_smoothness import settled_wheel_cost
            settled_wheel_smoothness_cost = settled_wheel_cost(
                self._policy_actions, self._previous_actions, command,
                command_linear_velocity, angular_velocity[:, 1],
                torch.acos((-gravity[:, 2]).clamp(-1., 1.)),
                action_scale=self.cfg.settled_wheel_smoothness_scale)
            reward = reward - self.cfg.settled_wheel_smoothness_weight * settled_wheel_smoothness_cost
        base_reward = reward.clone()
        command_progress_score = torch.zeros_like(reward)
        command_reverse_score = torch.zeros_like(reward)
        command_overspeed_score = torch.zeros_like(reward)
        command_idle_micro_speed_score = torch.zeros_like(reward)
        command_dynamic_tracking_score = torch.zeros_like(reward)
        command_sequence_overshoot_score = torch.zeros_like(reward)
        command_dynamic_stop_posture_score = torch.zeros_like(reward)
        command_underspeed_score = torch.zeros_like(reward)
        command_severe_underspeed_score = torch.zeros_like(reward)
        command_speed_error_excess_score = torch.zeros_like(reward)
        command_speed_coupling_score = torch.ones_like(reward)
        command_aligned_speed = torch.zeros_like(reward)
        command_high_speed_exploration_score = torch.zeros_like(reward)
        command_speed_acquisition_score = torch.zeros_like(reward)
        command_speed_regression_score = torch.zeros_like(reward)
        command_filtered_speed_error_score = torch.zeros_like(reward)
        command_speed_oscillation_score = torch.zeros_like(reward)
        command_pitch_excess_score = torch.zeros_like(reward)
        command_pitch_rate_excess_score = torch.zeros_like(reward)
        command_pitch_target_deg = torch.zeros_like(reward)
        command_pitch_tracking_score = torch.zeros_like(reward)
        command_high_speed_pitch_rate_score = torch.zeros_like(reward)
        command_filtered_aligned_speed = torch.zeros_like(reward)
        command_straight_alignment_cost = torch.zeros_like(reward)
        command_bank_tracking_score = torch.zeros_like(reward)
        command_bank_target_deg = torch.zeros_like(reward)
        command_bank_error_penalty_score = torch.zeros_like(reward)
        command_bank_error_abs_deg = torch.zeros_like(reward)
        # These terms are enabled only in the stand stage.  Each metric is
        # explicitly bounded to [0, 1], keeping the shaping numerically tame.
        if self.cfg.curriculum_stage == 0:
            contact_loss = 1.0 - stand_metrics["contact_score"]
            reward = reward + (
                self.cfg.stand_height_weight * stand_metrics["height_score"]
                + self.cfg.stand_leg_extension_weight * stand_metrics["leg_extension_score"]
                + self.cfg.stand_symmetry_weight * stand_metrics["symmetry_score"]
                + self.cfg.stand_geometry_symmetry_weight * stand_metrics["geometry_symmetry_score"]
                + self.cfg.stand_attitude_weight * stand_metrics["attitude_score"]
                + self.cfg.stand_contact_weight * stand_metrics["contact_score"]
                + self.cfg.stand_leg_length_weight * stand_metrics["leg_length_score"]
                - self.cfg.stand_no_contact_penalty * contact_loss
            )
        elif self.cfg.curriculum_stage == 1:
            exact_low_speed = self.cfg.command_exact_low_speed_reward_enabled
            target_speed, speed_normalizer, command_direction, speed_margin = tracking_targets(
                command[:,0], self.cfg.command_speed_error_margin_mps, exact_low_speed)
            if exact_low_speed:
                # Keep temporal reward state in one physical frame through
                # reverse/stop transitions. Forward-only behavior is unchanged.
                for history in (self._previous_command_aligned_speed,
                                self._previous_filtered_command_target):
                    history.copy_(rebase_aligned_history(
                        history, self._previous_reward_direction, command_direction))
                self._previous_reward_direction.copy_(command_direction)
            consistent_reward = self.cfg.command_reward_contract_by_command
            reward_scope = (torch.ones_like(self._rehearsal_mask) if consistent_reward
                            else ~self._rehearsal_mask)

            def reward_level_eligible(start_level):
                if consistent_reward:
                    # Reward eligibility follows the commanded physical skill,
                    # not which cohort or global curriculum level sampled it.
                    start_speed = abs(self.cfg.command_stage_linear_x_mps[start_level])
                    return target_speed >= start_speed - 1.0e-5
                return torch.full_like(self._rehearsal_mask, self._command_level >= start_level)

            # Use the same horizontal heading projection as velocity tracking
            # and the curriculum gate.  Mixing this with body-frame x makes
            # turning and tilted states receive contradictory speed gradients.
            aligned_speed = command_linear_velocity * command_direction
            command_progress_score = torch.clamp(aligned_speed / speed_normalizer, 0.0, 1.0)
            if exact_low_speed:
                command_progress_score = command_progress_score * (target_speed > 1e-6)
            # Keep a useful gradient when the policy is far below target.  A
            # tanh(error / tolerance) saturates after a few tolerance widths,
            # leaving only the progress term to pull a stalled policy forward.
            # These relative, bounded scores remain comparable across 1.0-3.0
            # m/s stages while the separate absolute-error gate stays strict.
            tracking_scale = target_speed + max(velocity_tolerance, 1.0e-3)
            command_aligned_speed = aligned_speed
            command_tracking_error_score = torch.clamp(
                torch.abs(velocity_error) / tracking_scale, 0.0, 1.0
            )
            # Enforce an explicit command-tracking dead band. Normalize and
            # clip the excess so this term is strong but cannot become
            # unbounded or overwhelm all posture and safety shaping.
            command_speed_error_excess_score = torch.clamp(
                (
                    torch.abs(velocity_error)
                    - speed_margin
                )
                / self.cfg.command_speed_error_scale_mps,
                0.0,
                1.0,
            )
            if self.cfg.command_tracking_error_tail_enabled:
                speed_excess = (
                    (torch.abs(velocity_error) - speed_margin)
                    / self.cfg.command_speed_error_scale_mps
                ).clamp_min(0.0)
                command_speed_error_excess_score = (
                    speed_excess.clamp_max(1.0)
                    + self.cfg.command_tracking_error_tail_gain
                    * (1.0 - 1.0 / speed_excess.clamp_min(1.0))
                )
            # A turn is successful only when yaw and forward speed are both
            # tracked. Without this coupling the policy can stop, pivot in
            # place, and collect nearly all yaw/bank/upright reward.
            command_speed_coupling_score = 1.0 - torch.clamp(
                (
                    torch.abs(velocity_error)
                    - speed_margin
                )
                / (0.5 * speed_normalizer),
                0.0,
                1.0,
            )
            underspeed_deficit = torch.clamp(target_speed - aligned_speed, min=0.0)
            command_underspeed_score = torch.clamp(
                underspeed_deficit / speed_normalizer,
                0.0,
                1.0,
            )
            severe_underspeed_progress = torch.clamp(
                (
                    underspeed_deficit
                    - self.cfg.command_severe_underspeed_margin_mps
                )
                / max(self.cfg.command_severe_underspeed_scale_mps, 1.0e-6),
                0.0,
                1.0,
            )
            command_severe_underspeed_score = severe_underspeed_progress.square() * (
                3.0 - 2.0 * severe_underspeed_progress
            )
            command_reverse_score = torch.clamp(-aligned_speed / speed_normalizer, 0.0, 1.0)
            command_overspeed_score = torch.clamp(
                (
                    (torch.where(target_speed <= 1e-6, aligned_speed.abs(), aligned_speed)
                     if exact_low_speed else aligned_speed)
                    - target_speed
                    - self.cfg.command_overspeed_margin_mps
                )
                / max(self.cfg.command_overspeed_scale_mps, 1.0e-6),
                0.0,
                1.0,
            )
            high_speed_factor = torch.clamp((target_speed - 1.0) / 2.0, 0.0, 1.0)
            if exact_low_speed:
                moving = (target_speed > 1e-6).to(reward.dtype)
                command_underspeed_score = command_underspeed_score * moving
                command_severe_underspeed_score = command_severe_underspeed_score * moving
                command_reverse_score = command_reverse_score * moving
            safe_speed_mask = (
                (-gravity[:, 2] > 0.90)
                & (torch.abs(linear_velocity[:, 1]) < 0.45)
                & (self.robot.data.root_pos_w.torch[:, 2] >= self.cfg.command_min_operating_root_height_m)
            ).float()
            severe_underspeed_eligible = (
                reward_level_eligible(self.cfg.command_severe_underspeed_start_level)
                & reward_scope
                & (
                    self._curriculum_step_counter
                    >= self.cfg.command_severe_underspeed_grace_steps
                )
            ).float()
            command_severe_underspeed_score = (
                command_severe_underspeed_score
                * safe_speed_mask
                * severe_underspeed_eligible
            )
            command_high_speed_exploration_score = (
                # Keep an acquisition gradient while far from the narrow
                # tracking kernel. The old min(progress, velocity_reward)
                # collapsed to near zero at 1.1/1.75 m/s and therefore did not
                # function as an exploration reward.
                command_progress_score
                * high_speed_factor
                * safe_speed_mask
                * reward_scope.float()
            ).clamp(0.0, 1.0)
            filter_alpha = 1.0 - math.exp(
                -float(self.cfg.sim.dt * self.cfg.decimation)
                / max(self.cfg.command_speed_delta_filter_tau_s, 1.0e-6)
            )
            command_filtered_aligned_speed = torch.where(
                self._previous_command_aligned_speed_valid,
                self._previous_command_aligned_speed
                + filter_alpha * (aligned_speed - self._previous_command_aligned_speed),
                aligned_speed,
            )
            # Use the same low-pass state for a bounded long-horizon tracking
            # cost. This distinguishes a stable 2 m/s trajectory from one whose
            # mean is 2 m/s only because repeated slow/fast cycles cancel.
            filtered_target = target_speed
            if self.cfg.command_dynamic_matched_filter_enabled:
                matched_target = matched_filtered_target(
                    self._previous_filtered_command_target, target_speed,
                    self._previous_command_aligned_speed_valid, filter_alpha)
                self._previous_filtered_command_target.copy_(matched_target)
                filtered_target = torch.where(self._continuous_command_mask,
                                              matched_target, target_speed)
            command_filtered_speed_error_score = torch.clamp(
                (
                    torch.abs(command_filtered_aligned_speed - filtered_target)
                    - speed_margin
                )
                / speed_normalizer,
                0.0,
                1.0,
            )
            command_filtered_speed_error_score = (
                command_filtered_speed_error_score
                * reward_scope.to(reward.dtype)
            )
            # Penalize the high-pass component of speed rather than increasing
            # the already strong absolute tracking cost. This distinguishes a
            # steady 2.0 m/s turn from alternating propulsion/braking whose
            # average speed happens to match the command. The post-reset grace
            # leaves the necessary inverted-pendulum acceleration transient
            # unpenalized, and the safety mask prevents falling from being used
            # as an artificial low-pass trajectory.
            speed_oscillation_residual = torch.abs(
                aligned_speed - command_filtered_aligned_speed
            )
            speed_oscillation_eligible = (
                reward_level_eligible(self.cfg.command_speed_oscillation_start_level)
                & (
                    self._curriculum_step_counter
                    >= self.cfg.command_severe_underspeed_grace_steps
                )
            ).to(reward.dtype)
            command_speed_oscillation_score = torch.clamp(
                (
                    speed_oscillation_residual
                    - self.cfg.command_speed_oscillation_tolerance_mps
                )
                / self.cfg.command_speed_oscillation_scale_mps,
                0.0,
                1.0,
            ).square() * speed_oscillation_eligible * safe_speed_mask
            speed_delta = torch.where(
                self._previous_command_aligned_speed_valid,
                command_filtered_aligned_speed - self._previous_command_aligned_speed,
                torch.zeros_like(command_filtered_aligned_speed),
            )
            speed_delta_scale = max(self.cfg.command_speed_delta_scale_mps, 1.0e-6)
            active_high_speed = (
                (target_speed > 1.0)
                & reward_scope
            ).float()
            # Weight by underspeed so this term disappears at the target and
            # cannot reward overspeed. Positive and negative changes are kept
            # separate so a bootstrapped state that is being braked is
            # unambiguously worse than one being maintained or accelerated.
            command_speed_acquisition_score = (
                torch.clamp(speed_delta / speed_delta_scale, 0.0, 1.0)
                * command_underspeed_score
                * safe_speed_mask
                * active_high_speed
            )
            command_speed_regression_score = (
                torch.clamp(-speed_delta / speed_delta_scale, 0.0, 1.0)
                * command_underspeed_score
                * active_high_speed
            )
            turning = (torch.abs(command[:, 2]) > 1.0e-4).float()
            turn_roll_scale = (1.0 - turning) + turning * self.cfg.command_turn_roll_scale
            turn_attitude_scale = (1.0 - turning) + turning * self.cfg.command_turn_attitude_scale
            lateral_speed = torch.clamp(torch.abs(linear_velocity[:, 1]), 0.0, 2.0)
            yaw_overspeed_limit = (torch.maximum(command[:, 2].abs(), torch.ones_like(command[:, 2]))
                if self.cfg.command_yaw_overspeed_relative_to_target else 1.0)
            excessive_yaw = torch.clamp(torch.abs(angular_velocity[:, 2]) - yaw_overspeed_limit, min=0.0, max=3.0)
            excessive_yaw = torch.where(spin_mask, spin_excess.clamp_max(3.0), excessive_yaw)
            # Preserve the small pitch lean needed for acceleration, but
            # penalize excessive pitch and pitch-rate excursions that precede
            # the high-speed tail falls seen in the survival gate.
            pitch_rad = torch.atan2(gravity[:, 0], -gravity[:, 2])
            pitch_tolerance_rad = torch.deg2rad(
                torch.full_like(command[:, 0], self.cfg.command_pitch_excess_tolerance_deg)
            )
            pitch_scale_rad = torch.deg2rad(
                torch.full_like(command[:, 0], self.cfg.command_pitch_excess_scale_deg)
            ).clamp_min(1.0e-3)
            command_pitch_excess_score = torch.clamp(
                (torch.abs(pitch_rad) - pitch_tolerance_rad) / pitch_scale_rad,
                0.0,
                1.0,
            )
            command_pitch_rate_excess_score = torch.clamp(
                (
                    torch.abs(angular_velocity[:, 1])
                    - self.cfg.command_pitch_rate_tolerance_rad_s
                )
                / max(self.cfg.command_pitch_rate_scale_rad_s, 1.0e-3),
                0.0,
                1.0,
            )
            high_speed_pitch_active = (
                reward_level_eligible(self.cfg.command_pitch_tracking_start_level)
                & reward_scope
            ).to(reward.dtype)
            desired_pitch_abs_deg = torch.clamp(
                target_speed * self.cfg.command_pitch_target_deg_per_mps
                + underspeed_deficit
                * self.cfg.command_pitch_target_underspeed_gain_deg_per_mps,
                0.0,
                self.cfg.command_pitch_target_max_deg,
            )
            command_pitch_target_deg = desired_pitch_abs_deg * command_direction
            pitch_error_deg = torch.abs(
                torch.rad2deg(pitch_rad) - command_pitch_target_deg
            )
            # Command-local early recovery shaping, independent of cohort/level.
            large_turn = (target_speed >= 2.0) & (command[:, 2].abs() >= .20)
            pitch_tracking_tolerance = torch.where(large_turn,
                torch.full_like(target_speed, self.cfg.command_large_turn_pitch_tolerance_deg),
                torch.full_like(target_speed, self.cfg.command_pitch_tracking_tolerance_deg))
            pitch_tracking_scale = torch.where(large_turn,
                torch.full_like(target_speed, self.cfg.command_large_turn_pitch_scale_deg),
                torch.full_like(target_speed, self.cfg.command_pitch_tracking_scale_deg))
            command_pitch_tracking_score = torch.clamp(
                (
                    pitch_error_deg
                    - pitch_tracking_tolerance
                )
                / pitch_tracking_scale.clamp_min(1.0e-3),
                0.0,
                1.0,
            ) * high_speed_pitch_active
            command_high_speed_pitch_rate_score = torch.clamp(
                (
                    torch.abs(angular_velocity[:, 1])
                    - self.cfg.command_high_speed_pitch_rate_tolerance_rad_s
                )
                / max(self.cfg.command_high_speed_pitch_rate_scale_rad_s, 1.0e-3),
                0.0,
                1.0,
            ) * high_speed_pitch_active
            # Dense early warning for the long, low-amplitude pitch-rate
            # oscillation seen in the weak negative-yaw branch.  This is
            # deliberately active only on high-speed turns and is bounded
            # below the terminal pitch-rate barrier above.
            turn_pitch_rate_active = (
                reward_level_eligible(self.cfg.command_turn_pitch_rate_start_level)
                & (turning > 0.5)
                & reward_scope
            ).to(reward.dtype)
            command_turn_pitch_rate_score = torch.clamp(
                (
                    torch.abs(angular_velocity[:, 1])
                    - self.cfg.command_turn_pitch_rate_tolerance_rad_s
                )
                / max(self.cfg.command_turn_pitch_rate_scale_rad_s, 1.0e-3),
                0.0,
                1.0,
            ) * turn_pitch_rate_active
            straight_weight = 1.0 - turning
            if self.cfg.command_straight_alignment_weight > 0:
                from .straight_alignment import alignment_cost
                command_straight_alignment_cost, _, _ = alignment_cost(
                    self.robot.data.joint_pos.torch[:,self._control_dof_ids],
                    command[:,2], angular_velocity[:,2],
                    self.cfg.command_straight_alignment_tolerance_deg,
                    self.cfg.command_straight_alignment_scale_deg,
                    command_vx=command[:,0],
                    reverse_scale_deg=self.cfg.command_reverse_alignment_scale_deg)
                reward = reward - self.cfg.command_straight_alignment_weight * command_straight_alignment_cost
            wheel_sync = torch.clamp((self._policy_actions[:, 2] - self._policy_actions[:, 5]).square(), 0.0, 4.0)
            leg_sync = torch.clamp(
                0.5 * (
                    (self._policy_actions[:, 0] - self._policy_actions[:, 3]).square()
                    + (self._policy_actions[:, 1] + self._policy_actions[:, 4]).square()
                ),
                0.0,
                4.0,
            )
            # Canonical y gravity is the true roll component. Using x here
            # penalized the pitch lean required for acceleration and speed.
            roll_error = torch.clamp(torch.abs(gravity[:, 1]), 0.0, 1.0)
            roll_rad, desired_bank_rad = bank_reference(gravity, command)
            bank_error_rad = roll_rad - desired_bank_rad
            bank_tolerance_rad = torch.deg2rad(
                torch.full_like(command[:, 0], self.cfg.command_bank_tolerance_deg)
            ).clamp_min(1.0e-3)
            command_bank_tracking_score = (
                turning
                * torch.exp(-0.5 * (bank_error_rad / bank_tolerance_rad).square())
            ).clamp(0.0, 1.0)
            command_bank_target_deg = torch.rad2deg(desired_bank_rad)
            command_bank_error_abs_deg = torch.rad2deg(torch.abs(bank_error_rad)) * turning
            command_bank_error_penalty_score = (
                command_bank_error_abs_deg
                / max(self.cfg.command_bank_error_scale_deg, 1.0e-3)
            ).clamp_min(0.0)
            # Preserve the original near-target slope, but retain a bounded
            # improvement signal beyond one scale (6 degrees). The previous
            # hard cap erased that signal at the inherited outward posture.
            # Value and first derivative are continuous at one; range [0, 2).
            command_bank_error_penalty_score = (
                command_bank_error_penalty_score.clamp_max(1.0)
                - torch.expm1(-(command_bank_error_penalty_score - 1.0).clamp_min(0.0))
            )
            yaw_error = torch.clamp(torch.abs(yaw_error_signed), 0.0, 2.0)
            yaw_error_scale = torch.maximum(torch.abs(command[:, 2]), torch.full_like(command[:, 2], yaw_tolerance))
            command_yaw_tracking_error_score = torch.clamp(yaw_error / yaw_error_scale, 0.0, 1.0)
            if self.cfg.command_tracking_error_tail_enabled:
                yaw_error_ratio = yaw_error / yaw_error_scale
                command_yaw_tracking_error_score = (
                    yaw_error_ratio.clamp_max(1.0)
                    + self.cfg.command_tracking_error_tail_gain
                    * (1.0 - 1.0 / yaw_error_ratio.clamp_min(1.0))
                )
            # Explicit adaptation domain: keep certified <=0.45 and other
            # speeds unchanged. Relative error alone makes a large absolute
            # yaw miss cheap when the target yaw increases.
            low_speed_turn_yaw_active = (
                (command[:, 0].abs() > 0.8) & (command[:, 0].abs() < 1.2)
                & (command[:, 2].abs() > 0.4501)
                & bool(getattr(self.cfg, "command_low_speed_turn_yaw_tracking_enabled", False))
                & reward_scope
            )
            command_turn_yaw_tracking_active = (
                (reward_level_eligible(self.cfg.command_turn_yaw_tracking_start_level)
                 | low_speed_turn_yaw_active)
                & (turning > 0.5)
                & reward_scope
            ).to(reward.dtype)
            command_turn_yaw_tracking_score = (
                command_yaw_tracking_error_score * command_turn_yaw_tracking_active
            )
            low_speed_yaw_excess = (
                (yaw_error - self.cfg.command_advance_yaw_error_rad_s).clamp_min(0.0)
                / max(float(getattr(self.cfg, "command_low_speed_turn_yaw_error_scale_rad_s", 0.10)), 1e-3)
            )
            command_turn_yaw_tracking_score = command_turn_yaw_tracking_score + (
                -torch.expm1(-low_speed_yaw_excess) * low_speed_turn_yaw_active
            )
            # Straight levels must not accept a stable but continuously
            # spinning solution. Turning levels retain yaw tracking and do not
            # receive this zero-yaw stabilization penalty.
            if self.cfg.command_small_turn_dense_yaw_weight > 0:
                from .low_speed_shaping import low_speed_yaw_cost
                reward = reward - self.cfg.command_small_turn_dense_yaw_weight * low_speed_yaw_cost(
                    command, angular_velocity[:, 2],
                    self.cfg.command_small_turn_dense_yaw_max_rad_s,
                ) * reward_scope
            yaw_stabilization = (1.0 - turning) * yaw_error.square()
            if self.cfg.command_dynamic_stop_posture_weight > 0:
                command_dynamic_stop_posture_score = dynamic_stop_posture_cost(
                    command, command_linear_velocity, pitch_rad, angular_velocity[:, 1],
                    self._continuous_command_mask)
            if self.cfg.command_dynamic_tracking_weight > 0:
                command_dynamic_tracking_score = dynamic_tracking_cost(
                    command[:, 0], command_linear_velocity,
                    tracking_cohort_mask(command, self._continuous_command_mask,
                        self.cfg.command_dynamic_tracking_include_fixed_reverse),
                    self.cfg.command_dynamic_tracking_scale_mps)
            if self.cfg.command_sequence_overshoot_weight > 0:
                from .command_sequence import sequence_overshoot_cost
                command_sequence_overshoot_score = sequence_overshoot_cost(
                    self._training_sequence_table, self._acquisition_command_age(),
                    command, command_linear_velocity, self._continuous_command_mask)
            if self.cfg.command_idle_micro_speed_penalty_weight > 0.0:
                idle_scale = self.cfg.command_idle_micro_speed_scale_mps
                if not math.isfinite(idle_scale) or idle_scale <= 0.0:
                    raise ValueError("Idle micro-speed scale must be finite and positive")
                idle_mask = (command.abs().amax(dim=1) <= 1.0e-6).to(reward.dtype)
                micro_speed_squared = (command_linear_velocity / idle_scale).square()
                command_idle_micro_speed_score = (
                    micro_speed_squared / (1.0 + micro_speed_squared) * idle_mask
                )
            bank_reward_scale = torch.where(
                ~(reward_level_eligible(self.cfg.command_turn_yaw_tracking_start_level)
                  if consistent_reward else reward_scope), torch.ones_like(reward),
                torch.full_like(reward, self.cfg.command_active_bank_reward_scale),
            )
            reward = reward + (
                self.cfg.command_forward_weight * velocity_reward
                + self.cfg.command_progress_weight * command_progress_score
                + self.cfg.command_high_speed_exploration_weight * command_high_speed_exploration_score
                + self.cfg.command_speed_acquisition_weight * command_speed_acquisition_score
                + self.cfg.command_yaw_weight * yaw_reward * command_speed_coupling_score
                + self.cfg.command_require_analytic_bank * bank_reward_scale * self.cfg.command_bank_tracking_weight * command_bank_tracking_score * command_speed_coupling_score
                - self.cfg.command_bank_error_penalty_weight
                * command_bank_error_penalty_score * bank_reward_scale * self.cfg.command_require_analytic_bank
                - self.cfg.command_tracking_error_penalty_weight * command_tracking_error_score
                - self.cfg.command_speed_error_penalty_weight * command_speed_error_excess_score
                - self.cfg.command_underspeed_penalty_weight * command_underspeed_score
                - self.cfg.command_severe_underspeed_penalty_weight
                * command_severe_underspeed_score
                - self.cfg.command_speed_regression_penalty_weight * command_speed_regression_score
                - self.cfg.command_filtered_speed_error_penalty_weight
                * command_filtered_speed_error_score
                - self.cfg.command_speed_oscillation_penalty_weight
                * command_speed_oscillation_score
                - self.cfg.command_reverse_penalty_weight * command_reverse_score
                - self.cfg.command_overspeed_penalty_weight * command_overspeed_score
                - self.cfg.command_idle_micro_speed_penalty_weight * command_idle_micro_speed_score
                - self.cfg.command_dynamic_tracking_weight * command_dynamic_tracking_score
                - self.cfg.command_sequence_overshoot_weight * command_sequence_overshoot_score
                - self.cfg.command_dynamic_stop_posture_weight * command_dynamic_stop_posture_score
                - self.cfg.command_yaw_tracking_penalty_weight * command_yaw_tracking_error_score
                - self.cfg.command_turn_yaw_tracking_penalty_weight
                * command_turn_yaw_tracking_score
                - self.cfg.command_lateral_weight * lateral_speed
                - self.cfg.command_yaw_rate_penalty_weight * excessive_yaw.square()
                - self.cfg.command_pitch_excess_penalty_weight * command_pitch_excess_score
                - self.cfg.command_pitch_rate_penalty_weight * command_pitch_rate_excess_score
                - self.cfg.command_pitch_tracking_penalty_weight
                * command_pitch_tracking_score
                - self.cfg.command_high_speed_pitch_rate_penalty_weight
                * command_high_speed_pitch_rate_score
                - self.cfg.command_turn_pitch_rate_penalty_weight
                * command_turn_pitch_rate_score
                - self.cfg.command_yaw_stabilization_weight * yaw_stabilization
                - straight_weight * self.cfg.command_wheel_sync_weight * wheel_sync
                - straight_weight * self.cfg.command_leg_sync_weight * leg_sync
                - turn_roll_scale * self.cfg.command_roll_penalty_weight * roll_error.square()
            )
            root_z = self.robot.data.root_pos_w.torch[:, 2]
            height_deficit = torch.clamp(
                self.cfg.command_min_height_m - root_z, min=0.0, max=0.12
            ) / 0.12
            # Locomotion has no attraction to the standing root-height target.
            # Keep the existing leg-length contribution at its original weight;
            # minimum height/contact/attitude constraints remain independent.
            posture_score = (0.5 * self.cfg.command_leg_length_preference_scale
                             * stand_metrics["leg_length_score"])
            # Preserve the stand value landscape during the first command
            # stage.  Contact and attitude are retained, while strict
            # left/right symmetry remains disabled for locomotion.
            stand_shape = (
                self.cfg.command_stand_shape_weight * (
                    posture_score + turn_attitude_scale * stand_metrics["attitude_score"]
                )
                + self.cfg.command_contact_weight * stand_metrics["contact_score"]
            )
            reward = reward + self.cfg.command_posture_weight * stand_shape
            reward = reward + self.cfg.command_survival_bonus - self.cfg.command_height_penalty_weight * height_deficit.square()
            reward = reward - max(self.cfg.stillness_weight - self.cfg.command_stillness_weight, 0.0) * stillness
            unsafe_termination = torch.stack(
                tuple(self._last_termination_reasons.values()), dim=0
            ).any(dim=0)
            reward = reward - (
                self.cfg.command_termination_penalty_weight
                * unsafe_termination.float()
            )
            if self.cfg.command_high_straight_alive_baseline != 0.0:
                from .survival_shaping import high_straight_alive_baseline
                reward = reward + high_straight_alive_baseline(
                    command, unsafe_termination,
                    self.cfg.command_high_straight_alive_baseline)
        # Strong, bounded one-sided spin acquisition penalty; preserve all gait rewards.
        reward = reward - self.cfg.spin_underspeed_penalty_weight * (
            spin_deficit / self.cfg.spin_underspeed_scale_rad_s).clamp(0.0, 1.0) * spin_mask
        reward = self._additional_reward_before_clip(reward)
        clipped_reward = torch.clamp(
            reward,
            self.cfg.reward_clip_min,
            self.cfg.reward_clip_max,
        )
        # Keep diagnostics in the standard RSL-RL episode-extra channel. The
        # logger averages these bounded per-environment signals over completed
        # episodes, while the termination masks identify the actual failure.
        tilt_deg = torch.rad2deg(torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0)))
        locomotion = self._locomotion_diagnostics(
            gravity,
            command,
            command_linear_velocity,
            stand_metrics["leg_length_diff_m"],
        )
        active_mask = ~self._rehearsal_mask & ~self._continuous_command_mask
        active_count = active_mask.float().sum().clamp_min(1.0)
        active_target_vx = (command[:, 0] * active_mask.float()).sum() / active_count
        active_actual_vx = (command_linear_velocity * active_mask.float()).sum() / active_count
        active_speed_ratio = active_actual_vx / torch.abs(active_target_vx).clamp_min(0.10)
        unassisted_active_mask = active_mask & ~self._reset_speed_bootstrap_mask
        unassisted_active_count = unassisted_active_mask.float().sum().clamp_min(1.0)
        unassisted_active_target_vx = (
            command[:, 0] * unassisted_active_mask.float()
        ).sum() / unassisted_active_count
        unassisted_active_actual_vx = (
            command_linear_velocity * unassisted_active_mask.float()
        ).sum() / unassisted_active_count
        unassisted_active_speed_ratio = unassisted_active_actual_vx / torch.abs(
            unassisted_active_target_vx
        ).clamp_min(0.10)
        active_turn_mask = active_mask & (torch.abs(command[:, 2]) > 1.0e-4)
        active_turn_count = active_turn_mask.float().sum().clamp_min(1.0)
        active_target_yaw_abs = (
            torch.abs(command[:, 2]) * active_turn_mask.float()
        ).sum() / active_turn_count
        active_actual_yaw_aligned = (
            angular_velocity[:, 2]
            * torch.sign(command[:, 2])
            * active_turn_mask.float()
        ).sum() / active_turn_count
        unassisted_active_turn_mask = (
            active_turn_mask & ~self._reset_speed_bootstrap_mask
        )
        unassisted_active_turn_count = (
            unassisted_active_turn_mask.float().sum().clamp_min(1.0)
        )
        unassisted_active_yaw_aligned = (
            angular_velocity[:, 2]
            * torch.sign(command[:, 2])
            * unassisted_active_turn_mask.float()
        ).sum() / unassisted_active_turn_count
        active_positive_turn_mask = active_turn_mask & (command[:, 2] > 1.0e-4)
        active_negative_turn_mask = active_turn_mask & (command[:, 2] < -1.0e-4)
        termination_active_positive_turn = (
            unsafe_termination & active_positive_turn_mask
        )
        termination_active_negative_turn = (
            unsafe_termination & active_negative_turn_mask
        )
        termination_rehearsal = unsafe_termination & self._rehearsal_mask

        def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            count = mask.float().sum().clamp_min(1.0)
            return (value * mask.float()).sum() / count

        active_positive_turn_actual_vx = masked_mean(
            command_linear_velocity, active_positive_turn_mask
        )
        active_negative_turn_actual_vx = masked_mean(
            command_linear_velocity, active_negative_turn_mask
        )
        active_positive_turn_speed_error = masked_mean(
            torch.abs(velocity_error), active_positive_turn_mask
        )
        active_negative_turn_speed_error = masked_mean(
            torch.abs(velocity_error), active_negative_turn_mask
        )
        wheel_common_action = 0.5 * (
            self._policy_actions[:, 2] + self._policy_actions[:, 5]
        )
        active_positive_turn_wheel_common = masked_mean(
            wheel_common_action, active_positive_turn_mask
        )
        active_negative_turn_wheel_common = masked_mean(
            wheel_common_action, active_negative_turn_mask
        )
        turn_roll_aligned = locomotion["roll_deg"] * torch.sign(command[:, 2])
        active_positive_turn_roll_aligned = masked_mean(
            turn_roll_aligned, active_positive_turn_mask
        )
        active_negative_turn_roll_aligned = masked_mean(
            turn_roll_aligned, active_negative_turn_mask
        )
        active_positive_turn_pitch_abs = masked_mean(
            torch.abs(locomotion["pitch_deg"]), active_positive_turn_mask
        )
        active_negative_turn_pitch_abs = masked_mean(
            torch.abs(locomotion["pitch_deg"]), active_negative_turn_mask
        )
        active_positive_turn_roll_rate_abs = masked_mean(
            torch.abs(angular_velocity[:, 0]), active_positive_turn_mask
        )
        active_negative_turn_roll_rate_abs = masked_mean(
            torch.abs(angular_velocity[:, 0]), active_negative_turn_mask
        )
        active_positive_turn_pitch_rate_abs = masked_mean(
            torch.abs(angular_velocity[:, 1]), active_positive_turn_mask
        )
        active_negative_turn_pitch_rate_abs = masked_mean(
            torch.abs(angular_velocity[:, 1]), active_negative_turn_mask
        )
        active_positive_turn_leg_diff = masked_mean(
            locomotion["leg_length_diff_m"], active_positive_turn_mask
        )
        active_negative_turn_leg_diff = masked_mean(
            locomotion["leg_length_diff_m"], active_negative_turn_mask
        )
        active_speed_errors = torch.abs(velocity_error[active_mask])
        speed_focus_rehearsal = (
            self.cfg.command_rehearsal_speed_focus_start_level
            <= self._command_level
            < self.cfg.command_rehearsal_speed_focus_end_level
        )
        rehearsal_target_fraction = (
            self.cfg.command_rehearsal_speed_focus_fraction
            if speed_focus_rehearsal
            else self.cfg.command_rehearsal_fraction
        )
        bootstrap_target_fraction = (
            (1.0 - rehearsal_target_fraction)
            * self.cfg.command_reset_speed_bootstrap_fraction
            if self.cfg.command_reset_speed_bootstrap_enabled
            and self._command_level >= self.cfg.command_reset_speed_bootstrap_start_level
            else 0.0
        )
        if active_speed_errors.numel() > 0:
            active_speed_error_mean = active_speed_errors.mean()
            active_speed_error_p90 = torch.quantile(active_speed_errors, 0.90)
            active_speed_within_gate = (
                active_speed_errors <= self._command_linear_error_limit()
            ).float().mean()
        else:
            active_speed_error_mean = torch.zeros((), device=self.device)
            active_speed_error_p90 = torch.zeros((), device=self.device)
            active_speed_within_gate = torch.zeros((), device=self.device)
        # Log each curriculum blocker separately.  A single aggregate gate
        # fraction hides whether a run is failing on speed tracking, survival,
        # or simply missing enough complete episodes.
        gate_linear_limit = self._command_linear_error_limit()
        gate_mask = ~self._rehearsal_mask & ~self._reset_speed_bootstrap_mask & ~self._continuous_command_mask
        gate_sample_target = max(int(gate_mask.sum().item()), 1)
        rehearsal_gate = self._rehearsal_gate_status()
        active_turn_gate = self._active_turn_gate_status()
        current_transition_steps = self._current_command_transition_steps()
        gate_hold_reached = (
            self._global_step_counter - self._command_level_start_step
            >= current_transition_steps + int(self.cfg.command_stage_hold_steps)
        )
        gate_conditions = {
            "hold": torch.as_tensor(gate_hold_reached, device=self.device, dtype=torch.bool),
            "damping_anneal": torch.as_tensor(
                self._wheel_damping_anneal_gate_complete(),
                device=self.device,
                dtype=torch.bool,
            ),
            "posture": self._command_safe_fraction_ema
            >= self.cfg.command_min_posture_env_fraction,
            "wheel_contact": self._command_wheel_contact_fraction_ema
            >= self.cfg.command_min_wheel_contact_fraction,
            "ready": self._command_ready_fraction >= self.cfg.command_min_ready_env_fraction,
            "stable": self._command_stable_fraction_ema >= self.cfg.command_min_stable_env_fraction,
            "linear": self._command_linear_error_ema <= gate_linear_limit,
            "yaw": self._command_yaw_error_ema <= self.cfg.command_advance_yaw_error_rad_s,
            "survival_samples": torch.as_tensor(
                self._command_survival_sample_count >= gate_sample_target,
                device=self.device,
                dtype=torch.bool,
            ),
            "survival": self._command_survival_fraction_ema
            >= self.cfg.command_min_survival_env_fraction,
            "completed_samples": torch.as_tensor(
                self._command_completed_error_sample_count >= gate_sample_target,
                device=self.device,
                dtype=torch.bool,
            ),
            "completed_linear": self._command_completed_linear_error_mean <= gate_linear_limit,
            "completed_yaw": self._command_completed_yaw_error_mean
            <= self.cfg.command_advance_yaw_error_rad_s,
            "turn_ready": active_turn_gate["ready"],
            "turn_linear": active_turn_gate["linear"],
            "turn_yaw": active_turn_gate["yaw"],
            "turn_stable": active_turn_gate["stable"],
            "turn_survival": active_turn_gate["survival"],
            "turn_bank": active_turn_gate["bank"],
            "retention_ready": rehearsal_gate["ready"],
            "retention_linear": rehearsal_gate["linear"],
            "retention_yaw": rehearsal_gate["yaw"],
            "retention_stable": rehearsal_gate["stable"],
            "retention_bank": rehearsal_gate["bank"],
            "retention_survival": rehearsal_gate["survival"],
        }
        self.extras["episode"] = {
            "command_level": torch.full_like(reward, float(self._command_level)),
            "command_target_vx": command[:, 0],
            "command_active_target_vx": active_target_vx.expand_as(reward),
            "command_target_yaw": command[:, 2],
            "command_target_yaw_abs": torch.abs(command[:, 2]),
            "command_active_target_yaw_abs": active_target_yaw_abs.expand_as(reward),
            "command_active_actual_yaw_rate_aligned": active_actual_yaw_aligned.expand_as(reward),
            "command_unassisted_active_actual_yaw_rate_aligned": (
                unassisted_active_yaw_aligned.expand_as(reward)
            ),
            "command_active_positive_turn_actual_vx": active_positive_turn_actual_vx.expand_as(reward),
            "command_active_negative_turn_actual_vx": active_negative_turn_actual_vx.expand_as(reward),
            "command_active_positive_turn_speed_error": active_positive_turn_speed_error.expand_as(reward),
            "command_active_negative_turn_speed_error": active_negative_turn_speed_error.expand_as(reward),
            "command_active_positive_turn_wheel_common": active_positive_turn_wheel_common.expand_as(reward),
            "command_active_negative_turn_wheel_common": active_negative_turn_wheel_common.expand_as(reward),
            "command_active_positive_turn_roll_aligned_deg": active_positive_turn_roll_aligned.expand_as(reward),
            "command_active_negative_turn_roll_aligned_deg": active_negative_turn_roll_aligned.expand_as(reward),
            "command_active_positive_turn_pitch_abs_deg": active_positive_turn_pitch_abs.expand_as(reward),
            "command_active_negative_turn_pitch_abs_deg": active_negative_turn_pitch_abs.expand_as(reward),
            "command_active_positive_turn_roll_rate_abs_rad_s": active_positive_turn_roll_rate_abs.expand_as(reward),
            "command_active_negative_turn_roll_rate_abs_rad_s": active_negative_turn_roll_rate_abs.expand_as(reward),
            "command_active_positive_turn_pitch_rate_abs_rad_s": active_positive_turn_pitch_rate_abs.expand_as(reward),
            "command_active_negative_turn_pitch_rate_abs_rad_s": active_negative_turn_pitch_rate_abs.expand_as(reward),
            "command_active_positive_turn_leg_length_diff_m": active_positive_turn_leg_diff.expand_as(reward),
            "command_active_negative_turn_leg_length_diff_m": active_negative_turn_leg_diff.expand_as(reward),
            "termination_active_positive_turn": termination_active_positive_turn.to(reward.dtype),
            "termination_active_negative_turn": termination_active_negative_turn.to(reward.dtype),
            "termination_rehearsal": termination_rehearsal.to(reward.dtype),
            "command_stable_fraction": self._command_stable_fraction.expand_as(reward),
            "command_stable_fraction_ema": self._command_stable_fraction_ema.expand_as(reward),
            "command_safe_fraction": self._command_safe_fraction.expand_as(reward),
            "command_safe_fraction_ema": self._command_safe_fraction_ema.expand_as(reward),
            "command_wheel_contact_fraction": self._command_wheel_contact_fraction.expand_as(reward),
            "command_wheel_contact_fraction_ema": self._command_wheel_contact_fraction_ema.expand_as(reward),
            "command_linear_error_ema": self._command_linear_error_ema.expand_as(reward),
            "command_gate_linear_error_limit": torch.full_like(
                reward,
                gate_linear_limit,
            ),
            "command_yaw_error_ema": self._command_yaw_error_ema.expand_as(reward),
            "command_survival_fraction": self._command_survival_fraction.expand_as(reward),
            "command_survival_fraction_ema": self._command_survival_fraction_ema.expand_as(reward),
            "command_ready_fraction": self._command_ready_fraction.expand_as(reward),
            "command_gate_confirmation_fraction": torch.full_like(
                reward,
                min(
                    self._command_gate_confirmation_steps
                    / max(float(self.cfg.command_advance_confirmation_steps), 1.0),
                    1.0,
                ),
            ),
            "command_survival_sample_count": torch.full_like(
                reward, float(self._command_survival_sample_count)
            ),
            "command_survival_success_count": torch.full_like(
                reward, float(self._command_survival_success_count)
            ),
            "command_completed_linear_error_mean": self._command_completed_linear_error_mean.expand_as(reward),
            "command_completed_yaw_error_mean": self._command_completed_yaw_error_mean.expand_as(reward),
            "command_completed_error_sample_count": torch.full_like(
                reward, float(self._command_completed_error_sample_count)
            ),
            "command_turn_gate_ready_fraction": active_turn_gate[
                "ready_fraction"
            ].expand_as(reward),
            "command_turn_gate_max_linear_error_ema": active_turn_gate[
                "max_linear_error"
            ].expand_as(reward),
            "command_turn_gate_max_yaw_error_ema": active_turn_gate[
                "max_yaw_error"
            ].expand_as(reward),
            "command_turn_gate_min_stable_fraction_ema": active_turn_gate[
                "min_stable_fraction"
            ].expand_as(reward),
            "command_turn_gate_min_sample_count": active_turn_gate[
                "min_sample_count"
            ].expand_as(reward),
            "command_turn_gate_max_bank_error_deg_ema": active_turn_gate[
                "max_bank_error_deg"
            ].expand_as(reward),
            "command_turn_gate_min_survival_fraction_ema": active_turn_gate[
                "min_survival_fraction"
            ].expand_as(reward),
            "command_turn_gate_min_survival_episode_count": active_turn_gate[
                "min_survival_episode_count"
            ].expand_as(reward),
            "command_active_positive_turn_bank_error_deg_ema": self._command_turn_gate_bank_error_deg_ema[0].expand_as(reward),
            "command_active_negative_turn_bank_error_deg_ema": self._command_turn_gate_bank_error_deg_ema[1].expand_as(reward),
            "command_active_positive_turn_survival_fraction_ema": self._command_turn_survival_fraction_ema[0].expand_as(reward),
            "command_active_negative_turn_survival_fraction_ema": self._command_turn_survival_fraction_ema[1].expand_as(reward),
            "command_active_positive_turn_survival_episode_count": self._command_turn_survival_episode_count[0].float().expand_as(reward),
            "command_active_negative_turn_survival_episode_count": self._command_turn_survival_episode_count[1].float().expand_as(reward),
            "command_retention_ready_fraction": rehearsal_gate[
                "ready_fraction"
            ].expand_as(reward),
            "command_retention_gate_enforced": rehearsal_gate[
                "enforced"
            ].float().expand_as(reward),
            "command_retention_max_linear_error_ema": rehearsal_gate[
                "max_linear_error"
            ].expand_as(reward),
            "command_retention_max_bank_error_deg_ema": rehearsal_gate["max_bank_error_deg"].expand_as(reward),
            "command_retention_max_yaw_error_ema": rehearsal_gate[
                "max_yaw_error"
            ].expand_as(reward),
            "command_retention_min_stable_fraction_ema": rehearsal_gate[
                "min_stable_fraction"
            ].expand_as(reward),
            "command_retention_min_sample_count": rehearsal_gate[
                "min_sample_count"
            ].expand_as(reward),
            "command_retention_min_survival_fraction_ema": rehearsal_gate[
                "min_survival_fraction"
            ].expand_as(reward),
            "command_retention_min_survival_episode_count": rehearsal_gate[
                "min_survival_episode_count"
            ].expand_as(reward),
            **{
                f"command_gate_blocked_{name}": (~passed).to(reward.dtype).expand_as(reward)
                for name, passed in gate_conditions.items()
            },
            "command_rehearsal_fraction": self._rehearsal_mask.float().mean().expand_as(reward),
            "command_continuous_sampling_fraction": self._continuous_command_mask.float().mean().expand_as(reward),
            "command_rehearsal_target_fraction": torch.full_like(
                reward, rehearsal_target_fraction
            ),
            "command_rehearsal_previous_fraction": (
                self._rehearsal_mask
                & (self._rehearsal_stage == max(self._command_level - 1, 0))
            ).float().mean().expand_as(reward),
            "command_rehearsal_anchor_fraction": (
                self._rehearsal_mask
                & (self._rehearsal_stage != max(self._command_level - 1, 0))
            ).float().mean().expand_as(reward),
            "command_rehearsal_priority_fraction": (
                self._rehearsal_mask
                & torch.isin(
                    self._rehearsal_stage,
                    torch.as_tensor(
                        self.cfg.command_rehearsal_priority_stage_ids,
                        device=self.device,
                        dtype=self._rehearsal_stage.dtype,
                    ),
                )
            ).float().mean().expand_as(reward),
            "command_transition_fraction": torch.full_like(
                reward,
                min(
                    float(self._global_step_counter - self._command_level_start_step)
                    / max(float(current_transition_steps), 1.0),
                    1.0,
                ),
            ),
            "command_hold_fraction": torch.full_like(
                reward,
                min(
                    max(
                        float(self._global_step_counter - self._command_level_start_step)
                        - float(current_transition_steps),
                        0.0,
                    )
                    / max(float(self.cfg.command_stage_hold_steps), 1.0),
                    1.0,
                ),
            ),
            "actual_vx": command_linear_velocity,
            "penalty_settled_wheel_smoothness": settled_wheel_smoothness_cost,
            "penalty_straight_measured_alignment": command_straight_alignment_cost,
            "command_active_actual_vx": active_actual_vx.expand_as(reward),
            "command_active_v_actual_over_v_cmd": active_speed_ratio.expand_as(reward),
            "command_unassisted_active_actual_vx": unassisted_active_actual_vx.expand_as(reward),
            "command_unassisted_active_v_actual_over_v_cmd": unassisted_active_speed_ratio.expand_as(reward),
            "command_active_speed_error_abs_mean": active_speed_error_mean.expand_as(reward),
            "command_active_speed_error_abs_p90": active_speed_error_p90.expand_as(reward),
            "command_active_speed_within_gate_fraction": active_speed_within_gate.expand_as(reward),
            "command_reset_speed_bootstrap_fraction": self._reset_speed_bootstrap_mask.float().mean().expand_as(reward),
            "command_reset_speed_bootstrap_target_fraction": torch.full_like(
                reward, bootstrap_target_fraction
            ),
            "command_reset_exact_target_fraction": self._reset_speed_bootstrap_exact_mask.float().mean().expand_as(reward),
            "command_reset_exact_target_within_bootstrap_fraction": (
                self._reset_speed_bootstrap_exact_mask.float().sum()
                / self._reset_speed_bootstrap_mask.float().sum().clamp_min(1.0)
            ).expand_as(reward),
            "command_reset_speed_bootstrap_mean_mps": self._reset_speed_bootstrap_speed.mean().expand_as(reward),
            "command_reset_speed_bootstrap_pitch_deg": self._reset_speed_bootstrap_pitch_deg.mean().expand_as(reward),
            "v_actual_over_v_cmd": locomotion["speed_ratio"],
            "command_aligned_speed": command_aligned_speed,
            "command_filtered_aligned_speed": command_filtered_aligned_speed,
            "command_speed_error_abs": torch.abs(velocity_error),
            "actual_body_vx": linear_velocity[:, 0],
            "actual_vy": linear_velocity[:, 1],
            "actual_yaw_rate": angular_velocity[:, 2],
            "actual_yaw_rate_abs": torch.abs(angular_velocity[:, 2]),
            "actual_yaw_rate_aligned": angular_velocity[:, 2] * torch.sign(command[:, 2]),
            "yaw_tracking_error_abs": torch.abs(yaw_error_signed),
            "wheel_turn_action_aligned": (
                self._policy_actions[:, 5] - self._policy_actions[:, 2]
            ) * torch.sign(command[:, 2]),
            "root_z": self.robot.data.root_pos_w.torch[:, 2],
            "tilt_deg": tilt_deg,
            "roll_deg": locomotion["roll_deg"],
            "roll_abs_deg": locomotion["roll_abs_deg"],
            "pitch_deg": locomotion["pitch_deg"],
            "turn_fraction": locomotion["turn_fraction"],
            "turn_roll_abs_deg": locomotion["turn_roll_abs_deg"],
            "turn_roll_aligned_deg": locomotion["turn_roll_aligned_deg"],
            "leg_length_diff_m": locomotion["leg_length_diff_m"],
            "turn_leg_length_diff_m": locomotion["turn_leg_length_diff_m"],
            "wheel_torque_abs_mean_nm": locomotion["wheel_torque_abs_mean_nm"],
            "wheel_torque_ratio_mean": locomotion["wheel_torque_ratio_mean"],
            "wheel_torque_saturation_fraction": locomotion["wheel_torque_saturation_fraction"],
            "wheel_effort_requested_abs_mean_nm": self._last_requested_wheel_effort_nm.abs().mean(dim=-1),
            "wheel_effort_target_abs_mean_nm": self._last_wheel_effort_target_nm.abs().mean(dim=-1),
            "wheel_effort_target_saturation_fraction": (
                torch.abs(self._last_wheel_effort_target_nm)
                >= (0.95 * self._last_effective_wheel_effort_limit).unsqueeze(-1)
            ).float().mean(dim=-1),
            "wheel_effort_above_authored_fraction": (
                torch.abs(self._last_wheel_effort_target_nm)
                >= (0.95 * self._wheel_authored_effort_limit).unsqueeze(0)
            ).float().mean(dim=-1),
            "wheel_authored_effort_limit_nm": self._wheel_authored_effort_limit.mean().expand_as(reward),
            "wheel_effective_effort_limit_nm": self._last_effective_wheel_effort_limit,
            "wheel_effective_torque_scale_nm": self._last_effective_wheel_torque_scale,
            "domain_randomization_strength": self._domain_randomization_strength,
            "turn_fixed_sign_cohort_enabled": torch.full_like(reward, float(self._turn_fixed_sign_mask is not None)),
            "turn_positive_cohort_env_count": torch.full_like(reward, float(self._turn_fixed_sign_mask.sum()) if self._turn_fixed_sign_mask is not None else 0.0),
            "domain_base_mass_scale": self._domain_base_mass_scale,
            "domain_base_inertia_shape_scale": self._domain_base_inertia_shape_scale,
            "domain_base_inertia_scale": self._domain_base_inertia_scale,
            "domain_friction_scale": self._domain_friction_scale,
            "domain_wheel_damping_scale": self._domain_wheel_damping_scale,
            "wheel_mechanical_viscous_nm_s_rad": self._wheel_mechanical_viscous_per_env.mean(dim=-1) if self._separated_wheel_passive else torch.zeros_like(reward),
            "wheel_coulomb_friction_nm": self._wheel_coulomb_per_env.mean(dim=-1) if self._separated_wheel_passive else torch.zeros_like(reward),
            "wheel_feedback_damping_nm_s_rad": self._wheel_feedback_damping_per_env.mean(dim=-1) if self._separated_wheel_passive else torch.zeros_like(reward),
            "domain_wheel_torque_gain": self._domain_wheel_torque_gain,
            "sensor_model_enabled": torch.full_like(reward, float(self._sensor_model_enabled)),
            "sensor_observation_delay_steps": self._sensor_delay_steps.to(reward.dtype),
            "sensor_observation_delay_ms": self._sensor_delay_steps * (1000.0 * self.step_dt),
            "communication_action_delay_steps": self._communication_delay_steps.to(reward.dtype),
            "communication_action_delay_ms": self._communication_delay_steps * (1000.0 * self.step_dt),
            "imu_gyro_bias_abs_max_rad_s": self._sensor_gyro_bias.abs().amax(dim=-1),
            "imu_accel_bias_abs_max_mps2": self._sensor_accel_bias.abs().amax(dim=-1),
            "domain_observation_noise_std": self._domain_observation_noise_std,
            "domain_observation_delay_steps": self._domain_observation_delay_steps.to(
                reward.dtype
            ),
            "wheel_damping_compensation_abs_mean_nm": (
                self._last_wheel_damping_compensation_nm.abs().mean(dim=-1)
            ),
            "wheel_damping_anneal_fraction": self._last_wheel_damping_anneal_fraction,
            "wheel_damping_anneal_advancing": self._last_wheel_damping_anneal_advancing,
            "wheel_effective_damping": self._last_effective_wheel_damping,
            "wheel_damping_anneal_target": torch.full_like(
                self._last_effective_wheel_damping,
                float(self.cfg.wheel_damping_anneal_target),
            ),
            "wheel_net_effort_delivery_ratio": (
                locomotion["wheel_torque_abs_mean_nm"]
                / self._last_wheel_effort_target_nm.abs().mean(dim=-1).clamp_min(0.05)
            ).clamp(0.0, 2.0),
            "wheel_slip_speed_mps": locomotion["wheel_slip_speed_mps"],
            "wheel_slip_ratio": locomotion["wheel_slip_ratio"],
            "wheel_action_right": self._policy_actions[:, 2],
            "wheel_action_left": self._policy_actions[:, 5],
            "action_feedback_delta_abs_mean": self._last_action_feedback_delta.abs().mean(dim=-1),
            "action_feedback_delta_wheel_abs_mean": self._last_action_feedback_delta[:, [2, 5]].abs().mean(dim=-1),
            "wheel_speed_feedback_action": self._last_speed_feedback_action,
            "wheel_speed_feedback_action_abs": torch.abs(self._last_speed_feedback_action),
            "wheel_highspeed_propulsion_action": self._last_highspeed_propulsion_action,
            "wheel_speed_feedback_damping_scale": (
                1.0
                - self._last_wheel_damping_anneal_fraction
                * (1.0 - self.cfg.wheel_speed_feedback_damping_anneal_target_scale)
            ),
            "wheel_pitch_feedback_action": self._last_pitch_feedback_action,
            "wheel_pitch_feedback_action_abs": torch.abs(self._last_pitch_feedback_action),
            "wheel_pitch_emergency_feedback_action": self._last_pitch_emergency_feedback_action,
            "wheel_pitch_emergency_feedback_action_abs": torch.abs(self._last_pitch_emergency_feedback_action),
            "leg_action_abs_mean": self._policy_actions[:, [0, 1, 3, 4]].abs().mean(dim=-1),
            "reward_step": clipped_reward,
            "dynamic_reward_clip_low_fraction": (
                ((reward < self.cfg.reward_clip_min) & self._continuous_command_mask).sum()
                / self._continuous_command_mask.sum().clamp_min(1)
            ).expand_as(reward),
            "dynamic_reward_clip_high_fraction": (
                ((reward > self.cfg.reward_clip_max) & self._continuous_command_mask).sum()
                / self._continuous_command_mask.sum().clamp_min(1)
            ).expand_as(reward),
            "reward_unclipped": reward,
            "reward_clip_low_fraction": (
                reward < self.cfg.reward_clip_min
            ).to(reward.dtype),
            "reward_clip_high_fraction": (
                reward > self.cfg.reward_clip_max
            ).to(reward.dtype),
            "reward_base": base_reward,
            "reward_stage_delta": reward - base_reward,
            "reward_velocity_tracking": velocity_reward,
            "reward_forward_progress": command_progress_score,
            "reward_high_speed_exploration": command_high_speed_exploration_score,
            "reward_speed_acquisition": command_speed_acquisition_score,
            "reward_bank_tracking": command_bank_tracking_score,
            "penalty_command_bank_error": command_bank_error_penalty_score,
            "reward_speed_coupling": command_speed_coupling_score,
            "command_bank_target_deg": command_bank_target_deg,
            "command_bank_error_abs_deg": command_bank_error_abs_deg,
            "penalty_command_tracking_error": (
                command_tracking_error_score
                if self.cfg.curriculum_stage == 1 else torch.zeros_like(reward)
            ),
            "penalty_command_speed_error_excess": command_speed_error_excess_score,
            "penalty_command_underspeed": (
                command_underspeed_score
                if self.cfg.curriculum_stage == 1 else torch.zeros_like(reward)
            ),
            "penalty_command_severe_underspeed": (
                command_severe_underspeed_score
                if self.cfg.curriculum_stage == 1 else torch.zeros_like(reward)
            ),
            "penalty_command_speed_regression": command_speed_regression_score,
            "penalty_command_filtered_speed_error": command_filtered_speed_error_score,
            "penalty_command_speed_oscillation": command_speed_oscillation_score,
            "penalty_command_pitch_excess": command_pitch_excess_score,
            "penalty_command_pitch_rate_excess": command_pitch_rate_excess_score,
            "command_pitch_target_deg": command_pitch_target_deg,
            "penalty_command_pitch_tracking": command_pitch_tracking_score,
            "penalty_command_high_speed_pitch_rate": command_high_speed_pitch_rate_score,
            "penalty_command_turn_pitch_rate": command_turn_pitch_rate_score,
            "penalty_command_turn_yaw_tracking": command_turn_yaw_tracking_score,
            "penalty_yaw_tracking_error": (
                command_yaw_tracking_error_score
                if self.cfg.curriculum_stage == 1 else torch.zeros_like(reward)
            ),
            "penalty_reverse_motion": command_reverse_score,
            "penalty_forward_overspeed": command_overspeed_score,
            "penalty_idle_micro_speed": command_idle_micro_speed_score,
            "penalty_dynamic_tracking": command_dynamic_tracking_score,
            "penalty_sequence_overshoot": command_sequence_overshoot_score,
            "penalty_dynamic_stop_posture": command_dynamic_stop_posture_score,
            "penalty_yaw_stabilization": yaw_stabilization if self.cfg.curriculum_stage == 1 else torch.zeros_like(reward),
            "penalty_wheel_sync": wheel_sync if self.cfg.curriculum_stage == 1 else torch.zeros_like(reward),
            "penalty_unsafe_termination": (
                unsafe_termination.float()
                if self.cfg.curriculum_stage == 1 else torch.zeros_like(reward)
            ),
        }
        for gate_index, stage_id in enumerate(self.cfg.command_rehearsal_gate_stage_ids):
            for direction_index, direction_name in enumerate(("positive", "negative")):
                prefix = f"command_retention_level{stage_id}_{direction_name}"
                self.extras["episode"][f"{prefix}_sample_count"] = self._rehearsal_gate_sample_count[
                    gate_index, direction_index
                ].float().expand_as(reward)
                self.extras["episode"][f"{prefix}_linear_error_ema"] = self._rehearsal_gate_linear_error_ema[
                    gate_index, direction_index
                ].expand_as(reward)
                self.extras["episode"][f"{prefix}_yaw_error_ema"] = self._rehearsal_gate_yaw_error_ema[
                    gate_index, direction_index
                ].expand_as(reward)
                self.extras["episode"][f"{prefix}_bank_error_deg_ema"] = self._rehearsal_gate_bank_error_deg_ema[gate_index, direction_index].expand_as(reward)
                self.extras["episode"][f"{prefix}_stable_fraction_ema"] = self._rehearsal_gate_stable_fraction_ema[
                    gate_index, direction_index
                ].expand_as(reward)
                self.extras["episode"][f"{prefix}_survival_fraction_ema"] = self._rehearsal_gate_survival_fraction_ema[
                    gate_index, direction_index
                ].expand_as(reward)
                self.extras["episode"][f"{prefix}_survival_episode_count"] = self._rehearsal_gate_survival_episode_count[
                    gate_index, direction_index
                ].float().expand_as(reward)
        for direction_index, direction_name in enumerate(("positive", "negative")):
            prefix = f"command_active_turn_{direction_name}"
            self.extras["episode"][f"{prefix}_sample_count"] = self._command_turn_gate_sample_count[
                direction_index
            ].float().expand_as(reward)
            self.extras["episode"][f"{prefix}_linear_error_ema"] = self._command_turn_gate_linear_error_ema[
                direction_index
            ].expand_as(reward)
            self.extras["episode"][f"{prefix}_yaw_error_ema"] = self._command_turn_gate_yaw_error_ema[
                direction_index
            ].expand_as(reward)
            self.extras["episode"][f"{prefix}_stable_fraction_ema"] = self._command_turn_gate_stable_fraction_ema[
                direction_index
            ].expand_as(reward)
        for reason, mask in getattr(self, "_last_termination_reasons", {}).items():
            self.extras["episode"][f"termination_{reason}"] = mask.float()
        denominator = self._level23_rehearsal_episode_count.clamp_min(1).float()
        diagnostics = {
            "episode_count": self._level23_rehearsal_episode_count.float(),
            "episode_length_mean": self._level23_rehearsal_episode_length_sum / denominator,
            "window_ready_fraction": self._level23_rehearsal_window_ready_count.float() / denominator,
            "steady_ready_fraction": self._level23_rehearsal_steady_ready_count.float() / denominator,
        }
        diagnostics.update({
            f"termination_{name}_count": count.float()
            for name, count in self._level23_rehearsal_termination_counts.items()
        })
        for name, value in diagnostics.items():
            self.extras["episode"][f"level23_rehearsal_{name}"] = value.expand_as(reward)
        self._previous_actions.copy_(self._policy_actions)
        self._previous_command_aligned_speed.copy_(command_filtered_aligned_speed)
        self._previous_command_aligned_speed_valid.fill_(self.cfg.curriculum_stage == 1)
        self._estop |= ~torch.isfinite(reward)
        return torch.nan_to_num(
            clipped_reward,
            nan=self.cfg.reward_clip_min,
            posinf=self.cfg.reward_clip_max,
            neginf=self.cfg.reward_clip_min,
        )

    def _current_command(self) -> torch.Tensor:
        """Return the current gated command with a smooth level transition."""
        if getattr(self.cfg, "play", False) and self._manual_command is not None:
            return self._manual_command
        if self.cfg.curriculum_stage != 1:
            return self._command
        command = self._command.clone()
        vx, yaw, _ = self._current_command_stage_values_tensor()
        command[:, 0] = vx
        command[:, 1] = 0.0
        command[:, 2] = yaw * self._command_turn_sign
        command[self._continuous_command_mask] = self._continuous_command_values()[self._continuous_command_mask]
        hold = getattr(self.cfg, 'command_turn_entry_hold_s', 0.)
        ramp = getattr(self.cfg, 'command_turn_entry_ramp_s', 0.)
        if hold + ramp > 0:
            mask = ~self._rehearsal_mask & ~self._continuous_command_mask
            abrupt_fraction = getattr(self.cfg, 'command_turn_entry_abrupt_fraction', 0.)
            # Fixed cohorts keep short failed episodes from stealing assisted slots.
            mask &= torch.arange(command.shape[0], device=command.device) >= round(command.shape[0] * abrupt_fraction)
            elapsed = self._acquisition_command_age().to(command.dtype) * self.step_dt - hold
            fraction = (elapsed / ramp).clamp(0, 1) if ramp > 0 else (elapsed >= 0).to(command.dtype)
            fraction = fraction.square() * (3. - 2. * fraction)
            command[:, 2] = torch.where(mask, command[:, 2] * fraction, command[:, 2])
        if self.cfg.command_reset_acquisition_ramp_s > 0:
            mask = self._acquisition_ramp_mask()
            fraction = (self._acquisition_command_age().to(command.dtype) * self.step_dt
                        / self.cfg.command_reset_acquisition_ramp_s).clamp(0, 1)
            command[:, 0] = torch.where(mask, 1.0 + fraction * (command[:, 0] - 1.0), command[:, 0])
        return command

    def _acquisition_ramp_mask(self) -> torch.Tensor:
        enabled = (self.cfg.command_reset_acquisition_ramp_s > 0
                   and self.cfg.curriculum_stage == 1 and not getattr(self.cfg, 'play', False)
                   and self.cfg.command_stage_linear_x_mps[self._command_level] >= 2.9
                   and abs(self.cfg.command_stage_yaw_rate_rad_s[self._command_level]) < 1e-4)
        return (~self._rehearsal_mask & ~self._continuous_command_mask) & enabled

    def _acquisition_command_age(self) -> torch.Tensor:
        # pre_physics increments the counter before consuming the prior observation.
        # Keep that observation's command through control, reward and termination.
        offset = int(getattr(self, '_command_in_physics_step', False))
        return (self._curriculum_step_counter - offset).clamp_min(0)

    def _acquisition_ramp_tracking_ready(self) -> torch.Tensor:
        end = math.ceil(self.cfg.command_reset_acquisition_ramp_s / self.step_dt) + self.cfg.command_gate_steps
        return ~self._acquisition_ramp_mask() | (self._acquisition_command_age() >= end)

    def _continuous_command_values(self) -> torch.Tensor:
        if getattr(self, '_random_speed_sequence', None) is not None:
            return self._random_speed_sequence.commands(self._acquisition_command_age())
        if self._training_sequence_table is None:
            return self._continuous_commands
        # One age convention for observation, control, scale, reward and done.
        return sequence_commands(self._training_sequence_table, self._acquisition_command_age())

    def _refresh_continuous_commands(self, env_ids: torch.Tensor) -> None:
        self._continuous_command_mask[env_ids] = False
        fraction = self.cfg.command_continuous_sampling_fraction
        if fraction <= 0 or self.cfg.curriculum_stage != 1 or getattr(self.cfg, 'play', False):
            return
        if not self.cfg.feedback_contract_by_command or not self.cfg.command_reward_contract_by_command:
            raise ValueError('Continuous commands require command-consistent feedback and rewards')
        if fraction + max(self.cfg.command_rehearsal_fraction, self.cfg.command_rehearsal_speed_focus_fraction) >= .9:
            raise ValueError('Continuous/rehearsal cohorts must leave at least10% fixed active environments')
        available = env_ids[~self._rehearsal_mask[env_ids]]
        missing = max(round(self.num_envs * fraction) - int(self._continuous_command_mask.sum()), 0)
        selected = available[torch.randperm(len(available), device=self.device)[:missing]]
        straight_range=getattr(self.cfg, 'command_continuous_straight_speed_range', ())
        if straight_range and getattr(self.cfg, 'command_speed_extension_sampling_box', ()):
            raise ValueError('Straight and turn-extension continuous samplers are mutually exclusive')
        signed_schedule = (
            (self._random_speed_sequence is not None and self._random_speed_sequence.lo < 0)
            or (self._training_sequence_table is not None
                and any(row[1] < 0 for row in self._training_sequence_table)))
        if signed_schedule or (
            (self._random_speed_sequence is not None or self._training_sequence_table is not None)
            and any(v < 0 for v in self.cfg.command_stage_linear_x_mps[:self._command_level+1])
        ):
            # Mixed-history courses may return to forward-only schedules.
            # The explicit schedule owns delivered commands. Do not ask the
            # forward-only envelope to sample an unused placeholder at negative vx.
            values = torch.zeros(len(selected), 3, device=self.device)
        else:
            # Preserve historical forward-course sampling/RNG exactly.
            values = sample_opened_commands(len(selected), self.cfg.command_stage_linear_x_mps,
                self.cfg.command_stage_yaw_rate_rad_s, self._command_level, device=self.device,
                final_turn_envelope=getattr(self.cfg, 'command_final_turn_envelope', None),
                extended_start_level=getattr(self.cfg, 'command_extended_sampling_start_level', None),
                straight_speed_range=straight_range)
        box=getattr(self.cfg, 'command_speed_extension_sampling_box', ())
        if box:
            from .command_contract import sample_speed_extension_commands
            values=sample_speed_extension_commands(len(selected), box, device=self.device,
                anchors=getattr(self.cfg, 'command_speed_extension_anchors', ()),
                anchor_fraction=getattr(self.cfg, 'command_speed_extension_anchor_fraction', 0.0))
        self._continuous_command_mask[selected] = True
        self._continuous_commands[selected] = values
        if self._random_speed_sequence is not None:
            self._random_speed_sequence.reset(selected)

    def set_manual_command(self, command: torch.Tensor) -> None:
        """Set a play-only command override for an external Isaac controller."""

        if command.ndim == 1:
            command = command.unsqueeze(0)
        if command.shape[-1] != 3:
            raise ValueError(f"RMUC manual command must have 3 values, got {tuple(command.shape)}")
        value = command.to(device=self.device, dtype=torch.float32)
        if value.shape[0] == 1 and self.num_envs > 1:
            value = value.expand(self.num_envs, -1)
        if value.shape[0] != self.num_envs:
            raise ValueError(
                f"RMUC manual command batch must be 1 or {self.num_envs}, got {value.shape[0]}"
            )
        self._manual_command = value.clone()

    def _wheel_damping_anneal_progress(self) -> float:
        """Return the cosine bridge fraction for the current command level."""

        if (
            self.cfg.wheel_damping_anneal_target < 0.0
            or self.cfg.curriculum_stage != 1
            or self._command_level < self.cfg.wheel_damping_anneal_start_level
        ):
            return 0.0
        elapsed_steps = (
            int(self._wheel_damping_anneal_transition_count)
            if self.cfg.wheel_damping_anneal_adaptive
            else max(
                int(self._global_step_counter - self._wheel_damping_anneal_start_step),
                0,
            )
        )
        progress = min(
            elapsed_steps / max(int(self.cfg.wheel_damping_anneal_steps), 1),
            1.0,
        )
        return 0.5 * (1.0 - math.cos(math.pi * progress))

    def _update_wheel_damping_anneal_progress(self) -> None:
        """Advance the actuator bridge only while the actor tracks safely."""

        self._last_wheel_damping_anneal_advancing.zero_()
        if (
            not self.cfg.wheel_damping_anneal_adaptive
            or self.cfg.wheel_damping_anneal_target < 0.0
            or self.cfg.curriculum_stage != 1
            or self._command_level < self.cfg.wheel_damping_anneal_start_level
            or self._global_step_counter
            < int(self.cfg.wheel_damping_anneal_warmup_steps)
            or self._wheel_damping_anneal_transition_count
            >= int(self.cfg.wheel_damping_anneal_steps)
        ):
            return
        metrics_ready = (
            self._command_stable_sample_count > 0
            and self._command_survival_sample_count > 0
        )
        # Do not reuse the final command-stable fraction here. That metric
        # already contains the strict per-stage speed gate, which creates a
        # circular dependency: the bridge cannot reduce damping until speed
        # passes, while the current damping is the reason speed cannot pass.
        # Advance on physical safety instead; the normal curriculum gate still
        # certifies strict speed/yaw tracking after the bridge is complete.
        tracking_ready = (
            metrics_ready
            and float(self._command_linear_error_ema)
            <= self.cfg.wheel_damping_anneal_max_linear_error_mps
            and float(self._command_safe_fraction_ema)
            >= max(
                self.cfg.wheel_damping_anneal_min_stable_fraction,
                self.cfg.command_min_posture_env_fraction,
            )
            and float(self._command_wheel_contact_fraction_ema)
            >= self.cfg.command_min_wheel_contact_fraction
            and float(self._command_survival_fraction_ema)
            >= self.cfg.wheel_damping_anneal_min_survival_fraction
        )
        if (
            tracking_ready
            and abs(self.cfg.command_stage_yaw_rate_rad_s[self._command_level])
            > 1.0e-4
        ):
            # Aggregate survival hid the weaker yaw sign and let the actuator
            # contract move faster than that branch could adapt. Require a
            # small but meaningful complete-episode sample for both signs and
            # pause the bridge as soon as either direction regresses.
            directional_ready = torch.all(
                self._command_turn_survival_episode_count >= 8
            )
            directional_safe = torch.all(
                self._command_turn_survival_fraction_ema
                >= self.cfg.wheel_damping_anneal_min_survival_fraction
            )
            tracking_ready = bool(directional_ready and directional_safe)
        if not tracking_ready:
            return
        # Changing actuator dynamics must not outrun retained skills. Active
        # survival alone cannot certify the frozen experts' command support.
        retention = self._rehearsal_gate_status()
        if not all(bool(retention[key]) for key in ("ready", "linear", "yaw", "stable", "survival", "bank")):
            return
        self._wheel_damping_anneal_transition_count = min(
            self._wheel_damping_anneal_transition_count + self.num_envs,
            int(self.cfg.wheel_damping_anneal_steps),
        )
        self._last_wheel_damping_anneal_advancing.fill_(1.0)

    def _wheel_damping_anneal_gate_complete(self) -> bool:
        """Require a requested actuator bridge to finish before advancement."""

        return (
            self.cfg.wheel_damping_anneal_target < 0.0
            or self._command_level < self.cfg.wheel_damping_anneal_start_level
            or self._wheel_damping_anneal_progress() >= 1.0 - 1.0e-6
        )

    def get_wheel_exploration_hold_mask(self) -> torch.Tensor:
        """Return envs that still need sustained wheel exploration."""

        if self.cfg.curriculum_stage != 1:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        command = self._current_command()
        heading_velocity = self._heading_forward_velocity(
            self.robot.data.root_lin_vel_w.torch
        )
        aligned_speed = heading_velocity * torch.sign(command[:, 0])
        target_speed = torch.abs(command[:, 0])
        return (
            (aligned_speed < target_speed - self.cfg.command_overspeed_margin_mps)
            & ~self._rehearsal_mask
        )

    def _current_command_stage_values(self) -> tuple[float, float, float]:
        """Interpolate vx, yaw magnitude and leg scale at a level boundary."""

        level = min(self._command_level, len(self.cfg.command_stage_names) - 1)
        transition_steps = self._current_command_transition_steps()
        alpha = min(
            max(
                float(self._global_step_counter - self._command_level_start_step)
                / max(float(transition_steps), 1.0),
                0.0,
            ),
            1.0,
        )
        if level == 0:
            previous_values = (
                self.cfg.command_bootstrap_linear_x_mps,
                self.cfg.command_bootstrap_yaw_rate_rad_s,
                self.cfg.command_bootstrap_leg_scale_rad,
            )
        else:
            previous_values = (
                self.cfg.command_stage_linear_x_mps[level - 1],
                self.cfg.command_stage_yaw_rate_rad_s[level - 1],
                self.cfg.command_stage_leg_scale_rad[level - 1],
            )
        current_values = (
            self.cfg.command_stage_linear_x_mps[level],
            self.cfg.command_stage_yaw_rate_rad_s[level],
            self.cfg.command_stage_leg_scale_rad[level],
        )
        return (
            (1.0 - alpha) * previous_values[0] + alpha * current_values[0],
            (1.0 - alpha) * previous_values[1] + alpha * current_values[1],
            (1.0 - alpha) * previous_values[2] + alpha * current_values[2],
        )

    def _current_command_transition_steps(self) -> int:
        level = min(self._command_level, len(self.cfg.command_stage_transition_steps) - 1)
        return int(self.cfg.command_stage_transition_steps[level])

    def _current_command_survival_gate_steps(self) -> int:
        level = min(
            self._command_level,
            len(self.cfg.command_stage_survival_gate_steps) - 1,
        )
        return int(self.cfg.command_stage_survival_gate_steps[level])

    def _current_command_stage_values_tensor(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return interpolated current values plus fixed previous-stage rehearsal values."""

        if getattr(self.cfg, "play", False) and self._manual_command is not None:
            command = self._manual_command
            leg_scale = interpolate_leg_scale(command[:, 0], self._manual_leg_speed_knots,
                                              self._manual_leg_scale_knots)
            return command[:, 0], command[:, 2].abs(), leg_scale
        vx, yaw, leg_scale = self._current_command_stage_values()
        vx_t = torch.full((self.num_envs,), vx, device=self.device)
        yaw_t = torch.full((self.num_envs,), yaw, device=self.device)
        leg_t = torch.full((self.num_envs,), leg_scale, device=self.device)
        if torch.any(self._rehearsal_mask):
            replay_ids = self._rehearsal_mask
            replay_stage = self._rehearsal_stage[replay_ids]
            stage_vx = torch.as_tensor(self.cfg.command_stage_linear_x_mps, device=self.device)[replay_stage]
            stage_yaw = torch.as_tensor(self.cfg.command_stage_yaw_rate_rad_s, device=self.device)[replay_stage]
            stage_leg = torch.as_tensor(self.cfg.command_stage_leg_scale_rad, device=self.device)[replay_stage]
            vx_t[replay_ids] = stage_vx
            yaw_t[replay_ids] = stage_yaw
            leg_t[replay_ids] = stage_leg
        random_ids = self._continuous_command_mask
        if torch.any(random_ids):
            sampled = self._continuous_command_values()[random_ids]
            vx_t[random_ids] = sampled[:, 0]
            yaw_t[random_ids] = sampled[:, 2].abs()
            leg_t[random_ids] = interpolate_leg_scale(sampled[:, 0], self._manual_leg_speed_knots,
                                                       self._manual_leg_scale_knots)
        if getattr(self.cfg, 'command_leg_scale_profile', None) is not None:
            leg_t = interpolate_leg_scale(vx_t, self._manual_leg_speed_knots,
                                           self._manual_leg_scale_knots)
        return vx_t, yaw_t, leg_t

    def _stand_metrics(self, gravity: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Bounded stand-shaping metrics shared by reward and curriculum gate."""
        if gravity is None:
            gravity, _, _ = self._state()
        root_z = self.robot.data.root_pos_w.torch[:, 2]
        height_error = torch.abs(root_z - self.cfg.stand_root_height_target_m)
        height_scale = max(self.cfg.stand_root_height_tolerance_m, 1.0e-3)
        height_score = torch.exp(-0.5 * (height_error / height_scale).square()).clamp(0.0, 1.0)
        leg_pos = self.robot.data.joint_pos.torch[:, self._dof_ids][:, self._leg_dof_ids]
        leg_target = self._standing_leg_pos
        leg_error = torch.abs(leg_pos - leg_target)
        leg_scale = max(self.cfg.stand_leg_pose_tolerance_rad, 1.0e-3)
        leg_extension_score = torch.exp(
            -0.5 * (torch.mean(leg_error, dim=-1) / leg_scale).square()
        ).clamp(0.0, 1.0)
        # Active-link signs are equal; hip signs are mirrored in the authored
        # standing profile, so compare them after reflecting the right hip.
        active_diff = torch.abs(
            leg_pos[:, self._leg_local_index["left_active_link_pitch"]]
            - leg_pos[:, self._leg_local_index["right_active_link_pitch"]]
        )
        hip_diff = torch.abs(
            leg_pos[:, self._leg_local_index["left_hip_pitch"]]
            + leg_pos[:, self._leg_local_index["right_hip_pitch"]]
        )
        symmetry_error = 0.5 * (active_diff + hip_diff)
        symmetry_scale = max(self.cfg.stand_symmetry_tolerance_rad, 1.0e-3)
        symmetry_score = torch.exp(-0.5 * (symmetry_error / symmetry_scale).square()).clamp(0.0, 1.0)
        attitude_error = torch.linalg.vector_norm(gravity[:, :2], dim=-1)
        attitude_score = torch.exp(-0.5 * (attitude_error / 0.35).square()).clamp(0.0, 1.0)
        contact_forces = self.contact_sensor.data.net_forces_w[:, self._wheel_contact_sensor_ids]
        wheel_force = torch.linalg.vector_norm(contact_forces, dim=-1)
        wheel_contact = wheel_force > float(self.cfg.contact_force_threshold_N)
        force_sum = wheel_force.sum(dim=-1).clamp_min(1.0e-6)
        force_imbalance = torch.abs(wheel_force[:, 0] - wheel_force[:, 1]) / force_sum
        contact_balance_score = torch.exp(-0.5 * (force_imbalance / max(self.cfg.stand_contact_balance_tolerance, 1.0e-3)).square()).clamp(0.0, 1.0)
        contact_score = wheel_contact.all(dim=-1).float() * contact_balance_score
        body_names = {name: index for index, name in enumerate(self.robot.body_names)}
        body_pos = self.robot.data.body_pos_w.torch
        body_quat = self.robot.data.body_quat_w.torch
        root_pos = body_pos[:, body_names["robot"]]
        wheel_pos = body_pos[:, [body_names["left_wheel"], body_names["right_wheel"]]]
        wheel_delta_w = wheel_pos - root_pos.unsqueeze(1)
        root_quat = body_quat[:, body_names["robot"]].unsqueeze(1).expand(-1, 2, -1)
        wheel_delta_source = quat_apply_inverse(root_quat, wheel_delta_w)
        # Map the authored +90-deg-X body frame into the canonical frame, then
        # exclude lateral half-track from the geometric leg extension.
        wheel_delta_forward = -wheel_delta_source[..., 2]
        wheel_delta_up = wheel_delta_source[..., 1]
        axle_distance = torch.sqrt(wheel_delta_forward.square() + wheel_delta_up.square())
        axle_error = torch.abs(axle_distance - self.cfg.stand_leg_length_target_m).mean(dim=-1)
        axle_scale = max(self.cfg.stand_leg_length_tolerance_m, 1.0e-3)
        leg_length_score = torch.exp(-0.5 * (axle_error / axle_scale).square()).clamp(0.0, 1.0)
        wheel_height_error = torch.abs(wheel_pos[:, 0, 2] - wheel_pos[:, 1, 2])
        wheel_height_score = torch.exp(
            -0.5 * (wheel_height_error / max(self.cfg.stand_wheel_height_tolerance_m, 1.0e-3)).square()
        ).clamp(0.0, 1.0)
        roll_error = torch.abs(gravity[:, 1])
        roll_score = torch.exp(-0.5 * (roll_error / max(self.cfg.stand_roll_tolerance, 1.0e-3)).square()).clamp(0.0, 1.0)
        geometry_symmetry_score = (wheel_height_score * roll_score * contact_balance_score).clamp(0.0, 1.0)
        symmetry_score = (symmetry_score * geometry_symmetry_score).sqrt().clamp(0.0, 1.0)
        height_ok = height_error <= self.cfg.stand_height_gate_tolerance_m
        leg_ok = leg_extension_score >= 0.70
        symmetry_ok = symmetry_error <= self.cfg.stand_symmetry_tolerance_rad
        attitude_ok = attitude_score >= 0.95
        geometry_ok = (leg_length_score >= 0.70) & (wheel_height_score >= 0.70) & (roll_score >= 0.70)
        standing_fraction = (height_ok & leg_ok & symmetry_ok & attitude_ok & geometry_ok & wheel_contact.all(dim=-1)).float()
        return {
            "height_error_m": height_error,
            "height_score": height_score,
            "leg_extension_score": leg_extension_score,
            "symmetry_error_rad": symmetry_error,
            "symmetry_score": symmetry_score,
            "attitude_score": attitude_score,
            "wheel_force_imbalance": force_imbalance,
            "contact_score": contact_score,
            "leg_length_m": axle_distance.mean(dim=-1),
            "leg_length_diff_m": torch.abs(axle_distance[:, 0] - axle_distance[:, 1]),
            "leg_length_score": leg_length_score,
            "wheel_height_error_m": wheel_height_error,
            "wheel_height_score": wheel_height_score,
            "roll_error": roll_error,
            "roll_score": roll_score,
            "geometry_symmetry_score": geometry_symmetry_score,
            "standing_fraction": standing_fraction,
        }

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        gravity, linear_velocity, angular_velocity = self._state()
        tilt = torch.rad2deg(torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0)))
        joint_pos = self.robot.data.joint_pos.torch[:, self._dof_ids]
        limit_violation = torch.any(
            (joint_pos < self._joint_pos_limits[..., 0]) | (joint_pos > self._joint_pos_limits[..., 1]), dim=-1
        )
        tilt_limit = self.cfg.estop_tilt_deg_real if self.cfg.run_mode == "real_hardware" else self.cfg.estop_tilt_deg_sim
        root_pos = self.robot.data.root_pos_w.torch
        reasons = {
            "nonfinite": (
                ~torch.isfinite(root_pos).all(dim=-1)
                | ~torch.isfinite(joint_pos).all(dim=-1)
                | ~torch.isfinite(linear_velocity).all(dim=-1)
                | ~torch.isfinite(angular_velocity).all(dim=-1)
            ),
            "joint_limit": limit_violation,
            "tilt": tilt > tilt_limit,
            "root_high": root_pos[:, 2] > self.cfg.max_height_m,
            "root_low": root_pos[:, 2] < self.cfg.min_root_height_m,
            "xy_distance": (
                torch.linalg.norm(root_pos[:, :2] - self.scene.env_origins[:, :2], dim=-1)
                > self.cfg.max_xy_distance_m
            ),
            "linear_speed": torch.linalg.norm(linear_velocity, dim=-1) > self.cfg.max_velocity_mps,
        }
        self._last_termination_reasons = reasons
        unsafe = torch.stack(tuple(reasons.values()), dim=0).any(dim=0)
        if bool(unsafe.any()) and not self._termination_debug_printed:
            failed = unsafe.nonzero(as_tuple=False).flatten()
            counts = {key: int(mask.sum().item()) for key, mask in reasons.items()}
            failed_command = self._current_command()[failed]
            failed_positive_yaw = int((failed_command[:, 2] > 1.0e-4).sum().item())
            failed_negative_yaw = int((failed_command[:, 2] < -1.0e-4).sum().item())
            print(
                "[RMUC_TERMINATION] "
                f"episode_step_range=({int(self.episode_length_buf[failed].min())}, "
                f"{int(self.episode_length_buf[failed].max())}) reasons={counts} "
                f"root_z_range=({float(root_pos[failed, 2].min()):.4f}, "
                f"{float(root_pos[failed, 2].max()):.4f}) "
                f"tilt_max_deg={float(tilt[failed].max()):.2f} "
                f"linear_speed_max={float(torch.linalg.norm(linear_velocity[failed], dim=-1).max()):.3f} "
                f"command_mean={failed_command.mean(dim=0).detach().cpu().tolist()} "
                f"yaw_sign_counts=(positive={failed_positive_yaw},negative={failed_negative_yaw}) "
                f"action_abs_max={float(self._policy_actions[failed].abs().max()):.3f} "
                f"action_mean={self._policy_actions[failed].mean(dim=0).detach().cpu().tolist()} "
                f"raw_action_mean={self._last_raw_policy_actions[failed].mean(dim=0).detach().cpu().tolist()} "
                f"speed_feedback_mean={float(self._last_speed_feedback_action[failed].mean()):.4f} "
                f"propulsion_mean={float(self._last_highspeed_propulsion_action[failed].mean()):.4f} "
                f"pitch_feedback_mean={float(self._last_pitch_feedback_action[failed].mean()):.4f} "
                f"pitch_emergency_feedback_mean={float(self._last_pitch_emergency_feedback_action[failed].mean()):.4f} "
                f"yaw_feedback_mean={float(self._last_yaw_feedback_action[failed].mean()):.4f} "
                f"angular_velocity_mean={angular_velocity[failed].mean(dim=0).detach().cpu().tolist()}",
                flush=True,
            )
            self._termination_debug_printed = True
        self._estop |= unsafe
        return self._estop, self.episode_length_buf >= self.max_episode_length - 1

    def get_termination_reasons(self) -> dict[str, torch.Tensor]:
        """Return the most recent safety termination masks for diagnostics."""

        return getattr(self, "_last_termination_reasons", {})

    def get_curriculum_metrics(self) -> dict[str, torch.Tensor]:
        """Return GPU-resident metrics used to gate staged training."""
        gravity, linear_velocity, angular_velocity = self._state()
        command = self._current_command()
        tilt_deg = torch.rad2deg(torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0)))
        contact_forces = self.contact_sensor.data.net_forces_w
        wheel_forces = contact_forces[:, self._wheel_contact_sensor_ids]
        wheel_contact = torch.linalg.vector_norm(wheel_forces, dim=-1) > float(self.cfg.contact_force_threshold_N)
        body_names = {name: index for index, name in enumerate(self.robot.body_names)}
        body_pos = self.robot.data.body_pos_w.torch
        body_quat = self.robot.data.body_quat_w.torch
        residuals = []
        for body0, body1, local0, local1 in _LOOP_POINTS:
            local0_t = torch.tensor(local0, device=self.device, dtype=body_pos.dtype).expand(self.num_envs, -1)
            local1_t = torch.tensor(local1, device=self.device, dtype=body_pos.dtype).expand(self.num_envs, -1)
            p0 = body_pos[:, body_names[body0]] + quat_apply(
                body_quat[:, body_names[body0]], local0_t
            )
            p1 = body_pos[:, body_names[body1]] + quat_apply(
                body_quat[:, body_names[body1]], local1_t
            )
            residuals.append(torch.linalg.vector_norm(p0 - p1, dim=-1))
        loop_residual_m = torch.stack(residuals, dim=-1).amax(dim=-1)
        stand = self._stand_metrics(gravity)
        root_heading_vx = self._heading_forward_velocity(self.robot.data.root_lin_vel_w.torch)
        axle_velocity_w = self.robot.data.body_lin_vel_w.torch[:, self._wheel_body_ids].mean(dim=1)
        axle_heading_vx = self._heading_forward_velocity(axle_velocity_w)
        locomotion = self._locomotion_diagnostics(
            gravity,
            command,
            root_heading_vx,
            stand["leg_length_diff_m"],
        )
        root_xy_distance = torch.linalg.norm(
            self.robot.data.root_pos_w.torch[:, :2] - self.scene.env_origins[:, :2], dim=-1
        )
        return {
            "tilt_deg": tilt_deg,
            "upright": (tilt_deg <= float(self.cfg.curriculum_max_tilt_deg)).float(),
            "wheel_contact_fraction": wheel_contact.float().mean(dim=-1),
            "loop_residual_m": loop_residual_m,
            "linear_velocity_error_mps": torch.abs(root_heading_vx - command[:, 0]),
            "yaw_velocity_error_rad_s": torch.abs(angular_velocity[:, 2] - command[:, 2]),
            "command_target_vx_mps": command[:, 0],
            "command_target_yaw_rad_s": command[:, 2],
            "actual_vx_mps": root_heading_vx,
            "v_actual_over_v_cmd": locomotion["speed_ratio"],
            "body_frame_vx_mps": linear_velocity[:, 0],
            "root_heading_vx_mps": root_heading_vx,
            "axle_heading_vx_mps": axle_heading_vx,
            "actual_yaw_rate_rad_s": angular_velocity[:, 2],
            "root_xy_distance_m": root_xy_distance,
            "root_height_m": self.robot.data.root_pos_w.torch[:, 2],
            "roll_deg": locomotion["roll_deg"],
            "roll_abs_deg": locomotion["roll_abs_deg"],
            "pitch_deg": locomotion["pitch_deg"],
            "turn_fraction": locomotion["turn_fraction"],
            "turn_roll_abs_deg": locomotion["turn_roll_abs_deg"],
            "turn_roll_aligned_deg": locomotion["turn_roll_aligned_deg"],
            "leg_length_diff_m": locomotion["leg_length_diff_m"],
            "turn_leg_length_diff_m": locomotion["turn_leg_length_diff_m"],
            "wheel_torque_abs_mean_nm": locomotion["wheel_torque_abs_mean_nm"],
            "wheel_torque_ratio_mean": locomotion["wheel_torque_ratio_mean"],
            "wheel_torque_saturation_fraction": locomotion["wheel_torque_saturation_fraction"],
            "wheel_slip_speed_mps": locomotion["wheel_slip_speed_mps"],
            "wheel_slip_ratio": locomotion["wheel_slip_ratio"],
            "command_rehearsal_priority_fraction": (
                self._rehearsal_mask
                & torch.isin(
                    self._rehearsal_stage,
                    torch.as_tensor(
                        self.cfg.command_rehearsal_priority_stage_ids,
                        device=self.device,
                        dtype=self._rehearsal_stage.dtype,
                    ),
                )
            ).float().mean(),
            "stand_height_error_m": stand["height_error_m"],
            "leg_extension_score": stand["leg_extension_score"],
            "leg_symmetry_error_rad": stand["symmetry_error_rad"],
            "attitude_score": stand["attitude_score"],
            "wheel_force_imbalance": stand["wheel_force_imbalance"],
            "standing_fraction": stand["standing_fraction"],
            "leg_length_m": stand["leg_length_m"],
            "leg_length_score": stand["leg_length_score"],
            "wheel_height_error_m": stand["wheel_height_error_m"],
            "wheel_height_score": stand["wheel_height_score"],
            "roll_error": stand["roll_error"],
            "roll_score": stand["roll_score"],
            "geometry_symmetry_score": stand["geometry_symmetry_score"],
        }

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self.cfg.curriculum_stage == 1 and hasattr(self, "_command_episode_started"):
            self._update_rehearsal_survival_gate(env_ids)
            # Measure survival on completed episodes. The old implementation
            # sampled current episode age every control step; even full-length
            # episodes spend an initial interval below any long survival gate;
            # count only the final length of completed episodes.
            completed_mask = (
                self._command_episode_started[env_ids]
                & ~self._rehearsal_mask[env_ids]
                & ~self._continuous_command_mask[env_ids]
                & ~self._reset_speed_bootstrap_mask[env_ids]
                & (self._curriculum_step_counter[env_ids] > 0)
            )
            completed_ids = env_ids[completed_mask]
            if completed_ids.numel() > 0:
                unsafe_completion = torch.stack(
                    tuple(self._last_termination_reasons.values()), dim=0
                ).any(dim=0)[completed_ids]
                survived = (
                    self._curriculum_step_counter[completed_ids]
                    >= self._current_command_survival_gate_steps()
                ) & ~unsafe_completion
                survived = survived.float()
                self._command_survival_fraction.copy_(survived.mean())
                batch_size = int(completed_ids.numel())
                previous_survival_samples = self._command_survival_sample_count
                self._command_survival_sample_count += batch_size
                self._command_survival_success_count += int(survived.sum().item())
                survival_ema_denominator = min(
                    self._command_survival_sample_count, self.num_envs
                )
                survival_ema_alpha = min(
                    batch_size / max(float(survival_ema_denominator), 1.0), 1.0
                )
                if previous_survival_samples == 0:
                    self._command_survival_fraction_ema.copy_(survived.mean())
                else:
                    self._command_survival_fraction_ema.lerp_(
                        survived.mean(), survival_ema_alpha
                    )
                if (
                    abs(self.cfg.command_stage_yaw_rate_rad_s[self._command_level])
                    > 1.0e-4
                ):
                    completed_sign = self._command_turn_sign[completed_ids]
                    for direction_index, direction_mask in enumerate(
                        (completed_sign > 0.0, completed_sign < 0.0)
                    ):
                        direction_count = int(direction_mask.sum().item())
                        if direction_count == 0:
                            continue
                        direction_survival = survived[direction_mask].mean()
                        previous_count = int(
                            self._command_turn_survival_episode_count[
                                direction_index
                            ].item()
                        )
                        updated_count = previous_count + direction_count
                        denominator = min(
                            updated_count,
                            int(self.cfg.command_rehearsal_gate_min_episodes),
                        )
                        alpha = min(
                            direction_count / max(float(denominator), 1.0), 1.0
                        )
                        target = self._command_turn_survival_fraction_ema[
                            direction_index
                        ]
                        if previous_count == 0:
                            target.copy_(direction_survival)
                        else:
                            target.lerp_(direction_survival, alpha)
                        self._command_turn_survival_episode_count[
                            direction_index
                        ] = updated_count
            tracking_completed_ids = completed_ids[
                self._command_episode_tracking_step_count[completed_ids] > 0
            ]
            if tracking_completed_ids.numel() > 0:
                completed_steps = self._command_episode_tracking_step_count[
                    tracking_completed_ids
                ].float()
                completed_linear_error = (
                    self._command_episode_linear_error_sum[tracking_completed_ids] / completed_steps
                )
                completed_yaw_error = (
                    self._command_episode_yaw_error_sum[tracking_completed_ids] / completed_steps
                )
                previous_error_samples = self._command_completed_error_sample_count
                tracking_batch_size = int(tracking_completed_ids.numel())
                self._command_completed_error_sample_count += tracking_batch_size
                # Exact cumulative mean during bootstrap, then a capped EMA
                # representing roughly the latest 128 complete unassisted
                # episodes. An all-stage cumulative mean kept obsolete early
                # actors in the gate long after the current policy improved.
                completed_ema_denominator = min(
                    self._command_completed_error_sample_count, self.num_envs
                )
                completed_ema_alpha = min(
                    tracking_batch_size / max(float(completed_ema_denominator), 1.0), 1.0
                )
                if previous_error_samples == 0:
                    self._command_completed_linear_error_mean.copy_(
                        completed_linear_error.mean()
                    )
                    self._command_completed_yaw_error_mean.copy_(completed_yaw_error.mean())
                else:
                    self._command_completed_linear_error_mean.lerp_(
                        completed_linear_error.mean(), completed_ema_alpha
                    )
                    self._command_completed_yaw_error_mean.lerp_(
                        completed_yaw_error.mean(), completed_ema_alpha
                    )
        super()._reset_idx(env_ids)
        self._apply_progressive_domain_randomization(env_ids)
        self._base_com_randomizer.reset(env_ids, self.cfg.base_com_randomization_strength)
        if self._arena_ground_randomizer is not None:
            self._arena_ground_randomizer.reset(env_ids)
        if self._gas_spring_body_ids is not None:
            self._gas_spring_torque_nm[env_ids]=sample_equivalent_torques(len(env_ids),self.cfg.gas_spring_assist_torque_nm,
                self._progressive_domain_randomization_strength(),self.cfg.gas_spring_torque_common_dr_fraction,
                self.cfg.gas_spring_torque_independent_dr_fraction,device=self.device)
        if self._radial_spring_body_ids is not None:
            extent = self.cfg.gas_spring_radial_dr_fraction * self._progressive_domain_randomization_strength()
            if extent > 0:
                scale = torch.empty((len(env_ids), 2), device=self.device).uniform_(1.0 - extent, 1.0 + extent)
                self._radial_spring_force_n[env_ids] = self.cfg.gas_spring_radial_force_n * scale
            else:
                self._radial_spring_force_n[env_ids] = self.cfg.gas_spring_radial_force_n

        self._estop[env_ids] = False
        self._curriculum_step_counter[env_ids] = 0
        self._obs_history_valid[env_ids] = False
        self._command_action_history[env_ids] = 0
        self._obs_history[env_ids] = 0.0
        self._policy_actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._last_speed_feedback_action[env_ids] = 0.0
        self._last_highspeed_propulsion_action[env_ids] = 0.0
        self._last_pitch_feedback_action[env_ids] = 0.0
        self._last_pitch_emergency_feedback_action[env_ids] = 0.0
        self._last_yaw_feedback_action[env_ids] = 0.0
        self._last_raw_policy_actions[env_ids] = 0.0
        self._last_wheel_damping_compensation_nm[env_ids] = 0.0
        self._last_wheel_damping_anneal_fraction[env_ids] = 0.0
        self._last_wheel_damping_anneal_advancing[env_ids] = 0.0
        self._last_effective_wheel_damping[env_ids] = (
            self._wheel_actuator_damping
            * self._domain_wheel_damping_scale[env_ids]
        )
        self._last_effective_wheel_torque_scale[env_ids] = (
            self.cfg.wheel_torque_scale_nm
            * self._domain_wheel_torque_gain[env_ids]
        )
        self._last_effective_wheel_effort_limit[env_ids] = float(
            self._wheel_effort_limit.mean()
        )
        self._last_requested_wheel_effort_nm[env_ids] = 0.0
        self._last_wheel_effort_target_nm[env_ids] = 0.0
        if self._separated_wheel_passive:
            self._last_wheel_mechanical_torque_nm[env_ids] = 0.0
            self._last_wheel_feedback_torque_nm[env_ids] = 0.0
            self._last_wheel_total_passive_torque_nm[env_ids] = 0.0
            self._last_wheel_mechanical_body_torques_w[env_ids] = 0.0
        self._last_action_feedback_delta[env_ids] = 0.0
        self._prepared_policy_action = None
        self._command_velocity_window[env_ids] = 0.0
        self._command_target_velocity_window[env_ids] = 0.0
        self._command_yaw_error_window[env_ids] = 0.0
        self._command_velocity_window_count[env_ids] = 0
        self._reset_speed_bootstrap_mask[env_ids] = False
        self._reset_speed_bootstrap_exact_mask[env_ids] = False
        self._reset_speed_bootstrap_speed[env_ids] = 0.0
        self._reset_speed_bootstrap_pitch_deg[env_ids] = 0.0
        self._previous_command_aligned_speed[env_ids] = 0.0
        self._previous_filtered_command_target[env_ids] = 0.0
        self._previous_reward_direction[env_ids] = 1.0
        self._previous_command_aligned_speed_valid[env_ids] = False
        self._command_episode_linear_error_sum[env_ids] = 0.0
        self._command_episode_yaw_error_sum[env_ids] = 0.0
        self._command_episode_tracking_step_count[env_ids] = 0
        self._level23_rehearsal_window_seen[env_ids] = False
        self._level23_rehearsal_steady_seen[env_ids] = False
        self._refresh_rehearsal_assignments(env_ids)
        self._rebalance_turn_assignments(env_ids)
        self._refresh_continuous_commands(env_ids)
        self._command_episode_started[env_ids] = True
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_pose[:, 3:7] = self._asset_frame_quat[env_ids]
        root_vel = self.robot.data.default_root_vel.torch[env_ids].clone()
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        if (
            self.cfg.curriculum_stage == 1
            and self.cfg.command_reset_speed_bootstrap_enabled
            and self._command_level >= self.cfg.command_reset_speed_bootstrap_start_level
        ):
            eligible = ~self._rehearsal_mask[env_ids] & ~self._continuous_command_mask[env_ids]
            # Use the same population-quota rule as rehearsal. Assisted
            # high-speed episodes have a different termination distribution;
            # Bernoulli-on-reset let their configured ~42% population decay
            # below 20%, removing the target-state samples PPO needed.
            target_bootstrap_count = int(
                round(
                    self.num_envs
                    * (1.0 - float(self._rehearsal_mask.float().mean()))
                    * self.cfg.command_reset_speed_bootstrap_fraction
                )
            )
            current_bootstrap_count = int(
                self._reset_speed_bootstrap_mask.sum().item()
            )
            missing_bootstrap_count = max(
                target_bootstrap_count - current_bootstrap_count, 0
            )
            eligible_local_ids = eligible.nonzero(as_tuple=False).flatten()
            selected_local_ids = eligible_local_ids[
                torch.randperm(eligible_local_ids.numel(), device=self.device)[
                    : min(missing_bootstrap_count, int(eligible_local_ids.numel()))
                ]
            ]
            selected = torch.zeros_like(eligible)
            selected[selected_local_ids] = True
            if torch.any(selected):
                selected_ids = env_ids[selected]
                ratio_low, ratio_high = self.cfg.command_reset_speed_bootstrap_ratio_range
                ratios = torch.empty(selected_ids.numel(), device=self.device).uniform_(
                    ratio_low, ratio_high
                )
                target_exact_count = int(
                    round(
                        target_bootstrap_count
                        * self.cfg.command_reset_speed_bootstrap_exact_target_fraction
                    )
                )
                current_exact_count = int(
                    self._reset_speed_bootstrap_exact_mask.sum().item()
                )
                missing_exact_count = min(
                    max(target_exact_count - current_exact_count, 0),
                    int(selected_ids.numel()),
                )
                exact_target = torch.zeros(
                    selected_ids.numel(), device=self.device, dtype=torch.bool
                )
                if missing_exact_count > 0:
                    exact_target[
                        torch.randperm(selected_ids.numel(), device=self.device)[
                            :missing_exact_count
                        ]
                    ] = True
                ratios[exact_target] = 1.0
                selected_speed = self._current_command()[selected_ids, 0] * ratios
                if self.cfg.command_reset_speed_bootstrap_match_pitch:
                    bootstrap_pitch_deg = torch.clamp(
                        torch.abs(selected_speed)
                        * self.cfg.command_pitch_target_deg_per_mps,
                        0.0,
                        self.cfg.command_pitch_target_max_deg,
                    ) * torch.sign(selected_speed)
                    zeros = torch.zeros_like(bootstrap_pitch_deg)
                    # Canonical pitch is rotation around authored local X after
                    # the asset's +90 deg X frame transform. Its Euler sign is
                    # opposite the canonical pitch diagnostic.
                    pitch_delta = quat_from_euler_xyz(
                        -torch.deg2rad(bootstrap_pitch_deg), zeros, zeros
                    )
                    root_pose[selected, 3:7] = quat_mul(
                        self._asset_frame_quat[selected_ids], pitch_delta
                    )
                    self._reset_speed_bootstrap_pitch_deg[selected_ids] = (
                        bootstrap_pitch_deg
                    )
                authored_forward = torch.zeros((selected_ids.numel(), 3), device=self.device)
                authored_forward[:, 2] = -1.0
                forward_w = quat_apply(root_pose[selected, 3:7], authored_forward)
                forward_xy = forward_w[:, :2] / torch.linalg.vector_norm(
                    forward_w[:, :2], dim=-1, keepdim=True
                ).clamp_min(1.0e-6)
                root_vel[selected, :2] = forward_xy * selected_speed.unsqueeze(-1)
                # The by-body joint order is [left, right], while policy wheel
                # signs are [right, left]. Matching root and surface velocity
                # avoids an artificial contact impulse on the first sim step.
                wheel_spin_sign_by_body = self._wheel_action_sign_tensor[[1, 0]]
                selected_joint_vel = joint_vel[selected]
                selected_joint_vel[:, self._wheel_joint_ids_by_body] = (
                    selected_speed.unsqueeze(-1)
                    / max(self.cfg.wheel_radius_m, 1.0e-6)
                    * wheel_spin_sign_by_body.unsqueeze(0)
                )
                joint_vel[selected] = selected_joint_vel
                self._reset_speed_bootstrap_mask[selected_ids] = True
                self._reset_speed_bootstrap_exact_mask[selected_ids[exact_target]] = True
                self._reset_speed_bootstrap_speed[selected_ids] = selected_speed
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
