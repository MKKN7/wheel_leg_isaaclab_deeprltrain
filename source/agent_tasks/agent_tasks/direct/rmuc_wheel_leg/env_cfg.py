"""Self-contained RMUC closed-loop environment configuration.

This task deliberately lives inside ``wheeled-legged_RL``.  It uses the
bundled USD and V14-style action semantics without importing an external
project at runtime.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml
from gymnasium import spaces

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.configclass import configclass

from agent_world.assets.rmuc_wheel_leg import RMUC_WHEEL_LEG_CFG
from .flat_terrain import RmucFlatTerrainImporterCfg
from .bank_geometry import BANK_CONTRACT


_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "rmuc_wheel_leg.yaml"
_CONFIG = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) if _CONFIG_PATH.exists() else {}
_CONFIG = _CONFIG if isinstance(_CONFIG, dict) else {}
_ROBOT = dict(_CONFIG.get("robot", {}))
_ENV = dict(_CONFIG.get("environment", {}))
_SAFETY = dict(_CONFIG.get("safety", {}))
_ESTOP = dict(_SAFETY.get("estop", {}))
_EXPLOSION = dict(_ESTOP.get("position_explosion", {}))
_TORQUE = dict(_SAFETY.get("torque", {}))
_CONTROL = dict(_CONFIG.get("control", {}))
_COMMAND = dict(_CONFIG.get("command", {}))
_REWARD = dict(_CONFIG.get("reward", {}))
_CURRICULUM = dict(_CONFIG.get("curriculum", {}))
_DOMAIN_RANDOMIZATION = dict(_CONFIG.get("domain_randomization", {}))
_COMMAND_STAGES = tuple(dict(stage) for stage in _COMMAND.get("stages", ()))

DOF_NAMES = tuple(str(name) for name in _ROBOT.get("dof_names", ()))
POLICY_DOF_NAMES = tuple(str(name) for name in _ROBOT.get("controlled_dof_names", ()))
STANDING_JOINT_POS = {
    str(name): float(value) for name, value in dict(_ROBOT.get("standing_joint_pos_profile", {})).items()
}
LEG_POLICY_DOF_NAMES = tuple(name for name in POLICY_DOF_NAMES if "wheel" not in name)
WHEEL_POLICY_DOF_NAMES = tuple(name for name in POLICY_DOF_NAMES if "wheel" in name)
legs_act_name = LEG_POLICY_DOF_NAMES
legs_inact_name = tuple(name for name in DOF_NAMES if name not in POLICY_DOF_NAMES)
wheel_name = WHEEL_POLICY_DOF_NAMES
use_spring = False
use_gimbal = False
OBSERVATION_DIM = 25
ACTION_DIM = len(POLICY_DOF_NAMES)


def _flat_terrain_cfg() -> TerrainImporterCfg:
    terrain_cfg = copy.deepcopy(ROUGH_TERRAINS_CFG)
    terrain_cfg.sub_terrains = {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
    terrain_cfg.num_rows = 1
    terrain_cfg.num_cols = 1
    terrain_cfg.curriculum = False
    # One local tile must cover the full cloned-environment grid. At the
    # default 128 envs and 3 m spacing, 8x8 m left 116 envs without ground.
    flat_size = float(_ENV.get("flat_terrain_size_m", 128.0))
    num_envs = max(int(_ENV.get("num_envs", 128)), 1)
    env_spacing = float(_ENV.get("env_spacing_m", 3.0))
    rows = max(int(math.floor(math.sqrt(num_envs))), 1)
    cols = int(math.ceil(num_envs / rows))
    grid_half_extent = 0.5 * max(rows - 1, cols - 1) * env_spacing
    safety_radius = float(_EXPLOSION.get("max_xy_distance_m", 40.0))
    required_half_extent = grid_half_extent + safety_radius + 1.0
    if 0.5 * flat_size < required_half_extent:
        raise ValueError(
            "RMUC flat terrain is too small for the cloned environment grid: "
            f"half_size={0.5 * flat_size:.1f} m, required={required_half_extent:.1f} m "
            f"for num_envs={num_envs}, spacing={env_spacing:.1f} m, "
            f"max_xy_distance={safety_radius:.1f} m"
        )
    terrain_cfg.size = (flat_size, flat_size)
    terrain_cfg.border_width = 1.0
    terrain_cfg.seed = int(_ENV.get("seed", 42))
    return RmucFlatTerrainImporterCfg(
        flat_mesh_subdivisions=int(_ENV.get("flat_mesh_subdivisions", 5)),
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_cfg,
        max_init_terrain_level=0,
    )


def _rough_terrain_cfg() -> TerrainImporterCfg:
    terrain_cfg = copy.deepcopy(ROUGH_TERRAINS_CFG)
    terrain_cfg.num_rows = 1
    terrain_cfg.num_cols = 1
    terrain_cfg.curriculum = False
    terrain_cfg.size = (8.0, 8.0)
    terrain_cfg.border_width = 1.0
    terrain_cfg.seed = int(_ENV.get("seed", 42))
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_cfg,
        max_init_terrain_level=0,
    )


@configclass
class RmucSceneCfg(InteractiveSceneCfg):
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=int(_CONFIG.get("sensors", {}).get("history_length", 3)),
        track_air_time=True,
        force_threshold=float(_CONFIG.get("sensors", {}).get("force_threshold_N", 2.0)),
    )


@configclass
class RmucWheelLegCourseEnvCfg(DirectRLEnvCfg):
    seed = int(_ENV.get("seed", 42))
    decimation = int(_ENV.get("decimation", 4))
    episode_length_s = float(_ENV.get("episode_length_s", 12.0))
    sim = SimulationCfg(dt=float(_ENV.get("sim_dt", 0.002)), render_interval=decimation)
    scene = RmucSceneCfg(
        num_envs=int(_ENV.get("num_envs", 128)),
        env_spacing=float(_ENV.get("env_spacing_m", 3.0)),
        replicate_physics=True,
    )
    robot_cfg = RMUC_WHEEL_LEG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    terrain = _flat_terrain_cfg()
    observation_space = spaces.Box(low=-1.0e6, high=1.0e6, shape=(OBSERVATION_DIM,), dtype=float)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=float)
    state_space = spaces.Box(low=-1.0e6, high=1.0e6, shape=(OBSERVATION_DIM,), dtype=float)

    # Explicit V14-style semantic bookkeeping. This task is custom because
    # the RMUC topology is not name/count compatible with Wheelbipe V14.
    legs_act_name = LEG_POLICY_DOF_NAMES
    legs_inact_name = tuple(name for name in DOF_NAMES if name not in POLICY_DOF_NAMES)
    wheel_name = WHEEL_POLICY_DOF_NAMES
    use_spring = False
    use_gimbal = False
    curriculum_stage = 0
    curriculum_stage_name = "stand"
    command_enabled = False
    rough_terrain_enabled = False
    friction_randomization_enabled = False
    push_enabled = False
    observation_delay_steps = 0
    push_interval_steps = 0
    push_velocity_xy_mps = 0.0
    friction_static_range = (0.5, 0.5)
    friction_dynamic_range = (0.5, 0.5)
    # Flat-terrain sim-to-real curriculum. Its strength follows command-stage
    # progress, so every increase is backed by the preceding skill gate.
    domain_randomization_enabled = bool(_DOMAIN_RANDOMIZATION.get("enabled", False))
    # Fixed evaluation can reproduce a training level's DR on any command.
    domain_randomization_strength_override = -1.0
    domain_randomization_command_consistent_noise = bool(
        _DOMAIN_RANDOMIZATION.get("command_consistent_noise", False)
    )
    domain_randomization_start_level = int(
        _DOMAIN_RANDOMIZATION.get("start_command_level", 29)
    )
    domain_randomization_full_level = int(
        _DOMAIN_RANDOMIZATION.get("full_command_level", 46)
    )
    domain_randomization_extended_start_level = int(_DOMAIN_RANDOMIZATION.get("extended_start_command_level", -1))
    domain_randomization_extended_full_level = int(_DOMAIN_RANDOMIZATION.get("extended_full_command_level", -1))
    if domain_randomization_extended_start_level >= 0:
        if not (domain_randomization_full_level < domain_randomization_extended_start_level
                <= domain_randomization_extended_full_level < len(_COMMAND_STAGES)):
            raise ValueError("Extended DR schedule must follow legacy DR and end within the curriculum")
    domain_randomization_apply_in_play = bool(
        _DOMAIN_RANDOMIZATION.get("apply_in_play", False)
    )
    domain_randomization_sensor_model = dict(_DOMAIN_RANDOMIZATION.get("sensor_model", {}))
    # Engineering bounds for an optional motor-current/effort response model.
    # It is not another communication FIFO or a measured RM3508 specification.
    motor_response_enabled = bool(_CONTROL.get("motor_response_enabled", False))
    motor_response_time_constant_min_s = float(_CONTROL.get("motor_response_time_constant_min_s", 0.002))
    motor_response_time_constant_max_s = float(_CONTROL.get("motor_response_time_constant_max_s", 0.030))
    if not (math.isfinite(motor_response_time_constant_min_s) and math.isfinite(motor_response_time_constant_max_s)
            and 0 < motor_response_time_constant_min_s <= motor_response_time_constant_max_s):
        raise ValueError("Motor response time constants must be finite, positive and ordered")
    if domain_randomization_sensor_model.get("enabled", False):
        for key, value in domain_randomization_sensor_model.items():
            if key != "enabled" and (not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"Invalid sensor model parameter {key}")
        if domain_randomization_sensor_model["delay_min_steps"] > domain_randomization_sensor_model["delay_max_steps"]:
            raise ValueError("Sensor delay range is reversed")
    domain_randomization_max_base_mass_fraction = float(
        _DOMAIN_RANDOMIZATION.get("max_base_mass_fraction", 0.05)
    )
    # Independent scalar shape uncertainty on top of mass-proportional inertia.
    # Default zero preserves the legacy physics and RNG sequence.
    domain_randomization_max_base_inertia_fraction = float(
        _DOMAIN_RANDOMIZATION.get("max_base_inertia_fraction", 0.0)
    )
    if (not math.isfinite(domain_randomization_max_base_inertia_fraction)
            or not 0.0 <= domain_randomization_max_base_inertia_fraction < 1.0):
        raise ValueError("Independent base inertia fraction must be finite and in [0,1)")
    domain_randomization_max_friction_fraction = float(
        _DOMAIN_RANDOMIZATION.get("max_friction_fraction", 0.15)
    )
    domain_randomization_max_wheel_damping_fraction = float(
        _DOMAIN_RANDOMIZATION.get("max_wheel_damping_fraction", 0.10)
    )
    separated_wheel_mechanical_viscous_range = tuple(
        _DOMAIN_RANDOMIZATION.get("separated_wheel_mechanical_viscous_range", (0.001, 0.005))
    )
    separated_wheel_coulomb_range = tuple(
        _DOMAIN_RANDOMIZATION.get("separated_wheel_coulomb_range", (0.001, 0.001))
    )
    separated_wheel_feedback_damping_range = tuple(
        _DOMAIN_RANDOMIZATION.get("separated_wheel_feedback_damping_range", (0.0, 0.0))
    )
    domain_randomization_max_wheel_torque_gain_fraction = float(
        _DOMAIN_RANDOMIZATION.get("max_wheel_torque_gain_fraction", 0.05)
    )
    domain_randomization_max_observation_noise_std = float(
        _DOMAIN_RANDOMIZATION.get("max_observation_noise_std", 0.01)
    )
    domain_randomization_max_observation_delay_steps = int(
        _DOMAIN_RANDOMIZATION.get("max_observation_delay_steps", 1)
    )
    domain_randomization_max_observation_delay_probability = float(
        _DOMAIN_RANDOMIZATION.get("max_observation_delay_probability", 0.35)
    )
    if domain_randomization_start_level < 0:
        raise ValueError("domain_randomization.start_command_level must be >= 0")
    if domain_randomization_full_level < domain_randomization_start_level:
        raise ValueError(
            "domain_randomization.full_command_level must be >= start_command_level"
        )
    for _name, _value in (
        ("max_base_mass_fraction", domain_randomization_max_base_mass_fraction),
        ("max_friction_fraction", domain_randomization_max_friction_fraction),
        ("max_wheel_damping_fraction", domain_randomization_max_wheel_damping_fraction),
        ("max_wheel_torque_gain_fraction", domain_randomization_max_wheel_torque_gain_fraction),
        ("max_observation_noise_std", domain_randomization_max_observation_noise_std),
    ):
        if _value < 0.0:
            raise ValueError(f"domain_randomization.{_name} must be >= 0")
    if domain_randomization_max_observation_delay_steps < 0:
        raise ValueError(
            "domain_randomization.max_observation_delay_steps must be >= 0"
        )
    if not 0.0 <= domain_randomization_max_observation_delay_probability <= 1.0:
        raise ValueError(
            "domain_randomization.max_observation_delay_probability must be in [0, 1]"
        )
    contact_force_threshold_N = float(_CONFIG.get("sensors", {}).get("force_threshold_N", 2.0))
    curriculum_max_tilt_deg = float(
        _CURRICULUM.get("gate", {}).get("stand", {}).get("max_mean_tilt_deg", 12.0)
    )
    # Stand shaping terms are bounded to [0, 1] in the environment.  Keep
    # their weights comparable so height/extension cannot drown out upright.
    stand_root_height_target_m = float(_REWARD.get("stand_root_height_target_m", _ROBOT.get("initial_pos_z_m", 0.40)))
    stand_root_height_tolerance_m = float(_REWARD.get("stand_root_height_tolerance_m", 0.08))
    stand_leg_pose_tolerance_rad = float(_REWARD.get("stand_leg_pose_tolerance_rad", 0.25))
    stand_symmetry_tolerance_rad = float(_REWARD.get("stand_symmetry_tolerance_rad", 0.20))
    stand_leg_length_target_m = float(_REWARD.get("stand_leg_length_target_m", 0.34))
    stand_leg_length_tolerance_m = float(_REWARD.get("stand_leg_length_tolerance_m", 0.05))
    stand_wheel_height_tolerance_m = float(_REWARD.get("stand_wheel_height_tolerance_m", 0.015))
    stand_roll_tolerance = float(_REWARD.get("stand_roll_tolerance", 0.10))
    stand_height_gate_tolerance_m = float(_CURRICULUM.get("gate", {}).get("stand", {}).get("max_mean_height_error_m", 0.08))
    stand_contact_balance_tolerance = float(_CURRICULUM.get("gate", {}).get("stand", {}).get("max_wheel_force_imbalance", 0.35))

    run_mode = str(_CONFIG.get("run_mode", "sim_training"))
    action_clip = float(_CONTROL.get("action_clip", 1.0))
    leg_mode = str(_CONTROL.get("leg_mode", "position"))
    leg_pair_mapping = str(_CONTROL.get("leg_pair_mapping", "independent"))
    leg_position_scale_rad = float(_CONTROL.get("leg_position_scale_rad", 0.35))
    stand_leg_position_scale_rad = float(_CONTROL.get("stand_leg_position_scale_rad", 0.18))
    command_leg_position_scale_mid_rad = float(_CONTROL.get("command_leg_position_scale_mid_rad", 0.22))
    command_leg_position_scale_final_rad = float(_CONTROL.get("command_leg_position_scale_final_rad", 0.28))
    # Opt-in diagnostic: mechanical drag is outside motor saturation/response.
    # Only separated_passive_v1 uses this explicit PD compute sample/hold.
    leg_pd_update_period_s = 0.002
    separated_wheel_physics_dt_s = 0.001  # 0.002 is an explicit training/diagnostic approximation.
    wheel_torque_contract = str(_CONTROL.get("wheel_torque_contract", "legacy_idealpd"))
    wheel_mechanical_viscous_nm_s_rad = float(_CONTROL.get("wheel_mechanical_viscous_nm_s_rad", 0.002))
    wheel_coulomb_friction_nm = float(_CONTROL.get("wheel_coulomb_friction_nm", 0.001))
    # Diagnostic smoothing, not an identified static-friction transition.
    wheel_coulomb_omega_eps_rad_s = float(_CONTROL.get("wheel_coulomb_omega_eps_rad_s", 0.1))
    # The controller feedback term is distinct from mechanical b and Coulomb.
    # postclip_loss_v1 preserves historical experiments; active_motor_v1 matches
    # the deployment controller's electromagnetic command before its 5.5 Nm clip.
    wheel_feedback_damping_nm_s_rad = float(_CONTROL.get("wheel_feedback_damping_nm_s_rad", 0.0))
    wheel_feedback_damping_mode = str(_CONTROL.get("wheel_feedback_damping_mode", "postclip_loss_v1"))
    wheel_torque_scale_nm = float(_CONTROL.get("wheel_torque_scale_nm", 5.5))
    # User-confirmed active-link output torque with equal opposite chassis reaction.
    # Preserve historical legacy checkpoint physics; separated mode forces xyzw.
    gas_spring_quaternion_order = "wxyz"
    gas_spring_model = str(_CONTROL.get("gas_spring_model", "equivalent_assist_joint_torque"))
    firmware_spring_enabled = bool(_CONTROL.get("firmware_spring_enabled", False))
    firmware_spring_target_kp = float(_CONTROL.get("firmware_spring_target_kp", 60.0))
    firmware_spring_scale = float(_CONTROL.get("firmware_spring_scale", 0.1))
    gas_spring_assist_torque_nm = float(_CONTROL.get("gas_spring_assist_torque_nm", 0.0))
    gas_spring_torque_common_dr_fraction = float(_CONTROL.get("gas_spring_torque_common_dr_fraction", 0.0))
    gas_spring_torque_independent_dr_fraction = float(_CONTROL.get("gas_spring_torque_independent_dr_fraction", 0.0))
    gas_spring_radial_force_n = float(_CONTROL.get("gas_spring_radial_force_n", 0.0))
    gas_spring_radial_dr_fraction = float(_CONTROL.get("gas_spring_radial_dr_fraction", 0.0))
    rehearsal_feedback_parity = bool(_CONTROL.get("rehearsal_feedback_parity", False))
    rehearsal_feedback_straight_only = bool(_CONTROL.get("rehearsal_feedback_straight_only", False))
    feedback_contract_by_command = bool(_CONTROL.get("feedback_contract_by_command", False))
    command_reward_contract_by_command = False
    command_bank_contract = BANK_CONTRACT
    # User: operator-command tracking; no prescribed analytic body lean.
    command_require_analytic_bank = False
    command_bank_reward_contract = "linear_near_exponential_tail_v1"
    # Transitional teacher only; final E2E acceptance requires False.
    policy_outer_feedback_enabled = True
    position_authority_enabled = False
    position_authority_speed_plateau_end = 1.0
    leg_actuator_dr_enabled = bool(_DOMAIN_RANDOMIZATION.get("leg_actuator_enabled", True))
    leg_actuator_velocity_dr_range = tuple(_DOMAIN_RANDOMIZATION.get("leg_velocity_limit_range_rad_s", (12.0, 17.0)))
    leg_actuator_armature_dr_range = tuple(_DOMAIN_RANDOMIZATION.get("leg_armature_range_kg_m2", (0.008, 0.024)))
    command_low_speed_turn_yaw_tracking_enabled = False
    command_small_turn_dense_yaw_weight = 0.0
    command_small_turn_dense_yaw_max_rad_s = 0.4501
    command_low_speed_turn_yaw_error_scale_rad_s = 0.10
    wheel_authored_effort_limit_nm = float(
        _CONTROL.get("wheel_authored_effort_limit_nm", float("inf"))
    )
    if wheel_authored_effort_limit_nm <= 0.0:
        raise ValueError("control.wheel_authored_effort_limit_nm must be positive")
    wheel_radius_m = float(_CONTROL.get("wheel_radius_m", 0.06))
    wheel_action_sign = tuple(float(value) for value in _CONTROL.get("wheel_action_sign", (-1.0, -1.0)))
    wheel_yaw_feedback_action_per_rad_s = float(
        _CONTROL.get("wheel_yaw_feedback_action_per_rad_s", 0.0)
    )
    wheel_yaw_feedback_action_clip = float(
        _CONTROL.get("wheel_yaw_feedback_action_clip", 0.0)
    )
    wheel_yaw_feedback_low_speed_action_clip = float(
        _CONTROL.get("wheel_yaw_feedback_low_speed_action_clip", wheel_yaw_feedback_action_clip)
    )
    wheel_yaw_feedback_straight_action_clip = float(
        _CONTROL.get(
            "wheel_yaw_feedback_straight_action_clip",
            wheel_yaw_feedback_action_clip,
        )
    )
    wheel_speed_feedback_action_per_mps = float(
        _CONTROL.get("wheel_speed_feedback_action_per_mps", 0.0)
    )
    wheel_speed_feedback_high_speed_threshold_mps = float(
        _CONTROL.get("wheel_speed_feedback_high_speed_threshold_mps", float("inf"))
    )
    wheel_speed_feedback_high_speed_action_per_mps = float(
        _CONTROL.get(
            "wheel_speed_feedback_high_speed_action_per_mps",
            wheel_speed_feedback_action_per_mps,
        )
    )
    wheel_speed_feedback_action_clip = float(
        _CONTROL.get("wheel_speed_feedback_action_clip", 0.0)
    )
    wheel_highspeed_feedback_overspeed_action_clip = float(
        _CONTROL.get("wheel_highspeed_feedback_overspeed_action_clip", 0.0)
    )
    if wheel_highspeed_feedback_overspeed_action_clip < 0.0:
        raise ValueError(
            "control.wheel_highspeed_feedback_overspeed_action_clip must be >= 0"
        )
    wheel_highspeed_propulsion_start_mps = float(
        _CONTROL.get("wheel_highspeed_propulsion_start_mps", float("inf"))
    )
    wheel_highspeed_propulsion_action_per_mps = float(
        _CONTROL.get("wheel_highspeed_propulsion_action_per_mps", 0.0)
    )
    wheel_highspeed_propulsion_action_clip = float(
        _CONTROL.get("wheel_highspeed_propulsion_action_clip", 0.0)
    )
    wheel_highspeed_propulsion_overspeed_fade_mps = float(
        _CONTROL.get("wheel_highspeed_propulsion_overspeed_fade_mps", 0.20)
    )
    if wheel_highspeed_propulsion_action_per_mps < 0.0:
        raise ValueError("control.wheel_highspeed_propulsion_action_per_mps must be >= 0")
    if wheel_highspeed_propulsion_action_clip < 0.0:
        raise ValueError("control.wheel_highspeed_propulsion_action_clip must be >= 0")
    if wheel_highspeed_propulsion_overspeed_fade_mps <= 0.0:
        raise ValueError("control.wheel_highspeed_propulsion_overspeed_fade_mps must be positive")
    wheel_speed_feedback_damping_anneal_target_scale = float(
        _CONTROL.get("wheel_speed_feedback_damping_anneal_target_scale", 0.0)
    )
    if not 0.0 <= wheel_speed_feedback_damping_anneal_target_scale <= 1.0:
        raise ValueError(
            "control.wheel_speed_feedback_damping_anneal_target_scale must be in [0, 1]"
        )
    wheel_speed_feedback_negative_turn_scale = float(
        _CONTROL.get("wheel_speed_feedback_negative_turn_scale", 1.0)
    )
    if wheel_speed_feedback_negative_turn_scale <= 0.0:
        raise ValueError("control.wheel_speed_feedback_negative_turn_scale must be positive")
    wheel_pitch_feedback_start_level = int(
        _CONTROL.get("wheel_pitch_feedback_start_level", 14)
    )
    wheel_pitch_feedback_action_per_rad = float(
        _CONTROL.get("wheel_pitch_feedback_action_per_rad", 0.0)
    )
    wheel_pitch_rate_feedback_action_per_rad_s = float(
        _CONTROL.get("wheel_pitch_rate_feedback_action_per_rad_s", 0.0)
    )
    wheel_pitch_rate_emergency_threshold_rad_s = float(
        _CONTROL.get("wheel_pitch_rate_emergency_threshold_rad_s", 1.0)
    )
    wheel_pitch_rate_emergency_action_per_rad_s = float(
        _CONTROL.get("wheel_pitch_rate_emergency_action_per_rad_s", 0.0)
    )
    wheel_pitch_feedback_action_clip = float(
        _CONTROL.get("wheel_pitch_feedback_action_clip", 0.0)
    )
    wheel_pitch_feedback_min_upright_cos = float(
        _CONTROL.get("wheel_pitch_feedback_min_upright_cos", 0.0)
    )
    if wheel_pitch_feedback_start_level < 0:
        raise ValueError("control.wheel_pitch_feedback_start_level must be >= 0")
    if wheel_pitch_feedback_action_clip < 0.0:
        raise ValueError("control.wheel_pitch_feedback_action_clip must be >= 0")
    if not -1.0 <= wheel_pitch_feedback_min_upright_cos <= 1.0:
        raise ValueError(
            "control.wheel_pitch_feedback_min_upright_cos must be in [-1, 1]"
        )
    # The legacy policy was trained with an explicit Isaac wheel damping of
    # 0.1, while the source XML wheel joints use 0.01 mechanical damping. A
    # zero default preserves checkpoint compatibility; high-speed runs can
    # ramp this feed-forward compensation instead of changing the actuator
    # model abruptly.
    wheel_damping_compensation_ratio = float(
        _CONTROL.get("wheel_damping_compensation_ratio", 0.0)
    )
    wheel_damping_compensation_start_level = int(
        _CONTROL.get("wheel_damping_compensation_start_level", 14)
    )
    wheel_damping_compensation_margin_mps = float(
        _CONTROL.get("wheel_damping_compensation_margin_mps", 0.20)
    )
    wheel_damping_compensation_scale_mps = float(
        _CONTROL.get("wheel_damping_compensation_scale_mps", 0.40)
    )
    # Runtime actuator bridge. A negative target disables the bridge and keeps
    # legacy checkpoints on their authored Isaac damping unchanged.
    wheel_damping_anneal_target = float(
        _CONTROL.get("wheel_damping_anneal_target", -1.0)
    )
    wheel_damping_anneal_steps = int(
        _CONTROL.get("wheel_damping_anneal_steps", 2_000_000)
    )
    wheel_damping_anneal_warmup_steps = int(
        _CONTROL.get("wheel_damping_anneal_warmup_steps", 0)
    )
    wheel_damping_anneal_start_level = int(
        _CONTROL.get("wheel_damping_anneal_start_level", 16)
    )
    wheel_damping_anneal_adaptive = bool(
        _CONTROL.get("wheel_damping_anneal_adaptive", True)
    )
    wheel_damping_anneal_max_linear_error_mps = float(
        _CONTROL.get("wheel_damping_anneal_max_linear_error_mps", 0.12)
    )
    wheel_damping_anneal_min_stable_fraction = float(
        _CONTROL.get("wheel_damping_anneal_min_stable_fraction", 0.45)
    )
    wheel_damping_anneal_min_survival_fraction = float(
        _CONTROL.get("wheel_damping_anneal_min_survival_fraction", 0.95)
    )
    if not 0.0 <= wheel_damping_compensation_ratio <= 1.0:
        raise ValueError("control.wheel_damping_compensation_ratio must be in [0, 1]")
    if wheel_damping_compensation_start_level < 0:
        raise ValueError("control.wheel_damping_compensation_start_level must be >= 0")
    if wheel_damping_compensation_margin_mps < 0.0:
        raise ValueError("control.wheel_damping_compensation_margin_mps must be >= 0")
    if wheel_damping_compensation_scale_mps <= 0.0:
        raise ValueError("control.wheel_damping_compensation_scale_mps must be positive")
    if wheel_damping_anneal_target >= 0.0 and wheel_damping_anneal_steps < 1:
        raise ValueError("control.wheel_damping_anneal_steps must be >= 1 when enabled")
    if wheel_damping_anneal_warmup_steps < 0:
        raise ValueError("control.wheel_damping_anneal_warmup_steps must be >= 0")
    if wheel_damping_anneal_start_level < 0:
        raise ValueError("control.wheel_damping_anneal_start_level must be >= 0")
    if wheel_damping_anneal_max_linear_error_mps <= 0.0:
        raise ValueError(
            "control.wheel_damping_anneal_max_linear_error_mps must be positive"
        )
    if not 0.0 <= wheel_damping_anneal_min_stable_fraction <= 1.0:
        raise ValueError(
            "control.wheel_damping_anneal_min_stable_fraction must be in [0, 1]"
        )
    if not 0.0 <= wheel_damping_anneal_min_survival_fraction <= 1.0:
        raise ValueError(
            "control.wheel_damping_anneal_min_survival_fraction must be in [0, 1]"
        )
    leg_joint_limit_margin_rad = float(_CONTROL.get("leg_joint_limit_margin_rad", 0.02))
    command_linear_x_mps = float(_COMMAND.get("linear_x_mps", 0.0))
    command_yaw_rate_rad_s = float(_COMMAND.get("yaw_rate_rad_s", 0.0))
    command_start_level = int(_COMMAND.get("start_level", 0))
    command_start_at_target = bool(_COMMAND.get("start_at_target", False))
    command_reset_acquisition_ramp_s = float(_COMMAND.get("reset_acquisition_ramp_s", 0.0))
    command_turn_entry_hold_s = 0.0
    command_turn_entry_ramp_s = 0.0
    command_turn_entry_abrupt_fraction = 0.0
    if not math.isfinite(command_reset_acquisition_ramp_s) or not 0 <= command_reset_acquisition_ramp_s <= 4:
        raise ValueError("Reset acquisition ramp must be within0..4seconds")
    command_observation_scale = tuple(
        float(value) for value in _COMMAND.get("observation_scale", (2.10, 0.50, 1.00))
    )
    command_observation_mode = str(_COMMAND.get("observation_mode", "tracking_error"))
    if command_observation_mode not in {
        "tracking_error",
        "tracking_error_with_target",
        "target_with_forward_velocity",
        "target",
    }:
        raise ValueError(
            f"Unsupported command.observation_mode: {command_observation_mode!r}"
        )
    command_observation_error_clip = float(_COMMAND.get("observation_error_clip", 2.0))
    command_yaw_observation_error_clip = float(_COMMAND.get("yaw_observation_error_clip", command_observation_error_clip))
    if not math.isfinite(command_yaw_observation_error_clip) or command_yaw_observation_error_clip <= 0:
        raise ValueError("Yaw observation error clip must be finite and positive")
    command_bootstrap_linear_x_mps = float(
        _COMMAND.get("bootstrap_linear_x_mps", command_linear_x_mps)
    )
    command_bootstrap_yaw_rate_rad_s = float(
        _COMMAND.get("bootstrap_yaw_rate_rad_s", command_yaw_rate_rad_s)
    )
    command_bootstrap_leg_scale_rad = float(
        _COMMAND.get("bootstrap_leg_scale_rad", _CONTROL.get("stand_leg_position_scale_rad", 0.18))
    )
    command_turn_start_ratio = float(_COMMAND.get("turn_start_ratio", 0.60))
    command_curriculum_mid_ratio = float(_COMMAND.get("curriculum_mid_ratio", 0.33))
    command_curriculum_early_speed_mps = float(_COMMAND.get("curriculum_early_speed_mps", 0.12))
    command_curriculum_mid_speed_mps = float(_COMMAND.get("curriculum_mid_speed_mps", 0.20))
    command_gate_steps = int(_COMMAND.get("gate_steps", 300))
    command_stage_hold_steps = int(_COMMAND.get("stage_hold_steps", 500_000))
    command_transition_steps = int(_COMMAND.get("transition_steps", 500_000))
    command_survival_gate_steps = int(_COMMAND.get("survival_gate_steps", 1200))
    command_rehearsal_enabled = bool(_COMMAND.get("rehearsal_enabled", True))
    # Explicit isolated-training sampling; -1 preserves adaptive direction allocation.
    command_turn_positive_fraction_override = -1.0
    command_turn_fixed_sign_cohorts = False
    command_rehearsal_fraction = float(_COMMAND.get("rehearsal_fraction", 0.30))
    # Additional training cohort; excluded from fixed-command certificates.
    command_continuous_sampling_fraction = float(_COMMAND.get("continuous_sampling_fraction", 0.0))
    command_speed_extension_sampling_box = ()
    command_speed_extension_anchors = ()
    command_speed_extension_anchor_fraction = 0.0
    command_final_turn_envelope = tuple(dict(row) for row in _COMMAND.get("final_turn_envelope", ()))
    command_extended_sampling_start_level = _COMMAND.get("extended_sampling_start_level", None)
    if command_extended_sampling_start_level is not None:
        command_extended_sampling_start_level = int(command_extended_sampling_start_level)
        if not command_final_turn_envelope or not 0 <= command_extended_sampling_start_level < len(_COMMAND_STAGES):
            raise ValueError("Extended sampling requires a final envelope and valid start level")
    if not 0.0 <= command_continuous_sampling_fraction <= 0.25:
        raise ValueError("Continuous command cohort must be in [0,0.25]")
    command_rehearsal_speed_focus_start_level = int(
        _COMMAND.get("rehearsal_speed_focus_start_level", 13)
    )
    command_rehearsal_speed_focus_end_level = int(
        _COMMAND.get("rehearsal_speed_focus_end_level", 15)
    )
    command_rehearsal_speed_focus_fraction = float(
        _COMMAND.get("rehearsal_speed_focus_fraction", 0.15)
    )
    command_rehearsal_start_level = int(_COMMAND.get("rehearsal_start_level", 8))
    command_rehearsal_anchor_stage_ids = tuple(
        int(value) for value in _COMMAND.get("rehearsal_anchor_stage_ids", (1, 3, 4, 5, 8))
    )
    command_rehearsal_priority_stage_ids = tuple(
        int(value) for value in _COMMAND.get("rehearsal_priority_stage_ids", (5, 8))
    )
    command_rehearsal_priority_stage_weights = tuple(
        float(value)
        for value in _COMMAND.get(
            "rehearsal_priority_stage_weights",
            (1.0,) * len(command_rehearsal_priority_stage_ids),
        )
    )
    command_rehearsal_priority_fraction = float(
        _COMMAND.get("rehearsal_priority_fraction", 0.75)
    )
    command_rehearsal_gate_stage_ids = tuple(
        int(value) for value in _COMMAND.get("rehearsal_gate_stage_ids", (4, 8, 13))
    )
    command_rehearsal_gate_min_samples = int(
        _COMMAND.get("rehearsal_gate_min_samples", command_gate_steps)
    )
    command_rehearsal_gate_min_episodes = int(
        _COMMAND.get("rehearsal_gate_min_episodes", 16)
    )
    command_rehearsal_gate_enforce_start_level = int(
        _COMMAND.get("rehearsal_gate_enforce_start_level", 15)
    )
    command_reset_speed_bootstrap_enabled = bool(
        _COMMAND.get("reset_speed_bootstrap_enabled", False)
    )
    command_reset_speed_bootstrap_start_level = int(
        _COMMAND.get("reset_speed_bootstrap_start_level", 9)
    )
    command_reset_speed_bootstrap_fraction = float(
        _COMMAND.get("reset_speed_bootstrap_fraction", 0.50)
    )
    command_reset_speed_bootstrap_exact_target_fraction = float(
        _COMMAND.get("reset_speed_bootstrap_exact_target_fraction", 0.50)
    )
    command_reset_speed_bootstrap_ratio_range = tuple(
        float(value)
        for value in _COMMAND.get("reset_speed_bootstrap_ratio_range", (0.80, 1.00))
    )
    command_reset_speed_bootstrap_match_pitch = bool(
        _COMMAND.get("reset_speed_bootstrap_match_pitch", True)
    )
    if not 0.0 <= command_rehearsal_fraction < 1.0:
        raise ValueError("command.rehearsal_fraction must be in [0, 1)")
    if not 0.0 <= command_rehearsal_speed_focus_fraction < 1.0:
        raise ValueError("command.rehearsal_speed_focus_fraction must be in [0, 1)")
    if command_rehearsal_speed_focus_end_level <= command_rehearsal_speed_focus_start_level:
        raise ValueError(
            "command.rehearsal_speed_focus_end_level must be above its start level"
        )
    if not 0.0 <= command_rehearsal_priority_fraction <= 1.0:
        raise ValueError("command.rehearsal_priority_fraction must be in [0, 1]")
    if (
        len(command_rehearsal_priority_stage_weights)
        != len(command_rehearsal_priority_stage_ids)
        or any(weight <= 0.0 for weight in command_rehearsal_priority_stage_weights)
        or sum(command_rehearsal_priority_stage_weights) <= 0.0
    ):
        raise ValueError(
            "command.rehearsal_priority_stage_weights must contain one positive "
            "weight per rehearsal_priority_stage_ids entry and have a positive sum"
        )
    if command_rehearsal_gate_min_samples < 1:
        raise ValueError("command.rehearsal_gate_min_samples must be >= 1")
    if command_rehearsal_gate_min_episodes < 1:
        raise ValueError("command.rehearsal_gate_min_episodes must be >= 1")
    if command_rehearsal_gate_enforce_start_level < 0:
        raise ValueError("command.rehearsal_gate_enforce_start_level must be >= 0")
    if not 0.0 <= command_reset_speed_bootstrap_fraction < 1.0:
        raise ValueError("command.reset_speed_bootstrap_fraction must be in [0, 1)")
    if not 0.0 <= command_reset_speed_bootstrap_exact_target_fraction <= 1.0:
        raise ValueError(
            "command.reset_speed_bootstrap_exact_target_fraction must be in [0, 1]"
        )
    if (
        len(command_reset_speed_bootstrap_ratio_range) != 2
        or command_reset_speed_bootstrap_ratio_range[0] < 0.0
        or command_reset_speed_bootstrap_ratio_range[1]
        < command_reset_speed_bootstrap_ratio_range[0]
        or command_reset_speed_bootstrap_ratio_range[1] > 1.0
    ):
        raise ValueError(
            "command.reset_speed_bootstrap_ratio_range must be [min, max] within [0, 1]"
        )
    command_min_posture_env_fraction = float(_COMMAND.get("min_posture_env_fraction", 0.90))
    command_min_wheel_contact_fraction = float(_COMMAND.get("min_wheel_contact_fraction", 0.55))
    command_min_instantaneous_wheel_contact_fraction = float(
        _COMMAND.get("min_instantaneous_wheel_contact_fraction", 0.50)
    )
    command_min_survival_env_fraction = float(_COMMAND.get("min_survival_env_fraction", 0.95))
    command_min_stable_env_fraction = float(_COMMAND.get("min_stable_env_fraction", 0.45))
    command_min_ready_env_fraction = float(_COMMAND.get("min_ready_env_fraction", 0.70))
    command_min_operating_root_height_m = float(_COMMAND.get("min_operating_root_height_m", 0.15))
    command_advance_linear_error_mps = float(_COMMAND.get("advance_linear_error_mps", 0.35))
    command_high_speed_gate_start_level = int(
        _COMMAND.get("high_speed_gate_start_level", 8)
    )
    command_high_speed_advance_linear_error_mps = float(
        _COMMAND.get("high_speed_advance_linear_error_mps", 0.15)
    )
    command_advance_yaw_error_rad_s = float(_COMMAND.get("advance_yaw_error_rad_s", 0.05))
    command_turn_gate_min_samples_per_direction = int(
        _COMMAND.get("turn_gate_min_samples_per_direction", command_gate_steps)
    )
    command_turn_gate_max_bank_error_deg = float(
        _COMMAND.get("turn_gate_max_bank_error_deg", 3.0)
    )
    if not 0.0 < command_turn_gate_max_bank_error_deg < 90.0:
        raise ValueError("command.turn_gate_max_bank_error_deg must be in (0,90)")
    command_turn_adaptive_focus_start_level = int(
        _COMMAND.get(
            "turn_adaptive_focus_start_level",
            _COMMAND.get("turn_positive_focus_start_level", 15),
        )
    )
    command_turn_adaptive_focus_fraction = float(
        _COMMAND.get(
            "turn_adaptive_focus_fraction",
            _COMMAND.get("turn_positive_focus_fraction", 0.65),
        )
    )
    command_advance_confirmation_steps = int(
        _COMMAND.get("advance_confirmation_steps", 125)
    )
    if not 0.0 <= command_min_ready_env_fraction <= 1.0:
        raise ValueError("command.min_ready_env_fraction must be in [0, 1]")
    if not 0.0 <= command_min_wheel_contact_fraction <= 1.0:
        raise ValueError("command.min_wheel_contact_fraction must be in [0, 1]")
    if not 0.0 <= command_min_instantaneous_wheel_contact_fraction <= 1.0:
        raise ValueError(
            "command.min_instantaneous_wheel_contact_fraction must be in [0, 1]"
        )
    if command_advance_confirmation_steps < 1:
        raise ValueError("command.advance_confirmation_steps must be >= 1")
    if command_turn_gate_min_samples_per_direction < 1:
        raise ValueError("command.turn_gate_min_samples_per_direction must be >= 1")
    if not 0 <= command_turn_adaptive_focus_start_level < len(_COMMAND_STAGES):
        raise ValueError(
            "command.turn_adaptive_focus_start_level must select a configured command stage"
        )
    if not 0.5 < command_turn_adaptive_focus_fraction < 1.0:
        raise ValueError("command.turn_adaptive_focus_fraction must be in (0.5, 1)")
    command_stage_names = tuple(str(stage.get("name", f"level_{index}")) for index, stage in enumerate(_COMMAND_STAGES))
    command_stage_linear_x_mps = tuple(float(stage.get("linear_x_mps", 0.0)) for stage in _COMMAND_STAGES)
    command_stage_yaw_rate_rad_s = tuple(float(stage.get("yaw_rate_rad_s", 0.0)) for stage in _COMMAND_STAGES)
    command_stage_leg_scale_rad = tuple(
        float(stage.get("leg_scale_rad", _CONTROL.get("stand_leg_position_scale_rad", 0.18)))
        for stage in _COMMAND_STAGES
    )
    command_stage_transition_steps = tuple(
        int(stage.get("transition_steps", _COMMAND.get("transition_steps", 500_000)))
        for stage in _COMMAND_STAGES
    )
    command_stage_survival_gate_steps = tuple(
        int(
            stage.get(
                "survival_gate_steps",
                int(_COMMAND.get("survival_gate_steps", 1200)),
            )
        )
        for stage in _COMMAND_STAGES
    )
    command_stage_advance_linear_error_mps = tuple(
        float(
            stage.get(
                "advance_linear_error_mps",
                float(_COMMAND.get("high_speed_advance_linear_error_mps", 0.15))
                if index >= int(_COMMAND.get("high_speed_gate_start_level", 8))
                else float(_COMMAND.get("advance_linear_error_mps", 0.35)),
            )
        )
        for index, stage in enumerate(_COMMAND_STAGES)
    )
    if any(steps <= 0 for steps in command_stage_transition_steps):
        raise ValueError("command stage transition_steps must be positive")
    if any(steps <= 0 for steps in command_stage_survival_gate_steps):
        raise ValueError("command stage survival_gate_steps must be positive")
    if any(limit <= 0.0 for limit in command_stage_advance_linear_error_mps):
        raise ValueError("command stage advance_linear_error_mps must be positive")
    if any(
        stage < 0 or stage >= len(_COMMAND_STAGES)
        for stage in command_rehearsal_gate_stage_ids
    ):
        raise ValueError("command.rehearsal_gate_stage_ids contains an invalid stage")
    command_forward_weight = float(_REWARD.get("command_forward_weight", 3.00))
    command_progress_weight = float(_REWARD.get("command_progress_weight", 1.50))
    command_tracking_error_penalty_weight = float(
        _REWARD.get("command_tracking_error_penalty_weight", 3.00)
    )
    command_speed_error_margin_mps = float(
        _REWARD.get("command_speed_error_margin_mps", 0.15)
    )
    command_speed_error_scale_mps = float(
        _REWARD.get("command_speed_error_scale_mps", 0.20)
    )
    command_speed_error_penalty_weight = float(
        _REWARD.get("command_speed_error_penalty_weight", 6.00)
    )
    command_tracking_error_tail_enabled = bool(
        _REWARD.get("command_tracking_error_tail_enabled", False)
    )
    command_tracking_error_tail_gain = float(
        _REWARD.get("command_tracking_error_tail_gain", 0.1)
    )
    if not 0.0 <= command_tracking_error_tail_gain <= 1.0:
        raise ValueError("reward.command_tracking_error_tail_gain must be in [0, 1]")
    if command_speed_error_margin_mps < 0.0:
        raise ValueError("reward.command_speed_error_margin_mps must be non-negative")
    if command_speed_error_scale_mps <= 0.0:
        raise ValueError("reward.command_speed_error_scale_mps must be positive")
    command_reverse_penalty_weight = float(_REWARD.get("command_reverse_penalty_weight", 2.00))
    command_overspeed_margin_mps = float(
        _REWARD.get("command_overspeed_margin_mps", 0.05)
    )
    command_overspeed_scale_mps = float(
        _REWARD.get("command_overspeed_scale_mps", 0.15)
    )
    command_overspeed_penalty_weight = float(_REWARD.get("command_overspeed_penalty_weight", 4.00))
    if command_overspeed_margin_mps < 0.0:
        raise ValueError("reward.command_overspeed_margin_mps must be non-negative")
    if command_overspeed_scale_mps <= 0.0:
        raise ValueError("reward.command_overspeed_scale_mps must be positive")
    command_yaw_weight = float(_REWARD.get("command_yaw_weight", 2.00))
    command_yaw_tracking_penalty_weight = float(
        _REWARD.get("command_yaw_tracking_penalty_weight", 1.50)
    )
    command_turn_yaw_tracking_start_level = int(
        _REWARD.get("command_turn_yaw_tracking_start_level", 28)
    )
    command_turn_yaw_tracking_penalty_weight = float(
        _REWARD.get("command_turn_yaw_tracking_penalty_weight", 5.00)
    )
    command_yaw_stabilization_weight = float(_REWARD.get("command_yaw_stabilization_weight", 1.00))
    command_lateral_weight = float(_REWARD.get("command_lateral_weight", 0.35))
    spin_asymmetric_reward_enabled = bool(_REWARD.get("spin_asymmetric_reward_enabled", False))
    spin_overspeed_deadband_rad_s = float(_REWARD.get("spin_overspeed_deadband_rad_s", 0.5))
    spin_underspeed_penalty_weight = float(_REWARD.get("spin_underspeed_penalty_weight", 10.0))
    spin_underspeed_scale_rad_s = float(_REWARD.get("spin_underspeed_scale_rad_s", 0.5))
    for value in (spin_overspeed_deadband_rad_s, spin_underspeed_penalty_weight, spin_underspeed_scale_rad_s):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Spin reward parameters must be finite and nonnegative")
    if spin_underspeed_scale_rad_s <= 0:
        raise ValueError("Spin underspeed scale must be positive")
    command_yaw_overspeed_relative_to_target = bool(_REWARD.get("command_yaw_overspeed_relative_to_target", False))
    command_yaw_rate_penalty_weight = float(_REWARD.get("command_yaw_rate_penalty_weight", 0.25))
    command_wheel_sync_weight = float(_REWARD.get("command_wheel_sync_weight", 1.50))
    command_leg_sync_weight = float(_REWARD.get("command_leg_sync_weight", 0.35))
    command_roll_penalty_weight = float(_REWARD.get("command_roll_penalty_weight", 0.40))
    command_bank_tracking_weight = float(_REWARD.get("command_bank_tracking_weight", 1.50))
    command_active_bank_reward_scale = float(_REWARD.get("command_active_bank_reward_scale", 1.0))
    if command_active_bank_reward_scale <= 0.0:
        raise ValueError("reward.command_active_bank_reward_scale must be positive")
    command_bank_tolerance_deg = float(_REWARD.get("command_bank_tolerance_deg", 1.50))
    command_bank_error_penalty_weight = float(
        _REWARD.get("command_bank_error_penalty_weight", 2.00)
    )
    command_bank_error_scale_deg = float(
        _REWARD.get("command_bank_error_scale_deg", 6.00)
    )
    command_turn_roll_scale = float(_REWARD.get("command_turn_roll_scale", 0.25))
    command_turn_attitude_scale = float(_REWARD.get("command_turn_attitude_scale", 0.25))
    command_stillness_weight = float(_REWARD.get("command_stillness_weight", 0.05))
    command_height_penalty_weight = float(_REWARD.get("command_height_penalty_weight", 0.50))
    command_posture_weight = float(_REWARD.get("command_posture_weight", 0.40))
    command_stand_shape_weight = float(_REWARD.get("command_stand_shape_weight", 0.50))
    command_leg_length_preference_scale = float(_REWARD.get("command_leg_length_preference_scale", 0.0))
    if not 0.0 <= command_leg_length_preference_scale <= 1.0:
        raise ValueError("Locomotion leg length preference scale must be finite in [0,1]")
    command_contact_weight = float(_REWARD.get("command_contact_weight", 0.25))
    command_min_height_m = float(_REWARD.get("command_min_height_m", 0.15))
    command_survival_bonus = float(_REWARD.get("command_survival_bonus", 0.10))
    command_termination_penalty_weight = float(
        _REWARD.get("command_termination_penalty_weight", 0.0)
    )
    reward_clip_min = float(_REWARD.get("clip_min", -10.0))
    reward_clip_max = float(_REWARD.get("clip_max", 10.0))
    if reward_clip_min >= reward_clip_max:
        raise ValueError("reward.clip_min must be below reward.clip_max")
    command_underspeed_penalty_weight = float(
        _REWARD.get("command_underspeed_penalty_weight", 1.50)
    )
    command_severe_underspeed_margin_mps = float(
        _REWARD.get("command_severe_underspeed_margin_mps", 0.25)
    )
    command_severe_underspeed_start_level = int(
        _REWARD.get("command_severe_underspeed_start_level", 13)
    )
    command_severe_underspeed_grace_steps = int(
        _REWARD.get("command_severe_underspeed_grace_steps", 250)
    )
    command_severe_underspeed_scale_mps = float(
        _REWARD.get("command_severe_underspeed_scale_mps", 0.25)
    )
    command_severe_underspeed_penalty_weight = float(
        _REWARD.get("command_severe_underspeed_penalty_weight", 0.0)
    )
    if command_severe_underspeed_margin_mps < 0.0:
        raise ValueError("reward.command_severe_underspeed_margin_mps must be >= 0")
    if command_severe_underspeed_start_level < 0:
        raise ValueError("reward.command_severe_underspeed_start_level must be >= 0")
    if command_severe_underspeed_grace_steps < 0:
        raise ValueError("reward.command_severe_underspeed_grace_steps must be >= 0")
    if command_severe_underspeed_scale_mps <= 0.0:
        raise ValueError("reward.command_severe_underspeed_scale_mps must be > 0")
    if command_severe_underspeed_penalty_weight < 0.0:
        raise ValueError("reward.command_severe_underspeed_penalty_weight must be >= 0")
    command_high_speed_exploration_weight = float(
        _REWARD.get("command_high_speed_exploration_weight", 0.75)
    )
    command_speed_delta_scale_mps = float(
        _REWARD.get("command_speed_delta_scale_mps", 0.02)
    )
    command_speed_delta_filter_tau_s = float(
        _REWARD.get("command_speed_delta_filter_tau_s", 0.16)
    )
    command_speed_acquisition_weight = float(
        _REWARD.get("command_speed_acquisition_weight", 0.75)
    )
    command_speed_regression_penalty_weight = float(
        _REWARD.get("command_speed_regression_penalty_weight", 1.00)
    )
    command_filtered_speed_error_penalty_weight = float(
        _REWARD.get("command_filtered_speed_error_penalty_weight", 2.00)
    )
    command_speed_oscillation_start_level = int(
        _REWARD.get("command_speed_oscillation_start_level", 13)
    )
    command_speed_oscillation_tolerance_mps = float(
        _REWARD.get("command_speed_oscillation_tolerance_mps", 0.08)
    )
    command_speed_oscillation_scale_mps = float(
        _REWARD.get("command_speed_oscillation_scale_mps", 0.20)
    )
    command_speed_oscillation_penalty_weight = float(
        _REWARD.get("command_speed_oscillation_penalty_weight", 2.00)
    )
    command_pitch_excess_tolerance_deg = float(
        _REWARD.get("command_pitch_excess_tolerance_deg", 6.0)
    )
    command_pitch_excess_scale_deg = float(
        _REWARD.get("command_pitch_excess_scale_deg", 8.0)
    )
    command_pitch_excess_penalty_weight = float(
        _REWARD.get("command_pitch_excess_penalty_weight", 0.80)
    )
    command_pitch_rate_tolerance_rad_s = float(
        _REWARD.get("command_pitch_rate_tolerance_rad_s", 1.50)
    )
    command_pitch_rate_scale_rad_s = float(
        _REWARD.get("command_pitch_rate_scale_rad_s", 1.50)
    )
    command_pitch_rate_penalty_weight = float(
        _REWARD.get("command_pitch_rate_penalty_weight", 0.25)
    )
    command_pitch_tracking_start_level = int(
        _REWARD.get("command_pitch_tracking_start_level", 14)
    )
    command_pitch_target_deg_per_mps = float(
        _REWARD.get("command_pitch_target_deg_per_mps", 1.50)
    )
    command_pitch_target_underspeed_gain_deg_per_mps = float(
        _REWARD.get("command_pitch_target_underspeed_gain_deg_per_mps", 1.50)
    )
    command_pitch_target_max_deg = float(
        _REWARD.get("command_pitch_target_max_deg", 5.00)
    )
    command_pitch_tracking_tolerance_deg = float(
        _REWARD.get("command_pitch_tracking_tolerance_deg", 2.50)
    )
    command_pitch_tracking_scale_deg = float(
        _REWARD.get("command_pitch_tracking_scale_deg", 5.00)
    )
    command_pitch_tracking_penalty_weight = float(
        _REWARD.get("command_pitch_tracking_penalty_weight", 1.25)
    )
    command_high_speed_pitch_rate_tolerance_rad_s = float(
        _REWARD.get("command_high_speed_pitch_rate_tolerance_rad_s", 0.50)
    )
    command_high_speed_pitch_rate_scale_rad_s = float(
        _REWARD.get("command_high_speed_pitch_rate_scale_rad_s", 1.00)
    )
    command_high_speed_pitch_rate_penalty_weight = float(
        _REWARD.get("command_high_speed_pitch_rate_penalty_weight", 0.50)
    )
    command_turn_pitch_rate_start_level = int(
        _REWARD.get("command_turn_pitch_rate_start_level", 28)
    )
    command_turn_pitch_rate_tolerance_rad_s = float(
        _REWARD.get("command_turn_pitch_rate_tolerance_rad_s", 0.08)
    )
    command_turn_pitch_rate_scale_rad_s = float(
        _REWARD.get("command_turn_pitch_rate_scale_rad_s", 0.16)
    )
    command_turn_pitch_rate_penalty_weight = float(
        _REWARD.get("command_turn_pitch_rate_penalty_weight", 0.50)
    )
    if command_speed_delta_filter_tau_s <= 0.0:
        raise ValueError("reward.command_speed_delta_filter_tau_s must be positive")
    if command_speed_delta_scale_mps <= 0.0:
        raise ValueError("reward.command_speed_delta_scale_mps must be positive")
    if command_speed_oscillation_start_level < 0:
        raise ValueError("reward.command_speed_oscillation_start_level must be >= 0")
    if command_speed_oscillation_tolerance_mps < 0.0:
        raise ValueError("reward.command_speed_oscillation_tolerance_mps must be >= 0")
    if command_speed_oscillation_scale_mps <= 0.0:
        raise ValueError("reward.command_speed_oscillation_scale_mps must be positive")
    if command_speed_oscillation_penalty_weight < 0.0:
        raise ValueError("reward.command_speed_oscillation_penalty_weight must be >= 0")
    if command_pitch_tracking_start_level < 0:
        raise ValueError("reward.command_pitch_tracking_start_level must be >= 0")
    if command_pitch_target_max_deg < 0.0:
        raise ValueError("reward.command_pitch_target_max_deg must be >= 0")
    if command_pitch_tracking_tolerance_deg < 0.0:
        raise ValueError("reward.command_pitch_tracking_tolerance_deg must be >= 0")
    command_large_turn_pitch_tolerance_deg = float(_REWARD.get(
        "command_large_turn_pitch_tolerance_deg", command_pitch_tracking_tolerance_deg))
    command_large_turn_pitch_scale_deg = float(_REWARD.get(
        "command_large_turn_pitch_scale_deg", command_pitch_tracking_scale_deg))
    if not 0 <= command_large_turn_pitch_tolerance_deg < float('inf'):
        raise ValueError("Large-turn pitch tolerance must be finite and nonnegative")
    if not 0 < command_large_turn_pitch_scale_deg < float('inf'):
        raise ValueError("Large-turn pitch scale must be finite and positive")
    if command_pitch_tracking_scale_deg <= 0.0:
        raise ValueError("reward.command_pitch_tracking_scale_deg must be positive")
    if command_high_speed_pitch_rate_tolerance_rad_s < 0.0:
        raise ValueError("reward.command_high_speed_pitch_rate_tolerance_rad_s must be >= 0")
    if command_high_speed_pitch_rate_scale_rad_s <= 0.0:
        raise ValueError("reward.command_high_speed_pitch_rate_scale_rad_s must be positive")
    if command_turn_pitch_rate_start_level < 0:
        raise ValueError("reward.command_turn_pitch_rate_start_level must be >= 0")
    if command_turn_pitch_rate_tolerance_rad_s < 0.0:
        raise ValueError("reward.command_turn_pitch_rate_tolerance_rad_s must be >= 0")
    if command_turn_pitch_rate_scale_rad_s <= 0.0:
        raise ValueError("reward.command_turn_pitch_rate_scale_rad_s must be positive")
    if command_turn_pitch_rate_penalty_weight < 0.0:
        raise ValueError("reward.command_turn_pitch_rate_penalty_weight must be >= 0")
    if command_turn_yaw_tracking_start_level < 0:
        raise ValueError("reward.command_turn_yaw_tracking_start_level must be >= 0")
    if command_turn_yaw_tracking_penalty_weight < 0.0:
        raise ValueError("reward.command_turn_yaw_tracking_penalty_weight must be >= 0")

    effort_limit_nm = float(_ROBOT.get("actuators", {}).get("leg", {}).get("effort_limit_nm", 40.0))
    torque_limit_ratio_sim = float(_TORQUE.get("limit_ratio_sim", 1.0))
    torque_limit_ratio_real = float(_TORQUE.get("limit_ratio_real", 0.8))
    current_limit_A = float(_TORQUE.get("current_limit_A", 60.0))
    estop_tilt_deg_sim = float(_ESTOP.get("tilt_deg", {}).get("sim", 85.0))
    estop_tilt_deg_real = float(_ESTOP.get("tilt_deg", {}).get("real", 30.0))
    max_height_m = float(_EXPLOSION.get("max_height_m", 5.0))
    min_root_height_m = float(_EXPLOSION.get("min_root_height_m", 0.08))
    max_velocity_mps = float(_EXPLOSION.get("max_velocity_mps", 10.0))
    max_xy_distance_m = float(_EXPLOSION.get("max_xy_distance_m", 40.0))

    upright_weight = float(_REWARD.get("upright_weight", 2.0))
    velocity_weight = float(_REWARD.get("velocity_weight", 0.5))
    yaw_weight = float(_REWARD.get("yaw_weight", 0.25))
    stillness_weight = float(_REWARD.get("stillness_weight", 0.5))
    action_rate_weight = float(_REWARD.get("action_rate_weight", 0.01))
    energy_weight = float(_REWARD.get("energy_weight", 0.002))
    tilt_weight = float(_REWARD.get("tilt_weight", 1.0))
    stand_height_weight = float(_REWARD.get("stand_height_weight", 0.75))
    stand_leg_extension_weight = float(_REWARD.get("stand_leg_extension_weight", 0.75))
    stand_symmetry_weight = float(_REWARD.get("stand_symmetry_weight", 0.50))
    stand_attitude_weight = float(_REWARD.get("stand_attitude_weight", 0.50))
    stand_contact_weight = float(_REWARD.get("stand_contact_weight", 0.75))
    stand_leg_length_weight = float(_REWARD.get("stand_leg_length_weight", 0.75))
    stand_geometry_symmetry_weight = float(_REWARD.get("stand_geometry_symmetry_weight", 0.75))
    stand_no_contact_penalty = float(_REWARD.get("stand_no_contact_penalty", 0.75))
    velocity_tolerance_mps = float(_REWARD.get("velocity_tolerance_mps", 0.4))
    command_velocity_tolerance_mps = float(_REWARD.get("command_velocity_tolerance_mps", 0.12))
    yaw_tolerance_rad_s = float(_REWARD.get("yaw_tolerance_rad_s", 0.5))
    command_yaw_tolerance_rad_s = float(_REWARD.get("command_yaw_tolerance_rad_s", 0.25))
    asset_frame_rpy_deg = tuple(float(value) for value in _ROBOT.get("asset_frame_rpy_deg", (90.0, 0.0, 0.0)))


@configclass
class RmucWheelLegRmucEnvCfg(RmucWheelLegCourseEnvCfg):
    scene = RmucSceneCfg(num_envs=1, env_spacing=float(_ENV.get("env_spacing_m", 3.0)), replicate_physics=True)


@configclass
class RmucWheelLegCommandEnvCfg(RmucWheelLegCourseEnvCfg):
    """Stage 1: flat terrain with forward/yaw commands and no perturbations."""

    curriculum_stage = 1
    curriculum_stage_name = "command"
    command_enabled = True
    command_linear_x_mps = float(_CURRICULUM.get("command", {}).get("linear_x_mps", 0.4))
    command_yaw_rate_rad_s = float(_CURRICULUM.get("command", {}).get("yaw_rate_rad_s", 0.15))
    curriculum_max_tilt_deg = float(
        _CURRICULUM.get("gate", {}).get("command", {}).get("max_mean_tilt_deg", 15.0)
    )


@configclass
class RmucWheelLegRoughEnvCfg(RmucWheelLegCommandEnvCfg):
    """Stage 2: rough terrain plus bounded robustness randomization."""

    curriculum_stage = 2
    curriculum_stage_name = "rough_robust"
    terrain = _rough_terrain_cfg()
    rough_terrain_enabled = True
    friction_randomization_enabled = True
    push_enabled = True
    observation_delay_steps = int(_CURRICULUM.get("rough_robust", {}).get("observation_delay_steps", 2))
    push_interval_steps = int(_CURRICULUM.get("rough_robust", {}).get("push_interval_steps", 250))
    push_velocity_xy_mps = float(_CURRICULUM.get("rough_robust", {}).get("push_velocity_xy_mps", 0.5))
    curriculum_max_tilt_deg = float(
        _CURRICULUM.get("gate", {}).get("rough_robust", {}).get("max_mean_tilt_deg", 20.0)
    )
    friction_static_range = tuple(
        float(value) for value in _CURRICULUM.get("rough_robust", {}).get("friction_static_range", (0.4, 1.0))
    )
    friction_dynamic_range = tuple(
        float(value) for value in _CURRICULUM.get("rough_robust", {}).get("friction_dynamic_range", (0.3, 0.8))
    )


@configclass
class RmucWheelLegRoughPlayEnvCfg(RmucWheelLegRoughEnvCfg):
    scene = RmucSceneCfg(num_envs=1, env_spacing=float(_ENV.get("env_spacing_m", 3.0)), replicate_physics=True)
