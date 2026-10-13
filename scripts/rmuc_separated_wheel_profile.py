"""Pure configuration adapter for explicitly uncalibrated wheel diagnostics."""
import json
import math
from pathlib import Path

FIELDS = ('wheel_mechanical_viscous_nm_s_rad', 'wheel_coulomb_friction_nm',
          'wheel_feedback_damping_nm_s_rad', 'wheel_coulomb_omega_eps_rad_s')


def validate_profile(profile):
    required = {'schema_version', 'scope', 'wheel_torque_contract',
                'hardware_sign_verified', 'hardware_feedback_identified', *FIELDS}
    if isinstance(profile, dict) and profile.get('schema_version') == 2:
        required.add('wheel_feedback_damping_mode')
    if not isinstance(profile, dict) or set(profile) != required:
        raise ValueError('Separated profile requires exactly the declared schema diagnostic fields')
    if type(profile['schema_version']) is not int or profile['schema_version'] not in (1, 2):
        raise ValueError('Separated profile requires schema_version=1 or 2')
    if profile.get('wheel_feedback_damping_mode', 'postclip_loss_v1') not in ('postclip_loss_v1', 'active_motor_v1'):
        raise ValueError('Invalid wheel_feedback_damping_mode')
    if profile['scope'] != 'diagnostic' or profile['wheel_torque_contract'] != 'separated_passive_v1':
        raise ValueError('Separated profile requires diagnostic scope and separated_passive_v1')
    if profile['hardware_sign_verified'] is not False or profile['hardware_feedback_identified'] is not False:
        raise ValueError('Hardware sign and feedback must remain explicitly unverified/unidentified')
    for name in FIELDS:
        value = profile[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be explicit finite nonnegative numeric SI value')
    if profile['wheel_coulomb_omega_eps_rad_s'] <= 0:
        raise ValueError('Coulomb smoothing epsilon must be positive, not a measurement')
    return dict(profile)


def load_profile(path):
    path = Path(path)
    if path.suffix.lower() == '.json':
        profile = json.loads(path.read_text())
    elif path.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        profile = yaml.safe_load(path.read_text())
    else:
        raise ValueError('Separated profile must be JSON or YAML')
    return validate_profile(profile)


def validate_options(args):
    """Run before AppLauncher; legacy callers are not changed."""
    if getattr(args, 'diagnostic_separated_wheel_profile', None) is None:
        return
    for name in ('diagnostic_hardware_profile', 'diagnostic_wheel_velocity_profile',
                 'diagnostic_wheel_velocity_equivalence'):
        if getattr(args, name, None):
            raise ValueError('Separated wheel profile cannot combine with legacy hardware/velocity profiles')
    if not args.diagnostic_disable_outer_feedback or not args.position_authority_consumer:
        raise ValueError('Separated wheel profile requires --diagnostic-disable-outer-feedback and --position-authority-consumer')


def readback(cfg):
    actuator = cfg.robot_cfg.actuators['wheel_motors']
    return dict(**{name: getattr(cfg, name) for name in FIELDS},
                separated_wheel_mechanical_viscous_range=getattr(cfg, 'separated_wheel_mechanical_viscous_range', None),
                separated_wheel_coulomb_range=getattr(cfg, 'separated_wheel_coulomb_range', None),
                separated_wheel_feedback_damping_range=getattr(cfg, 'separated_wheel_feedback_damping_range', None),
                wheel_torque_contract=cfg.wheel_torque_contract,
                wheel_feedback_damping_mode=getattr(cfg, 'wheel_feedback_damping_mode', 'postclip_loss_v1'),
                gas_spring_quaternion_order=getattr(cfg,"gas_spring_quaternion_order","wxyz"),
                physics_dt=cfg.sim.dt, decimation=cfg.decimation,
                policy_dt=cfg.sim.dt * cfg.decimation,
                leg_pd_update_period_s=cfg.leg_pd_update_period_s,
                wheel_action_scale_nm=cfg.wheel_torque_scale_nm,
                wheel_peak_nm=actuator.effort_limit,
                wheel_motor_actuator_pd_damping=actuator.damping,
                wheel_feedback_applied_after_motor_clip=(
                    getattr(cfg, 'wheel_feedback_damping_mode', 'postclip_loss_v1') == 'postclip_loss_v1'
                ),
                wheel_damping_anneal_target=cfg.wheel_damping_anneal_target,
                wheel_damping_compensation_ratio=cfg.wheel_damping_compensation_ratio,
                policy_outer_feedback_enabled=cfg.policy_outer_feedback_enabled,
                position_authority_enabled=cfg.position_authority_enabled)


def apply_separated_wheel_profile(cfg, profile, *, physics_dt=.001):
    profile = validate_profile(profile)
    if physics_dt not in (.001, .002):
        raise ValueError('Diagnostic physics timestep must be 1ms or 2ms')
    actuator = cfg.robot_cfg.actuators['wheel_motors']
    if cfg.wheel_torque_scale_nm != 5.5 or actuator.effort_limit != 5.5:
        raise ValueError('Separated wheel diagnostic requires existing peak5.5Nm and scale5.5Nm')
    for name in (*FIELDS, 'wheel_torque_contract'):
        setattr(cfg, name, profile[name])
    cfg.wheel_feedback_damping_mode = profile.get('wheel_feedback_damping_mode', 'postclip_loss_v1')
    for value_name, range_name in (
        ('wheel_mechanical_viscous_nm_s_rad', 'separated_wheel_mechanical_viscous_range'),
        ('wheel_coulomb_friction_nm', 'separated_wheel_coulomb_range'),
        ('wheel_feedback_damping_nm_s_rad', 'separated_wheel_feedback_damping_range'),
    ):
        value = profile[value_name]
        bounds = getattr(cfg, range_name, (value, value))
        if not bounds[0] <= value <= bounds[1]:
            setattr(cfg, range_name, (value, value))
    cfg.gas_spring_quaternion_order = "xyzw"
    cfg.separated_wheel_physics_dt_s = physics_dt
    cfg.sim.dt = physics_dt
    cfg.decimation = round(.008 / physics_dt)
    cfg.leg_pd_update_period_s = .002
    cfg.wheel_damping_anneal_target = -1.
    cfg.wheel_damping_compensation_ratio = 0.
    cfg.policy_outer_feedback_enabled = False
    cfg.position_authority_enabled = True
    actuator.damping = 0.0
    return dict(profile=profile, readback=readback(cfg), behavior_certified=False,
                parameter_status='Mechanical prior and smoothing diagnostic; feedback and hardware sign unidentified')
