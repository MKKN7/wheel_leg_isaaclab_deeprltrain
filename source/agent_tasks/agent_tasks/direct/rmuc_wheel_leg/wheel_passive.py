"""Wheel-side passive drag, applied as internal body wrenches outside motors.

The tanh transition is a diagnostic regularization, not measured hardware data.
Zero feedback D denotes a no-feedback diagnostic, not identified hardware D.
"""
import math
import torch

POSTCLIP_LOSS_V1 = "postclip_loss_v1"
ACTIVE_MOTOR_V1 = "active_motor_v1"


def feedback_damping_is_active_motor(cfg):
    """Whether feedback D is part of the clipped electromagnetic command."""
    mode = getattr(cfg, "wheel_feedback_damping_mode", POSTCLIP_LOSS_V1)
    if mode not in (POSTCLIP_LOSS_V1, ACTIVE_MOTOR_V1):
        raise ValueError(f"Unknown wheel feedback damping mode: {mode}")
    return mode == ACTIVE_MOTOR_V1


def configure_wheel_torque_contract(cfg):
    """Validate opt-in separation before actuator construction; legacy is a no-op."""
    mode = getattr(cfg, 'wheel_torque_contract', 'legacy_idealpd')
    if mode == 'legacy_idealpd':
        return False
    if mode != 'separated_passive_v1':
        raise ValueError(f'Unknown wheel torque contract: {mode}')
    fields = ('wheel_mechanical_viscous_nm_s_rad', 'wheel_coulomb_friction_nm',
              'wheel_feedback_damping_nm_s_rad', 'wheel_coulomb_omega_eps_rad_s')
    values = [float(getattr(cfg, name)) for name in fields]
    if not all(math.isfinite(v) and v >= 0 for v in values) or values[-1] <= 0:
        raise ValueError('Wheel drag/D must be finite nonnegative and omega epsilon positive')
    if cfg.wheel_damping_anneal_target >= 0 or cfg.wheel_damping_compensation_ratio != 0:
        raise ValueError('Separated wheel passive contract forbids legacy damping anneal/compensation')
    actuator = cfg.robot_cfg.actuators['wheel_motors']
    declared_peak = getattr(cfg, 'wheel_authored_effort_limit_nm', 5.5)
    # Historical configs can leave the USD limit unspecified (infinity).
    # Explicit deployment contracts must also be usable by the training path.
    declared_peak = declared_peak if math.isfinite(declared_peak) else 5.5
    expected_peak = getattr(cfg, 'diagnostic_wheel_peak_nm', declared_peak)
    if not math.isfinite(expected_peak) or not 0 < expected_peak <= 5.5:
        raise ValueError('Diagnostic electromagnetic peak must be finite in (0,5.5]')
    if cfg.wheel_torque_scale_nm != 5.5 or actuator.effort_limit != expected_peak:
        raise ValueError('Separated wheel contract requires action scale 5.5 and actuator peak matching the declared contract')
    simulator_peak = getattr(actuator, 'effort_limit_sim', None)
    if simulator_peak is not None and simulator_peak != expected_peak:
        raise ValueError('Simulator and electromagnetic wheel peaks must match')
    if getattr(cfg, 'leg_pd_update_period_s', 0.002) != 0.002:
        raise ValueError('Separated contract requires leg PD update period 0.002 s')
    feedback_damping_is_active_motor(cfg)
    physics_dt = getattr(cfg, 'separated_wheel_physics_dt_s', .001)
    if physics_dt not in (.001, .002):
        raise ValueError('Separated wheel physics timestep must be 1ms or explicit 2ms diagnostic')
    cfg.gas_spring_quaternion_order = "xyzw"
    cfg.sim.dt = physics_dt
    cfg.decimation = round(.008 / physics_dt)
    # The IdealPD actuator owns only electromagnetic torque. Mechanical losses
    # remain external; feedback D is either an explicitly legacy post-clip loss
    # or an active contribution before this actuator's single peak clip.
    actuator.damping = 0.0
    return True


def mechanical_torque(omega, viscous, coulomb, omega_epsilon):
    """Wheel-side generalized torque; never pass this through a motor actuator."""
    return -viscous * omega - coulomb * torch.tanh(omega / omega_epsilon)


def feedback_damping_torque(omega, damping):
    """Feedback D contribution in wheel output-shaft torque units."""
    return -damping * omega


def sample_separated_wheel_parameters(cfg, count, strength, device):
    """Sample actual post-clip losses, independently for both wheels."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError('Separated wheel DR strength must be in [0,1]')
    result = []
    for nominal_name, range_name in (
        ('wheel_mechanical_viscous_nm_s_rad', 'separated_wheel_mechanical_viscous_range'),
        ('wheel_coulomb_friction_nm', 'separated_wheel_coulomb_range'),
        ('wheel_feedback_damping_nm_s_rad', 'separated_wheel_feedback_damping_range'),
    ):
        nominal = float(getattr(cfg, nominal_name))
        bounds = getattr(cfg, range_name, (nominal, nominal))
        if (len(bounds) != 2 or not all(math.isfinite(float(v)) for v in bounds)
                or not 0 <= float(bounds[0]) <= nominal <= float(bounds[1])):
            raise ValueError(f'Invalid separated wheel DR range: {range_name}')
        low = nominal + strength * (float(bounds[0]) - nominal)
        high = nominal + strength * (float(bounds[1]) - nominal)
        value = torch.full((count, 2), low, device=device)
        if high > low:
            value.uniform_(low, high)
        result.append(value)
    return tuple(result)


def rotate_axis(quaternion_w, local_axis):
    """Rotate body-local axis with Isaac Lab 3 scalar-last (x,y,z,w) quaternion."""
    vector = quaternion_w[..., :3]
    axis = torch.broadcast_to(local_axis, vector.shape)
    cross = 2 * torch.cross(vector, axis, dim=-1)
    return axis + quaternion_w[..., 3:4] * cross + torch.cross(vector, cross, dim=-1)


def body_torque_pair(parent_quaternion_w, parent_axis, torque):
    """[body0 reaction, body1 wheel] torques, in world frame."""
    child = rotate_axis(parent_quaternion_w, parent_axis) * torque.unsqueeze(-1)
    return torch.stack((-child, child), dim=-2)


def read_wheel_joint_geometry(usd_path, joint_names):
    """Read joint axes and connected bodies, refusing ambiguous/reversed topology."""
    from pxr import Usd, UsdPhysics, Gf
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise ValueError('Cannot read wheel joint geometry')
    result = []
    for name in joint_names:
        prims = [p for p in stage.Traverse() if p.GetName() == name]
        if len(prims) != 1:
            raise ValueError(f'Expected one wheel joint: {name}')
        joint = UsdPhysics.RevoluteJoint(prims[0])
        if not joint:
            raise ValueError(f'Wheel joint must be revolute: {name}')
        parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(parents) != 1 or len(children) != 1 or children[0].name != name.removesuffix('_spin'):
            raise ValueError(f'Wheel joint requires body0 parent and body1 named wheel: {name}')
        axis_name = joint.GetAxisAttr().Get()
        if axis_name not in ('X', 'Y', 'Z'):
            raise ValueError(f'Unsupported wheel joint axis: {axis_name}')
        axis = Gf.Vec3d(*(float(i == 'XYZ'.index(axis_name)) for i in range(3)))
        rotation = joint.GetLocalRot0Attr().Get()
        axis = Gf.Rotation(Gf.Quatd(rotation)).TransformDir(axis)
        result.append((parents[0].name, children[0].name, tuple(axis)))
    return result


class ExplicitPDSampleHold:
    """Hold actual PD effort between scheduled computes, not just its target.

    Reset clears only selected environments and preserves the global clock. A
    reset on a hold tick therefore commands zero until the next update tick.
    Install inside any continuous motor-response wrapper so physical response
    continues at physics rate against the held PD command.
    """
    def __init__(self, template, physics_dt, update_period):
        if not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError("PD physics dt must be finite and positive")
        ratio = update_period / physics_dt
        if not math.isfinite(ratio) or physics_dt <= 0 or ratio < 1 or abs(ratio-round(ratio)) > 1e-9:
            raise ValueError('PD period must be a positive integer multiple of physics dt')
        self.physics_dt = physics_dt
        self.update_period = update_period
        self.interval_steps = round(ratio)
        self.physics_calls = 0
        self.update_count = 0
        self.last_was_update = False
        self.held_computed_effort = torch.zeros_like(template)
        self.held_applied_effort = torch.zeros_like(template)

    def reset(self, env_ids):
        self.held_computed_effort[env_ids] = 0
        self.held_applied_effort[env_ids] = 0


def install_explicit_pd_sample_hold(actuator, physics_dt, update_period):
    """Skip the wrapped explicit compute entirely on intermediate ticks."""
    if getattr(actuator, 'is_implicit_model', True):
        raise ValueError('PD sample/hold requires explicit actuator, not PhysX implicit PD')
    if hasattr(actuator, '_rmuc_pd_sample_hold') or hasattr(actuator, '_rmuc_effort_response'):
        raise ValueError('Install PD sample/hold once, before motor response')
    hold = ExplicitPDSampleHold(actuator.damping, physics_dt, update_period)
    original_compute, original_reset = actuator.compute, actuator.reset
    def compute(control_action, joint_pos, joint_vel):
        hold.last_was_update = hold.physics_calls % hold.interval_steps == 0
        if hold.last_was_update:
            result = original_compute(control_action, joint_pos, joint_vel)
            hold.held_computed_effort.copy_(actuator.computed_effort)
            hold.held_applied_effort.copy_(result.joint_efforts)
            hold.update_count += 1
        # Independent snapshots prevent an outer motor-response wrapper from
        # aliasing and changing the held pre-response PD command.
        actuator.computed_effort = hold.held_computed_effort.clone()
        actuator.applied_effort = hold.held_applied_effort.clone()
        control_action.joint_efforts = actuator.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        hold.physics_calls += 1
        return control_action
    def reset(env_ids):
        original_reset(env_ids)
        hold.reset(env_ids)
    actuator.compute, actuator.reset = compute, reset
    actuator._rmuc_pd_sample_hold = hold
    return hold
