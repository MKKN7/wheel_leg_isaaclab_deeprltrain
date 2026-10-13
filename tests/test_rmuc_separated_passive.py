"""CPU contracts and joint-space reaction; no behavior acceptance claims."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sim2real_RL/sim2sim'))
from rmuc_controller import compute_effort
from wheel_passive import (validate_wheel_contract, passive_components,
                           feedback_damping_torque, feedback_damping_is_active_motor,
                           configure_passive_runtime, apply_passive)


def contract():
    return dict(wheel_torque_contract='separated_passive_v1',
                wheel_feedback_damping_nm_s_rad=0., wheel_mechanical_viscous_nm_s_rad=.002,
                wheel_coulomb_friction_nm=.001, wheel_coulomb_omega_eps_rad_s=.1,
                hardware_sign_verified=False, leg_pd_dt=.002, leg_scale_rad=.22, wheel_peak_nm=5.5,
                wheel_action_scale_nm=5.5, wheel_action_sign=[-1., -1.])


def test_motor_passive_and_no_outer_clip():
    c = contract()
    action = np.array([1., 1., -2., 1., 1., 2.])
    vel = np.array([0., 0., -100., 0., 0., 100.])
    motor, targets, _ = compute_effort(c, action, np.zeros(6), vel, np.zeros(6),
                                     np.tile([-3., 3.], (6, 1)))
    np.testing.assert_allclose(motor[[2, 5]], [5.5, -5.5])
    np.testing.assert_allclose(motor[[0, 1, 3, 4]], 13.2)
    viscous, coulomb = passive_components(c, vel[[2, 5]])
    np.testing.assert_allclose(viscous, [.2, -.2])
    np.testing.assert_allclose(coulomb, [.001, -.001])
    np.testing.assert_allclose(motor[[2, 5]] + viscous + coulomb, [5.701, -5.701])
    assert np.all((viscous+coulomb) * vel[[2, 5]] <= 0)
    np.testing.assert_array_equal(passive_components(c, [0., 0.]), np.zeros((2, 2)))


def test_feedback_and_mechanical_losses_are_after_motor_clip_without_outer_clip():
    c = contract()
    c['wheel_feedback_damping_nm_s_rad'] = .5
    vel = np.array([0., 0., 100., 0., 0., -100.])
    action = np.array([0., 0., -1., 0., 0., 1.])
    motor, _, requested = compute_effort(c, action, np.zeros(6), vel, np.zeros(6),
                                np.tile([-3., 3.], (6, 1)))
    np.testing.assert_allclose(requested, [5.5, -5.5])
    np.testing.assert_allclose(motor[[2, 5]], [5.5, -5.5])
    feedback = feedback_damping_torque(c, vel[[2, 5]])
    viscous, coulomb = passive_components(c, vel[[2, 5]])
    np.testing.assert_allclose(feedback, [-50., 50.])
    np.testing.assert_allclose(motor[[2, 5]] + feedback + viscous + coulomb,
                               [-44.701, 44.701])


def test_active_feedback_d_is_inside_the_motor_peak_clip():
    c = contract()
    c.update(wheel_feedback_damping_nm_s_rad=.05, wheel_feedback_damping_mode='active_motor_v1')
    vel = np.array([0., 0., 20., 0., 0., -300.])
    # Exercise action clipping before D, motor clipping after D, and no net clip.
    action = np.array([0., 0., -2., 0., 0., 2.])
    motor, _, requested = compute_effort(c, action, np.zeros(6), vel, np.zeros(6),
                                         np.tile([-3., 3.], (6, 1)))
    assert feedback_damping_is_active_motor(c)
    np.testing.assert_allclose(requested, [4.5, 9.5])
    np.testing.assert_allclose(motor[[2, 5]], [4.5, 5.5])
    viscous, coulomb = passive_components(c, vel[[2, 5]])
    np.testing.assert_allclose(motor[[2, 5]] + viscous + coulomb, [4.459, 6.101])


@pytest.mark.parametrize('update', [dict(wheel_damping=0.), dict(wheel_feedback_damping_mode='bad'),
    dict(wheel_coulomb_omega_eps_rad_s=0.), dict(wheel_feedback_damping_nm_s_rad=float('nan')),
    dict(hardware_sign_verified=True), dict(wheel_torque_contract='unknown')])
def test_reject_ambiguous_or_invalid(update):
    with pytest.raises(ValueError):
        validate_wheel_contract(dict(contract(), **update))


def test_joint_generalized_force_reacts_on_parent_without_world_torque():
    mj = pytest.importorskip('mujoco')
    model = mj.MjModel.from_xml_string('''<mujoco><option timestep="0.001" gravity="0 0 0"/>
    <worldbody><body><joint name="parent" axis="0 0 1"/>
    <geom type="sphere" size=".1" mass="1"/>
    <body><joint name="wheel" axis="0 0 1" damping=".2" frictionloss=".1"/>
    <geom type="sphere" size=".05" mass="1"/></body></body></worldbody></mujoco>''')
    data = mj.MjData(model)
    old = configure_passive_runtime(model, np.array([1]))
    assert old == dict(damping=[.2], frictionloss=[.1])
    assert model.dof_damping[1] == model.dof_frictionloss[1] == 0.
    apply_passive(data, np.array([1]), np.array([-.002]), np.array([-.001]))
    mj.mj_step(model, data)
    assert data.qvel[0] > 0 and data.qvel.sum() < 0
    # The internal joint torque preserves total angular momentum (sphere inertias).
    assert abs(.004 * data.qvel[0] + .001 * data.qvel.sum()) < 1e-12


def test_loader_requires_new_period_and_explicit_fields(tmp_path):
    import hashlib
    import json
    from rmuc_controller import load_contract
    artifact = tmp_path / 'source'
    artifact.write_bytes(b'cpu test artifact')
    c = dict(contract(), observation_dim=25, action_dim=6, policy_dt=.008,
             physics_dt=.001, wheel_continuous_nm=3., leg_kp=60., leg_kd=2.,
             leg_limit_nm=40., outer_feedback=False, radial_force_n=0., wheel_gear=1.,
             gas_spring_torque_nm=1.0775)
    for key in ('checkpoint', 'onnx', 'jit', 'mjcf'):
        c[key] = 'source'
        c[key+'_sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    path = tmp_path / 'contract.json'
    path.write_text(json.dumps(c))
    assert load_contract(path)['physics_dt'] == .001
    c['physics_dt'] = .002
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='physics_dt'):
        load_contract(path)
    c['physics_dt'] = .001
    del c['wheel_feedback_damping_nm_s_rad']
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='wheel_feedback_damping'):
        load_contract(path)


def test_independent_wheel_1ms_leg_2ms_hold():
    from rmuc_controller import LowLevelEffortScheduler
    c = contract()
    c['wheel_feedback_damping_nm_s_rad'] = .1
    scheduler = LowLevelEffortScheduler(c)
    args = (np.ones(6), np.zeros(6))
    zero = np.zeros(6)
    ranges = np.tile([-3., 3.], (6, 1))
    t0, _, _ = scheduler.compute(*args, zero, zero, ranges)
    t1, _, _ = scheduler.compute(*args, -np.ones(6), zero, ranges)
    t2, _, _ = scheduler.compute(*args, -np.ones(6)*2, zero, ranges)
    np.testing.assert_array_equal(t0[[0, 1, 3, 4]], t1[[0, 1, 3, 4]])
    np.testing.assert_allclose(t2[[0, 1, 3, 4]]-t0[[0, 1, 3, 4]], 4.)
    np.testing.assert_array_equal(t1[[2, 5]], t0[[2, 5]])
    np.testing.assert_array_equal(t2[[2, 5]], t1[[2, 5]])
    # D is a 1ms post-clip joint torque and is evaluated from each live wheel speed.
    d0 = feedback_damping_torque(c, zero[[2, 5]])
    d1 = feedback_damping_torque(c, (-np.ones(6))[[2, 5]])
    np.testing.assert_allclose(d1-d0, [.1, .1])
