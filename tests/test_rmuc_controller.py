import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / 'sim2real_RL/sim2sim'
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('rmuc_controller', HERE / 'rmuc_controller.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def contract():
    return dict(leg_scale_rad=.22, wheel_peak_nm=5.5, wheel_action_scale_nm=5.5,
                wheel_action_sign=[-1., -1.], wheel_damping=.080625,
                wheel_continuous_nm=3., observation_contract=dict(
                    schema_version=1, observation_dim=25,
                    command_observation_mode='tracking_error_with_target',
                    command_observation_scale=[2.1, .5, 1.],
                    command_observation_error_clip=2., command_yaw_observation_error_clip=2.))


def test_diagnostic_period_is_explicit_and_leg_pd_remains_two_ms():
    values=dict(wheel_torque_contract='separated_passive_v1',
        wheel_feedback_damping_nm_s_rad=.05,wheel_mechanical_viscous_nm_s_rad=.002,
        wheel_coulomb_friction_nm=.001,wheel_coulomb_omega_eps_rad_s=.1,
        hardware_sign_verified=False,leg_pd_dt=.002)
    assert m.LowLevelEffortScheduler(values).leg_stride==2
    diagnostic=dict(values,diagnostic_physics_dt_s=.002,certified=False)
    assert m.LowLevelEffortScheduler(diagnostic).leg_stride==1
    for changes in [dict(certified=True),dict(diagnostic_physics_dt_s=.004)]:
        with pytest.raises(ValueError):
            m.LowLevelEffortScheduler(dict(diagnostic,**changes))


def test_observation_yaw_clip_is_explicit_and_preserves_legacy_default():
    from rmuc_policy_interface import build_observation
    values = dict(angular_velocity_body=np.zeros(3), gravity_body=[0., 0., -1.],
                  heading_forward_velocity=0., command_vx=0., joint_position=np.zeros(6),
                  joint_velocity=np.zeros(6), default_joint_position=np.zeros(6),
                  previous_executed_action=np.zeros(6))
    old3 = build_observation(command_yaw=3., **values)
    old4 = build_observation(command_yaw=4., **values)
    assert np.array_equal(old3, old4)
    new3 = build_observation(command_yaw=3., yaw_error_clip=6., **values)
    new4 = build_observation(command_yaw=4., yaw_error_clip=6., **values)
    assert new3[8] == 3. and new4[8] == 4.
    np.testing.assert_array_equal(new3[:8], new4[:8])


def test_controller_rejects_export_observation_mismatch(tmp_path):
    artifact = tmp_path / 'artifact'
    artifact.write_bytes(b'fixed')
    values = dict(observation_dim=25, action_dim=6, policy_dt=.008, physics_dt=.002,
                  leg_kp=60., leg_kd=2., leg_limit_nm=40., outer_feedback=False,
                  radial_force_n=0., wheel_gear=1., gas_spring_torque_nm=10.775,
                  **contract())
    for key in ('checkpoint', 'onnx', 'jit', 'mjcf'):
        values[key] = 'artifact'
        values[key+'_sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    export = tmp_path / 'export'
    export.mkdir()
    (export / 'action_contract.json').write_text(json.dumps(dict(observation_contract=dict(
        values['observation_contract'], command_yaw_observation_error_clip=6.))))
    path = tmp_path / 'contract.json'
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match='observation contracts differ'):
        m.load_contract(path)


def test_wheel_peak_is_not_continuous_clip_and_damping_applies_after_target_clip():
    action = np.array([0., 0., -2., 0., 0., 2.])
    velocity = np.array([0., 0., 10., 0., 0., -10.])
    torque, _, _ = m.compute_effort(contract(), action, np.zeros(6), velocity,
                                   np.zeros(6), np.tile([-3., 3.], (6, 1)))
    np.testing.assert_allclose(torque[[2, 5]], [4.69375, -4.69375])
    assert np.all(abs(torque[[2, 5]]) > 3.)


def test_leg_pd_uses_live_physics_state_and_40nm_limit():
    ranges = np.tile([-3., 3.], (6, 1))
    action = np.ones(6)
    a, _, _ = m.compute_effort(contract(), action, np.zeros(6), np.zeros(6), np.zeros(6), ranges)
    b, _, _ = m.compute_effort(contract(), action, np.zeros(6), np.ones(6)*30, np.zeros(6), ranges)
    np.testing.assert_allclose(a[[0, 1, 3, 4]], 13.2)
    np.testing.assert_allclose(b[[0, 1, 3, 4]], -40.)


def test_invalid_inputs_fail_before_actuation():
    with pytest.raises(ValueError):
        m.compute_effort(contract(), np.full(6, np.nan), np.zeros(6), np.zeros(6),
                         np.zeros(6), np.tile([-3., 3.], (6, 1)))


def test_contract_rejects_hidden_derating_and_artifact_changes(tmp_path):
    artifact = tmp_path / 'artifact'
    artifact.write_bytes(b'original')
    values = dict(observation_dim=25, action_dim=6, policy_dt=.008, physics_dt=.002,
                  leg_kp=60., leg_kd=2., leg_limit_nm=40., outer_feedback=False,
                  radial_force_n=0., wheel_gear=1., gas_spring_torque_nm=10.775,
                  **contract())
    for key in ('checkpoint', 'onnx', 'jit', 'mjcf'):
        values[key] = 'artifact'
        values[key+'_sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    path = tmp_path / 'contract.json'
    path.write_text(json.dumps(values))
    assert m.load_contract(path)['gas_spring_torque_nm'] == 10.775
    values['wheel_peak_nm'] = 3.
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match='wheel_peak_nm'):
        m.load_contract(path)
    values['wheel_peak_nm'] = 5.5
    path.write_text(json.dumps(values))
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        m.load_contract(path)


def test_manual_leg_mapping_matches_actual_training_function_and_duplicate_rules():
    import torch
    source = HERE.parents[1] / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/command_contract.py'
    spec = importlib.util.spec_from_file_location('training_command_contract', source)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    params = dict(stand_leg_position_scale_rad=.18,
                  command_stage_linear_x_mps=[1., 2., 1., 3.],
                  command_stage_leg_scale_rad=[.19, .25, .22, .28])
    c = dict(contract(), **m.manual_leg_knots(params))
    assert c['manual_leg_scale_knots'] == [.18, .22, .25, .28]
    speeds = torch.tensor([-5., -2.5, -1., 0., .5, 1., 1.5, 2., 3., 5.])
    expected = training.interpolate_leg_scale(speeds, torch.tensor(c['manual_leg_speed_knots']),
                                             torch.tensor(c['manual_leg_scale_knots']))
    np.testing.assert_array_equal([m.command_leg_scale(c, float(v)) for v in speeds], expected.numpy())
    inputs = (np.ones(6), np.zeros(6), np.zeros(6), np.zeros(6), np.tile([-3., 3.], (6, 1)))
    old = m.compute_effort(contract(), *inputs)
    new = m.compute_effort(c, *inputs, command_vx=1.)
    for a, b in zip(old, new):
        np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-7)
    switched = m.compute_effort(c, *inputs, command_vx=3.)
    np.testing.assert_allclose(switched[1], .28, rtol=1e-7)
    assert m.command_leg_scale(contract(), 3.) == .22


def test_invalid_or_incomplete_manual_knots_fail_closed():
    for update in [dict(manual_leg_speed_knots=[0., 1.]),
                   dict(manual_leg_scale_knots=[.18, .22]),
                   dict(manual_leg_speed_knots=[0., 0.], manual_leg_scale_knots=[.18, .22])]:
        with pytest.raises(ValueError):
            m.command_leg_scale(dict(contract(), **update), 1.)
    c = dict(contract(), manual_leg_speed_knots=[0., 1.], manual_leg_scale_knots=[.18, .22])
    with pytest.raises(ValueError, match='command_vx'):
        m.command_leg_scale(c)


def test_v458_controller_builder_restores_actual_speed_dependent_leg_targets():
    root = HERE.parents[1]
    builder_path = root / 'scripts/build_rmuc_controller_contract.py'
    builder_spec = importlib.util.spec_from_file_location('contract_builder', builder_path)
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    base = root / 'artifacts/continuation_v458'
    built = builder.build_contract(base/'cpu/mujoco_controller_contract.json',
                                   base/'actual_env.yaml', base/'quick_eval.json')
    assert built['controller_contract_version'] == 2
    assert m.command_leg_scale(built, 1.0) == pytest.approx(.22)
    assert m.command_leg_scale(built, 1.5) == pytest.approx(.24)
    assert m.command_leg_scale(built, 2.0) == pytest.approx(.25)
    assert m.command_leg_scale(built, 3.2) == pytest.approx(.286)
    assert m.command_leg_scale(built, 0.) == pytest.approx(.18)
    old = m.compute_effort(contract(), np.ones(6), np.zeros(6), np.zeros(6),
                           np.zeros(6), np.tile([-3., 3.], (6, 1)), command_vx=1.5)
    new = m.compute_effort(built, np.ones(6), np.zeros(6), np.zeros(6),
                           np.zeros(6), np.tile([-3., 3.], (6, 1)), command_vx=1.5)
    np.testing.assert_allclose(new[1], .24)
    assert np.max(np.abs(new[0][[0, 1, 3, 4]] - old[0][[0, 1, 3, 4]])) == pytest.approx(1.2)
