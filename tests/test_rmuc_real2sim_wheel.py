"""CPU checks for identified wheel dynamics without requiring an Isaac launch."""
import importlib.util
from pathlib import Path
import json

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "real2sim_wheel", ROOT / "source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/real2sim_wheel.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def profile(mode="none"):
    return dict(
        schema_version=1, scope="real2sim_diagnostic", physical_order=["Left", "Right"],
        wheel_joint_order=["right_wheel_spin", "left_wheel_spin"],
        mechanical_viscous_nm_s_rad=[.0025485, .00066957],
        coulomb_friction_nm=[.054699, .052843], compensation_enabled=True,
        damping_compensation_nm_s_rad=[.0494515, .05133043],
        coulomb_compensation_nm=[.053699, .051843],
        friction_epsilon_rad_s=.1, delay_s=[.011, .007], delay_mode=mode,
        identification_status="reported_fit_not_independently_validated",
        equivalent_inertia_kg_m2_metadata=[.0042075, .004204],
    )


def runtime(mode="none", count=2, dt=.002):
    return m.Real2SimWheel(torch.zeros(count, 2, dtype=torch.float64), dt, profile(mode))


def test_unsaturated_compensation_recovers_target_law_and_side_mapping():
    model = runtime(count=7)
    omega = torch.tensor([-30., -2., -.03, 0., .03, 2., 30.], dtype=torch.float64)[:, None].repeat(1, 2)
    mechanical = -model.mechanical_viscous * omega - model.coulomb * torch.tanh(omega / .1)
    motor = model.feedback(omega)
    assert motor.abs().max() < 5.5
    expected = -.052 * omega - .001 * torch.tanh(omega / .1)
    torch.testing.assert_close(motor + mechanical, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(motor[0], -motor[-1])
    assert model.mechanical_viscous[0, 0] == .0025485
    assert model.mechanical_viscous[0, 1] == .00066957
    assert model.readback()["inertia_applied"] is False


def test_saturation_gap_is_retained_not_erased_by_outer_clip():
    model = runtime(count=1)
    omega = torch.tensor([[-30., -30.]], dtype=torch.float64)
    motor = model.motor_effort((5.5 + model.feedback(omega)).clamp(-5.5, 5.5))
    mechanical = -model.mechanical_viscous * omega - model.coulomb * torch.tanh(omega / .1)
    total = motor + mechanical
    torch.testing.assert_close(total, torch.tensor([[5.631154, 5.5729301]], dtype=torch.float64))
    old = torch.tensor([[5.561, 5.561]], dtype=torch.float64)
    assert (total > old).all() and (total > 5.5).all()
    assert model.readback()["motor_peak_hit_fraction_by_side"] == [1., 1.]


def test_fractional_motor_delay_impulse_exact_two_taps_and_side_order():
    model = runtime("motor_transport", count=1)
    outputs = []
    for step in range(9):
        x = torch.ones(1, 2, dtype=torch.float64) if step == 0 else torch.zeros(1, 2, dtype=torch.float64)
        outputs.append(model.motor_effort(x).clone())
    output = torch.cat(outputs)
    expected = torch.zeros(9, 2, dtype=torch.float64)
    expected[5:7, 0] = .5  # Left 11 ms -> physics samples at 10 and 12 ms.
    expected[3:5, 1] = .5  # Right 7 ms -> samples at 6 and 8 ms.
    torch.testing.assert_close(output, expected)
    metadata = model.readback()
    assert metadata["delay_floor_steps"] == [5, 3]
    assert metadata["delay_interpolation_fraction"] == [.5, .5]
    assert metadata["delay_discretization"] == "causal_two_tap_fractional_approximation"


def test_one_ms_delay_is_integer_and_reset_does_not_leak_other_environment():
    model = runtime("motor_transport", count=2, dt=.001)
    for _ in range(15):
        model.motor_effort(torch.ones(2, 2, dtype=torch.float64))
    model.reset(torch.tensor([0]))
    result = model.motor_effort(torch.ones(2, 2, dtype=torch.float64))
    torch.testing.assert_close(result[0], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(result[1], torch.ones(2, dtype=torch.float64))
    assert model.readback()["delay_floor_steps"] == [11, 7]
    assert model.readback()["delay_discretization"] == "integer_physics_steps"
    model.reset(None)
    torch.testing.assert_close(model._delay.history, torch.zeros_like(model._delay.history))


def test_feedback_hypothesis_delays_only_lowlevel_feedback_not_motor_transport():
    model = runtime("feedback_sensor", count=1)
    omega = torch.full((1, 2), 30., dtype=torch.float64)
    torch.testing.assert_close(model.feedback(omega), torch.zeros_like(omega))
    effort = torch.tensor([[2., -3.]], dtype=torch.float64)
    assert model.motor_effort(effort) is effort
    for _ in range(8):
        model.feedback(omega)
    torch.testing.assert_close(model.last_feedback_omega, omega)
    assert "policy_observation_unchanged" in model.readback()["delay_hypothesis_scope"]


def test_none_bypasses_delay_and_disabled_compensation_is_zero():
    p = profile()
    p["compensation_enabled"] = False
    p["damping_compensation_nm_s_rad"] = [0., 0.]
    p["coulomb_compensation_nm"] = [0., 0.]
    model = m.Real2SimWheel(torch.zeros(2, 2), .002, p)
    omega = torch.tensor([[30., -50.], [0., 1.]])
    assert torch.equal(model.feedback(omega), torch.zeros_like(omega))
    effort = torch.tensor([[5.5, -5.5], [.01, -.02]])
    assert model.motor_effort(effort) is effort
    assert model.readback()["effective_delay_s"] == [0., 0.]


@pytest.mark.parametrize("key,value", [
    ("physical_order", ["Right", "Left"]),
    ("wheel_joint_order", ["left_wheel_spin", "right_wheel_spin"]),
    ("schema_version", True), ("scope", "deployment_certified"),
    ("identification_status", "verified"), ("compensation_enabled", 1),
    ("mechanical_viscous_nm_s_rad", [-.1, .01]),
    ("coulomb_friction_nm", [float("nan"), .01]),
    ("damping_compensation_nm_s_rad", [True, .05]),
    ("delay_s", [.013, .007]), ("delay_s", [.011]),
    ("delay_mode", "both"), ("friction_epsilon_rad_s", 0),
    ("equivalent_inertia_kg_m2_metadata", [0., .004]),
    ("armature_override_kg_m2", [-.001, .0035]),
    ("armature_override_kg_m2", [.0035]),
    ("armature_override_kg_m2", [.0035, float("nan")]),
    ("armature_override_kg_m2", [.0035, float("inf")]),
    ("armature_override_kg_m2", [True, .0035]),
    ("unexpected", 0), ("notes", "not a list"),
])
def test_invalid_profile_fails_closed(key, value):
    p = profile()
    p[key] = value
    with pytest.raises(ValueError):
        m.validate_profile(p)


def test_fixed_compensation_does_not_oracle_cancel_changed_friction():
    model = runtime(count=1)
    omega = torch.tensor([[10., -10.]], dtype=torch.float64)
    motor = model.feedback(omega)
    b_real = 1.2 * model.mechanical_viscous
    c_real = 1.2 * model.coulomb
    total = motor - b_real * omega - c_real * torch.tanh(omega / .1)
    nominal = -.052 * omega - .001 * torch.tanh(omega / .1)
    residual = -.2 * (model.mechanical_viscous * omega + model.coulomb * torch.tanh(omega / .1))
    torch.testing.assert_close(total - nominal, residual)


def test_load_profile_and_defensive_copy(tmp_path):
    p = profile()
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(p))
    assert m.load_profile(path) == p
    model = m.Real2SimWheel(torch.zeros(2, 2), .002, p)
    p["mechanical_viscous_nm_s_rad"][0] = 99
    readback = model.readback()
    assert readback["profile"]["mechanical_viscous_nm_s_rad"][0] == .0025485
    readback["profile"]["mechanical_viscous_nm_s_rad"][0] = 42
    assert model.readback()["profile"]["mechanical_viscous_nm_s_rad"][0] == .0025485


def test_armature_override_is_optional_metadata_not_an_engine_mutation():
    p = profile()
    model = m.Real2SimWheel(torch.zeros(2, 2), .002, p)
    assert model.readback()["armature_override_requested_kg_m2"] is None
    assert model.readback()["armature_override_interpretation"] == "not_requested"
    p["armature_override_kg_m2"] = [.003539909281, .003536409377]
    model = m.Real2SimWheel(torch.zeros(2, 2), .002, p)
    assert model.readback()["armature_override_requested_kg_m2"] == [.003539909281, .003536409377]
    assert model.readback()["inertia_applied"] is False
    assert model.readback()["armature_override_interpretation"] == "nominal_joint_equivalent_approximation_not_exact_fixed_parent_identification"
    # Zero is physically permitted for an explicit actuator armature override.
    p["armature_override_kg_m2"] = [0., 0.]
    assert m.validate_profile(p)["armature_override_kg_m2"] == [0., 0.]


def test_twelve_ms_delay_ceiling_is_inclusive_and_not_per_stage_permission():
    p = profile("motor_transport")
    p["delay_s"] = [.012, .012]
    model = m.Real2SimWheel(torch.zeros(2, 2), .002, p)
    assert model.readback()["configured_delay_cap_s"] == .012
    assert model.readback()["delay_floor_steps"] == [6, 6]
    assert "total_observation_to_execution_pipeline_requires_separate_verification" in model.readback()["delay_cap_scope"]
    p["delay_s"][0] = .012000001
    with pytest.raises(ValueError):
        m.validate_profile(p)


def test_total_delay_disables_duplicate_buffers_without_erasing_noise():
    from types import SimpleNamespace
    original={'enabled':True,'gyro_noise_std_rad_s':.005,'observation_delay_probability':.05,
              'communication_delay_probability':.02,'action_delay_enabled':True}
    cfg=SimpleNamespace(observation_delay_steps=2,domain_randomization_max_observation_delay_steps=3,
                        domain_randomization_max_observation_delay_probability=.2,
                        domain_randomization_sensor_model=original)
    m.configure_total_delay(cfg,profile('motor_transport'))
    assert cfg.observation_delay_steps==cfg.domain_randomization_max_observation_delay_steps==0
    assert cfg.domain_randomization_max_observation_delay_probability==0
    assert cfg.domain_randomization_sensor_model['gyro_noise_std_rad_s']==.005
    assert cfg.domain_randomization_sensor_model['observation_delay_probability']==0
    assert cfg.domain_randomization_sensor_model['communication_delay_probability']==0
    assert cfg.domain_randomization_sensor_model['action_delay_enabled'] is False
    assert original['action_delay_enabled'] is True
    with pytest.raises(ValueError):m.configure_total_delay(cfg,profile('feedback_sensor'))


def test_mixed_transport_is_per_env_and_reset_does_not_change_global_rng():
    p = profile('motor_transport')
    p.update(delay_s=[.002,.002], transport_delay_probability=.5, transport_delay_seed=491)
    before = torch.random.get_rng_state().clone()
    model = m.Real2SimWheel(torch.zeros(64, 2), .002, p)
    assert torch.equal(before, torch.random.get_rng_state())
    delayed = model._transport_active.clone()
    assert delayed.any() and (~delayed).any()
    effort = torch.ones(64,2)
    torch.testing.assert_close(model.motor_effort(effort), (~delayed)[:,None].expand(64,2).float())
    torch.testing.assert_close(model.motor_effort(effort*2), torch.where(delayed[:,None], effort, effort*2))
    untouched = model._transport_active[1:].clone()
    model.reset(torch.tensor([0]))
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(model._transport_active[1:],untouched)
    out = model.motor_effort(effort*3)
    torch.testing.assert_close(out[1:],torch.where(delayed[1:,None],effort[1:]*2,effort[1:]*3))
    expected_first = torch.zeros(2) if model._transport_active[0] else torch.ones(2)*3
    torch.testing.assert_close(out[0],expected_first)
    rb=model.readback()
    assert set(v for row in rb['effective_delay_s_per_env'] for v in row) <= {0., .0020000000949949026}
    assert all(v==3 for v in rb['transport_total_calls_per_env'])


def test_mixed_transport_zero_and_one_endpoints_match_legacy():
    for probability in [0.,1.]:
        p=profile('motor_transport');p.update(delay_s=[.002,.002],transport_delay_probability=probability)
        mixed=m.Real2SimWheel(torch.zeros(3,2),.002,p)
        fixed=profile('motor_transport' if probability else 'none');fixed['delay_s']=[.002,.002]
        legacy=m.Real2SimWheel(torch.zeros(3,2),.002,fixed)
        for i in range(5):
            a=torch.full((3,2),float(i-2))
            assert torch.equal(mixed.motor_effort(a),legacy.motor_effort(a))


@pytest.mark.parametrize('field,value',[('transport_delay_probability',-.1),('transport_delay_probability',1.1),('transport_delay_probability',True),('transport_delay_probability',float('nan')),('transport_delay_seed',True),('transport_delay_seed',-1)])
def test_invalid_transport_exposure_rejected(field,value):
    p=profile('motor_transport');p[field]=value
    with pytest.raises(ValueError):m.validate_profile(p)


def test_peak_statistics_use_actual_limits_without_changing_torque():
    model=runtime('none',count=1)
    effort=torch.tensor([[4.89,-4.89]],dtype=torch.float64)
    limit=torch.full_like(effort,4.89)
    output=model.motor_effort(effort,limit)
    assert output is effort
    rb=model.readback()
    assert rb['motor_peak_hit_fraction_by_side']==[1.,1.]
    assert rb['motor_peak_statistic_effort_limit_range_nm']==[4.89,4.89]
    assert rb['motor_peak_statistic_is_actual_clipping_count'] is False
