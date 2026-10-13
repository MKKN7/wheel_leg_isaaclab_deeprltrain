import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from rmuc_joint_dr_profile import apply_joint_dr_profile, validate_joint_dr_profile


def profile():
    sensor = yaml.safe_load((ROOT/'configs/rmuc_wheel_leg.yaml').read_text())['domain_randomization']['sensor_model']
    return {'overrides': dict(
        domain_randomization_max_base_mass_fraction=.05,
        domain_randomization_max_base_inertia_fraction=.05,
        domain_randomization_max_friction_fraction=.15,
        domain_randomization_max_wheel_torque_gain_fraction=.05,
        domain_randomization_max_observation_delay_probability=0.,
        motor_response_enabled=True, leg_actuator_dr_enabled=True,
        motor_response_time_constant_min_s=.002, motor_response_time_constant_max_s=.03,
        domain_randomization_max_observation_noise_std=0.,
        domain_randomization_max_observation_delay_steps=0,
        domain_randomization_sensor_model=sensor)}


def test_profile_enables_joint_factors_without_mutating_input_or_other_dynamics():
    p = profile(); original = copy.deepcopy(p)
    cfg = SimpleNamespace(wheel_torque_scale_nm=5.5, gas_spring_assist_torque_nm=1.0775)
    apply_joint_dr_profile(cfg, p)
    assert cfg.motor_response_enabled and cfg.leg_actuator_dr_enabled
    assert cfg.domain_randomization_sensor_model['enabled']
    assert cfg.domain_randomization_max_base_inertia_fraction == .05
    assert cfg.wheel_torque_scale_nm == 5.5 and cfg.gas_spring_assist_torque_nm == 1.0775
    cfg.domain_randomization_sensor_model['enabled'] = False
    assert p == original


def test_optional_separated_wheel_ranges_must_cover_nominal_and_remain_explicit():
    p = profile()
    p['overrides']['separated_wheel_mechanical_viscous_range'] = [.001, .005]
    p['overrides']['separated_wheel_feedback_damping_range'] = [0., .08]
    cfg = SimpleNamespace(wheel_mechanical_viscous_nm_s_rad=.002,
                          wheel_feedback_damping_nm_s_rad=0.)
    apply_joint_dr_profile(cfg, p)
    assert cfg.separated_wheel_feedback_damping_range == [0., .08]
    p['overrides']['separated_wheel_mechanical_viscous_range'] = [.003, .005]
    with pytest.raises(ValueError, match='excludes nominal'):
        apply_joint_dr_profile(cfg, p)


def test_independent_delay_probabilities_preserve_physics_strength_and_delay_amplitude():
    p=profile();s=p['overrides']['domain_randomization_sensor_model']
    s.update(observation_delay_probability=.1,communication_delay_probability=0.,
             delay_min_steps=1,delay_max_steps=1)
    cfg=SimpleNamespace(domain_randomization_strength_override=.02)
    apply_joint_dr_profile(cfg,p)
    assert cfg.domain_randomization_strength_override==.02
    assert cfg.domain_randomization_sensor_model['observation_delay_probability']==.1
    assert cfg.domain_randomization_sensor_model['communication_delay_probability']==0
    assert cfg.domain_randomization_sensor_model['delay_max_steps']==1


@pytest.mark.parametrize('value',[-.1,1.01,float('nan'),True])
def test_invalid_independent_delay_probability_rejected(value):
    p=profile();p['overrides']['domain_randomization_sensor_model']['observation_delay_probability']=value
    with pytest.raises(ValueError):validate_joint_dr_profile(p)


@pytest.mark.parametrize('case', ['unknown', 'legacy_delay', 'nan', 'fraction', 'delay_reversed'])
def test_invalid_profile_rejected_before_partial_configuration_change(case):
    p = profile(); v = p['overrides']
    if case == 'unknown': v['wheel_torque_scale_nm'] = 3.
    if case == 'legacy_delay': v['domain_randomization_max_observation_delay_steps'] = 1
    if case == 'nan': v['domain_randomization_sensor_model']['gyro_noise_std_rad_s'] = float('nan')
    if case == 'fraction': v['domain_randomization_max_base_inertia_fraction'] = 1.
    if case == 'delay_reversed': v['domain_randomization_sensor_model']['delay_min_steps'] = 4
    cfg = SimpleNamespace(unchanged=True)
    with pytest.raises(ValueError): apply_joint_dr_profile(cfg, p)
    assert vars(cfg) == {'unchanged': True}
