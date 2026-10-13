import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('hardware_profile',ROOT/'scripts/rmuc_hardware_profile.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

# Immutable historical schema2 experiment: never reinterpret it as passive damping.
def profile():return yaml.safe_load((ROOT/'artifacts/continuation_v436/hardware_profile.yaml').read_text())
def target():return yaml.safe_load((ROOT/'configs/rmuc_hardware_target.yaml').read_text())
def config():return SimpleNamespace(robot_cfg=SimpleNamespace(actuators={'wheel_motors':SimpleNamespace(damping=.1,effort_limit=5.5),
    'leg_motors':SimpleNamespace(damping=2.,effort_limit=40.)}),torque_limit_ratio_real=.8,
    wheel_torque_scale_nm=5.5,domain_randomization_max_wheel_damping_fraction=.1,
    gas_spring_radial_force_n=0.,gas_spring_assist_torque_nm=0.,motor_response_enabled=False,
    motor_response_time_constant_min_s=.002,motor_response_time_constant_max_s=.03)

def test_target_maps_to_actual_actuator_and_DR_without_changing_legs():
    c=config();p=profile();before=copy.deepcopy(p);result=m.apply_hardware_profile(c,p)
    assert c.robot_cfg.actuators['wheel_motors'].effort_limit==5.5
    assert c.robot_cfg.actuators['wheel_motors'].damping==.078625
    assert c.wheel_torque_scale_nm==5.5 and c.torque_limit_ratio_real==.8
    assert c.gas_spring_radial_force_n==0 and c.gas_spring_assist_torque_nm==1.0775
    assert c.gas_spring_radial_dr_fraction==0.
    assert c.gas_spring_torque_common_dr_fraction+c.gas_spring_torque_independent_dr_fraction==.2
    assert c.robot_cfg.actuators['leg_motors'].effort_limit==40
    assert c.motor_response_enabled and result['measured_calibration'] is False and p==before

@pytest.mark.parametrize('kind',['gear','wrong_spring_model','torque','damping','nan'])
def test_inconsistent_target_rejected_without_partial_config_mutation(kind):
    p=profile();c=config()
    if kind=='gear':p['wheel']['multiply_by_gear_ratio']=True
    if kind=='wrong_spring_model':p['gas_spring']['model']='radial_force'
    if kind=='torque':p['wheel']['target_actuator_effort_limit_nm']=3.0
    if kind=='damping':p['wheel']['damping']['full_dr_fraction']=.1
    if kind=='nan':p['motor_response']['time_constant_range_s'][1]=float('nan')
    with pytest.raises(ValueError):m.apply_hardware_profile(c,p)
    assert c.robot_cfg.actuators['wheel_motors'].effort_limit==5.5

def test_zero_fraction_leaves_entire_configuration_unchanged():
    c=config();before=copy.deepcopy(c)
    r=m.apply_hardware_profile(c,profile(),0.)
    assert vars(c)==vars(before) and r['applied_overrides']=={}

@pytest.mark.parametrize('fraction',[.25,.5,1.])
def test_preserved_action_units_do_not_preserve_excess_physical_torque(fraction):
    c=config();p=profile();p['wheel']['target_action_torque_scale_nm']=5.5
    m.apply_hardware_profile(c,p,fraction)
    assert c.wheel_torque_scale_nm==5.5
    assert c.robot_cfg.actuators['wheel_motors'].effort_limit==pytest.approx(5.5)

@pytest.mark.parametrize('scale',[0.,-1.,float('nan'),float('inf')])
def test_invalid_action_scale_is_rejected_without_mutation(scale):
    c=config();p=profile();before=copy.deepcopy(c)
    p['wheel']['target_action_torque_scale_nm']=scale
    with pytest.raises(ValueError):m.apply_hardware_profile(c,p)
    assert vars(c)==vars(before)

@pytest.mark.parametrize('fraction',[.05,.25,.5,1.])
def test_partial_migration_interpolates_mechanics_and_full_DR_bounds(fraction):
    c=config();r=m.apply_hardware_profile(c,profile(),fraction)
    assert c.wheel_torque_scale_nm==pytest.approx(5.5)
    assert c.robot_cfg.actuators['wheel_motors'].effort_limit==c.wheel_torque_scale_nm
    assert c.gas_spring_radial_force_n==0
    assert c.gas_spring_assist_torque_nm==pytest.approx(1.0775*fraction)
    low,high=r['damping_full_DR_range'];center=c.robot_cfg.actuators['wheel_motors'].damping
    extent=c.domain_randomization_max_wheel_damping_fraction
    assert low==pytest.approx(.09*(1-fraction)+.0393125*fraction)
    assert high==pytest.approx(.11*(1-fraction)+.1179375*fraction)
    assert center*(1-extent)==pytest.approx(low) and center*(1+extent)==pytest.approx(high)
    assert c.motor_response_time_constant_max_s==pytest.approx(.03*fraction)


def test_explicit_legacy_derating_profile_is_not_reinterpreted():
    c=config();p=profile();p['wheel'].pop('max_peak_output_torque_nm')
    p['wheel']['target_actuator_effort_limit_nm']=3.
    r=m.apply_hardware_profile(c,p)
    assert c.robot_cfg.actuators['wheel_motors'].effort_limit==3.
    assert r['legacy_continuous_as_hard_limit']

def test_continuous_reference_does_not_clip_peak():
    c=config();r=m.apply_hardware_profile(c,profile())
    assert r['continuous_torque_reference_nm']==3.
    assert r['peak_torque_limit_nm']==5.5 and c.wheel_authored_effort_limit_nm==5.5


@pytest.mark.parametrize('fraction', [0., .5, 1.])
def test_unidentified_feedback_target_fails_closed_without_mutation(fraction):
    c=config(); before=copy.deepcopy(c); p=target()
    with pytest.raises(ValueError, match='unidentified feedback damping'):
        m.apply_hardware_profile(c,p,fraction)
    assert vars(c)==vars(before)


def test_identified_feedback_does_not_silently_enable_legacy_adapter():
    c=config(); before=copy.deepcopy(c); p=target()
    p['wheel']['feedback_damping'].update(value=.1, identified=True)
    with pytest.raises(ValueError, match='multirate runtime adapter'):
        m.apply_hardware_profile(c,p)
    assert vars(c)==vars(before)


def test_target_component_separation_and_timing():
    p=target(); w=p['wheel']
    assert p['schema_version']==3 and 'damping' not in w
    assert w['mechanical_viscous']['nominal']==.002
    assert w['mechanical_viscous']['range']==[.001,.005]
    assert w['coulomb_friction']['nominal']==.001
    assert w['feedback_damping']['value'] is None
    assert not w['combined_torque_outer_clip']
    assert p['timing']['wheel_execution_period_s']==.001
    assert p['timing']['leg_pd_period_s']==.002
    assert p['timing']['policy_period_s']==.008
    assert p['gas_spring']['equivalent_torque_nm_per_leg']==10.775


def test_target_audit_reports_unidentified_feedback_and_no_launch_readiness():
    spec=importlib.util.spec_from_file_location('hardware_audit',ROOT/'scripts/audit_rmuc_hardware_target.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.audit(target())
    assert result['supplied_parameters_checked']
    assert not result['training_launch_ready'] and not result['hardware_certified']
    assert 'feedback_damping_absolute_value_unidentified' in result['unresolved_items']
    assert result['mechanical_viscous_range_nm_s_rad']==[.001,.005]
