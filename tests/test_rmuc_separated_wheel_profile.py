import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('separated_profile',ROOT/'scripts/rmuc_separated_wheel_profile.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def profile():
    return dict(schema_version=1,scope='diagnostic',wheel_torque_contract='separated_passive_v1',
                hardware_sign_verified=False,hardware_feedback_identified=False,
                wheel_mechanical_viscous_nm_s_rad=.003, wheel_coulomb_friction_nm=.004,
                wheel_feedback_damping_nm_s_rad=.005,wheel_coulomb_omega_eps_rad_s=.2)


def config():
    return NS(sim=NS(dt=.002),decimation=4,wheel_torque_scale_nm=5.5,
              robot_cfg=NS(actuators={'wheel_motors':NS(effort_limit=5.5,damping=.080625),
                                      'leg_motors':NS(stiffness=60.,damping=2.,effort_limit=40.)}))


def test_explicit_components_and_multirate_without_leg_changes():
    c=config(); p=profile(); before=copy.deepcopy(p);leg=copy.deepcopy(c.robot_cfg.actuators['leg_motors'])
    result=m.apply_separated_wheel_profile(c,p)
    assert p==before and c.robot_cfg.actuators['leg_motors']==leg
    for field in m.FIELDS: assert getattr(c,field)==p[field]
    assert c.sim.dt==.001 and c.decimation==8 and c.leg_pd_update_period_s==.002
    assert c.robot_cfg.actuators['wheel_motors'].damping==0.
    assert c.wheel_damping_anneal_target==-1 and c.wheel_damping_compensation_ratio==0
    assert c.position_authority_enabled and not c.policy_outer_feedback_enabled
    assert result['readback']['wheel_feedback_damping_nm_s_rad']==.005
    assert result['readback']['wheel_motor_actuator_pd_damping']==0.
    assert result['readback']['wheel_feedback_applied_after_motor_clip'] is True
    assert result['behavior_certified'] is False


def test_schema2_active_mode_and_schema1_explicitly_restore_historical_location():
    c=config();p=profile()
    p.update(schema_version=2, wheel_feedback_damping_mode='active_motor_v1')
    result=m.apply_separated_wheel_profile(c,p)
    assert c.wheel_feedback_damping_mode == 'active_motor_v1'
    assert result['readback']['wheel_feedback_applied_after_motor_clip'] is False
    result=m.apply_separated_wheel_profile(c,profile())
    assert c.wheel_feedback_damping_mode == 'postclip_loss_v1'
    assert result['readback']['wheel_feedback_applied_after_motor_clip'] is True


def test_two_ms_diagnostic_preserves_policy_leg_periods_and_actuator_contract():
    one, two = config(), config()
    m.apply_separated_wheel_profile(one, profile())
    result = m.apply_separated_wheel_profile(two, profile(), physics_dt=.002)
    assert two.sim.dt == .002 and two.decimation == 4
    assert two.leg_pd_update_period_s == one.leg_pd_update_period_s == .002
    assert result['readback']['policy_dt'] == .008
    assert two.robot_cfg == one.robot_cfg
    for name in m.FIELDS:
        assert getattr(one, name) == getattr(two, name)
    assert result['behavior_certified'] is False


def test_unsupported_timestep_rejected_before_mutation():
    c = config()
    before = copy.deepcopy(c)
    with pytest.raises(ValueError):
        m.apply_separated_wheel_profile(c, profile(), physics_dt=.003)
    assert c == before


@pytest.mark.parametrize('key,value', [('schema_version',True),('schema_version',2),('scope','hardware'),
 ('wheel_torque_contract','legacy_idealpd'),('hardware_sign_verified',True),
 ('hardware_feedback_identified',True),('wheel_feedback_damping_nm_s_rad',True),
 ('wheel_feedback_damping_nm_s_rad','0'),('wheel_mechanical_viscous_nm_s_rad',-.1),
 ('wheel_coulomb_friction_nm',float('inf')),('wheel_coulomb_omega_eps_rad_s',0),
 ('wheel_feedback_damping_nm_s_rad',float('nan'))])
def test_invalid_profile_rejected_without_config_mutation(key,value):
    c=config(); before=copy.deepcopy(c);p=profile();p[key]=value
    with pytest.raises(ValueError):m.apply_separated_wheel_profile(c,p)
    assert c==before


@pytest.mark.parametrize('field',m.FIELDS)
def test_missing_component_never_inherits_default(field):
    p=profile();del p[field]
    with pytest.raises(ValueError):m.validate_profile(p)


@pytest.mark.parametrize('target',['scale','peak'])
def test_reject_derated_or_rescaled_config(target):
    c=config()
    if target=='scale':c.wheel_torque_scale_nm=3.
    else:c.robot_cfg.actuators['wheel_motors'].effort_limit=3.
    before=copy.deepcopy(c)
    with pytest.raises(ValueError):m.apply_separated_wheel_profile(c,profile())
    assert c==before


@pytest.mark.parametrize('extension',['json','yaml'])
def test_load_json_yaml(tmp_path,extension):
    import yaml
    p=tmp_path/('profile.'+extension)
    p.write_text(json.dumps(profile()) if extension=='json' else yaml.safe_dump(profile()))
    assert m.load_profile(p)==profile()


def options():
    return NS(diagnostic_separated_wheel_profile=Path('profile.json'),
              diagnostic_disable_outer_feedback=True,position_authority_consumer=True)


@pytest.mark.parametrize('field,value',[('diagnostic_hardware_profile','legacy.yaml'),
 ('diagnostic_wheel_velocity_profile','velocity.yaml'),('diagnostic_wheel_velocity_equivalence',True),
 ('diagnostic_disable_outer_feedback',False),('position_authority_consumer',False)])
def test_ambiguous_or_feedback_consumer_conflict(field,value):
    a=options();setattr(a,field,value)
    with pytest.raises(ValueError):m.validate_options(a)


def test_legacy_options_are_noop():
    m.validate_options(NS())
    m.validate_options(options())


def test_evaluator_integration_cpu_ast():
    import ast
    source=(ROOT/'scripts/evaluate_rmuc_grid.py').read_text()
    tree=ast.parse(source)
    main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    # Capture provenance before gym.make and make diagnostic result non-certifying.
    assert source.index("task_root/'wheel_passive.py'") < source.index("gym.make(")
    assert source.index('applied_separated_wheel_profile = apply_separated_wheel_profile(') < source.index('source_hashes =')
    blocks=[node for node in ast.walk(main) if isinstance(node,ast.If) and
            ast.unparse(node.test)=='applied_separated_wheel_profile is not None']
    assert any("report['passed'] = False" in ast.unparse(block) for block in blocks)
    assert 'runtime_readback' in source and 'runtime_leg_pd_interval_steps' in source
