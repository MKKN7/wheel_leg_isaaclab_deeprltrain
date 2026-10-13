"""Exercise the real allocator across changing focus and asynchronous resets."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch

SOURCE=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'
CLASS=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RmucWheelLegEnv')
def method(name):
    fn=next(n for n in CLASS.body if isinstance(n,ast.FunctionDef) and n.name==name)
    scope={'torch':torch}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),scope)
    return scope[name],fn
rebalance,_=method('_rebalance_turn_assignments')
positive_fraction,_=method('_turn_positive_fraction_for_stage')
sampling_spec=importlib.util.spec_from_file_location('rmuc_command_sampling',SOURCE.with_name('command_sampling.py'))
sampling=importlib.util.module_from_spec(sampling_spec)
sampling_spec.loader.exec_module(sampling)


@pytest.mark.parametrize('fraction,expected',[(1.,[100,0]),(0.,[0,100])])
def test_explicit_direction_survives_asynchronous_resets(fraction,expected):
    e=environment(1)
    e.cfg.command_turn_positive_fraction_override=fraction
    e._turn_positive_fraction_for_stage=lambda stage:positive_fraction(e,stage)
    for _ in range(100):rebalance(e,torch.tensor([0]))
    assert e._turn_episode_assignment_count[0].tolist()==expected


@pytest.mark.parametrize('fraction',[-2.,1.1,float('nan'),float('inf')])
def test_invalid_direction_override_rejected(fraction):
    with pytest.raises(ValueError):
        positive_fraction(NS(cfg=NS(command_turn_positive_fraction_override=fraction)),0)

def environment(count=1000):
    return NS(cfg=NS(curriculum_stage=1),device='cpu',_command_level=0,
        _rehearsal_mask=torch.zeros(count,dtype=torch.bool),_rehearsal_stage=torch.zeros(count,dtype=torch.long),
        _turn_episode_assignment_count=torch.zeros(2,2,dtype=torch.long),
        _turn_positive_assignment_credit=torch.zeros(2,dtype=torch.float64),
        _command_turn_sign=torch.ones(count),_turn_positive_fraction_for_stage=lambda stage:.5)

@pytest.mark.parametrize('old,new',[(.8,.2),(.2,.8),(.8,.5)])
def test_focus_change_does_not_reinterpret_old_episodes(old,new):
    e=environment();e._turn_positive_fraction_for_stage=lambda stage:old
    rebalance(e,torch.arange(1000))
    previous=e._turn_episode_assignment_count.clone()
    e._turn_positive_fraction_for_stage=lambda stage:new
    rebalance(e,torch.arange(128))
    assigned=int((e._turn_episode_assignment_count-previous)[0,0])
    assert abs(assigned-128*new)<=.500001
    assert 0<assigned<128

def test_single_environment_reset_tracks_integrated_fraction_after_reversal():
    e=environment(1);expected=0.
    for fraction in [.8]*100+[.2]*100+[.8]*100:
        e._turn_positive_fraction_for_stage=lambda stage:fraction
        rebalance(e,torch.tensor([0]));expected+=fraction
        assert abs(float(e._turn_episode_assignment_count[0,0])-expected)<=.500001
        assert abs(float(e._turn_positive_assignment_credit[0]))<=.500001
    assert int(e._turn_episode_assignment_count.sum())==300

def test_stages_have_independent_credit_and_nonreset_signs_are_preserved():
    e=environment(4);e._rehearsal_mask[2:]=True;e._rehearsal_stage[2:]=1
    e._turn_positive_fraction_for_stage=lambda stage:[.2,.8][stage]
    for _ in range(100):
        for index in (0,2):
            old=e._command_turn_sign.clone();rebalance(e,torch.tensor([index]))
            mask=torch.arange(4)!=index
            assert torch.equal(old[mask],e._command_turn_sign[mask])
    assert e._turn_episode_assignment_count.tolist()==[[20,80],[80,20]]

def test_measurement_reset_clears_credit_with_assignment_counts():
    reset,fn=method('_reset_active_command_gate_measurements')
    e=environment(1)
    # Provide the real method's unrelated metric tensors without a simulator.
    for node in ast.walk(fn):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='zero_':
            setattr(e,node.func.value.attr,torch.ones(2))
    e._turn_positive_assignment_credit.fill_(.4)
    reset(e)
    assert not e._turn_episode_assignment_count.any()
    assert not e._turn_positive_assignment_credit.any()


def test_fixed_cohorts_preserve_control_step_direction_across_short_resets():
    e=environment(128)
    e._turn_fixed_sign_mask=sampling.fixed_turn_sign_cohorts(128,.8,458,'cpu')
    assert int(e._turn_fixed_sign_mask.sum())==102
    first=None
    for _ in range(20):
        rebalance(e,torch.arange(128))
        if first is None:first=e._command_turn_sign.clone()
        else:torch.testing.assert_close(e._command_turn_sign,first)
    assert e._turn_episode_assignment_count[0].tolist()==[2040,520]
