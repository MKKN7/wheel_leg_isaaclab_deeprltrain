import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg'
spec=importlib.util.spec_from_file_location('rmuc_entry_test.reverse_entry',TASK/'reverse_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
package=types.ModuleType('rmuc_entry_test');package.__path__=[str(TASK)]
sys.modules['rmuc_entry_test']=package;sys.modules['rmuc_entry_test.reverse_entry']=entry


def methods():
    tree=ast.parse((TASK/'env.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RmucWheelLegEnv')
    selected=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['_current_command','_acquisition_command_age']]
    ns={'torch':torch,'__package__':'rmuc_entry_test','__name__':'rmuc_entry_test.methods'}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(TASK/'env.py'),'exec'),ns)
    return ns


def test_shared_entry_matches_accepted_evaluation_without_mutating_target():
    sys.path.insert(0,str(ROOT/'scripts'))
    from rmuc_eval_commands import ramp_reverse_commands
    target=torch.tensor([[-1.3,0,0],[-1.4,0,0],[1.5,0,0],[0,0,0],[-1.,0,.5]])
    old=target.clone()
    for tick in [0,1,125,187,188,750]:
        age=torch.full((5,),tick)
        assert torch.equal(entry.reverse_entry_command(target,age,.008,1.5),ramp_reverse_commands(target,age,.008,1.5))
    assert torch.equal(target,old)
    assert entry.reverse_entry_command(target,torch.zeros(5),.008,0) is target


def test_current_command_training_play_and_physics_phase_are_aligned():
    ns=methods()
    class Fake:
        _current_command=ns['_current_command']
        _acquisition_command_age=ns['_acquisition_command_age']
    target=torch.tensor([[-1.3,0,0],[-1.35,0,0],[-1.4,0,0]])
    u=Fake();u.cfg=SimpleNamespace(play=False,curriculum_stage=1,command_turn_entry_hold_s=0.,command_turn_entry_ramp_s=0.,command_reverse_entry_ramp_s=1.5,command_reset_acquisition_ramp_s=0.)
    u.step_dt=.008;u._command=target;u._command_turn_sign=torch.ones(3);u._rehearsal_mask=torch.tensor([False,True,False]);u._continuous_command_mask=torch.tensor([False,False,True]);u._active_targets=[]
    u._continuous_command_values=lambda:target
    u._current_command_stage_values_tensor=lambda:(target[:,0],target[:,2],torch.ones(3))
    for tick in [0,1,187,188,750]:
        u._curriculum_step_counter=torch.tensor([tick,tick,tick]);u._command_in_physics_step=False
        observed=u._current_command()
        assert torch.equal(observed[1:],target[1:]) # Old rehearsal and dynamic schedule unchanged.
        u._curriculum_step_counter+=1;u._command_in_physics_step=True
        assert torch.equal(observed,u._current_command()) # Same command through action, reward, done.
        u.cfg.play=True;u._manual_command=target
        assert torch.equal(observed[0],u._current_command()[0])
        u.cfg.play=False
    u._curriculum_step_counter=torch.tensor([0,500,500]);u._command_in_physics_step=False
    assert u._current_command()[0,0]==0 # Independent episode reset restarts only this entry.


@pytest.mark.parametrize('seconds',[-1,float('nan'),4.01])
def test_invalid_entry_rejected(seconds):
    with pytest.raises(ValueError):entry.reverse_entry_command(torch.zeros(1,3),torch.zeros(1),.008,seconds)


def test_ramp_intermediates_not_mislabeled_as_fixed_targets():
    sys.path.insert(0,str(ROOT/'scripts/rsl_rl'))
    from rmuc_target_policy_gradient import target_gate_from_commands
    from rmuc_command_value_training import exact_command_samples
    targets=[[-1.3,0],[-1.35,0],[-1.4,0]]
    command=torch.tensor(targets)[:,0].repeat_interleave(188)
    age=torch.arange(188).repeat(3);physical=command*(age*.008/1.5).clamp(0,1)
    obs=torch.zeros(len(physical)+3,25);obs[:-3,7]=physical*.5;obs[-3:,7]=torch.tensor(targets)[:,0]*.5
    mask=target_gate_from_commands(obs,targets,tolerance=1e-5).bool()
    assert not mask[:-3].any();assert mask[-3:].all()
    assert torch.equal(mask,exact_command_samples(obs,torch.tensor(targets)))
    assert target_gate_from_commands(obs,targets)[:-3].any() # Old broad tolerance aliases a passing ramp.
