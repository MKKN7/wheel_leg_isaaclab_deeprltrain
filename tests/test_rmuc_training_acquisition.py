"""Execute training command/gate methods, including observation/action timing."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace, MethodType
import torch

ROOT=Path(__file__).resolve().parents[1]
TREE=ast.parse((ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py').read_text())
CLS=next(n for n in TREE.body if isinstance(n,ast.ClassDef) and n.name=='RmucWheelLegEnv')

def setup(seconds=2.):
    cfg=SimpleNamespace(command_reset_acquisition_ramp_s=seconds,curriculum_stage=1,play=False,
        command_stage_linear_x_mps=[3.],command_stage_yaw_rate_rad_s=[0.],command_gate_steps=300)
    e=SimpleNamespace(cfg=cfg,step_dt=.008,_command_level=0,_command=torch.zeros(4,3),
        _rehearsal_mask=torch.tensor([False,True,False,False]),
        _continuous_command_mask=torch.tensor([False,False,True,False]),
        _continuous_commands=torch.tensor([[0.,0.,0.],[0.,0.,0.],[2.9,0.,.1],[0.,0.,0.]]),
        _command_turn_sign=torch.ones(4),_curriculum_step_counter=torch.tensor([0,0,0,250]),
        _current_command_stage_values_tensor=lambda:(torch.tensor([3.,1.5,3.,3.]),torch.zeros(4),None))
    for name in ['_current_command','_acquisition_ramp_mask','_acquisition_command_age','_acquisition_ramp_tracking_ready']:
        node=next(n for n in CLS.body if isinstance(n,ast.FunctionDef) and n.name==name)
        ns={'torch':torch,'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual command>','exec'),ns)
        setattr(e,name,MethodType(ns[name],e))
    return e

def test_only_fixed_highspeed_straight_active_cohort_ramps():
    e=setup();torch.testing.assert_close(e._current_command()[:,0],torch.tensor([1.,1.5,2.9,3.]))
    assert e._current_command()[2,2]==.1
    assert torch.equal(e._acquisition_ramp_mask(),torch.tensor([True,False,False,True]))
    e.cfg.command_stage_yaw_rate_rad_s=[.3]
    assert not e._acquisition_ramp_mask().any()
    e.cfg.command_stage_yaw_rate_rad_s=[0.];e.cfg.command_stage_linear_x_mps=[2.5]
    assert not e._acquisition_ramp_mask().any()

def test_disabled_profile_preserves_commands_and_gate_readiness():
    e=setup(0.)
    assert torch.equal(e._current_command()[:,0],torch.tensor([3.,1.5,2.9,3.]))
    assert e._acquisition_ramp_tracking_ready().all()


def test_turn_entry_keeps_rehearsal_and_continuous_commands_and_physics_timing():
    e=setup(0.)
    e.cfg.command_turn_entry_hold_s=1.
    e.cfg.command_turn_entry_ramp_s=1.
    e._current_command_stage_values_tensor=lambda:(torch.ones(4),torch.full((4,),1.4),None)
    e._command_turn_sign=torch.tensor([1.,-1.,1.,-1.])
    e._curriculum_step_counter=torch.tensor([0,0,0,250])
    torch.testing.assert_close(e._current_command()[:,2],torch.tensor([0.,-1.4,.1,-1.4]))
    e._curriculum_step_counter.fill_(187)
    before=e._current_command().clone()
    assert 0 < before[0,2] < 1.4
    e._command_in_physics_step=True
    e._curriculum_step_counter+=1
    torch.testing.assert_close(e._current_command(),before)
    e._command_in_physics_step=False
    assert e._current_command()[0,2]>before[0,2]


def test_abrupt_entry_cohort_keeps_full_target_at_reset():
    e=setup(0.)
    e.cfg.command_turn_entry_hold_s=0.
    e.cfg.command_turn_entry_ramp_s=.25
    e.cfg.command_turn_entry_abrupt_fraction=.5
    e._rehearsal_mask.zero_()
    e._continuous_command_mask.zero_()
    e._current_command_stage_values_tensor=lambda:(torch.ones(4),torch.full((4,),1.4),None)
    e._curriculum_step_counter.zero_()
    torch.testing.assert_close(e._current_command()[:,2],torch.tensor([1.4,1.4,0.,0.]))
    e._curriculum_step_counter.fill_(32)
    torch.testing.assert_close(e._current_command()[:,2],torch.full((4,),1.4))

def test_observation_command_is_held_through_the_following_physics_step():
    e=setup();e._curriculum_step_counter.fill_(125)
    before=e._current_command().clone()
    e._command_in_physics_step=True;e._curriculum_step_counter+=1
    assert torch.equal(e._current_command(),before)
    e._command_in_physics_step=False
    assert e._current_command()[0,0]>before[0,0]

def test_tracking_requires_entire_window_after_ramp_but_early_failure_still_counts():
    e=setup();e._curriculum_step_counter=torch.tensor([549,0,0,550])
    assert torch.equal(e._acquisition_ramp_tracking_ready(),torch.tensor([False,True,True,True]))
    fn=next(n for n in CLS.body if isinstance(n,ast.FunctionDef) and n.name=='_reset_idx')
    node=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='completed_mask')
    e._curriculum_step_counter[0]=10;e._command_episode_started=torch.ones(4,dtype=torch.bool)
    e._reset_speed_bootstrap_mask=torch.zeros(4,dtype=torch.bool)
    ns={'self':e,'env_ids':torch.tensor([0])};exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual completed mask>','exec'),ns)
    assert ns['completed_mask'].item() and not e._acquisition_ramp_tracking_ready()[0]
