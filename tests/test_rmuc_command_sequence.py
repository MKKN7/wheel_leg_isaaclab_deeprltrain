import pytest
import torch
from rmuc_command_sequence import validate_sequence,sequence_commands


def test_reverse_oscillation_acquisition_does_not_suppress_runaway_or_other_domains():
    from rmuc_command_sequence import reverse_oscillation_ready, reverse_runaway_cost
    cmd=torch.tensor([[-.6,0.,0.],[-.6,0.,0.],[-.6,0.,0.],[-.6,0.,0.],[.6,0.,0.],[0.,0.,0.],[-.6,0.,.2]])
    age=torch.tensor([150,374,375,150,150,150,150]);dynamic=torch.tensor([False,False,False,True,False,False,False])
    assert reverse_oscillation_ready(cmd,age,dynamic,.008,0).all()
    assert reverse_oscillation_ready(cmd,age,dynamic,.008,3).tolist()==[False,False,True,True,True,True,True]
    # Large initial overspeed remains penalized even while oscillation is gated.
    cost=reverse_runaway_cost(cmd,torch.full((7,),-.9),torch.full((7,),-.91),torch.full((7,),-.1),torch.ones(7,dtype=torch.bool),.008)
    assert cost[0]>0 and cost[1]>0
    for invalid in [-1,float('nan'),6]:
        with pytest.raises(ValueError):reverse_oscillation_ready(cmd,age,dynamic,.008,invalid)


def test_boundary_reset_and_reverse_preserve_existing_command_slots():
    table=validate_sequence([[0,0,0],[2,.5,0],[4,1.,0],[6,0,0],[8,-.5,0]],.008,20)
    ages=torch.tensor([0,249,250,499,500,750,1000,0])
    cmd=sequence_commands(table,ages)
    torch.testing.assert_close(cmd[:,0],torch.tensor([0.,0.,.5,.5,1.,0.,-.5,0.]))
    assert not cmd[:,1:].any()


@pytest.mark.parametrize('rows',[
 [[1,0,0],[2,1,0]],[[0,0,0],[0,1,0]],[[0,0,0],[.003,1,0]],
 [[0,0,0],[20,1,0]],[[0,0,0],[2,3.3,0]],[[0,0,0],[2,0,9]],
 [[0,0,0],[2,float('nan'),0]],[[0,0,0],[True,1,0]]])
def test_invalid_sequences_rejected(rows):
    with pytest.raises(ValueError):validate_sequence(rows,.008,20)


def test_dynamic_reward_distinguishes_braking_and_preserves_static_cohorts():
    from rmuc_command_sequence import dynamic_tracking_cost
    v=torch.tensor([.0,.1,.2,.5,1.,1.5],requires_grad=True)
    c=dynamic_tracking_cost(torch.zeros_like(v),v,torch.ones_like(v,dtype=torch.bool),.5)
    assert c[0]==0 and torch.all(torch.diff(c)>0)
    c.sum().backward();assert torch.all(v.grad[1:]>0)
    torch.testing.assert_close(c,dynamic_tracking_cost(torch.zeros_like(v),-v,torch.ones_like(v,dtype=torch.bool),.5))
    assert not dynamic_tracking_cost(torch.ones_like(v),v,torch.zeros_like(v,dtype=torch.bool),.5).any()
    assert dynamic_tracking_cost(torch.tensor([0.]),torch.tensor([100.]),torch.tensor([True]),.5).item()==4


def test_fixed_reverse_dense_tracking_has_correct_braking_signal_and_scope():
    from rmuc_command_sequence import dynamic_tracking_cost,tracking_cohort_mask
    command=torch.tensor([[-.1,0,0],[-.1,0,0],[.1,0,0],[0,0,0],[-.1,0,.3]])
    dynamic=torch.tensor([False,True,False,False,False])
    assert tracking_cohort_mask(command,dynamic) is dynamic
    mask=tracking_cohort_mask(command,dynamic,True)
    assert torch.equal(mask,torch.tensor([True,True,False,False,False]))
    vx=torch.tensor([-.7,-.7,.7,.7,-.7],requires_grad=True)
    cost=dynamic_tracking_cost(command[:,0],vx,mask,.5)
    assert cost[0]==cost[1] and not cost[2:].any()
    cost.sum().backward()
    assert vx.grad[0]<0 and vx.grad[1]<0 and not vx.grad[2:].any()
    # Both severe overspeed and wrong-way motion have a useful error slope.
    v=torch.tensor([-1.5,-.7,-.3,-.1,.3],requires_grad=True)
    c=dynamic_tracking_cost(torch.full_like(v,-.1),v,torch.ones_like(v,dtype=torch.bool),.5)
    c.sum().backward()
    assert (v.grad[:3]<0).all() and v.grad[3]==0 and v.grad[4]>0


def test_stop_posture_only_zero_command_dynamic_cohort():
    from rmuc_command_sequence import dynamic_stop_posture_cost
    commands=torch.tensor([[0.,0,0],[.5,0,0],[0,0,.5],[0,0,0]])
    result=dynamic_stop_posture_cost(commands,torch.zeros(4),torch.ones(4)*.25,torch.ones(4),torch.tensor([True,True,True,False]))
    assert result[0]>0 and not result[1:].any()


def test_stop_posture_symmetric_bounded_and_near_stop_emphasis():
    from rmuc_command_sequence import dynamic_stop_posture_cost
    cmd=torch.zeros(3,3);vx=torch.tensor([0.,.2,.5]);angle=torch.ones(3)*.15;rate=torch.ones(3)*.3;mask=torch.ones(3,dtype=torch.bool)
    value=dynamic_stop_posture_cost(cmd,vx,angle,rate,mask)
    assert torch.all(torch.diff(value)<0)
    torch.testing.assert_close(value,dynamic_stop_posture_cost(cmd,-vx,-angle,-rate,mask))
    assert not dynamic_stop_posture_cost(cmd,vx,torch.ones(3)*.01,torch.zeros(3),mask).any()
    assert dynamic_stop_posture_cost(cmd,torch.zeros(3),angle*100,rate*100,mask).max()<=4


def test_alternate_timing_preserves_commands_but_changes_phase_and_reward():
 import runpy
 from pathlib import Path
 m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/command_sequence.py'))
 short=[[0,0,0],[2,-.25,0],[4,-.6,0],[6,-.8,0],[9,-.4,0],[11,0,0]]
 long=[[0,0,0],[2,-.25,0],[6,-.6,0],[10,-.8,0],[14,-.4,0],[17,0,0]]
 a=validate_sequence(short,.008,20);b=m['alternate_sequence'](short,long,.008,20)
 ages=torch.tensor([625,625]);ca=sequence_commands(a,ages);cb=sequence_commands(b,ages)
 assert ca[0,0]==-.6 and cb[0,0]==-.25
 cost=m['sequence_overshoot_cost'];mask=torch.ones(2,dtype=torch.bool);speed=torch.tensor([-.3,-.3])
 assert not cost(a,ages,ca,speed,mask).any()
 assert (cost(b,ages,cb,speed,mask)>0).all()
 for table in [a,b]:assert sequence_commands(table,torch.zeros(2,dtype=torch.long)).abs().max()==0
 with pytest.raises(ValueError):m['alternate_sequence'](short,[[0,0,0],[2,.25,0]],.008,20)
 assert m['alternate_sequence'](short,[],.008,20) is None
