import runpy
from pathlib import Path
import torch

f=runpy.run_path(str(Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/low_speed_tracking.py'))['tracking_targets']

def test_legacy_targets_and_regular_speed_behavior_preserved():
    v=torch.tensor([0.,.05,-.05,.1,.2,.225,.5,1.5,3.2])
    target,norm,direction,margin=f(v,.02)
    assert torch.equal(target,v.abs().clamp_min(.1))
    assert torch.equal(norm,target) and torch.equal(direction,v.sign())
    assert torch.equal(margin,torch.full_like(v,.02))
    new=f(v,.02,True)
    for before,after in zip((target,norm,direction,margin),new):
        assert torch.equal(before[v.abs()>=.2],after[v.abs()>=.2])

def test_zero_and_creep_track_actual_target_not_point1_floor():
    v=torch.tensor([0.,.05,-.05]);target,norm,direction,margin=f(v,.02,True)
    assert torch.equal(target,v.abs()) and (norm==.1).all()
    measured=v.clone();filtered=measured*direction
    cost=((filtered-target).abs()-margin).clamp_min(0)/norm
    assert torch.equal(cost,torch.zeros_like(v))
    # Both signs of zero-command drift remain visible and equally penalized.
    drift=torch.tensor([-.1,.1],requires_grad=True)
    t,n,d,m=f(torch.zeros(2),.02,True)
    cost=((drift*d-t).abs()-m).clamp_min(0)/n
    assert cost[0]==cost[1] and cost[0]>.9
    cost.sum().backward();assert drift.grad[0]<0 and drift.grad[1]>0

def test_creep_deadband_is_below_strict_relative_gate():
    v=torch.tensor([.05,.1,.15]);_,_,_,margin=f(v,.02,True)
    assert (margin<.1*v).all() and torch.isfinite(margin).all()


def test_direction_changes_match_signed_physical_filter_without_false_acceleration():
    rebase=runpy.run_path(str(Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/low_speed_tracking.py'))['rebase_aligned_history']
    # Includes reversal while still rolling forward, reverse-to-stop and reset.
    commands=[.5,-.5,-.5,0.,.25]
    measured=[.4,.4,-.1,-.1,.1]
    signed_speed=torch.tensor(.4);signed_target=torch.tensor(.5)
    aligned_speed=signed_speed.clone();aligned_target=signed_target.clone()
    previous_direction=torch.tensor(1.)
    for c,v in zip(commands,measured):
        direction=torch.tensor(-1. if c<0 else 1.)
        old=rebase(aligned_speed,previous_direction,direction)
        old_target=rebase(aligned_target,previous_direction,direction)
        aligned_speed=old+.1*(torch.tensor(v)*direction-old)
        aligned_target=old_target+.1*(abs(c)-old_target)
        new_signed=signed_speed+.1*(v-signed_speed)
        signed_target=signed_target+.1*(c-signed_target)
        torch.testing.assert_close(aligned_speed,new_signed*direction)
        torch.testing.assert_close(aligned_target,signed_target*direction)
        torch.testing.assert_close(aligned_speed-old,(new_signed-signed_speed)*direction)
        signed_speed=new_signed;previous_direction=direction
