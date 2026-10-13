import importlib.util
from pathlib import Path
import torch
import pytest

path=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/recovery_rewards.py'
spec=importlib.util.spec_from_file_location('recovery_rewards',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def evaluate(up,height,previous=None,valid=None,established=None):
    up=torch.tensor(up);n=len(up)
    return m.compute_recovery_rewards(up,torch.tensor(height),
        torch.tensor(established if established is not None else [False]*n),
        torch.tensor(previous) if previous is not None else up,
        torch.tensor(valid if valid is not None else [False]*n),
        torch.zeros(n,6),torch.zeros(n,6),torch.zeros(n,4),m.DEFAULT_WEIGHTS)


def test_inverted_height_has_no_reward_and_upright_hold_is_preferred():
    rewards,terms=evaluate([-1.,-1.,0.,1.,1.],[.07,.4,.4,.4,.4],established=[False,False,False,False,True])
    assert terms['upright_height'][:3].tolist()==[0.,0.,0.]
    assert rewards[0]==rewards[1] and rewards[0]<0
    assert rewards[4]>rewards[3]>rewards[2]>rewards[0]


def test_reset_does_not_award_progress_and_reversal_is_penalized():
    _,terms=evaluate([.5,.5,-.5],[.3]*3,previous=[-.5,-.5,.5],valid=[False,True,True])
    assert terms['upright_progress'].tolist()==[0.,2.,-2.]


def test_action_effort_costs_cannot_create_reward():
    zeros=torch.zeros(2);actions=torch.ones(2,6);effort=torch.ones(2,4)
    reward,terms=m.compute_recovery_rewards(zeros,zeros,zeros.bool(),zeros,zeros.bool(),
        actions,torch.zeros_like(actions),effort,m.DEFAULT_WEIGHTS)
    assert (terms['action_rate']<0).all() and (terms['leg_effort']<0).all()
    assert torch.isfinite(reward).all()


def test_unknown_reward_key_rejected():
    with pytest.raises(ValueError):
        m.compute_recovery_rewards(None,None,None,None,None,None,None,None,{'track_velocity':1.})


def test_standing_scores_reject_airborne_inverted_and_cap_height():
    up=torch.tensor([1.,-1.,1.,1.,1.])
    height=torch.tensor([.5,.5,.14,.29,.6])
    supported=torch.tensor([False,True,True,True,True])
    ready=torch.tensor([False,False,False,True,True])
    terms=m.standing_recovery_terms(up,height,supported,ready,
        torch.tensor([.5,.5,0.,.1,.5]),.28,.978,.5)
    assert terms['supported_height'].tolist()==[0.,0.,.5,1.,1.]
    assert terms['height_reached'].tolist()==[0.,0.,0.,1.,1.]
    assert torch.allclose(terms['stable_hold'],torch.tensor([0.,0.,0.,.2,1.]))


def test_hold_reward_vanishes_when_not_ready_even_with_stale_counter():
    terms=m.standing_recovery_terms(torch.ones(1),torch.ones(1),torch.ones(1,dtype=torch.bool),
        torch.zeros(1,dtype=torch.bool),torch.ones(1),.28,.978,.5)
    assert terms['stable_hold'].item()==0


def test_refall_initial_fall_dwell_single_event_and_rearm():
    armed=torch.tensor([False]);bad=torch.tensor([0]);ready=torch.tensor([0])
    for _ in range(10):
        armed,bad,event=m.refall_event(armed,bad,ready,torch.tensor([True]),2,3)
        assert not event.item()
    armed,bad,event=m.refall_event(armed,bad,torch.tensor([2]),torch.tensor([False]),2,3)
    assert armed.item()
    for i in range(5):
        armed,bad,event=m.refall_event(armed,bad,ready,torch.tensor([True]),2,3)
        assert event.item()==(i==2)
    armed,bad,_=m.refall_event(armed,bad,torch.tensor([2]),torch.tensor([False]),2,3)
    assert armed.item()
    armed,bad,event=m.refall_event(armed,bad,ready,torch.tensor([True]),2,3)
    armed,bad,event=m.refall_event(armed,bad,ready,torch.tensor([False]),2,3)
    assert bad.item()==0 and not event.item()


def test_no_static_side_income_and_progress_regresses():
    up=torch.tensor([-1.,0.,.5,1.])
    assert m.upright_without_side_income(up).tolist()==[0.,0.,0.,1.]
    progress=m.recovery_phase_progress(
        torch.tensor([-1.,0.,1.,1.,0.]),torch.tensor([.07,.18,.28,.28,.18]),
        torch.tensor([False,False,True,True,False]),torch.tensor([0.,0.,0.,.5,0.]),
        .28,.978,.5)
    assert ((progress>=0)&(progress<=1)).all()
    assert progress[0]==0 and progress[2]==.8 and progress[3]==1.
    assert progress[4]<progress[3]


def test_rise_shaping_available_before_strict_standing_without_side_income():
    up=torch.tensor([0.,.5,.6,.866,1.,1.])
    height=torch.tensor([.28,.28,.14,.28,.28,.6])
    supported=torch.tensor([True,True,True,True,False,True])
    r=m.supported_rise_shaping(up,height,supported,.28)
    assert r[0]==0 and r[1]==0 and r[4]==0
    assert 0<r[2]<r[3]<r[5] and r[5]==1

def test_stable_pose_shaping_rewards_low_speed_supported_pose():
    up=torch.tensor([.6,.9,.99,.99])
    h=torch.tensor([.28,.28,.28,.28])
    supported=torch.tensor([True,True,True,False])
    slow=torch.tensor([.01,.01,.01,.01])
    fast=torch.tensor([.3,.3,.3,.3])
    r=m.stable_pose_shaping(up,h,supported,slow,slow,.28,.36)
    rf=m.stable_pose_shaping(up,h,supported,fast,fast,.28,.36)
    assert r[0] < r[1] < r[2] and r[3]==0
    assert (rf[:3] < r[:3]).all() and rf[3]==r[3]==0

def test_safety_terms_grow_before_limit_and_impact():
    limits=torch.tensor([[[-1.,1.],[-1.,1.]]])
    q=torch.tensor([[0.,.9]])
    barrier=m.joint_limit_barrier(q,limits,.2)
    assert barrier.item()>0
    forces=torch.tensor([[[0.,0.,100.],[0.,0.,500.]]])
    assert m.contact_impact_penalty(forces,300.).item()>0

def test_slew_schedule_is_bounded_and_reaches_terminal_rate():
    assert m.linear_slew_schedule(4.,1.5,0,100)==4.
    assert m.linear_slew_schedule(4.,1.5,50,100)==2.75
    assert m.linear_slew_schedule(4.,1.5,100,100)==1.5
    assert m.linear_slew_schedule(4.,1.5,1000,100)==1.5

def test_braking_cost_no_side_income_and_slowing_reduces_cost():
    speed=torch.tensor([0.,.1,.3,.6],requires_grad=True)
    cost=m.supported_braking_cost(torch.ones(4),torch.ones(4,dtype=torch.bool),speed)
    assert cost[0]==0 and (cost[1:]>cost[:-1]).all() and (cost<1).all()
    grad=torch.autograd.grad(cost.sum(),speed)[0]
    assert (grad[1:]>0).all()
    assert m.supported_braking_cost(torch.tensor([0.,.5]),torch.ones(2,dtype=torch.bool),torch.ones(2)).sum()==0
    assert m.supported_braking_cost(torch.ones(1),torch.zeros(1,dtype=torch.bool),torch.ones(1)).item()==0

def test_double_contact_is_one_shot_transition_only():
    prev=torch.tensor([False,True,False]);cur=torch.tensor([True,True,False])
    assert m.double_contact_acquired(prev,cur).tolist()==[1.,0.,0.]


def test_curriculum_requires_consecutive_success_and_respects_evaluation_ceiling():
    stage=torch.tensor([0,0,1,4]);streak=torch.tensor([7,7,7,7])
    success=torch.tensor([True,False,True,True])
    new,count=m.advance_recovery_stage(stage,streak,success,8,4)
    assert new.tolist()==[1,0,2,4] and count.tolist()==[0,0,0,8]
    new,count=m.advance_recovery_stage(torch.tensor([0]),torch.tensor([8]),torch.tensor([True]),8,0)
    assert new.item()==0 and count.item()==8


def test_gap_progress_no_reset_static_takeoff_inverted_or_front_income():
    prev=torch.full((8,),.3);gap=torch.tensor([.15,.3,.45,.15,.15,.15,.15,.15])
    valid=torch.tensor([True,True,True,False,True,True,True,True])
    supported=torch.tensor([True,True,True,True,False,True,True,True])
    prior_supported=torch.tensor([True,True,True,True,True,False,True,True])
    up=torch.tensor([.8,.8,.8,.8,.8,.8,-.1,.8])
    side=torch.tensor([True,True,True,True,True,True,True,False])
    result=m.supported_wheel_gap_progress(prev,gap,valid,prior_supported,supported,
        torch.ones(8),up,side)
    torch.testing.assert_close(result,torch.tensor([.5,0.,-.5,0.,0.,0.,0.,0.]))
    # A closed gap cycle has zero undiscounted progress reward.
    a=m.supported_wheel_gap_progress(prev,gap,torch.ones(8,dtype=torch.bool),
        torch.ones(8,dtype=torch.bool),torch.ones(8,dtype=torch.bool),torch.ones(8),torch.ones(8),torch.ones(8,dtype=torch.bool))
    b=m.supported_wheel_gap_progress(gap,prev,torch.ones(8,dtype=torch.bool),
        torch.ones(8,dtype=torch.bool),torch.ones(8,dtype=torch.bool),torch.ones(8),torch.ones(8),torch.ones(8,dtype=torch.bool))
    torch.testing.assert_close(a+b,torch.zeros(8))


def test_contact_once_debounce_chatter_loss_recontact_and_reset():
    dwell=torch.zeros(2,dtype=torch.long);paid=torch.zeros(2,dtype=torch.bool)
    total=torch.zeros(2)
    for supported in ([True,True],[False,True],[True,True],[True,False],[True,True],[False,True],[True,True],[True,True],[True,True]):
        dwell,paid,event=m.debounced_contact_event(dwell,paid,torch.tensor(supported),3)
        total+=event
    assert total.tolist()==[1.,1.] and paid.all()
    # A reset rearms only the reset environment.
    dwell[0]=0;paid[0]=False
    for _ in range(3):
        dwell,paid,event=m.debounced_contact_event(dwell,paid,torch.ones(2,dtype=torch.bool),3)
        total+=event
    assert total.tolist()==[2.,1.]


def test_reset_cohorts_repeat_without_changing_rng_and_reject_bad_direction():
    before=torch.get_rng_state().clone()
    layout=m.recovery_cohort_layout(8,[(15,1),(0,0),(5,2),(15,3)],device='cpu')
    assert layout.tolist()==[[15,1],[0,0],[5,2],[15,3]]*2
    assert torch.equal(before,torch.get_rng_state())
    assert layout[torch.tensor([7,0,5])].tolist()==[[15,3],[15,1],[0,0]]
    for bad in ([],[(15,4)],[(12,1)],[(15.,1)]):
        with pytest.raises(ValueError):m.recovery_cohort_layout(4,bad,device='cpu')
