import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_position_authority import PositionAuthorityHead, execute_authority_action

def inputs():
    x=torch.zeros(6,25);x[:,7]=torch.tensor([.5,.5,.5,.5,1.25,-.5])
    x[:,8]=torch.tensor([1.,-1.,.45,-.45,1.,1.])
    a=torch.tensor([.2,-.2,.4,1.2,.1,-.3]).repeat(6,1)
    return x,a

def test_zero_install_preserves_legacy_execution_but_protected_raw_exact():
    x,a=inputs();head=PositionAuthorityHead()
    result=head(x,a)
    torch.testing.assert_close(result[:2],a[:2].clamp(-1,1),rtol=0,atol=0)
    torch.testing.assert_close(result[2:],a[2:],rtol=0,atol=0)
    # Protected commands require the legacy consumer, not globally wider clip.
    executed=execute_authority_action(x,result)
    torch.testing.assert_close(executed,a.clamp(-1,1),rtol=0,atol=0)

def test_positive_branch_does_not_touch_negative_and_target_has_declared_margin():
    x,a=inputs();head=PositionAuthorityHead()
    with torch.no_grad():
        for expert in head.positive.experts:expert[-1].bias.fill_(100.)
    out=head(x,a)
    assert abs(float(out[0,3].detach()*.22)-.37)<1e-6
    assert float(out[0,2].detach())<=1 and float(out[0,5].detach())<=1
    torch.testing.assert_close(out[1],a[1].clamp(-1,1))
    torch.testing.assert_close(out[2:],a[2:])
    assert out[0,1]==a[0,1] and out[0,4]==a[0,4]

def test_saturated_old_leg_has_nonzero_new_gradient():
    x,a=inputs();head=PositionAuthorityHead()
    head(x,a)[0,3].backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.positive.parameters())
    assert all(p.grad is None or p.grad.abs().sum()==0 for p in head.negative.parameters())

def test_real_environment_consumer_matches_head_contract():
    import ast
    from types import SimpleNamespace
    path=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'
    tree=ast.parse(path.read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_compute_policy_action')
    scope={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),scope)
    x,a=inputs()
    env=SimpleNamespace(cfg=SimpleNamespace(action_clip=1.,policy_outer_feedback_enabled=False,position_authority_enabled=True),
                        _position_authority_observation=x,_last_raw_policy_actions=torch.zeros_like(a))
    for name in ['_last_speed_feedback_action','_last_highspeed_propulsion_action','_last_pitch_feedback_action',
                 '_last_pitch_emergency_feedback_action','_last_yaw_feedback_action','_last_action_feedback_delta']:
        setattr(env,name,torch.zeros(len(x)))
    torch.manual_seed(235)
    for _ in range(20):
        actions=torch.randn_like(a)*3
        actual=scope['_compute_policy_action'](env,actions)
        torch.testing.assert_close(actual,execute_authority_action(x,actions),rtol=0,atol=0)

    # The same opt-in range must reach the real consumer, not only the actor.
    env.cfg.position_authority_speed_plateau_end=1.5
    x[:,7]=torch.tensor([.5,.525,.625,.75,.85,.75])
    x[:,8]=torch.tensor([1.,1.,1.,-1.,1.,.3])
    actions=torch.full_like(a,1.4)
    actual=scope['_compute_policy_action'](env,actions)
    torch.testing.assert_close(actual,execute_authority_action(x,actions,1.5),rtol=0,atol=0)
    assert actual[2,3]==1.4 and actual[3,3]==1.4
    assert actual[4,3]==1.0 and actual[5,3]==1.0
    assert torch.all(actual[:,[2,5]]<=1.)

def test_checkpoint_speed_support_opt_in_preserves_old_commands():
    from copy import deepcopy
    from rmuc_local_authority import LocalAuthorityCorrection
    torch.manual_seed(31)
    head=PositionAuthorityHead();head.local_correction=LocalAuthorityCorrection([1.,-1.,1.4,-1.4])
    with torch.no_grad():
        for p in head.parameters():p.uniform_(-.2,.2)
    extended=deepcopy(head);extended.register_buffer('speed_plateau_end',torch.tensor(1.5))
    x=torch.randn(128,25);x[:,7]=.5;x[:,8]=1.4-x[:,2]/.25
    actions=torch.randn(128,6)
    torch.testing.assert_close(head(x,actions),extended(x,actions),rtol=0,atol=0)
    x[:,7]=.75;x[:,8]=.3-x[:,2]/.25
    torch.testing.assert_close(head(x,actions),extended(x,actions),rtol=0,atol=0)
    x[:,7]=.625;x[:,8]=1.-x[:,2]/.25
    assert head.local_correction.gates(x).max()==0
    assert extended.local_correction.gates(x,1.5).max()>0.99
    assert not torch.equal(head(x,actions),extended(x,actions))

def test_authority_speed_contract_rejects_invalid_and_restores_buffer():
    from rmuc_position_authority import checkpoint_authority_speed_plateau_end
    from rmuc_command_adapter import install_rmuc_position_authority
    import pytest
    assert checkpoint_authority_speed_plateau_end({})==1.
    key='rmuc_position_authority.speed_plateau_end'
    for value in [float('nan'),float('inf'),.9,1.6]:
        with pytest.raises(ValueError):checkpoint_authority_speed_plateau_end({key:value})
    actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_turn_adapter=torch.nn.Identity()
    state={'rmuc_position_authority.positive.router.0.weight':torch.zeros(1),key:torch.tensor(1.5)}
    head=install_rmuc_position_authority(actor,state)
    assert head.speed_plateau_end.item()==1.5
