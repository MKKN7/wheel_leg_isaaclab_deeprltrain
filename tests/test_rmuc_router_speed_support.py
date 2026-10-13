import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_state_router import IsolatedTurnStateRouter
from rmuc_command_adapter import install_rmuc_state_router

def test_support_keeps_old_anchors_and_opens_both_new_directions():
    a=IsolatedTurnStateRouter(extended_turn_1p0=True,e2e_turn_1p0=True)
    b=IsolatedTurnStateRouter(extended_turn_1p0=True,e2e_turn_1p0=True)
    with torch.no_grad():
        for p in a.parameters():p.uniform_(-.2,.2)
    b.load_state_dict(a.state_dict());b.register_buffer('turn_speed_plateau_end',torch.tensor(1.5))
    x=torch.zeros(8,25);x[:,7]=torch.tensor([.5,.5,.75,.75,.625,.625,.85,.85]);x[:,8]=torch.tensor([1.,-1.,.3,-.3,1.,-1.,1.,-1.])
    torch.testing.assert_close(a(x)[:4],b(x)[:4],rtol=0,atol=0)
    torch.testing.assert_close(a(x)[6:],b(x)[6:],rtol=0,atol=0)
    pos,neg=b.extended_turn_eligibility(x)
    assert pos[4]==1 and neg[5]==1
    assert not torch.equal(a(x)[4:6],b(x)[4:6])

def test_restore_and_invalid_support():
    r=IsolatedTurnStateRouter(extended_turn_1p0=True)
    s={'rmuc_state_router.'+k:v for k,v in r.state_dict().items()}
    key='rmuc_state_router.turn_speed_plateau_end'
    def actor():
        a=torch.nn.Module();a.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));a.rmuc_turn_adapter=torch.nn.Identity();return a
    for v in [float('nan'),float('inf'),.9,1.6]:
        with pytest.raises(ValueError):install_rmuc_state_router(actor(),dict(s,**{key:torch.tensor(v)}))
    out=install_rmuc_state_router(actor(),dict(s,**{key:torch.tensor(1.5)}))
    assert out.turn_speed_plateau_end.item()==1.5

def test_copied_extension_isolates_old_speed_and_other_direction_after_optimizer_step():
    torch.manual_seed(5)
    r=IsolatedTurnStateRouter(extended_turn_1p0=True,e2e_turn_1p0=True)
    r.register_buffer('turn_speed_plateau_end',torch.tensor(1.5))
    with torch.no_grad():
        for p in r.parameters():p.uniform_(-.15,.15)
    x=torch.zeros(6,25);x[:,7]=torch.tensor([.5,.5,.625,.625,.75,.75]);x[:,8]=torch.tensor([1.,-1.,1.,-1.,.3,-.3])
    before=r(x).detach();r.install_speed_extensions()
    torch.testing.assert_close(r(x),before,rtol=0,atol=0)
    r.requires_grad_(False)
    selected=[v for k,v in r.speed_extensions.items() if 'positive' in k]
    for m in selected:m.requires_grad_(True)
    opt=torch.optim.Adam([p for m in selected for p in m.parameters()],lr=.01)
    loss=-r(x)[2,2];loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for m in selected for p in m.parameters())
    opt.step();after=r(x).detach()
    torch.testing.assert_close(after[[0,1,3,4,5]],before[[0,1,3,4,5]],rtol=0,atol=0)
    assert after[2,2]!=before[2,2]
    assert all(not p.requires_grad for n,p in r.named_parameters() if not n.startswith('speed_extensions.'))

def test_speed_target_gate_rejects_protected_command():
    from rmuc_training_contract import validate_trainable_target_gate
    state={'rmuc_state_router.speed_extensions.e2e_positive_1p0.experts.0.0.weight':torch.zeros(1),
           'rmuc_state_router.turn_speed_plateau_end':torch.tensor(1.5)}
    r=validate_trainable_target_gate(state,1.25,1.,router_speed_direction='positive')
    assert r['gates']==[1.,0.]
    with pytest.raises(ValueError):validate_trainable_target_gate(state,1.,1.,router_speed_direction='positive')

def test_signed_continuous_speed_box():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/command_contract.py'
    spec=importlib.util.spec_from_file_location('contract',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    x=m.sample_speed_extension_commands(4096,[1.05,1.3,.6,1.4],device='cpu')
    assert ((x[:,0]>=1.05)&(x[:,0]<=1.3)).all() and (x[:,1]==0).all()
    assert ((x[:,2]>=.6)&(x[:,2]<=1.4)).all()
    assert torch.unique(x[:,0]).numel()>4000
    for bounds in [[1.,1.6,.6,1.4],[1.,1.5,-.6,.6],[1.,1.5,.6,float('nan')]]:
        with pytest.raises(ValueError):m.sample_speed_extension_commands(1,bounds,device='cpu')

def test_shifted_handover_keeps_anchor_and_trained_upper_domain():
    from copy import deepcopy
    r=IsolatedTurnStateRouter(extended_turn_1p0=True,e2e_turn_1p0=True)
    r.register_buffer('turn_speed_plateau_end',torch.tensor(1.5));r.install_speed_extensions()
    with torch.no_grad():
        for p in r.speed_extensions.parameters():p.uniform_(-.2,.2)
    shifted=deepcopy(r);shifted.register_buffer('speed_extension_blend_bounds',torch.tensor([1.05,1.15]))
    x=torch.zeros(4,25);x[:,7]=torch.tensor([.525,.55,.575,.625]);x[:,8]=1.
    assert shifted.speed_extension_gate(x)[0]==0
    assert 0<shifted.speed_extension_gate(x)[1]<1
    torch.testing.assert_close(r(x)[2:],shifted(x)[2:],rtol=0,atol=0)
    base=deepcopy(r);del base.speed_extensions
    torch.testing.assert_close(base(x)[0],shifted(x)[0],rtol=0,atol=0)

def test_extension_sampling_rehearses_exact_anchors_and_keeps_continuous_samples():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/command_contract.py'
    spec=importlib.util.spec_from_file_location('contract',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    anchors=[[1.05,1.],[1.05,1.4],[1.25,.6],[1.25,1.],[1.5,1.]]
    torch.manual_seed(4)
    x=m.sample_speed_extension_commands(100,[1.05,1.3,.6,1.4],device='cpu',anchors=anchors,anchor_fraction=.5)
    hits=torch.zeros(100,dtype=torch.bool)
    for v,w in anchors:
        match=(x[:,0]==v)&(x[:,2]==w);assert match.sum()==10;hits|=match
    assert hits.sum()==50 and (x[:,1]==0).all()
    with pytest.raises(ValueError):m.sample_speed_extension_commands(5,[1.05,1.3,.6,1.4],device='cpu',anchors=[[1.2,-1]],anchor_fraction=.5)

def test_single_environment_resets_do_not_eliminate_anchor_rehearsal():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/command_contract.py'
    spec=importlib.util.spec_from_file_location('contract',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    torch.manual_seed(17)
    draws=torch.cat([m.sample_speed_extension_commands(1,[1.05,1.3,.6,1.4],device='cpu',anchors=[[1.5,1.]],anchor_fraction=.5) for _ in range(2000)])
    rate=(draws[:,0]==1.5).float().mean().item()
    assert .46<rate<.54

def test_low_yaw_handover_preserves_high_yaw_and_old_speed():
    from copy import deepcopy
    from rmuc_training_contract import validate_trainable_target_gate
    r=IsolatedTurnStateRouter(extended_turn_1p0=True,e2e_turn_1p0=True)
    r.register_buffer('turn_speed_plateau_end',torch.tensor(1.5));r.install_speed_extensions()
    r.register_buffer('speed_extension_blend_bounds',torch.tensor([1.05,1.15]))
    new=deepcopy(r);new.register_buffer('low_yaw_speed_blend',torch.tensor([1.,1.05,.6001,.8]))
    x=torch.zeros(6,25);x[:,7]=torch.tensor([.5,.525,.55,.525,.625,.525]);x[:,8]=torch.tensor([.6,.6,-.6,1.4,.6,.7])
    a=r.speed_extension_gate(x);b=new.speed_extension_gate(x)
    assert b[0]==a[0]==0 and b[1]==b[2]==1
    assert b[3]==a[3] and b[4]==a[4] and 0<b[5]<1
    state={'rmuc_state_router.'+k:v for k,v in new.state_dict().items()}
    gate=validate_trainable_target_gate(state,1.05,.6,router_speed_direction='negative')
    assert gate['gates']==[0.,1.]
