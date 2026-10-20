import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_state_router import IsolatedTurnStateRouter
from rmuc_position_authority import PositionAuthorityHead,execute_authority_action

def setup():
    torch.manual_seed(13)
    r=IsolatedTurnStateRouter(extended_turn_1p0=True,e2e_turn_1p0=True)
    r.register_buffer('turn_speed_plateau_end',torch.tensor(1.5));r.install_speed_extensions()
    x=torch.zeros(4,25);x[:,7]=torch.tensor([.5,.75,.75,.75]);x[:,8]=torch.tensor([1.4,-1.4,1.4,.1])
    r.install_post_authority_extensions();r.requires_grad_(False)
    for n,m in r.speed_extensions.items():
        if 'positive' in n:m.requires_grad_(True)
    h=PositionAuthorityHead();h.register_buffer('speed_plateau_end',torch.tensor(1.5));h.requires_grad_(False)
    return r,h,x

def test_zero_initialization_and_postclip_leg_gradient_and_retention():
    r,h,x=setup();before=r(x).detach();reference=torch.zeros(4,6);reference[:,3]=1.2
    base=h(x,reference+before).detach();assert torch.equal(r.post_authority_correction(x),torch.zeros(4,6))
    opt=torch.optim.Adam([p for p in r.parameters() if p.requires_grad],lr=.01)
    out=execute_authority_action(x,base+r.post_authority_correction(x),1.5)
    out[2,3].backward()
    assert sum(float(p.grad.abs().sum()) for p in r.parameters() if p.grad is not None)>0
    opt.step();after=r.post_authority_correction(x)
    assert after[2,3]<0
    assert torch.equal(after[[0,1,3]],torch.zeros(3,6))
    assert torch.equal(before,r(x))  # Frozen pre-clip reference does not move.
    assert all(p.grad is None for p in r.post_authority_anchors.parameters())

def test_postclip_correction_and_consumer_remain_bounded():
    r,h,x=setup()
    with torch.no_grad():
        for n,m in r.speed_extensions.items():
            if 'positive' in n:
                for expert in m.experts:expert[-1].bias.fill_(100.)
    d=r.post_authority_correction(x);assert (d.abs()<=r.post_authority_budget).all()
    out=execute_authority_action(x,torch.full((4,6),2.)+d,1.5)
    assert (out[:,[1,2,4,5]].abs()<=1).all()
    assert (out[:,[0,3]].abs()<=1+.15/.22).all()

def test_migration_budget_rejected():
    for b in [[.4]*6,[.1]*6,[float('nan')]*6]:
        r=IsolatedTurnStateRouter(extended_turn_1p0=True);r.install_speed_extensions()
        with pytest.raises(ValueError):r.install_post_authority_extensions(b)
