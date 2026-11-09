import copy,sys
from pathlib import Path
import pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_state_router import IsolatedTurnStateRouter

def make():
    r=IsolatedTurnStateRouter(extended_turn_1p0=True,extended_turn_0p8=True,e2e_turn_1p0=True)
    r.register_buffer('turn_speed_plateau_end',torch.tensor(1.5));r.install_speed_extensions()
    for m in r.modules():
        if isinstance(m,torch.nn.Linear):torch.nn.init.constant_(m.weight,.01);torch.nn.init.constant_(m.bias,.03)
    return r

def test_router_support_preserves_domains_and_extends_learned_path():
    a=make();b=copy.deepcopy(a);b.install_forward_turn_support([2.,2.1])
    x=torch.zeros(6,25);x[:,7]=torch.tensor([-.4,1.,1.5,1.6,1.8,2.])*.5;x[:,8]=.8
    assert torch.equal(a(x[:3]),b(x[:3]));assert (b.extended_speed_support(x[3:])==1).all()
    assert a(x[4:]).abs().max()==0 and b(x[4:]).abs().max()>0
    y=x.clone();y[:,8]=0.;assert torch.equal(a(y),b(y))
    y[:,8]=2.5;assert torch.equal(a(y),b(y))
    assert (b(x).abs()<=b.correction_bound(x)+1e-6).all()
    z=x.clone();z[:,8]=2.;assert torch.equal(a(z),b(z))

@pytest.mark.parametrize('domain',[[1.4,2.],[2.,2.],[2.,2.6],[float('nan'),2.],[2.]])
def test_reject_invalid_support(domain):
    with pytest.raises(ValueError):make().install_forward_turn_support(domain)
