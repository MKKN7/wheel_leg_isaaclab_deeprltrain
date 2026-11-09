import copy,sys
from pathlib import Path
import pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_physics_adaptation import PhysicsAdaptation
from rmuc_command_adapter import install_rmuc_physics_adaptation,RmucTurnResidualAdapter
from rmuc_physics_selection import network_gate,state_network_gate

def restore(module):
    a=torch.nn.Module();a.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));a.rmuc_position_authority=torch.nn.Identity()
    state={'rmuc_physics_adaptation.'+k:v for k,v in module.state_dict().items()}
    out=install_rmuc_physics_adaptation(a,state);out.load_state_dict(module.state_dict(),strict=True)
    return out,state

def test_forward_turn_extension_keeps_old_domains_and_budget():
    torch.manual_seed(11860);inner=PhysicsAdaptation();inner.network[-1].bias.data.fill_(2)
    outer=PhysicsAdaptation(inner,update_domain=[.25,.45,.35,.5]);old=copy.deepcopy(outer)
    inner.install_legacy_forward_turn_extension([2.,2.1])
    x=torch.randn(4096,25)*.2;x[:,7]=torch.linspace(-2.,3.,len(x))*.5;x[:,8]=.8-x[:,2]/.25
    before=old.correction(x);after=outer.correction(x);keep=x[:,7]<=.75
    assert torch.equal(before[keep],after[keep])
    assert (after.abs()<=outer.budget+1e-6).all()
    active=(x[:,7]>=.9)&(x[:,7]<=1.)&(x[:,8].abs()<1.999);assert after[active,2].min()>.3
    clipped=x[:,8].abs()>=1.999;assert torch.equal(before[clipped],after[clipped])
    straight=x.clone();straight[:,8]=-straight[:,2]/.25
    assert torch.equal(old.correction(straight),outer.correction(straight))
    large=x.clone();large[:,8]=2.5-large[:,2]/.25
    assert torch.equal(old.correction(large),outer.correction(large))
    z=torch.zeros(3,25);z[:,7]=torch.tensor([1.5,1.8,2.1])*.5;z[:,8]=.8
    torch.testing.assert_close(inner.update_gate(z),torch.tensor([1.,1.,0.]),atol=1e-6,rtol=0)
    assert not outer.update_gate(z).any()
    restored,state=restore(outer);assert torch.equal(restored.correction(x),after)
    assert torch.equal(state_network_gate(state,'rmuc_physics_adaptation.frozen_source.network',z),network_gate(inner,'network',z))

def test_forward_leg_response_is_bounded_and_wheels_unchanged():
    m=PhysicsAdaptation();m.install_forward_base_leg_response(1.2,torch.zeros(6),[1.6,1.7,2.,2.1],.03)
    x=torch.zeros(3000,25);x[:,7]=torch.linspace(-2.,3.,len(x))*.5;b=torch.randn(len(x),6)
    out=m.forward_base_leg_response(x,b)
    assert torch.equal(out[:,[2,5]],b[:,[2,5]])
    keep=(x[:,7]<=.8)|(x[:,7]>=1.05);assert torch.equal(out[keep],b[keep])
    assert (out-b).abs().max()<.030001
    turn=x.clone();turn[:,8]=.1;assert torch.equal(m.forward_base_leg_response(turn,b),b)
    restored,_=restore(m);assert torch.equal(restored.forward_base_leg_response(x,b),out)

@pytest.mark.parametrize('gain,domain,limit',[(2.,[1.6,1.7,2.,2.1],.03),(1.2,[1.,1.7,2.,2.1],.03),(1.2,[1.6,1.7,2.,2.1],.1)])
def test_reject_unbounded_forward_leg_response(gain,domain,limit):
    with pytest.raises(ValueError):PhysicsAdaptation().install_forward_base_leg_response(gain,torch.zeros(6),domain,limit)

def test_optin_expert_speed_extension_preserves_original_and_roundtrip():
    a=RmucTurnResidualAdapter(25,6);x=torch.zeros(7,25)
    x[:,7]=torch.tensor([-1.,1.,1.5,1.6,1.8,2.,2.1])*.5;x[:,8]=.8
    old=a.effective_expert_gate(x);assert torch.equal(old,a.expert_gate(x))
    a.install_expert_speed_extension([2.,2.1]);new=a.effective_expert_gate(x)
    assert torch.equal(new[:3],old[:3]);torch.testing.assert_close(new[3:],torch.tensor([1.,1.,1.,0.]),atol=1e-6,rtol=0)
    y=x.clone();y[:,8]=0.;assert torch.equal(a.effective_expert_gate(y),a.expert_gate(y))
    y[:,8]=2.5;assert torch.equal(a.effective_expert_gate(y),a.expert_gate(y))
    b=RmucTurnResidualAdapter(25,6);b.load_state_dict(a.state_dict(),strict=True)
    assert torch.equal(b.effective_expert_gate(x),new)

def test_invalid_saved_expert_extension_fails_closed():
    a=RmucTurnResidualAdapter(25,6);state=a.state_dict();state['expert_speed_extension']=torch.tensor([2.1,2.])
    with pytest.raises(ValueError):a.load_state_dict(state,strict=True)

def test_scoped_upper_gain_preserves_both_verified_endpoints():
    a=PhysicsAdaptation();a.install_forward_base_leg_response(1.2,torch.randn(6),[1.6,1.7,1.9,2.],.03)
    b=copy.deepcopy(a);b.forward_base_leg_gain.fill_(1.3)
    m=copy.deepcopy(a);m.install_forward_base_leg_upper_gain(1.3,[1.8,1.9])
    x=torch.zeros(4096,25);x[:,7]=torch.linspace(-2,3.2,len(x))*.5;actions=torch.randn(len(x),6)
    lo=x[:,7]<=.9;hi=x[:,7]>=.95
    out=m.forward_base_leg_response(x,actions)
    assert torch.equal(out[lo],a.forward_base_leg_response(x,actions)[lo])
    assert torch.equal(out[hi],b.forward_base_leg_response(x,actions)[hi])
    assert torch.equal(out[:,[2,5]],actions[:,[2,5]])
    assert (out-actions).abs().max()<.030001
    restored,_=restore(m);assert torch.equal(restored.forward_base_leg_response(x,actions),out)
    x[:,8]=.8;assert torch.equal(m.forward_base_leg_response(x,actions),actions)

@pytest.mark.parametrize('gain,domain',[(2.,[1.8,1.9]),(1.3,[1.7,2.]),(1.3,[1.8,1.8]),(1.3,[float('nan'),1.9])])
def test_invalid_upper_gain(gain,domain):
    m=PhysicsAdaptation();m.install_forward_base_leg_response(1.2,torch.zeros(6),[1.6,1.7,1.9,2.],.03)
    with pytest.raises(ValueError):m.install_forward_base_leg_upper_gain(gain,domain)
