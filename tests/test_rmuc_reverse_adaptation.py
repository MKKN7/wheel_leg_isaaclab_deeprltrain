import copy
import pytest
import torch
from rmuc_physics_adaptation import PhysicsAdaptation
from rmuc_physics_selection import state_network_gate
from rmuc_training_contract import validate_trainable_target_gate
from rmuc_command_adapter import install_rmuc_physics_adaptation


def test_reverse_branch_zero_init_freeze_and_executed_gate_gradient():
    torch.manual_seed(53)
    head=PhysicsAdaptation()
    with torch.no_grad():head.network[-1].weight.normal_(0,.1)
    x=torch.randn(10,25)*.1
    x[:,7]=torch.tensor([-.5,-.25,-.1,-.01,0.,.05,.5,1.5,-.25,-.7])*.5
    x[:,2]=0;x[:,8]=torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,.3,0.])
    before=head.correction(x).detach().clone()
    head.install_reverse_network([.15,.15,.25,.15,.15,.25],[-.7,-.5,-.02,0.])
    assert torch.equal(before,head.correction(x))
    head.requires_grad_(False);head.reverse_network.requires_grad_(True)
    frozen={k:v.clone() for k,v in head.state_dict().items() if not k.startswith('reverse_network.')}
    opt=torch.optim.Adam(head.reverse_network.parameters(),lr=.01)
    for _ in range(3):
        opt.zero_grad();loss=(head.correction(x)[:3]-.2).square().mean();loss.backward();opt.step()
    after=head.correction(x);assert not torch.equal(before[:3],after[:3])
    assert torch.equal(before[4:],after[4:])
    assert all(torch.equal(v,head.state_dict()[k]) for k,v in frozen.items())
    assert (after.abs()<=head.correction_bound()+1e-6).all()
    state={'rmuc_physics_adaptation.'+k:v for k,v in head.state_dict().items()}
    path='rmuc_physics_adaptation.reverse_network'
    assert torch.equal(state_network_gate(state,path,x),head.reverse_gate(x))
    for speed in [-.5,-.25,-.1]:
        assert validate_trainable_target_gate(state,speed,0,physics_only=True,physics_modules=[path])['passed']
    with pytest.raises(ValueError):validate_trainable_target_gate(state,.1,0,physics_only=True,physics_modules=[path])
    actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_position_authority=torch.nn.Identity()
    restored=install_rmuc_physics_adaptation(actor,state);restored.load_state_dict(head.state_dict(),strict=True)
    assert torch.equal(after,restored.correction(x))
    traced=torch.jit.trace(head,(x,torch.zeros(10,6)))
    torch.testing.assert_close(traced(x,torch.zeros(10,6)),after,atol=0,rtol=0)


@pytest.mark.parametrize('budget,domain', [([.6]*6,[-.7,-.5,-.02,0]),([.1]*6,[-.7,-.5,0,.1]),([.1]*6,[-.7,-.5,-.5,-.5]),([float('nan')]*6,[-.7,-.5,-.02,0])])
def test_invalid_reverse_contract_fails_closed(budget,domain):
    with pytest.raises(ValueError):PhysicsAdaptation().install_reverse_network(budget,domain)
