import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_physics_adaptation import PhysicsAdaptation
from rmuc_physics_selection import state_network_gate

def test_stop_preserves_initial_policy_and_excludes_moving_and_turning():
    h=PhysicsAdaptation();x=torch.randn(12,25);x[:,2]=.1;x[:,7]=torch.tensor([0,.01,-.01,.049,-.049,.05,-.05,.1,-.4,2.5,0,0])*.5;x[:,8]=-x[:,2]/.25
    x[10,8]+=.3;x[11,8]=2
    before=h.correction(x).detach();h.install_stop_network([.08,.08,.2,.08,.08,.2]);assert torch.equal(before,h.correction(x))
    with torch.no_grad():h.stop_network[-1].bias.fill_(1)
    delta=h.correction(x)-before
    assert torch.equal(delta[5:],torch.zeros_like(delta[5:]))
    assert (delta[0].abs()>0).all();assert (delta.abs()<=h.stop_budget+1e-6).all()
    state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
    assert torch.equal(state_network_gate(state,'rmuc_physics_adaptation.stop_network',x),h.stop_gate(x))
    h2=PhysicsAdaptation();h2.install_stop_network(h.stop_budget);h2.load_state_dict(h.state_dict());assert torch.equal(h2.correction(x),h.correction(x))

def test_stop_invalid_and_duplicate():
    h=PhysicsAdaptation()
    for b in [[.21]*6,[float('nan')]*6,[-.1]*6,[.1]*5]:
        with pytest.raises(ValueError):h.install_stop_network(b)
    h.install_stop_network([.1]*6)
    with pytest.raises(ValueError):h.install_stop_network([.1]*6)

def test_stop_sequence_frozen_approach_is_explicit_and_bounded():
    from rmuc_training_contract import validate_trainable_sequence_gates
    h=PhysicsAdaptation();h.install_stop_network([.1]*6)
    state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
    kw=dict(physics_only=True,physics_modules=['rmuc_physics_adaptation.stop_network'])
    rows=validate_trainable_sequence_gates(state,[[0,0,0],[2,1,0],[4,0,0]],**kw)
    assert rows[1]['phase']=='frozen_teacher_approach' and rows[1]['gates']==[0]
    assert rows[2]['phase']=='trainable_stop' and rows[2]['gates']==[1]
    for seq in [[[0,1,0]],[[0,0,0],[2,1.6,0]],[[0,0,0],[2,.5,.1]],[[0,0,0],[2,-.1,0]]]:
        with pytest.raises(ValueError):validate_trainable_sequence_gates(state,seq,**kw)
