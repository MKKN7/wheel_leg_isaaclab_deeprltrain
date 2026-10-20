import copy
import pytest
import torch
from rmuc_physics_adaptation import PhysicsAdaptation
from rmuc_physics_selection import network_gate, state_network_gate

def test_zero_migration_and_independent_capacity():
    torch.manual_seed(61)
    source = PhysicsAdaptation()
    with torch.no_grad():
        source.network[-1].bias.fill_(2.)
    head = copy.deepcopy(source)
    head.install_alignment_network([.4,.4,.35,.4,.4,.35],[.15,.2,.3,.35])
    x=torch.randn(128,25);x[:,7]=.125;x[:,2]=0;x[:,8]=0
    assert torch.equal(source.correction(x),head.correction(x))
    with torch.no_grad():head.alignment_network[-1].bias[0]=1.
    delta=head.correction(x)-source.correction(x)
    assert (delta[:,0]>.15).all()  # inherited .15 envelope cannot eat this correction
    assert (head.correction(x).abs()<=head.correction_bound()+1e-6).all()
    head.alignment_network.requires_grad_(True)
    g=torch.autograd.grad(head.correction(x)[:,0].mean(),head.alignment_network[-1].bias)[0]
    assert g[0]>0

def test_turns_other_speeds_and_checkpoint_gate():
    head=PhysicsAdaptation();head.install_alignment_network([.4]*6,[.15,.2,.3,.35])
    x=torch.zeros(6,25);x[:,7]=torch.tensor([.25,.25,.25,0.,.5,-.25])*.5;x[:,8]=torch.tensor([0.,.1,-.1,0.,0.,0.])
    assert torch.equal(head.alignment_gate(x),torch.tensor([1.,0.,0.,0.,0.,0.]))
    state={'rmuc_physics_adaptation.'+k:v for k,v in head.state_dict().items()}
    assert torch.equal(state_network_gate(state,'rmuc_physics_adaptation.alignment_network',x),network_gate(head,'alignment_network',x))
    with torch.no_grad():head.alignment_network[-1].bias.fill_(1.)
    assert torch.equal(head.correction(x)[1:],head._legacy_correction(x)[1:])

def test_invalid_budget_rejected():
    for b in ([float('nan')]*6, [.6]*6, [-.1]*6):
        with pytest.raises(ValueError):PhysicsAdaptation().install_alignment_network(b,[.15,.2,.3,.35])


def test_straight_extension_preserves_crawl_turns_and_excludes_clipped_alias():
    head=PhysicsAdaptation();head.install_alignment_network([.4,.4,.35,.4,.4,.35],[.15,.2,.3,.35])
    with torch.no_grad():head.alignment_network[-1].bias.fill_(.3)
    original=copy.deepcopy(head);head.install_straight_extension([.4,.4,.35,.4,.4,.35],[.35,.5,1.5,1.7])
    x=torch.zeros(8,25);x[:,7]=torch.tensor([.225,.25,.275,.35,.5,1.,1.5,1.])* .5
    x[-1,2]=-.5;x[-1,8]=2.  # clipped yaw-error can falsely decode zero
    assert torch.equal(head.correction(x),original.correction(x))
    expected=torch.tensor([0.,0.,0.,0.,1.,1.,1.,0.])
    assert torch.equal(head.straight_extension_gate(x),expected)
    with torch.no_grad():head.straight_extension_network[-1].bias.fill_(1.)
    assert torch.equal(head.correction(x)[expected==0],original.correction(x)[expected==0])
    assert (head.correction(x).abs()<=head.correction_bound()+1e-6).all()
    assert torch.equal(head.correction_bound(),head.budget+head.alignment_budget)
    state={'rmuc_physics_adaptation.'+k:v for k,v in head.state_dict().items()}
    assert torch.equal(state_network_gate(state,'rmuc_physics_adaptation.straight_extension_network',x),expected)
    x[:,2]=0;x[:,8]=.01
    assert not head.straight_extension_gate(x).any()


def test_straight_extension_rejects_overlapping_domains():
    head=PhysicsAdaptation();head.install_alignment_network([.4]*6,[.15,.2,.3,.35])
    with pytest.raises(ValueError):head.install_straight_extension([.4]*6,[.3,.5,1.5,1.7])


def test_optin_bridge_fills_transition_without_changing_certified_anchors_or_turns():
    head=PhysicsAdaptation();head.install_alignment_network([.4]*6,[.15,.2,.3,.35])
    head.install_straight_extension([.3]*6,[.35,.5,1.5,1.7])
    with torch.no_grad():
        head.alignment_network[-1].bias.fill_(.4)
        head.straight_extension_network[-1].bias.fill_(-.7)
    old=copy.deepcopy(head);head.install_straight_bridge()
    x=torch.zeros(1001,25);x[:,7]=torch.linspace(.3,.5,len(x))*.5
    low=head.alignment_gate(x);high=head.straight_extension_gate(x)
    assert torch.allclose(low+high,torch.ones_like(low),atol=1e-6)
    assert torch.all(head.correction(x).abs()<=head.correction_bound()+1e-6)
    # Continuity is measured in the actual neural correction, not only its gate.
    assert (head.correction(x)[1:]-head.correction(x)[:-1]).abs().max()<.005
    protected=torch.zeros(10,25);protected[:,7]=torch.tensor([0,.1,.225,.25,.275,.3,.5,.75,1.,1.5])*.5
    assert torch.equal(head.correction(protected),old.correction(protected))
    x[:,8]=.01
    assert torch.equal(head.correction(x),old.correction(x))
    x[:,8]=0;x[:,7]=.35*.5
    assert (head.alignment_gate(x)>0).all() and (head.straight_extension_gate(x)>0).all()
    state={'rmuc_physics_adaptation.'+k:v for k,v in head.state_dict().items()}
    assert torch.equal(state_network_gate(state,'rmuc_physics_adaptation.alignment_network',x),head.alignment_gate(x))
    assert torch.equal(state_network_gate(state,'rmuc_physics_adaptation.straight_extension_network',x),head.straight_extension_gate(x))


def test_explicit_idle_domain_opens_zero_without_changing_prior_plateau():
    old=PhysicsAdaptation();old.install_alignment_network([.4]*6,[.15,.2,.3,.35])
    with torch.no_grad():old.alignment_network[-1].bias.fill_(.3)
    new=copy.deepcopy(old);new.alignment_domain.copy_(torch.tensor([-.05,0.,.3,.35]))
    reconstructed=PhysicsAdaptation();reconstructed.install_alignment_network([.4]*6,[-.05,0.,.3,.35])
    x=torch.zeros(6,25);x[:,7]=torch.tensor([-.1,0,.05,.15,.225,.275])*.5
    assert torch.equal(reconstructed.alignment_gate(x),torch.tensor([0.,1.,1.,1.,1.,1.]))
    assert torch.equal(old.correction(x)[[0,4,5]],new.correction(x)[[0,4,5]])
    x[:,8]=.01
    assert torch.equal(old.correction(x),new.correction(x))
    with pytest.raises(ValueError):PhysicsAdaptation().install_alignment_network([.4]*6,[-.1,0.,.3,.35])
