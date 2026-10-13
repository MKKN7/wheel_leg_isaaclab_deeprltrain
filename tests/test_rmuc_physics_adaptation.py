import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('physics_adaptation',Path(__file__).resolve().parents[1]/'scripts/rsl_rl/rmuc_physics_adaptation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_reference_update_preserves_crawl_negative_straight_and_fast_turns():
    head=m.PhysicsAdaptation(positive_update_domain=[.3,.5,1.5,1.7,.45,.6],
        reference_update_domain=[.3,.5,1.5,1.7,0.,.05,.35,.45])
    obs=torch.zeros(8,25)
    obs[:,7]=torch.tensor([.25,1.,1.,1.,1.,.5,1.5,1.])*.5
    obs[:,8]=torch.tensor([.1,-.1,0.,1.4,.45,.1,.3,.2])
    before=head.correction(obs).detach().clone()
    with torch.no_grad():head.reference_network[-1].bias.fill_(1.)
    after=head.correction(obs)
    assert torch.equal(before[:5],after[:5])
    assert (after[5:]-before[5:]).abs().max()>0
    assert (after.abs()<=head.budget+1e-7).all()
    traced=torch.jit.trace(head,(obs,torch.zeros(8,6)))
    torch.testing.assert_close(traced(obs,torch.zeros(8,6)),head(obs,torch.zeros(8,6)),atol=0,rtol=0)
    # Values approach zero continuously at the positive-yaw support edge.
    edge=torch.zeros(3,25);edge[:,7]=.5;edge[:,8]=torch.tensor([.45-1e-5,.45,.45+1e-5])
    assert head.correction(edge).abs().max()<1e-6


def test_reference_scope_requires_reference_network_and_ordered_bounds():
    import pytest
    with pytest.raises(ValueError):
        m.PhysicsAdaptation(reference_update_domain=[.3,.5,1.5,1.7,0.,.05,.35,.45])
    with pytest.raises(ValueError):
        m.PhysicsAdaptation(positive_update_domain=[.3,.5,1.5,1.7,.45,.6],
            reference_update_domain=[.5,.3,1.5,1.7,0.,.05,.35,.45])


def test_negative_reference_scope_preserves_straight_positive_and_faster_commands():
    head=m.PhysicsAdaptation(m.PhysicsAdaptation(),update_domain=[.25,.45,.35,.5],
        positive_update_domain=[0.,.1,.3,.45,0.,.05],positive_straight_enabled=0.,
        reference_update_domain=[0.,.1,.3,.45,-.2,-.15,-.05,0.])
    obs=torch.zeros(7,25);obs[:,7]=torch.tensor([.25,.25,.25,.25,.5,0.,.25])*.5
    obs[:,8]=torch.tensor([-.1,0.,.1,-.3,-.1,-.1,-.05])
    before=head.correction(obs).detach().clone()
    with torch.no_grad():head.reference_network[-1].bias.fill_(.2)
    after=head.correction(obs)
    assert torch.equal(before[1:6],after[1:6])
    assert not torch.equal(before[0],after[0])
    traced=torch.jit.trace(head,(obs,torch.zeros(7,6)))
    torch.testing.assert_close(traced(obs,torch.zeros(7,6)),head(obs,torch.zeros(7,6)),atol=0,rtol=0)


def test_positive_training_preserves_negative_straight_and_large_turns():
    head=m.PhysicsAdaptation(m.PhysicsAdaptation(),update_domain=[1.5,1.7,.45,.6],
        positive_update_domain=[.3,.5,1.5,1.7,.45,.6])
    obs=torch.zeros(7,25)
    obs[:,7]=torch.tensor([1.,1.,1.,1.,0.,.5,1.5])*.5
    obs[:,8]=torch.tensor([-.3,0.,1.4,.5,.3,-.3,.5])
    before=head.correction(obs).detach().clone()
    opt=torch.optim.Adam(head.network.parameters(),lr=.01)
    for _ in range(3):
        opt.zero_grad();loss=(head.correction(obs)[[1,3,6]]-.1).square().mean();loss.backward();opt.step()
    after=head.correction(obs)
    assert torch.equal(before[[0,2,4,5]],after[[0,2,4,5]])
    assert not torch.equal(before[[1,3,6]],after[[1,3,6]])
    assert all(not p.requires_grad for p in head.reference_network.parameters())
    assert (after.abs()<=head.budget+1e-7).all()


def test_scoped_network_preserves_protected_outputs_bounds_and_trace():
    head=m.PhysicsAdaptation(scoped_network_domain=[.3,.45,.55,.75,.02,.1])
    with torch.no_grad():
        head.network[-1].bias.fill_(-2.)
        head.scoped_network[-1].bias.fill_(2.)
    obs=torch.zeros(7,25)
    obs[:,7]=torch.tensor([-.5,0.,.5,.5,.75,1.,.6])*.5
    obs[:,8]=torch.tensor([0.,0.,0.,.3,0.,-.2,0.])
    expected=head.update_gate(obs)[:,None]*head.budget*torch.tanh(head.network(obs))
    actual=head.correction(obs)
    protected=torch.tensor([0,1,3,4,5])
    assert torch.equal(actual[protected],expected[protected])
    assert actual[2,2]>0 and expected[2,2]<0
    assert (actual.abs()<=head.budget+1e-7).all()
    assert all(not p.requires_grad for p in head.scoped_network.parameters())
    source=torch.zeros(7,6)
    traced=torch.jit.trace(head,(obs,source))
    torch.testing.assert_close(traced(obs,source),head(obs,source),atol=0,rtol=0)

def test_positive_only_crawl_updates_preserve_negative_straight_and_fast():
    head=m.PhysicsAdaptation(m.PhysicsAdaptation(),update_domain=[.25,.45,.35,.5],
        positive_update_domain=[0.,.1,.3,.45,0.,.05],positive_straight_enabled=0.)
    obs=torch.zeros(7,25);obs[:,7]=torch.tensor([.25,.25,.25,.25,.25,.5,0.])*.5
    obs[:,8]=torch.tensor([-.3,-.001,0.,.1,.3,.3,.3])
    before=head.correction(obs).detach().clone()
    opt=torch.optim.Adam(head.network.parameters(),lr=.01)
    for _ in range(3):
        opt.zero_grad();(head.correction(obs)[3:5]-.1).square().mean().backward();opt.step()
    after=head.correction(obs)
    assert torch.equal(before[[0,1,2,5,6]],after[[0,1,2,5,6]])
    assert not torch.equal(before[3:5],after[3:5])
    traced=torch.jit.trace(head,(obs,torch.zeros(7,6)))
    torch.testing.assert_close(traced(obs,torch.zeros(7,6)),head(obs,torch.zeros(7,6)),atol=0,rtol=0)


def test_zero_initialization_preserves_raw_source_even_outside_action_limits():
    torch.manual_seed(263)
    head=m.PhysicsAdaptation();obs=torch.randn(512,25);source=torch.randn(512,6)*3
    assert torch.equal(head(obs,source),source)


def test_nonzero_correction_is_bounded_and_outside_domain_is_exact():
    head=m.PhysicsAdaptation()
    with torch.no_grad():head.network[-1].bias.fill_(10.)
    obs=torch.zeros(4,25);obs[:,7]=torch.tensor([.5,.85,.5,0.]);obs[:,8]=torch.tensor([1.,1.,4.2,0.])
    source=torch.randn(4,6)
    delta=head.correction(obs)
    assert (delta.abs()<=head.budget).all()
    assert (delta[:,[1,4]]==0).all()
    assert torch.equal(head(obs,source)[1:3],source[1:3])
    assert delta[0].abs().sum()>0 and delta[3].abs().sum()>0


def test_gradient_reaches_new_head_while_explicitly_frozen_source_stays_fixed():
    torch.manual_seed(264)
    source=torch.nn.Linear(25,6).requires_grad_(False)
    head=m.PhysicsAdaptation();optimizer=torch.optim.Adam(head.parameters(),lr=1e-4)
    before={k:v.clone() for k,v in source.state_dict().items()}
    obs=torch.zeros(16,25);obs[:,7]=.5;obs[:,8]=1.
    loss=(head(obs,source(obs))-torch.ones(16,6)).square().mean()
    loss.backward();optimizer.step()
    assert all(torch.equal(before[k],v) for k,v in source.state_dict().items())
    assert head.correction(obs).abs().sum()>0


def test_frozen_migration_starts_exact_and_never_expands_total_budget():
    old=m.PhysicsAdaptation()
    with torch.no_grad():old.network[-1].bias.fill_(2.)
    head=m.PhysicsAdaptation(old)
    obs=torch.randn(512,25);obs[:,7]=.5;obs[:,8]=1.-obs[:,2]/.25
    assert torch.equal(head.correction(obs),old.correction(obs))
    for value in [-10.,10.]:
        with torch.no_grad():head.network[-1].bias.fill_(value)
        assert (head.correction(obs).abs()<=head.budget+1e-7).all()
    assert all(not p.requires_grad for p in old.parameters())


def test_scoped_migration_learns_low_domain_and_preserves_large_turns():
    old=m.PhysicsAdaptation()
    with torch.no_grad():old.network[-1].bias.fill_(.3)
    head=m.PhysicsAdaptation(old,update_domain=[1.5,1.7,.45,.6])
    obs=torch.zeros(8,25)
    obs[:,7]=torch.tensor([0.,.25,.5,.75,.5,.5,.5,1.])
    obs[:,8]=torch.tensor([0.,-.3,.1,.45,.8,-1.,1.4,0.])
    assert torch.equal(head.correction(obs),old.correction(obs))
    assert (head.update_gate(obs[:4])==1).all()
    assert (head.update_gate(obs[4:])==0).all()
    optimizer=torch.optim.Adam(head.network.parameters(),lr=.001)
    before={k:v.clone() for k,v in old.state_dict().items()}
    for _ in range(3):
        loss=(head.correction(obs[:4])-torch.ones(4,6)*.1).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    assert not torch.equal(head.correction(obs[:4]),old.correction(obs[:4]))
    assert torch.equal(head.correction(obs[4:]),old.correction(obs[4:]))
    assert all(torch.equal(before[k],v) for k,v in old.state_dict().items())
    assert (head.correction(obs).abs()<=head.budget+1e-7).all()


def test_second_migration_keeps_one_shared_budget():
    seed=m.PhysicsAdaptation()
    first=m.PhysicsAdaptation(seed)
    second=m.PhysicsAdaptation(first,update_domain=[1.5,1.7,.45,.6])
    obs=torch.zeros(8,25);obs[:,7]=.5;obs[:,8]=.3
    with torch.no_grad():
        seed.network[-1].bias.fill_(4.)
        first.network[-1].bias.fill_(-4.)
        second.network[-1].bias.fill_(4.)
    assert (second.correction(obs).abs()<=second.budget+1e-7).all()
    assert all(not p.requires_grad for p in first.parameters())


def test_frozen_reference_local_update_exact_retention_and_bounds():
    h=m.PhysicsAdaptation(positive_update_domain=[.3,.5,1.5,1.7,.45,.6],
        reference_update_domain=[.3,.5,1.5,1.7,0.,.05,.35,.45],
        frozen_reference_update_domain=[1.35,1.45,1.55,1.65,-.29,-.24,-.16,-.11])
    x=torch.zeros(9,25);x[:,7]=torch.tensor([1.5,.5,1.5,1.5,.25,1.,1.5,1.4,1.6])*.5
    x[:,8]=torch.tensor([-.2,-.1,-.1,-.3,.1,-.2,.1,-.2,-.2])
    before=h.correction(x).detach().clone()
    with torch.no_grad():h.frozen_reference_network[-1].bias.fill_(2.)
    after=h.correction(x);gate=h.frozen_reference_gate(x)
    assert torch.equal(before[gate==0],after[gate==0])
    assert not torch.equal(before[0],after[0])
    assert (after.abs()<=h.budget+1e-7).all()
    assert not any(p.requires_grad for p in h.reference_anchor_network.parameters())
