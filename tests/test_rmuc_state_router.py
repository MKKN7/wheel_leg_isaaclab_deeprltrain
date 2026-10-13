import importlib.util
from pathlib import Path
import torch
import pytest
p=Path(__file__).resolve().parents[1]/'scripts/rsl_rl/rmuc_state_router.py'
s=importlib.util.spec_from_file_location('state_router',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def obs(n=64):
    x=torch.randn(n,25)*.2;x[:,7]=1.2;x[:,8]=.3-x[:,2]/.25
    return x

def test_identity_install_and_normalized_state_routing():
    torch.manual_seed(12);a=m.IsolatedTurnStateRouter();x=obs()
    assert a(x).count_nonzero()==0
    w=a.positive.probabilities(x)
    torch.testing.assert_close(w.sum(-1),torch.ones(len(x)))
    y=x.clone();y[:,3]+=1
    assert not torch.equal(w,a.positive.probabilities(y))

def test_combined_budget_and_zero_support_survive_extreme_experts():
    a=m.IsolatedTurnStateRouter();x=obs(600)
    with torch.no_grad():
        for p in a.parameters():p.uniform_(-50,50)
    x[:200,8]=-.3-x[:200,2]/.25
    x[400:,7]=torch.linspace(0,2,200)
    out=a(x);assert torch.isfinite(out).all()
    assert (out.abs()<=a.budget+1e-7).all()
    x[:,7]=1.1;assert a(x).count_nonzero()==0
    x[:,7]=1.3;assert a(x).count_nonzero()==0
    x[:,7]=1.2;x[:,8]=.15-x[:,2]/.25
    torch.testing.assert_close(a(x),torch.zeros_like(out),atol=1e-10,rtol=0)

def test_positive_update_does_not_change_negative_or_reference_parameters():
    torch.manual_seed(3);a=m.IsolatedTurnStateRouter();reference=torch.nn.Linear(25,6).requires_grad_(False)
    fixed={k:p.clone() for k,p in reference.named_parameters()}
    x=obs();negative=x.clone();negative[:,8]=-.3-negative[:,2]/.25
    original=a(negative).detach().clone();old={k:p.clone() for k,p in a.negative.named_parameters()}
    opt=torch.optim.Adam(a.parameters(),lr=.01)
    loss=(reference(x)+a(x)-.2).square().mean()+.01*a.balance_loss(x)
    loss.backward();opt.step()
    assert a(x).abs().max()>0
    assert torch.equal(a(negative),original)
    assert all(torch.equal(p,old[k]) for k,p in a.negative.named_parameters())
    assert all(torch.equal(p,fixed[k]) for k,p in reference.named_parameters())

def test_balance_empty_batch_and_collapsed_routing_gradients():
    a=m.IsolatedTurnStateRouter();x=obs()
    with torch.no_grad():a.positive.router[-1].bias[0]=3
    loss=a.balance_loss(x);assert loss>0;loss.backward()
    assert a.positive.router[-1].bias.grad[0]>0
    x[:,7]=.5;a.zero_grad();empty=a.balance_loss(x);assert empty==0
    empty.backward();assert all(p.grad is None or p.grad.count_nonzero()==0 for p in a.parameters())

@pytest.mark.parametrize('budget',[[0]*6,[float('nan')]*6,[.4]*6,[.1]*5])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):m.IsolatedTurnStateRouter(budget)


def test_inactive_branch_protected_from_existing_adam_momentum_and_decay():
    a=m.IsolatedTurnStateRouter();x=obs();negative=x.clone();negative[:,8]=-.3-negative[:,2]/.25
    opt=torch.optim.AdamW(a.parameters(),lr=.01,weight_decay=.1)
    for _ in range(3):
        opt.zero_grad();(a(negative)-.2).square().mean().backward()
        a.isolate_optimizer_step(negative);opt.step()
    before={k:p.detach().clone() for k,p in a.negative.named_parameters()}
    opt.zero_grad();(a(x)-.2).square().mean().backward();a.isolate_optimizer_step(x);opt.step()
    assert all(torch.equal(p,before[k]) for k,p in a.negative.named_parameters())


def test_recovery_zero_install_and_disjoint_bounded_support():
    old=m.IsolatedTurnStateRouter()
    with torch.no_grad():
        for p in old.parameters():p.uniform_(-.3,.3)
    new=m.IsolatedTurnStateRouter(recovery_positive_2p0=True)
    missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
    assert missing and all(k == 'recovery_positive_2p0_wheel_seed' or k.startswith('recovery_positive_2p0.') for k in missing)
    assert not unexpected
    x=obs(1000);x[:,7]=torch.linspace(0,1.7,len(x))
    # Adding a zero-initialized branch cannot change any existing command.
    assert torch.equal(new(x), old(x))
    with torch.no_grad():
        for p in new.recovery_positive_2p0.parameters():p.uniform_(-50,50)
    changed=(new(x)-old(x)).abs().amax(-1)>1e-7
    assert changed.any()
    assert not changed[(x[:,7]<=.9)|(x[:,7]>=1.1)].any()
    assert (new(x).abs()<=new.budget+1e-6).all()
    x[:,8]=-.3-x[:,2]/.25
    assert torch.equal(new(x),old(x))


def test_recovery_step_does_not_move_original_router_with_adam_momentum():
    a=m.IsolatedTurnStateRouter(recovery_positive_2p0=True)
    original=obs();recovery=obs();recovery[:,7]=1.
    optimizer=torch.optim.AdamW(a.parameters(),lr=.01,weight_decay=.1)
    for _ in range(3):
        optimizer.zero_grad();(a(original)-.2).square().mean().backward()
        a.isolate_optimizer_step(original);optimizer.step()
    before={k:p.detach().clone() for k,p in a.named_parameters() if not k.startswith('recovery_positive_2p0.')}
    optimizer.zero_grad()
    ((a(recovery)-.2).square().mean()+.01*a.balance_loss(recovery)).backward()
    a.isolate_optimizer_step(recovery);optimizer.step()
    assert a(recovery).abs().max()>0
    assert all(torch.equal(p,before[k]) for k,p in a.named_parameters() if k in before)


def test_high_speed_straight_support_and_combined_budget():
    a=m.IsolatedTurnStateRouter(recovery_positive_2p0=True,recovery_straight_3p0=True)
    x=obs(3000);x[:,7]=torch.linspace(0,1.7,len(x));x[:,8]=-x[:,2]/.25
    assert a(x).count_nonzero()==0
    with torch.no_grad():
        for p in a.parameters():p.uniform_(-20,20)
    gate=a.straight_recovery_eligibility(x)
    assert gate[x[:,7]<=1.45].count_nonzero()==0
    assert (gate[x[:,7]>=1.5]==1).all()
    for yaw in [-.3,-.06,0.,.02,.03,.06,.3]:
        x[:,8]=yaw-x[:,2]/.25
        assert (a(x).abs()<=a.budget+1e-6).all()
        if abs(yaw)>=.06:
            assert (a.straight_recovery_eligibility(x)<1e-6).all()


def test_straight_recovery_update_preserves_all_turn_experts_with_momentum():
    a=m.IsolatedTurnStateRouter(recovery_positive_2p0=True,recovery_straight_3p0=True)
    turn=obs();turn[:32,7]=1.
    straight=obs();straight[:,7]=1.5;straight[:,8]=-straight[:,2]/.25
    optimizer=torch.optim.AdamW(a.parameters(),lr=.01,weight_decay=.1)
    for x in [turn,turn]:
        optimizer.zero_grad();(a(x)-.2).square().mean().backward()
        a.isolate_optimizer_step(x);optimizer.step()
    old={k:p.detach().clone() for k,p in a.named_parameters() if not k.startswith('recovery_straight_3p0.')}
    optimizer.zero_grad();((a(straight)-.2).square().mean()+.01*a.balance_loss(straight)).backward()
    a.isolate_optimizer_step(straight);optimizer.step()
    assert a(straight).abs().max()>0
    assert all(torch.equal(p,old[k]) for k,p in a.named_parameters() if k in old)


def test_extended_turn_identity_support_and_shared_budget():
    torch.manual_seed(198)
    old=m.IsolatedTurnStateRouter(recovery_positive_2p0=True,recovery_straight_3p0=True)
    with torch.no_grad():
        for p in old.parameters():p.uniform_(-.3,.3)
    new=m.IsolatedTurnStateRouter(recovery_positive_2p0=True,recovery_straight_3p0=True,extended_turn_1p0=True)
    missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
    assert missing and all(k.startswith('extended_') for k in missing) and not unexpected
    grid=torch.cartesian_prod(torch.linspace(0,3.3,67),torch.tensor([-2.5,-.8,-.6,-.45,0.,.45,.6,.8,2.5]))
    x=obs(len(grid));x[:,7]=grid[:,0]*.5;x[:,8]=grid[:,1]-x[:,2]/.25
    assert torch.equal(old(x),new(x))
    with torch.no_grad():
        for name,branch in new.named_children():
            if name.startswith('extended_'):
                for p in branch.parameters():p.uniform_(-50,50)
    protected=(grid[:,0]<=.8)|(grid[:,0]>=1.2)|(grid[:,1].abs()<=.45)
    assert torch.equal(old(x)[protected],new(x)[protected])
    assert (new(x)-old(x)).abs().max()>0
    assert (new(x).abs()<=new.budget+1e-6).all()


def test_extended_direction_momentum_isolation():
    a=m.IsolatedTurnStateRouter(extended_turn_1p0=True)
    x=obs();x[:,7]=.5;x[:,8]=-.6-x[:,2]/.25
    optimizer=torch.optim.AdamW(a.parameters(),lr=.01,weight_decay=.1)
    for _ in range(2):
        optimizer.zero_grad();(a(x)-.2).square().mean().backward()
        a.isolate_optimizer_step(x);optimizer.step()
    before={k:p.detach().clone() for k,p in a.named_parameters() if not k.startswith('extended_positive_1p0.')}
    x[:,8]=.6-x[:,2]/.25
    optimizer.zero_grad();((a(x)-.2).square().mean()+.01*a.balance_loss(x)).backward()
    a.isolate_optimizer_step(x);optimizer.step()
    assert all(torch.equal(p,before[k]) for k,p in a.named_parameters() if k in before)
    assert a(x).abs().max()>0


def test_seeded_extended_support_bounds_and_legacy_identity():
    a=m.IsolatedTurnStateRouter(extended_turn_1p0=True)
    b=m.IsolatedTurnStateRouter(extended_turn_1p0=True,extended_turn_seed=True)
    b.load_state_dict(a.state_dict(),strict=False)
    grid=torch.cartesian_prod(torch.linspace(0.,3.3,67),torch.tensor([-.8,-.6,-.45,0.,.45,.6,.8]))
    x=obs(len(grid));x[:,7]=grid[:,0]*.5;x[:,8]=grid[:,1]-x[:,2]/.25
    assert torch.equal(a(x),b(x))
    with torch.no_grad():
        b.extended_positive_1p0_seed.copy_(torch.tensor([0.,0.,-.05,0.,0.,-.03]))
        b.extended_negative_1p0_seed.copy_(torch.tensor([0.,0.,.04,0.,0.,.04]))
    protected=(grid[:,0]<=.8)|(grid[:,0]>=1.2)|(grid[:,1].abs()<=.45)
    assert torch.equal(a(x)[protected],b(x)[protected])
    with torch.no_grad():
        for p in b.parameters():p.uniform_(-50,50)
        b.extended_positive_1p0_seed.fill_(99.)
    assert (b(x).abs()<=b.correction_bound(x)+1e-6).all()
    assert (b(x)[protected].abs()<=b.budget+1e-6).all()

def test_turn08_seed_and_learning_preserve_turn06_and_bound():
    torch.manual_seed(205)
    old=m.IsolatedTurnStateRouter(extended_turn_1p0=True,extended_turn_seed=True)
    new=m.IsolatedTurnStateRouter(extended_turn_1p0=True,extended_turn_seed=True,extended_turn_0p8=True)
    missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
    assert not unexpected and missing and all('_1p0_0p8.' in k for k in missing)
    grid=torch.cartesian_prod(torch.linspace(0.,3.3,67),torch.tensor([-1.,-.8,-.7,-.6,-.45,0.,.45,.6,.7,.8,1.]))
    x=obs(len(grid));x[:,7]=grid[:,0]*.5;x[:,8]=grid[:,1]-x[:,2]/.25
    assert torch.equal(old(x),new(x))
    with torch.no_grad():
        new.extended_positive_1p0_0p8.seed.copy_(torch.tensor([0.,0.,-.05,0.,0.,.01]))
        new.extended_negative_1p0_0p8.seed.copy_(torch.tensor([0.,0.,.02,0.,0.,.02]))
        for name,branch in new.named_children():
            if '_1p0_0p8' in name:
                for p in branch.parameters():p.uniform_(-40,40)
    protected=(grid[:,0]<=.8)|(grid[:,0]>=1.2)|(grid[:,1].abs()<=.6)
    assert torch.equal(new(x)[protected],old(x)[protected])
    assert (new(x).abs()<=new.correction_bound(x)+1e-6).all()
    assert (new.extended_positive_1p0_0p8(x).abs()<=new.extended_positive_1p0_0p8.local_budget+1e-6).all()

def test_turn08_selected_training_leaves_previous_graph_unchanged():
    a=m.IsolatedTurnStateRouter(extended_turn_1p0=True,extended_turn_seed=True,extended_turn_0p8=True)
    a.requires_grad_(False);a.extended_positive_1p0_0p8.requires_grad_(True)
    frozen={k:p.detach().clone() for k,p in a.named_parameters() if not p.requires_grad}
    x=obs();x[:,7]=.5;x[:,8]=.8-x[:,2]/.25
    opt=torch.optim.AdamW([p for p in a.parameters() if p.requires_grad],lr=.01)
    opt.zero_grad();((a(x)-.03).square().mean()+.01*a.balance_loss(x)).backward()
    a.isolate_optimizer_step(x);opt.step()
    assert all(torch.equal(p,frozen[k]) for k,p in a.named_parameters() if k in frozen)
    assert a(x).abs().max()>0

def test_positive_turn10_preserves_all_negative_and_previous_commands():
    torch.manual_seed(207)
    old=m.IsolatedTurnStateRouter(extended_turn_1p0=True,extended_turn_seed=True,extended_turn_0p8=True)
    new=m.IsolatedTurnStateRouter(extended_turn_1p0=True,extended_turn_seed=True,extended_turn_0p8=True,extended_positive_turn_1p0=True)
    with torch.no_grad():
        for p in old.parameters():p.uniform_(-.2,.2)
    missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
    assert not unexpected and missing and all(k.startswith('extended_positive_1p0_1p0.') for k in missing)
    grid=torch.cartesian_prod(torch.linspace(0.,3.3,67),torch.tensor([-2.5,-1.2,-1.,-.8,-.6,0.,.6,.8,.9,1.,1.2,2.5]))
    x=obs(len(grid));x[:,7]=grid[:,0]*.5;x[:,8]=grid[:,1]-x[:,2]/.25
    assert torch.equal(old(x),new(x))
    with torch.no_grad():
        for p in new.extended_positive_1p0_1p0.parameters():p.uniform_(-40,40)
    protected=(grid[:,0]<=.8)|(grid[:,0]>=1.2)|(grid[:,1]<=.8)
    assert torch.equal(old(x)[protected],new(x)[protected])
    assert (new(x).abs()<=new.correction_bound(x)+1e-6).all()
    assert (new(x)-old(x)).abs().max()>0

def test_positive_turn10_optimizer_updates_only_new_branch():
    a=m.IsolatedTurnStateRouter(extended_turn_1p0=True,extended_turn_0p8=True,extended_positive_turn_1p0=True)
    a.requires_grad_(False);a.extended_positive_1p0_1p0.requires_grad_(True)
    frozen={k:p.detach().clone() for k,p in a.named_parameters() if not p.requires_grad}
    x=obs();x[:,7]=.5;x[:,8]=1.-x[:,2]/.25
    opt=torch.optim.AdamW([p for p in a.parameters() if p.requires_grad],lr=.01)
    opt.zero_grad();((a(x)-.03).square().mean()+.01*a.balance_loss(x)).backward()
    a.isolate_optimizer_step(x);opt.step()
    assert all(torch.equal(p,frozen[k]) for k,p in a.named_parameters() if k in frozen)
    assert a(x).abs().max()>0


def test_e2e_branches_have_local_support_budget_and_direction_isolation():
    a=m.IsolatedTurnStateRouter(e2e_turn_1p0=True)
    x=obs(6)
    x[:,7]=torch.tensor([.5,.5,.5,.5,1.2,.5])
    yaw=torch.tensor([.8,-.8,.45,-.45,.8,1.2])
    x[:,8]=yaw-x[:,2]/.25
    assert torch.count_nonzero(a(x))==0
    with torch.no_grad():
        for expert in a.e2e_positive_1p0.experts:
            expert[-1].bias.fill_(100.)
    y=a(x)
    torch.testing.assert_close(y[[0,5]],torch.full((2,6),.3))
    assert torch.count_nonzero(y[[1,2,3,4]])==0
    assert (y.abs()<=a.correction_bound(x)+1e-7).all()
    branches={name:mask for name,_,mask in a.branches(x)}
    assert branches['e2e_positive_1p0'].tolist()==[True,False,False,False,False,True]
    assert branches['e2e_negative_1p0'].tolist()==[False,True,False,False,False,False]
