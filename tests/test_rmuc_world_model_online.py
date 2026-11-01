import torch
from rmuc_world_model.gae import lambda_returns, ReturnCritic
from rmuc_world_model.gpu_replay import WindowReplay
from rmuc_world_model.dynamics import Dynamics

def test_gae_termination_and_truncation():
    r=torch.tensor([[1.,1.],[2.,2.]],requires_grad=True)
    v=torch.tensor([[0.,0.],[3.,3.],[10.,10.]],requires_grad=True)
    c=torch.tensor([[1.,1.],[0.,1.]])
    y=lambda_returns(r,v,c,gamma=.9,lam=1.)
    torch.testing.assert_close(y,torch.tensor([[2.8,10.9],[2.,11.]]))
    assert not y.requires_grad
    z=lambda_returns(r,v,c,gamma=.9,lam=0.)
    torch.testing.assert_close(z,torch.tensor([[3.7,3.7],[2.,11.]]))

def test_vectorized_windows_never_cross_sequences():
    seq=[]
    for n,off in [(15,0),(23,100)]:
        s=(torch.arange(n)+off).float()[:,None]
        seq.append(dict(state=s,action=s.expand(-1,6),context=s,starts=torch.arange(4,n-3+1)))
    replay=WindowReplay(seq,4,3,'cpu')
    s,ah,a,y,c=replay.sample(1024,torch.Generator().manual_seed(1))
    torch.testing.assert_close(ah[...,0],s[...,0])
    torch.testing.assert_close(a[...,0],y[...,0]);torch.testing.assert_close(c,y)
    torch.testing.assert_close(y[:,0,0]-s[:,-1,0],torch.ones(1024))
    assert ((y[:,-1,0]-s[:,0,0])==6).all()

def test_default_depth_preserves_keys_and_deeper_model_gradients():
    args=[torch.zeros(3),torch.ones(3),torch.ones(3),torch.zeros(2),torch.ones(2)]
    old=Dynamics(*args,width=8)
    assert 'net.4.weight' in old.state_dict()
    deep=Dynamics(*args,width=8,depth=3)
    assert 'net.6.weight' in deep.state_dict()
    s=torch.randn(2,4,3); ah=torch.randn(2,4,6);a=torch.randn(2,6);c=torch.randn(2,2)
    loss=deep(s,ah,a,c).square().mean();loss.backward()
    assert torch.isfinite(deep.net[-1].weight.grad).all()

def test_critic_update_real_gradient():
    torch.manual_seed(1)
    critic=ReturnCritic(torch.zeros(5),torch.ones(5));opt=torch.optim.Adam(critic.parameters(),lr=.001)
    x=torch.randn(64,5);target=torch.ones(64)*2
    before=(critic(x)-target).square().mean()
    for _ in range(5):
        opt.zero_grad();loss=(critic(x)-target).square().mean();loss.backward();opt.step()
    assert (critic(x)-target).square().mean()<before

def test_step_overshoot_is_directional_and_ramps_excluded():
    from rmuc_world_model.transition_reward import step_metadata,overshoot_cost
    commands=torch.tensor([0.,0.,.5,.5,.5,.35,.35,0.,0.])
    meta=step_metadata(commands)
    torch.testing.assert_close(meta[3],torch.tensor([1.,.5]))
    torch.testing.assert_close(meta[6],torch.tensor([-1.,.15]))
    torch.testing.assert_close(meta[8],torch.tensor([-1.,.35]))
    target=torch.tensor([.35,.35,0.,.5]);vel=torch.tensor([.5,.30,-.10,.65])
    info=torch.tensor([[-1.,.15],[-1.,.15],[-1.,.5],[1.,.5]])
    cost=overshoot_cost(vel,target,info)
    assert cost[0]==0 and (cost[1:]>0).all()
    assert step_metadata(torch.linspace(0,.5,101)).count_nonzero()==0

def test_reward_context_is_present_and_old_critic_rejected():
    import pytest
    from rmuc_world_model.critic_context import features,validate_reward_schema
    state=torch.zeros(2,3);context=torch.ones(2,4);event=torch.tensor([[1.,.5],[-1.,.5]])
    conditioned=features(state,context,event,True)
    assert conditioned.shape==(2,9)
    assert not torch.equal(conditioned[0],conditioned[1])
    assert features(state,context,event,False).shape==(2,7)
    with pytest.raises(ValueError,match='schema'):
        validate_reward_schema({},True,4.)
    with pytest.raises(ValueError,match='reward'):
        validate_reward_schema({'event_conditioned':True,'overshoot_weight':1.},True,4.)
    validate_reward_schema({'event_conditioned':True,'overshoot_weight':4.},True,4.)

def test_queue_padding_requires_inactive_transport_and_preserves_physics():
    import pytest
    from rmuc_world_model.queue_schema import align_delay_queues
    names=['root_state_w:0']+[f'leg_history:{i}' for i in range(4)]+[f'wheel_delay_history:{i}' for i in range(4)]
    target=['root_state_w:0']+[f'leg_history:{i}' for i in range(8)]+[f'wheel_delay_history:{i}' for i in range(6)]
    x=torch.arange(25.).expand(3,-1).clone();c=torch.zeros(35)
    q,meta=align_delay_queues([{'state':x,'context':c}],{'extra_state_names':names},target)
    torch.testing.assert_close(q[0]['state'][:,:21],x[:,:21])
    assert q[0]['state'][:,21:25].count_nonzero()==0
    torch.testing.assert_close(q[0]['state'][:,25:29],x[:,21:25])
    c[12]=1
    with pytest.raises(ValueError,match='active leg'):
        align_delay_queues([{'state':x,'context':c}],{'extra_state_names':names},target)
    with pytest.raises(ValueError,match='Only zero'):
        align_delay_queues([],{'extra_state_names':names},target+['joint_vel:99'])

def test_delay_balance_independent_of_dataset_size():
    from types import SimpleNamespace
    from rmuc_world_model.domain_sampling import DelayDomainSampler
    c=torch.zeros(1100,35);c[:1000,12:14]=1
    replay=SimpleNamespace(context=c,starts=torch.arange(1100))
    sampler=DelayDomainSampler(replay)
    idx=sampler.sample(256,torch.Generator().manual_seed(9))
    assert (c[idx,12]==0).sum()==128
    assert (c[idx,12]==1).sum()==128

def test_exploration_is_bounded_and_rejected_for_acceptance_inputs():
    import pytest
    from rmuc_world_model.exploration import validate_exploration,excite
    a=torch.ones(10000,6)
    assert excite(a,0.) is a
    x=excite(a,.03,torch.Generator().manual_seed(1))
    assert (x-a).abs().max()<=.090001
    assert (x-a).std()>.025
    for std,replay,distill,audit in [(.03,False,False,False),(.03,True,True,False),(.03,True,False,True),(float('nan'),True,False,False)]:
        with pytest.raises(ValueError):validate_exploration(std,replay,distill,audit)
    validate_exploration(.03,True)

def test_randomized_response_loss_detects_wrong_gain_not_constant_bias():
    from rmuc_world_model.response_loss import randomized_residual_loss
    torch.manual_seed(8)
    noise=torch.randn(4096,2)*.03;gain=torch.tensor([[.02],[-.04]])
    target=noise@gain
    good=randomized_residual_loss(target+2.,target,noise,torch.ones(1))
    wrong=randomized_residual_loss(-target,target,noise,torch.ones(1))
    assert good<1e-10 and wrong>.001
    parameter=torch.nn.Parameter(-gain.clone());optimizer=torch.optim.SGD([parameter],lr=.5)
    for _ in range(5):
        optimizer.zero_grad();loss=randomized_residual_loss(noise@parameter,target,noise,torch.ones(1));loss.backward();optimizer.step()
    assert randomized_residual_loss(noise@parameter,target,noise,torch.ones(1))<wrong*.01

def test_critic_constant_mask_blocks_value_and_input_gradient():
    from rmuc_world_model.gae import ReturnCritic
    torch.manual_seed(14)
    c=ReturnCritic(torch.zeros(3),torch.ones(3),torch.tensor([1.,0.,1.]))
    with torch.no_grad():c.net[-1].weight.fill_(.1)
    x=torch.randn(8,3,requires_grad=True);y=x.detach().clone();y[:,1]+=100
    torch.testing.assert_close(c(x),c(y))
    c(x).sum().backward();assert x.grad[:,1].count_nonzero()==0
    assert x.grad[:,[0,2]].abs().sum()>0
    restored=ReturnCritic(c.mean,c.scale,c.input_mask);restored.load_state_dict(c.state_dict())
    torch.testing.assert_close(restored(y),c(y))
    legacy=ReturnCritic(torch.zeros(3),torch.ones(3));assert 'input_mask' not in legacy.state_dict()

def test_physical_speed_objective_bounds_transient_gradient_and_keeps_idle_precision():
    from rmuc_world_model.tracking_objective import speed_cost,LEGACY,PHYSICAL
    target=torch.tensor([0.,.5,1.5]);error=torch.ones(3,requires_grad=True)*.5
    old=torch.autograd.grad(speed_cost(error,target,LEGACY).sum(),error,retain_graph=True)[0]
    new=torch.autograd.grad(speed_cost(error,target,PHYSICAL).sum(),error)[0]
    assert old[0]>4.9*old[1]
    assert (new<=20).all() and new[0]/new[1]<1.01
    fine=torch.tensor([.001,.001],requires_grad=True)
    gradient=torch.autograd.grad(speed_cost(fine,torch.tensor([0.,.5]),PHYSICAL).sum(),fine)[0]
    assert gradient[0]>4.9*gradient[1]
    for version in [LEGACY,PHYSICAL]:
        grid=torch.linspace(0,1,101);cost=speed_cost(grid,torch.zeros_like(grid),version)
        assert cost[0]==0 and (cost[1:]>cost[:-1]).all()
        torch.testing.assert_close(cost,speed_cost(-grid,torch.zeros_like(grid),version))

def test_constraint_projection_checks_actual_step_and_conflicting_constraints():
    from rmuc_world_model.constraint_update import project_update
    delta=torch.tensor([1.,2.,3.]);grads=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
    out,info=project_update(delta,grads)
    torch.testing.assert_close(out,torch.tensor([0.,0.,3.]))
    assert (grads@out<=1e-7).all() and out.norm()<=delta.norm()
    opposed=torch.tensor([[1.,0.],[-1.,0.],[1.,1.]])
    out,info=project_update(torch.tensor([1.,1.]),opposed)
    assert (opposed@out<=1e-6).all()
    empty=torch.empty(0,3);out,_=project_update(delta,empty);torch.testing.assert_close(out,delta)

def test_late_phase_resets_and_worst_quartile_does_not_hide_spike():
    from rmuc_world_model.constraint_update import late_phase,upper_quartile
    c=torch.cat([torch.zeros(100),torch.ones(300)*.5,torch.zeros(300)])
    late=late_phase(c)
    assert late[:100].all() and not late[100:288].any() and late[288:400].all()
    assert not late[400:588].any() and late[588:].all()
    x=torch.tensor([0.,0.,0.,4.],requires_grad=True);y=upper_quartile(x)
    assert y==4;y.backward();assert x.grad[-1]==1 and x.grad[:-1].count_nonzero()==0
