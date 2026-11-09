import torch
import pytest
from tensordict import TensorDict
from rmuc_critic_command_balance import command_weights, weighted_value_output, KEY


def test_minority_command_gain_is_bounded_without_erasing_majority():
    c=torch.zeros(1000,3);c[-10:,0]=1.5;c[-5:,2]=.8
    w,r=command_weights(c)
    assert w.mean().item()==pytest.approx(1.,abs=1e-6)
    assert w.min()>=.25 and w.max()<=3
    assert w[-1]>w[0]>0
    z=TensorDict({'policy':torch.zeros(1000,25),KEY:w},batch_size=[1000]);order=torch.randperm(1000)
    assert torch.equal(z[order][KEY],w[order])


def test_weighted_output_exactly_matches_explicit_mse_and_leaves_actor_gradient():
    torch.manual_seed(4)
    v=torch.randn(20,1,requires_grad=True);copy=v.detach().clone().requires_grad_(True)
    actor=torch.randn(20,6,requires_grad=True);other=actor.detach().clone().requires_grad_(True)
    y=torch.randn_like(v);w=torch.rand_like(v)+.25;before=v.detach().clone()
    weighted_value_output(v,w)
    ((v-y).square().mean()+actor.square().mean()).backward()
    ((copy-y).square().mul(w).mean()+other.square().mean()).backward()
    torch.testing.assert_close(v.grad,copy.grad);torch.testing.assert_close(actor.grad,other.grad)
    assert torch.equal(v.detach(),before)


def test_zero_fraction_is_original_and_continuous_microsteps_share_bins():
    c=torch.zeros(20,3);c[:,0]=torch.linspace(.49,.51,20)
    w,r=command_weights(c,fraction=0)
    assert torch.allclose(w,torch.ones_like(w))
    assert len(r['groups'])==1
    with pytest.raises(ValueError):command_weights(torch.full((3,3),float('nan')))


def test_rollout_labels_follow_shuffled_minibatches_and_cleanup(tmp_path):
    from types import SimpleNamespace as NS
    from rmuc_critic_command_balance import configure_critic_command_balance
    class Critic(torch.nn.Module):
        is_recurrent=False
        def __init__(self):
            super().__init__();self.net=torch.nn.Linear(25,1)
        def forward(self,obs):return self.net(obs['policy'])
    critic=Critic();obs=TensorDict({'policy':torch.randn(2,5,25)},batch_size=[2,5])
    returns=torch.randn(2,5,1);returns_before=returns.clone()
    storage=NS(observations=obs)
    alg=NS(critic=critic,actor=NS(is_recurrent=False),symmetry=None,use_clipped_value_loss=False,
           storage=storage,num_learning_epochs=1,num_mini_batches=2,act=lambda x:x)
    def update():
        flat=obs.reshape(10);order=torch.tensor([9,0,8,1,7,2,6,3,5,4]);yy=returns.reshape(10,1)
        for ids in order.split(5):
            critic.zero_grad();v=critic(flat[ids]);loss=(v-yy[ids]).square().mean();loss.backward()
            hooked=critic.net.weight.grad.clone()
            # Evaluate the same intended weighted loss without triggering its hook.
            critic.zero_grad();direct=critic.net(flat[ids]['policy']);((direct-yy[ids]).square()*flat[ids][KEY]).mean().backward()
            torch.testing.assert_close(hooked,critic.net.weight.grad)
        return {'value':1.}
    alg.update=update
    c=torch.zeros(5,3);c[-1,0]=1.5
    env=NS(_current_command=lambda:c);env.unwrapped=env
    runner=NS(alg=alg,logger=NS(log_dir=str(tmp_path)))
    configure_critic_command_balance(runner,env)
    for _ in range(2):
        alg.act(obs[0]);alg.act(obs[1]);alg.update()
        assert KEY not in obs.keys()
        assert torch.equal(returns,returns_before)
    assert len((tmp_path/'critic_command_balance.jsonl').read_text().splitlines())==2
