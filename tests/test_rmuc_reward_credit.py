import json
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from rmuc_reward_credit import (penalty_reward_effect, fixed_value_gae_effect,
                                normalized_advantage_effect, configure_reward_credit_audit)


def test_reward_clip_can_erase_nonzero_penalty_and_counterfactual_detects_it():
    raw=torch.tensor([-110.,-101.,0.,19.,22.]);cost=torch.tensor([2.,3.,2.,2.,1.])
    torch.testing.assert_close(penalty_reward_effect(raw,cost,-100,20),torch.tensor([0.,-2.,-2.,-1.,0.]))
    with pytest.raises(ValueError):penalty_reward_effect(raw,-cost,-100,20)


def test_term_gae_matches_reward_removal_and_does_not_cross_reset():
    reward=torch.tensor([[2.,3.],[1.,4.],[5.,8.]])
    effect=torch.tensor([[-.2,-.5],[-.4,-.3],[-.1,-.7]])
    done=torch.tensor([[False,True],[True,False],[False,False]])
    before=fixed_value_gae_effect(reward,done,.9,.8)
    after=fixed_value_gae_effect(reward-effect,done,.9,.8)
    contribution=fixed_value_gae_effect(effect,done,.9,.8)
    torch.testing.assert_close(before-after,contribution)
    assert contribution[0,1]==effect[0,1]
    assert contribution[1,0]==effect[1,0]
    torch.testing.assert_close(normalized_advantage_effect(before,contribution),
        (before-before.mean())/(before.std()+1e-8)-(after-after.mean())/(after.std()+1e-8))


@pytest.mark.parametrize('pitch_weight',[0.,2.])
def test_instrumentation_preserves_optimizer_and_measures_term_gradient(tmp_path,monkeypatch,pitch_weight):
    monkeypatch.setenv('RMUC_CAPTURE_PPO_ROLLOUT_UPDATE','1')
    class Actor(torch.nn.Module):
        is_recurrent=False
        def __init__(self):
            super().__init__();self.net=torch.nn.Linear(25,6)
        def forward(self,obs):return self.net(obs['policy'])
    torch.manual_seed(9);actor=Actor();reference=Actor();reference.load_state_dict(actor.state_dict())
    T,N=3,4;obs=torch.randn(T,N,25)*.01;obs[:,:,7]=-.3;obs[:,:,2]=0;obs[:,:,8]=0
    with torch.no_grad():mean=actor(TensorDict({'policy':obs},batch_size=[T,N]))
    std=torch.ones_like(mean)*.03;actions=mean+torch.randn_like(mean)*std
    raw_reward=torch.tensor([[2.,3.,4.,5.],[5.,4.,3.,2.],[1.,2.,4.,8.]])
    dones=torch.zeros(T,N,1,dtype=torch.bool);values=torch.zeros(T,N,1)
    returns=fixed_value_gae_effect(raw_reward,dones[...,0],.998,.99)[...,None]
    adv=(returns-returns.mean())/returns.std()
    storage=SimpleNamespace(observations=TensorDict({'policy':obs},batch_size=[T,N]),
        dones=dones,values=values,returns=returns,advantages=adv,actions=actions,distribution_params=(mean,std))
    optimizer=torch.optim.SGD(actor.parameters(),lr=.001)
    def do_update(model,opt):
        out=model(TensorDict({'policy':obs.reshape(-1,25)},batch_size=[T*N])).reshape(T,N,6)
        loss=-(((actions-mean)/std.square()).detach()*out*adv).sum(-1).mean()
        opt.zero_grad();loss.backward();opt.step();return {'loss':float(loss.detach())}
    alg=SimpleNamespace(get_policy=lambda:actor,storage=storage,gamma=.998,lam=.99,
                        update=lambda:do_update(actor,optimizer),process_env_step=lambda *args:None)
    cfg=SimpleNamespace(command_active_targets=[[-.6,0]],command_reverse_runaway_weight=2.,
                        command_reverse_pitch_oscillation_weight=pitch_weight,
                        command_speed_oscillation_penalty_weight=2.,reward_clip_min=-100.,reward_clip_max=20.,
                        position_authority_speed_plateau_end=1.)
    env=SimpleNamespace(cfg=cfg);env.unwrapped=env
    runner=SimpleNamespace(alg=alg,logger=SimpleNamespace(log_dir=tmp_path))
    configure_reward_credit_audit(runner,env)
    for i in range(T):
        metrics={'reward_unclipped':raw_reward[i],
                 'penalty_reverse_runaway':torch.arange(N,dtype=torch.float32)*.1*(i+1),
                 'penalty_command_speed_oscillation':torch.arange(N,dtype=torch.float32).flip(0)*.03*(i+1),
                 'penalty_reverse_pitch_oscillation':torch.arange(N,dtype=torch.float32)*.04*(i+1),
                 'command_aligned_speed':torch.full((N,),.7),
                 'command_filtered_aligned_speed':torch.full((N,),.68)}
        alg.process_env_step(None,raw_reward[i],dones[i,:,0],{'episode':metrics})
    alg.update();do_update(reference,torch.optim.SGD(reference.parameters(),lr=.001))
    for a,b in zip(actor.parameters(),reference.parameters()):assert torch.equal(a,b)
    report=json.loads((tmp_path/'reward_credit_audit.jsonl').read_text())
    row=report['rows'][0]
    assert row['actual_parameter_update_norm']>0
    assert row['terms']['reverse_runaway']['actor_parameter_gradient_norm']>0
    assert row['terms']['speed_oscillation']['actor_parameter_gradient_norm']>0
    if pitch_weight:
        assert row['terms']['reverse_pitch_oscillation']['actor_parameter_gradient_norm']>0
        assert row['terms']['reverse_pitch_oscillation']['normalized_advantage_effect_rms']>0
        assert row['terms']['reverse_pitch_oscillation']['cosine_actual_update_with_term_ascent'] is not None
    else:
        assert 'reverse_pitch_oscillation' not in row['terms']
    assert row['reverse_overspeed_samples']==T*N
    assert not report['unbroken_chain_certified']
    snapshot=torch.load(tmp_path/'reward_credit_snapshot.pt',weights_only=False)
    assert snapshot['contract']['gradient_max_samples']==0
    assert row['gradient_samples']==T*N
    assert snapshot['actual_parameter_displacement'].norm()>0
    torch.testing.assert_close(snapshot['weighted_costs']['reverse_runaway'][0],torch.arange(N)*.2)
    assert not torch.equal(snapshot['actor_state_before']['net.weight'],snapshot['actor_state_after']['net.weight'])


def test_chunked_full_gradient_matches_whole_batch_without_dropping_tail():
    from rmuc_reward_credit import score_gradients
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(25,8),torch.nn.Tanh(),torch.nn.Linear(8,6))
        def forward(self,x):return self.net(x['policy'])
    torch.manual_seed(19);a=Actor();obs=torch.randn(13,25)
    mean=a(TensorDict({'policy':obs},batch_size=[13])).detach();score=torch.randn(13,6)
    adv={'full':torch.randn(13),'cost':torch.randn(13)};params=list(a.parameters())
    full,_,_=score_gradients(a,obs,mean,score,adv,params,13)
    chunks,_,_=score_gradients(a,obs,mean,score,adv,params,4)
    for k in adv:torch.testing.assert_close(full[k],chunks[k],atol=2e-7,rtol=2e-6)
    assert all(p.grad is None for p in params)
