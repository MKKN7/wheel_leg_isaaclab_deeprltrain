from types import SimpleNamespace as N
import copy
import hashlib
import json

import torch
import pytest

from rmuc_command_value import CommandValueNetwork,mlp
from rmuc_command_value_training import configure_command_value_training,validate_actor_critic_coverage


@pytest.mark.parametrize('minimum_age', [0.,6.])
@pytest.mark.parametrize('mixed_reference', [False,True])
@pytest.mark.parametrize('exact_on_policy', [False,True])
def test_training_actor_frozen_serialization_and_calibration_replay(tmp_path, minimum_age, mixed_reference, exact_on_policy, monkeypatch):
    torch.manual_seed(17)
    base=mlp();experts=torch.nn.ModuleList([mlp(),mlp()])
    spec=dict(arm='isolated_command_scale',grid=[[0,0],[.5,.3]],mean=torch.zeros(25),
        scale=torch.ones(25),command_mean=torch.tensor([[50.],[100.]]),command_scale=torch.tensor([[3.],[7.]]),model_state_dict=experts.state_dict())
    bank=CommandValueNetwork(base.state_dict(),spec)
    validate_actor_critic_coverage(bank,None)
    validate_actor_critic_coverage(bank,[[0,0],[.5,.3]])
    with pytest.raises(ValueError,match='Uncalibrated actor targets'):
        validate_actor_critic_coverage(bank,[[0,0],[1.8,0],[-.6,0]])
    with pytest.raises(ValueError,match='explicit target'):
        validate_actor_critic_coverage(bank,[])
    initial=tmp_path/'critic.pt';torch.save(bank.checkpoint(),initial)
    x=torch.randn(32,25);x[:,7]=0;x[:,8]=-x[:,2]/.25
    x[16:,7]=.25;x[16:,8]=.3-x[16:,2]/.25
    path=tmp_path/'data.pt'
    torch.save(dict(observations=x,returns=torch.ones(32)*60,episode_step=torch.ones(32)*(10 if minimum_age==0 else 900),
        tail_error_bounds=torch.zeros(32),env_ids=torch.arange(32)%4,gamma=.998),path)
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    cfg=dict(critic_checkpoint=str(initial),critic_sha256=sha(initial),gamma=.998,
        learning_rate=.0001,seed=99,epochs=1,minibatches=2,replay_weight=1.,
        datasets=[dict(path=str(path),sha256=sha(path),minimum_episode_age_s=minimum_age)])
    if mixed_reference:
        import sys
        source=tmp_path/'source.pt';source.write_bytes(b'test frozen actor')
        data=torch.load(path,weights_only=False);data['checkpoint']=str(source);torch.save(data,path)
        cfg['datasets'][0].update(sha256=sha(path),commands=[[0,0],[.5,.3]],source_policy_sha256=sha(source))
        cfg.update(mc_policy_references=[dict(checkpoint=str(source),sha256=sha(source),commands=[[0,0],[.5,.3]])],max_mc_policy_kl=1e6)
        class Reference(torch.nn.Module):
            def forward(self,obs):return obs['policy'].new_zeros(len(obs['policy']),6)
        monkeypatch.setitem(sys.modules,'verify_rmuc_export',N(restore_actor=lambda p:Reference()))
    if exact_on_policy:
        cfg['on_policy_exact_commands_only']=True
        # Replay remains exact; on-policy ramp observations pass close to heads.
        x=x.clone();x[::4,7]+=.0004
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    class Critic(torch.nn.Module):
        obs_dim=25;is_recurrent=False;obs_normalization=False
        def __init__(self):super().__init__();self.mlp=copy.deepcopy(base)
        def forward(self,obs):return self.mlp(obs['policy'])
    critic=Critic();actor=torch.nn.Linear(25,6)
    original_actor=copy.deepcopy(actor.state_dict())
    optimizer=torch.optim.Adam([{'params':actor.parameters()},{'params':critic.parameters()}],lr=.001)
    alg=N(critic=critic,actor=actor,optimizer=optimizer,gamma=.998,use_clipped_value_loss=False,is_multi_gpu=False,
          storage=N(observations={'policy':x.reshape(2,16,25)},returns=torch.ones(2,16,1)*65))
    def update():
        # Mirrors value-only warmup; actor is unused, standard optimizer must
        # not step the replaced critic parameters.
        optimizer.zero_grad();critic({'policy':x}).square().mean().backward();optimizer.step()
        return {}
    alg.get_policy=lambda:lambda obs:actor(obs['policy'])
    alg.update=update
    alg.save=lambda:dict(critic_state_dict=critic.state_dict(),actor_state_dict=actor.state_dict())
    runner=N(alg=alg,device='cpu',logger=N(log_dir=str(tmp_path)))
    model=configure_command_value_training(runner,config)
    expected=copy.deepcopy(model.state_dict())
    alg.update()
    assert all(torch.equal(v,actor.state_dict()[k]) for k,v in original_actor.items())
    assert all(torch.equal(v,model.base.state_dict()[k]) for k,v in base.state_dict().items())
    assert any(not torch.equal(v,model.state_dict()[k]) for k,v in expected.items() if k.startswith('experts.'))
    state=alg.save();restored=CommandValueNetwork.from_checkpoint(state['rmuc_command_value'])
    torch.testing.assert_close(model(x),restored(x),rtol=0,atol=0)
    fallback=mlp();fallback.load_state_dict({k.removeprefix('mlp.'):v for k,v in state['critic_state_dict'].items()})
    torch.testing.assert_close(fallback(x),base(x),rtol=0,atol=0)
    assert 'rmuc_command_value_optimizer' in state
    record=json.loads((tmp_path/'command_value_updates.jsonl').read_text())
    assert not record['fallback_changed'] and record['critic_steps']==2
    assert record['critic_update_samples']==(24 if exact_on_policy else 32)
    if mixed_reference:assert len(record['mc_source_policy_kl_by_command'])==2
def test_calibrated_sidecar_resume_cannot_silently_use_fallback():
    from rmuc_command_value_training import validate_command_value_resume
    import pytest
    validate_command_value_resume({})
    with pytest.raises(ValueError):validate_command_value_resume({'rmuc_command_value':{}})
    migrated={'rmuc_command_value_migration':{'requires_explicit_command_value_training':True,'critic_sha256':'expected'}}
    with pytest.raises(ValueError):validate_command_value_resume(migrated)
    with pytest.raises(ValueError):validate_command_value_resume(migrated,{'critic_sha256':'wrong'})
    validate_command_value_resume(migrated,{'critic_sha256':'expected'})


def test_mixed_policy_replay_excludes_other_actor_labels_and_validates_sources(tmp_path,monkeypatch):
 import sys
 from rmuc_command_value_training import load_replay,load_command_policy_references
 oldpolicy=tmp_path/'old.pt';newpolicy=tmp_path/'new.pt';oldpolicy.write_bytes(b'old-actor');newpolicy.write_bytes(b'new-actor')
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 base=mlp();experts=torch.nn.ModuleList([mlp(),mlp()]);bank=CommandValueNetwork(base.state_dict(),dict(arm='isolated_command_scale',grid=[[-1,0],[-1.1,0]],mean=torch.zeros(25),scale=torch.ones(25),command_mean=torch.zeros(2,1),command_scale=torch.ones(2,1),model_state_dict=experts.state_dict()),half_width=(.01,.04))
 x=torch.zeros(16,25);x[:8,7]=-.5;x[8:,7]=-.55
 datasets=[]
 for name,policy,commands,labels in [('old',oldpolicy,[[-1,0]],torch.cat((torch.ones(8)*5,torch.ones(8)*999))),('new',newpolicy,[[-1.1,0]],torch.ones(16)*20)]:
  f=tmp_path/(name+'data.pt');torch.save(dict(checkpoint=str(policy),observations=x,returns=labels,episode_step=torch.ones(16),tail_error_bounds=torch.zeros(16),env_ids=torch.arange(16)%4,gamma=.998),f)
  datasets.append(dict(path=str(f),sha256=sha(f),minimum_episode_age_s=0.,commands=commands,source_policy_sha256=sha(policy)))
 refs=[dict(checkpoint=str(newpolicy),sha256=sha(newpolicy),commands=[[-1.1,0]]),dict(checkpoint=str(oldpolicy),sha256=sha(oldpolicy),commands=[[-1,0]])]
 cfg=dict(gamma=.998,datasets=datasets,mc_policy_references=refs,max_mc_policy_kl=.5)
 rx,ry,rb,pools=load_replay(cfg,bank,'cpu');assert len(rx)==8;assert torch.equal(ry[pools[0]],torch.ones(4,1)*5);assert torch.equal(ry[pools[1]],torch.ones(4,1)*20);assert not (ry==999).any()
 class Ref(torch.nn.Module):
  def __init__(self,value):super().__init__();self.bias=torch.nn.Parameter(torch.ones(6)*value)
  def forward(self,obs):return self.bias.expand(len(obs['policy']),6)
 monkeypatch.setitem(sys.modules,'verify_rmuc_export',N(restore_actor=lambda p:Ref(1 if p==oldpolicy else 2)))
 policies=load_command_policy_references(cfg,bank.centers,'cpu');assert policies[0].bias[0]==1 and policies[1].bias[0]==2;assert not policies[0].bias.requires_grad
 bad=copy.deepcopy(cfg);bad['datasets'][0]['commands']=[[-1,0],[-1.1,0]]
 with pytest.raises(ValueError,match='reference disagree'):load_replay(bad,bank,'cpu')
 bad=copy.deepcopy(cfg);bad['datasets'][1]['source_policy_sha256']='wrong'
 with pytest.raises(ValueError,match='provenance mismatch'):load_replay(bad,bank,'cpu')
 bad=copy.deepcopy(cfg);bad['mc_policy_references']=refs[:1]
 with pytest.raises(ValueError,match='cover every'):load_command_policy_references(bad,bank.centers,'cpu')
 bad=copy.deepcopy(cfg);bad['mc_policy_references'].append(refs[0])
 with pytest.raises(ValueError,match='more than once'):load_command_policy_references(bad,bank.centers,'cpu')
 bad=copy.deepcopy(cfg);bad['mc_policy_reference_checkpoint']=str(oldpolicy)
 with pytest.raises(ValueError,match='mixed legacy'):load_command_policy_references(bad,bank.centers,'cpu')
 assert load_command_policy_references({},bank.centers,'cpu') is None
