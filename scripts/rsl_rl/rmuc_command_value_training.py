"""Isolated critic updates with fixed-actor MC calibration replay.

Enabled only by an explicit checkpoint/replay manifest. Actor PPO advantages,
observations, rewards and sampling are unchanged. The physical distribution is
owned by the experiment's environment config, not by this helper.
"""
import copy
import hashlib
import json
from pathlib import Path

import torch

from rmuc_command_value import CommandValueNetwork, protect_inactive_expert_momentum


def validate_command_value_resume(payload, manifest=None):
    """Require explicit extraction/restoration of a calibrated critic sidecar."""
    if 'rmuc_command_value' in payload:
        raise ValueError('Isolated command critic resume requires explicit sidecar migration; do not silently resume the fallback critic')
    migration=payload.get('rmuc_command_value_migration')
    if migration and migration.get('requires_explicit_command_value_training'):
        if manifest is None or manifest.get('critic_sha256')!=migration['critic_sha256']:
            raise ValueError('Migrated actor requires its explicitly extracted calibrated critic manifest')


def mc_policy_drift(mean, reference, commands, samples_per_command=64, std=.03):
    """Fixed-source Gaussian mean KL proxy, grouped by replay command."""
    if mean.shape != reference.shape or mean.shape != (commands*samples_per_command, 6):
        raise ValueError('MC policy probe shape mismatch')
    if not torch.isfinite(mean).all() or not torch.isfinite(reference).all():
        raise ValueError('Nonfinite MC policy probe')
    return (.5*((mean-reference)/std).square().sum(-1)
            .reshape(commands,samples_per_command).mean(-1))


def exact_command_samples(observations, centers):
    physical=torch.stack((observations[:,7]/.5,observations[:,2]/.25+observations[:,8]),-1)
    return ((physical[:,None]-centers).abs().amax(-1)<1e-5).any(-1)


def load_replay(manifest, model, device):
    grid=model.centers.cpu().tolist();xs=[];ys=[];groups=[]
    for entry in manifest['datasets']:
        path=Path(entry['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('MC replay source hash differs')
        data=torch.load(path,map_location='cpu',weights_only=False)
        if float(data['gamma'])!=float(manifest['gamma']):raise ValueError('MC replay discount mismatch')
        minimum_age=float(entry.get('minimum_episode_age_s',6.))
        if minimum_age not in (0.,6.):
            raise ValueError('MC replay phase must explicitly be all ages or steady6s')
        keep=(data['episode_step']*.008>=minimum_age)&(data['tail_error_bounds']<=5)&(data['env_ids']%4>=2)
        obs=data['observations'];labels=model.gates(obs).argmax(-1)
        exact=model.gates(obs).amax(-1)==1
        keep &= exact
        # Only explicitly calibrated commands are admitted. Source manifests
        # bind reward semantics; preserve failed-entry samples when requested.
        # Hard command matching excludes nearby, unvalidated reward contracts.
        physical=torch.stack((obs[:,7]/.5,obs[:,2]/.25+obs[:,8]),-1)
        distance=(physical[:,None]-model.centers.cpu()).abs().amax(-1)
        keep &= distance.amin(-1)<1e-5
        if 'commands' in entry:
            commands=torch.as_tensor(entry['commands'],dtype=physical.dtype)
            if commands.ndim!=2 or commands.shape[1]!=2 or not len(commands) or not torch.isfinite(commands).all():
                raise ValueError('Explicit replay commands must be a nonempty finite Nx2 list')
            keep &= (physical[:,None]-commands).abs().amax(-1).amin(-1)<1e-5
        if manifest.get('mc_policy_references'):
            if not entry.get('commands') or not entry.get('source_policy_sha256') or not data.get('checkpoint'):
                raise ValueError('Mixed-policy replay requires command selection and source provenance')
            if hashlib.sha256(Path(data['checkpoint']).read_bytes()).hexdigest()!=entry['source_policy_sha256']:
                raise ValueError('Replay source policy provenance mismatch')
            for command in entry['commands']:
                matching=[ref for ref in manifest['mc_policy_references']
                          if any(max(abs(float(a)-float(b)) for a,b in zip(command,c))<1e-5
                                 for c in ref['commands'])]
                if len(matching)!=1 or matching[0]['sha256']!=entry['source_policy_sha256']:
                    raise ValueError('Replay command and policy reference disagree')
        xs.append(obs[keep]);ys.append(data['returns'][keep,None]);groups.append(labels[keep])
    x,y,g=torch.cat(xs).to(device),torch.cat(ys).to(device),torch.cat(groups).to(device)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite replay')
    pools=[torch.nonzero(g==i).flatten() for i in range(len(grid))]
    if any(not len(p) for p in pools):raise ValueError('MC calibration lacks a command')
    return x,y,g,pools


def load_command_policy_references(config, centers, device):
    """Explicit per-command MC source policies; optional, old manifests unchanged."""
    entries=config.get('mc_policy_references')
    if entries is None:return None
    if not entries or config.get('mc_policy_reference_checkpoint'):
        raise ValueError('Choose one explicit MC source contract, not mixed legacy and per-command references')
    from verify_rmuc_export import restore_actor
    grid=centers.detach().cpu();owners=[None]*len(grid);cache={}
    for entry in entries:
        path=Path(entry['checkpoint']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=entry['sha256']:raise ValueError('MC per-command source hash mismatch')
        commands=torch.as_tensor(entry['commands'],dtype=grid.dtype)
        if commands.ndim!=2 or commands.shape[1]!=2 or not len(commands) or not torch.isfinite(commands).all():
            raise ValueError('MC source commands must be a finite nonempty Nx2 list')
        if digest not in cache:cache[digest]=restore_actor(path).to(device).eval().requires_grad_(False)
        for command in commands:
            matched=((grid-command).abs().amax(-1)<1e-5).nonzero().flatten()
            if len(matched)!=1 or owners[int(matched[0])] is not None:
                raise ValueError('MC source command unknown or assigned more than once')
            owners[int(matched[0])]=cache[digest]
    if any(x is None for x in owners):raise ValueError('MC source does not cover every trained command')
    if float(config.get('max_mc_policy_kl',0))<=0:raise ValueError('Positive MC policy drift limit required')
    return owners


def validate_actor_critic_coverage(bank, actor_targets):
    """A frozen fallback is not a calibrated critic for an actor repair target."""
    if actor_targets is None:
        return  # Explicit critic-only calibration, with no actor updates.
    if not actor_targets:
        raise ValueError('Actor updates with command critic require explicit target commands')
    targets=torch.as_tensor(actor_targets,device=bank.centers.device,dtype=bank.centers.dtype)
    if targets.ndim!=2 or targets.shape[1]!=2 or not torch.isfinite(targets).all():
        raise ValueError('Invalid actor target commands')
    obs=targets.new_zeros(len(targets),25)
    obs[:,7]=targets[:,0]*.5;obs[:,8]=targets[:,1]
    covered=bank.gates(obs).sum(-1)>=1-1e-6
    if not covered.all():
        missing=targets[~covered].cpu().tolist()
        raise ValueError(f'Uncalibrated actor targets would use frozen fallback critic: {missing}')


def configure_command_value_training(runner, config_path, *, actor_targets=None):
    config=json.loads(Path(config_path).read_text());alg=runner.alg
    if (alg.critic.obs_dim!=25 or alg.critic.is_recurrent or alg.critic.obs_normalization
            or alg.use_clipped_value_loss or alg.is_multi_gpu):
        raise ValueError('Command critic requires25D feedforward/no normalization/unclipped/single GPU')
    if alg.gamma!=config['gamma']:raise ValueError('PPO/MC gamma mismatch')
    init=Path(config['critic_checkpoint'])
    if hashlib.sha256(init.read_bytes()).hexdigest()!=config['critic_sha256']:
        raise ValueError('Isolated critic initialization hash mismatch')
    payload=torch.load(init,map_location='cpu',weights_only=False)
    bank=CommandValueNetwork.from_checkpoint(payload).to(runner.device)
    validate_actor_critic_coverage(bank,actor_targets)
    reference=copy.deepcopy(alg.critic.mlp.state_dict())
    if any(not torch.equal(v,bank.base.state_dict()[k]) for k,v in reference.items()):
        raise ValueError('Fallback critic differs from the resumed source')
    oldids={id(p) for p in alg.critic.parameters()}
    for group in alg.optimizer.param_groups:
        group['params']=[p for p in group['params'] if id(p) not in oldids]
    alg.critic.mlp=bank
    bank.normalize_value_gradients=False
    optimizer=torch.optim.Adam(bank.experts.parameters(),lr=float(config['learning_rate']))
    protect_inactive_expert_momentum(bank,optimizer)
    rx,ry,rb,pools=load_replay(config,bank.cpu(),runner.device)
    bank.to(runner.device)
    rng=torch.Generator(device=runner.device).manual_seed(config['seed'])
    replay_rng=torch.Generator(device=runner.device).manual_seed(config['seed']+100000)
    policy_reference=None
    command_policy_references=load_command_policy_references(config,bank.centers,runner.device)
    if config.get('mc_policy_reference_checkpoint'):
        reference_path=Path(config['mc_policy_reference_checkpoint'])
        if hashlib.sha256(reference_path.read_bytes()).hexdigest()!=config['mc_policy_reference_sha256']:
            raise ValueError('MC actor reference hash mismatch')
        from verify_rmuc_export import restore_actor
        policy_reference=restore_actor(reference_path).to(runner.device).requires_grad_(False)
        if float(config.get('max_mc_policy_kl',0))<=0:
            raise ValueError('Finite positive actor-change bound required for replay use')
    logs=Path(runner.logger.log_dir)
    (logs/'command_value_contract.json').write_text(json.dumps(dict(**config,model=bank.metadata(),
        actor_changed_by_install=False,only_train_envs_in_replay=True,
        critic_gradient_owner='Separate normalized critic optimizer; PPO optimizer contains no critic'),indent=2))
    original_update=alg.update;iteration=0

    def update(*args,**kwargs):
        nonlocal iteration
        x=alg.storage.observations['policy'].detach().reshape(-1,25).clone()
        targets=alg.storage.returns.detach().reshape(-1,1).clone()
        total_samples=len(x)
        if config.get('on_policy_exact_commands_only',False):
            # A ramp passing near a head is not the fixed future command that
            # calibrated that head. The actor uses the same exact-command gate.
            keep=exact_command_samples(x,bank.centers)
            x=x[keep];targets=targets[keep]
            if not len(x):raise RuntimeError('No exact command samples for calibrated critic update')
        # PPO still evaluates these unnormalized values for GAE and reporting,
        # but its optimizer owns no critic parameters.
        result=original_update(*args,**kwargs)
        critic_steps=int(config['epochs'])*int(config['minibatches'])
        losses=[];mc_losses=[]
        active=(bank.gates(x).sum(-1)>0)
        if not active.any():raise RuntimeError('No command critic training samples')
        for epoch in range(int(config['epochs'])):
            order=torch.randperm(len(x),device=x.device,generator=rng)
            for ids in order.tensor_split(int(config['minibatches'])):
                optimizer.zero_grad(set_to_none=True)
                gate=bank.gates(x[ids]);scale=gate@bank.value_scale+(1-gate.sum(-1,keepdim=True))*bank.value_scale.mean()
                prediction=bank(x[ids]);loss=.5*((prediction-targets[ids])/scale).square().mean()
                mc=torch.zeros((),device=x.device)
                if config['replay_weight']:
                    ri=torch.cat([pool[torch.randint(len(pool),(64,),device=x.device,generator=replay_rng)] for pool in pools])
                    mc=((bank(rx[ri])-ry[ri])/bank.value_scale[rb[ri]]).square().mean()
                    loss=loss+config['replay_weight']*mc
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite command critic loss')
                if loss.requires_grad:
                    loss.backward();torch.nn.utils.clip_grad_norm_(bank.experts.parameters(),1.);optimizer.step()
                losses.append(float(loss.detach()));mc_losses.append(float(mc.detach()))
        if any(not torch.equal(v,bank.base.state_dict()[k]) for k,v in reference.items()):
            raise RuntimeError('Frozen fallback critic changed')
        record=dict(update=iteration+1,active_samples=int(active.sum()),total_samples=total_samples,
            critic_update_samples=len(x),on_policy_exact_commands_only=bool(config.get('on_policy_exact_commands_only',False)),critic_steps=critic_steps,
            loss=sum(losses)/len(losses),mc_loss=sum(mc_losses)/len(mc_losses),fallback_changed=False)
        if policy_reference is not None or command_policy_references is not None:
            from tensordict import TensorDict
            probe_ids=torch.cat([pool[torch.linspace(0,len(pool)-1,64,device=pool.device).long()] for pool in pools])
            with torch.no_grad():
                obs=TensorDict({'policy':rx[probe_ids]},batch_size=[len(probe_ids)])
                # Proxy only: fixed-source mean KL at the recorded std=.03.
                reference_mean=(policy_reference(obs) if command_policy_references is None else
                    torch.cat([reference(TensorDict({'policy':rx[probe_ids[i*64:(i+1)*64]]},batch_size=[64]))
                               for i,reference in enumerate(command_policy_references)]))
                kl=mc_policy_drift(alg.get_policy()(obs),reference_mean,len(pools))
            record['mc_source_policy_kl_by_command']=kl.cpu().tolist()
            if float(kl.max())>config['max_mc_policy_kl']:
                with (logs/'command_value_updates.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(**record,stopped='MC policy drift bound exceeded'))+'\n')
                raise RuntimeError('Actor moved beyond the fixed-policy MC replay budget; recollect only affected domains')
        with (logs/'command_value_updates.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        iteration+=1
        return result

    alg.update=update
    original_save=alg.save
    def save():
        saved=original_save()
        saved['rmuc_command_value']=bank.checkpoint()
        saved['rmuc_command_value_optimizer']=optimizer.state_dict()
        saved['rmuc_command_value_config']=copy.deepcopy(config)
        # Standard actor-only evaluators remain compatible. Resuming training
        # must explicitly restore the sidecar instead of silently losing it.
        saved['critic_state_dict']={'mlp.'+k:v for k,v in bank.base.state_dict().items()}
        return saved
    alg.save=save
    return bank
