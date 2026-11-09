"""Opt-in diagnostics from weighted reward terms through GAE and actor updates.

One-term counterfactuals hold states, actions, termination and critic fixed.
They measure local credit paths, not the return of a counterfactual policy.
"""
import json
import os
from pathlib import Path

import torch

TERMS = {
    'reverse_runaway': ('penalty_reverse_runaway', 'command_reverse_runaway_weight'),
    'speed_oscillation': ('penalty_command_speed_oscillation', 'command_speed_oscillation_penalty_weight'),
    'reverse_pitch_oscillation': ('penalty_reverse_pitch_oscillation', 'command_reverse_pitch_oscillation_weight'),
}


def penalty_reward_effect(unclipped, weighted_cost, low, high):
    if unclipped.shape != weighted_cost.shape or not torch.isfinite(unclipped).all() or not torch.isfinite(weighted_cost).all():
        raise ValueError('Finite matching reward and cost required')
    if (weighted_cost < 0).any() or not low < high:
        raise ValueError('Nonnegative cost and ordered reward limits required')
    return unclipped.clamp(low, high) - (unclipped + weighted_cost).clamp(low, high)


def fixed_value_gae_effect(reward_effect, dones, gamma, lam):
    """Exact GAE difference when only recorded rewards change; fixed bootstrap."""
    if reward_effect.shape != dones.shape or reward_effect.ndim != 2:
        raise ValueError('Expected matching time x environment matrices')
    result=torch.empty_like(reward_effect);carry=torch.zeros_like(reward_effect[0])
    for step in range(len(reward_effect)-1,-1,-1):
        carry=reward_effect[step]+gamma*lam*(~dones[step].bool())*carry
        result[step]=carry
    return result


def normalized_advantage_effect(raw_advantage, component):
    """Include global normalization, using the same sample std as RSL-RL."""
    normalize=lambda x:(x-x.mean())/(x.std()+1e-8)
    return normalize(raw_advantage)-normalize(raw_advantage-component)


def cosine(a,b):
    denominator=a.norm()*b.norm()
    return float(torch.dot(a,b)/denominator) if denominator>1e-12 else None


def score_gradients(actor, observations, old_mean, score, advantages, parameters, chunk_size):
    """All selected rows, accumulated at rollout batch size (TF32 parity).

    A single giant GEMM can choose a different TF32 kernel than collection.
    Chunking preserves the collection precision and bounds diagnostic memory;
    it does not subsample or change the command-mean reduction.
    """
    from tensordict import TensorDict
    size=len(observations)
    if size<=0 or chunk_size<=0:raise ValueError('Positive sample and chunk counts required')
    vectors={k:torch.zeros(sum(p.numel() for p in parameters),device=observations.device)
             for k in advantages}
    predictions=[];largest=0.
    for start in range(0,size,chunk_size):
        end=min(start+chunk_size,size);chunk=observations[start:end]
        # Keep the final GEMM at the collection batch size as well. This actor
        # is feedforward and sample independent; padded rows never enter loss.
        if len(chunk)<chunk_size:
            chunk=torch.cat((chunk,chunk[:1].expand(chunk_size-len(chunk),-1)))
        mean=actor(TensorDict({'policy':chunk},batch_size=[len(chunk)]))[:end-start]
        error=float((mean.detach()-old_mean[start:end]).abs().max());largest=max(largest,error)
        if not torch.allclose(mean.detach(),old_mean[start:end],atol=2e-4,rtol=2e-4):
            raise RuntimeError(f'Captured means do not match pre-update actor: chunk={start}:{end}, max_error={error}')
        scores=(mean*score[start:end]).sum(-1)
        for j,(label,adv) in enumerate(advantages.items()):
            scalar=(scores*adv[start:end].detach()).sum()/size
            gradients=torch.autograd.grad(scalar,parameters,retain_graph=j+1<len(advantages),allow_unused=True)
            vectors[label]+=torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                      for p,g in zip(parameters,gradients)])
        predictions.append(mean.detach())
    return vectors,torch.cat(predictions),largest


def configure_reward_credit_audit(runner, env):
    """Read-only training instrumentation. Install after other update wrappers."""
    env=env.unwrapped;alg=runner.alg;actor=alg.get_policy()
    if actor.is_recurrent or getattr(alg,'normalize_advantage_per_mini_batch',False) or getattr(alg,'symmetry',None):
        raise ValueError('Reward credit audit requires plain feedforward/global-advantage PPO')
    if not env.cfg.command_active_targets:
        raise ValueError('Explicit audit target commands required')
    from rmuc_target_policy_gradient import target_gate_from_commands
    from rmuc_position_authority import execute_authority_action
    from tensordict import TensorDict
    path=Path(runner.logger.log_dir);frames=[];update_count=0
    process=alg.process_env_step;original_update=alg.update
    terms={name:pair for name,pair in TERMS.items()
           if name in ('reverse_runaway','speed_oscillation') or getattr(env.cfg,pair[1],0)>0}
    weights={name:float(getattr(env.cfg,weight)) for name,(_,weight) in terms.items()}
    if weights['speed_oscillation']<=0:
        raise ValueError('Audited anti-oscillation training requires a positive penalty weight')
    if any(vx<0 and abs(yaw)<1e-6 for vx,yaw in env.cfg.command_active_targets) and weights['reverse_runaway']<=0:
        raise ValueError('Audited reverse targets require an enabled runaway penalty')
    low=float(env.cfg.reward_clip_min);high=float(env.cfg.reward_clip_max)
    max_samples=int(os.environ.get('RMUC_REWARD_CREDIT_MAX_SAMPLES','0'))
    if max_samples<0:raise ValueError('Credit sample limit must be nonnegative; zero means all target samples')
    capture_update=int(os.environ.get('RMUC_CAPTURE_PPO_ROLLOUT_UPDATE','1'))
    contract={'version':'full_target_credit_v3_chunked','gradient_max_samples':max_samples,
              'weights':weights,'limits':[low,high],'targets':[list(x) for x in env.cfg.command_active_targets],
              'scope':'Read-only: exact fixed-critic reward-removal GAE difference; sampled pre-update PPO score gradient; actual optimizer displacement. Not a causal policy rollout or guarantee of convergence.'}
    (path/'reward_credit_contract.json').write_text(json.dumps(contract,indent=2))

    def step(obs,rewards,dones,extras):
        metrics=extras['episode']
        raw=metrics['reward_unclipped'].detach().clone()
        if not torch.allclose(raw.clamp(low,high),rewards,atol=2e-5,rtol=1e-6):
            raise RuntimeError('Reward capture does not match environment output before timeout bootstrap')
        costs={name:metrics[metric].detach().clone()*weights[name] for name,(metric,_) in terms.items()}
        frames.append((raw,costs,metrics['command_aligned_speed'].detach().clone(),
                       metrics['command_filtered_aligned_speed'].detach().clone()))
        return process(obs,rewards,dones,extras)

    def update(*args,**kwargs):
        nonlocal update_count
        storage=alg.storage;obs=storage.observations['policy']
        if len(frames)!=len(obs):raise RuntimeError('Reward/rollout timestamps are misaligned')
        raw=torch.stack([x[0] for x in frames]);dones=storage.dones.squeeze(-1).bool()
        aligned_speed=torch.stack([x[2] for x in frames])
        raw_adv=storage.returns.squeeze(-1)-storage.values.squeeze(-1)
        normalized=(raw_adv-raw_adv.mean())/(raw_adv.std()+1e-8)
        if not torch.allclose(normalized,storage.advantages.squeeze(-1),atol=2e-4,rtol=2e-4):
            raise RuntimeError('Advantage normalization differs; refusing misleading credit attribution')
        parameter_pairs=[(n,p) for n,p in actor.named_parameters() if p.requires_grad]
        parameters=[p for _,p in parameter_pairs]
        before=[p.detach().clone() for p in parameters]
        term_effects={};term_adv={};cost_matrices={}
        for name in terms:
            cost=torch.stack([x[1][name] for x in frames]);cost_matrices[name]=cost
            effect=penalty_reward_effect(raw,cost,low,high)
            contribution=fixed_value_gae_effect(effect,dones,alg.gamma,alg.lam)
            term_effects[name]=effect;term_adv[name]=normalized_advantage_effect(raw_adv,contribution)
        flat_obs=obs.detach().reshape(-1,25).clone()
        old_mean,old_std=[x.detach().reshape(-1,6) for x in storage.distribution_params]
        actions=storage.actions.detach().reshape(-1,6)
        score=(actions-old_mean)/old_std.square()
        rows=[];pending=[]
        for command in env.cfg.command_active_targets:
            tolerance=1e-5 if getattr(env.cfg,'command_reverse_entry_ramp_s',0)>0 else 1e-3
            mask=target_gate_from_commands(obs,[command],tolerance=tolerance)>0
            ids=mask.reshape(-1).nonzero().flatten()
            if not len(ids):continue
            sampled=(ids if max_samples==0 else
                     ids[torch.linspace(0,len(ids)-1,min(max_samples,len(ids)),device=ids.device).long()])
            groups={};gradient_vectors={}
            if parameters:
                gradient_vectors,mean,mean_error=score_gradients(actor,flat_obs[sampled],old_mean[sampled],score[sampled],
                    {k:v.detach().reshape(-1)[sampled] for k,v in {'full':normalized,**term_adv}.items()},
                    parameters,chunk_size=obs.shape[1])
                plateau=getattr(env.cfg,'position_authority_speed_plateau_end',1.)
                executed=execute_authority_action(flat_obs[sampled],mean,plateau)
                saturation=((mean.detach()-executed.detach()).abs()>1e-6).float().mean(0).tolist()
            else:saturation=None
            for name in terms:
                cost=cost_matrices[name][mask];effect=term_effects[name][mask]
                vector=gradient_vectors.get(name)
                groups[name]={'weight':weights[name],'active_fraction':float((cost>0).float().mean()),
                    'weighted_cost_mean':float(cost.mean()),'weighted_cost_max':float(cost.max()),
                    'after_clip_effect_mean':float(effect.mean()),
                    'positive_cost_but_zero_effect_fraction':float(((cost>0)&(effect==0)).float().mean()),
                    'normalized_advantage_effect_rms':float(term_adv[name][mask].square().mean().sqrt()),
                    'actor_parameter_gradient_norm':float(vector.norm()) if vector is not None else None,
                    'cosine_with_full_surrogate':cosine(vector,gradient_vectors['full']) if vector is not None else None}
            row={'command':list(command),'samples':len(ids),'gradient_samples':len(sampled),'actor_frozen':not bool(parameters),
                 'mean_replay_max_error':mean_error if parameters else None,
                 'consumer_mean_clip_fraction':saturation,'terms':groups,
                 'full_surrogate_gradient_norm':float(gradient_vectors['full'].norm()) if parameters else None,
                 'gradient_reduction':'Mean within this command for full and each term; actual PPO uses full minibatch reduction'}
            reverse_excess=mask&(aligned_speed>abs(command[0])+.05)&(command[0]<0)
            row['reverse_overspeed_samples']=int(reverse_excess.sum())
            row['reverse_overspeed_with_zero_penalty_samples']=int((reverse_excess&(cost_matrices['reverse_runaway']==0)).sum())
            rows.append(row);pending.append((row,gradient_vectors))
        frames.clear()
        snapshot=None
        if update_count+1==capture_update:
            snapshot={'observations':obs.detach().cpu().clone(),'actions':storage.actions.detach().cpu().clone(),
                'means':old_mean.cpu().clone(),'std':old_std.cpu().clone(),'dones':dones.cpu().clone(),
                'unclipped_reward':raw.detach().cpu().clone(),'raw_advantage':raw_adv.detach().cpu().clone(),
                'normalized_advantage':normalized.detach().cpu().clone(),
                'weighted_costs':{k:v.cpu().clone() for k,v in cost_matrices.items()},
                'after_clip_reward_effects':{k:v.cpu().clone() for k,v in term_effects.items()},
                'normalized_advantage_effects':{k:v.cpu().clone() for k,v in term_adv.items()},
                'actor_state_before':{k:v.detach().cpu().clone() for k,v in actor.state_dict().items()},
                'trainable_parameter_names':[n for n,_ in parameter_pairs],
                'gradient_vectors_by_command':[{k:v.cpu().clone() for k,v in vectors.items()} for _,vectors in pending],
                'gamma':alg.gamma,'lambda':alg.lam,'contract':contract}
        result=original_update(*args,**kwargs)
        if parameters:
            displacement=torch.cat([(p.detach()-old).flatten() for p,old in zip(parameters,before)])
            for row,vectors in pending:
                row['actual_parameter_update_norm']=float(displacement.norm())
                for name in terms:
                    row['terms'][name]['cosine_actual_update_with_term_ascent']=cosine(displacement,vectors[name])
        if snapshot is not None:
            snapshot['actor_state_after']={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
            snapshot['actual_parameter_displacement']=displacement.cpu().clone() if parameters else None
            snapshot['rows']=rows
            output=path/'reward_credit_snapshot.pt'
            if output.exists():raise RuntimeError('Refusing to overwrite reward credit snapshot')
            torch.save(snapshot,output)
        update_count+=1
        with (path/'reward_credit_audit.jsonl').open('a') as f:
            f.write(json.dumps({'update':update_count,'rows':rows,'unbroken_chain_certified':False,
                'limitation':'Nonzero sampled gradients do not guarantee helpful executed behavior. Adam, clipping, retention and other objectives can rotate the final update; behavior gates remain required.'})+'\n')
        return result
    alg.process_env_step=step;alg.update=update
