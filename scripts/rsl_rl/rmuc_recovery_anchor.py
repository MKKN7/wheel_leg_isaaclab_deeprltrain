"""Opt-in fixed-reference mean-action loss for recovery rehearsal cohorts.

The mask travels with shuffled observations, never enters the 25-D actor
input, and does not alter PPO rewards, actions, or stored log probabilities.
This is a soft update constraint, not a behavioral retention guarantee.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import torch

MASK_KEY = '_recovery_anchor_mask'


def regularize_masked_mean(mean, reference, mask, coefficient):
    if mean.shape != reference.shape or mask.shape != mean.shape[:-1]+(1,):
        raise ValueError('Recovery anchor mask/action shapes do not match')
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Anchor coefficient must be finite and nonnegative')
    difference=(mean-reference).detach()*mask
    denominator=(mask.sum()*mean.shape[-1]).clamp_min(1)
    gradient=2*coefficient*difference/denominator
    if mean.requires_grad and coefficient:
        mean.register_hook(lambda incoming: incoming+gradient)
    return float(coefficient*difference.square().sum()/denominator)


def configure_recovery_anchor(runner, env, checkpoint, coefficient):
    if not math.isfinite(coefficient) or coefficient<=0:
        raise ValueError('Enabled recovery anchor requires positive coefficient')
    env=getattr(env,'unwrapped',env)
    cohorts=getattr(env,'_recovery_cohorts',None)
    if cohorts is None or env.cfg.recovery_curriculum_enabled:
        raise ValueError('Recovery anchor requires fixed explicit rehearsal cohorts')
    protected=cohorts[:,0]<=15
    if not protected.any():raise ValueError('No protected near-standing rehearsal environments')
    actor=runner.alg.get_policy()
    if actor.is_recurrent or getattr(runner.alg,'symmetry',None):
        raise ValueError('Recovery anchor requires feed-forward PPO without symmetry')
    distribution=actor.distribution
    cached=distribution._distribution;distribution._distribution=None
    try:reference=copy.deepcopy(actor).eval()
    finally:distribution._distribution=cached
    payload=torch.load(checkpoint,map_location=env.device,weights_only=False)
    reference.load_state_dict(payload['actor_state_dict'],strict=True)
    reference.requires_grad_(False)
    provenance=dict(checkpoint=str(Path(checkpoint).resolve()),
        sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        coefficient=coefficient,protected_initial_angles='<=15 degrees',
        protected_envs=int(protected.sum()),total_envs=int(protected.numel()),
        penalty='mean squared raw action difference, normalized over protected samples and six actions')
    directory=Path(runner.logger.log_dir)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'recovery_anchor_contract.json').write_text(json.dumps(provenance,indent=2)+'\n')
    state=dict(updating=False,anchor=None,mask=None,losses=[],samples=0,updates=0)
    start_iteration=int(runner.current_learning_iteration)
    original_forward=actor.forward
    original_distribution_update=distribution.update
    original_update=runner.alg.update

    def forward(obs,masks=None,hidden_state=None,stochastic_output=False):
        state['anchor']=None
        if state['updating'] and stochastic_output and torch.is_grad_enabled():
            state['mask']=obs[MASK_KEY]
            with torch.no_grad():state['anchor']=reference(obs,masks=masks,hidden_state=hidden_state)
        return original_forward(obs,masks=masks,hidden_state=hidden_state,
                                stochastic_output=stochastic_output)

    def distribution_update(mean):
        if state['anchor'] is not None:
            state['losses'].append(regularize_masked_mean(mean,state['anchor'],state['mask'],coefficient))
            state['samples']+=int(state['mask'].sum())
        return original_distribution_update(mean)

    def update(*args,**kwargs):
        observations=runner.alg.storage.observations
        if MASK_KEY in observations.keys():raise ValueError('Anchor mask key collision')
        steps,environments=observations.batch_size
        if environments!=protected.numel():raise ValueError('Rehearsal mask/storage mismatch')
        observations[MASK_KEY]=protected[None,:,None].expand(steps,-1,-1).clone()
        state.update(updating=True,losses=[],samples=0)
        try:result=original_update(*args,**kwargs)
        finally:
            state.update(updating=False,anchor=None,mask=None)
            del observations[MASK_KEY]
        if not state['losses']:raise RuntimeError('Recovery anchor did not enter PPO backward path')
        result['recovery_anchor_loss']=sum(state['losses'])/len(state['losses'])
        with (directory/'recovery_anchor.jsonl').open('a') as f:
            f.write(json.dumps(dict(iteration=start_iteration+state['updates'],
                protected_minibatch_samples=state['samples'],minibatches=len(state['losses']),
                loss=result['recovery_anchor_loss'],reference_sha256=provenance['sha256']))+'\n')
        state['updates']+=1
        return result

    actor.forward=forward
    distribution.update=distribution_update
    runner.alg.update=update
