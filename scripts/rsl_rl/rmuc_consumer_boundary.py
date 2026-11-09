"""Opt-in mean-action boundary loss. Never changes sampled/executed actions."""
import json,math
from pathlib import Path
import torch
from rmuc_position_authority import execute_authority_action
from rmuc_target_policy_gradient import target_gate_from_commands

def boundary_loss(mean, observation, targets, weight, plateau=1.):
    if not math.isfinite(weight) or weight<=0:raise ValueError('Positive finite boundary weight required')
    mask=target_gate_from_commands(observation,targets)[...,None]
    boundary=execute_authority_action(observation,mean.detach(),plateau)
    # Detached projection gives exact gradient of squared distance to the box.
    return .5*weight*((mean-boundary).square()*mask).sum()/(mask.sum()*mean.shape[-1]).clamp_min(1)

def configure_consumer_boundary(runner, env, weight):
    if not math.isfinite(weight) or weight<=0:raise ValueError('Positive finite boundary weight required')
    env=env.unwrapped;actor=runner.alg.get_policy();dist=actor.distribution
    targets=env.cfg.command_active_targets
    if not targets or actor.is_recurrent or getattr(runner.alg,'symmetry',None):raise ValueError('Explicit targets and plain feedforward PPO required')
    plateau=getattr(env.cfg,'position_authority_speed_plateau_end',1.)
    state={'updating':False,'obs':None,'losses':[],'index':0}
    old_forward=actor.forward;old_dist=dist.update;old_update=runner.alg.update
    def forward(obs,masks=None,hidden_state=None,stochastic_output=False):
        if state['updating'] and stochastic_output and torch.is_grad_enabled():state['obs']=obs['policy']
        return old_forward(obs,masks=masks,hidden_state=hidden_state,stochastic_output=stochastic_output)
    def distribution(mean):
        if state['updating'] and mean.requires_grad and state['obs'] is not None:
            loss=boundary_loss(mean,state['obs'],targets,weight,plateau)
            mask=target_gate_from_commands(state['obs'],targets)[...,None]
            boundary=execute_authority_action(state['obs'],mean.detach(),plateau)
            grad=(weight*(mean.detach()-boundary)*mask/(mask.sum()*mean.shape[-1]).clamp_min(1)).detach()
            mean.register_hook(lambda g:g+grad)
            state['losses'].append(float(loss.detach()))
        return old_dist(mean)
    path=Path(runner.logger.log_dir)
    (path/'consumer_boundary_contract.json').write_text(json.dumps({'weight':weight,'targets':[list(x) for x in targets],'scope':'Squared mean distance to existing consumer action bounds; no actionmapping/hardwarelimit changes','plateau':plateau},indent=2))
    def update(*a,**kw):
        actor_frozen_at_entry=not any(p.requires_grad for p in actor.parameters())
        state.update(updating=True,obs=None,losses=[])
        try:result=old_update(*a,**kw)
        finally:state.update(updating=False,obs=None)
        if not state['losses'] and not actor_frozen_at_entry:raise RuntimeError('Consumer boundary hook missed PPO update')
        with (path/'consumer_boundary.jsonl').open('a') as f:f.write(json.dumps({'update':state['index'],'loss':sum(state['losses'])/max(1,len(state['losses'])),'calls':len(state['losses']),'actor_frozen_at_entry':actor_frozen_at_entry})+'\n')
        state['index']+=1;return result
    actor.forward=forward;dist.update=distribution;runner.alg.update=update
