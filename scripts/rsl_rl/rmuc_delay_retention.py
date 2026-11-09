"""Training-only source-action KL on low/middle delay and outside target domain.

Labels follow shuffled samples; delay never enters the actor. No action changes,
no hard step rollback, no reward edits. This is a soft retention proxy only.
"""
import json,math,hashlib
from pathlib import Path
import torch
KEY='_rmuc_delay_retention_mask'

def protected_mask(action_steps, target_gate, target_all_delays=False):
    if action_steps.shape!=target_gate.shape:raise ValueError('Delay/gate shape mismatch')
    if target_all_delays:
        return (target_gate<=0)[:,None]
    return ((action_steps<=2)|(target_gate<=0))[:,None]


def retention_delta(mean, reference, observation=None, targets=(), plateau=1.):
    """Raw delta by default; projected mean proxy only on target clipped-reference dimensions."""
    delta=(mean-reference).detach()
    derivative=torch.ones_like(delta)
    if observation is not None:
        from rmuc_position_authority import execute_authority_action
        from rmuc_target_policy_gradient import target_gate_from_commands
        projected_reference=execute_authority_action(observation,reference.detach(),plateau)
        projected_mean=execute_authority_action(observation,mean.detach(),plateau)
        use=(target_gate_from_commands(observation,targets)>0)[...,None] & ((reference.detach()-projected_reference).abs()>1e-6)
        delta=torch.where(use,projected_mean-projected_reference,delta)
        derivative=torch.where(use,(mean.detach()==projected_mean).to(mean.dtype),derivative)
    return delta,derivative

def add_kl_gradient(mean,reference,mask,std,beta,diagnostics,observation=None,targets=(),plateau=1.):
    if mean.shape!=reference.shape or mask.shape!=mean.shape[:-1]+(1,):raise ValueError('Mask/action mismatch')
    if not math.isfinite(beta) or beta<=0 or not torch.isfinite(std).all() or (std<=0).any():raise ValueError('Positive finite KL parameters required')
    delta,derivative=retention_delta(mean,reference,observation,targets,plateau);den=(mask.sum()*mean.shape[-1]).clamp_min(1);variance=std.detach().square()
    grad=beta*delta*derivative*mask/variance/den
    if mean.requires_grad:
        def hook(g):
            n=g.detach().norm();k=grad.norm()
            diagnostics.append({'ppo_norm':float(n),'retention_norm':float(k),'ratio':float(k/n) if n>0 else None})
            return g+grad
        mean.register_hook(hook)
    return float(.5*beta*(delta.square()/variance*mask).sum()/den)

def configure_delay_retention(runner,env,checkpoint,beta=1.,execution_aware=False,target_all_delays=False):
    env=env.unwrapped;actor=runner.alg.get_policy();dist=actor.distribution
    # Keep the mean-retention scale fixed if exploration std becomes trainable.
    retention_std=(dist.std_param if dist.std_type=='scalar' else torch.exp(dist.log_std_param)).detach().clone()
    if actor.is_recurrent or getattr(runner.alg,'symmetry',None) or env._joint_delay_dr is None:raise ValueError('Requires feedforward joint-delay training')
    # The reference owns its checkpoint architecture. A newly scoped actor may
    # contain extra frozen anchors absent from that checkpoint; copying the
    # learner then loading the old state either fails or weakens strict restore.
    from verify_rmuc_export import restore_actor
    reference=restore_actor(Path(checkpoint)).to(env.device).eval().requires_grad_(False)
    p=Path(runner.logger.log_dir);p.mkdir(parents=True,exist_ok=True)
    contract={'reference':str(checkpoint),'sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'beta':beta,'protected':'action delay0/2/4ms OR outside high-straight gate','actor_input':25,'penalty':'beta * average per-action mean distance using fixed initial std; variance-distribution retention is not enforced','behavior_retention':'unproven; paired evaluations mandatory'}
    if getattr(getattr(env,'cfg',None),'command_active_targets',()):
        contract['protected']='action delay0/2/4ms OR outside explicit target commands'
        contract['target_commands']=[list(x) for x in env.cfg.command_active_targets]
    if target_all_delays:
        if not getattr(env.cfg,'command_active_targets',()):
            raise ValueError('All-delay target learning requires explicit command targets')
        contract['protected']='Outside explicit target commands at all delays; target behavior retention must be evaluated'
    contract['target_all_delays']=target_all_delays
    contract['execution_aware_reference_saturated_targets']=execution_aware
    contract['mean_retention_std_fixed']=retention_std.cpu().tolist()
    if execution_aware:contract['penalty']='Raw Gaussian mean KL except reference-saturated target dimensions: projected consumer mean distance/std2 proxy; not censored-distribution KL'
    (p/'delay_retention_contract.json').write_text(json.dumps(contract,indent=2))
    labels=[];state={'updating':False,'anchor':None,'mask':None,'losses':[],'gradients':[],'update':0}
    old_act=runner.alg.act;old_update=runner.alg.update;old_forward=actor.forward;old_dist=dist.update
    def act(obs):
        targets=getattr(env.cfg,'command_active_targets',()) if hasattr(env,'cfg') else ()
        if targets:
            from rmuc_target_policy_gradient import target_gate_from_commands
            tolerance=1e-5 if getattr(env.cfg,'command_reverse_entry_ramp_s',0)>0 else 1e-3
            gate=target_gate_from_commands(obs['policy'],targets,tolerance=tolerance)
        else:
            gate=actor.rmuc_physics_adaptation.high_straight_gate(obs['policy'])
        labels.append(protected_mask(env._joint_delay_dr.action.steps,gate,target_all_delays).detach().clone())
        return old_act(obs)
    def forward(obs,masks=None,hidden_state=None,stochastic_output=False):
        state['anchor']=None
        if state['updating'] and stochastic_output and torch.is_grad_enabled():
            state['mask']=obs[KEY]
            state['policy_obs']=obs['policy']
            with torch.no_grad():state['anchor']=reference(obs,masks=masks,hidden_state=hidden_state)
        return old_forward(obs,masks=masks,hidden_state=hidden_state,stochastic_output=stochastic_output)
    def distribution(mean):
        if state['anchor'] is not None:
            state['losses'].append(add_kl_gradient(mean,state['anchor'],state['mask'],retention_std,beta,state['gradients'],
                state.get('policy_obs') if execution_aware else None,
                getattr(env.cfg,'command_active_targets',()),
                getattr(env.cfg,'position_authority_speed_plateau_end',1.)))
        return old_dist(mean)
    def update(*args,**kwargs):
        obs=runner.alg.storage.observations;mask=torch.stack(labels);labels.clear()
        if KEY in obs.keys() or tuple(mask.shape[:2])!=tuple(obs.batch_size):raise ValueError('Retention rollout alignment mismatch')
        obs[KEY]=mask;state.update(updating=True,losses=[],gradients=[])
        try:result=old_update(*args,**kwargs)
        finally:state.update(updating=False,anchor=None,mask=None);del obs[KEY]
        if not state['losses']:raise RuntimeError('Retention hook never reached PPO means')
        record={'update':state['update'],'protected_samples':int(mask.sum()),'total_samples':mask.numel(),'loss':sum(state['losses'])/len(state['losses']),'gradient_diagnostics':state['gradients']}
        with (p/'delay_retention.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        state['update']+=1;return result
    runner.alg.act=act;runner.alg.update=update;actor.forward=forward;dist.update=distribution
