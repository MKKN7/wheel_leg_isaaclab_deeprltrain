"""PPO integration: frozen legacy actor, isolated router, explicit auxiliary gradients."""
import json
import math
from pathlib import Path
import torch


def select_router_branches(actor, names=None):
    """Freeze the whole actor, then explicitly open only selected direct branches."""
    router = actor.rmuc_state_router
    available = dict(router.named_children())
    selected = tuple(available) if names is None else tuple(names)
    if not selected or len(set(selected)) != len(selected) or set(selected)-set(available):
        raise ValueError(f'Expected unique router branches from {tuple(available)}, got {selected}')
    actor.requires_grad_(False)
    for name in selected:
        available[name].requires_grad_(True)
    return selected


def prepare_router_optimizer(runner, state, learning_rate, critic_rate, train_branches=None):
    from rmuc_command_adapter import install_rmuc_state_router
    if not 0 < learning_rate <= 1e-3 or not 0 < critic_rate <= 1e-3:
        raise ValueError('Router/critic rates must be in (0,1e-3]')
    actor=runner.alg.get_policy()
    if actor.is_recurrent or runner.alg.schedule!='fixed' or getattr(runner.alg,'is_multi_gpu',False):
        raise ValueError('Isolated router currently supports single-device fixed-schedule feedforward PPO')
    router=install_rmuc_state_router(actor,state)
    if router is None:raise ValueError('Explicit router initialization checkpoint is required')
    runner.rmuc_trainable_router_branches = select_router_branches(actor, train_branches)
    runner.alg.optimizer=torch.optim.Adam([
        {'params':[p for p in router.parameters() if p.requires_grad],'lr':learning_rate,'rmuc_group':'state_router',
         'rmuc_branches':list(runner.rmuc_trainable_router_branches)},
        {'params':list(runner.alg.critic.parameters()),'lr':critic_rate,'rmuc_group':'critic'}])


def configure_router_training(runner, coefficient):
    if not math.isfinite(coefficient) or coefficient<0:raise ValueError('Invalid router auxiliary coefficient')
    actor=runner.alg.get_policy();router=actor.rmuc_state_router
    selected = select_router_branches(actor, getattr(runner, 'rmuc_trainable_router_branches', None))
    params=tuple(p for p in router.parameters() if p.requires_grad);pending={};current={'collecting':False,'observation':None,'losses':[]}
    frozen={n:p.detach().clone() for n,p in actor.named_parameters() if not p.requires_grad}
    buffers={n:b.detach().clone() for n,b in actor.named_buffers()}
    path=Path(runner.logger.log_dir)/'state_router_training.jsonl'
    # autograd.grad computes auxiliary derivatives without writing .grad. During
    # that computation hooks must not inject stale auxiliary derivatives.
    def parameter_hook(parameter):
        def hook(gradient):
            extra=pending.get(id(parameter))
            return gradient if current['collecting'] or extra is None else gradient+extra
        return hook
    for parameter in params:parameter.register_hook(parameter_hook(parameter))
    def forward_hook(module, inputs, output):
        if not torch.is_grad_enabled() or not output.requires_grad:return
        observation=inputs[0];current['observation']=observation.detach();pending.clear()
        loss=coefficient*module.balance_loss(observation)
        current['collecting']=True
        try:
            grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        finally:current['collecting']=False
        pending.update({id(p):g.detach() for p,g in zip(params,grads) if g is not None})
        current['losses'].append(float(loss.detach()))
    router.register_forward_hook(forward_hook)
    def before_step(optimizer,args,kwargs):
        if current['observation'] is not None:
            router.isolate_optimizer_step(current['observation'])
    def after_step(optimizer,args,kwargs):
        pending.clear();current['observation']=None
    runner.alg.optimizer.register_step_pre_hook(before_step)
    runner.alg.optimizer.register_step_post_hook(after_step)
    original_update=runner.alg.update
    start_iteration=int(runner.current_learning_iteration)
    update_count=0
    def update(*args,**kwargs):
        nonlocal update_count
        current['losses']=[]
        result=original_update(*args,**kwargs)
        changed=[n for n,p in actor.named_parameters() if n in frozen and not torch.equal(p,frozen[n])]
        changed_buffers=[n for n,b in actor.named_buffers() if not torch.equal(b,buffers[n])]
        if changed or changed_buffers:raise RuntimeError(f'Isolated router changed frozen graph: {changed+changed_buffers}')
        storage=runner.alg.storage.observations['policy']
        from rmuc_residual_safety import rollout_probe
        probe=rollout_probe(storage)
        with torch.no_grad():
            correction=router(probe);positive,negative=router.eligibility(probe)
            loads={}
            for name,branch,mask in router.branches(probe):
                loads[name]={'samples':int(mask.sum()),'probability_mean':branch.probabilities(probe[mask]).mean(0).tolist() if mask.any() else None}
            if not torch.isfinite(correction).all() or not (correction.abs()<=router.correction_bound(probe)+1e-6).all():
                raise RuntimeError('Router correction violated finite action budget')
        record={'iteration':start_iteration+update_count,'aux_coefficient':coefficient,
            'trainable_branches':list(selected),
            'aux_loss_mean':sum(current['losses'])/max(1,len(current['losses'])),
            'aux_minibatches':len(current['losses']),'frozen_graph_changed':changed,'frozen_buffers_changed':changed_buffers,
            'correction_abs_max_per_action':correction.abs().amax(0).tolist(),'router_loads':loads}
        with path.open('a') as f:f.write(json.dumps(record)+'\n')
        update_count+=1
        return result
    runner.alg.update=update
