"""Checkpoint-preserving critic expansion and descriptive value diagnostics."""
import torch

def expand_critic_state(state, dimension):
    result={k:v.clone() for k,v in state.items()};w=state['mlp.0.weight']
    if w.shape[1]!=25 or dimension<=25:raise ValueError('Expected25D source critic and expanded input')
    if any('normalizer' in k for k in state):raise ValueError('Normalizer migration requires explicit implementation')
    result['mlp.0.weight']=w.new_zeros((len(w),dimension));result['mlp.0.weight'][:,:25]=w
    return result


def prepare_critic_state(payload, dimension, context_version):
    """Expand a25D critic, or resume only an explicitly matching context."""
    state=payload['critic_state_dict'];source_dimension=state['mlp.0.weight'].shape[1]
    if source_dimension==25:
        return expand_critic_state(state,dimension)
    contract=(payload.get('infos') or {}).get('rmuc_critic_context_contract')
    if source_dimension!=dimension or contract!={'dimension':dimension,'version':context_version}:
        raise ValueError('Expanded critic resume requires matching recorded input context')
    return {k:v.clone() for k,v in state.items()}

def value_diagnostics(values, returns, advantages, action_steps, observations=None, targets=()):
    v,y,a=(x.detach().reshape(-1).float() for x in [values,returns,advantages]);s=action_steps.reshape(-1)
    if len(v)!=len(s):raise ValueError('Delay label alignment mismatch')
    out={}
    masks=[('all',torch.ones_like(s,dtype=torch.bool)),('low',s<=1),('middle',s==2),('high',s>=3)]
    if observations is not None and targets:
        from rmuc_target_policy_gradient import target_gate_from_commands
        if observations.numel()//25!=len(v):raise ValueError('Command diagnostic observation alignment mismatch')
        for vx,yaw in targets:
            mask=target_gate_from_commands(observations,[[vx,yaw]]).reshape(-1)>0
            key=f'command_vx{vx:g}_yaw{yaw:g}'
            masks.append((key,mask))
            for label,delaymask in [('low',s<=1),('middle',s==2),('high',s>=3)]:masks.append((key+'_'+label,mask&delaymask))
    for name,mask in masks:
        if not mask.any():continue
        vv,yy,aa=v[mask],y[mask],a[mask];var=yy.var(unbiased=False)
        out[name]={'n':len(yy),'explained_variance':float(1-(yy-vv).var(unbiased=False)/var) if var>1e-8 else None,'return_target_variance':float(var),'value_residual_mean':float((yy-vv).mean()),'value_residual_rmse':float((yy-vv).square().mean().sqrt()),'advantage_mean':float(aa.mean()),'advantage_std':float(aa.std(unbiased=False))}
    return {'groups':out,'scope':'Pre-update predictions vs bootstrapped GAE returns, not independent Monte Carlo truth; advantage signs are not correctness labels'}
