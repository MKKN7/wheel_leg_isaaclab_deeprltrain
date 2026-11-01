"""Project the actual optimizer step, not its preconditioner input gradient.

Only a first-order local no-increase check. Nonlinear imagined and real behavior
still require independent acceptance. Never interprets this as a safety proof.
"""
import torch

def project_update(delta, gradients, iterations=80, tolerance=1e-7):
    if delta.ndim!=1 or gradients.ndim!=2 or gradients.shape[1]!=delta.numel():
        raise ValueError('Constraint gradient shape mismatch')
    if not torch.isfinite(delta).all() or not torch.isfinite(gradients).all():
        raise ValueError('Nonfinite optimizer proposal')
    norms=gradients.norm(dim=1)
    active=norms>1e-12
    g=gradients[active]/norms[active,None]
    out=delta.clone()
    if not len(g):return out,dict(active=0,fallback_zero=False)
    for _ in range(iterations):
        for row in g:
            out=out-torch.relu(torch.dot(row,out))*row
        if (g@out).max()<=tolerance:break
    # Numerical nonconvergence must not silently permit constraint increase.
    fallback=bool((g@out).max()>tolerance)
    if fallback:out.zero_()
    original_norm=delta.norm();new_norm=out.norm()
    if new_norm>original_norm:out=out*(original_norm/new_norm.clamp_min(1e-20))
    return out,dict(active=int(active.sum()),fallback_zero=fallback,proposal_norm=float(original_norm),applied_norm=float(out.norm()),max_normalized_directional_increase=float((g@out).max()))

def late_phase(command,dt=.008):
    """Training metadata: true after step arrival budget, resets on every change.

    Smooth ramps do not gain a relaxed steady gate; this is only discrete-step
    local-constraint selection. Final physical evaluation is unchanged.
    """
    if command.ndim!=1:raise ValueError('One trajectory required')
    out=torch.ones_like(command,dtype=torch.bool)
    changes=(command[1:]-command[:-1]).abs()>1e-6
    start=0;limit=0.
    for i in range(len(command)):
        if i and bool(changes[i-1]):
            jump=abs(float(command[i]-command[i-1]));start=i;limit=1.5 if jump<=.5+1e-6 else 2.
        out[i]=(i-start)*dt>=limit
    return out

def upper_quartile(values):
    if not values.numel():return values.sum()
    return values.topk(max(1,(values.numel()+3)//4)).values.mean()
