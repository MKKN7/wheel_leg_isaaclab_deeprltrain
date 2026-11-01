"""Privileged step metadata for training reward only; never an actor observation."""
import torch

def step_metadata(command, minimum_jump=.025):
    """Identify discrete steps, not smooth ramps. Each event ends at next command change.

    Input [T] in m/s, output [T,2]: signed direction and absolute step size.
    A command must remain unchanged the following tick to count as a step.
    """
    if command.ndim != 1:raise ValueError('One trajectory required')
    out=command.new_zeros(len(command),2)
    changes=torch.nonzero((command[1:]-command[:-1]).abs()>1e-6).flatten()+1
    for j,start in enumerate(changes.tolist()):
        stop=int(changes[j+1]) if j+1<len(changes) else len(command)
        jump=command[start]-command[start-1]
        if stop-start>=2 and abs(float(jump))>=minimum_jump:
            out[start:stop,0]=jump.sign();out[start:stop,1]=jump.abs()
    return out

def overshoot_cost(vx,target,metadata,margin=.18):
    direction,size=metadata.unbind(-1)
    excess=torch.relu((vx-target)*direction-margin*size)
    return (excess/(.1*size).clamp_min(.01)).square()*(size>0)


def robust_overshoot_cost(vx,target,metadata,margin=.18):
    """Same overshoot boundary and local quadratic curvature; linear far tail.

    Training objective only. Does not relax the actual overshoot/behavior gate.
    """
    direction,size=metadata.unbind(-1)
    excess=torch.relu((vx-target)*direction-margin*size)
    z=excess/(.1*size).clamp_min(.01)
    # Rational form avoids cancellation near zero.
    return (2*z.square()/(torch.sqrt(1+z.square())+1))*(size>0)
