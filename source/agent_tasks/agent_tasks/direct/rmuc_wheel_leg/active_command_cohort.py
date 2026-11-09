"""Training-only explicit fixed command batches, no inference routing change."""
import math,torch

def validate_targets(values):
    result=[]
    for row in values:
        if len(row)!=2 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in row):raise ValueError('Finite vx/yaw pairs required')
        v,w=map(float,row)
        if abs(v)>3.2 or abs(w)>8:raise ValueError('Command outside physical task envelope')
        result.append((v,w))
    if len(set(result))!=len(result):raise ValueError('Duplicate active targets')
    return tuple(result)

def assign(vx,yaw,rehearsal,continuous,targets,signed=False):
    if not targets:return vx,yaw
    table=torch.as_tensor(targets,device=vx.device,dtype=vx.dtype);selected=table[torch.arange(vx.numel(),device=vx.device)%len(targets)];mask=~rehearsal&~continuous
    return torch.where(mask,selected[:,0],vx),torch.where(mask,selected[:,1] if signed else selected[:,1].abs(),yaw)
