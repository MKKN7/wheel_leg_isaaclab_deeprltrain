"""Explicit repair quotas over training-only rows; no dataset duplication."""
from pathlib import Path
import torch

def repair_pools(paths, data, repair_dirs, device):
    dirs=[Path(d).resolve() for d in repair_dirs]
    if len(set(dirs))!=len(dirs):raise ValueError('Duplicate repair directory')
    pools=[[] for _ in dirs];offset=0
    for i,(path,shard) in enumerate(zip(paths,data)):
        if i%10==9:continue  # Exactly the trainer's held-shard split.
        n=len(shard['observations'])
        parent=Path(path).resolve().parent
        if parent in dirs:pools[dirs.index(parent)].append(torch.arange(offset,offset+n,device=device))
        offset+=n
    result=[]
    for directory,parts in zip(dirs,pools):
        if not parts or not sum(len(x) for x in parts):raise ValueError(f'No training rows for repair directory: {directory}')
        result.append(torch.cat(parts))
    return result

def sample_batch(total, pools, batch_size=4096, fraction=.1, device='cpu'):
    if not 0<=fraction<1 or len(pools)*fraction>=1:raise ValueError('Repair quotas must leave global rehearsal')
    quota=round(batch_size*fraction)
    ids=[torch.randint(total,(batch_size-quota*len(pools),),device=device)]
    ids += [pool[torch.randint(len(pool),(quota,),device=device)] for pool in pools]
    return torch.cat(ids)
