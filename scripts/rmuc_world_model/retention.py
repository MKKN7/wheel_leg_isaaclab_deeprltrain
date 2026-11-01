"""Offline phase labels only; never append episode time to deployed observations."""
import torch

def phase_labels(episode_age,dt=.008):
    seconds=episode_age.to(torch.float64)*dt
    # 0=ordinary rehearsal;1=initial idle;2=all stop tails;3=validated early final stop.
    labels=torch.zeros_like(episode_age,dtype=torch.long)
    labels[seconds<2]=1
    for start,stop in [(4.4,6.),(12.4,15.),(17.4,20.)]:
        labels[(seconds>=start-1e-9)&(seconds<stop-1e-9)]=2
    labels[(seconds>=17.-1e-9)&(seconds<17.4-1e-9)]=3
    return labels

def balanced_indices(labels,count_per_group,generator):
    groups=[]
    for k in range(4):
        pool=torch.nonzero(labels==k).flatten()
        if not len(pool):raise ValueError(f'Missing phase{k}')
        groups.append(pool[torch.randint(len(pool),(count_per_group,),generator=generator,device=labels.device)])
    return torch.cat(groups)

def phase_maxima(difference,labels):
    return {str(k):float(difference[labels==k].abs().max()) for k in range(4)}

def phase_accepts(maxima):
    return all(maxima[str(k)]<=limit for k,limit in enumerate([.01,.0005,.0005,.01]))
