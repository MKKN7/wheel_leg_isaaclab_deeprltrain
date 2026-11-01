"""Explicit replay prefix, preserving raw evidence and rejecting crossing windows."""
import math

def prefix_sequences(sequences,seconds,history=4,horizon=32,dt=.008):
    if not math.isfinite(seconds) or seconds<=0:raise ValueError('Positive finite cutoff required')
    cutoff=int(seconds/dt);out=[]
    for source in sequences:
        n=min(cutoff,len(source['state']))
        if n<history+horizon:continue
        q=dict(source)
        for key in ['state','action','observations','sensor_history']:
            if key in q:q[key]=q[key][:n]
        if q['context'].ndim==2:q['context']=q['context'][:n]
        q['starts']=q['starts'][(q['starts']>=history)&(q['starts']+horizon<=n)]
        if len(q['starts']):out.append(q)
    if not out:raise ValueError('No full windows before cutoff')
    return out
