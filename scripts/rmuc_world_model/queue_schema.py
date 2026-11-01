"""Pad only unused delay-queue tail slots; physical states may never be fabricated."""
import torch

def align_delay_queues(sequences, metadata, target_names):
    old=metadata['extra_state_names']
    if old==target_names:return sequences,metadata
    missing=[n for n in target_names if n not in old]
    extra=[n for n in old if n not in target_names]
    prefixes=('leg_history:','wheel_delay_history:')
    if extra or any(not n.startswith(prefixes) for n in missing):
        raise ValueError('Only zero padding of unused delay queue tails is allowed')
    for prefix,width in [('leg_history:',4),('wheel_delay_history:',2)]:
        source=[n for n in old if n.startswith(prefix)]
        target=[n for n in target_names if n.startswith(prefix)]
        if source!=target[:len(source)] or len(source)%width or len(target)%width:
            raise ValueError('Delay queue channel/order mismatch')
    source_index={n:16+i for i,n in enumerate(old)}
    for q in sequences:
        c=q['context'][0] if q['context'].ndim==2 else q['context']
        # Context 12/13 are actual leg/wheel transport activation.
        if any(n.startswith('leg_history:') for n in missing) and c[12]!=0:raise ValueError('Cannot pad active leg transport')
        if any(n.startswith('wheel_delay_history:') for n in missing) and c[13]!=0:raise ValueError('Cannot pad active wheel transport')
        x=q['state'];out=x.new_zeros(len(x),16+len(target_names));out[:,:16]=x[:,:16]
        for i,name in enumerate(target_names):
            if name in source_index:out[:,16+i]=x[:,source_index[name]]
        q['state']=out
    metadata=dict(metadata,original_extra_state_names=old,extra_state_names=list(target_names),zero_padded_unused_queue_fields=missing)
    return sequences,metadata
