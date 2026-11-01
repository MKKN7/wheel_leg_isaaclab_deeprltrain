"""Offline identification replay. Physical history is training-only, never actor input."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch

STATE_COLUMNS = ['root_z','vx','yaw_rate','roll_deg','pitch_deg','pitch_rate'] + [f'joint_position_{i}' for i in [0,1,3,4]] + [f'joint_velocity_{i}' for i in range(6)]

def valid_starts(age, history, horizon):
    """post-state[t-1] + raw_action[t] -> post-state[t]; no reset crossings."""
    if history<1 or horizon<1:raise ValueError('Positive history/horizon required')
    age=np.asarray(age)
    if age.ndim!=1:raise ValueError('One environment per sequence')
    return np.asarray([i for i in range(history,len(age)-horizon+1)
        if np.all(np.diff(age[i-history:i+horizon])==1)],dtype=np.int64)


def load_report(path, history=4, horizon=8):
    path=Path(path);r=json.loads(path.read_text());trace=Path(r['policy_trace']['path'])
    if hashlib.sha256(trace.read_bytes()).hexdigest()!=r['policy_trace']['sha256']:raise ValueError('Trace hash mismatch')
    z=np.load(trace);f=z['frames'];columns={str(k):i for i,k in enumerate(z['columns'])}
    state=f[:,:,[columns[k] for k in STATE_COLUMNS]].copy();state[:,:,[3,4]]=np.deg2rad(state[:,:,[3,4]])
    action=f[:,:,[columns[f'raw_action_{i}'] for i in range(6)]].copy()
    if float(z['dt'])!=.008:raise ValueError('Expected8ms policy')
    if not np.isfinite(state).all() or not np.isfinite(action).all():raise ValueError('Nonfinite replay')
    n=state.shape[1];mass=r['initial_mass_inertia_readback'];context=np.concatenate([np.asarray(mass['mass_kg']).reshape(n,1),np.asarray(mass['inertia_tensor_kg_m2']).reshape(n,-1)],axis=-1).astype('float32')
    seq=[]
    for env in range(n):
        age=f[:,env,columns['episode_step']]
        # Initial DR readback is only valid in the first episode.
        cuts=np.flatnonzero(np.diff(age)!=1)+1;end=int(cuts[0]) if len(cuts) else len(age)
        if end<history+horizon:continue
        starts=valid_starts(age[:end],history,horizon)
        seq.append(dict(state=torch.from_numpy(state[:end,env]),action=torch.from_numpy(action[:end,env]),context=torch.from_numpy(context[env]),starts=torch.from_numpy(starts),env=env))
    return seq,dict(report=str(path.resolve()),trace_sha256=r['policy_trace']['sha256'],checkpoint_sha256=r['checkpoint_sha256'],seed=r['seed'],state_columns=STATE_COLUMNS,angle_unit='radians after conversion',scope='Partial privileged kinematics, missing lateral velocity/contact and full actuator states. Not Markov-complete; identification only.')


def sample(sequences,batch,history,horizon,generator):
    rows=[]
    for _ in range(batch):
        q=sequences[int(torch.randint(len(sequences),(1,),generator=generator))]
        t=int(q['starts'][int(torch.randint(len(q['starts']),(1,),generator=generator))])
        rows.append((q['state'][t-history:t],q['action'][t-history:t],q['action'][t:t+horizon],q['state'][t:t+horizon],q['context'][t:t+horizon] if q['context'].ndim==2 else q['context']))
    return tuple(torch.stack([r[i] for r in rows]) for i in range(5))
