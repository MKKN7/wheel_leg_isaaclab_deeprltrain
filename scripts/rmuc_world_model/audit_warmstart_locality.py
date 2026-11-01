"""Read-only audit of pulse-target fitting away from its labeled phase."""
import argparse,json,sys
from pathlib import Path
import torch,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from verify_rmuc_export import restore_actor
p=argparse.ArgumentParser()
for k in ['capture','source','candidate','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
src=restore_actor(a.source).eval().requires_grad_(False);candidate=restore_actor(a.candidate).eval().requires_grad_(False)
j=json.loads((a.capture/'evaluation.json').read_text());z=np.load(j['policy_trace']['path']);f=z['frames'];cols={str(k):i for i,k in enumerate(z['columns'])};dt=float(z['dt']);obs=[];times=[];envs=[]
for e in range(f.shape[1]):
 t=f[:,e,0]*dt;cut=np.flatnonzero(np.diff(t)<0);end=int(cut[0]+1) if len(cut) else len(t)
 obs.append(torch.tensor(f[:end,e][:,[cols['action_input_observation_'+str(k)] for k in range(25)]]));times.append(t[:end]);envs.append(np.full(end,e))
x=torch.cat(obs);t=np.concatenate(times);env=np.concatenate(envs)
with torch.no_grad():delta=torch.cat([candidate({'policy':q},stochastic_output=False)-src({'policy':q},stochastic_output=False) for q in x.split(2048)])
rows=[]
for split in ['train','heldout']:
 for lo,hi in [(7,8),(8,8.5),(8.5,8.756),(8.756,9.5),(9.5,10.5)]:
  m=(t>=lo)&(t<hi)&((env%4==3) if split=='heldout' else (env%4!=3));d=delta[m]
  if not len(d):continue
  rows.append(dict(split=split,window=[lo,hi],samples=len(d),mean_delta=d.mean(0).tolist(),mean_common_active=float(d[:,[0,3]].mean()),mean_common_wheel=float(d[:,[2,5]].mean()),p95_abs_delta=torch.quantile(d.abs(),.95,dim=0).tolist()))
report=dict(rows=rows,scope='Same recorded observations, first episode only. Fitting uses phase labels; actor receives no time. Off-window response is function generalization, not proof of closed-loop benefit.',promoted=False)
a.output.write_text(json.dumps(report,indent=2));print(json.dumps([{k:v for k,v in row.items() if k not in ['mean_delta','p95_abs_delta']} for row in rows],indent=2))
