"""Recorded-state actor decomposition; not a closed-loop causal certificate."""
import json,sys,argparse,torch,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from verify_rmuc_export import restore_actor
p=argparse.ArgumentParser()
for k in ['capture','checkpoint','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
j=json.loads((a.capture/'evaluation.json').read_text());z=np.load(j['policy_trace']['path']);c={str(k):i for i,k in enumerate(z['columns'])};f=z['frames'];dt=float(z['dt']);actor=restore_actor(a.checkpoint).eval().requires_grad_(False);head=actor.rmuc_physics_adaptation;rows=[]
for lo,hi in [(7,8),(8,8.5),(8.5,9),(9,10),(10,11),(11,12)]:
 chunks=[]
 for e in range(f.shape[1]):
  t=f[:,e,c['episode_step']]*dt;cut=np.flatnonzero(np.diff(t)<0);end=int(cut[0]+1) if len(cut) else len(t);v=f[:end,e];m=(t[:end]>=lo)&(t[:end]<hi);chunks.append(v[m][::4])
 v=np.concatenate(chunks);o=torch.tensor(v[:,[c['action_input_observation_'+str(i)] for i in range(25)]],dtype=torch.float32)
 with torch.no_grad():
  full=actor({'policy':o},stochastic_output=False);old=head.reverse_gate;head.reverse_gate=lambda x:torch.zeros(x.shape[:-1],device=x.device)
  base=actor({'policy':o},stochastic_output=False);head.reverse_gate=old
  plus=o.clone();minus=o.clone();plus[:,6]+=.02;minus[:,6]-=.02
  derivative=(actor({'policy':plus},stochastic_output=False)[:,[2,5]].mean(-1)-actor({'policy':minus},stochastic_output=False)[:,[2,5]].mean(-1))/.04
  output=torch.tanh(head.reverse_network(o));parity=float((full-torch.tensor(v[:,[c['raw_action_'+str(i)] for i in range(6)]])).abs().max())
 rows.append(dict(window=[lo,hi],samples=len(v),actor_parity_max=parity,vx_median=float(np.median(v[:,c['vx']])),pitch_deg_median=float(np.median(v[:,c['pitch_deg']])),obs6_median=float(o[:,6].median()),obs6_clip_fraction=float((o[:,6].abs()>=1.999).float().mean()),wheel_full_mean=full[:,[2,5]].mean(0).tolist(),wheel_without_reverse_mean=base[:,[2,5]].mean(0).tolist(),reverse_effect_mean=(full-base)[:,[2,5]].mean(0).tolist(),d_common_wheel_d_obs6_median=float(derivative.median()),head_tanh_saturation=float((output.abs()>.95).float().mean()),motor_wheel_torque_mean=np.mean(v[:,[c['applied_motor_torque_2'],c['applied_motor_torque_5']]],axis=0).tolist()))
assert max(x['actor_parity_max'] for x in rows)<1e-4
a.output.write_text(json.dumps(dict(rows=rows,scope='Local input sensitivity and branch ablation on recorded states. Wheel sign and pitch balance require physical closed-loop interpretation; no unique causal claim.'),indent=2));print(json.dumps(rows,indent=2))
