"""Exact current task cost on real post-step policy traces, no learned bootstrap."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from tracking_objective import speed_cost,LEGACY,PHYSICAL
from transition_reward import step_metadata,overshoot_cost
p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2);rows=[]
for d in a.captures:
 r=json.loads((d/'evaluation.json').read_text());path=Path(r['policy_trace']['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['policy_trace']['sha256'];z=np.load(path);cols={str(k):i for i,k in enumerate(z['columns'])};f=torch.tensor(z['frames']);world=r['world_trajectory'];wp=Path(world['trace_path']);assert hashlib.sha256(wp.read_bytes()).hexdigest()==world['trace_sha256'];w=np.load(wp);wc={str(k):i for i,k in enumerate(w['columns'])};cmd=torch.tensor(w['frames'][:,:,[wc['vx_command'],wc['yaw_command']]])
 gates=json.loads((d/'dynamic_domain_report.json').read_text())['rows']
 for version in [LEGACY,PHYSICAL]:
  for start,stop in [(2,4),(4,6),(6,9),(9,12),(12,15),(15,17),(17,20)]:
   costs=[]
   for env in range(f.shape[1]):
    age=f[:,env,cols['episode_step']];cuts=torch.nonzero(age[1:]-age[:-1]!=1).flatten();end=int(cuts[0])+1 if len(cuts) else len(age)
    begin=round(start/.008);finish=min(round(stop/.008),end)
    if finish-begin<round((stop-start)/.008)-2:continue
    x=f[begin:finish,env];c=cmd[begin:finish,env];get=lambda k:x[:,cols[k]]
    vx=get('vx');yaw=get('yaw_rate');pitch=torch.deg2rad(get('pitch_deg'));roll=torch.deg2rad(get('roll_deg'));active=get('joint_position_0')-get('joint_position_3');hip=get('joint_position_1')+get('joint_position_4');straight=c[:,1].abs()<.05
    huber=lambda v:torch.sqrt(1+v.square())-1
    tracking=speed_cost(vx-c[:,0],c[:,0],version)+huber((yaw-c[:,1])/.05)
    symmetry=(huber(active/np.deg2rad(2))+huber(hip/np.deg2rad(2)))*straight
    posture=huber(torch.relu(pitch.abs()-np.deg2rad(5))/.05)+huber(torch.relu(roll.abs()-np.deg2rad(5))/.05)*straight
    survival=torch.sigmoid((get('root_z')-.12)/.015)*torch.sigmoid((1-pitch.abs())/.05)*torch.sigmoid((1.1-roll.abs())/.05)
    reward=.008*(2-tracking-.25*symmetry-.1*posture)-10*(1-survival)
    metadata=step_metadata(cmd[:end,env,0])[begin:finish];reward-=.008*4*overshoot_cost(vx,c[:,0],metadata)
    gate=next(v for v in gates if v['env']==env and v['start']==start)
    costs.append(dict(env=env,cost=float(-reward.sum()),passed=gate['full_dynamic_pass']))
   rows.append(dict(capture=str(d),version=version,start=start,stop=stop,count=len(costs),mean_cost=sum(v['cost'] for v in costs)/len(costs) if costs else None,pass_count=sum(bool(v['passed']) for v in costs),environments=costs))
a.output.write_text(json.dumps(dict(rows=rows,scope='Real post-state trace cost, same20s schedule; no critic. Validate against rich-state path before drawing conclusions.'),indent=2));print('Wrote',a.output)
