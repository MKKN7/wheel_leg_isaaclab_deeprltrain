"""Rank complete actual stop trajectories, separating objective terms and phases."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from tracking_objective import speed_cost,LEGACY,PHYSICAL
from transition_reward import overshoot_cost
p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2);rows=[]
for d in a.captures:
 report=json.loads((d/'evaluation.json').read_text());ref=report['policy_trace'];path=Path(ref['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256'];z=np.load(path);f=torch.tensor(z['frames']);cols=list(z['columns']);dt=float(z['dt']);spec=json.loads((d/'command.json').read_text());cmd=spec['command'];schedule=json.loads(Path(cmd[cmd.index('--command-sequence')+1]).read_text());duration=float(cmd[cmd.index('--episode-length-seconds')+1])
 for event in range(1,len(schedule)):
  start,target,yaw=schedule[event];old=schedule[event-1][1]
  if target!=0 or yaw!=0 or old==0:continue
  stop=schedule[event+1][0] if event+1<len(schedule) else duration
  for env in range(f.shape[1]):
   data=f[:,env];t=data[:,cols.index('episode_step')]*dt;cuts=torch.nonzero(t[1:]<t[:-1]).flatten();end=int(cuts[0])+1 if len(cuts) else len(t);data=data[:end];t=t[:end];mask=(t>=start-1e-6)&(t<stop-1e-6)
   if int(mask.sum())<round((stop-start)/dt)-2:continue
   x=data[mask];age=t[mask]-start;get=lambda k:x[:,cols.index(k)];vx=get('vx');pitch=torch.deg2rad(get('pitch_deg'));roll=torch.deg2rad(get('roll_deg'));active=get('joint_position_0')-get('joint_position_3');hip=get('joint_position_1')+get('joint_position_4');hub=lambda v:torch.sqrt(1+v.square())-1
   eventinfo=vx.new_tensor([-np.sign(old),abs(old)])[None].expand(len(vx),-1)
   survival=torch.sigmoid((get('root_z')-.12)/.015)*torch.sigmoid((1-pitch.abs())/.05)*torch.sigmoid((1.1-roll.abs())/.05)
   terms={'speed':dt*speed_cost(vx,torch.zeros_like(vx),LEGACY),'yaw':dt*hub(get('yaw_rate')/.05),'leg':dt*.25*(hub(active/np.deg2rad(2))+hub(hip/np.deg2rad(2))),'posture':dt*.1*(hub(torch.relu(pitch.abs()-np.deg2rad(5))/.05)+hub(torch.relu(roll.abs()-np.deg2rad(5))/.05)),'survival':10*(1-survival),'overshoot':dt*4*overshoot_cost(vx,torch.zeros_like(vx),eventinfo),'constant':torch.full_like(vx,-2*dt)}
   phases={}
   for name,m in [('initial',age<.256),('pitch_transition',(age>=.256)&(age<.5)),('rebound',(age>=.5)&(age<1.2)),('settle',age>=1.2),('all',torch.ones_like(age,dtype=torch.bool))]:
    discount=.998**torch.arange(len(vx));phases[name]={k:float((value[m]*discount[m]).sum()) for k,value in terms.items()};phases[name]['total']=sum(phases[name].values())
   rows.append(dict(capture=str(d),event=start,old_vx=old,env=env,phases=phases,reverse_peak=float(torch.relu(-np.sign(old)*vx).max()),reverse_distance=float(torch.relu(-np.sign(old)*vx).sum()*dt)))
summary=[]
for capture,event in sorted(set((x['capture'],x['event']) for x in rows)):
 rr=[x for x in rows if x['capture']==capture and x['event']==event];summary.append(dict(capture=capture,event=event,n=len(rr),mean_reverse_peak=float(np.mean([x['reverse_peak'] for x in rr])),mean_reverse_distance=float(np.mean([x['reverse_distance'] for x in rr])),phase_mean_cost={phase:{term:float(np.mean([x['phases'][phase][term] for x in rr])) for term in rr[0]['phases'][phase]} for phase in rr[0]['phases']}))
a.output.write_text(json.dumps(dict(summary=summary,rows=rows,scope='Lower physical discounted stage cost is better; same implemented reward, no learned bootstrap. Fixed sequence/seed only; does not establish causal gradient validity.'),indent=2));print(json.dumps([{k:v for k,v in x.items() if k!='phase_mean_cost'}|{'cost':x['phase_mean_cost']['all']} for x in summary],indent=2))
