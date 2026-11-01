"""First-episode joint vx/yaw diagnostics; no turning symmetry penalty or false certificates."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'artifacts/teacher_reward_audit_20261017'))
from analyze_dynamic_events import event_metrics
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();d=a.directory
spec=json.loads((d/'command.json').read_text());r=json.loads((d/'evaluation.json').read_text());argv=spec['command'];seq=json.loads(Path(argv[argv.index('--command-sequence')+1]).read_text())
if r.get('world_model_action_noise_std',0.):raise ValueError('Exploration capture is not deterministic behavior acceptance')
l=np.load(r['leg_tracking_trace']['path']);pf=np.load(r['policy_trace']['path']);w=np.load(r['world_trajectory']['trace_path']);wc={str(k):i for i,k in enumerate(w['columns'])};pc={str(k):i for i,k in enumerate(pf['columns'])};dt=float(l['dt']);rows=[]
assert l['frames'].shape[:2]==pf['frames'].shape[:2]==w['frames'].shape[:2]
for env in range(l['frames'].shape[1]):
 f=l['frames'][:,env];pol=pf['frames'][:,env];world=w['frames'][:,env];cuts=np.flatnonzero(np.diff(f[:,0])<0)+1;end=int(cuts[0]) if len(cuts) else len(f);f=f[:end];pol=pol[:end];world=world[:end];t=f[:,0]*dt
 for j in range(1,len(seq)):
  start,v,yaw=seq[j];stop=seq[j+1][0] if j+1<len(seq) else 20;m=(t>=start-1e-6)&(t<stop-1e-6);tail=m&(t>=stop-.5-1e-6)
  if not m.any():rows.append(dict(env=env,start=start,complete=False,reason='not reached in first episode'));continue
  terminated=bool(world[m,wc['terminated']].any());complete=bool(t[-1]>=stop-(2 if j==len(seq)-1 else 1)*dt-1e-5 and not terminated)
  cmdvx=pol[:,pc['action_input_observation_7']]/.5
  # Use recorded true command, never reconstruct clipped yaw-error obs.
  parity=max(float(abs(world[m,wc['vx_command']]-v).max()),float(abs(world[m,wc['yaw_command']]-yaw).max()),float(abs(cmdvx[m]-v).max()))
  yawerr=abs(pol[m,pc['yaw_rate']]-yaw);bankerr=abs(pol[m,pc['roll_deg']]-np.rad2deg(np.arctan2(v*yaw,9.81)))
  ev=event_metrics(t,f[:,1],cmdvx,np.rad2deg(f[:,2]-f[:,5]),np.rad2deg(f[:,3]+f[:,6]),start=start,stop=stop,old=seq[j-1][1],target=v,ramp_s=0,dt=dt,final_event=j==len(seq)-1)
  vx_tail=float(abs(f[tail,1]-v).mean()) if tail.any() else None;yt=float(abs(pol[tail,pc['yaw_rate']]-yaw).mean()) if tail.any() else None;bt=float(abs(pol[tail,pc['roll_deg']]-np.rad2deg(np.arctan2(v*yaw,9.81))).mean()) if tail.any() else None
  turn=abs(yaw)>1e-6;tol=.01 if v==0 else .1*abs(v);steady=bool(complete and parity<1e-5 and vx_tail is not None and vx_tail<tol and yt<.05 and (bt<r.get('max_bank_error_deg',3.) if turn else ev['steady_tail_pass']))
  rows.append(dict(env=env,start=start,target_vx=v,target_yaw=yaw,complete=complete,terminated=terminated,command_parity_max=parity,vx_tail_mae=vx_tail,yaw_mae=float(yawerr.mean()),yaw_tail_mae=yt,bank_tail_error_deg=bt,steady_joint_pass=steady,symmetry_gate_applied=not turn,vx_event=ev,full_dynamic_pass=None if turn else bool(steady and ev['step_speed_pass'])))
summary=[]
for start,v,yaw in seq[1:]:
 rr=[x for x in rows if x['start']==start];reached=[x for x in rr if 'vx_event' in x]
 summary.append(dict(start=start,vx=v,yaw=yaw,complete=sum(x['complete'] for x in rr),reached=len(reached),steady_joint_pass=sum(x.get('steady_joint_pass',False) for x in rr),yaw_tail_mae_median=float(np.median([x['yaw_tail_mae'] for x in reached if x['yaw_tail_mae'] is not None])) if any(x['yaw_tail_mae'] is not None for x in reached) else None,full_dynamic_pass=None if yaw else sum(bool(x.get('full_dynamic_pass')) for x in rr)))
out=dict(summary=summary,rows=rows,promoted=False,scope='First episode only; yaw/roll from postphysics trace (8ms relative to pre-action speed). Steady joint gate is diagnostic, not complete dynamic yaw acceptance. No symmetry gate while turning. Missing/reset-shortened events fail coverage; no extrapolation to unvisited speeds.')
(d/'dynamic_domain_report.json').write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))
