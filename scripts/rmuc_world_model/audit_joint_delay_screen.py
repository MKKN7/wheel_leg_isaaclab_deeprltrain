"""Strict first-episode screening; never promotes or substitutes for32rep acceptance."""
import argparse,json
from pathlib import Path
import numpy as np

def analyze(directory):
 report=json.loads((directory/'evaluation.json').read_text());spec=json.loads((directory/'command.json').read_text());args=spec['command'];grid=json.loads(Path(args[args.index('--command-grid')+1]).read_text());duration=float(args[args.index('--episode-length-seconds')+1]);z=np.load(report['leg_tracking_trace']['path']);f=z['frames'];dt=float(z['dt']);buckets=z['bucket_ids'];rows=[]
 # leg trace contract: pre-step episode_step, heading vx, six control q.
 cols=[str(v) for v in z['columns']];assert f.shape[-1]>=8
 policy=np.load(report['policy_trace']['path']);pc={str(k):i for i,k in enumerate(policy['columns'])};pf=policy['frames'];rows=[]
 for env in range(f.shape[1]):
  v=f[:,env];t=v[:,0]*dt;reset=np.flatnonzero(np.diff(t)<0);end=int(reset[0]+1) if len(reset) else len(t);v=v[:end];t=t[:end];vx,yaw=grid[int(buckets[env])];complete=bool(t[-1]>=duration-2*dt);tail=t>=duration-2.;speed=float(np.abs(v[tail,1]-vx).mean()) if tail.any() else None;angles=np.rad2deg(np.stack([v[:,2]-v[:,5],v[:,3]+v[:,6]],-1));angle=np.quantile(np.abs(angles[tail]),.95,axis=0).tolist() if tail.any() else None
  q=pf[:,env];qt=q[:,pc['episode_step']]*dt;reset=np.flatnonzero(np.diff(qt)<0);qe=int(reset[0]+1) if len(reset) else len(qt);q=q[:qe];qt=qt[:qe];ym=qt>=duration-2.;yawerr=float(np.abs(q[ym,pc['yaw_rate']]-yaw).mean()) if ym.any() else None
  speed_limit=.01 if vx==0 else abs(vx)*.1;yaw_limit=.1 if vx<0 and yaw==0 else .05
  checks=dict(complete=complete,speed=speed is not None and speed<speed_limit,yaw=yawerr is not None and yawerr<=yaw_limit,straight_angles=(angle is not None and max(angle)<=2.) if yaw==0 else True)
  rows.append(dict(env=env,bucket=int(buckets[env]),vx=vx,yaw=yaw,complete=complete,speed_tail_mae=speed,speed_limit=speed_limit,yaw_tail_mae=yawerr,yaw_limit=yaw_limit,angle_p95_deg=angle,checks=checks,screen_pass=all(checks.values())))
 summary=[]
 for bucket,(vx,yaw) in enumerate(grid):
  r=[x for x in rows if x['bucket']==bucket];summary.append(dict(vx=vx,yaw=yaw,replicas=len(r),complete=sum(x['complete'] for x in r),strict_screen_pass=sum(x['screen_pass'] for x in r)))
 result=dict(summary=summary,rows=rows,promoted=False,scope='First12s episode, last2s strict speed/yaw/straight angle screening only.2rep insufficient; no drift/arc/step/multi-seed certification. Turn angle gate not applied. Built-in loose evaluator passed field ignored.');(directory/'strict_screen.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();r=analyze(a.directory);print(json.dumps(r['summary'],indent=2))
