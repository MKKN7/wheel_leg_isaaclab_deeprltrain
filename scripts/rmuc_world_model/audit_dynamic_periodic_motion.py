"""Event-local oscillation evidence, separate from task tracking acceptance."""
import argparse,json
from pathlib import Path
import numpy as np
from audit_periodic_motion import periodic_metrics

def analyze(directory):
 spec=json.loads((directory/'command.json').read_text());args=spec['command'];seq=json.loads(Path(args[args.index('--command-sequence')+1]).read_text());duration=float(args[args.index('--episode-length-seconds')+1]);r=json.loads((directory/'evaluation.json').read_text());z=np.load(r['policy_trace']['path']);c={str(k):i for i,k in enumerate(z['columns'])};f=z['frames'];dt=float(np.load(r['leg_tracking_trace']['path'])['dt']);rows=[]
 for env in range(f.shape[1]):
  v=f[:,env];t=v[:,c['episode_step']]*dt;cuts=np.flatnonzero(np.diff(t)<0);end=int(cuts[0]+1) if len(cuts) else len(t);v=v[:end];t=t[:end]
  for j,(start,vx,yaw) in enumerate(seq):
   stop=seq[j+1][0] if j+1<len(seq) else duration
   # Fixed0.5s exclusion is an analysis window choice, not a relaxed arrival gate.
   mask=(t>=start+.5)&(t<stop);window=stop-start-.5
   complete=bool(len(t) and t[-1]>=stop-2*dt)
   row=dict(env=env,start=start,stop=stop,vx=vx,yaw=yaw,window_seconds=window,complete=complete,valid=bool(complete and window>=1.5 and mask.sum()>=int(window/dt)-2),metrics={})
   if row['valid']:
    a=v[mask];signals={k:a[:,c[k]] for k in ['vx','yaw_rate','pitch_deg','action_input_observation_3','wheel_target_right_nm','wheel_target_left_nm']}
    signals['active_difference_deg']=np.rad2deg(a[:,c['joint_position_0']]-a[:,c['joint_position_3']]);signals['hip_difference_deg']=np.rad2deg(a[:,c['joint_position_1']]+a[:,c['joint_position_4']])
    for key,value in signals.items():
     m=periodic_metrics(value,dt);m['resolved_cycles']=m.get('dominant_hz',0)*len(value)*dt;m['at_least_three_cycles']=m['resolved_cycles']>=3;row['metrics'][key]=m
   rows.append(row)
 out=dict(scope='First episode, each command event excluding first0.5s; at least1.5s window, report cycles. Not a standalone pass gate; transient decay, turn geometry and insufficient cycles prevent causal/steady claims. Original arrival/overshoot gates unchanged.',rows=rows,promoted=False)
 (directory/'dynamic_periodic_audit.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();r=analyze(a.directory);print('events',len(r['rows']),'valid',sum(x['valid'] for x in r['rows']))
