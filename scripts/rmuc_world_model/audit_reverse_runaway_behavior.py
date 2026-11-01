"""First-episode reverse dynamics; truncated episodes never count as success."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def analyze(directory):
 spec=json.loads((directory/'command.json').read_text());evaluation=json.loads((directory/'evaluation.json').read_text());args=spec['command'];seq=json.loads(Path(args[args.index('--command-sequence')+1]).read_text());trace=Path(evaluation['policy_trace']['path']);z=np.load(trace);f=z['frames'];cols={str(k):i for i,k in enumerate(z['columns'])};dt=float(z['dt']);rows=[]
 assert hashlib.sha256(trace.read_bytes()).hexdigest()==evaluation['policy_trace']['sha256']
 for env in range(f.shape[1]):
  data=f[:,env];time=data[:,cols['episode_step']]*dt;resets=np.flatnonzero(np.diff(time)<0);end=int(resets[0]+1) if len(resets) else len(time);time=time[:end];data=data[:end];vx=data[:,cols['vx']]
  for i,(start,target,yaw) in enumerate(seq):
   if target>=0 or yaw!=0:continue
   stop=seq[i+1][0] if i+1<len(seq) else 20.;mask=(time>=start)&(time<stop);indices=np.flatnonzero(mask)
   if not len(indices):rows.append(dict(env=env,start=start,target=target,reached=False,complete=False));continue
   t=time[mask];v=vx[mask];excess=np.maximum(target-v,0.);acceleration=np.r_[0.,-np.diff(vx)/dt][mask];worsening=(excess>.02)&(acceleration>.1)
   crossed=np.flatnonzero(excess>.02);runaway=np.flatnonzero(excess>.2);complete=bool(time[-1]>=stop-dt*1.1)
   # Common first-second comparison is valid only when fully observed.
   early=(t<start+1.);early_complete=bool(time[-1]>=start+1.-dt*1.1)
   rows.append(dict(env=env,start=start,target=target,reached=True,complete=complete,first_episode_last_time=float(time[-1]),observed_phase_seconds=float(len(t)*dt),peak_excess_mps=float(excess.max()),excess_area_m=float(excess.sum()*dt),worsening_overspeed_seconds=float(worsening.sum()*dt),first_excess_time=float(t[crossed[0]]) if len(crossed) else None,first_runaway_time=float(t[runaway[0]]) if len(runaway) else None,common_first_second_complete=early_complete,common_first_second_mean_excess_mps=float(excess[early].mean()) if early_complete else None,common_first_second_peak_pitch_deg=float(np.abs(data[mask,cols['pitch_deg']][early]).max()) if early_complete else None))
 summary=[]
 for start,target,yaw in seq:
  selected=[r for r in rows if r['start']==start];reached=[r for r in selected if r['reached']]
  if not selected:continue
  item=dict(start=start,target=target,expected=len(selected),reached=len(reached),complete=sum(r['complete'] for r in selected))
  for key in ['first_episode_last_time','peak_excess_mps','worsening_overspeed_seconds','common_first_second_mean_excess_mps','common_first_second_peak_pitch_deg']:
   values=[r[key] for r in reached if r[key] is not None];item['median_'+key]=float(np.median(values)) if values else None
  summary.append(item)
 return dict(checkpoint=evaluation['checkpoint_sha256'],trace_sha256=evaluation['policy_trace']['sha256'],summary=summary,rows=rows,scope='Post-step first episode only; incomplete phases are failures. Integrated/peak metrics across unequal lifetimes are not standalone improvement evidence. First-second comparison requires complete first second. Acceleration from sampled speed is diagnostic, not noise-free.')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);a=parser.parse_args();result=analyze(a.directory);(a.directory/'runaway_behavior_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['summary'],indent=2))
