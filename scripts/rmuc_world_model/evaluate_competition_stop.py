"""Re-score saved deterministic stop trajectories, never overwrite strict evidence."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from competition_stop import stop_metrics
p=argparse.ArgumentParser();p.add_argument('captures',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();results=[]
for d in a.captures:
 r=json.loads((d/'evaluation.json').read_text());assert not r.get('world_model_action_noise_std',0.)
 strict=json.loads((d/'dynamic_domain_report.json').read_text());ref=r['leg_tracking_trace'];path=Path(ref['path'])
 if 'sha256' in ref:assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256']
 z=np.load(path);f=z['frames'];dt=float(z['dt']);rows=[]
 for row in strict['rows']:
  if row['target_vx']!=0 or row['target_yaw']!=0 or 'vx_event' not in row:continue
  ev=row['vx_event'];old=ev['old']
  if old==0:continue
  x=f[:,row['env']];cuts=np.flatnonzero(np.diff(x[:,0])<0);x=x[:cuts[0]+1] if len(cuts) else x
  t=x[:,0]*dt;mask=(t>=ev['start_s']-1e-6)&(t<ev['stop_s']-1e-6)
  m=stop_metrics(t[mask]-ev['start_s'],x[mask,1],old,dt,row['complete'],row.get('terminated',True),ev['tail_angle_p95_deg'],row['yaw_tail_mae'],row['command_parity_max'])
  pre=(t>=ev['start_s']-.5-1e-6)&(t<ev['start_s']-1e-6)
  pre_error=float(np.abs(x[pre,1]-old).mean()) if pre.sum()>=round(.45/dt) else None
  m['checks']['entry_speed_reached']=pre_error is not None and pre_error<=.1*abs(old)
  m['passed']=all(m['checks'].values())
  rows.append(dict(env=row['env'],start=row['start'],old_vx=old,pre_stop_speed_mae=pre_error,strict_pass=row['full_dynamic_pass'],**m))
 summary=[]
 for start in sorted(set(x['start'] for x in rows)):
  rr=[x for x in rows if x['start']==start];summary.append(dict(start=start,old_vx=rr[0]['old_vx'],episodes=len(rr),strict_pass=sum(x['strict_pass'] for x in rr),competition_pass=sum(x['passed'] for x in rr),failures={k:sum(not x['checks'][k] for x in rr) for k in rr[0]['checks']}))
 results.append(dict(capture=str(d),checkpoint_sha256=r['checkpoint_sha256'],summary=summary,rows=rows))
a.output.write_text(json.dumps(dict(version='competition_stop_20261028_v3',authorization='User explicitly relaxed stop precision toward stable quick stopping',limits=dict(speed_band=.05,hold_s=.3,arrival_s_small=1.5,arrival_s_large=2.,max_reverse_speed=.15,max_continuous_reverse_s=.5,reverse_threshold_mps=.05,tail_speed_mae=.03,leg_p95_deg=2,yaw_tail_mae=.05),results=results,promoted=False,scope='Saved single-seed fixed-sequence stop evidence only; no new dynamics or deployment certification. Strict reports preserved. Non-stop command gates unchanged.'),indent=2));print(json.dumps([{k:v for k,v in x.items() if k!='rows'} for x in results],indent=2))
