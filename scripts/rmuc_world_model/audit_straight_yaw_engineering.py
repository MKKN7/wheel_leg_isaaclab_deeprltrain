"""Separate authorized .08rad/s straight yaw screen; never overwrite strict gates."""
import argparse,json
from pathlib import Path
import numpy as np

def analyze(p):
 d=json.loads((p/'highspeed_engineering_screen.json').read_text());z=np.load(p/'evaluation.world_trace.npz');f=z['frames'];c={str(k):i for i,k in enumerate(z['columns'])};dt=float(z['dt']);rows=[]
 for x in d['rows']:
  checks=dict(x['engineering_checks']);straight=abs(x['yaw'])<=1e-6
  if straight:checks['yaw']=x['yaw_tail_mae'] is not None and x['yaw_tail_mae']<=.08
  v=f[:,x['env']];reset=np.flatnonzero(np.diff(v[:,c['pre_age']])<0);v=v[:reset[0]+1] if len(reset) else v
  end=np.flatnonzero((v[:,c['terminated']]>0)|(v[:,c['truncated']]>0));v=v[:end[0]+1] if len(end) else v
  h=np.unwrap(np.r_[v[0,c['pre_heading']],v[:,c['post_heading']]])
  start=np.searchsorted(v[:,c['pre_age']]*dt,6.);tail=h[start:];heading=float(np.rad2deg(tail[-1]-tail[0])) if len(tail)>1 else None
  xy=v[:,[c['post_x'],c['post_y']]];offset=xy-v[0,[c['pre_x'],c['pre_y']]];cross=-np.sin(h[0])*offset[:,0]+np.cos(h[0])*offset[:,1]
  rows.append(dict(env=x['env'],vx=x['vx'],yaw=x['yaw'],strict_pass=x['screen_pass'],angle_engineering_pass=x['engineering_screen_pass'],yaw08_component_pass=all(checks.values()),yaw_mae=x['yaw_tail_mae'],heading_change_6to12_deg=heading,heading_change_episode_deg=float(np.rad2deg(h[-1]-h[0])),cross_track_max_m=float(np.max(np.abs(cross))),full_acceptance=False))
 out={'rows':rows,'strict':sum(x['strict_pass'] for x in rows),'angle_engineering':sum(x['angle_engineering_pass'] for x in rows),'yaw08_component':sum(x['yaw08_component_pass'] for x in rows),'scope':'User allowed modest yaw relaxation; straight only .08rad/s, original .05 retained. Heading and path drift reported, not silently accepted. Turning/yaw rewards unchanged.','promoted':False}
 (p/'straight_yaw08_screen.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();d=analyze(a.directory);print({k:v for k,v in d.items() if k!='rows'})
