"""Separate steady signed offset from variation; first complete episode only."""
from pathlib import Path
import argparse,json
import numpy as np

def analyze(directory):
 report=json.loads((directory/'evaluation.json').read_text());z=np.load(report['policy_trace']['path']);c={str(k):i for i,k in enumerate(z['columns'])};f=z['frames'];screen=json.loads((directory/'strict_screen.json').read_text());rows=[]
 for row in screen['rows']:
  if row['yaw']!=0 or not row['complete']:continue
  v=f[:,row['env']];t=v[:,c['episode_step']]*.008;reset=np.flatnonzero(np.diff(t)<0);end=int(reset[0]+1) if len(reset) else len(t);v=v[:end];t=t[:end];v=v[t>=10]
  if len(v)<200:continue
  signals=dict(active_deg=np.rad2deg(v[:,c['joint_position_0']]-v[:,c['joint_position_3']]),hip_deg=np.rad2deg(v[:,c['joint_position_1']]+v[:,c['joint_position_4']]),yaw_rad_s=v[:,c['yaw_rate']],speed_error_mps=v[:,c['vx']]-row['vx'])
  stats={k:dict(signed_mean=float(x.mean()),mean_abs=float(np.abs(x).mean()),std=float(x.std()),p95_abs=float(np.quantile(np.abs(x),.95)),p95_centered_abs=float(np.quantile(np.abs(x-x.mean()),.95))) for k,x in signals.items()}
  rows.append(dict(env=row['env'],vx=row['vx'],screen_pass=row['screen_pass'],metrics=stats))
 summary=[]
 for vx in sorted({x['vx'] for x in rows}):
  a=[x for x in rows if x['vx']==vx];summary.append(dict(vx=vx,replicas=len(a),metrics={k:{m:float(np.median([x['metrics'][k][m] for x in a])) for m in a[0]['metrics'][k]} for k in a[0]['metrics']}))
 out=dict(scope='First completed12s episode,10–12s; centered variation is not proof of periodicity; signed offset and oscillation must both be controlled',summary=summary,rows=rows,promoted=False);(directory/'bias_oscillation_audit.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();analyze(args.directory)
