"""User-authorized relaxed reverse yaw review. Preserve original strict report."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();out=a.directory/'reverse_practical_gate.json';assert not out.exists()
j=json.loads((a.directory/'dynamic_domain_report.json').read_text());rows=[]
for x in j['rows']:
 if x.get('target_vx',0)>=0 or abs(x.get('target_yaw',0))>1e-6:continue
 ev=x['vx_event'];steady=bool(x['complete'] and x['command_parity_max']<1e-5 and x['vx_tail_mae']<.1*abs(x['target_vx']) and x['yaw_tail_mae']<=.1 and ev['steady_tail_pass'])
 rows.append(dict(env=x['env'],start=x['start'],target_vx=x['target_vx'],complete=x['complete'],strict_steady=x['steady_joint_pass'],practical_steady=steady,practical_dynamic=bool(steady and ev['step_speed_pass']),overshoot_fraction=ev['overshoot_fraction'],arrival_s=ev['arrival_s'],yaw_tail_mae=x['yaw_tail_mae']))
summary=[]
for t in sorted(set(x['start'] for x in rows)):
 rr=[x for x in rows if x['start']==t];summary.append(dict(start=t,target_vx=rr[0]['target_vx'],reached=len(rr),complete=sum(x['complete'] for x in rr),strict_steady=sum(x['strict_steady'] for x in rr),practical_steady=sum(x['practical_steady'] for x in rr),practical_dynamic=sum(x['practical_dynamic'] for x in rr)))
out.write_text(json.dumps(dict(yaw_gate=.1,strict_yaw_gate=.05,scope='Negative vx and yaw0 only. No relaxation of speed/leg/step/complete gates. Original strict report unchanged. Not deployment certification.',priority=['limit overshoot and runaway','complete survival','reverse speed tracking'],summary=summary,rows=rows,promoted=False),indent=2));print(json.dumps(summary,indent=2))
