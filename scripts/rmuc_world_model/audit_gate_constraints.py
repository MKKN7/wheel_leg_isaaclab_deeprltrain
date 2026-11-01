"""Report separate gate violations; never allow better mean reward to offset failures."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();out=[]
for d in a.captures:
 rows=json.loads((d/'dynamic_domain_report.json').read_text())['rows']
 for start in sorted(set(r['start'] for r in rows)):
  events=[]
  for r in rows:
   if r['start']!=start:continue
   if 'vx_event' not in r:events.append(dict(env=r['env'],coverage=False));continue
   e=r['vx_event'];target=e['target'];tol=.01 if target==0 else .1*abs(target)
   ratios=dict(overshoot=e['overshoot_fraction']/.2 if e['overshoot_fraction'] is not None else None,arrival=e['arrival_s']/e['arrival_limit_s'] if e['arrival_s'] is not None else None,speed=e['tail_speed_mae']/tol if e['tail_speed_mae'] is not None else None,yaw=r['yaw_tail_mae']/.05 if r['yaw_tail_mae'] is not None else None,active=e['tail_angle_p95_deg'][0]/2,hip=e['tail_angle_p95_deg'][1]/2)
   violations={k:None if v is None else max(0.,v-1.) for k,v in ratios.items()}
   events.append(dict(env=r['env'],coverage=r['complete'],ratios=ratios,positive_violations=violations,passed=r['full_dynamic_pass']))
  summary={}
  for key in ['overshoot','arrival','speed','yaw','active','hip']:
   values=[e['ratios'][key] for e in events if 'ratios' in e and e['ratios'][key] is not None]
   values=sorted(values,reverse=True);n=max(1,(len(values)+3)//4)
   summary[key]=dict(worst_quartile_mean_ratio=sum(values[:n])/n if values else None,missing=sum('ratios' not in e or e['ratios'][key] is None for e in events),max_ratio=max(values) if values else None)
  out.append(dict(capture=str(d),start=start,summary=summary,events=events))
a.output.write_text(json.dumps(dict(rows=out,scope='Exact event gate components, reported separately; no composite reward score or gate relaxation. Missing arrival is failure, not zero cost. Equality rules remain in original gate.'),indent=2))
print(json.dumps(dict(captures=len(a.captures),events=len(out))))
