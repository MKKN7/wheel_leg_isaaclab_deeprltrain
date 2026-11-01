"""Evaluate unchanged cost versions on real recorded trajectories, without bootstrap."""
import argparse,json
from pathlib import Path
import torch
from rich_replay import load_capture
from train_imagination import stage_objective
from transition_reward import step_metadata,overshoot_cost
from tracking_objective import LEGACY,PHYSICAL
p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
rows=[]
for d in a.captures:
 seq,meta=load_capture(d,4,32,True,True);names=meta['extra_state_names'];ix=lambda key:[16+i for i,n in enumerate(names) if n.startswith(key+':')]
 gates=json.loads((d/'dynamic_domain_report.json').read_text())['rows']
 for version in [LEGACY,PHYSICAL]:
  for start,stop in [(2,4),(4,6),(6,9),(9,12),(12,15),(15,17),(17,20)]:
   per=[]
   for q in seq:
    begin=round(start/.008);end=min(round(stop/.008),len(q['state']))
    if end-begin<round((stop-start)/.008)-2:continue
    x=q['state'][begin:end];c=q['context'][begin:end];event=step_metadata(q['context'][:,-3])[begin:end]
    with torch.no_grad():
     reward,continuation,m=stage_objective(x,c,ix,version)
     penalty=.008*4*overshoot_cost(m['vx'],c[:,-3],event)
     reward=reward-penalty
    gate=next(r for r in gates if r['env']==q['env'] and r['start']==start)
    per.append(dict(env=q['env'],cost=float(-reward.sum()),overshoot_cost=float(penalty.sum()),passed=gate['full_dynamic_pass'],overshoot=gate['vx_event']['overshoot_fraction'],arrival=gate['vx_event']['arrival_s'],steady=gate['steady_joint_pass']))
   rows.append(dict(capture=str(d),version=version,start=start,stop=stop,count=len(per),mean_cost=sum(r['cost'] for r in per)/len(per) if per else None,pass_count=sum(bool(r['passed']) for r in per),environments=per))
a.output.write_text(json.dumps(dict(rows=rows,scope='Undiscounted full event cost on actual simulator post-states, no learned critic. Same-seed paired trajectories; cumulative preceding events affect state. Not a causal gradient test.'),indent=2))
print(json.dumps([{k:v for k,v in r.items() if k!='environments'} for r in rows],indent=2))
