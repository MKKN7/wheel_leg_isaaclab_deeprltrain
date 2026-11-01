"""Independent pulse-end validation of a same-policy finite continuation value."""
import argparse,json,torch
from pathlib import Path
from rich_replay import load_capture
from train_imagination import stage_objective
from transition_reward import step_metadata,robust_overshoot_cost
from gae import ReturnCritic
p=argparse.ArgumentParser()
for k in ['baseline','probe','critic','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
base,bm=load_capture(a.baseline,4,250,True,True);probe,pm=load_capture(a.probe,4,250,True,True)
assert bm['checkpoint_sha256']==pm['checkpoint_sha256'];assert bm['extra_state_names']==pm['extra_state_names']
pr=json.loads((a.probe/'evaluation.json').read_text());br=json.loads((a.baseline/'evaluation.json').read_text())
for k in ['seed','initial_state_audit','initial_mass_inertia_readback','initial_material_readback']:assert pr[k]==br[k]
# Start after pulse ends; the same source policy supplies all future actions.
t=round(float(pr['diagnostic_action_bias_end_seconds'])/.008)+1;h=250
sd=torch.load(a.critic,map_location='cpu',weights_only=False)['model'];m=ReturnCritic(sd['mean'],sd['scale'],sd.get('input_mask'));m.load_state_dict(sd);m.eval()
ix=lambda k:[16+i for i,n in enumerate(bm['extra_state_names']) if n.startswith(k+':')]
base={q['env']:q for q in base};probe={q['env']:q for q in probe};rows=[]
with torch.no_grad():
 for e in sorted(set(base)&set(probe)):
  if e%4!=3:continue
  row={'env':e}
  for name,q in [('base',base[e]),('probe',probe[e])]:
   assert t+h<=len(q['state'])
   x=q['state'];c=q['context'];ev=step_metadata(c[:,-3]);r,cont,z=stage_objective(x[t:t+h],c[t:t+h],ix);r=r-.008*4*robust_overshoot_cost(z['vx'],c[t:t+h,-3],ev[t:t+h]);weights=.998**torch.arange(h)*torch.cat([torch.ones(1),cont.cumprod(0)[:-1]])
   row[name+'_actual']=float((r*weights).sum());feat=torch.cat([x[t-1],c[t],ev[t]]);row[name+'_predicted']=float(m(feat[None]))
  row['actual_delta']=row['probe_actual']-row['base_actual'];row['predicted_delta']=row['probe_predicted']-row['base_predicted'];rows.append(row)
summary=dict(n=len(rows),wrong_sign=sum(q['actual_delta']*q['predicted_delta']<0 for q in rows),delta_mae=sum(abs(q['actual_delta']-q['predicted_delta']) for q in rows)/len(rows),actual_abs_delta_mean=sum(abs(q['actual_delta']) for q in rows)/len(rows))
a.output.write_text(json.dumps(dict(summary=summary,rows=rows,start_seconds=t*.008,actor_modified=False,qualified_for_actor_update=False,scope='Same frozen actor after unseen pulse dose; heldout envs; finite2s continuation. No new-policy guarantee.'),indent=2));print(json.dumps(summary,indent=2))
