"""Separate model terminal-state error from frozen-value policy mismatch."""
import argparse,json,torch
from pathlib import Path
from rich_replay import load_capture
from dynamics import Dynamics
from gae import ReturnCritic
from train_imagination import stage_objective
from transition_reward import step_metadata,robust_overshoot_cost
p=argparse.ArgumentParser()
for k in ['baseline','candidate','model','critic','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
b,meta=load_capture(a.baseline,4,282,True,True);c,cm=load_capture(a.candidate,4,282,True,True);assert meta['extra_state_names']==cm['extra_state_names'];b={q['env']:q for q in b};c={q['env']:q for q in c}
ck=torch.load(a.model,map_location='cpu',weights_only=False);sd=ck['model'];wm=Dynamics(sd['mean'],sd['scale'],sd['delta_scale'],sd['context_mean'],sd['context_scale'],4,ck['width'],ck['depth']);wm.load_state_dict(sd);wm.eval()
sd=torch.load(a.critic,map_location='cpu',weights_only=False)['model'];value=ReturnCritic(sd['mean'],sd['scale'],sd.get('input_mask'));value.load_state_dict(sd);value.eval();ix=lambda k:[16+i for i,n in enumerate(meta['extra_state_names']) if n.startswith(k+':')];rows=[];t=1000;h=32;future=250
with torch.no_grad():
 for e in sorted(set(b)&set(c)):
  if e%4!=3:continue
  assert torch.equal(b[e]['state'][:t],c[e]['state'][:t]);row={'env':e}
  for label,q in [('base',b[e]),('candidate',c[e])]:
   x,ctx=q['state'],q['context'];ev=step_metadata(ctx[:,-3]);assert t+h+future<=len(x)
   pred=wm.rollout(x[t-4:t][None],q['action'][t-4:t][None],q['action'][t:t+h][None],ctx[t:t+h][None])[0]
   real_feature=torch.cat([x[t+h-1],ctx[t+h],ev[t+h]])
   model_feature=torch.cat([pred[-1],ctx[t+h],ev[t+h]])
   row[label+'_value_real_state']=float(value(real_feature[None]));row[label+'_value_model_state']=float(value(model_feature[None]))
   xx=x[t+h:t+h+future];cc=ctx[t+h:t+h+future];r,cont,z=stage_objective(xx,cc,ix);r=r-.008*4*robust_overshoot_cost(z['vx'],cc[:,-3],ev[t+h:t+h+future]);weights=.998**torch.arange(future)*torch.cat([torch.ones(1),cont.cumprod(0)[:-1]])
   row[label+'_actual_future']=float((r*weights).sum())
  for k in ['value_real_state','value_model_state','actual_future']:row['delta_'+k]=row['candidate_'+k]-row['base_'+k]
  rows.append(row)
summary={k:sum(q['delta_'+k] for q in rows)/len(rows) for k in ['value_real_state','value_model_state','actual_future']}
a.output.write_text(json.dumps(dict(summary=summary,rows=rows,start_seconds=8,terminal_seconds=8.256,scope='Recorded candidate actions over32steps; source-policy value vs real candidate continuation. Value-on-real-state disagreement mixes value error and changed continuation policy; model-vs-real terminal disagreement isolates input-state effects. Not exact policy-gradient certification.',promoted=False),indent=2));print(json.dumps(summary,indent=2))
