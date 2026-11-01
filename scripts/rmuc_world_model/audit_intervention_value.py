"""Separate short physical reward, imagined reward and bootstrap ranking for paired pulses."""
import argparse,json,torch
from pathlib import Path
from rich_replay import load_capture
from dynamics import Dynamics
from gae import ReturnCritic
from train_imagination import stage_objective
from transition_reward import step_metadata,overshoot_cost
from critic_context import features
p=argparse.ArgumentParser()
for key in ['baseline','probe','model','critic','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
b,meta=load_capture(a.baseline,4,32,True,True);q,_=load_capture(a.probe,4,32,True,True);r=json.loads((a.probe/'evaluation.json').read_text());start=float(r['diagnostic_action_bias_start_seconds']);t=round(start/.008);h=32;end=round(13/.008)
ck=torch.load(a.model,map_location='cpu',weights_only=False);sd=ck['model'];m=Dynamics(sd['mean'],sd['scale'],sd['delta_scale'],sd['context_mean'],sd['context_scale'],ck['history'],ck['width'],ck.get('depth',2));m.load_state_dict(sd);m.eval()
v=torch.load(a.critic,map_location='cpu',weights_only=False);sd=v['model'];critic=ReturnCritic(sd['mean'],sd['scale'],sd.get('input_mask'));critic.load_state_dict(sd);critic.eval()
assert ck['state_columns']==v['state_columns']==meta['state_columns']+meta['extra_state_names']
ix=lambda key:[16+i for i,n in enumerate(meta['extra_state_names']) if n.startswith(key+':')]
b={x['env']:x for x in b};q={x['env']:x for x in q};envs=sorted(set(b)&set(q));ctx=torch.stack([b[e]['context'][t:end+1] for e in envs]);event=torch.stack([step_metadata(b[e]['context'][:,-3])[t:end+1] for e in envs]);gamma=v['gamma']

def reward(states):
 n=states.shape[1];c=ctx[:,:n];ev=event[:,:n]
 rw,cont,met=stage_objective(states.flatten(0,1),c.flatten(0,1),ix,v.get('tracking_version','tolerance_normalized_v1'))
 rw=rw-.008*v.get('overshoot_weight',0)*overshoot_cost(met['vx'],c[...,-3].flatten(),ev.flatten(0,1));rw=rw.reshape(len(envs),n);cont=cont.reshape(len(envs),n)
 surv=torch.cat([torch.ones_like(cont[:,:1]),cont.cumprod(1)],1)
 return (rw*surv[:,:-1]*gamma**torch.arange(n)).sum(1),surv[:,-1]

def bootstrap(states,survival):
 n=states.shape[1];f=features(states[:,-1],ctx[:,n],event[:,n],v.get('event_conditioned',False));return gamma**n*survival*critic(f)

scores={}
with torch.no_grad():
 past=torch.stack([b[e]['state'][t-4:t] for e in envs]);ah=torch.stack([b[e]['action'][t-4:t] for e in envs])
 for name,seq in [('baseline',b),('probe',q)]:
  for e in envs:assert torch.equal(b[e]['state'][:t-1],seq[e]['state'][:t-1])
  real=torch.stack([seq[e]['state'][t:end] for e in envs]);act=torch.stack([seq[e]['action'][t:t+h] for e in envs]);pred=m.rollout(past,ah,act,ctx[:,:h]);sr,sv=reward(real[:,:h]);pr,pv=reward(pred);lr,_=reward(real)
  scores[name]={'real_h32_stage':sr,'real_h32_bootstrap':bootstrap(real[:,:h],sv),'model_h32_stage':pr,'model_h32_bootstrap':bootstrap(pred,pv),'real_stop_window_stage':lr}
 delta={k:(scores['probe'][k]-scores['baseline'][k]) for k in scores['baseline']};groups=r['environment_bucket_ids'];cases=json.loads((a.probe/'bias.json').read_text());rows=[]
 for k,case in enumerate(cases):
  mask=torch.tensor([groups[e]==k for e in envs]);row=dict(case=case['name'],n=int(mask.sum()),**{name:float(x[mask].mean()) for name,x in delta.items()});row['model_h32_total']=row['model_h32_stage']+row['model_h32_bootstrap'];row['real_h32_total']=row['real_h32_stage']+row['real_h32_bootstrap'];rows.append(row)
a.output.write_text(json.dumps(dict(rows=rows,horizon=32,real_window=[start,13],critic=str(a.critic),scope='Higher reward delta is better. Recorded post-intervention action streams include policy feedback; critic is approximate and trained on nearby candidate, so this is ranking diagnosis, not proof of policy gradient or counterfactual optimal action.',promoted=False),indent=2));print(json.dumps(rows,indent=2))
