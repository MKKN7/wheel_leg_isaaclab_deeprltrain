"""Compare learned bootstrap with real H32 returns, not imagined rewards."""
import argparse,json
from pathlib import Path
import torch
from gae import ReturnCritic
from rich_replay import load_capture
from train_imagination import stage_objective
from transition_reward import step_metadata,overshoot_cost,robust_overshoot_cost
p=argparse.ArgumentParser()
for k in ['critic','capture','output']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--overshoot-weight',type=float,default=0.);p.add_argument('--start-seconds',type=float,nargs='+',default=[2.,4.,6.,9.,12.,15.,17.]);a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
v=torch.load(a.critic,map_location='cpu',weights_only=False);sd=v['model'];critic=ReturnCritic(sd['mean'],sd['scale'],sd.get('input_mask'));critic.load_state_dict(sd);critic.eval()
seq,meta=load_capture(a.capture,4,33,True,True);assert v['state_columns']==meta['state_columns']+meta['extra_state_names']
names=meta['extra_state_names'];ix=lambda key:[16+i for i,n in enumerate(names) if n.startswith(key+':')];rows=[]
with torch.no_grad():
 for seconds in a.start_seconds:
  t=round(seconds/.008);h=32;eligible=[q for q in seq if q['env']%4==3 and len(q['state'])>t+h]
  if not eligible:continue
  state=torch.stack([q['state'][t:t+h] for q in eligible],1);context=torch.stack([q['context'][t:t+h] for q in eligible],1)
  trans=torch.stack([step_metadata(q['context'][:,-3])[t:t+h] for q in eligible],1)
  reward,cont,metrics=stage_objective(state.flatten(0,1),context.flatten(0,1),ix,v.get("tracking_version","tolerance_normalized_v1"))
  reward=reward-.008*a.overshoot_weight*(robust_overshoot_cost if v.get('overshoot_loss','quadratic')=='smooth_linear_tail' else overshoot_cost)(metrics['vx'],context[...,-3].flatten(),trans.flatten(0,1));reward=reward.reshape(h,-1);cont=cont.reshape(h,-1)
  start=torch.stack([torch.cat([q['state'][t-1],q['context'][t]]) for q in eligible]);end=torch.stack([torch.cat([q['state'][t+h-1],q['context'][t+h]]) for q in eligible])
  if v.get('event_conditioned',False):
   start=torch.cat([start,torch.stack([step_metadata(q['context'][:,-3])[t] for q in eligible])],-1)
   end=torch.cat([end,torch.stack([step_metadata(q['context'][:,-3])[t+h] for q in eligible])],-1)
  sv=critic(start);ev=critic(end);survival=torch.cat([torch.ones_like(cont[:1]),cont.cumprod(0)],0);discount=v['gamma']**torch.arange(h+1)
  measured=(reward*survival[:-1]*discount[:-1,None]).sum(0)+discount[-1]*survival[-1]*ev
  rows.append(dict(start_seconds=seconds,heldout_envs=len(eligible),predicted_value_mean=float(sv.mean()),physical_bootstrap_target_mean=float(measured.mean()),physical_bellman_mae=float((sv-measured).abs().mean()),physical_discounted_stage_return=float((reward*survival[:-1]*discount[:-1,None]).sum(0).mean())))
a.output.write_text(json.dumps(dict(rows=rows,scope='Same-seed held-out environments; real32step rewards and own terminal bootstrap. Value consistency diagnostic, not exact infinite-horizon value or causal proof.'),indent=2));print(json.dumps(rows,indent=2))
