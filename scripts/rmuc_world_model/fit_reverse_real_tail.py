"""Finite real-policy tail reference, no self-bootstrap and no actor update.
Not an on-policy critic certificate; only diagnostic 2-second continuation value.
"""
from pathlib import Path
import argparse,json,torch
from rich_replay import load_capture
from train_imagination import stage_objective
from transition_reward import step_metadata,robust_overshoot_cost
from gae import ReturnCritic
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--response-capture',type=Path);p.add_argument('--start-seconds',type=float,default=5.);p.add_argument('--end-seconds',type=float,default=9.);p.add_argument('--command-range',type=float,nargs=2);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2);torch.manual_seed(7891)
seq,meta=load_capture(a.capture,4,250,True,True);ix=lambda k:[16+i for i,n in enumerate(meta['extra_state_names']) if n.startswith(k+':')];train=[];val=[];h=250
for q in seq:
 x=q['state'];c=q['context'];ev=step_metadata(c[:,-3]);r,cont,z=stage_objective(x,c,ix);r=r-.008*4*robust_overshoot_cost(z['vx'],c[:,-3],ev)
 starts=torch.arange(max(4,round(a.start_seconds/.008)),min(round(a.end_seconds/.008),len(x)-h),4)
 if a.command_range is not None:
  lo,hi=a.command_range;starts=starts[(c[starts,-3]>=lo)&(c[starts,-3]<=hi)]
 if not len(starts):continue
 total=torch.zeros(len(starts));discount=torch.ones_like(total)
 for k in range(h):total+=discount*r[starts+k];discount*=.998*cont[starts+k]
 features=torch.cat([x[starts-1],c[starts],ev[starts]],-1)
 (val if q['env']%4==3 else train).append((features,total,starts,q['env']))
x=torch.cat([q[0] for q in train]);y=torch.cat([q[1] for q in train]);vx=torch.cat([q[0] for q in val]);vy=torch.cat([q[1] for q in val]);mu=x.mean(0);scale=x.std(0).clamp_min(.02);mask=(x.amax(0)!=x.amin(0)).float();m=ReturnCritic(mu,scale,mask);opt=torch.optim.Adam(m.parameters(),lr=.0005);ys=y.std().clamp_min(1.)
pair=None
if a.response_capture:
 response,rm=load_capture(a.response_capture,4,250,True,True)
 assert rm['checkpoint_sha256']==meta['checkpoint_sha256'] and rm['extra_state_names']==meta['extra_state_names']
 er=json.loads((a.response_capture/'evaluation.json').read_text());t=round(float(er['diagnostic_action_bias_end_seconds'])/.008)+1
 bd={q['env']:q for q in seq};pairs=[]
 for q in response:
  if q['env']%4==3:continue
  values=[]
  for source in [bd[q['env']],q]:
   assert t+h<=len(source['state'])
   st,ct=source['state'],source['context'];event=step_metadata(ct[:,-3]);rw,co,z=stage_objective(st[t:t+h],ct[t:t+h],ix);rw=rw-.008*4*robust_overshoot_cost(z['vx'],ct[t:t+h,-3],event[t:t+h])
   target=(rw*.998**torch.arange(h)*torch.cat([torch.ones(1),co.cumprod(0)[:-1]])).sum()
   values.append((torch.cat([st[t-1],ct[t],event[t]]),target))
  pairs.append(values)
 pair=(torch.stack([z[0][0] for z in pairs]),torch.stack([z[1][0] for z in pairs]),torch.stack([z[1][1]-z[0][1] for z in pairs]))
for step in range(1000):
 ids=torch.randint(len(x),(512,));pred=m(x[ids]);loss=((pred-y[ids])/ys).square().mean()
 if pair is not None:
  pb,pp,dy=pair;loss=loss+((m(pp)-m(pb)-dy)/dy.std().clamp_min(1.)).square().mean()
 assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10);opt.step()
with torch.no_grad():pred=m(vx);mae=(pred-vy).abs();report=dict(state='diagnostic_finite_tail_fit_complete',horizon=h,seconds=2,source_policy=meta['checkpoint_sha256'],train_windows=len(x),heldout_windows=len(vx),response_capture=str(a.response_capture),heldout_mae=float(mae.mean()),constant_mean_baseline_mae=float((vy-y.mean()).abs().mean()),target_min=float(vy.min()),target_max=float(vy.max()),pred_min=float(pred.min()),pred_max=float(pred.max()),actor_modified=False,qualified_for_actor_update=False,scope='Fixed source-policy finite2s returns; no bootstrap. Does not establish candidate-policy value or beyond2s stability.')
torch.save(dict(model=m.state_dict(),target='real_policy_finite2s',actor_compatible=False),a.output/'diagnostic_value.pt');(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
