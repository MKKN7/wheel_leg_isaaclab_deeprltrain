"""Frozen-policy real-return anchor; never emits or modifies actor weights."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import torch
from gae import ReturnCritic
from rich_replay import load_capture
from train_imagination import stage_objective
from transition_reward import step_metadata,overshoot_cost,robust_overshoot_cost
from critic_context import features
p=argparse.ArgumentParser()
for k in ['critic','capture','teacher','output']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--steps',type=int,default=1200);p.add_argument('--prefix-seconds',type=float);p.add_argument('--horizons',type=int,nargs='+',default=[32,128]);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(191)
vc=torch.load(a.critic,map_location='cpu',weights_only=False);sd=vc['model'];critic=ReturnCritic(sd['mean'],sd['scale'],sd.get('input_mask'));critic.load_state_dict(sd);baseline=copy.deepcopy(critic).eval().requires_grad_(False)
assert a.horizons and all(1<=n<=128 for n in a.horizons)
seq,meta=load_capture(a.capture,4,max(a.horizons)+1,True,True);sha=hashlib.sha256(a.teacher.read_bytes()).hexdigest();assert sha==meta['checkpoint_sha256'];assert vc['state_columns']==meta['state_columns']+meta['extra_state_names']
if a.prefix_seconds is not None:
 from replay_window import prefix_sequences
 seq=prefix_sequences(seq,a.prefix_seconds,horizon=max(a.horizons)+1)
 meta['explicit_prefix_seconds']=a.prefix_seconds
names=meta['extra_state_names'];ix=lambda k:[16+i for i,n in enumerate(names) if n.startswith(k+':')]
splits={'train':[],'heldout':[]};gamma=vc['gamma']
with torch.no_grad():
 for q in seq:
  state=q['state'];context=q['context'];event=step_metadata(context[:,-3]);reward,cont,m=stage_objective(state,context,ix,vc.get('tracking_version','tolerance_normalized_v1'));reward-=.008*vc.get('overshoot_weight',0.)*(robust_overshoot_cost if vc.get('overshoot_loss','quadratic')=='smooth_linear_tail' else overshoot_cost)(m['vx'],context[:,-3],event)
  for horizon in a.horizons:
   # Use only complete first-episode windows; t+h must have a real next context.
   t=q['starts'][q['starts']+horizon<len(state)]
   x=features(state[t-1],context[t],event[t],vc.get('event_conditioned',False))
   end=features(state[t+horizon-1],context[t+horizon],event[t+horizon],vc.get('event_conditioned',False))
   target=torch.zeros(len(t));survival=torch.ones(len(t))
   for j in range(horizon):
    target+=gamma**j*survival*reward[t+j];survival*=cont[t+j]
   target+=gamma**horizon*survival*baseline(end)
   splits['heldout' if q['env']%4==3 else 'train'].append((x,target,torch.full((len(t),),horizon)))
data={key:tuple(torch.cat([r[i] for r in rows]) for i in range(3)) for key,rows in splits.items()};x,y,h=data['train'];hx,hy,hh=data['heldout']
opt=torch.optim.Adam(critic.parameters(),lr=1e-4);gen=torch.Generator().manual_seed(192)
def metrics(model):
 with torch.no_grad():
  pred=torch.cat([model(z) for z in hx.split(2048)])
  return {str(n):dict(mae=float((pred[hh==n]-hy[hh==n]).abs().mean()),rmse=float((pred[hh==n]-hy[hh==n]).square().mean().sqrt())) for n in a.horizons}
before=metrics(critic);start=time.monotonic();history=[]
for step in range(a.steps):
 idx=torch.randint(len(x),(2048,),generator=gen);loss=torch.nn.functional.smooth_l1_loss(critic(x[idx]),y[idx],beta=1.)
 if not torch.isfinite(loss):raise RuntimeError('Nonfinite real anchor loss')
 opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);opt.step()
 if step%100==0:
  row=dict(step=step+1,loss=float(loss.detach()),seconds=time.monotonic()-start);history.append(row);print(json.dumps(row),flush=True)
 if time.monotonic()-start>180:break
vc['model']=critic.state_dict();vc['optimizer']=opt.state_dict();vc['real_anchor']=dict(source_teacher_sha256=sha,capture=str(a.capture),horizons=a.horizons,frozen_bootstrap=True)
torch.save(vc,a.output/'critic.pt');after=metrics(critic)
report=dict(source_teacher_sha256=sha,source_critic_sha256=hashlib.sha256(a.critic.read_bytes()).hexdigest(),capture=meta,steps=step+1,seconds=time.monotonic()-start,train_samples=len(x),heldout_samples=len(hx),before=before,after=after,history=history,actor_modified=False,promoted=False,scope='Same-policy real32/128step targets with frozen source bootstrap; only complete first-episode windows; no terminal-zero or reset crossing. Holdout environment IDs, same seed. Fitting return levels does NOT verify value gradients or behavioral improvement.')
(a.output/'report.json').write_text(json.dumps(report,indent=2));(a.output/'command.json').write_text(json.dumps(dict(argv=__import__('sys').argv),indent=2));print(json.dumps(dict(before=before,after=after,seconds=report['seconds']),indent=2))
