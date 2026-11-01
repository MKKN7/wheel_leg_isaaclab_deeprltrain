"""Fit bounded-scope MC bootstrap cost on frozen teacher's final stopping event."""
import argparse,json,hashlib,time
from pathlib import Path
import torch
from rich_replay import load_capture
from value import StopValue,discounted_cost_to_go,stopping_stage_cost
from observation import rotate,canonical
p=argparse.ArgumentParser()
for k in ['capture','output']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--overshoot-weight',type=float,default=0.);p.add_argument('--condition-commands',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--cost-profile',choices=['legacy','settling_v1'],default='legacy');p.add_argument('--gamma',type=float,default=.99);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(27)
seq,meta=load_capture(a.capture,4,32,True,a.condition_commands);names=meta['extra_state_names'];ix=lambda k:[16+i for i,n in enumerate(names) if n.startswith(k+':')]
xtr=[];ytr=[];xval=[];yval=[];gamma=a.gamma;dt=.008
assert 0<gamma<=1
for s in seq:
 x=s['state'][2125:2499] # stop command17s, exclude time-limit reset transition
 r=x[:,ix('root_state_w')];q=torch.nn.functional.normalize(r[:,[3,0,1,2]],dim=-1);g=torch.zeros_like(r[:,:3]);g[:,2]=-1;f=rotate(q,g)[:,:2];f=f/f.norm(dim=-1,keepdim=True).clamp_min(1e-6);vx=(r[:,4:6]*f).sum(-1);g=canonical(q,g);pitch=torch.atan2(g[:,0],-g[:,2]);qp=x[:,ix('joint_pos')];joint=torch.atan2(qp[:,:38],qp[:,38:]);leg=joint[:,2]-joint[:,0]
 yaw=canonical(q,r[:,7:10])[:,2];hip=joint[:,3]+joint[:,1]
 cost=stopping_stage_cost(vx,pitch,leg,a.overshoot_weight,dt,profile=a.cost_profile,yaw=yaw,hip=hip)
 # V(x_t) predicts subsequent-step cost, not cost already incurred at x_t.
 future=discounted_cost_to_go(cost,gamma);target=torch.cat([future[1:],torch.zeros_like(future[:1])])
 context=s['context'][2125:2499] if s['context'].ndim==2 else s['context'][None].expand(len(x),-1)
 assert len(x)==374 and len(context)==374,'Final stop must be fully captured without reset crossing'
 if s['context'].ndim==2:assert (context[:,-3:-1]==0).all(),'Value model only supports final straight stop'
 inputs=torch.cat([x,context],-1)
 xx,yy=(xval,yval) if s['env']%4==3 else (xtr,ytr);xx.append(inputs);yy.append(target)
device=torch.device(a.device);x=torch.cat(xtr).to(device);y=torch.cat(ytr).to(device);v=torch.cat(xval).to(device);vy=torch.cat(yval).to(device);mean=x.mean(0);scale=x.std(0).clamp_min(.02);ts=y.square().mean().sqrt().clamp_min(.0001)
m=StopValue(mean,scale,ts).to(device);opt=torch.optim.Adam(m.parameters(),lr=.001);start=time.monotonic()
for step in range(1600):
 ids=torch.randint(len(x),(256,),device=device);pred=m(x[ids]);loss=((pred-y[ids])/ts).square().mean();opt.zero_grad();
 if not torch.isfinite(loss):raise RuntimeError('Nonfinite value objective')
 loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
with torch.no_grad():
 pv=m(v);rmse=(pv-vy).square().mean().sqrt();rel=rmse/vy.square().mean().sqrt();r2=1-(pv-vy).square().sum()/((vy-vy.mean()).square().sum())
ck=a.output/'stop_value.pt';torch.save(dict(cost_profile=a.cost_profile,condition_commands=a.condition_commands,state_columns=meta['state_columns']+meta.get('extra_state_names',[]),model=m.state_dict(),gamma=gamma,dt=dt,overshoot_weight=a.overshoot_weight,scope='Frozen-teacher final17-20s stop, finite-horizon MC target; no general skill value claims'),ck)
report=dict(cost_profile=a.cost_profile,gamma=gamma,device=str(device),condition_commands=a.condition_commands,overshoot_weight=a.overshoot_weight,steps=1600,seconds=time.monotonic()-start,held_out_envs=8,held_out_rmse=float(rmse),held_out_relative_rmse=float(rel),held_out_r2=float(r2),qualified_for_diagnostic=bool(rel<.15),qualified_for_policy_update=False,source=meta,value_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),scope='Same-seed held-out envs, frozen teacher MC tail. Terminal timeout is finite diagnostic horizon, not physical termination. Not GAE critic or production reward replacement.')
(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='source'},indent=2))
