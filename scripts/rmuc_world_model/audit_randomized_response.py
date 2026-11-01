"""Local randomized-action response diagnostic; not full closed-loop qualification.

Known zero-mean independent action excitation identifies an average local response.
Per-environment estimates expose sign variation; no iid confidence claim is made
for temporally correlated physics trajectories.
"""
import argparse,json,sys
from pathlib import Path
import torch
from dynamics import Dynamics
from rich_replay import load_capture
from queue_schema import align_delay_queues
from train_imagination import physical
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_rmuc_export import restore_actor

p=argparse.ArgumentParser()
for k in ['model','capture','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
verified=json.loads((a.capture/'excitation_verified.json').read_text());assert verified['passed']
ck=torch.load(a.model,map_location='cpu',weights_only=False);sd=ck['model'];wm=Dynamics(sd['mean'],sd['scale'],sd['delta_scale'],sd['context_mean'],sd['context_scale'],ck['history'],ck['width'],ck.get('depth',2));wm.load_state_dict(sd);wm.eval().requires_grad_(False)
seq,meta=load_capture(a.capture,4,32,True,True);seq,meta=align_delay_queues(seq,meta,ck['state_columns'][16:]);names=meta['extra_state_names'];ix=lambda k:[16+i for i,n in enumerate(names) if n.startswith(k+':')]
cp=Path(json.loads((a.capture/'command.json').read_text())['source_checkpoint']);actor=restore_actor(cp).eval().requires_grad_(False)
def metric(state):
 root,q,dq,vx,yaw,pitch,roll=physical(state,ix)
 return torch.stack([vx,yaw,pitch,q[:,2]-q[:,0],q[:,3]+q[:,1]],-1)
def fit(noise,y):
 x=noise.double()-noise.double().mean(0);y=y.double()-y.double().mean(0)
 return torch.linalg.lstsq(x,y).solution
noise_all=[];real_all=[];pred_all=[];rows=[]
with torch.no_grad():
 for q in seq:
  if q['env']%4!=3:continue
  starts=q['starts'][::4];noises=[];actuals=[];preds=[]
  for t in starts.split(128):
   state=q['state'][t[:,None]+torch.arange(-4,0)];ah=q['action'][t[:,None]+torch.arange(-4,0)];ctx=q['context'][t]
   baseline=actor({'policy':q['observations'][t]},stochastic_output=False);action=q['action'][t];noise=action-baseline
   base_prediction=metric(wm(state,ah,baseline,ctx));prediction=metric(wm(state,ah,action,ctx))
   actual=metric(q['state'][t]);noises.append(noise);actuals.append(actual-base_prediction);preds.append(prediction-base_prediction)
  noise=torch.cat(noises);real=torch.cat(actuals);pred=torch.cat(preds)
  noise_all.append(noise);real_all.append(real);pred_all.append(pred)
  rows.append(dict(env=q['env'],samples=len(noise),physics_coefficient=fit(noise,real).tolist(),model_coefficient=fit(noise,pred).tolist()))
n=torch.cat(noise_all);r=torch.cat(real_all);d=torch.cat(pred_all);rc=fit(n,r);mc=fit(n,d)
out=dict(metrics=['vx_mps','yaw_rad_s','pitch_rad','active_difference_rad','hip_difference_rad'],action_columns=list(range(6)),heldout_envs=len(rows),samples=len(n),physics_coefficient=rc.tolist(),model_coefficient=mc.tolist(),coefficient_mae_per_metric=(rc-mc).abs().mean(0).tolist(),rows=rows,qualified_for_policy_update=False,scope='One-step average local response from independent random action excitation, same seed, heldout envs. Baseline model is a variance-reduction control. Nonlinear/delayed responses and policy feedback require longer real closed-loop validation. No iid confidence interval or true full Jacobian claim.')
a.output.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
