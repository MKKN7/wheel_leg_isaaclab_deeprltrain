"""Held-out-environment domain breakdown with physical root-derived metrics."""
from pathlib import Path
import argparse,json,torch
from rich_replay import load_capture
from replay import sample
from dynamics import Dynamics
from queue_schema import align_delay_queues
from observation import rotate,canonical
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--captures',type=Path,nargs='+',required=True);p.add_argument('--pad-unused-delay-queues',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
ck=torch.load(a.model,map_location='cpu',weights_only=False);sd=ck['model'];m=Dynamics(sd['mean'],sd['scale'],sd['delta_scale'],sd['context_mean'],sd['context_scale'],ck['history'],ck['width'],ck.get('depth',2));m.load_state_dict(sd);m.eval();rows=[]
for path in a.captures:
 seq,meta=load_capture(path,4,32,True,ck.get('condition_commands',False))
 if a.pad_unused_delay_queues:seq,meta=align_delay_queues(seq,meta,ck['state_columns'][16:])
 assert ck['state_columns']==meta['state_columns']+meta['extra_state_names']
 val=[q for q in seq if q['env']%4==3];ids=[16+i for i,n in enumerate(meta['extra_state_names']) if n.startswith('root_state_w:')]
 def physical(x):
  r=x[...,ids];q=torch.nn.functional.normalize(r[...,[3,0,1,2]],dim=-1);g=torch.zeros_like(r[...,:3]);g[...,2]=-1;f=rotate(q,g)[...,:2];f=f/f.norm(dim=-1,keepdim=True).clamp_min(1e-6);v=(r[...,4:6]*f).sum(-1);gravity=canonical(q,g);pitch=torch.atan2(gravity[...,0],-gravity[...,2]);yaw=canonical(q,r[...,7:10])[...,2];return v,pitch,yaw
 with torch.no_grad():
  s,ah,act,target,c=sample(val,256,4,32,torch.Generator().manual_seed(883));pred=m.rollout(s,ah,act,c);pv,pp,py=physical(pred);tv,tp,ty=physical(target)
  assert (tv-target[...,1]).abs().max()<1e-4
  rows.append(dict(capture=str(path),heldout_envs=len(val),windows=256,first_episode_only=True,physical_vx_h32_mae=float((pv[:,-1]-tv[:,-1]).abs().mean()),physical_pitch_h32_mae_deg=float(torch.rad2deg(pp[:,-1]-tp[:,-1]).abs().mean()),physical_yaw_rate_h32_mae=float((py[:,-1]-ty[:,-1]).abs().mean()),finite=bool(torch.isfinite(pred).all())))
out=dict(rows=rows,qualified_for_policy_update=False,scope='Held-out envs, recorded-action open-loop32step; no actor closed-loop/gradient certification. Samples from first episode only, including early failure trajectories. No claim for command regions not reached.')
a.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
