"""Fixed 25-128-128-6 student training. Export remains blocked pending closed-loop gate."""
import argparse, hashlib, json, copy
from pathlib import Path
import torch
from torch import nn
from rmuc_distill_sampling import repair_pools, sample_batch

def validate_resume_teacher(previous, teacher_sha, allow_teacher_change=False):
 if previous and previous['teacher_sha256'] != teacher_sha and not allow_teacher_change:
  raise ValueError('Resume teacher changed; use explicit --allow-teacher-change only with newly relabelled data')

def action_distill_loss(prediction, target, wheel_difference_weight=0.):
 error=prediction-target
 return error.square().mean()+wheel_difference_weight*(error[...,2]-error[...,5]).square().mean()

def main():
 p=argparse.ArgumentParser();p.add_argument('--exclude-negative-vx-commands',action='store_true');p.add_argument('--hidden-dims',type=int,nargs=2,default=[128,128],choices=[128,160,152]);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--extra-dataset',type=Path,nargs='+');p.add_argument('--resume',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--teacher-sha',required=True);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--steps',type=int,default=12000);p.add_argument('--device',default='cuda:0');p.add_argument('--straight-retention-weight',type=float,default=0.);p.add_argument('--repair-dataset',type=Path,nargs='+',default=[]);p.add_argument('--repair-fraction',type=float,default=.1);p.add_argument('--allow-teacher-change',action='store_true',help='Explicit warm start under a new teacher; every dataset shard must carry the new verified teacher hash');p.add_argument('--wheel-difference-weight',type=float,default=0.);a=p.parse_args()
 if not 0<=a.wheel_difference_weight<=10:raise ValueError('Wheel difference weight must be in [0,10]')
 if not 0<=a.straight_retention_weight<=100:raise ValueError('Retention weight must be in[0,100]')
 if a.straight_retention_weight and not a.resume:raise ValueError('Retention requires an explicit frozen resume student')
 torch.manual_seed(723);torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=False)
 paths=sorted(a.dataset.glob('shard_*.pt'));assert len(paths)>=4
 if a.extra_dataset:
  for folder in a.extra_dataset:paths+=sorted(folder.glob('shard_*.pt'))
 data=[torch.load(x,map_location='cpu',weights_only=False) for x in paths]
 assert all(x['checkpoint_sha256']==a.teacher_sha for x in data)
 if a.exclude_negative_vx_commands:
  for shard in data:
   keep=shard['observations'][:,7]>=-1e-7
   shard['observations']=shard['observations'][keep];shard['actions']=shard['actions'][keep]
 # Whole chronological shards held out, never random rows from the same trajectory.
 hold=[x for i,x in enumerate(data) if i%10==9];train=[x for i,x in enumerate(data) if i%10!=9];assert hold
 x=torch.cat([v['observations'] for v in train]);y=torch.cat([v['actions'] for v in train]);vx=torch.cat([v['observations'] for v in hold]);vy=torch.cat([v['actions'] for v in hold]);assert len(x)+len(vx)>=1000000
 previous=torch.load(a.resume,map_location='cpu',weights_only=False) if a.resume else None
 validate_resume_teacher(previous,a.teacher_sha,a.allow_teacher_change)
 retention_mask=(x[:,7]>=0)&(x[:,7]<=.75)&((x[:,2]/.25+x[:,8]).abs()<1e-4)
 retention_hold_mask=(vx[:,7]>=0)&(vx[:,7]<=.75)&((vx[:,2]/.25+vx[:,8]).abs()<1e-4)
 mean=previous['mean'] if previous else x.mean(0);std=previous['std'] if previous else x.std(0).clamp_min(.01);x=((x-mean)/std).to(a.device);y=y.to(a.device);vx=((vx-mean)/std).to(a.device);vy=vy.to(a.device)
 architecture=[25,*a.hidden_dims,6]
 if architecture not in ([25,128,128,6],[25,160,152,6]):raise ValueError('Unsupported student shape')
 if previous and previous['architecture']!=architecture:raise ValueError('Resume architecture mismatch')
 h1,h2=a.hidden_dims
 net=nn.Sequential(nn.Linear(25,h1),nn.ReLU(),nn.Linear(h1,h2),nn.ReLU(),nn.Linear(h2,6)).to(a.device)
 parameter_count=sum(t.numel() for t in net.parameters())
 if previous:net.load_state_dict(previous['student_state_dict'],strict=True)
 reference=copy.deepcopy(net).eval() if a.straight_retention_weight else None
 if reference is not None:
  reference.requires_grad_(False);anchor_x=x[retention_mask.to(a.device)];anchor_vx=vx[retention_hold_mask.to(a.device)];assert len(anchor_x)>0 and len(anchor_vx)>0
 pools=repair_pools(paths,data,a.repair_dataset,a.device)
 opt=torch.optim.Adam(net.parameters(),lr=a.lr);best=float('inf');history=[]
 for step in range(a.steps):
  ids=sample_batch(len(x),pools,fraction=a.repair_fraction,device=a.device);z=x[ids];opt.zero_grad();loss=action_distill_loss(net(z+torch.randn_like(z)*.002),y[ids],a.wheel_difference_weight);
  if reference is not None:
   anchors=anchor_x[torch.randint(len(anchor_x),(1024,),device=a.device)]
   with torch.no_grad():anchor_y=reference(anchors)
   loss=loss+a.straight_retention_weight*(net(anchors)-anchor_y).square().mean()
  loss.backward();opt.step()
  if (step+1)%250==0:
   with torch.no_grad():
    errors=[(net(vx[j:j+4096])-vy[j:j+4096]).square().sum().item() for j in range(0,len(vx),4096)];score=sum(errors)/vy.numel()
    teacher_score=score;retention_score=0.
    wheel_difference_score=sum(((net(vx[j:j+4096])-vy[j:j+4096])[:,2]-(net(vx[j:j+4096])-vy[j:j+4096])[:,5]).square().sum().item() for j in range(0,len(vx),4096))/len(vx) if a.wheel_difference_weight else 0.
    score+=a.wheel_difference_weight*wheel_difference_score
    if reference is not None:
     retention_score=sum((net(anchor_vx[j:j+4096])-reference(anchor_vx[j:j+4096])).square().sum().item() for j in range(0,len(anchor_vx),4096))/(len(anchor_vx)*6)
     score+=a.straight_retention_weight*retention_score
   history.append(dict(step=step+1,validation_mse=teacher_score,wheel_difference_mse=wheel_difference_score,retention_mse=retention_score,selection_objective=score));print(history[-1],flush=True)
   if score<best:
    best=score;torch.save(dict(student_state_dict={k:v.cpu().clone() for k,v in net.state_dict().items()},mean=mean,std=std,teacher_sha256=a.teacher_sha,architecture=architecture,promoted=False),a.output/'student.pt')
 report=dict(wheel_difference_weight=a.wheel_difference_weight,repair_sampling=dict(directories=[str(d) for d in a.repair_dataset],fraction_each=a.repair_fraction,training_pool_rows=[len(v) for v in pools],global_rehearsal_includes_repair_rows=True),straight_retention_weight=a.straight_retention_weight,retention_reference_sha256=hashlib.sha256(a.resume.read_bytes()).hexdigest() if reference is not None else None,state='trained_not_behavior_validated',promoted=False,teacher_sha256=a.teacher_sha,samples=len(x)+len(vx),parameters=parameter_count,weight_bytes=4*parameter_count,exclude_negative_vx_commands=a.exclude_negative_vx_commands,history=history,dataset_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},export_blocked='Requires same-contract closed-loop retention and mandatory survival gates')
 report['resume_provenance']=dict(checkpoint=str(a.resume) if a.resume else None,sha256=hashlib.sha256(a.resume.read_bytes()).hexdigest() if a.resume else None,previous_teacher_sha256=previous['teacher_sha256'] if previous else None,teacher_change_explicit=a.allow_teacher_change)
 (a.output/'training_report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
