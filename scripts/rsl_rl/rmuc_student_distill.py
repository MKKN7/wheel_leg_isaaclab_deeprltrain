"""Fixed 25-128-128-6 student training. Export remains blocked pending closed-loop gate."""
import argparse, hashlib, json
from pathlib import Path
import torch
from torch import nn

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--extra-dataset',type=Path,nargs='+');p.add_argument('--resume',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--teacher-sha',required=True);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--steps',type=int,default=12000);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 torch.manual_seed(723);torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=False)
 paths=sorted(a.dataset.glob('shard_*.pt'));assert len(paths)>=4
 if a.extra_dataset:
  for folder in a.extra_dataset:paths+=sorted(folder.glob('shard_*.pt'))
 data=[torch.load(x,map_location='cpu',weights_only=False) for x in paths]
 assert all(x['checkpoint_sha256']==a.teacher_sha for x in data)
 # Whole chronological shards held out, never random rows from the same trajectory.
 hold=[x for i,x in enumerate(data) if i%10==9];train=[x for i,x in enumerate(data) if i%10!=9];assert hold
 x=torch.cat([v['observations'] for v in train]);y=torch.cat([v['actions'] for v in train]);vx=torch.cat([v['observations'] for v in hold]);vy=torch.cat([v['actions'] for v in hold]);assert len(x)+len(vx)>=1000000
 previous=torch.load(a.resume,map_location='cpu',weights_only=False) if a.resume else None
 if previous:assert previous['teacher_sha256']==a.teacher_sha
 mean=previous['mean'] if previous else x.mean(0);std=previous['std'] if previous else x.std(0).clamp_min(.01);x=((x-mean)/std).to(a.device);y=y.to(a.device);vx=((vx-mean)/std).to(a.device);vy=vy.to(a.device)
 net=nn.Sequential(nn.Linear(25,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,6)).to(a.device);assert sum(t.numel() for t in net.parameters())==20614
 if previous:net.load_state_dict(previous['student_state_dict'],strict=True)
 opt=torch.optim.Adam(net.parameters(),lr=a.lr);best=float('inf');history=[]
 for step in range(a.steps):
  ids=torch.randint(len(x),(4096,),device=a.device);z=x[ids];opt.zero_grad();loss=(net(z+torch.randn_like(z)*.002)-y[ids]).square().mean();loss.backward();opt.step()
  if (step+1)%250==0:
   with torch.no_grad():
    errors=[(net(vx[j:j+4096])-vy[j:j+4096]).square().sum().item() for j in range(0,len(vx),4096)];score=sum(errors)/vy.numel()
   history.append(dict(step=step+1,validation_mse=score));print(history[-1],flush=True)
   if score<best:
    best=score;torch.save(dict(student_state_dict={k:v.cpu().clone() for k,v in net.state_dict().items()},mean=mean,std=std,teacher_sha256=a.teacher_sha,architecture=[25,128,128,6],promoted=False),a.output/'student.pt')
 report=dict(state='trained_not_behavior_validated',promoted=False,teacher_sha256=a.teacher_sha,samples=len(x)+len(vx),parameters=20614,weight_bytes=82456,history=history,dataset_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},export_blocked='Requires same-contract closed-loop retention and mandatory survival gates')
 (a.output/'training_report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
