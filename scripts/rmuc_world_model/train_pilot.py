"""CPU identification gate before any policy optimization. Never emits actor weights."""
from pathlib import Path
import argparse,json,hashlib,time,torch
from replay import load_report,sample,STATE_COLUMNS
from dynamics import Dynamics
from gpu_replay import WindowReplay
from queue_schema import align_delay_queues
from response_loss import randomized_residual_loss
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--reports',type=Path,nargs='+',default=[]);p.add_argument('--captures',type=Path,nargs='+',default=[]);p.add_argument('--rich',action='store_true');p.add_argument('--steps',type=int,default=1200);p.add_argument('--train-horizon',type=int,default=8);p.add_argument('--actor-state-loss-weight',type=float,default=0.);p.add_argument('--resume',type=Path);p.add_argument('--condition-commands',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=128);p.add_argument('--pad-unused-delay-queues',action='store_true');p.add_argument('--width',type=int,default=128);p.add_argument('--depth',type=int,default=2);p.add_argument('--lr',type=float,default=.001);p.add_argument('--budget-seconds',type=float,default=240);p.add_argument('--response-loss-weight',type=float,default=0.);p.add_argument('--response-batch-size',type=int,default=4096);p.add_argument('--paired-baseline',type=Path);p.add_argument('--paired-probe',type=Path);p.add_argument('--paired-response-weight',type=float,default=0.);p.add_argument('--paired-absolute-weight',type=float,default=0.);p.add_argument('--capture-prefix-config',type=Path);p.add_argument('--paired-policy-start-seconds',type=float);p.add_argument('--paired-additional-probes',type=Path,nargs='*',default=[]);args=p.parse_args()
assert bool(args.reports)!=bool(args.captures), 'Exactly one input mode required'
assert 1<=args.train_horizon<=32
assert args.batch_size>0
assert args.actor_state_loss_weight>=0 and (args.actor_state_loss_weight==0 or args.rich)
args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(27);g=torch.Generator().manual_seed(27);train=[];val=[];provenance=[];response_sequences=[]
prefixes={str(Path(k).resolve()):v for k,v in json.loads(args.capture_prefix_config.read_text()).items()} if args.capture_prefix_config else {}
for path in (args.reports or args.captures):
 if args.captures:
  from rich_replay import load_capture
  seq,meta=load_capture(path,4,32,args.rich,args.condition_commands)
 else:seq,meta=load_report(path,4,32)
 if str(path.resolve()) in prefixes:
  from replay_window import prefix_sequences
  seconds=prefixes[str(path.resolve())];seq=prefix_sequences(seq,seconds)
  meta['explicit_prefix_seconds']=seconds
 if args.pad_unused_delay_queues and provenance:
  seq,meta=align_delay_queues(seq,meta,provenance[0]['extra_state_names'])
 if provenance and meta.get('extra_state_names')!=provenance[0].get('extra_state_names'):raise ValueError('Capture schema mismatch')
 if args.response_loss_weight and args.captures:
  evaluation=json.loads((path/'evaluation.json').read_text())
  if evaluation.get('world_model_action_noise_std',0.):
   assert json.loads((path/'excitation_verified.json').read_text())['passed']
   import sys
   sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
   from verify_rmuc_export import restore_actor
   spec=json.loads((path/'command.json').read_text());teacher=restore_actor(Path(spec['source_checkpoint'])).eval().requires_grad_(False)
   with torch.no_grad():
    for q in seq:
     if q['env']%4==3:continue
     baseline=torch.cat([teacher({'policy':o},stochastic_output=False) for o in q['observations'].split(2048)])
     noise=q['action']-baseline
     assert noise.abs().max()<=3*evaluation['world_model_action_noise_std']+2e-5
     response_sequences.append(dict(q,context=torch.cat([q['context'],noise],-1)))
   del teacher
 provenance.append(meta)
 for q in seq:(val if q['env']%4==3 else train).append(q)
assert train and val
allstate=torch.cat([q['state'] for q in train]);diff=torch.cat([q['state'][1:]-q['state'][:-1] for q in train]);ctx=torch.cat([q['context'] if q['context'].ndim==2 else q['context'][None] for q in train]);mean=allstate.mean(0);scale=allstate.std(0).clamp_min(.02);ds=diff.std(0).clamp_min(.002);cm=ctx.mean(0);cs=ctx.std(0).clamp_min(.001)
m=Dynamics(mean,scale,ds,cm,cs,width=args.width,depth=args.depth)
if args.resume:
 previous=torch.load(args.resume,map_location='cpu',weights_only=False)
 assert previous['state_columns']==STATE_COLUMNS+provenance[0].get('extra_state_names',[])
 assert previous['history']==4 and previous['width']==args.width and previous.get('depth',2)==args.depth
 m.load_state_dict(previous['model'],strict=True)
 # Preserve the learned model's coordinate system; do not silently renormalize weights.
 mean,scale,ds,cm,cs=m.mean,m.scale,m.delta_scale,m.context_mean,m.context_scale
device=torch.device(args.device);m.to(device)
mean,scale,ds,cm,cs=m.mean,m.scale,m.delta_scale,m.context_mean,m.context_scale
for trajectory in train+val:
 for key in ['state','action','context']:trajectory[key]=trajectory[key].to(device)
training_replay=WindowReplay(train,4,args.train_horizon,device)
validation_replay=WindowReplay(val,4,32,device)
g=torch.Generator(device=device).manual_seed(27)
opt=torch.optim.Adam(m.parameters(),lr=args.lr)
if args.resume and 'optimizer' in previous:
 opt.load_state_dict(previous['optimizer'])
 for group in opt.param_groups:group['lr']=args.lr
fixed=validation_replay.sample(256,torch.Generator(device=device).manual_seed(943));history=[];start=time.monotonic()
(args.output/'command.json').write_text(json.dumps(dict(argv=__import__('sys').argv,configuration=vars(args)),default=str,indent=2))
def evaluate():
 with torch.no_grad():
  s,ah,a,y,c=fixed;pred=m.rollout(s,ah,a,c);persist=s[:,-1:, :].expand_as(y);result={}
  for k in [1,8,16,32]:
   err=pred[:,k-1]-y[:,k-1];base=persist[:,k-1]-y[:,k-1]
   result[str(k)]=dict(model_normalized_rmse=float((err/scale).square().mean().sqrt()),persistence_normalized_rmse=float((base/scale).square().mean().sqrt()),vx_mae=float(err[:,1].abs().mean()),pitch_mae_deg=float(torch.rad2deg(err[:,4]).abs().mean()),state_mae=err.abs().mean(0).tolist())
 return result
# Rich-state losses otherwise average actor-critical sensors with hundreds of auxiliary fields.
actor_names={f'root_state_w:{i}' for i in range(10)}
for j in [2,3,35,0,1,33]:
 actor_names.update([f'joint_pos:{j}',f'joint_pos:{j+38}',f'joint_vel:{j}'])
actor_indices=[16+i for i,n in enumerate(provenance[0].get('extra_state_names',[])) if n in actor_names]
if args.actor_state_loss_weight:assert len(actor_indices)==28
response_replay=None
if args.response_loss_weight:
 assert args.rich and args.condition_commands and response_sequences
 response_replay=WindowReplay(response_sequences,4,1,device)
 from train_imagination import physical
 response_names=provenance[0]['extra_state_names'];response_indices=lambda k:[16+i for i,n in enumerate(response_names) if n.startswith(k+':')]
 def response_metric(x):
  root,q,dq,vx,yaw,pitch,roll=physical(x,response_indices)
  return torch.stack([vx,yaw,pitch,q[:,2]-q[:,0],q[:,3]+q[:,1]],-1)
 response_scale=torch.tensor([.01,.05,.005,.01,.01],device=device)
paired=None
if args.paired_response_weight:
 from paired_response import PairedResponse
 assert args.paired_response_weight>0 and args.paired_baseline and args.paired_probe
 paired=PairedResponse(args.paired_baseline,args.paired_probe,device,policy_start_seconds=args.paired_policy_start_seconds)
 assert paired.columns==STATE_COLUMNS+provenance[0].get('extra_state_names',[])
 for probe in args.paired_additional_probes:
  extra=PairedResponse(args.paired_baseline,probe,device,policy_start_seconds=args.paired_policy_start_seconds)
  assert extra.columns==paired.columns and extra.ids==paired.ids
  for group in ['base','probe']:
   left,right=getattr(paired,group),getattr(extra,group)
   for key in left:left[key]=torch.cat([left[key],right[key]],0)
initial=evaluate()
for step in range(args.steps):
 s,ah,a,y,c=training_replay.sample(args.batch_size,g);pred=m.rollout(s,ah,a,c);error=((pred-y)/scale).square();loss=error[:,:,:16].mean()+(0.1*error[:,:,16:].mean() if error.shape[-1]>16 else 0)
 if args.actor_state_loss_weight:loss=loss+args.actor_state_loss_weight*error[:,:,actor_indices].mean()
 response_loss=loss.new_zeros(())
 if response_replay:
  rs,rh,ra,ry,rc=response_replay.sample(args.response_batch_size,g)
  rp=m(rs,rh,ra[:,0],rc[:,0,:-6])
  response_loss=randomized_residual_loss(response_metric(rp),response_metric(ry[:,0]),rc[:,0,-6:],response_scale)
  loss=loss+args.response_loss_weight*response_loss
 paired_loss=loss.new_zeros(())
 if paired:
  paired_loss=paired.loss(m,args.paired_absolute_weight);loss=loss+args.paired_response_weight*paired_loss
 if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
 opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 if step%10==0 or step==args.steps-1:
  progress=dict(paired_absolute_loss=float(paired.last_absolute) if paired else 0.,paired_delta_loss=float(paired.last_response) if paired else 0.,paired_response_loss=float(paired_loss.detach()),step=step+1,loss=float(loss.detach()),response_loss=float(response_loss.detach()),seconds=time.monotonic()-start,device=str(device),width=args.width,depth=args.depth)
  history.append(progress);temp=args.output/'progress.tmp';temp.write_text(json.dumps(progress));temp.replace(args.output/'progress.json');print(json.dumps(progress),flush=True)
 if time.monotonic()-start>args.budget_seconds:break
result=evaluate();torch.save(dict(condition_commands=args.condition_commands,model=m.state_dict(),history=4,width=args.width,depth=args.depth,optimizer=opt.state_dict(),state_columns=STATE_COLUMNS+provenance[0].get('extra_state_names',[]),context='root mass/inertia plus capture controller context when available',deployment_actor_modified=False),args.output/'world_model.pt')
report=dict(paired_absolute_weight=args.paired_absolute_weight,paired_response_weight=args.paired_response_weight,paired_training_envs=paired.ids if paired else [],response_loss_weight=args.response_loss_weight,response_batch_size=args.response_batch_size,response_training_trajectories=len(response_sequences),device=str(device),batch_size=args.batch_size,condition_commands=args.condition_commands,resume_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,normalization='source model preserved' if args.resume else 'training split only',actor_state_loss_weight=args.actor_state_loss_weight,actor_state_indices=actor_indices,state='identification_complete',width=args.width,depth=args.depth,learning_rate=args.lr,parameter_count=sum(p.numel() for p in m.parameters()),steps=step+1,train_horizon=args.train_horizon,seconds=time.monotonic()-start,state_dim=len(mean),context_dim=len(cm),auxiliary_loss_weight=.1 if len(mean)>16 else 0,training_trajectories=len(train),validation_trajectories=len(val),split='Fixed environment IDs mod4=3 held out across all source rollouts; same simulator seed distribution. Not independent seed or trajectory-family test.',provenance=provenance,initial=initial,validation=result,loss_history=history,qualified_for_policy_update=False,reason='No verified counterfactual gradient or on-policy imagination. Solver state/context not exhaustive. Prediction error alone cannot qualify BPTT policy training.',deployment=dict(obs=25,action=6,frame_history=1,target='STM32H723 + CubeAI',world_model_on_mcu=False),model_sha256=hashlib.sha256((args.output/'world_model.pt').read_bytes()).hexdigest())
(args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['provenance','loss_history','initial']},indent=2))
