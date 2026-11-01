"""BPTT scalar probe or isolated actor candidate; production checkpoint never edited.
Independent physics evaluation is mandatory before any actor fitting.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from dynamics import Dynamics
from rich_replay import load_capture
from observation import observe,native_root_to_wxyz,rotate,canonical
from sensor import push_action,push_sensor
from horizon import require_complete_intervention
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_rmuc_export import restore_actor
from rmuc_position_authority import execute_authority_action
from rmuc_eval_commands import bounded_residual_probe,diagnostic_bias_fraction

p=argparse.ArgumentParser()
for k in ['capture','model','teacher','output']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--critic',type=Path);p.add_argument('--evaluate-cases',type=Path);p.add_argument('--train-stop-head',action='store_true');p.add_argument('--actor-lr',type=float,default=1e-5);p.add_argument('--device',default='cpu');p.add_argument('--phase-retention',action='store_true');p.add_argument('--imagination-horizon',type=int,choices=[16,32,64],default=None);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
ck=torch.load(a.model,map_location='cpu',weights_only=False);sd=ck['model'];wm=Dynamics(sd['mean'],sd['scale'],sd['delta_scale'],sd['context_mean'],sd['context_scale'],ck['history'],ck['width'],ck.get('depth',2));wm.load_state_dict(sd);wm.eval().requires_grad_(False)
critic=None
if a.critic:
 from value import StopValue,stopping_stage_cost
 vc=torch.load(a.critic,map_location='cpu',weights_only=False);v=vc['model'];critic=StopValue(v['mean'],v['scale'],v['target_scale']);critic.load_state_dict(v);critic.eval().requires_grad_(False)
 assert json.loads((a.critic.parent/'report.json').read_text())['qualified_for_diagnostic']
actor=restore_actor(a.teacher).eval().requires_grad_(False);seq,meta=load_capture(a.capture,4,100,True,ck.get('condition_commands',False))
device=torch.device(a.device);wm.to(device);actor.to(device)
if critic is not None:critic.to(device)
for item in seq:
 for key in ['state','action','context']:item[key]=item[key].to(device)
assert hashlib.sha256(a.teacher.read_bytes()).hexdigest()==meta['checkpoint_sha256']
assert ck['state_columns']==meta['state_columns']+meta.get('extra_state_names',[])
if critic is not None:
 assert vc.get('condition_commands',False)==ck.get('condition_commands',False)
 assert len(vc['model']['mean'])==len(wm.mean)+len(wm.context_mean)
manifest=json.loads((a.capture/'replay/manifest.json').read_text());shards=[torch.load(x['path'],weights_only=True) for x in manifest['shards']]
keys=['root_state_w','joint_pos','joint_vel','_default_control_pos','policy_actions','obs_history','action_history','sensor_delay_cycles','action_delay_cycles','episode_step']
pre={k:torch.cat([s['pre'][k] for s in shards]) for k in keys};obs=torch.cat([s['observation'] for s in shards]);del shards
pre={k:v.to(device) for k,v in pre.items()};obs=obs.to(device)
assert pre['sensor_delay_cycles'].unique().tolist()==[1] and pre['action_delay_cycles'].unique().tolist()==[0]
commands=torch.zeros(*obs.shape[:2],3,device=device);commands[...,0]=obs[...,7]/.5
assert (obs[...,2]/.25+obs[...,8]).abs().max()<1e-5, 'Only unclipped zero-yaw diagnostic supported'
ids=[2,3,35,0,1,33];scale=torch.tensor([2.1,.5,1.],device=device)
clean=observe(native_root_to_wxyz(pre['root_state_w']),pre['joint_pos'][...,ids],pre['joint_vel'][...,ids],pre['_default_control_pos'],pre['policy_actions'],commands,scale,2.,2.)
noise=pre['obs_history'][:,:,0]-clean
names=meta['extra_state_names'];ix=lambda k:[16+i for i,n in enumerate(names) if n.startswith(k+':')]
head=actor.rmuc_physics_adaptation;hook={}
def capture(module,inputs,output):
 hook['residual']=head.stop_residual(inputs[0],output);hook['gate']=head.stop_gate(inputs[0])
handle=head.stop_network.register_forward_hook(capture)
# Model-training envs only propose; held-out envs evaluate the proposal separately.
train_ids=torch.tensor([q['env'] for q in seq if q['env']%4!=3],device=device);held=torch.tensor([q['env'] for q in seq if q['env']%4==3],device=device);by={q['env']:q for q in seq}
t=2125;H=a.imagination_horizon or (32 if a.train_stop_head else 64)
if not a.train_stop_head:require_complete_intervention(H,.008,.4)

def rollout(amplitude,envs,direction=None):
 states=torch.stack([by[int(e)]['state'][t-4:t] for e in envs]);ah=torch.stack([by[int(e)]['action'][t-4:t] for e in envs]);ctx=torch.stack([by[int(e)]['context'][t:t+H+1] if by[int(e)]['context'].ndim==2 else by[int(e)]['context'] for e in envs]);hist=pre['obs_history'][t,envs];acthist=pre['action_history'][t,envs];o=obs[t,envs];vs=[];pitches=[];asym=[];yaws=[];hips=[]
 for j in range(H):
  raw=actor({'policy':o},stochastic_output=False)
  frac=diagnostic_bias_fraction(pre['episode_step'][t+j,envs],.008,17.,.04,17.4)
  delta=amplitude*frac[:,None]*(raw.new_tensor([1,0,0,1,0,0]) if direction is None else direction)
  raw=raw+bounded_residual_probe(hook['residual'],head.stop_budget,delta,hook['gate'])
  executed,acthist=push_action(acthist,execute_authority_action(o,raw),pre['action_delay_cycles'][t+j,envs])
  nxt=wm(states,ah,raw,ctx[:,j] if ctx.ndim==3 else ctx)
  # Functional clone/index writes avoid modifying tensors needed by autograd.
  nxt=nxt.clone();nxt[:,ix('policy_actions')]=executed;nxt[:,ix('action_history')]=acthist.flatten(1);nxt[:,ix('obs_history')]=hist.flatten(1)
  root=torch.cat([nxt.new_zeros(len(envs),2),nxt[:,0:1],nxt[:,ix('root_state_w')]],-1);root=native_root_to_wxyz(root)
  quat=torch.nn.functional.normalize(root[:,3:7],dim=-1);root=torch.cat([root[:,:3],quat,root[:,7:]],-1)
  qp=nxt[:,ix('joint_pos')];q=torch.atan2(qp[:,:38],qp[:,38:]);dq=nxt[:,ix('joint_vel')]
  fresh=observe(root,q[:,ids],dq[:,ids],pre['_default_control_pos'][t,envs],executed,commands[t+j+1,envs],scale,2.,2.)+noise[t+j+1,envs]
  fresh=torch.cat([fresh[:,:6],fresh[:,6:9].clamp(-2,2),fresh[:,9:19],executed],-1)
  o,hist=push_sensor(hist,fresh,torch.ones(len(envs),dtype=torch.bool,device=device),pre['sensor_delay_cycles'][t+j+1,envs],commands[t+j+1,envs,2])
  f=root.new_zeros(len(envs),3);f[:,2]=-1;fw=rotate(quat,f)[:,:2];fw=fw/fw.norm(dim=-1,keepdim=True).clamp_min(1e-6)
  vs.append((root[:,7:9]*fw).sum(-1));g=canonical(quat,f);pitches.append(torch.atan2(g[:,0],-g[:,2]));asym.append(q[:,2]-q[:,0]);hips.append(q[:,3]+q[:,1]);yaws.append(canonical(quat,root[:,10:13])[:,2])
  states=torch.cat([states[:,1:],nxt[:,None]],1);ah=torch.cat([ah[:,1:],raw[:,None]],1)
 vx=torch.stack(vs);pitch=torch.stack(pitches);leg=torch.stack(asym)
 # Diagnostic objective, not a replacement for the production reward/gates.
 if critic is None:
  cost=vx[-32:].square().mean()+.2*pitch.square().mean()+.05*leg.square().mean()+.1*amplitude.square()
 else:
  discount=vx.new_tensor(vc['gamma'])**torch.arange(H,device=device)
  stage=stopping_stage_cost(vx,pitch,leg,vc.get('overshoot_weight',0.),profile=vc.get('cost_profile','legacy'),yaw=torch.stack(yaws),hip=torch.stack(hips))
  cost=(stage*discount[:,None]).sum(0).mean()+vc['gamma']**H*critic(torch.cat([nxt,ctx[:,H] if ctx.ndim==3 else ctx],-1)).mean()+.1*amplitude.square()
 return cost,dict(stage_cost=float((stage*discount[:,None]).sum(0).mean().detach()) if critic is not None else None,bootstrap_cost=float((vc['gamma']**H*critic(torch.cat([nxt,ctx[:,H] if ctx.ndim==3 else ctx],-1)).mean()).detach()) if critic is not None else None,vx_tail_rms=float(vx[-32:].detach().square().mean().sqrt()),pitch_rms_deg=float(torch.rad2deg(pitch.detach()).square().mean().sqrt()))

if a.train_stop_head:
 assert critic is not None and not a.evaluate_cases
 # Only active-leg rows of the top-level final stop layer may change.
 layer=head.stop_network[4];layer.requires_grad_(True)
 original_w=layer.weight.detach().clone();original_b=layer.bias.detach().clone()
 original_state={k:v.detach().clone() for k,v in actor.state_dict().items()}
 generator=torch.Generator(device=device).manual_seed(73)
 # Old trajectory rehearsal includes startup/deceleration/standstill, including rare states.
 replay_obs=obs[:].reshape(-1,25).clone()
 from retention import phase_labels,balanced_indices,phase_maxima,phase_accepts
 labels=phase_labels(pre['episode_step']).flatten()
 selection_ids=torch.cat([torch.nonzero(labels==k).flatten()[::8] for k in range(4)])
 phase_checks=None
 with torch.no_grad():
  baseline_replay=torch.cat([actor({'policy':x},stochastic_output=False) for x in replay_obs.split(1024)])
  baseline_cost,_=rollout(torch.tensor(0.,device=device),train_ids)
 optimizer=torch.optim.Adam(layer.parameters(),lr=a.actor_lr);history=[];best_cost=float(baseline_cost);best=(original_w.clone(),original_b.clone());start=time.monotonic()
 for step in range(24):
  idx=balanced_indices(labels,256,generator) if a.phase_retention else torch.randint(len(replay_obs),(1024,),generator=generator,device=device)
  optimizer.zero_grad();cost,metrics=rollout(torch.tensor(0.,device=device),train_ids)
  deviation=actor({'policy':replay_obs[idx]},stochastic_output=False)-baseline_replay[idx]
  retention=deviation.square().mean()
  if a.phase_retention:
   weights=torch.where((labels[idx]==1)|(labels[idx]==2),100000.,50.)
   retention_cost=(deviation.square().mean(-1)*weights).mean()
  else:retention_cost=50.*retention
  loss=cost+retention_cost
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite actor objective')
  loss.backward()
  for parameter in layer.parameters():
   parameter.grad[[1,2,4,5]]=0
  torch.nn.utils.clip_grad_norm_(layer.parameters(),.1);optimizer.step()
  with torch.no_grad():
   measured_cost,_=rollout(torch.tensor(0.,device=device),train_ids)
   deviation=actor({'policy':replay_obs[idx]},stochastic_output=False)-baseline_replay[idx]
   maxdiff=float(deviation.abs().max())
   phase_ok=True
   if a.phase_retention:
    check=actor({'policy':replay_obs[selection_ids]},stochastic_output=False)-baseline_replay[selection_ids]
    phase_checks=phase_maxima(check,labels[selection_ids]);phase_ok=phase_accepts(phase_checks)
   if phase_ok and maxdiff<=.01 and float(measured_cost)<best_cost:
    best_cost=float(measured_cost);best=(layer.weight.clone(),layer.bias.clone())
  history.append(dict(step=step,cost=float(cost.detach()),retention_mse=float(retention.detach()),after_cost=float(measured_cost),replay_sample_maxdiff=maxdiff,phase_maxima=phase_checks))
  progress=dict(state='training',step=step+1,best_model_cost=best_cost,last=history[-1],elapsed_seconds=time.monotonic()-start,promoted=False)
  temp=a.output/'progress.tmp';temp.write_text(json.dumps(progress,indent=2));temp.replace(a.output/'progress.json')
  print(json.dumps(progress),flush=True)
  if time.monotonic()-start>180:break
 with torch.no_grad():
  layer.weight.copy_(best[0]);layer.bias.copy_(best[1])
  after_replay=torch.cat([actor({'policy':x},stochastic_output=False) for x in replay_obs.split(1024)])
  maxdiff=float((after_replay-baseline_replay).abs().max());retention_mse=float((after_replay-baseline_replay).square().mean())
  final_phase=phase_maxima(after_replay-baseline_replay,labels)
  held_cost,_=rollout(torch.tensor(0.,device=device),held)
 changed=[k for k,v in actor.state_dict().items() if not torch.equal(v,original_state[k])]
 allowed=['rmuc_physics_adaptation.stop_network.4.weight','rmuc_physics_adaptation.stop_network.4.bias']
 assert set(changed)<=set(allowed)
 assert torch.equal(layer.weight[[1,2,4,5]],original_w[[1,2,4,5]]) and torch.equal(layer.bias[[1,2,4,5]],original_b[[1,2,4,5]])
 accepted=bool(changed and maxdiff<=.01 and best_cost<float(baseline_cost) and (not a.phase_retention or phase_accepts(final_phase)))
 if accepted:
  payload=torch.load(a.teacher,map_location='cpu',weights_only=False)
  payload['actor_state_dict']={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
  payload.pop('optimizer_state_dict',None) # Old PPO optimizer is not a valid resume state.
  payload['rmuc_world_model_update']=dict(source_sha256=meta['checkpoint_sha256'],backend='bounded_stop_last_layer_bptt_pilot',ppo_optimizer_resume_valid=False,physical_validation_passed=False)
  torch.save(payload,a.output/'candidate.pt')
 result=dict(phase_retention=a.phase_retention,phase_maxima=final_phase,horizon=H,cost_profile=vc.get('cost_profile','legacy'),gamma=vc['gamma'],device=str(device),actor_lr=a.actor_lr,overshoot_weight=vc.get('overshoot_weight',0.),candidate_emitted=accepted,promoted=False,baseline_model_cost=float(baseline_cost),candidate_model_cost=best_cost,held_out_model_cost=float(held_cost),replay_max_action_difference=maxdiff,replay_action_mse=retention_mse,changed_keys=changed,trainable_output_rows=[0,3],history=history,seconds=time.monotonic()-start,source_teacher_sha256=meta['checkpoint_sha256'],scope='First isolated BPTT actor pilot, frozen teacher data/model/value. No time or pulse input to actor. Raw replay retention is not closed-loop retention; real Isaac validation mandatory.')
 (a.output/'actor_update_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));handle.remove();sys.exit(0)
if a.evaluate_cases:
 cases=json.loads(a.evaluate_cases.read_text());results=[]
 with torch.no_grad():
  for index,case in enumerate(cases):
   envs=torch.arange(index*8,(index+1)*8,device=device);vector=torch.tensor(case['action'],device=device,dtype=torch.float32)
   assert vector.shape==(6,) and torch.isfinite(vector).all() and vector.abs().max()<=.02
   amp=float(vector.abs().max());direction=vector/max(amp,1e-12)
   baseline,bm=rollout(torch.tensor(0.,device=device),envs);candidate,cm=rollout(torch.tensor(amp,device=device),envs,direction)
   results.append(dict(case=index,name=case['name'],amplitude=amp,baseline_cost=float(baseline),proposal_cost=float(candidate),predicted_delta=float(candidate-baseline),baseline=bm,proposal=cm))
 (a.output/'predictions.json').write_text(json.dumps(dict(horizon=H,pulse_duration_s=.4,pulse_fully_inside_horizon=H*.008>=.4,cases=results,scope='Same matched environment draws as pending physics. Frozen model/critic/teacher; no fitting.'),indent=2));handle.remove();sys.exit(0)
start=time.monotonic();theta=torch.tensor(0.,device=device,requires_grad=True);cost,_=rollout(theta,train_ids);cost.backward();grad=float(theta.grad)
with torch.no_grad():
 eps=.0001;plus,_=rollout(torch.tensor(eps,device=device),train_ids);minus,_=rollout(torch.tensor(-eps,device=device),train_ids);fd=float((plus-minus)/(2*eps))
if not torch.isfinite(torch.tensor(grad)) or abs(grad-fd)>max(1e-5,abs(fd)*.05):raise RuntimeError(f'Autograd finite-difference mismatch {grad} / {fd}')
opt=torch.optim.Adam([theta],lr=.002);history=[];best=(float(cost.detach()),0.)
for step in range(8):
 opt.zero_grad();loss,metrics=rollout(theta,train_ids);evaluated_amplitude=float(theta.detach())
 if float(loss.detach())<best[0]:best=(float(loss.detach()),evaluated_amplitude)
 if not torch.isfinite(loss):raise RuntimeError('Nonfinite probe objective')
 loss.backward();opt.step()
 with torch.no_grad():theta.clamp_(-.01,.01)
 history.append(dict(step=step,loss=float(loss.detach()),amplitude=evaluated_amplitude,next_amplitude=float(theta.detach()),**metrics))
 if time.monotonic()-start>180:break
with torch.no_grad():
 final_loss,_=rollout(theta,train_ids)
 if float(final_loss)<best[0]:best=(float(final_loss),float(theta))
 theta.fill_(best[1]);results={}
 for name,envs in [('proposal_training_envs',train_ids),('held_out_envs',held)]:
  base,bm=rollout(torch.tensor(0.,device=device),envs);candidate,cm=rollout(theta.detach(),envs);results[name]=dict(base_cost=float(base),proposal_cost=float(candidate),baseline=bm,proposal=cm)
out=dict(device=str(device),critic_sha256=hashlib.sha256(a.critic.read_bytes()).hexdigest() if a.critic else None,horizon=H,source_teacher_sha256=meta['checkpoint_sha256'],world_model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),autograd=grad,finite_difference=fd,amplitude=float(theta.detach()),history=history,results=results,elapsed_seconds=time.monotonic()-start,actor_updated=False,promoted=False,scope='Conditional model BPTT scalar pulse proposal, same development seed. Not a policy update or simulator validation. Optional critic is narrow frozen-teacher MC bootstrap, not full SuperMimic critic.',physics_validation_required=True)
(a.output/'proposal.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));handle.remove()
