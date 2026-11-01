"""Compare matched-state physical and model rewards; no training or GPU needed."""
import argparse,json,sys,hashlib
from pathlib import Path
import torch
from rich_replay import load_capture
from dynamics import Dynamics
from train_imagination import stage_objective,physical
from transition_reward import step_metadata,robust_overshoot_cost
from tracking_objective import speed_cost
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_rmuc_export import restore_actor
p=argparse.ArgumentParser()
for k in ['baseline','candidate','model','teacher','output']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--start-seconds',type=float,default=8.)
p.add_argument('--allow-different-initial-states',action='store_true')
a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
b,meta=load_capture(a.baseline,4,32,True,True);c,cm=load_capture(a.candidate,4,32,True,True)
assert meta['extra_state_names']==cm['extra_state_names']
b={q['env']:q for q in b};c={q['env']:q for q in c};ix=lambda k:[16+i for i,n in enumerate(meta['extra_state_names']) if n.startswith(k+':')]
ck=torch.load(a.model,map_location='cpu',weights_only=False);sd=ck['model'];m=Dynamics(sd['mean'],sd['scale'],sd['delta_scale'],sd['context_mean'],sd['context_scale'],4,ck['width'],ck['depth']);m.load_state_dict(sd);m.eval()
actor=restore_actor(a.teacher).eval();head=actor.rmuc_physics_adaptation
rows=[];t=round(a.start_seconds/.008)
def components(x,ctx,event):
 root,j,dq,v,y,pitch,roll=physical(x,ix)
 hub=lambda z:torch.sqrt(1+z*z)-1
 r,cont,_=stage_objective(x,ctx,ix)
 parts={'speed':-.008*speed_cost(v-ctx[:,-3],ctx[:,-3]),
        'yaw':-.008*hub((y-ctx[:,-2])/.05),
        'overshoot':-.008*4*robust_overshoot_cost(v,ctx[:,-3],event)}
 parts['other']=r-parts['speed']-parts['yaw']
 return {k:float((value*.998**torch.arange(len(x))).sum()) for k,value in parts.items()}
def reward(x,ctx,event):
 r,cont,z=stage_objective(x,ctx,ix);over=.008*4*robust_overshoot_cost(z['vx'],ctx[:,-3],event)
 return r-over,over
with torch.no_grad():
 for e in sorted(b):
  if e%4!=3:continue
  q=b[e];z=c[e]
  if not a.allow_different_initial_states:assert torch.equal(q['state'][:t],z['state'][:t])
  ev=step_metadata(q['context'][:,-3]);past=q['state'][t-4:t][None];ah=q['action'][t-4:t][None];ctx=q['context'][t:t+32]
  metrics={};parts={}
  for name,s in [('base',q),('candidate',z)]:
   own_past=s['state'][t-4:t][None] if a.allow_different_initial_states else past
   own_ah=s['action'][t-4:t][None] if a.allow_different_initial_states else ah
   assert torch.equal(ctx,s['context'][t:t+32])
   pred=m.rollout(own_past,own_ah,s['action'][t:t+32][None],ctx[None])[0]
   for tag,x in [('real',s['state'][t:t+32]),('model',pred)]:
    rw,ov=reward(x,ctx,ev[t:t+32]);metrics[name+'_'+tag+'_h32']=float((rw*.998**torch.arange(32)).sum())
    parts[name+'_'+tag]=components(x,ctx,ev[t:t+32])
   end=min(t+250,len(q['state']),len(z['state']))
   rw,ov=reward(s['state'][t:end],s['context'][t:end],ev[t:end]);metrics[name+'_real_2s']=float((rw*.998**torch.arange(end-t)).sum())
  obs=q['observations'][t:min(t+250,len(q['observations']))];gate=head.reverse_gate(obs);out=torch.tanh(head.reverse_network(obs))
  rows.append(dict(env=e,**metrics,reward_components=parts,gate_min=float(gate.min()),reverse_tanh_saturation_fraction=float((out.abs()>.95).float().mean()),delta_model_h32=metrics['candidate_model_h32']-metrics['base_model_h32'],delta_real_h32=metrics['candidate_real_h32']-metrics['base_real_h32'],delta_real_2s=metrics['candidate_real_2s']-metrics['base_real_2s']))
# Signed error gradient check at representative reverse speeds, excluding attitude changes.
v=torch.tensor([-.9,-.65,-.6,-.55,-.3],requires_grad=True);target=torch.full_like(v,-.6)
cost=speed_cost(v-target,target)+4*robust_overshoot_cost(v,target,torch.tensor([[-1.,.2]]).expand(len(v),2));grad=torch.autograd.grad(cost.sum(),v)[0]
summary={k:sum(x[k] for x in rows)/len(rows) for k in ['delta_model_h32','delta_real_h32','delta_real_2s','gate_min','reverse_tanh_saturation_fraction']}
summary['component_deltas']={tag:{part:sum(x['reward_components']['candidate_'+tag][part]-x['reward_components']['base_'+tag][part] for x in rows)/len(rows) for part in ['speed','yaw','overshoot','other']} for tag in ['real','model']}
summary['wrong_sign_environments']=sum(x['delta_model_h32']*x['delta_real_h32']<0 for x in rows)
reasons=[]
if summary['wrong_sign_environments']>len(rows)/4:
 reasons.append('More than25percent heldout reward rankings have opposite signs')
if summary['delta_model_h32']>0 and summary['delta_real_h32']<0:
 reasons.append('Model ranks harmful candidate as beneficial')
for part in ['speed','yaw','overshoot','other']:
 real=summary['component_deltas']['real'][part];model=summary['component_deltas']['model'][part]
 if model>1e-3 and real < -1e-3:reasons.append('False improvement in '+part)
a.output.write_text(json.dumps(dict(start_seconds=a.start_seconds,matched_initial_history=not a.allow_different_initial_states,summary=summary,rows=rows,ranking_gate_passed=not reasons,failed_conditions=reasons,model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),reward_gradient=dict(vx=v.detach().tolist(),cost=cost.detach().tolist(),dcost_dvx=grad.tolist()),scope=('Independent initial histories: conditional prediction comparison, not matched-state causal experiment.' if a.allow_different_initial_states else 'Matched initial-state recorded-action response, not autonomous policy-gradient certification.'),promoted=False),indent=2));print(json.dumps(summary,indent=2))
