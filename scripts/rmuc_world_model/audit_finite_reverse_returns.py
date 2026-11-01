"""Independent finite-window reward audit; no value bootstrap or terminal assumption."""
import argparse,json,torch
from pathlib import Path
from rich_replay import load_capture
from train_imagination import stage_objective,physical
from tracking_objective import speed_cost
from transition_reward import step_metadata,overshoot_cost
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
seq,meta=load_capture(a.capture,4,33,True,True);names=meta['extra_state_names'];ix=lambda k:[16+i for i,n in enumerate(names) if n.startswith(k+':')];rows=[];parity=0.
with torch.no_grad():
 for q in seq:
  if q['env']%4!=3:continue
  ev=step_metadata(q['context'][:,-3])
  for seconds in [7.,8.,8.5,9.,9.5]:
   t=round(seconds/.008);h=32
   if t+h>=len(q['state']):continue
   x=q['state'][t:t+h];ctx=q['context'][t:t+h];root,j,dq,v,y,pitch,roll=physical(x,ix);target=ctx[:,-3];straight=ctx[:,-2].abs()<.05;hub=lambda z:torch.sqrt(1+z*z)-1
   reward,cont,metric=stage_objective(x,ctx,ix);overs=overshoot_cost(v,target,ev[t:t+h]);full=reward-.008*4*overs
   parts={'constant':torch.full_like(v,.016),'speed':-.008*speed_cost(v-target,target),'yaw':-.008*hub((y-ctx[:,-2])/.05),'leg':-.008*.25*(hub((j[:,2]-j[:,0])/torch.deg2rad(v.new_tensor(2.)))+hub((j[:,3]+j[:,1])/torch.deg2rad(v.new_tensor(2.))))*straight,'posture':-.008*.1*(hub(torch.relu(pitch.abs()-torch.deg2rad(v.new_tensor(5.)))/.05)+hub(torch.relu(roll.abs()-torch.deg2rad(v.new_tensor(5.)))/.05)*straight),'survival':-10*(1-cont),'overshoot':-.008*4*overs}
   parity=max(parity,float((sum(parts.values())-full).abs().max()));discount=.998**torch.arange(h);surv=torch.cat([torch.ones(1),cont.cumprod(0)[:-1]]);weight=discount*surv
   values={k:float((z*weight).sum()) for k,z in parts.items()};rows.append({'env':q['env'],'start':seconds,'target':float(target[0]),'vx_start':float(v[0]),'vx_end':float(v[-1]),'step_size':float(ev[t,1]),'finite_return':float((full*weight).sum()),'parts':values})
assert parity<1e-4
summary=[]
for t in sorted(set(x['start'] for x in rows)):
 rr=[x for x in rows if x['start']==t];summary.append({'start':t,'n':len(rr),'mean_return':sum(x['finite_return'] for x in rr)/len(rr),'mean_parts':{k:sum(x['parts'][k] for x in rr)/len(rr) for k in rr[0]['parts']}})
a.output.write_text(json.dumps({'summary':summary,'rows':rows,'reward_reconstruction_max_error':parity,'scope':'Real first-episode H32 discounted reward, soft continuation, no bootstrap. Not infinite-horizon value or actor certification.'},indent=2));print(json.dumps(summary,indent=2))
