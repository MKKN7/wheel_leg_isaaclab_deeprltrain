"""Initialize a trainable neural branch from measured response directions, CPU only.

Initialization is not learned closed-loop competence. PPO and unbiased rollout
must follow; neither inference nor the environment receives a bias schedule.
"""
from pathlib import Path
import argparse,hashlib,json,sys,torch,numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/teacher_course_20261009/low_speed_domain_20261011'
OUT=BASE/'straight_domain_neural_seed_20261014'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'scripts/rsl_rl')]
from verify_rmuc_export import restore_actor

def main():
 global OUT
 parser=argparse.ArgumentParser();parser.add_argument('--startup-aware',action='store_true');args=parser.parse_args()
 if args.startup_aware:OUT=BASE/'straight_domain_neural_startup_seed_20261014'
 OUT.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(618)
 response=BASE/'straight_domain_response_20261014'
 proposals=json.loads((response/'initialization_proposals.json').read_text())
 source=response/'zero_extension_seed.pt'
 actor=restore_actor(source).eval().requires_grad_(False)
 head=actor.rmuc_physics_adaptation.straight_extension_network
 head.requires_grad_(True)
 if args.startup_aware:
  # Fit the seed using two physically meaningful observation slots, avoiding
  # spurious wheel-phase correlations. PPO subsequently unlocks all inputs.
  with torch.no_grad():
   head[0].weight.zero_();head[0].weight[:,6:8].normal_(0,.3);head[0].bias.zero_()
   torch.nn.init.xavier_uniform_(head[2].weight);head[2].bias.zero_()
 data=torch.load(response/'evaluation.startup.pt',map_location='cpu',weights_only=False)
 x=data['observations'].reshape(-1,25).clone();v=x[:,7]/.5
 if args.startup_aware:
  failed=torch.load(BASE/'straight_domain_neural_seed_20261014/evaluation.startup.pt',map_location='cpu',weights_only=False)
  extra=failed['observations'].reshape(-1,25);extra=extra[(extra[:,7]/.5-.5).abs()<1e-4]
  x=torch.cat((x,extra))
 # Avoid relying on one fixed initial state. Include all diagnostic trajectories
 # and continuously varied commands within this opened straight interval.
 jitter=x.clone();jitter[:,7]=(.5+torch.rand(len(x)))*.5
 if args.startup_aware:jitter[:,6]+=(jitter[:,7]-x[:,7])/.5*2.1
 x=torch.cat((x,jitter));v=x[:,7]/.5
 knots=np.asarray([r['vx'] for r in proposals]);leg=np.interp(v.numpy(),knots,[r['proposed_action_leg_wheel'][0] for r in proposals]);wheel=np.interp(v.numpy(),knots,[r['proposed_action_leg_wheel'][1] for r in proposals])
 target=torch.tensor(np.stack([leg,leg,wheel,-leg,leg,wheel],-1),dtype=x.dtype)
 if args.startup_aware:
  # Supervised initialization label only. No gate/controller is added to the
  # deployed actor; PPO may change every weight in this branch afterward.
  measured_vx=v-x[:,6]/2.1
  t=(measured_vx/.4).clamp(0,1);ready=t*t*(3-2*t)
  slow=((.75-v)/.25).clamp(0,1)
  target*=((1-slow)+slow*ready)[:,None]
 budget=actor.rmuc_physics_adaptation.straight_extension_budget
 order=torch.randperm(len(x));held=order[:len(x)//5];train=order[len(x)//5:]
 opt=torch.optim.Adam(head.parameters(),lr=.001)
 updates=2400 if args.startup_aware else 800
 for step in range(updates):
  ids=train[torch.randint(len(train),(2048,))]
  pred=budget*head(x[ids]).tanh();loss=(pred-target[ids]).square().mean()
  opt.zero_grad();loss.backward()
  if args.startup_aware:
   head[0].weight.grad[:,:6]=0;head[0].weight.grad[:,8:]=0
  opt.step()
 with torch.no_grad():
  error=(budget*head(x[held]).tanh()-target[held]).abs()
  mse=float(error.square().mean());maxerr=float(error.max());p99=float(torch.quantile(error.flatten(),.99))
 assert p99<(.025 if args.startup_aware else .015),(mse,maxerr,p99)
 payload=torch.load(source,map_location='cpu',weights_only=False);payload['actor_state_dict']=actor.state_dict()
 candidate=OUT/'candidate.pt';torch.save(payload,candidate)
 reloaded=restore_actor(candidate).eval().requires_grad_(False)
 reference=restore_actor(source).eval().requires_grad_(False)
 protected=x[::8].clone();protected[:,8]+=.1
 with torch.no_grad():
  gate=reloaded.rmuc_physics_adaptation.straight_extension_gate(protected)
  protected=protected[gate==0]
  delta=float((reference({'policy':protected})-reloaded({'policy':protected})).abs().max())
 assert delta==0
 spec=json.loads((response/'evaluation_command.json').read_text());cmd=spec['command']
 for key in ['--action-bias-sweep','--action-bias-start-seconds','--action-bias-ramp-seconds']:
  i=cmd.index(key);del cmd[i:i+2]
 grid=json.loads((BASE/'straight0275_balanced_20261014/grid64.json').read_text())
 if args.startup_aware:grid=[[v,0.] for v in [.225,.25,.275,.5,.625,.75,.875,1.,1.125,1.25,1.375,1.5]]
 (OUT/'grid.json').write_text(json.dumps(grid)+'\n')
 for key,val in [('--checkpoint',str(candidate)),('--output',str(OUT/'evaluation.json')),('--command-grid',str(OUT/'grid.json'))]:cmd[cmd.index(key)+1]=val
 sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
 spec.update(source_checkpoint=str(candidate),source_sha256=sha,state='prepared',diagnostic_only=False,
             unique_variable='Neural straight branch initialized from response data; no external action bias or ramp',scope='Unbiased full actor retention and starting-state screen, not formal acceptance')
 for name in ['command.json','evaluation_command.json']:(OUT/name).write_text(json.dumps(spec,indent=2)+'\n')
 contract=json.loads((response/'experiment_contract.json').read_text());contract.update(source_checkpoint=str(candidate),source_sha256=sha,experiment_type='neural_branch_initialization_screen',unique_variable=spec['unique_variable'],retention='All original64, including five straight anchors; intermediate speeds evaluated after PPO',budget=dict(steps=3200,envs=2*len(grid),min_episodes=4,timeout_seconds=720),evaluation_command_file=str(OUT/'evaluation_command.json'))
 (OUT/'experiment_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
 audit=dict(state='initialized_not_behavior_validated',source_checkpoint=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),candidate_sha256=sha,
            supervised_initialization_updates=updates,ppo_updates=0,samples=len(x),held_mse=mse,held_max_action_error=maxerr,held_p99_action_error=p99,
            protected_samples=len(protected),protected_full_actor_maxdiff=delta,proposal=proposals,
            startup_aware_initialization=args.startup_aware,
            limitations=['Response regression extrapolates wheel action from .06 to about .15; unbiased rollout required','Neural weights are initialized, not a deployed external controller','Input-state response and disturbance rejection must be learned and validated in PPO'])
 (OUT/'seed_preflight.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps({k:v for k,v in audit.items() if k!='proposal'},indent=2))
if __name__=='__main__':main()
