"""Prepare one joint reachability screen for five straight speeds; no launch."""
from pathlib import Path
import copy,json,hashlib,sys,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/teacher_course_20261009/low_speed_domain_20261011';OUT=BASE/'straight_domain_response_20261014'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'scripts/rsl_rl')]
from verify_rmuc_export import restore_actor

def main():
 OUT.mkdir(exist_ok=False)
 source=BASE/'straight0275_balanced_20261014/final_checkpoint.pt';sha=hashlib.sha256(source.read_bytes()).hexdigest()
 speeds=[.5,.75,1.,1.25,1.5];grid=[[v,0.] for v in speeds];(OUT/'grid.json').write_text(json.dumps(grid)+'\n')
 cases=[]
 for leg in [0.,.15,.3]:
  for wheel in [0.,-.06,.06]:cases.append(dict(name='baseline' if leg==wheel==0 else f'leg{leg}_wheel{wheel}',action=[leg,leg,wheel,-leg,leg,wheel]))
 (OUT/'bias.json').write_text(json.dumps(cases,indent=2)+'\n')
 spec=json.loads((BASE/'straight0275_balanced_20261014/evaluation_command.json').read_text());cmd=spec['command']
 for key,val in [('--checkpoint',str(source)),('--output',str(OUT/'evaluation.json')),('--command-grid',str(OUT/'grid.json')),('--replicas','2'),('--steps','3200'),('--min-episodes','4')]:cmd[cmd.index(key)+1]=val
 cmd+=['--action-bias-sweep',str(OUT/'bias.json'),'--action-bias-start-seconds','2','--action-bias-ramp-seconds','1','--arena-ground-progress','.1','--base-com-dr-strength','.1','--capture-startup-steps','500','--capture-observation-stride','6']
 spec.update(source_checkpoint=str(source),source_sha256=sha,state='prepared',scope='Frozen5speed x9action directions with first-stage arena/COMDR; not policy acceptance')
 (OUT/'evaluation_command.json').write_text(json.dumps(spec,indent=2)+'\n')
 contract=json.loads((BASE/'straight0275_balanced_20261014/experiment_contract.json').read_text());contract.update(source_checkpoint=str(source),source_sha256=sha,unique_variable='Joint leg-wheel action direction scan across five speeds under explicit first-stage arena/COM; no optimizer',optimizer_updates=0,trainable_parameters=[],budget={'steps':3200,'envs':90,'min_episodes':4,'timeout_seconds':720},ground_progress=.1,ground_effective_static_range=[.495,.515],ground_effective_dynamic_range=[.488,.505],ground_restitution=.002,rootCOM_max_offset_flu_m=[.001,.001,.0005],retention='Independent low-speed head remains unchanged; artificial action perturbations are diagnostic only, no deployment/promotion')
 (OUT/'experiment_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
 # Prepare zero-output extension seed but do not infer feasible initialization.
 torch.set_num_threads(2);torch.manual_seed(61);a=restore_actor(source).eval().requires_grad_(False);b=restore_actor(source).eval().requires_grad_(False)
 b.rmuc_physics_adaptation.install_straight_extension([.4,.4,.35,.4,.4,.35],[.35,.5,1.5,1.7])
 payload=torch.load(source,map_location='cpu',weights_only=False);payload['actor_state_dict']=b.state_dict();candidate=OUT/'zero_extension_seed.pt';torch.save(payload,candidate);b=restore_actor(candidate).eval().requires_grad_(False)
 x=torch.randn(8192,25);x[:,7]=torch.linspace(-1.5,3.2,len(x))*.5;x[:,2]=0;x[:,8]=torch.linspace(-2.5,2.5,len(x));x[:1024,7]=.125;x[:1024,8]=0
 with torch.no_grad():
  parity=float((a({'policy':x})-b({'policy':x})).abs().max());assert parity==0
  b.rmuc_physics_adaptation.straight_extension_network[-1].bias.fill_(.2)
  gate=b.rmuc_physics_adaptation.straight_extension_gate(x);outside=float((a({'policy':x[gate==0]})-b({'policy':x[gate==0]})).abs().max());assert outside==0
 (OUT/'seed_preflight.json').write_text(json.dumps(dict(passed=True,source_sha256=sha,seed_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),zero_migration_maxdiff=parity,perturbed_head_protected_maxdiff=outside,samples=len(x),scope='CPU full actor reconstruction and isolation only; no behavior claim'),indent=2)+'\n')
 print(OUT);print(sha)
if __name__=='__main__':main()
