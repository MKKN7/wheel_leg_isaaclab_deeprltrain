"""Prepare isolated shared straight PPO, without launching or relaxing gates."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/teacher_course_20261009/low_speed_domain_20261011'
OLD=BASE/'straight0275_balanced_20261014'
SEED=BASE/'straight_domain_neural_startup_seed_20261014'
OUT=BASE/'straight_domain_ppo_20261014'

def main():
 OUT.mkdir(exist_ok=False)
 source=SEED/'candidate.pt';sha=hashlib.sha256(source.read_bytes()).hexdigest()
 spec=json.loads((OLD/'command.json').read_text());cmd=spec['command']
 def flag(k,v):cmd[cmd.index(k)+1]=str(v)
 def cfg(k,v):
  token='env.'+k+'='+json.dumps(v,separators=(',',':'))
  index=next((i for i,t in enumerate(cmd) if t.startswith('env.'+k+'=')),None)
  if index is None:cmd.append(token)
  else:cmd[index]=token
 def get(k):return json.loads(next(t.split('=',1)[1] for t in cmd if t.startswith('env.'+k+'=')))
 for k,v in [('--checkpoint',source),('--run_name',OUT.name),('--max_iterations',80),('--command_level',24)]:flag(k,v)
 # Move the variable-arity option ahead of positional Hydra tokens.
 i=cmd.index('--physics_train_modules');del cmd[i:i+2]
 modules=['rmuc_physics_adaptation.alignment_network','rmuc_physics_adaptation.straight_extension_network']
 cmd[2:2]=['--physics_train_modules',*modules]
 for key,value in [('command_stage_linear_x_mps',.5),('command_stage_yaw_rate_rad_s',0.),('command_stage_leg_scale_rad',.2),('command_stage_names','straight05_shared_alignment')]:
  values=get(key);values[24]=value;cfg(key,values)
 weights=[4,1,1,1,1,8,8,1,1,1,1,1,1,1,1,1,8,1,8,1,1,1,4,4]
 for key in ['command_rehearsal_anchor_stage_ids','command_rehearsal_priority_stage_ids']:cfg(key,list(range(24)))
 cfg('command_rehearsal_gate_stage_ids',list(range(25)));cfg('command_rehearsal_priority_stage_weights',weights)
 cfg('command_continuous_sampling_fraction',.1);cfg('command_continuous_straight_speed_range',[.5,1.5])
 cfg('command_straight_alignment_tolerance_deg',1.)
 for key,value in [('arena_ground_severity',-1.),('arena_ground_migration_progress',.1),('arena_ground_randomization_enabled',True),('arena_ground_severity_range',[0.,1.]),('friction_randomization_enabled',False),('domain_randomization_max_friction_fraction',0.),('base_com_randomization_enabled',True),('base_com_max_offset_flu_m',[.01,.01,.005]),('base_com_randomization_strength',.1)]:cfg(key,value)
 spec.update(source_checkpoint=str(source),source_sha256=sha,run_name=OUT.name,state='prepared_pending_seed_screen',
             unique_variable='Train low and shared straight neural heads jointly with 1deg training deadband, five anchors and continuous .5–1.5 cohort; frozen remainder',
             training_plot_conditions=[[v,0.] for v in [.225,.25,.275,.5,.75,1.,1.25,1.5]])
 spec['env']['RMUC_TRAINING_TAIL_START_UPDATE']='71'
 spec['prerequisites']=['Unbiased seed screen and source hashes reviewed','CPU gate/rehearsal/runtime preflight passed']
 (OUT/'command.json').write_text(json.dumps(spec,indent=2)+'\n')
 contract=json.loads((SEED/'experiment_contract.json').read_text());contract.update(experiment_type='shared_straight_neural_ppo',source_checkpoint=str(source),source_sha256=sha,unique_variable=spec['unique_variable'],trainable_parameters=modules+['critic'],optimizer_updates=80,
   budget=dict(updates=80,envs=512,steps_per_env=128,timeout_seconds=2400),
   stop_conditions=['80updates or2400s','Runtime contract/nonfinite/freeze/coverage failure','No automatic promotion'],
   reward_change='Measured alignment tolerance2→1deg for margin; weight8, speed margin.02, old action-sync0 unchanged',
   retention='Frozen old actor and all turn networks; 75% staged rehearsal, original64 quick plus conditional8point32; midpoints separately recorded',
   sampling=dict(fixed_target05=.15,continuous_straight_range=[.5,1.5],continuous_fraction=.1,rehearsal_fraction=.75,priority_ids=list(range(24)),priority_weights=weights),
   policy_requirement='Neural correction only; no environment bias, no task-level feedback, no forced bilateral targets',
   local_acceptance='Each replica active/hip absP95<=2deg and speedMAE<10%;32episodes per point and no observed failure',
   limitations=['Training scope only; idle and below.2 still unverified','Current4.31 spring/no added transport/J unchanged, not final contract','Static64 and local32 do not establish step/arc/random-switch/multiseed/full-domain robustness'])
 contract['evaluation_command_file']=str(OUT/'evaluation_command.json')
 (OUT/'experiment_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
 shutil.copy2(OLD/'grid64.json',OUT/'grid64.json');shutil.copy2(OLD/'fitcomp_checks.py',OUT/'fitcomp_checks.py')
 e=json.loads((SEED/'evaluation_command.json').read_text())
 for key,val in [('--checkpoint',str(OUT/'final_checkpoint.pt')),('--output',str(OUT/'evaluation.json')),('--command-grid',str(OUT/'grid64.json'))]:e['command'][e['command'].index(key)+1]=val
 for key in ['--capture-startup-steps','--capture-observation-stride']:
  i=e['command'].index(key);del e['command'][i:i+2]
 e.update(source_checkpoint=str(OUT/'final_checkpoint.pt'),source_sha256=None,state='waiting_training',scope='Full actor deterministic original64 retention; no external action bias')
 (OUT/'evaluation_command.json').write_text(json.dumps(e,indent=2)+'\n')
 formal=OUT/'formal_straight8';formal.mkdir();grid=[[v,0.] for v in [.225,.25,.275,.5,.75,1.,1.25,1.5]]
 (formal/'grid.json').write_text(json.dumps(grid)+'\n');f=json.loads(json.dumps(e))
 for key,val in [('--output',str(formal/'evaluation.json')),('--command-grid',str(formal/'grid.json')),('--replicas','16'),('--min-episodes','32'),('--steps','4800')]:f['command'][f['command'].index(key)+1]=val
 (formal/'evaluation_command.json').write_text(json.dumps(f,indent=2)+'\n')
 # Existing run/audit structure, restricted to selected networks and same physics.
 run=(OLD/'run.py').read_text().replace('subprocess.check_call([sys.executable,str(out/\'preflight.py\')],cwd=root)','subprocess.check_call([sys.executable,str(out/\'preflight.py\')],cwd=root)')
 run=run.replace("checks.update(training_checks(r,yaml.load(cfgtext,Loader=yaml.BaseLoader),out))","checks.update(training_checks(r,yaml.load(cfgtext,Loader=yaml.BaseLoader),out))\n    from rmuc_arena_com_runtime_audit import audit as arena_audit\n    checks.update(arena_audit(r)['checks'])\n    checks['training_tolerance1deg']=bool(re.search(r'^command_straight_alignment_tolerance_deg: 1(?:\\.0)?$',cfgtext,re.M))")
 run=run.replace("import sys\n", "import sys\nsys.path.insert(0,str(root/'scripts'))\n")
 run=run.replace("('rmuc_physics_adaptation.alignment_network.',)","('rmuc_physics_adaptation.alignment_network.','rmuc_physics_adaptation.straight_extension_network.')")
 start=run.index("fe=json.loads(");end=run.index("write('final_report.json'",start)
 run=run[:start]+"fe=json.loads((out/'formal_straight8/evaluation_command.json').read_text());fe.update(source_sha256=sha,state='prepared_not_launched');write('formal_straight8/evaluation_command.json',fe)\n"+run[end:]
 (OUT/'run.py').write_text(run)
 authority=(OLD/'audit_trained_authority.py').read_text().replace("gate=network_gate(candidate.rmuc_physics_adaptation,'alignment_network',x)","gate=torch.maximum(network_gate(candidate.rmuc_physics_adaptation,'alignment_network',x),network_gate(candidate.rmuc_physics_adaptation,'straight_extension_network',x))")
 (OUT/'audit_trained_authority.py').write_text(authority)
 evaluate=(OLD/'evaluate.py').read_text().replace('timeout=600','timeout=720')
 (OUT/'evaluate.py').write_text(evaluate)
 print(OUT);print(sha)
if __name__=='__main__':main()
