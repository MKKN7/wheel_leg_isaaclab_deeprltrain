"""Prepare next teacher batch for verified idle/creep/bridge gaps; no launch."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/teacher_course_20261009/low_speed_domain_20261011'
OLD=BASE/'straight_domain_ppo_20261014'
SEED=BASE/'straight_idle_domain_screen_20261014'
OUT=BASE/'low_command_joint_ppo_20261014'

def main():
 OUT.mkdir(exist_ok=False)
 source=SEED/'candidate.pt';sha=hashlib.sha256(source.read_bytes()).hexdigest()
 s=json.loads((OLD/'command.json').read_text());cmd=s['command']
 def flag(k,v):cmd[cmd.index(k)+1]=str(v)
 def cfg(k,v):
  token='env.'+k+'='+json.dumps(v,separators=(',',':'))
  index=next((i for i,t in enumerate(cmd) if t.startswith('env.'+k+'=')),None)
  if index is None:cmd.append(token)
  else:cmd[index]=token
 def get(k):return json.loads(next(t.split('=',1)[1] for t in cmd if t.startswith('env.'+k+'=')))
 for k,v in [('--checkpoint',source),('--run_name',OUT.name),('--command_level',27),('--num_envs',1024)]:flag(k,v)
 changes={'command_stage_linear_x_mps':[.35,.1,0.], 'command_stage_yaw_rate_rad_s':[0.,0.,0.],
          'command_stage_leg_scale_rad':[.194,.19,.19], 'command_stage_names':['straight035_bridge','straight010_creep','idle_neural_alignment']}
 for key,values in changes.items():
  allvalues=get(key);allvalues[25:28]=values;cfg(key,allvalues)
 weights=[4,1,1,1,1,5,5,1,1,1,1,1,1,1,1,1,5,1,5,1,1,1,4,4,6,12,12]
 for key in ['command_rehearsal_anchor_stage_ids','command_rehearsal_priority_stage_ids']:cfg(key,list(range(27)))
 cfg('command_rehearsal_gate_stage_ids',list(range(28)));cfg('command_rehearsal_priority_stage_weights',weights)
 cfg('command_continuous_straight_speed_range',[0.,.5]);cfg('command_exact_low_speed_reward_enabled',True)
 s.update(source_checkpoint=str(source),source_sha256=sha,run_name=OUT.name,state='prepared_pending_idle_screen',
   unique_variable='Joint idle/creep/bridge neural learning with true low-speed reward reference; unchanged physics, freeze all other actor weights',
   training_plot_conditions=[[v,0.] for v in [0.,.1,.225,.25,.275,.35,.5,.75,1.,1.25,1.5]])
 (OUT/'command.json').write_text(json.dumps(s,indent=2)+'\n')
 c=json.loads((OLD/'experiment_contract.json').read_text());c.update(source_checkpoint=str(source),source_sha256=sha,unique_variable=s['unique_variable'],
  experiment_type='low_command_joint_neural_ppo',budget=dict(updates=80,envs=1024,steps_per_env=128,timeout_seconds=2400),
  reward_change='Opt-in true low-speed reference; denominator floor separated; no idle forward-progress reward; low-speed deadband narrowed; >=.2 reward identical',
  sampling=dict(fixed_idle_fraction=.15,continuous_straight_range=[0.,.5],continuous_fraction=.1,rehearsal_fraction=.75,priority_ids=list(range(27)),priority_weights=weights),
  retention='Frozen all other actor parameters; original64 plus focused low15; conditional nonzero16x32 formal; zero absolute speed/drift gate remains unspecified',
  evaluation_command_file=str(OUT/'evaluation_command.json'),
  limitations=['No zero-speed release gate asserted from relative10percent','No full domain/multiseed/dynamics certificate','Spring4.31,DR.1,zero addedtransport,J unchanged'])
 (OUT/'experiment_contract.json').write_text(json.dumps(c,indent=2)+'\n')
 for name in ['run.py','evaluate.py','report.py','fitcomp_checks.py','audit_trained_authority.py','preflight.py','grid64.json']:
  shutil.copy2(OLD/name,OUT/name)
 run=(OUT/'run.py').read_text().replace('formal_straight8','formal_straight16')
 run=run.replace("checks['training_tolerance1deg']=", "checks['exact_low_speed_reward']=bool(re.search(r'^command_exact_low_speed_reward_enabled: true$',cfgtext,re.M|re.I))\n    checks['training_tolerance1deg']=")
 (OUT/'run.py').write_text(run)
 pre=(OUT/'preflight.py').read_text().replace("cfg.command_continuous_straight_speed_range==[.5,1.5]","cfg.command_continuous_straight_speed_range==[0.,.5]")
 pre=pre.replace("set(range(24))==set(cfg.command_rehearsal_priority_stage_ids)","set(range(27))==set(cfg.command_rehearsal_priority_stage_ids)")
 pre=pre.replace("for v in [.225,.25,.275,.5,.625,.75,.875,1.,1.125,1.25,1.375,1.5]", "for v in [0.,.05,.1,.15,.225,.25,.275,.35,.5,.625,.75,.875,1.,1.125,1.25,1.375,1.5]")
 pre=pre.replace("g['gates']==[1.]", "all(v>0 for v in g['gates'])")
 pre=pre.replace("checks['no_duplicate']=not duplicates", "checks['no_duplicate']=not duplicates\nchecks['exact_low_speed_reward']=cfg.command_exact_low_speed_reward_enabled is True")
 (OUT/'preflight.py').write_text(pre)
 report=(OUT/'report.py').read_text().replace("out.parent/'straight_domain_neural_seed_20261014/evaluation.json'", "out.parent/'straight_domain_ppo_20261014/evaluation.json'")
 report=report.replace("screen_targets=[x for x in rows if x['vx'] in [.225,.25,.5,.75,1.,1.25,1.5]]", "screen_targets=[x for x in rows if x['vx']>0]")
 report=report.replace("'formal_pass=a['passed']", "'formal_pass=a['passed']")
 (OUT/'report.py').write_text(report)
 e=json.loads((OLD/'evaluation_command.json').read_text())
 for key,value in [('--checkpoint',str(OUT/'final_checkpoint.pt')),('--output',str(OUT/'evaluation.json')),('--command-grid',str(OUT/'grid64.json'))]:e['command'][e['command'].index(key)+1]=value
 e.update(source_checkpoint=str(OUT/'final_checkpoint.pt'),source_sha256=None,state='waiting_training')
 (OUT/'evaluation_command.json').write_text(json.dumps(e,indent=2)+'\n')
 for name,grid,reps,episodes,steps in [
  ('focused_low15',json.loads((SEED/'grid.json').read_text()),2,4,3200),
  ('formal_straight16',[[v,0.] for v in [.05,.1,.15,.175,.2,.225,.25,.275,.3,.35,.4,.5,.75,1.,1.25,1.5]],8,32,6500)]:
  d=OUT/name;d.mkdir();(d/'grid.json').write_text(json.dumps(grid)+'\n');f=json.loads(json.dumps(e))
  for key,value in [('--output',str(d/'evaluation.json')),('--command-grid',str(d/'grid.json')),('--replicas',str(reps)),('--min-episodes',str(episodes)),('--steps',str(steps))]:f['command'][f['command'].index(key)+1]=value
  (d/'evaluation_command.json').write_text(json.dumps(f,indent=2)+'\n')
 (OUT/'source_checkpoint_hash.json').write_text(json.dumps(dict(checkpoint=str(source),sha256=sha),indent=2)+'\n')
 print(OUT);print(sha)
if __name__=='__main__':main()
