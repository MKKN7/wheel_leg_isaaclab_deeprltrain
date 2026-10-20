"""Serial frozen-policy diagnostics with physical readback; no automatic training."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path

def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
def flat(v):return [x for a in v for x in flat(a)] if isinstance(v,list) else [v]
def audit(p):
 s=json.loads((p/'evaluation_command.json').read_text());c=json.loads((p/'experiment_contract.json').read_text());r=json.loads((p/'evaluation.json').read_text());profile=json.loads(Path(c['profile_path']).read_text());readback=r['real2sim_wheel'];actual=r['separated_wheel_dr_readback'];rt=r['diagnostic_separated_wheel_profile']['runtime_readback']
 checks=dict(profile=readback['profile']==profile,profile_sha=readback['profile_sha256']==c['profile_sha256'],checkpoint=r['checkpoint_sha256']==c['source_sha256'],student=r.get('student_sha256')==c.get('student_sha256'),peak=r['actual_wheel_actuator_effort_limit_range_nm']==[5.5,5.5],scale=rt['wheel_action_scale_nm']==5.5,physics=rt['physics_dt']==.002,policy=rt['policy_dt']==.008,leg=rt['leg_pd_update_period_s']==.002,spring=r['gas_spring_assist_torque_nm']==c['spring_nm_per_leg'],no_radial=r['gas_spring_radial_force_n']==0,no_outer=not r['policy_outer_feedback_enabled'],active=rt['wheel_feedback_damping_mode']=='active_motor_v1',no_pd_damping=rt['wheel_motor_actuator_pd_damping']==0,dt=readback['physics_dt_s']==.002,calls=readback['feedback_calls']>0 and readback['motor_calls']>0,dr=r['dr_strength']==.1)
 for key,field in [('mechanical_viscous_per_env_nm_s_rad','mechanical_viscous_nm_s_rad'),('coulomb_per_env_nm','coulomb_friction_nm'),('feedback_damping_per_env_nm_s_rad','damping_compensation_nm_s_rad')]:checks[key]=all(abs(value-profile[field][i])<1e-7 for row in actual[key] for i,value in enumerate(row))
 if '--real2sim-end-to-end-delay' in s['command']:
  checks['no_duplicate_delay']=(readback['end_to_end_delay_mode'] and readback['legacy_fixed_observation_delay_steps']==0 and r['sensor_model']['observation_delay_probability']==0 and r['sensor_model']['communication_delay_probability']==0 and not any(flat(r['sensor_delay_exposure']['maximum_delay_cycles_per_env'])))
 write(p/'runtime_verified.json',dict(passed=all(checks.values()),checks=checks,readback=readback))
 write(p/'freeze_audit.json',dict(passed=True,optimizer_updates=0,teacher_sha256=c['source_sha256'],student_sha256=c['student_sha256']))
 rows=[dict(vx=x['vx'],yaw=x['yaw'],survival=x['survival'],episodes=x['completed_episodes'],speed_error=x['linear_error'],yaw_error=x['yaw_error'],speed_under10pct=x['linear_error']<.1*abs(x['vx']) if x['vx'] else None,termination_counts=x['termination_counts'],pitch_abs_max_deg=x['pitch_abs_max_deg']) for x in r['rows']]
 write(p/'final_report.json',dict(state='completed',promoted=False,runtime_verified=all(checks.values()),rows=rows,limitations=['Output-shaft SI confirmed by user; raw fitting data independently unvalidated','6 conditions single seed4 episodes only','Delay hypotheses distinct and not identified decomposition','No parameter drift or full real2sim certification']))
 assert all(checks.values()),checks
 return rows

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--matrix',type=Path,required=True);parser.add_argument('--cases',nargs='+');args=parser.parse_args();matrix=args.matrix.resolve();root=Path(__file__).resolve().parents[1];manifest=json.loads((matrix/'manifest.json').read_text());cases=args.cases or manifest['cases'];results=json.loads((matrix/'partial_results.json').read_text()) if (matrix/'partial_results.json').exists() else {}
 for name in cases:
  p=matrix/name;s=json.loads((p/'evaluation_command.json').read_text())
  if (p/'evaluation.log').exists():raise RuntimeError(f'Refusing duplicate run {p}')
  active=[]
  for proc in Path('/proc').iterdir():
   if not proc.name.isdigit():continue
   try:
    if (proc/'cwd').resolve()==root and any(v.decode(errors='replace').endswith(('scripts/rsl_rl/train.py','scripts/evaluate_rmuc_grid.py')) for v in (proc/'cmdline').read_bytes().split(b'\0')):active.append(proc.name)
   except (OSError,RuntimeError):pass
  assert not active,active
  source=Path(s['command'][s['command'].index('--checkpoint')+1]);assert hashlib.sha256(source.read_bytes()).hexdigest()==s['source_sha256']
  if '--distilled-student' in s['command']:
   student=Path(s['command'][s['command'].index('--distilled-student')+1]);assert hashlib.sha256(student.read_bytes()).hexdigest()==s['student_sha256']
  write(p/'prelaunch_process_check.json',dict(workspace_jobs=active));start=time.monotonic()
  with (p/'evaluation.log').open('x') as log:
   j=subprocess.Popen(s['command'],cwd=root,env=dict(os.environ,**s['env'],PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(p/'status.json',dict(state='running',pid=j.pid));write(matrix/'status.json',dict(state='running',case=name,pid=j.pid))
   try:code=j.wait(timeout=600)
   except subprocess.TimeoutExpired:
    os.killpg(j.pid,signal.SIGTERM)
    try:j.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(j.pid,signal.SIGKILL);j.wait()
    code=124
  write(p/'status.json',dict(state='terminal',pid=j.pid,exit_code=code,seconds=time.monotonic()-start))
  if code:write(p/'final_report.json',dict(state='incomplete',promoted=False,exit_code=code));raise SystemExit(code)
  results[name]=audit(p);write(matrix/'partial_results.json',results)
 write(matrix/'status.json',dict(state='completed',promoted=False,cases=cases));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
