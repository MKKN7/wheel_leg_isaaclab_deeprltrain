"""Prepare immutable frozen-policy wheel hypotheses, never launch GPU here."""
from pathlib import Path
import json,hashlib,copy
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/real2sim_wheel_20261014'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def main():
 OUT.mkdir(exist_ok=False)
 base=dict(schema_version=1,scope='real2sim_diagnostic',physical_order=['Left','Right'],wheel_joint_order=['right_wheel_spin','left_wheel_spin'],mechanical_viscous_nm_s_rad=[.0025485,.00066957],coulomb_friction_nm=[.054699,.052843],compensation_enabled=True,damping_compensation_nm_s_rad=[.0494515,.05133043],coulomb_compensation_nm=[.053699,.051843],friction_epsilon_rad_s=.1,delay_s=[.011,.007],delay_mode='none',identification_status='reported_fit_not_independently_validated',equivalent_inertia_kg_m2_metadata=[.0042075,.004204],notes=['Assume output-shaft SI and identified Left is policy physical Left; user confirmation pending','No J overwrite: equivalent fit cannot be added to existing body inertia blindly','Same tanh epsilon.1 for physical/compensating friction is an unvalidated diagnostic assumption','Motor_transport is extra transport-delay sensitivity atop retained existing DR motor lag; not identified decomposition','Feedback_sensor delays low-level damping/friction feedback only; obs25 retains existing sensor DR, not a full common-sensor deployment model','Fixed fitted B/C and fixed compensator, no perfect per-reset compensation estimator'])
 profiles={}
 profiles['nominal']=copy.deepcopy(base);profiles['nominal'].update(mechanical_viscous_nm_s_rad=[.002,.002],coulomb_friction_nm=[.001,.001],damping_compensation_nm_s_rad=[.05,.05],coulomb_compensation_nm=[0,0],delay_s=[0,0])
 profiles['fit_uncompensated']=copy.deepcopy(base);profiles['fit_uncompensated'].update(compensation_enabled=False,damping_compensation_nm_s_rad=[0,0],coulomb_compensation_nm=[0,0],delay_s=[0,0])
 profiles['fit_compensated']=copy.deepcopy(base);profiles['fit_compensated'].update(delay_s=[0,0])
 for name,mode in [('fit_motor_delay','motor_transport'),('fit_feedback_delay','feedback_sensor')]:profiles[name]=copy.deepcopy(base);profiles[name]['delay_mode']=mode
 for name,profile in profiles.items():write(OUT/(name+'.json'),profile)
 teacher=ROOT/'artifacts/continuation_v588/candidate.pt';student=ROOT/'artifacts/h723_distill_v8/student/student.pt'
 source=json.loads((ROOT/'artifacts/h723_distill_v8/formal_a/evaluation_command.json').read_text())
 grid=[[0,0],[.25,0],[.5,0],[1,0],[1,.8],[1,-.8]]
 for name in profiles:
  p=OUT/('v8_'+name);p.mkdir();write(p/'grid.json',grid);spec=copy.deepcopy(source);cmd=spec['command']
  for key,value in [('--output',str(p/'evaluation.json')),('--command-grid',str(p/'grid.json')),('--replicas','2'),('--min-episodes','4'),('--steps','3200')]:cmd[cmd.index(key)+1]=value
  cmd+=['--real2sim-wheel-profile',str(OUT/(name+'.json')),'--record-leg-tracking']
  spec.update(source_sha256=digest(teacher),student_sha256=digest(student),state='prepared',experiment='frozen v8 '+name)
  write(p/'evaluation_command.json',spec)
  write(p/'experiment_contract.json',dict(source_checkpoint=str(teacher),source_sha256=digest(teacher),student_checkpoint=str(student),student_sha256=digest(student),profile_path=str(OUT/(name+'.json')),profile_sha256=digest(OUT/(name+'.json')),unique_variable=name+' wheel profile; fixed6command screen on same frozen student and same seed/layout',peak_nm=5.5,continuous_nm=3,action_scale_nm=5.5,leg_kp=60,leg_kd=2,leg_effort_nm=40,physics_leg_policy_ms=[2,2,8],spring_nm_per_leg=2.424375,radial_force_n=0,dr_strength=.1,wheel_fit_fixed_not_randomized=True,other_DR='Legacy joint profile retained: mass/friction/inertia-shape+/-2%, motor gain+/-0.5%, motor response .2..3ms; sensor/communication one-cycle probabilities .05/.02',delay_scope=profiles[name]['notes'][3:5],budget=dict(envs=12,steps=3200,min_episodes=4,timeout_seconds=600),retention='Matched6command comparisons, not formal retention, zero-speed gate unassigned; no optimizer',stop_conditions=['600s timeout','nonzero exit','runtime mismatch'],promoted=False))
 write(OUT/'manifest.json',dict(state='prepared',cases=['v8_'+name for name in profiles],scope='Conditional frozen student real2sim hypotheses, not hardware or full sim2sim certification',model_parameters_changed=False,source_fit_sha256=digest(ROOT/'artifacts/hardware_wheel_identification_20261013/reported_fit.json')))
 print(OUT)
if __name__=='__main__':main()
