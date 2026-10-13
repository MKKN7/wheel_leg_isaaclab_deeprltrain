"""Compare student with frozen teacher per condition; never treat quick screens as delivery."""
import argparse,json,hashlib,math
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--student',type=Path,required=True);p.add_argument('--evaluation',type=Path,nargs='+',required=True);p.add_argument('--teacher-evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sha=hashlib.sha256(a.student.read_bytes()).hexdigest();baseline=json.loads(a.teacher_evaluation.read_text());old={(r['vx'],r['yaw']):r for r in baseline['rows']};rows=[]
 for path in a.evaluation:
  e=json.loads(path.read_text());assert e['student_sha256']==sha and e['checkpoint_sha256']==baseline['checkpoint_sha256']
  for k in ['dr_strength','gas_spring_assist_torque_nm','checkpoint_observation_contract','sensor_model','actual_wheel_actuator_effort_limit_range_nm']:
   assert e[k]==baseline[k],k
  rows+=e['rows']
 assert len({(r['vx'],r['yaw']) for r in rows})==len(rows)
 required={key for key,r in old.items() if r['passed']};by={(r['vx'],r['yaw']):r for r in rows};assert required<=by.keys()
 retained=[key for key in required if by[key]['passed']];lost=[by[k] for k in sorted(required-set(retained))]
 reg=[r for key,r in by.items() if r['survival']+1e-6<old[key]['survival']]
 minimum=min(r['completed_episodes'] for r in rows);ratio=len(retained)/len(required)
 report=dict(student_sha256=sha,teacher_sha256=baseline['checkpoint_sha256'],passed=minimum>=32 and ratio>=.9 and not reg and not lost,promoted=False,minimum_episodes=minimum,retention_fraction=ratio,retained=len(retained),teacher_passed=len(required),total_passed=sum(r['passed'] for r in rows),lost_conditions=lost,survival_regressions=reg,required_count_90_percent=math.ceil(.9*len(required)),mandatory_skill_policy='All previously accepted teacher skills retained;90percent alone does not override required skills',limitations=['Isaac same-contract gate only; MuJoCo/FSM/CubeAI/board timing require separate evidence'])
 a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['lost_conditions','survival_regressions']},indent=2))
if __name__=='__main__':main()
