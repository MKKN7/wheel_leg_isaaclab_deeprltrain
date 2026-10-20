from pathlib import Path
import copy,hashlib,json
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'artifacts/real2sim_wheel_20261014'
b=ROOT/'artifacts/teacher_course_20261009/low_speed_domain_20261011';source=b/'straight0225_coupled_finetune_20261013/final_checkpoint.pt';sha=hashlib.sha256(source.read_bytes()).hexdigest()
base=json.loads((b/'straight0225_finetune_formal_20261013/evaluation_command.json').read_text())
for profile in ['nominal','fit_compensated']:
 p=out/('teacher_'+profile);p.mkdir();grid=[[.225,0],[.25,0],[.275,0],[.5,0],[1,.8],[1,-.8]];(p/'grid.json').write_text(json.dumps(grid)+'\n');spec=copy.deepcopy(base);cmd=spec['command']
 for key,value in [('--checkpoint',str(source)),('--output',str(p/'evaluation.json')),('--command-grid',str(p/'grid.json')),('--replicas','2'),('--min-episodes','4'),('--steps','3200')]:cmd[cmd.index(key)+1]=value
 cmd+=['--real2sim-wheel-profile',str(out/(profile+'.json'))]
 spec.update(source_sha256=sha,state='prepared',student_sha256=None);(p/'evaluation_command.json').write_text(json.dumps(spec,indent=2)+'\n')
 contract=json.loads((out/('v8_'+profile)/'experiment_contract.json').read_text());contract.update(source_checkpoint=str(source),source_sha256=sha,student_checkpoint=None,student_sha256=None,spring_nm_per_leg=4.31,unique_variable='Matched current teacher wheel profile '+profile,retention='6point frozen comparison; no complete formal32 retention',promoted=False)
 (p/'experiment_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
print('Prepared teacher matched real2sim cases')
