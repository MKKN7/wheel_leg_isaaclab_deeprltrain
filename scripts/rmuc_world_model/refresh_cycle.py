"""Collect candidate physics trajectories, gate retention, refresh WM from new data.

Never promotes a teacher. Candidate evaluation may be rejected while its valid
trajectories still improve the model. Delay schemas are kept in separate buffers.
"""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/home/kellen/anaconda3/envs/issaclab/bin/python'
BASE=ROOT/'artifacts/supermimic_pilot_20261027'

def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)

def run(command,log,budget,env=None):
    with log.open('x') as stream:
        p=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,**(env or {})),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=budget)
        except BaseException:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    if code:raise RuntimeError(f'Command failed ({code}): {log}')

def main():
    parser=argparse.ArgumentParser()
    for k in ['candidate','model','output']:parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--dual-delay-replay',action='store_true');parser.add_argument('--prepare-only',action='store_true');a=parser.parse_args()
    a.candidate=a.candidate.resolve();a.model=a.model.resolve();a.output=a.output.resolve()
    sha=hashlib.sha256(a.candidate.read_bytes()).hexdigest()
    manifest=a.output/'evaluation_command.json'
    if not manifest.exists():
        a.output.mkdir(parents=True,exist_ok=False);commands=[]
        for label,template in [('joint10',BASE/'joint_delay10_low_v1'),('sensor8',BASE/'privileged_capture_v1')]:
            d=a.output/label;d.mkdir();s=json.loads((template/'command.json').read_text());cmd=s['command']
            for flag,value in [('--checkpoint',a.candidate),('--output',d/'evaluation.json'),('--world-model-replay-dir',d/'replay')]:cmd[cmd.index(flag)+1]=str(value)
            # Only current-delay capture is fed to this WM: sensor-only has different queue shapes.
            if label=='sensor8':i=cmd.index('--world-model-replay-dir');del cmd[i:i+2]
            s.update(source_checkpoint=str(a.candidate),source_sha256=sha,baseline=str(template/'evaluation.json'),unique_variable='BPTT+GAE candidate only; paired original physical draws',state='prepared',budget_seconds=900)
            write(d/'command.json',s);commands.append(s)
        write(manifest,dict(candidate_sha256=sha,commands=commands,promoted=False))
    else:assert json.loads(manifest.read_text())['candidate_sha256']==sha
    if a.prepare_only:return
    if (a.output/'status.json').exists():raise RuntimeError('Refusing duplicate cycle; inspect existing status')
    write(a.output/'status.json',dict(state='collecting_and_evaluating',pid=os.getpid(),started=time.time()))
    # Launch independent evaluators together; both are bounded by their own runners.
    processes=[]
    try:
        for label in ['joint10','sensor8']:
            d=a.output/label
            (d/'run.py').write_text((BASE/'joint_delay10_low_v1/run.py').read_text())
            processes.append((d,subprocess.Popen([PYTHON,str(d/'run.py')],cwd=ROOT)))
        for d,p in processes:
            rc=p.wait();state=json.loads((d/'status.json').read_text())
            if rc or state['state']!='evaluation_complete':raise RuntimeError(f'Evaluation failed: {d}')
            run([PYTHON,'scripts/rmuc_world_model/analyze_dynamic_domain.py',str(d)],d/'analysis.log',120)
        run([PYTHON,'scripts/rmuc_world_model/audit_actor_runtime.py',str(a.output/'joint10')],a.output/'runtime_audit.log',120)
        reports={}
        for label,baseline in [('joint10',BASE/'joint_delay10_low_v1'),('sensor8',BASE/'privileged_capture_v1')]:
            old=json.loads((baseline/'dynamic_domain_report.json').read_text())['summary']
            new=json.loads((a.output/label/'dynamic_domain_report.json').read_text())['summary']
            reports[label]=[dict(start=x['start'],before=x['full_dynamic_pass'],after=y['full_dynamic_pass'],complete=y['complete']) for x,y in zip(old,new)]
        retained=all(x['after']>=x['before'] and x['complete']==32 for rows in reports.values() for x in rows)
        write(a.output/'retention_audit.json',dict(local_sequence_retained=retained,comparison=reports,promoted=False,scope='Two delay conditions, seven low-speed events; full old-skill domain not covered'))
        write(a.output/'status.json',dict(state='refreshing_world_model',pid=os.getpid(),local_retention=retained))
        captures=[str(BASE/'joint_delay10_low_v1'),str(BASE/'joint_delay10_vxyaw_v1'),str(BASE/'joint_delay10_direction_probe_v1'),str(a.output/'joint10')]
        if a.dual_delay_replay:captures.append(str(BASE/'privileged_capture_v1'))
        cmd=[PYTHON,'scripts/rmuc_world_model/train_pilot.py','--output',str(a.output/'refreshed_model'),'--captures',*captures,'--rich','--condition-commands','--device','cuda:0','--width','2048','--depth','3','--batch-size','1024','--steps','300','--train-horizon','8','--lr','.0002','--actor-state-loss-weight','1','--budget-seconds','300','--resume',str(a.model)]
        if a.dual_delay_replay:cmd.append('--pad-unused-delay-queues')
        write(a.output/'model_refresh_command.json',dict(command=cmd,source_model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),fresh_candidate_sha256=sha))
        run(cmd,a.output/'model_refresh.log',600,{'CUDA_VISIBLE_DEVICES':'0'})
        write(a.output/'final_report.json',dict(state='physics_and_fresh_model_refresh_complete',local_retention=retained,promoted=False,comparison=reports,next_model=str(a.output/'refreshed_model/world_model.pt'),scope='One actual model-based iteration. Not full SuperMimic reproduction or all-skill acceptance.'))
        write(a.output/'status.json',dict(state='complete',finished=time.time(),promoted=False))
    except BaseException as exc:
        write(a.output/'status.json',dict(state='failed',error=repr(exc),promoted=False));raise

if __name__=='__main__':main()
