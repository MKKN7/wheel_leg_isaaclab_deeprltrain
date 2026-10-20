"""Measured straight/idle linkage-alignment gate; turns are not forced symmetric."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def audit(report,frames,bucket_ids,tolerance_deg=2.,warmup=1200,allow_creep_exception=False):
    rows=[]
    for bucket,row in enumerate(report['rows']):
        if abs(row['yaw'])>1e-4:continue
        creep_exception=bool(allow_creep_exception and abs(row['vx']-.05)<1e-8 and abs(row['yaw'])<1e-8)
        speed_limit=.006 if creep_exception else .1*abs(row['vx'])
        env_ids=np.flatnonzero(bucket_ids==bucket);per=[]
        for env in env_ids:
            x=frames[:,env];x=x[x[:,0]>=warmup]
            finite=len(x)>0 and np.isfinite(x).all()
            active=float(np.percentile(abs(np.rad2deg(x[:,2]-x[:,5])),95)) if finite else None
            hip=float(np.percentile(abs(np.rad2deg(x[:,3]+x[:,6])),95)) if finite else None
            speed_mae=float(np.abs(x[:,1]-row['vx']).mean()) if finite else None
            per.append(dict(env=int(env),samples=len(x),active_abs_p95_deg=active,hip_abs_p95_deg=hip,
                            speed_mae_mps=speed_mae,
                            speed_under10percent=bool(finite and speed_mae<.1*abs(row['vx'])) if row['vx'] else None,
                            speed_within_declared_limit=bool(finite and speed_mae<speed_limit) if row['vx'] else None,
                            passed=bool(finite and active<=tolerance_deg and hip<=tolerance_deg)))
        vx=row['vx'];speed=(row['linear_error'] is not None and row['linear_error']<.1*abs(vx)) if vx else None
        no_failure=row['survival']==1 and bool(row.get('termination_counts')) and all(v==0 for v in row['termination_counts'].values())
        checks=dict(all_replicas_alignment=bool(per) and all(x['passed'] for x in per),speed_under10percent=speed,
                    all_replicas_speed_under10percent=bool(per) and all(x['speed_under10percent'] is True for x in per),
                    episodes_at_least32=row['completed_episodes']>=32,no_observed_failure=no_failure)
        if allow_creep_exception:
            del checks['speed_under10percent'];del checks['all_replicas_speed_under10percent']
            checks['speed_within_declared_limit']=bool(row['linear_error'] is not None and row['linear_error']<speed_limit) if vx else None
            checks['all_replicas_speed_within_declared_limit']=bool(per) and all(x['speed_within_declared_limit'] is True for x in per)
        rows.append(dict(vx=vx,yaw=row['yaw'],replicas=per,checks=checks,passed=all(v is True for v in checks.values()),
                         speed_limit_mps=speed_limit,creep_exception_applied=creep_exception,raw_speed_under10percent=speed))
    return dict(scope='Straight/idle physical linkage deltas only; NOT whole-leg angle or full delivery certificate',
                tolerance_deg=tolerance_deg,warmup_control_steps=warmup,rows=rows,passed=bool(rows) and all(r['passed'] for r in rows),
                zero_speed_gate='Separate idle gate required; never auto-pass',promoted=False,
                creep_exception_enabled=allow_creep_exception,
                creep_exception_scope='User-authorized positive0.05m/s straight only, absoluteMAE<.006; no episode/posture/idle gate relaxation')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();e=a.experiment
    report=json.loads((e/'evaluation.json').read_text());path=e/'evaluation.leg_tracking.npz';data=np.load(path)
    result=audit(report,data['frames'],data['bucket_ids']);result.update(checkpoint_sha256=report['checkpoint_sha256'],trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with (e/'alignment_acceptance.json').open('x') as f:json.dump(result,f,indent=2)
    print('Alignment conditions:',len(result['rows']),'passed:',result['passed'])
