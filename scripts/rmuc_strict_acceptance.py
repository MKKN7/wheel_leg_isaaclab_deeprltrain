"""Fail-closed absolute acceptance shared by teacher and distilled student reports.

Legacy rows are preserved for comparisons. Missing trajectory/hardware evidence
is never inferred from scalar tracking averages. This is not a deployment certificate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

CONTRACT='rmuc_absolute_acceptance_20261013_v1'

def below(value,limit):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value < limit

def audit(report):
    rows=[]
    for row in report.get('rows',[]):
        vx=row.get('vx');yaw=row.get('yaw');missing=[]
        checks={
            'episodes_at_least32':row.get('completed_episodes',0)>=32,
            'no_observed_failure':row.get('survival')==1 and isinstance(row.get('termination_counts'),dict) and bool(row['termination_counts']) and all(v==0 for v in row['termination_counts'].values()),
            'yaw_error_under0p05':below(row.get('yaw_error'),.05),
        }
        if vx is None or vx==0:
            checks['speed_under10percent']=None
            missing.append('zero-command absolute speed/drift gate requires separately specified limits and time-series evidence')
        else:checks['speed_under10percent']=below(row.get('linear_error'),.1*abs(vx))
        # Do not fabricate waveform properties from mean roll or max absolute pitch.
        missing.extend(['roll/pitch fluctuation under5deg and no leg resonance require validated time series',
                        'step overshoot under20percent and arrival under0.5s require step-response evidence',
                        'dynamic switching and multiseed retention not established by this row'])
        if vx and yaw:
            dynamics=row.get('turn_dynamics_steady') or {}
            lean=dynamics.get('physical_inward_lean_deg')
            if lean is None and isinstance(row.get('roll_mean_deg'),(int,float)) and report.get('bank_contract')=='world_heading_inward_v2':
                lean=row['roll_mean_deg']*(1 if vx*yaw>0 else -1)
            checks['mean_lean_inward']=below(0,lean) if isinstance(lean,(int,float)) else None
            missing.append('arc cross-track under0.1m and continuously controllable radius require world trajectory evidence')
        checks['required_trajectory_evidence']=None
        rows.append(dict(vx=vx,yaw=yaw,legacy_stage_passed=row.get('passed'),checks=checks,
                         failed=[k for k,v in checks.items() if v is False],missing_evidence=missing,passed=False))
    return dict(contract=CONTRACT,applies_to=['teacher','small_mlp'],passed=False,promoted=False,
                scope='Available-metric audit only; complete absolute acceptance requires additional measured evidence',
                nonzero_speed_tested=sum(r.get('vx') not in (None,0) for r in rows),
                nonzero_speed_passed=sum(r['checks']['speed_under10percent'] is True for r in rows),
                legacy_stage_passed=sum(r['legacy_stage_passed'] is True for r in rows),rows=rows,
                limitation='No static report-only route to release; teacher90percent retention cannot override missing absolute gates')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(json.loads(a.evaluation.read_text()));result['source_sha256']=hashlib.sha256(a.evaluation.read_bytes()).hexdigest()
    with a.output.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print({k:v for k,v in result.items() if k!='rows'})
