"""Per-direction nominal recovery retention; never a deployment certificate."""
import argparse
import hashlib
import json
from pathlib import Path

CASES=('front_down','back_down','left_wheel_side_down','right_wheel_side_down')
ANGLES=(0,5,15,30,90)


def direction_counts(report):
    if (report.get('diagnostic_communication_delay_steps') is not None
            or report.get('diagnostic_communication_delay_probability') is not None
            or report.get('evaluation_communication_delay_probability_override',-1.) != -1.
            or report.get('diagnostic_communication_delay_cap_steps') is not None
            or report.get('evaluation_communication_delay_cap_steps',-1) != -1):
        raise ValueError('Communication-delay diagnostic cannot validate nominal retention')
    if (report.get('recovery_ground',{}).get('inclination_degrees',0.) != 0.
            or report.get('diagnostic_slope_degrees') is not None
            or report.get('diagnostic_slope_azimuth_degrees') is not None):
        raise ValueError('Slope evaluation cannot validate flat nominal retention')
    if report.get('diagnostic_dr_strength_override') is not None or report.get('diagnostic_hardware_profile') is not None:
        raise ValueError('DR/hardware diagnostic cannot validate nominal retention')
    if report.get('diagnostic_zero_action_after_seconds') is not None:
        raise ValueError('Intervention report cannot validate policy retention')
    if report.get('policy_behavior_evaluation',True) is not True:
        raise ValueError('Not an unmodified policy evaluation')
    if report.get('diagnostic_joint_tolerance_override') is not None:
        raise ValueError('Diagnostic tolerance cannot validate retention')
    rows=report['rows']
    if len(rows)!=16 or any(sum(r['case']==case for r in rows)!=4 for case in CASES):
        raise ValueError('Require four complete replicas per direction')
    result={}
    for case in CASES:
        group=[r for r in rows if r['case']==case]
        for row in group:
            terminal=row['terminal']
            expected=bool(terminal['final_established'] and not terminal['unsafe'])
            if bool(row['recovery_hold_passed'])!=expected:
                raise ValueError('Hold/unsafe report inconsistency')
        result[case]=dict(passed=sum(r['recovery_hold_passed'] for r in group),
            unsafe=sum(r['terminal']['unsafe'] for r in group))
    return result


def compare_reports(reference,candidate):
    if set(reference)!=set(ANGLES) or set(candidate)!=set(ANGLES):
        raise ValueError('All five scheduled angles are required')
    for reports in (reference,candidate):
        if len({r['checkpoint_sha256'] for r in reports.values()})!=1:
            raise ValueError('Mixed checkpoint hashes')
    rows=[];regressions=[];reference_passed_groups=0;provenance_gaps=[]
    for angle in ANGLES:
        a,b=reference[angle],candidate[angle]
        if a['initial_tilt_degrees']!=angle or b['initial_tilt_degrees']!=angle:
            raise ValueError('Wrong angle')
        for key in ('gas_spring_assist_torque_nm','gas_spring_radial_force_n',
                    'applied_leg_target_slew_rate_rad_s','joint_limit_tolerance_rad'):
            if a[key]!=b[key]:raise ValueError('Mismatched contract: '+key)
        for key in ('recovery_criteria','evaluation_seed','evaluation_reset_mode'):
            if a.get(key) is None or b.get(key) is None:
                # Older reports remain descriptive historical evidence. Missing
                # metadata must never be silently treated as a matched contract.
                provenance_gaps.append(dict(angle=angle,field=key,
                    missing_reference=a.get(key) is None,missing_candidate=b.get(key) is None))
            elif a[key]!=b[key]:
                raise ValueError('Mismatched evaluation provenance: '+key)
        ca,cb=direction_counts(a),direction_counts(b)
        for case in CASES:
            protected=ca[case]['passed']==4
            lost=protected and cb[case]['passed']!=4
            reference_passed_groups+=int(protected)
            if lost:regressions.append(dict(angle=angle,case=case))
            rows.append(dict(angle=angle,case=case,reference=ca[case],candidate=cb[case],
                protected=protected,regressed=lost))
    target=next(r for r in rows if r['angle']==15 and r['case']=='back_down')
    return dict(rows=rows,regressions=regressions,protected_groups=reference_passed_groups,
        nominal_retention_passed=not regressions,
        back15_all_replicas_passed=target['candidate']['passed']==4,
        eligible_for_nominal_review=(not regressions and target['candidate']['passed']==4),
        evaluation_provenance_matched=not provenance_gaps,
        evaluation_provenance_gaps=provenance_gaps,
        promoted=False,certified=False,maximum_dr_verified=False,
        scope='Descriptive fixed-start DR0 retention; full physics/threshold provenance, multi-seed and randomized robustness require separate review')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    loaded=[];hashes={}
    for directory in (args.reference,args.candidate):
        status=json.loads((directory/'status.json').read_text())
        if status['state']!='complete':raise RuntimeError('Worker not complete: '+str(directory))
        reports={}
        for angle in ANGLES:
            path=directory/f'evaluation_{angle}.json'
            report=json.loads(path.read_text())
            if report['checkpoint_sha256']!=status['checkpoint_sha256']:
                raise ValueError('Report does not match worker checkpoint')
            reports[angle]=report
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        loaded.append(reports)
    result=compare_reports(*loaded);result['report_sha256']=hashes
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('nominal_retention_passed','back15_all_replicas_passed','regressions')},indent=2))


if __name__=='__main__':main()
