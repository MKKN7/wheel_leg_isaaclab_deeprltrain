import copy
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/compare_rmuc_recovery.py'
spec=importlib.util.spec_from_file_location('recovery_comparison',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def reports():
    result={}
    for angle in m.ANGLES:
        rows=[]
        for case in m.CASES:
            passed=angle in (0,5) or (angle==15 and case!='back_down') or (angle==30 and case=='front_down')
            rows.extend(dict(case=case,recovery_hold_passed=passed,
                terminal=dict(final_established=passed,unsafe=not passed)) for _ in range(4))
        result[angle]=dict(checkpoint_sha256='reference',initial_tilt_degrees=angle,rows=rows,
            gas_spring_assist_torque_nm=10.775,gas_spring_radial_force_n=0.,
            applied_leg_target_slew_rate_rad_s=3.5,joint_limit_tolerance_rad=0.)
    return result


def test_target_improvement_cannot_hide_retention_loss():
    a=reports();b=copy.deepcopy(a)
    for r in b[15]['rows']:
        if r['case']=='back_down':r.update(recovery_hold_passed=True,terminal=dict(final_established=True,unsafe=False))
    b[0]['rows'][0].update(recovery_hold_passed=False,terminal=dict(final_established=False,unsafe=False))
    result=m.compare_reports(a,b)
    assert result['back15_all_replicas_passed']
    assert not result['nominal_retention_passed'] and not result['eligible_for_nominal_review']
    assert result['regressions']==[dict(angle=0,case='front_down')]


@pytest.mark.parametrize('bad',['intervention','hash','missing','contract','fake_pass','dr','profile','communication'])
def test_rejects_incomplete_or_incompatible_evidence(bad):
    a=reports();b=copy.deepcopy(a)
    if bad=='intervention':b[0]['diagnostic_zero_action_after_seconds']=2.
    if bad=='hash':b[5]['checkpoint_sha256']='different'
    if bad=='missing':del b[90]
    if bad=='contract':b[0]['gas_spring_assist_torque_nm']=0.
    if bad=='fake_pass':b[15]['rows'][4]['recovery_hold_passed']=True
    if bad=='dr':b[0]['diagnostic_dr_strength_override']=1.
    if bad=='profile':b[0]['diagnostic_hardware_profile']={'wheel_effort_limit_nm':3.}
    if bad=='communication':b[0]['diagnostic_communication_delay_steps']=0
    with pytest.raises(ValueError):m.compare_reports(a,b)


@pytest.mark.parametrize('field,reference,candidate',[
    ('recovery_criteria',{'min_height_m':.28,'hold_seconds':.5},
                         {'min_height_m':.25,'hold_seconds':.5}),
    ('recovery_criteria',{'min_height_m':.28,'hold_seconds':.5},
                         {'min_height_m':.28,'hold_seconds':.2}),
    ('evaluation_seed',268,269),
    ('evaluation_reset_mode','fixed_angle_four_directions_equal_replicas','training_cohorts'),
])
def test_rejects_changed_thresholds_seed_or_reset_distribution(field,reference,candidate):
    a=reports();b=copy.deepcopy(a)
    a[5][field]=reference;b[5][field]=candidate
    with pytest.raises(ValueError,match='Mismatched evaluation provenance'):
        m.compare_reports(a,b)


def test_legacy_metadata_missing_is_reported_not_claimed_matched():
    result=m.compare_reports(reports(),reports())
    assert not result['evaluation_provenance_matched']
    assert len(result['evaluation_provenance_gaps'])==15
    assert not result['certified']


@pytest.mark.parametrize('fields',[
    {'recovery_ground':{'inclination_degrees':2.}},
    {'diagnostic_slope_degrees':0.},
    {'diagnostic_slope_azimuth_degrees':90.},
])
def test_slope_reports_cannot_enter_flat_retention(fields):
    a=reports();b=copy.deepcopy(a);b[0].update(fields)
    with pytest.raises(ValueError,match='Slope evaluation'):
        m.compare_reports(a,b)
