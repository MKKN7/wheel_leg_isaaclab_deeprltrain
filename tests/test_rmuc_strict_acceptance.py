import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rmuc_strict_acceptance import audit

def report(speed=.025,vx=.25):
 return dict(bank_contract='world_heading_inward_v2',rows=[dict(vx=vx,yaw=.1,linear_error=speed,yaw_error=.01,survival=1,completed_episodes=32,termination_counts={'tilt':0},roll_mean_deg=1,passed=True)])

def test_strict_boundary_and_legacy_disagreement():
 r=audit(report());assert r['nonzero_speed_passed']==0
 assert audit(report(.0249))['nonzero_speed_passed']==1
 assert audit(report(.057))['rows'][0]['legacy_stage_passed'] is True
 assert not audit(report(.057))['passed']

def test_missing_evidence_cannot_pass_even_perfect_scalars():
 r=audit(report(0));assert not r['passed'];assert r['rows'][0]['missing_evidence']

def test_zero_and_invalid_values_never_count_as_relative_pass():
 assert audit(report(0,0))['rows'][0]['checks']['speed_under10percent'] is None
 for value in [None,float('nan'),float('inf')]:assert audit(report(value))['nonzero_speed_passed']==0

def test_fall_and_outward_lean_not_masked_by_stage_pass():
 d=report(0);d['rows'][0]['termination_counts']['tilt']=1;d['rows'][0]['roll_mean_deg']=-1
 c=audit(d)['rows'][0]['checks'];assert c['no_observed_failure'] is False;assert c['mean_lean_inward'] is False
