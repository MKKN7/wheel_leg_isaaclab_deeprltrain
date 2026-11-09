import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('batch_audit',Path(__file__).resolve().parents[1]/'scripts/rmuc_world_model/audit_batch_progress.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def sample(passes,yaw):
 return dict(summary=[dict(vx=1.8,yaw=0,replicas=2,complete=2,strict_screen_pass=passes)],rows=[dict(vx=1.8,yaw=0,speed_tail_mae=.05,yaw_tail_mae=yaw),dict(vx=1.8,yaw=0,speed_tail_mae=None,yaw_tail_mae=None)])
def test_rejects_pass_count_gain_when_tracking_worsens():
 d=m.review(sample(0,.06),sample(1,.09));assert not d['eligible_for_broad_evaluation'];assert d['net_pass_gain']==1

def test_allows_improving_candidate_for_further_review_only():
 d=m.review(sample(0,.09),sample(1,.04));assert d['eligible_for_broad_evaluation'];assert d['promoted'] is False

def test_high_delay_gain_does_not_hide_low_delay_loss():
 def row(i,angle):return dict(env=i,vx=1.8,yaw=0,angle_p95_deg=[angle,angle],checks=dict(complete=True,speed=True,yaw=True,straight_angles=angle<=2))
 a=dict(rows=[row(0,2.5),row(1,2.5),row(2,4)])
 b=dict(rows=[row(0,4),row(1,4),row(2,2.5)])
 d=m.engineering_retention(a,b,[0,1,4]);assert not d['eligible'];assert d['groups']['low']['source']==2;assert d['groups']['high']['candidate']==1

def test_missing_tail_and_pairing_are_not_success():
 import pytest
 a=dict(rows=[dict(env=0,vx=1.8,yaw=0,angle_p95_deg=[2.5,2.5],checks=dict(complete=True,speed=True,yaw=True,straight_angles=False))])
 b=dict(rows=[dict(a['rows'][0],angle_p95_deg=None)])
 assert not m.engineering_retention(a,b,[0])['eligible']
 b['rows'][0]['env']=1
 with pytest.raises(ValueError):m.engineering_retention(a,b,[0])

def test_targeted_high_delay_objective_rejects_only_low_delay_gains():
 def row(i,ok):return dict(env=i,vx=1.8,yaw=0,angle_p95_deg=[1,1],checks=dict(complete=True,speed=True,yaw=ok,straight_angles=True))
 a=dict(rows=[row(0,False),row(1,False)])
 b=dict(rows=[row(0,True),row(1,False)])
 assert m.engineering_retention(a,b,[0,4])['eligible']
 assert not m.engineering_retention(a,b,[0,4],True)['eligible']
 assert not m.engineering_retention(a,b,None,True)['eligible']
 b['rows'][1]=row(1,True)
 assert m.engineering_retention(a,b,[0,4],True)['eligible']
