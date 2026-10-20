import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rmuc_alignment_acceptance import audit

def test_each_replica_required_and_hip_mirror():
 f=np.zeros((20,2,8));f[:,:,0]=1300;f[:,:,1]=.25;f[:,:,3]=.1;f[:,:,6]=-.1
 row=dict(vx=.25,yaw=0,linear_error=.01,survival=1,completed_episodes=32,termination_counts={'tilt':0})
 r=dict(rows=[row]);assert audit(r,f,np.array([0,0]))['passed']
 f[:,1,2]=np.deg2rad(8)
 assert not audit(r,f,np.array([0,0]))['passed']
def test_turns_missing_data_and_zero_speed_not_passed():
 f=np.zeros((20,1,8));r=dict(rows=[dict(vx=0,yaw=0,linear_error=0,survival=1,completed_episodes=32,termination_counts={'tilt':0})])
 assert not audit(r,f,np.array([0]))['passed']
 f[:,:,0]=1300;assert not audit(r,f,np.array([0]))['passed']
 r['rows'][0]['yaw']=.1;assert audit(r,f,np.array([0]))['rows']==[]

def test_pooled_speed_cannot_hide_bad_replica():
 f=np.zeros((20,2,8));f[:,:,0]=1300;f[:,0,1]=.25;f[:,1,1]=.21
 row=dict(vx=.25,yaw=0,linear_error=.02,survival=1,completed_episodes=32,termination_counts={'tilt':0})
 result=audit(dict(rows=[row]),f,np.array([0,0]))
 assert result['rows'][0]['checks']['speed_under10percent']
 assert not result['rows'][0]['checks']['all_replicas_speed_under10percent']
 assert not result['passed']

def test_user_creep_exception_is_optin_and_does_not_hide_raw_failure():
 f=np.zeros((20,2,8));f[:,:,0]=1300;f[:,:,1]=.0555
 row=dict(vx=.05,yaw=0,linear_error=.0055,survival=1,completed_episodes=32,termination_counts={'tilt':0})
 report=dict(rows=[row]);ids=np.array([0,0])
 assert not audit(report,f,ids)['passed']
 r=audit(report,f,ids,allow_creep_exception=True)
 assert r['passed'] and r['rows'][0]['creep_exception_applied']
 assert not r['rows'][0]['raw_speed_under10percent']
 assert not r['rows'][0]['replicas'][0]['speed_under10percent']
 f[:,1,1]=.057
 assert not audit(report,f,ids,allow_creep_exception=True)['passed']
 f[:,:,1]=.0555;row['completed_episodes']=4
 assert not audit(report,f,ids,allow_creep_exception=True)['passed']
 row['completed_episodes']=32;f[:,0,2]=np.deg2rad(2.1)
 assert not audit(report,f,ids,allow_creep_exception=True)['passed']

def test_creep_exception_does_not_relax_other_speeds_or_reverse():
 for v in [.049,.051,.1,-.05]:
  f=np.zeros((20,1,8));f[:,:,0]=1300;f[:,:,1]=v+.11*abs(v)
  row=dict(vx=v,yaw=0,linear_error=.11*abs(v),survival=1,completed_episodes=32,termination_counts={'tilt':0})
  r=audit(dict(rows=[row]),f,np.array([0]),allow_creep_exception=True)
  assert not r['passed'] and not r['rows'][0]['creep_exception_applied']
