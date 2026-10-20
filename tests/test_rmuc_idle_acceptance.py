import copy
import numpy as np
from rmuc_idle_acceptance import audit

def fixture(count=32):
    dt=.008;n=1501;w=np.zeros((n,count,14));q=np.zeros((n,count,8))
    age=np.arange(n)[:,None];w[:,:,4]=age;w[:,:,9]=age+1;q[:,:,0]=age
    w[-1,:,11]=1
    report={'rows':[dict(vx=0,yaw=0,survival=1.,termination_counts={'tilt':0},completed_episodes=count)]}
    return report,w,q,np.zeros(count,dtype=int),dt

def test_still_complete_32_pass_but_short_or_missing_do_not():
    data=fixture();assert audit(*data)['passed']
    r,w,q,b,dt=data
    assert not audit(r,w[:1400],q[:1400],b,dt)['passed']
    assert not audit(*fixture(2))['passed']
    assert audit(*fixture(2))['short_screen_pass']

def test_max_drift_not_end_displacement_and_bad_replica_cannot_hide():
    r,w,q,b,dt=fixture();w[800:900,3,0]=.051
    a=audit(r,w,q,b,dt);assert not a['passed']
    bad=next(x for x in a['rows'][0]['windows'] if x['env']==3)
    assert bad['maximum_xy_drift_m']==.051
    r,w,q,b,dt=fixture();w[:,5,3]=.0101
    assert not audit(r,w,q,b,dt)['passed']

def test_exact_duration_and_missing_endpoint_fail_closed():
    r,w,q,b,dt=fixture()
    result=audit(r,w,q,b,dt)
    assert result['rows'][0]['windows'][0]['duration_s']==10.
    assert not audit(r,w[:-1],q[:-1],b,dt)['passed']

def test_joint_mirror_and_reset_integrity():
    r,w,q,b,dt=fixture();q[:,:,2]=.2;q[:,:,5]=.2;q[:,:,3]=.3;q[:,:,6]=-.3
    assert audit(r,w,q,b,dt)['passed']
    q[:,7,2]+=np.deg2rad(2.1);assert not audit(r,w,q,b,dt)['passed']
    r,w,q,b,dt=fixture();w[700,:,10]=1;w[701:,:,4]-=701;w[701:,:,9]-=701;q[701:,:,0]-=701
    assert not audit(r,w,q,b,dt)['passed']
