import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rmuc_trajectory_metrics import arc_reference,arc_metrics,step_metrics

def test_arc_geometry_phase_lag_not_lateral_error():
 t=np.linspace(0,10,1001);p,_=arc_reference(t,1,1)
 q=p[50:];m=arc_metrics(t[:len(q)],q,1,1,initial_xy=(0,0),initial_heading=0)
 assert m['cross_track_max_m']<1e-12 and m['time_aligned_error_max_m']>.4

def test_radial_offset_rejected():
 t=np.linspace(0,10,1001);p,_=arc_reference(t,1,1);center=np.array([0,1]);p=center+1.12*(p-center)
 assert not arc_metrics(t,p,1,1,initial_xy=(0,0),initial_heading=0)['geometric_gate_passed']

def test_rotated_reverse_straight():
 t=np.linspace(0,10,1001);p,v=arc_reference(t,-1,0,(2,3),np.pi/2)
 assert np.allclose(p[-1],[2,-7])
 assert arc_metrics(t,p,-1,0,initial_xy=(2,3),initial_heading=np.pi/2)['cross_track_max_m']<1e-12

def test_braking_and_transient_band_entry():
 t=np.arange(0,1.001,.001);v=np.maximum(0,1-t/.3)
 assert step_metrics(t,v,1,0)['passed']
 v=np.ones_like(t);v[100:120]=0
 assert step_metrics(t,v,1,0)['arrival_time_s'] is None

def test_overshoot_rejected():
 t=np.linspace(0,1,1001);v=np.ones_like(t);v[50]=1.21
 assert not step_metrics(t,v,0,1)['passed']

def test_low_speed_arrival_cannot_use_old_point05_absolute_floor():
 t=np.linspace(0,1,1001);v=np.full_like(t,.025)
 r=step_metrics(t,v,0,.05)
 assert r['arrival_band_mps']==pytest.approx(.005)
 assert r['arrival_time_s'] is None and not r['passed']
 v[:]=.05;assert step_metrics(t,v,0,.05)['passed']

def test_zero_target_uses_confirmed_point01_and_validates_parameters():
 t=np.linspace(0,1,1001);v=np.full_like(t,.02)
 r=step_metrics(t,v,.1,0)
 assert r['arrival_band_mps']==.01 and not r['passed']
 with pytest.raises(ValueError):step_metrics(t,v,.1,0,hold_s=0)
 with pytest.raises(ValueError):step_metrics(t,v,.1,0,zero_tolerance=float('nan'))


def test_relaxed_response_preserves_legacy_and_rejects_large_steps():
 t=np.arange(0,4,.008)
 v=np.minimum(t/1.2,1)*.5
 r=step_metrics(t,v,0,.5)
 assert r['passed'] and not r['legacy_05s_passed'] and r['arrival_limit_s']==1.5
 v=np.minimum(t/1.9,1)*1.5
 r=step_metrics(t,v,0,1.5)
 assert r['passed'] and r['arrival_limit_s']==2
 r=step_metrics(t,np.full_like(t,2.),0,2.)
 assert not r['passed'] and not r['step_within_declared_scope']


def test_relaxation_does_not_accept_large_reverse_stop_overshoot():
 t=np.arange(0,4,.008);v=np.zeros_like(t)
 v[t<.4]=.5;v[(t>=.4)&(t<.8)]=-.3
 r=step_metrics(t,v,.5,0)
 assert r['arrival_time_s']<1.5 and not r['passed']
 assert r['overshoot_fraction']>.2


def test_smooth_tracking_does_not_inherit_relaxed_step_limit():
 from rmuc_trajectory_metrics import smooth_tracking_metrics
 t=np.arange(0,4,.008);c=.25+.25*np.minimum(t/2,1)
 assert smooth_tracking_metrics(t,c,c)['passed']
 v=.25+.25*np.minimum(np.maximum(t-.8,0)/2,1)
 r=smooth_tracking_metrics(t,v,c)
 assert not r['passed'] and not r['response_passed'] and r['response_limit_s']==.5
 assert r['tracking_max_error_mps']>.09


def test_smooth_rejects_step_and_keeps_zero_absolute_error():
 from rmuc_trajectory_metrics import smooth_tracking_metrics
 t=np.arange(0,3,.008);c=np.where(t<1,0.,.5)
 with pytest.raises(ValueError):smooth_tracking_metrics(t,c,c)
 c=.5*(1-np.minimum(t,1));v=c.copy();v[t>1]=.02
 assert not smooth_tracking_metrics(t,v,c)['speed_tracking_passed']
