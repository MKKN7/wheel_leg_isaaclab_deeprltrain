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
