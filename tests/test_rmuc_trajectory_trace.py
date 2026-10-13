import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rmuc_trajectory_trace import analyze_trace


def straight(n=100,dt=.1):
 x=np.zeros((n,1,14));i=np.arange(n);x[:,0,0]=i*dt;x[:,0,5]=(i+1)*dt
 x[:,0,3]=x[:,0,8]=x[:,0,12]=1;x[:,0,4]=i;x[:,0,9]=i+1
 return x


def test_terminal_position_and_reset_never_joined():
 x=straight();x[49,0,10]=1;x[50:,0,[0,5]]+=100
 x[50:,0,4]-=50;x[50:,0,9]-=50
 r=analyze_trace(x,.1,minimum_duration_s=5)['rows']
 assert len(r)==2 and not r[0]['arc_segment_passed'] and r[1]['arc_segment_passed']
 assert r[1]['arc']['cross_track_max_m']==0
 assert all(a['step'] is None for a in r)


def test_geometry_alone_never_full_acceptance():
 r=analyze_trace(straight(),.1)
 assert r['rows'][0]['arc_segment_passed'] and not r['full_acceptance']
 x=straight();x[20,0,9]+=1
 assert not analyze_trace(x,.1)['rows'][0]['episode_integrity']


def test_transition_is_not_reset_step():
 x=straight(200);x[100:,0,12]=2;x[100:,0,8]=2;x[101:,0,3]=2
 rows=analyze_trace(x,.1)['rows']
 assert len(rows)==2 and rows[0]['step'] is None and rows[1]['step'] is not None
 assert rows[1]['step']['arrival_time_s']==.1
