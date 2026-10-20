import runpy
from pathlib import Path
import numpy as np
f=runpy.run_path(str(Path(__file__).resolve().parents[1]/'artifacts/teacher_reward_audit_20261017/analyze_dynamic_events.py'))['event_metrics']

def metric(v,cmd,t,active=None,**kw):
    zeros=np.zeros_like(t)
    return f(t,v,cmd,zeros if active is None else active,zeros,start=0.,stop=2.,old=0.,target=.3,ramp_s=0.,dt=.008,**kw)

def test_step_overshoot_cannot_hide_behind_good_final_speed():
    t=np.arange(250)*.008;cmd=np.full_like(t,.3);v=cmd.copy();v[:30]=0;v[30:45]=.4
    r=metric(v,cmd,t)
    assert r['complete'] and r['tail_speed_mae']==0 and r['arrival_s']<1.5
    assert r['overshoot_fraction']>.2 and not r['step_speed_pass']

def test_transient_angles_recorded_without_counting_as_steady_symmetry():
    t=np.arange(250)*.008;cmd=np.full_like(t,.3);angle=np.zeros_like(t);angle[:10]=12
    r=metric(cmd,cmd,t,angle)
    assert r['transient_angle_absmax_deg']==[12.,0.] and r['tail_angle_p95_deg']==[0.,0.]
    assert r['steady_tail_pass']

def test_incomplete_event_and_command_mismatch_never_pass():
    t=np.arange(150)*.008;c=np.full_like(t,.3)
    assert not metric(c,c,t)['step_speed_pass']
    t=np.arange(250)*.008;c=np.full_like(t,.3)
    assert not metric(c,c+.1,t)['step_speed_pass']

def test_smooth_tracking_uses_ramp_command_not_final_target():
    t=np.arange(250)*.008;cmd=.3*np.clip(t/.8,0,1);z=np.zeros_like(t)
    r=f(t,cmd,cmd,z,z,start=0,stop=2,old=0,target=.3,ramp_s=.8,dt=.008)
    assert r['speed_mae']==0 and r['instantaneous_band_fraction']==1 and r['step_speed_pass'] is None
