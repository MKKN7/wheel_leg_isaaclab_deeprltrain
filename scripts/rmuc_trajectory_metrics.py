"""Read-only metrics for EXISTING body vx+yaw commands; no new command channel."""
import numpy as np


def arc_reference(time_s, vx, yaw_rate, initial_xy=(0.,0.), initial_heading=0.):
    t=np.asarray(time_s,dtype=float)
    if t.ndim!=1 or len(t)<1 or not np.isfinite(t).all() or not np.isfinite([vx,yaw_rate,initial_heading,*initial_xy]).all():
        raise ValueError('Finite time and existing vx/yaw required')
    t=t-t[0];a=initial_heading+yaw_rate*t
    if abs(yaw_rate)<1e-10:
        xy=np.asarray(initial_xy)+t[:,None]*vx*np.array([np.cos(initial_heading),np.sin(initial_heading)])
    else:
        xy=np.asarray(initial_xy)+(vx/yaw_rate)*np.column_stack((np.sin(a)-np.sin(initial_heading),np.cos(initial_heading)-np.cos(a)))
    return xy,np.column_stack((vx*np.cos(a),vx*np.sin(a)))


def arc_metrics(time_s, position_xy, vx, yaw_rate, *, initial_xy, initial_heading):
    t=np.asarray(time_s,dtype=float);p=np.asarray(position_xy,dtype=float)
    if len(t)<2 or p.shape!=(len(t),2) or (np.diff(t)<=0).any() or not np.isfinite(p).all():
        raise ValueError('Aligned finite trajectory with increasing time required')
    ref,velocity=arc_reference(t,vx,yaw_rate,initial_xy,initial_heading)
    # Geometric distance to reference line/circle; time-aligned error separately.
    if abs(vx)<1e-10:
        cross=np.linalg.norm(p-np.asarray(initial_xy),axis=1)
    elif abs(yaw_rate)<1e-10:
        normal=np.array([-np.sin(initial_heading),np.cos(initial_heading)])
        cross=np.abs((p-np.asarray(initial_xy))@normal)
    else:
        center=np.asarray(initial_xy)+(vx/yaw_rate)*np.array([-np.sin(initial_heading),np.cos(initial_heading)])
        cross=np.abs(np.linalg.norm(p-center,axis=1)-abs(vx/yaw_rate))
    return dict(duration_s=float(t[-1]-t[0]),cross_track_max_m=float(cross.max()),
                cross_track_rms_m=float(np.mean(cross**2)**.5),
                time_aligned_error_max_m=float(np.linalg.norm(p-ref,axis=1).max()),
                reference_radius_m=abs(vx/yaw_rate) if abs(yaw_rate)>1e-10 and abs(vx)>1e-10 else None,
                geometric_gate_passed=bool(cross.max()<.1),
                full_acceptance=False,limitations=['Requires survival, episode integrity and dynamic/retention gates; geometry alone is insufficient'])


def step_metrics(time_s, measured, before, target, *, hold_s=.2, zero_tolerance=.05):
    t=np.asarray(time_s,dtype=float);v=np.asarray(measured,dtype=float)
    if t.ndim!=1 or v.shape!=t.shape or len(t)<3 or (np.diff(t)<=0).any() or not np.isfinite(t).all() or not np.isfinite(v).all() or not np.isfinite([before,target]).all():
        raise ValueError('Finite increasing time and aligned scalar speed required')
    delta=target-before
    if abs(delta)<1e-9:raise ValueError('Nonzero step required')
    overshoot=max(0.,float(((v-target)*np.sign(delta)).max()))/abs(delta)
    band=max(.1*abs(target),zero_tolerance);good=np.abs(v-target)<=band;arrival=None
    for i in range(len(t)):
        j=np.searchsorted(t,t[i]+hold_s)
        if j<len(t) and good[i:j+1].all():arrival=float(t[i]-t[0]);break
    steady=t>=t[0]+.5*(t[-1]-t[0]);mae=float(np.abs(v[steady]-target).mean())
    steady_ok=mae<(.1*abs(target) if abs(target)>=zero_tolerance else zero_tolerance)
    return dict(steady_mae_mps=mae,overshoot_fraction=overshoot,arrival_time_s=arrival,
                passed=steady_ok and overshoot<.2 and arrival is not None and arrival<.5,
                arrival_band_mps=band,hold_s=hold_s,
                definition='Fixed commanded step; first in-band interval held .2s. Zero target absolute tolerance .05m/s provisional.')
