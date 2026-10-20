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


def step_metrics(time_s, measured, before, target, *, hold_s=.2, zero_tolerance=.01):
    t=np.asarray(time_s,dtype=float);v=np.asarray(measured,dtype=float)
    if t.ndim!=1 or v.shape!=t.shape or len(t)<3 or (np.diff(t)<=0).any() or not np.isfinite(t).all() or not np.isfinite(v).all() or not np.isfinite([before,target]).all():
        raise ValueError('Finite increasing time and aligned scalar speed required')
    delta=target-before
    if abs(delta)<1e-9:raise ValueError('Nonzero step required')
    if not np.isfinite([hold_s,zero_tolerance]).all() or hold_s<=0 or zero_tolerance<=0:
        raise ValueError('Positive finite arrival hold and zero-command tolerance required')
    overshoot=max(0.,float(((v-target)*np.sign(delta)).max()))/abs(delta)
    band=.1*abs(target) if abs(target)>1e-9 else zero_tolerance
    good=np.abs(v-target)<=band;arrival=None
    for i in range(len(t)):
        j=np.searchsorted(t,t[i]+hold_s)
        if j<len(t) and good[i:j+1].all():arrival=float(t[i]-t[0]);break
    steady=t>=t[0]+.5*(t[-1]-t[0]);mae=float(np.abs(v[steady]-target).mean())
    steady_ok=mae<band if abs(target)>1e-9 else mae<=band
    in_scope=abs(delta)<=1.5+1e-9
    arrival_limit=1.5 if abs(delta)<=.5+1e-9 else 2.0
    common=steady_ok and overshoot<.2 and arrival is not None
    return dict(steady_mae_mps=mae,overshoot_fraction=overshoot,arrival_time_s=arrival,
                passed=bool(common and in_scope and arrival<=arrival_limit),
                legacy_05s_passed=bool(common and arrival<.5),
                acceptance_profile='2026-10-15_relaxed_step_response',
                step_within_declared_scope=bool(in_scope),arrival_limit_s=arrival_limit if in_scope else None,
                arrival_band_mps=band,hold_s=hold_s,
                definition='Fixed commanded step; first in-band interval held for hold_s. Nonzero target10percent; zero target absolute tolerance, default user-confirmed.01m/s. Idle maximum drift and posture require separate trajectory gates.')


def smooth_tracking_metrics(time_s, measured, command):
    """Strict time-aligned tracking of a single monotonic ramp (not a step).

    Threshold crossing delays at10..90% are diagnostics; no shifting measured
    data to hide tracking error. Caller supplies the full ramp and its tail.
    """
    t=np.asarray(time_s,dtype=float);v=np.asarray(measured,dtype=float);c=np.asarray(command,dtype=float)
    if t.ndim!=1 or v.shape!=t.shape or c.shape!=t.shape or len(t)<3 or not all(np.isfinite(x).all() for x in (t,v,c)) or (np.diff(t)<=0).any():
        raise ValueError('Finite aligned ramp trace with increasing time required')
    delta=c[-1]-c[0]
    if abs(delta)<1e-9:raise ValueError('A single nonzero monotonic ramp required')
    sign=np.sign(delta);dc=np.diff(c)*sign
    if (dc < -1e-7).any() or (dc>1e-7).sum()<2:
        raise ValueError('Step or nonmonotonic command cannot use smooth ramp metrics')
    error=np.abs(v-c);zero=np.abs(c)<1e-9
    good=np.where(zero,error<=.01,error<.1*np.abs(c))
    delays=[]
    for fraction in np.linspace(.1,.9,9):
        level=c[0]+fraction*delta
        ci=np.flatnonzero((c-level)*sign>=0);vi=np.flatnonzero((v-level)*sign>=0)
        delays.append(None if not len(vi) else float(t[vi[0]]-t[ci[0]]))
    delay_ok=all(d is not None and d<.5 for d in delays)
    return dict(acceptance_profile='strict_smooth_tracking',tracking_mae_mps=float(error.mean()),
                tracking_max_error_mps=float(error.max()),tracking_in_band_fraction=float(good.mean()),
                crossing_delays_s=delays,response_limit_s=.5,
                speed_tracking_passed=bool(good.all()),response_passed=bool(delay_ok),
                passed=bool(good.all() and delay_ok),full_acceptance=False,
                limitations=['Single monotonic ramp; reject reset or truncated trajectories upstream',
                             'No time alignment adjustment; posture, survival and leg-angle gates remain separate'])
