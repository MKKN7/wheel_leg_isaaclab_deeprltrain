"""User-authorized practical stop gate, separate from historical precision gate."""
import numpy as np

def stop_metrics(t,v,old,dt,complete,terminated,leg_p95,yaw_tail,command_error):
    t=np.asarray(t);v=np.asarray(v)
    if not len(t) or not np.isfinite(v).all():return dict(passed=False,reason='Missing/nonfinite trajectory')
    limit=1.5 if abs(old)<=.5+1e-6 else 2.
    hold=max(1,int(np.ceil(.3/dt)));inside=np.abs(v)<=.05;arrival=None
    for i in range(len(v)-hold+1):
        if inside[i:i+hold].all():arrival=float(t[i]);break
    reverse=np.maximum(0,-np.sign(old)*v);tail=t>=t[-1]-.5+dt/2
    run=longest=0
    for flag in reverse>.05:
        run=run+1 if flag else 0;longest=max(longest,run)
    checks=dict(complete=bool(complete),no_termination=not terminated,command_matches=command_error<1e-5,
                arrival=arrival is not None and arrival<=limit,
                reverse_speed=float(reverse.max())<=.15,
                no_sustained_reverse=longest*dt<=.5,
                tail_speed=float(np.abs(v[tail]).mean())<=.03,
                leg_alignment=max(leg_p95)<=2.,yaw=yaw_tail<.05)
    return dict(passed=all(checks.values()),checks=checks,arrival_s=arrival,arrival_limit_s=limit,
                peak_reverse_mps=float(reverse.max()),reverse_distance_m=float(reverse.sum()*dt),longest_reverse_s=longest*dt,tail_speed_mae=float(np.abs(v[tail]).mean()))
