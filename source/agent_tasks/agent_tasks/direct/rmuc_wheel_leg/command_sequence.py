"""Deterministic episode-time vx/yaw steps; no new observation channels."""
import math
import torch


def matched_filtered_target(previous, target, valid, alpha):
    """Target filter matching the existing measured-speed reward filter."""
    return torch.where(valid, previous + alpha * (target - previous), target)


def validate_sequence(rows, dt, episode_seconds):
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(episode_seconds) or episode_seconds <= 0:
        raise ValueError('Positive finite timing required')
    if not isinstance(rows,list) or len(rows)<2:
        raise ValueError('At least two [time_s,vx,yaw] entries required')
    result=[];last=-1
    for row in rows:
        if not isinstance(row,list) or len(row)!=3 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in row):
            raise ValueError('Finite numeric [time_s,vx,yaw] entries required')
        t,v,w=map(float,row);step=round(t/dt)
        if t<0 or t>=episode_seconds or abs(step*dt-t)>1e-7 or step<=last or abs(v)>3.2 or abs(w)>8:
            raise ValueError('Ordered policy-aligned times and bounded vx/yaw required')
        result.append([step,v,w]);last=step
    if result[0][0]!=0:raise ValueError('Sequence must start at zero')
    return result


def sequence_commands(step_table, episode_age):
    table=torch.as_tensor(step_table,device=episode_age.device,dtype=torch.float32)
    ids=torch.searchsorted(table[:,0].contiguous(),episode_age.to(torch.float32).contiguous(),right=True)-1
    ids=ids.clamp(0,len(step_table)-1)
    command=torch.zeros((*episode_age.shape,3),device=episode_age.device)
    command[...,0]=table[ids,1];command[...,2]=table[ids,2]
    return command


def dynamic_tracking_cost(command_vx, measured_vx, cohort_mask, scale_mps):
    """Smooth L1-like error for moving-to-stop states, bounded for safety.

    At scale .5, errors0.1..1.5 remain distinguishable. Static cohorts are
    exactly zero; this is a training reward, never an actuator outer loop.
    """
    if not math.isfinite(scale_mps) or scale_mps <= 0:
        raise ValueError('Positive finite dynamic tracking scale required')
    error = (command_vx - measured_vx) / scale_mps
    return ((1.0 + error.square()).sqrt() - 1.0).clamp_max(4.0) * cohort_mask.to(error.dtype)


def tracking_cohort_mask(command, dynamic_mask, include_fixed_reverse=False):
    """Opt-in dense tracking for fixed reverse without double-counting dynamics."""
    if not include_fixed_reverse:
        return dynamic_mask
    reverse_straight = (command[..., 0] < 0) & (command[..., 2].abs() <= 1e-4)
    return dynamic_mask | reverse_straight


def dynamic_stop_posture_cost(command, measured_vx, pitch_rad, pitch_rate, cohort_mask):
    """Bounded stop-only settling cost, gradually emphasized near zero speed.

    No force, command filtering or feedback is applied. Two-degree pitch dead
    band preserves the nominal standing lean; rate cost encourages settling.
    """
    active = cohort_mask & (command.abs().amax(dim=-1) <= 1.e-6)
    near_stop = 1.0 / (1.0 + (measured_vx / .2).square())
    angle = ((pitch_rad.abs() - math.radians(2)).clamp_min(0) / math.radians(10)).square()
    rate = (pitch_rate / .5).square()
    return (angle + .25 * rate).clamp_max(4.) * near_stop * active.to(measured_vx.dtype)
