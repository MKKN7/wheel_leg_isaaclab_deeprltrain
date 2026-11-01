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


def stop_tracking_scale_adjustment(command, measured_vx, cohort_mask, base_scale, stop_scale):
    """Replace only dynamic zero-command tracking scale; leave idle/reverse/turn unchanged."""
    if not math.isfinite(stop_scale) or stop_scale < 0:
        raise ValueError('Stop scale must be finite and nonnegative')
    if stop_scale == 0:
        return torch.zeros_like(measured_vx)
    active = cohort_mask & (command.abs().amax(dim=-1) <= 1.e-6)
    return (dynamic_tracking_cost(command[..., 0], measured_vx, active, stop_scale)
            - dynamic_tracking_cost(command[..., 0], measured_vx, active, base_scale))


def sequence_overshoot_cost(step_table, episode_age, command, measured_vx, cohort_mask):
    """Penalize crossing a step target, not the braking torque needed to stop.

    Fixed straight sequences only. Ten percent of step size is free; beyond
    that the cost has a linear slope up to four. Initial phase, static cohorts
    and nonmatching/yaw commands are exactly excluded. No controller change.
    """
    table = torch.as_tensor(step_table, device=episode_age.device, dtype=measured_vx.dtype)
    ids = torch.searchsorted(table[:, 0].contiguous(), episode_age.to(table.dtype).contiguous(), right=True) - 1
    ids = ids.clamp(0, len(step_table) - 1)
    target = table[ids, 1]
    previous = table[(ids - 1).clamp_min(0), 1]
    delta = target - previous
    active = (cohort_mask & (ids > 0) & (delta.abs() > 1.e-6)
              & (table[ids, 2].abs() <= 1.e-6)
              & (table[(ids - 1).clamp_min(0), 2].abs() <= 1.e-6)
              & ((command[:, 0] - target).abs() <= 1.e-5)
              & (command[:, 2].abs() <= 1.e-6))
    crossing = delta.sign() * (measured_vx - target) / delta.abs().clamp_min(.05)
    return torch.where(active, (crossing - .1).clamp(0., 4.), torch.zeros_like(measured_vx))


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


def reverse_runaway_cost(command, measured_vx, previous_vx, filtered_backward_accel,
                         history_valid, dt):
    """Reverse-straight reward only; never changes actions or observations.

    Penalize actual overspeed plus a 120ms warning when both raw and filtered
    velocities indicate increasing backward speed. Correct braking gets no
    acceleration surcharge. The extrapolation is shaping, not a plant model.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Positive policy dt required')
    active = (command[..., 0] < -1.e-6) & (command[..., 2].abs() <= 1.e-6)
    excess = (-measured_vx - command[..., 0].abs() - .02).clamp_min(0.)
    raw_backward_accel = (previous_vx - measured_vx) / dt
    acceleration = torch.minimum(raw_backward_accel, filtered_backward_accel).clamp(0., 3.)
    acceleration = torch.where(history_valid, acceleration, torch.zeros_like(acceleration))
    projected_excess = (-measured_vx + .12*acceleration - command[..., 0].abs() - .02).clamp_min(0.)
    severity = (excess / .2).clamp_max(4.)
    escalation = ((projected_excess - excess) / .2).clamp(0., 2.)
    return torch.where(active, severity + escalation, torch.zeros_like(severity))


def reverse_settled_tracking_cost(command, measured_vx, episode_age_steps,
                                  step_table, sequence_mask, policy_dt):
    """Opt-in reverse steady precision after2s; no extra entry acceleration reward.

    Reward-only phase labels are privileged training data, not actor inputs.
    The symmetric error cost also penalizes overspeed. Turn/idle/old reverse
    commands>=-.65 receive exactly zero additional cost.
    """
    if not math.isfinite(policy_dt) or policy_dt<=0:raise ValueError('Positive policy dt required')
    phase_age=episode_age_steps.to(measured_vx.dtype)*policy_dt
    if step_table is not None:
        table=torch.as_tensor(step_table,device=measured_vx.device,dtype=measured_vx.dtype)
        ids=(torch.searchsorted(table[:,0].contiguous(),episode_age_steps.to(table.dtype).contiguous(),right=True)-1).clamp(0,len(table)-1)
        phase_age=torch.where(sequence_mask,(episode_age_steps-table[ids,0])*policy_dt,phase_age)
    ramp=((phase_age-2.)/1.).clamp(0.,1.)
    active=(command[:,0]<-.65)&(command[:,2].abs()<=1.e-6)
    scale=(command[:,0].abs()*.1).clamp_min(.05)
    error=(measured_vx-command[:,0])/scale
    cost=error.square()/(1.+error.square())
    return torch.where(active,cost*ramp,torch.zeros_like(cost))
