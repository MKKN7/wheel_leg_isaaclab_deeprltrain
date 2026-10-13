"""Explicit diagnostic command schedules; final targets remain unchanged."""
import math
import torch

def offset_leg_targets(target, offset, fraction, limits, margin):
    """Diagnostic joint-side offset after policy clipping; keep physical margins."""
    shifted = target + offset * fraction[:, None]
    return torch.maximum(torch.minimum(shifted, limits[..., 1]-margin), limits[..., 0]+margin)


class HeldLegReference:
    """Diagnostic only: latch clipped leg actions per episode, retain live wheels."""
    def __init__(self, count, device, start_s):
        if not math.isfinite(start_s) or start_s <= 0:
            raise ValueError('Positive finite latch time required')
        self.start_s = start_s
        self.valid = torch.zeros(count, dtype=torch.bool, device=device)
        self.previous_age = torch.zeros(count, device=device)
        self.reference = torch.zeros(count, 4, device=device)

    def apply(self, action, episode_age, dt):
        age = episode_age.to(torch.float32)
        reset = age < self.previous_age
        self.valid[reset] = False
        ready = age * dt >= self.start_s
        self.valid[~ready] = False
        capture = ready & ~self.valid
        legs = action[:, [0, 1, 3, 4]]
        self.reference[capture] = legs[capture].clamp(-1., 1.)
        self.valid[capture] = True
        out = action.clone()
        out[:, [0, 1, 3, 4]] = torch.where(self.valid[:, None], self.reference, legs)
        self.previous_age.copy_(age)
        return out


def diagnostic_bias_fraction(episode_age, dt, start_s, ramp_s):
    """Per-episode smoothstep; reset returns to zero when a delay is requested."""
    if not all(math.isfinite(x) for x in (dt, start_s, ramp_s)) or dt <= 0 or min(start_s, ramp_s) < 0:
        raise ValueError('Finite nonnegative diagnostic times and positive dt required')
    elapsed = episode_age.to(torch.float32) * dt - start_s
    if ramp_s == 0:
        return (elapsed >= 0).to(torch.float32)
    t = (elapsed / ramp_s).clamp(0., 1.)
    return t.square() * (3. - 2. * t)


def ramp_straight_commands(target, episode_age, dt, duration_s):
    if not math.isfinite(duration_s) or duration_s<=0 or not math.isfinite(dt) or dt<=0:
        raise ValueError('Positive finite ramp duration and control timestep required')
    if target.ndim!=2 or target.shape[1]!=3 or not torch.isfinite(target).all():
        raise ValueError('Finite [N,3] command targets required')
    if (target[:,0]<1.).any() or (target[:,1:]!=0).any():
        raise ValueError('Diagnostic acquisition ramp supports forward straight targets >=1m/s only')
    fraction=(episode_age.to(target.dtype)*dt/duration_s).clamp(0,1)
    command=target.clone()
    command[:,0]=1.+fraction*(target[:,0]-1.)
    return command
