"""Continuous training commands inside the currently opened curriculum envelope."""
import torch


def fixed_turn_sign_cohorts(num_envs, positive_fraction, seed, device):
    """Keep direction exposure stable in control steps across episode resets."""
    if num_envs <= 0 or not 0.0 <= positive_fraction <= 1.0:
        raise ValueError('Invalid fixed turn cohort size or fraction')
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    order = torch.randperm(num_envs, generator=generator)
    positive = torch.zeros(num_envs, dtype=torch.bool)
    positive[order[:round(num_envs * positive_fraction)]] = True
    return positive.to(device=device)


def _interpolate_inside(values, knots, limits):
    """Linear training envelope, zero outside the opened anchor interval."""
    if len(knots) == 1:
        return torch.where(values == knots[0], limits[0], 0.)
    hi = torch.searchsorted(knots, values.contiguous(), right=True).clamp(1, len(knots)-1)
    fraction = ((values-knots[hi-1])/(knots[hi]-knots[hi-1])).clamp(0., 1.)
    result = limits[hi-1] + fraction*(limits[hi]-limits[hi-1])
    return result*((values >= knots[0]) & (values <= knots[-1]))


def sample_opened_commands(count, stage_speeds, stage_yaws, level, *, device='cpu', generator=None,
                           final_turn_envelope=None, extended_start_level=None):
    """Sample vx uniformly and signed yaw under conservative adjacent bounds.

    Opened means eligible for training, not certified. No command is a gate
    certificate. Repeated speeds retain their largest opened yaw magnitude.
    """
    if count < 0 or len(stage_speeds) != len(stage_yaws) or not 0 <= level < len(stage_speeds):
        raise ValueError('Invalid command sampling dimensions or level')
    speeds=torch.as_tensor(stage_speeds[:level+1],device=device,dtype=torch.float32)
    yaws=torch.as_tensor(stage_yaws[:level+1],device=device,dtype=torch.float32).abs()
    if not torch.isfinite(speeds).all() or not torch.isfinite(yaws).all() or (speeds<0).any():
        raise ValueError('Opened forward command bounds must be finite and nonnegative')
    knots=torch.unique(speeds,sorted=True)
    vx=knots[0]+torch.rand(count,device=device,generator=generator)*(knots[-1]-knots[0])
    # A straight-only stage is not evidence that turns are forbidden there.
    # Interpolate conservatively between opened TURN anchors separately.
    turn_knots=torch.unique(speeds[yaws>0],sorted=True)
    if len(turn_knots)==0:
        bound=torch.zeros_like(vx)
    elif len(turn_knots)==1:
        bound=torch.where(vx==turn_knots[0],yaws[speeds==turn_knots[0]].max(),0.)
    else:
        limits=torch.stack([yaws[speeds==v].max() for v in turn_knots])
        hi=torch.searchsorted(turn_knots,vx,right=True).clamp(1,len(turn_knots)-1)
        bound=torch.minimum(limits[hi-1],limits[hi])
        bound=bound*((vx>=turn_knots[0])&(vx<=turn_knots[-1]))
    # Extension anchors supersede legacy small-yaw knots only BETWEEN opened
    # extension anchors. The final envelope caps exploration, never opens a
    # future skill by itself. Its yaw_min is a target-band label, not a ban on
    # straight driving or small corrections. Legacy levels consume identical RNG.
    if final_turn_envelope is not None and extended_start_level is not None:
        if not 0 <= extended_start_level < len(stage_speeds):
            raise ValueError('Extended sampling start must name an existing stage')
        target_speeds = torch.tensor([row['speed_mps'] for row in final_turn_envelope], device=device)
        target_limits = torch.tensor([row['yaw_max_rad_s'] for row in final_turn_envelope], device=device)
        if (len(target_speeds) < 2 or not torch.isfinite(target_speeds).all()
                or not torch.isfinite(target_limits).all() or not (target_limits > 0).all()
                or not (target_speeds[1:] > target_speeds[:-1]).all()):
            raise ValueError('Final turn envelope requires ordered finite speeds and positive yaw limits')
        if level >= extended_start_level:
            ext_speeds = speeds[extended_start_level:]
            ext_yaws = yaws[extended_start_level:]
            ext_knots = torch.unique(ext_speeds[ext_yaws > 0], sorted=True)
            if len(ext_knots):
                ext_limits = torch.stack([ext_yaws[ext_speeds == speed].max() for speed in ext_knots])
                opened = _interpolate_inside(vx, ext_knots, ext_limits)
                target = _interpolate_inside(vx, target_speeds, target_limits)
                bound = torch.maximum(bound, torch.minimum(opened, target))
    yaw=(2*torch.rand(count,device=device,generator=generator)-1)*bound
    return torch.stack((vx,torch.zeros_like(vx),yaw),dim=-1)
