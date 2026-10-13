"""Joint-side actuator uncertainty, interpolated from nominal physical parameters."""
import math
import torch

def sample_leg_parameters(velocity, armature, strength, velocity_range, armature_range):
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('DR strength must be finite and within [0,1]')
    for bounds in (velocity_range, armature_range):
        if len(bounds)!=2 or not all(math.isfinite(x) for x in bounds) or not 0 < bounds[0] <= bounds[1]:
            raise ValueError('Positive finite ordered joint-side ranges required')
    if velocity.shape != armature.shape:
        raise ValueError('Matching joint parameter shapes required')
    if strength == 0:
        return velocity.clone(), armature.clone()
    def sample(center, bounds):
        draw = bounds[0] + torch.rand_like(center)*(bounds[1]-bounds[0])
        return center + strength*(draw-center)
    return sample(velocity, velocity_range), sample(armature, armature_range)
