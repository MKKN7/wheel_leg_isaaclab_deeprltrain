"""Physical speed-error cost variants; acceptance tolerances are not altered."""
import torch

LEGACY='tolerance_normalized_v1'
PHYSICAL='bounded_physical_slope_v2'

def speed_cost(error,target,version=LEGACY):
    tolerance=torch.where(target.abs()<.025,.01,.1*target.abs()).clamp_min(.01)
    huber=torch.sqrt(1+(error/tolerance).square())-1
    if version==LEGACY:return huber
    if version==PHYSICAL:
        # Same asymptotic derivative 1/.05 per m/s for any command. The local
        # curvature still grows as tolerance shrinks, preserving fine idle error sensitivity.
        return huber*tolerance/.05
    raise ValueError('Unknown speed objective version')
