"""Continuous command mapping shared by keyboard and fixed-grid inference."""
from __future__ import annotations

import torch
from collections.abc import Mapping


def validate_leg_scale_profile(profile):
    """Explicit absolute-speed action mapping shared by all command cohorts."""
    import math
    if not isinstance(profile, Mapping) or set(profile) != {'schema_version','speed_abs_mps','scale_rad'} or profile['schema_version'] != 1:
        raise ValueError('Explicit schema1 leg-scale profile required')
    speeds, scales = profile['speed_abs_mps'], profile['scale_rad']
    if (len(speeds) < 2 or len(speeds) != len(scales)
            or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in [*speeds,*scales])
            or speeds[0] != 0 or speeds[-1] > 3.2
            or any(b <= a for a,b in zip(speeds,speeds[1:]))
            or any(not 0 < v <= 1 for v in scales)):
        raise ValueError('Ordered nonnegative speed knots and positive leg scales required')
    return list(speeds), list(scales)


def sample_speed_extension_commands(count, bounds, *, device, anchors=(), anchor_fraction=0.0):
    """Explicit signed vx/yaw rectangle for an isolated speed-extension course."""
    import math
    if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds):
        raise ValueError('Four finite speed/yaw bounds required')
    v0,v1,w0,w1 = bounds
    if not (1.0 <= v0 < v1 <= 1.5 and -.0 < abs(w0) <= 1.8
            and abs(w1) <= 1.8 and w0 < w1 and w0*w1 > 0):
        raise ValueError('Speed-extension box must be signed, vx[1,1.5], yaw within +/-1.8')
    values=torch.zeros(count,3,device=device)
    values[:,0]=v0+(v1-v0)*torch.rand(count,device=device)
    values[:,2]=w0+(w1-w0)*torch.rand(count,device=device)
    if not math.isfinite(anchor_fraction) or not 0 <= anchor_fraction <= .75:
        raise ValueError('Anchor fraction must be finite and within [0,.75]')
    if anchor_fraction:
        if not anchors or any(len(a)!=2 or not all(math.isfinite(v) for v in a)
                              or not 1.0 <= a[0] <= 1.5 or not 0 < abs(a[1]) <= 1.8
                              or a[1]*w0 <= 0 for a in anchors):
            raise ValueError('Finite signed speed/yaw anchors within extension support required')
        # Reset batches often contain one environment. Deterministic rounding
        # turns round(1*.5) into zero and silently eliminates anchor rehearsal.
        expected=count*anchor_fraction
        n=math.floor(expected)
        if expected>n:
            n+=int(torch.rand((),device=device).item()<expected-n)
        if n:
            table=torch.tensor(anchors,device=device,dtype=values.dtype)
            offset=torch.randint(len(anchors),(1,),device=device)
            ids=(torch.arange(n,device=device)+offset)%len(anchors)
            selected=torch.randperm(count,device=device)[:n]
            values[selected,0]=table[ids,0];values[selected,2]=table[ids,1]
    return values


def interpolate_leg_scale(speed: torch.Tensor, knots: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Interpolate the certified stage scales by absolute command speed."""
    value = speed.abs().clamp(float(knots[0]), float(knots[-1]))
    upper = torch.searchsorted(knots, value.contiguous()).clamp(1, knots.numel() - 1)
    lower = upper - 1
    alpha = (value - knots[lower]) / (knots[upper] - knots[lower])
    return scales[lower] + alpha * (scales[upper] - scales[lower])
