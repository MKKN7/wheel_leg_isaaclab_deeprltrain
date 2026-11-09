"""Opt-in observation-only speed-source diagnostic; no controller changes."""
from __future__ import annotations

import math
import torch


def select_policy_forward_velocity(source, root_heading_velocity, control_velocity, wheel_radius_m):
    """Select the scalar used before obs6 scaling, clipping, delay and noise.

    ``desktop2_proxy`` copies the local firmware's coincident-pivot symmetric
    linkage formula. Policy DOF order is activeR, hipR, wheelR, activeL, hipL,
    wheelL. Physical second hip has the opposite sign to policy slot4.
    Uses instantaneous physical joint rates: existing observation queues/noise
    remain outside this function. Does not emulate asynchronous CAN feedback.
    """
    if source == 'root_heading':
        return root_heading_velocity
    if source != 'desktop2_proxy':
        raise ValueError(f'Unknown diagnostic policy speed source: {source!r}')
    if control_velocity.shape[-1] != 6 or control_velocity.shape[:-1] != root_heading_velocity.shape:
        raise ValueError('Speed-source diagnostic requires aligned scalar root speed and six policy DOF rates')
    if not math.isfinite(wheel_radius_m) or wheel_radius_m <= 0:
        raise ValueError('Wheel radius must be finite and positive')
    # getLegData() zeros leg rates with abs(rate)<.022 before FK. Preserve
    # all recorded/raw observation channels; apply this only to the proxy.
    legs = control_velocity[..., [0, 1, 3, 4]]
    legs = torch.where(legs.abs() < .022, torch.zeros_like(legs), legs)
    left_leg_w = -.5 * (legs[..., 0] + legs[..., 1])
    right_leg_w = -.5 * (legs[..., 2] - legs[..., 3])
    return .5 * wheel_radius_m * (
        -control_velocity[..., 2] - control_velocity[..., 5] + left_leg_w + right_leg_w)
