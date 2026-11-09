"""Reward-only temporal tracking-error variation; no observation filtering."""
import math
import torch

def tracking_error_variation(command, speed, filtered_command, filtered_speed,
                             ready, tolerance=.005, scale=.05):
    if not all(math.isfinite(v) for v in [tolerance,scale]) or tolerance<0 or scale<=0:
        raise ValueError('Finite nonnegative tolerance and positive scale required')
    # Same filter on target and measurement prevents charging perfect tracking
    # for a step or ramp merely because the target itself is changing.
    highpass_error=(command-speed)-(filtered_command-filtered_speed)
    excess=(highpass_error.abs()-tolerance).clamp_min(0)/scale
    return excess.square()/(1+excess.square())*ready.to(speed.dtype)
