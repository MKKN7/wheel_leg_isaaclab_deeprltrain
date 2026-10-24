"""Opt-in 26D leg-length command contract. Not wired into legacy25D runtime.

The appended command is a hip-pivot to wheel-axis mean planar distance in m.
It is not root height. Current approved geometry envelope must be supplied by
caller; legacy hardware limits do not establish simulation reachability.
"""
import numpy as np
REFERENCE_M = .34
NORMALIZATION_M = .10

def append_leg_length_command(observation25, target_m, *, low_m=.30, high_m=.36):
    obs = np.asarray(observation25, dtype=np.float32)
    target = np.asarray(target_m, dtype=np.float32)
    if obs.ndim < 1 or obs.shape[-1] != 25 or not np.isfinite(obs).all():
        raise ValueError('Finite legacy25D observation required')
    if not np.isfinite([low_m, high_m]).all() or not .16 <= low_m < high_m <= .36:
        raise ValueError('Explicit candidate length envelope within legacy mechanical bounds required')
    if not np.isfinite(target).all() or (target < low_m-1e-7).any() or (target > high_m+1e-7).any():
        raise ValueError('Target outside validated/candidate envelope; do not silently clip')
    target = np.broadcast_to(target, obs.shape[:-1])
    return np.concatenate((obs, ((target-REFERENCE_M)/NORMALIZATION_M)[...,None]), axis=-1)

def leg_length_tracking_cost(left_m, right_m, target_m, *, scale_m=.02):
    """Bounded-shape pseudo-Huber gradient, mean length preserves bank differential.

    Candidate reward term only, weight not set. Collision/contact/attitude and
    legacy tracking are independent; a low-cost pose alone never means success.
    """
    if not np.isfinite(scale_m) or scale_m <= 0:
        raise ValueError('Positive scale required')
    left,right,target=np.broadcast_arrays(*[np.asarray(x,dtype=float) for x in (left_m,right_m,target_m)])
    if not all(np.isfinite(x).all() for x in (left,right,target)):
        raise ValueError('Finite lengths required')
    error=(.5*(left+right)-target)/scale_m
    return np.hypot(1.,error)-1.
