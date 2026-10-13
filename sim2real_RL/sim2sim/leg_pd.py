"""Joint-output-side IdealPD law; shared deployment reference, no hidden loops."""
import numpy as np

KP = 60.0  # Nm/rad, joint/output side
KD = 2.0   # Nm/(rad/s), joint/output side
EFFORT_LIMIT_NM = 40.0  # current simulation contract, not measured continuous rating
PD_PERIOD_S = .002


def leg_pd_torque(target_position, position, velocity, target_velocity=0., feedforward=0.):
    arrays = [np.asarray(x,dtype=np.float64) for x in
              [target_position,position,velocity,target_velocity,feedforward]]
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError('PD inputs must be finite')
    target,actual,speed,desired_speed,effort=arrays
    return np.clip(KP*(target-actual)+KD*(desired_speed-speed)+effort,
                   -EFFORT_LIMIT_NM,EFFORT_LIMIT_NM)
