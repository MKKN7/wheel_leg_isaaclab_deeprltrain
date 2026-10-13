"""Recovery deployment phase and physics-step target slew semantics."""
import numpy as np


def phase_progress(up, height, ready, hold_seconds, min_height, upright_cos, required_hold):
    orientation=np.clip((np.clip(up,-1.,1.)+1.)/(1.+upright_cos),0.,1.)
    progress=.5*orientation
    progress=np.where(up>upright_cos,.5+.3*np.clip(height/min_height,0.,1.),progress)
    return np.clip(np.where(ready,.8+.2*np.clip(hold_seconds/required_hold,0.,1.),progress),0.,1.)


def slew_target(requested, previous, rate, physics_dt):
    return previous+np.clip(requested-previous,-rate*physics_dt,rate*physics_dt)
