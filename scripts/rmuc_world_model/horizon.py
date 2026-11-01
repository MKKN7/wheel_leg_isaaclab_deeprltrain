"""Reject a baseline-critic bootstrap while an external intervention is unfinished."""
import math

def require_complete_intervention(horizon,dt,duration):
    if isinstance(horizon,bool) or not isinstance(horizon,int) or horizon<1:
        raise ValueError('Positive integer horizon required')
    if not all(math.isfinite(x) and x>0 for x in (dt,duration)):
        raise ValueError('Positive finite timing required')
    if horizon*dt+1e-9<duration:
        raise ValueError('Imagination ends before external pulse: baseline-teacher value omits remaining intervention; extend horizon or model continuation explicitly')
