"""Opt-in high-straight alive baseline; never reward unsafe termination."""
import math
import torch


def high_straight_alive_baseline(command, unsafe_termination, amount):
    if not math.isfinite(amount) or not 0 <= amount <= 25:
        raise ValueError('High-straight alive baseline must be within [0,25]')
    active = ((command[..., 0] >= 2.1) & (command[..., 2].abs() <= 1e-4)
              & ~unsafe_termination.bool())
    return active.to(command.dtype) * amount


def fast_reverse_alive_baseline(command, unsafe_termination, amount):
    """Training-only survival priority for straight reverse vx<=-.95.

    No bonus on unsafe termination, forward, turning or protected slow skills.
    Zero preserves all earlier reward contracts.
    """
    if not math.isfinite(amount) or not 0<=amount<=25:
        raise ValueError('Fast reverse alive baseline must be within [0,25]')
    active=((command[...,0]<=-.95)&(command[...,0]>=-1.8)
            &(command[...,2].abs()<=1e-4)&~unsafe_termination.bool())
    return active.to(command.dtype)*amount
