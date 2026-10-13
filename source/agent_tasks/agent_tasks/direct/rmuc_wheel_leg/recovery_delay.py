"""Independent curriculum cap; preserve zero-delay samples and sampling RNG."""
import math


def communication_delay_probability(strength, override=-1.0):
    if not math.isfinite(strength) or not 0<=strength<=1:
        raise ValueError('DR strength must be finite within [0,1]')
    if not math.isfinite(override) or (override != -1.0 and not 0<=override<=1):
        raise ValueError('Delay probability override must be -1 or within [0,1]')
    # Nominal retention is always zero-delay, regardless of training exposure.
    return 0.0 if strength==0 else (strength if override==-1.0 else override)


def cap_communication_delay(sampled, maximum_steps):
    if type(maximum_steps) is not int or maximum_steps not in (-1,0,1,2,3):
        raise ValueError('Recovery communication cap must be -1 or 0..3 control steps')
    return sampled if maximum_steps == -1 else sampled.clamp(max=maximum_steps)
