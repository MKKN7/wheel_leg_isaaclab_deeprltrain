"""Opt-in pure DR samplers; no environment integration or hidden defaults.

Mechanical drag is not feedback damping. Total inertia is sampled independently
of mass, never as mass times a second uncertainty factor. Zero strength consumes
no random numbers. Bounds are engineering priors, not measured calibration.
"""
import math
import torch


def _strength(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('DR strength must be finite within [0,1]')
    return float(value)


def _template(template):
    if not isinstance(template, torch.Tensor) or not template.is_floating_point():
        raise ValueError('Floating tensor template required')
    if not bool(torch.isfinite(template).all()):
        raise ValueError('Template must be finite')


def _bounds(nominal, bounds):
    if isinstance(nominal, bool) or not isinstance(nominal, (int, float)) or not math.isfinite(nominal):
        raise ValueError('Nominal must be finite numeric')
    if len(bounds) != 2 or any(isinstance(x, bool) or not isinstance(x, (int,float)) or not math.isfinite(x) for x in bounds):
        raise ValueError('Two finite numeric bounds required')
    low, high = map(float, bounds)
    if not 0 <= low <= nominal <= high:
        raise ValueError('Nonnegative ordered bounds must include nominal')
    return float(nominal), low, high


def mechanical_viscous_bounds(strength, nominal=0.002, bounds=(0.001, 0.005)):
    strength = _strength(strength)
    nominal, low, high = _bounds(nominal, bounds)
    return nominal + strength*(low-nominal), nominal + strength*(high-nominal)


def sample_mechanical_viscous(template, strength, nominal=0.002, bounds=(0.001, 0.005), *, generator=None):
    """Return per-template SI b; caller chooses env/wheel shape and stores it."""
    _template(template)
    low, high = mechanical_viscous_bounds(strength, nominal, bounds)
    if low == high:
        return torch.full_like(template, low)
    return torch.empty_like(template).uniform_(low, high, generator=generator)


def sample_mass_and_total_inertia_scales(template, strength, mass_fraction=0.2, inertia_fraction=0.2, *, generator=None):
    """Two independent scales; apply mass=M0*m and inertia=I0*i directly."""
    _template(template); strength = _strength(strength)
    for fraction in (mass_fraction, inertia_fraction):
        if isinstance(fraction, bool) or not isinstance(fraction, (int,float)) or not math.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError('Mass and total inertia fractions must be finite within [0,1)')
    def sample(fraction):
        extent = strength*fraction
        if extent == 0:
            return torch.ones_like(template)
        return torch.empty_like(template).uniform_(1-extent,1+extent,generator=generator)
    return sample(mass_fraction), sample(inertia_fraction)
