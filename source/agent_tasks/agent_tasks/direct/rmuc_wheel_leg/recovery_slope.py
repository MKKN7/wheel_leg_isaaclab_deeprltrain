"""Ground-frame geometry for recovery slopes; no gravity or actuator changes."""
import torch


def ground_frame(inclination_degrees, uphill_azimuth_degrees):
    """Return normal and wxyz quaternion mapping a flat frame onto the slope.

    Positive inclination rises along the specified world-XY azimuth. Inputs
    share a shape; negative inclination reverses the uphill direction.
    """
    if inclination_degrees.shape != uphill_azimuth_degrees.shape:
        raise ValueError('Slope inclination and azimuth shapes must match')
    if not (torch.isfinite(inclination_degrees).all()
            and torch.isfinite(uphill_azimuth_degrees).all()):
        raise ValueError('Slope angles must be finite')
    if torch.any(inclination_degrees.abs() >= 90):
        raise ValueError('Slope must be strictly below a vertical plane')
    theta=torch.deg2rad(inclination_degrees)
    phi=torch.deg2rad(uphill_azimuth_degrees)
    # Rotation about (sin(phi), -cos(phi), 0) tilts the +X direction uphill.
    normal=torch.stack((-theta.sin()*phi.cos(),-theta.sin()*phi.sin(),theta.cos()),-1)
    half=theta*.5
    quaternion=torch.stack((half.cos(),half.sin()*phi.sin(),
                            -half.sin()*phi.cos(),torch.zeros_like(half)),-1)
    return normal,quaternion


def ground_clearance(points, origin, normal):
    """Signed normal distance; callers broadcast origins/normals for body arrays."""
    return ((points-origin)*normal).sum(-1)


def plane_elevation(xy, origin, normal):
    """World Z of the plane at XY, for collider/reset consistency checks."""
    if torch.any(normal[...,2] <= 0):
        raise ValueError('Ground normal must point upwards')
    return origin[...,2]-((xy-origin[...,:2])*normal[...,:2]).sum(-1)/normal[...,2]
