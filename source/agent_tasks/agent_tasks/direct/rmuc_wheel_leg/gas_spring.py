"""Constant generalized-torque approximation of the reference gas springs.

For the bundled USD, both active-link joints use parent-local +X with identity
localRot0. Apply child torque and equal opposite chassis reaction in world axes.
This is passive mechanical assistance, not an action or motor torque command.
"""
import torch


def firmware_spring_motor_compensation(leg_length_m: torch.Tensor,
                                       front_angle_rad: torch.Tensor,
                                       back_angle_rad: torch.Tensor) -> torch.Tensor:
    """Old controller's front/back motor compensation, in its signed motor frame.

    This is an observed firmware command law, not a calibrated passive spring
    wrench or an active-link generalized torque.
    """
    if leg_length_m.shape != front_angle_rad.shape or leg_length_m.shape != back_angle_rad.shape:
        raise ValueError("Firmware spring inputs must have matching shapes")
    if not (bool(torch.isfinite(leg_length_m).all())
            and bool(torch.isfinite(front_angle_rad).all())
            and bool(torch.isfinite(back_angle_rad).all())):
        raise ValueError("Firmware spring inputs must be finite")
    spring_tor = ((-83.121 * leg_length_m + 40.242) * 0.8).clamp(16.5, 27.0)
    front = -spring_tor * torch.sin(front_angle_rad - back_angle_rad)
    return torch.stack((front, -front), dim=-1)


def firmware_leg_length(front_angle_rad: torch.Tensor,
                        back_angle_rad: torch.Tensor) -> torch.Tensor:
    """Closed-link L0 used by the legacy controller's spring law."""
    if front_angle_rad.shape != back_angle_rad.shape:
        raise ValueError("Leg angle inputs must have matching shapes")
    l1 = 0.210
    l2 = 0.250
    l3 = 0.250
    l4 = 0.210
    l5 = 0.0
    xb = -l5 / 2.0 + l1 * torch.cos(front_angle_rad)
    yb = l1 * torch.sin(front_angle_rad)
    xd = l5 / 2.0 + l4 * torch.cos(back_angle_rad)
    yd = l4 * torch.sin(back_angle_rad)
    dx = xd - xb
    dy = yd - yb
    bd2 = dx.square() + dy.square()
    a = 2.0 * l2 * dx
    b = 2.0 * l2 * dy
    c = l2 * l2 + bd2 - l3 * l3
    discriminant = (a.square() + b.square() - c.square()).clamp_min(0.0)
    phi2 = 2.0 * torch.atan2(b + torch.sqrt(discriminant), a + c)
    xc = -l5 / 2.0 + l1 * torch.cos(front_angle_rad) + l2 * torch.cos(phi2)
    yc = l1 * torch.sin(front_angle_rad) + l2 * torch.sin(phi2)
    return torch.sqrt((xc.square() + yc.square()).clamp_min(1.0e-8))


def firmware_spring_torque_from_joint_positions(joint_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [right-front,right-back,left-front,left-back] torque and L0.

    Joint positions use the RMUC policy order. Offsets/signs are copied from
    the read-only firmware encoder mapping; the result is a motor compensation
    target, not a passive body wrench.
    """
    if joint_positions.shape[-1] != 4:
        raise ValueError("Expected four controlled leg joint positions")
    # USD q is already in physical joint radians.  Do not add firmware encoder
    # offsets: those offsets convert raw motor feedback into this frame and are
    # not part of the USD joint coordinate.
    signs = torch.tensor([1.0, 1.0, 1.0, -1.0], device=joint_positions.device, dtype=joint_positions.dtype)
    angles = joint_positions * signs
    right_l0 = firmware_leg_length(angles[..., 0], angles[..., 1])
    left_l0 = firmware_leg_length(angles[..., 2], angles[..., 3])
    right = firmware_spring_motor_compensation(right_l0, angles[..., 0], angles[..., 1])
    left = firmware_spring_motor_compensation(left_l0, angles[..., 2], angles[..., 3])
    return torch.stack((right[..., 0], right[..., 1], left[..., 0], left[..., 1]), dim=-1), torch.stack((right_l0, left_l0), dim=-1)


def spring_body_torques(parent_quaternion_w: torch.Tensor, magnitude_nm: float,
                        quaternion_order: str = "wxyz") -> torch.Tensor:
    """Return [chassis, left active link, right active link] world torques."""
    # Historical default reproduces older checkpoints; Isaac Lab 3 is xyzw.
    if quaternion_order == "xyzw":
        x, y, z, w = parent_quaternion_w.unbind(-1)
    elif quaternion_order == "wxyz":
        w, x, y, z = parent_quaternion_w.unbind(-1)
    else:
        raise ValueError("Gas spring quaternion_order must be xyzw or historical wxyz")
    axis = torch.stack((1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)), dim=-1)
    if isinstance(magnitude_nm,torch.Tensor) and magnitude_nm.ndim>0:
        if magnitude_nm.shape!=parent_quaternion_w.shape[:-1]+(2,):
            raise ValueError('Per-leg spring torque must have shape [...,2]')
        children=-magnitude_nm.unsqueeze(-1)*axis.unsqueeze(-2)
        return torch.cat((-children.sum(-2,keepdim=True),children),dim=-2)
    child = -float(magnitude_nm) * axis
    return torch.stack((-2*child, child, child), dim=-2)


def validate_reference_spring_joints(usd_path: str) -> None:
    """Fail closed if the fixed-axis approximation is applied to another asset."""
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise ValueError('Cannot read gas-spring joint geometry')
    for side in ('left', 'right'):
        prims = [p for p in stage.Traverse() if p.GetName() == f'{side}_active_link_pitch']
        if len(prims) != 1:
            raise ValueError('Gas-spring approximation requires unique active-link joints')
        joint = UsdPhysics.RevoluteJoint(prims[0])
        if not joint or joint.GetAxisAttr().Get() != 'X':
            raise ValueError('Gas-spring approximation requires parent-local X axes')
        if [p.name for p in joint.GetBody0Rel().GetTargets()] != ['robot'] or [p.name for p in joint.GetBody1Rel().GetTargets()] != [f'{side}_active_link']:
            raise ValueError('Gas-spring joint bodies differ from reference geometry')
        rotation = joint.GetLocalRot0Attr().Get()
        if abs(abs(float(rotation.GetReal())) - 1.) > 1e-6 or any(abs(float(x)) > 1e-6 for x in rotation.GetImaginary()):
            raise ValueError('Gas-spring parent joint frame differs from reference geometry')


def radial_spring_force_pair(hip_positions_w: torch.Tensor,
                             wheel_positions_w: torch.Tensor,
                             extension_force_n: torch.Tensor):
    """Return hip/wheel force pairs and application points in world coordinates.

    Positive force extends the virtual hip-to-wheel leg. Inputs have shapes
    [...,3], [...,3], [...]. The caller provides a force law or sampled force;
    this geometry helper does not assume a measured force or stroke curve.
    Force must be applied at these points, not at the bodies' centers of mass.
    """
    if hip_positions_w.shape != wheel_positions_w.shape or hip_positions_w.shape[-1] != 3:
        raise ValueError('Hip and wheel points must have matching [...,3] shapes')
    if extension_force_n.shape != hip_positions_w.shape[:-1]:
        raise ValueError('Radial force must have one value per leg')
    delta = wheel_positions_w - hip_positions_w
    length = torch.linalg.vector_norm(delta, dim=-1)
    if (not bool(torch.isfinite(delta).all()) or not bool(torch.isfinite(extension_force_n).all())
            or bool((length <= 1e-6).any()) or bool((extension_force_n < 0).any())):
        raise ValueError('Radial extension requires finite positive leg length and nonnegative force')
    wheel_force = delta / length.unsqueeze(-1) * extension_force_n.unsqueeze(-1)
    forces = torch.stack((-wheel_force, wheel_force), dim=-2)
    positions = torch.stack((hip_positions_w, wheel_positions_w), dim=-2)
    return forces, positions


def radial_attachment_points(usd_path: str):
    """Read virtual-leg endpoints from USD joint frames; do not edit the asset."""
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise ValueError('Cannot read radial spring attachment geometry')
    points = []
    for side in ('left', 'right'):
        joints = {}
        for name in (f'{side}_hip_pitch', f'{side}_wheel_spin'):
            prims = [p for p in stage.Traverse() if p.GetName() == name]
            if len(prims) != 1:
                raise ValueError('Radial approximation requires unique hip and wheel joints')
            joints[name] = UsdPhysics.Joint(prims[0])
        hip, wheel = joints[f'{side}_hip_pitch'], joints[f'{side}_wheel_spin']
        if [p.name for p in hip.GetBody0Rel().GetTargets()] != ['robot'] or [p.name for p in wheel.GetBody1Rel().GetTargets()] != [f'{side}_wheel']:
            raise ValueError('Radial approximation requires chassis hip pivot and wheel axle')
        points.append((tuple(hip.GetLocalPos0Attr().Get()), tuple(wheel.GetLocalPos1Attr().Get())))
    return points


def sample_equivalent_torques(count, nominal, strength, common_fraction, independent_fraction, *, device):
    """Additive common/independent deviations, total extent is their sum."""
    import math
    if not all(math.isfinite(float(v)) for v in (nominal,strength,common_fraction,independent_fraction)):
        raise ValueError('Nonfinite spring DR parameters')
    if nominal<0 or not 0<=strength<=1 or min(common_fraction,independent_fraction)<0 or common_fraction+independent_fraction>1:
        raise ValueError('Invalid spring DR extent')
    result=torch.full((count,2),float(nominal),device=device)
    if strength==0 or nominal==0 or common_fraction+independent_fraction==0:
        return result  # preserve RNG stream when disabled
    common=torch.empty(count,1,device=device).uniform_(-1,1)*common_fraction
    independent=torch.empty(count,2,device=device).uniform_(-1,1)*independent_fraction
    return result*(1.+strength*(common+independent))
