"""Sensor-only no-slip root-reference speed in the legacy F/right/up axes.

Both angular and polar vectors follow the policy's determinant -1 map; physical
cross products therefore have the opposite sign to the coordinate cross.
This is an explicit kinematic estimator, not a ground-truth replacement under
wheel slip or loss of contact. No root velocity is an input.
"""
import torch

# Root COM minus the midpoint of hip pivots, from the training asset nominal
# mass frame. This is a reference POINT, not an identified real mass centre.
REFERENCE_FROM_HIP_FRU = (0.0055423, -0.0052237, 0.1277691)
ACTIVE_ZERO = 137.1081 * torch.pi / 180.
HIP_ZERO = 42.8966 * torch.pi / 180.


def leg_endpoint(phi_front, phi_back, dfront, dback):
    """Same 0.21/0.25 m, coincident-pivot branch as firmware getPhi()."""
    middle = .5 * (phi_front + phi_back)
    spread = .5 * (phi_front - phi_back)
    dm = .5 * (dfront + dback)
    dd = .5 * (dfront - dback)
    s, c = torch.sin(spread), torch.cos(spread)
    height = torch.sqrt((.25**2 - .21**2 * s.square()).clamp_min(1e-10))
    length = .21 * c + height
    dlength = (-.21 * s - .21**2 * s*c/height) * dd
    cm, sm = torch.cos(middle), torch.sin(middle)
    x, y = length*cm, length*sm
    dx, dy = dlength*cm-length*sm*dm, dlength*sm+length*cm*dm
    # Wheel stator is on the LOWER link D->C, not on virtual leg O->C.
    ex = x-.21*torch.cos(phi_back); ey = y-.21*torch.sin(phi_back)
    edx = dx+.21*torch.sin(phi_back)*dback
    edy = dy-.21*torch.cos(phi_back)*dback
    carrier_rate = (ex*edy-ey*edx)/(ex.square()+ey.square()).clamp_min(1e-10)
    return x, y, dx, dy, carrier_rate


def root_reference_velocity(leg_position, control_velocity, gravity, angular_velocity,
                            wheel_radius=.06, half_track=.2215,
                            reference_from_hip=REFERENCE_FROM_HIP_FRU):
    """Return heading speed and two independently reconstructed side speeds.

    leg_position is four policy joint offsets, control_velocity six policy
    rates. Inputs are sampled sensors; caller owns delay/noise/fusion.
    """
    offsets=leg_position.new_tensor([ACTIVE_ZERO,HIP_ZERO,ACTIVE_ZERO,HIP_ZERO])
    signs=leg_position.new_tensor([1.,1.,1.,-1.])
    phi=leg_position*signs+offsets
    rates=control_velocity[..., [0,1,3,4]]*signs
    rates=torch.where(rates.abs()<.022,torch.zeros_like(rates),rates)
    g=gravity / torch.linalg.vector_norm(gravity,dim=-1,keepdim=True).clamp_min(1e-8)
    heading=torch.zeros_like(g);heading[...,0]=1.
    heading=heading-g*g[...,0:1]
    heading=heading/torch.linalg.vector_norm(heading,dim=-1,keepdim=True).clamp_min(1e-5)
    # Circular-wheel contact radius lies perpendicular to its lateral axle.
    contact=g.clone();contact[...,1]=0.
    contact=wheel_radius*contact/torch.linalg.vector_norm(contact,dim=-1,keepdim=True).clamp_min(1e-5)
    reference=leg_position.new_tensor(reference_from_hip)
    result=[]
    for side,(i,j,wslot,lateral) in enumerate([(0,1,2,-half_track),(2,3,5,half_track)]):
        x,y,dx,dy,carrier=leg_endpoint(phi[...,i],phi[...,j],rates[...,i],rates[...,j])
        position=torch.stack((-x,torch.full_like(x,lateral),-y),dim=-1)-reference
        relative=torch.stack((-dx,torch.zeros_like(x),-dy),dim=-1)
        wheel_omega=angular_velocity.clone()
        wheel_omega[...,1]+=carrier+control_velocity[...,wslot]
        # v_C = cross(omega_wheel, contact); v_P = v_C + cross(omega,r)-dr.
        v=torch.cross(wheel_omega,contact,dim=-1)+torch.cross(angular_velocity,position,dim=-1)-relative
        result.append((v*heading).sum(-1))
    side_speeds=torch.stack(result,dim=-1)
    return side_speeds.mean(-1),side_speeds
