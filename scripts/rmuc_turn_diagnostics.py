"""Read-only flat-ground turn diagnostics; never supplies policy observations or rewards."""
import torch
COLUMNS = (
    'root_heading_vx_mps', 'root_heading_vy_mps', 'root_heading_vy_abs_mps',
    'root_body_vy_abs_mps', 'body_yaw_rate_rad_s', 'world_vertical_rate_rad_s',
    'true_com_height_m', 'root_height_m', 'com_lateral_from_axle_m',
    'wheel_center_longitudinal_slip_mps', 'root_based_wheel_mismatch_mps',
    'wheel_center_lateral_speed_abs_mps', 'min_normal_load_fraction',
    'normal_force_horizontal_vertical_ratio', 'contact_normal_force_n',
    'steady_turn_lateral_accel_proxy_mps2', 'quasistatic_support_margin_proxy_m',
)
def world_bank_angle(root_quat, quat_apply):
    """Signed lean toward horizontal heading-left, independent of travel direction."""
    forward_local=torch.zeros_like(root_quat[:,:3]);forward_local[:,2]=-1.
    up_local=torch.zeros_like(forward_local);up_local[:,1]=1.
    forward=quat_apply(root_quat,forward_local)[:,:2]
    forward=forward/torch.linalg.vector_norm(forward,dim=-1,keepdim=True).clamp_min(1e-6)
    left=torch.stack((-forward[:,1],forward[:,0]),dim=-1)
    up=quat_apply(root_quat,up_local)
    return torch.atan2((up[:,:2]*left).sum(-1),up[:,2])

def bank_aligned_with_turn_proxy(bank_angle, heading_speed, world_yaw_rate):
    """Align bank to signed v*yaw; a steady-turn proxy, not measured acceleration."""
    return bank_angle * torch.sign(heading_speed * world_yaw_rate)

def turn_metrics(root_pos, root_vel, body_velocity, body_yaw_rate, world_vertical_rate,
                 forward_xy, body_com_pos, body_mass, wheel_pos, wheel_vel,
                 signed_wheel_surface_speed, wheel_force, ground_height):
    lateral=torch.stack((-forward_xy[:,1],forward_xy[:,0]),dim=-1)
    vx=(root_vel[:,:2]*forward_xy).sum(-1);vy=(root_vel[:,:2]*lateral).sum(-1)
    mass_sum=body_mass.sum(-1).clamp_min(1e-9)
    com=(body_com_pos*body_mass[...,None]).sum(-2)/mass_sum[:,None]
    height=com[:,2]-ground_height
    com_lateral=((com[:,:2]-wheel_pos[:,:,:2].mean(1))*lateral).sum(-1)
    wheel_forward=(wheel_vel[:,:,:2]*forward_xy[:,None,:]).sum(-1)
    wheel_lateral=(wheel_vel[:,:,:2]*lateral[:,None,:]).sum(-1)
    normal=wheel_force[:,:,2].clamp_min(0.)
    contact=normal>2.
    count=contact.sum(-1).clamp_min(1)
    slip=((signed_wheel_surface_speed-wheel_forward).abs()*contact).sum(-1)/count
    mismatch=(signed_wheel_surface_speed-vx[:,None]).abs().mean(-1)
    normal_sum=normal.sum(-1)
    load=normal.min(-1).values/normal_sum.clamp_min(1e-6)
    tangential=torch.linalg.vector_norm(wheel_force[:,:,:2],dim=-1).sum(-1)/normal_sum.clamp_min(1.)
    accel=vx*world_vertical_rate
    half_track=(((wheel_pos[:,1,:2]-wheel_pos[:,0,:2])*lateral).sum(-1)).abs()/2
    support=half_track-(com_lateral-height*accel/9.81).abs()
    return torch.stack((vx,vy,vy.abs(),body_velocity[:,1].abs(),body_yaw_rate,world_vertical_rate,
        height,root_pos[:,2]-ground_height,com_lateral,slip,mismatch,
        (wheel_lateral.abs()*contact).sum(-1)/count,load,tangential,normal_sum,accel,support),dim=-1)
