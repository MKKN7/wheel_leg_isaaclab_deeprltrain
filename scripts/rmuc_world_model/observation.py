"""Differentiable noise-free25D contract. Delays/noise must be applied separately."""
import torch

def rotate(q,v,inverse=False):
    xyz=-q[...,1:] if inverse else q[...,1:]
    uv=torch.linalg.cross(xyz,v,dim=-1)
    return v+2*(q[...,:1]*uv+torch.linalg.cross(xyz,uv,dim=-1))

def canonical(q,v):
    body=rotate(q,v,inverse=True)
    return torch.stack([-body[...,2],body[...,0],body[...,1]],-1)

def observe(root,control_pos,control_vel,default_pos,previous_applied_action,command,command_scale,error_clip,yaw_clip):
    """tracking_error_with_target only. Inputs remain SI; quaternion orderwxyz."""
    q=root[...,3:7]
    gravity=torch.zeros_like(root[...,:3]);gravity[...,2]=-1
    gravity=canonical(q,gravity);omega=canonical(q,root[...,10:13])
    forward=torch.zeros_like(gravity);forward[...,2]=-1
    forward=rotate(q,forward)[...,:2];forward=forward/forward.norm(dim=-1,keepdim=True).clamp_min(1.e-6)
    vx=(root[...,7:9]*forward).sum(-1)
    commands=torch.stack([command[...,0]-vx,command[...,0],command[...,2]-omega[...,2]],-1)*command_scale
    cmds=torch.cat([commands[...,:2].clamp(-error_clip,error_clip),commands[...,2:].clamp(-yaw_clip,yaw_clip)],-1)
    legs=[0,1,3,4]
    return torch.cat([omega*.25,gravity,cmds,control_pos[...,legs]-default_pos[...,legs],control_vel*.05,previous_applied_action],-1)


def native_root_to_wxyz(root):
    """Isaac native root_state_w tensor uses xyzw; actor math uses wxyz.

    Do not apply to root_quat_w accessor, which is already wxyz.
    """
    return torch.cat([root[...,:3],root[...,6:7],root[...,3:6],root[...,7:]],-1)
