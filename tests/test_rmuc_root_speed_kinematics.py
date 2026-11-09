from pathlib import Path
import runpy
import torch

M=runpy.run_path(str(Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/root_speed_kinematics.py'))


def test_stationary_legs_level_forward_rolling():
    q=torch.zeros(2,4);dq=torch.zeros(2,6);dq[:,[2,5]]=-5
    speed,sides=M['root_reference_velocity'](q,dq,torch.tensor([[0.,0.,-1.]]).repeat(2,1),torch.zeros(2,3))
    torch.testing.assert_close(speed,torch.full((2,),.3));torch.testing.assert_close(sides,torch.full((2,2),.3))


def test_leg_jacobian_and_carrier_rate_match_finite_difference():
    torch.manual_seed(42);q=torch.rand(300,2,dtype=torch.float64)*.6-.3+torch.tensor([M['ACTIVE_ZERO'],M['HIP_ZERO']]);dq=torch.randn_like(q)
    x,y,dx,dy,carrier=M['leg_endpoint'](q[:,0],q[:,1],dq[:,0],dq[:,1])
    eps=1e-6
    ends=[];phis=[]
    for sign in [-1,1]:
        qq=q+sign*eps*dq
        xx,yy,*_=M['leg_endpoint'](qq[:,0],qq[:,1],dq[:,0],dq[:,1]);ends.append(torch.stack([xx,yy],-1))
        phis.append(torch.atan2(yy-.21*torch.sin(qq[:,1]),xx-.21*torch.cos(qq[:,1])))
    torch.testing.assert_close((ends[1]-ends[0])/(2*eps),torch.stack([dx,dy],-1),atol=1e-8,rtol=1e-7)
    torch.testing.assert_close((phis[1]-phis[0])/(2*eps),carrier,atol=1e-8,rtol=1e-7)


def test_rigid_pitch_motion_about_fixed_reference_does_not_look_like_translation():
    # Level body, fixed legs, known wheel-centre velocity from rigid rotation
    # around the stationary reference. Add carrier body rotation to encoder.
    q=torch.zeros(1,4,dtype=torch.float64);dq=torch.zeros(1,6,dtype=torch.float64)
    q0=torch.tensor([M['ACTIVE_ZERO']],dtype=torch.float64);q1=torch.tensor([M['HIP_ZERO']],dtype=torch.float64)
    _,length,*_=M['leg_endpoint'](q0,q1,q0*0,q1*0)
    omega=torch.tensor([[0.,1.,0.]],dtype=torch.float64)
    wheel_centre_forward=length.item()+M['REFERENCE_FROM_HIP_FRU'][2]
    dq[:,[2,5]]=-wheel_centre_forward/.06-1.
    v,_=M['root_reference_velocity'](q,dq,torch.tensor([[0.,0.,-1.]],dtype=torch.float64),omega)
    torch.testing.assert_close(v,torch.zeros(1,dtype=torch.float64),atol=1e-10,rtol=0)
