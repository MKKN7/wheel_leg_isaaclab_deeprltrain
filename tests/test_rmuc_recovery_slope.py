import importlib.util
from pathlib import Path
import pytest
import torch

path=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/recovery_slope.py'
spec=importlib.util.spec_from_file_location('recovery_slope',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_flat_frame_and_height_match_existing_world_height():
    n,q=m.ground_frame(torch.zeros(4),torch.tensor([0.,90.,180.,270.]))
    assert torch.equal(n,torch.tensor([[0.,0.,1.]]).expand(4,-1))
    assert torch.equal(q,torch.tensor([[1.,0.,0.,0.]]).expand(4,-1))
    origin=torch.tensor([[4.,7.,2.]]).expand(4,-1)
    point=origin+torch.tensor([1.,2.,.28])
    torch.testing.assert_close(m.ground_clearance(point,origin,n),point[:,2]-origin[:,2])


def test_uphill_elevation_rotation_and_clearance_agree():
    angle=torch.tensor([10.,10.,10.,10.],dtype=torch.float64)
    phi=torch.tensor([0.,90.,180.,270.],dtype=torch.float64)
    n,q=m.ground_frame(angle,phi)
    # Quaternion rotating world +Z must produce the physical plane normal.
    v=q[:,1:];z=torch.tensor([0.,0.,1.],dtype=q.dtype).expand_as(v)
    rotated=z+2*torch.cross(v,torch.cross(v,z,dim=-1)+q[:,:1]*z,dim=-1)
    torch.testing.assert_close(rotated,n)
    xy=torch.stack((torch.deg2rad(phi).cos(),torch.deg2rad(phi).sin()),-1)
    origin=torch.zeros(4,3,dtype=q.dtype)
    height=m.plane_elevation(xy,origin,n)
    torch.testing.assert_close(height,torch.deg2rad(angle).tan())
    point=torch.cat((xy,height[:,None]),-1)
    torch.testing.assert_close(m.ground_clearance(point,origin,n),torch.zeros(4,dtype=q.dtype))
    torch.testing.assert_close(m.ground_clearance(point+.28*n,origin,n),torch.full((4,),.28,dtype=q.dtype))


@pytest.mark.parametrize('angle',[90.,-90.,float('nan')])
def test_invalid_slope_rejected(angle):
    with pytest.raises(ValueError):m.ground_frame(torch.tensor([angle]),torch.zeros(1))
