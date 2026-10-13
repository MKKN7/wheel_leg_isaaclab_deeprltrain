import importlib.util
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gas_spring',ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/gas_spring.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_internal_reaction_and_virtual_work_under_random_orientation():
    torch.manual_seed(41)
    q=torch.randn(64,4,dtype=torch.float64);q=q/q.norm(dim=-1,keepdim=True)
    torque=m.spring_body_torques(q,10.775)
    torch.testing.assert_close(torque.sum(dim=1),torch.zeros(64,3,dtype=torch.float64),atol=1e-12,rtol=0)
    omega=torch.randn(64,3,3,dtype=torch.float64)
    axis=-torque[:,1]/10.775
    relative_power=-10.775*((omega[:,1]-omega[:,0])*axis).sum(-1)-10.775*((omega[:,2]-omega[:,0])*axis).sum(-1)
    torch.testing.assert_close((torque*omega).sum((1,2)),relative_power)
    assert torch.count_nonzero(m.spring_body_torques(q,0.))==0


def test_reference_joint_direction_and_asset_geometry():
    m.validate_reference_spring_joints(str(ROOT/'source/agent_world/agent_world/assets/usd_files/rmuc_wheel_leg/rmuc_robot_closed_loop.usd'))
    torque=m.spring_body_torques(torch.tensor([[1.,0.,0.,0.]]),10.775)
    torch.testing.assert_close(torque[0,:,0],torch.tensor([21.55,-10.775,-10.775]))
    assert torch.count_nonzero(torque[...,1:])==0


def test_radial_pair_conserves_force_moment_and_has_only_extension_power():
    torch.manual_seed(12)
    hip = torch.randn(64, 2, 3, dtype=torch.float64)
    displacement = torch.randn_like(hip)
    wheel = hip + displacement
    magnitude = torch.rand(64, 2, dtype=torch.float64) * 100
    force, points = m.radial_spring_force_pair(hip, wheel, magnitude)
    torch.testing.assert_close(force.sum(-2), torch.zeros_like(hip), atol=1e-12, rtol=0)
    moment = torch.cross(points, force, dim=-1).sum(-2)
    torch.testing.assert_close(moment, torch.zeros_like(hip), atol=1e-12, rtol=0)
    velocity = torch.randn_like(force)
    axis = displacement / displacement.norm(dim=-1,keepdim=True)
    extension_speed = ((velocity[...,1,:]-velocity[...,0,:])*axis).sum(-1)
    torch.testing.assert_close((force*velocity).sum((-1,-2)),magnitude*extension_speed)
    rotation = torch.randn_like(hip)
    tangential = torch.cross(rotation,displacement,dim=-1)
    torch.testing.assert_close((force[...,1,:]*tangential).sum(-1),torch.zeros_like(magnitude),atol=1e-12,rtol=0)


def test_radial_zero_force_and_invalid_geometry():
    import pytest
    hip=torch.zeros(2,3);wheel=torch.tensor([[0.,0.,-.3],[0.,0.,-.4]])
    forces,_=m.radial_spring_force_pair(hip,wheel,torch.zeros(2))
    assert torch.count_nonzero(forces)==0
    for positions,force in [(hip,torch.ones(2)),(wheel,torch.tensor([-1.,1.])),(wheel,torch.tensor([float('nan'),1.]))]:
        with pytest.raises(ValueError):m.radial_spring_force_pair(hip,positions,force)


def test_radial_attachment_points_match_actual_usd():
    points=m.radial_attachment_points(str(ROOT/'source/agent_world/agent_world/assets/usd_files/rmuc_wheel_leg/rmuc_robot_closed_loop.usd'))
    assert len(points)==2
    assert abs(points[0][0][0]-.1775)<1e-6
    assert abs(points[1][0][0]+.1775)<1e-6
    assert points[0][1]==points[1][1]==(0.,0.,0.)


def test_equivalent_torque_sampling_bounds_zero_rng_and_asymmetric_reaction():
    torch.manual_seed(349)
    before=torch.get_rng_state().clone()
    zero=m.sample_equivalent_torques(32,10.775,0.,.1,.1,device='cpu')
    assert torch.equal(before,torch.get_rng_state())
    torch.testing.assert_close(zero,torch.full((32,2),10.775))
    common=m.sample_equivalent_torques(128,10.775,1.,.2,0.,device='cpu')
    assert torch.equal(common[:,0],common[:,1])
    values=m.sample_equivalent_torques(4096,10.775,1.,.1,.1,device='cpu')
    assert values.min()>=8.62 and values.max()<=12.93
    assert not torch.equal(values[:,0],values[:,1])
    q=torch.randn(4096,4);q/=q.norm(dim=-1,keepdim=True)
    torques=m.spring_body_torques(q,values)
    torch.testing.assert_close(torques.sum(1),torch.zeros(4096,3),atol=2e-6,rtol=0)
    torch.testing.assert_close(torques[:,1:].norm(dim=-1),values,atol=1e-5,rtol=1e-6)


def test_xyzw_matches_installed_isaac_math_and_historical_wxyz_is_explicit():
    import pytest
    from isaaclab.utils.math import quat_apply,quat_from_euler_xyz
    q=quat_from_euler_xyz(torch.tensor([torch.pi/2,.2]),torch.tensor([0.,-.4]),torch.tensor([0.,.7]))
    axis=quat_apply(q,torch.tensor([[1.,0.,0.]]).expand(2,-1))
    correct=m.spring_body_torques(q,1.0775,quaternion_order='xyzw')
    torch.testing.assert_close(correct[:,1],-1.0775*axis)
    torch.testing.assert_close(correct.sum(1),torch.zeros(2,3),atol=1e-7,rtol=0)
    legacy=m.spring_body_torques(q,1.0775)
    assert torch.equal(legacy,m.spring_body_torques(q,1.0775,quaternion_order='wxyz'))
    assert not torch.allclose(correct,legacy)
    torch.testing.assert_close(correct[0,1],torch.tensor([-1.0775,0.,0.]))
    with pytest.raises(ValueError):m.spring_body_torques(q,1.,quaternion_order='unknown')
