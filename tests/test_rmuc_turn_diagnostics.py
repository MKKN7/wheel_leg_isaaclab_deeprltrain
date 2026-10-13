import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'scripts/rmuc_turn_diagnostics.py'
spec=importlib.util.spec_from_file_location('turn_diagnostics',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_pure_turn_wheel_speed_difference_is_not_slip_and_com_is_mass_weighted():
    root=torch.tensor([[0.,0.,.4]])
    vel=torch.tensor([[1.,0.,0.]])
    # Pure no-slip turn: inner/outer wheel speeds differ from root by yaw*half-track.
    wheels=torch.tensor([[[0.,.2,.06],[0.,-.2,.06]]])
    wheelvel=torch.tensor([[[.8,0.,0.],[1.2,0.,0.]]])
    bodies=torch.tensor([[[0.,0.,.4],[0.,0.,.2]]])
    masses=torch.tensor([[1.,3.]])
    force=torch.tensor([[[0.,0.,20.],[0.,0.,20.]]])
    result=m.turn_metrics(root,vel,vel,torch.ones(1),torch.ones(1),
        torch.tensor([[1.,0.]]),bodies,masses,wheels,wheelvel,
        torch.tensor([[.8,1.2]]),force,torch.zeros(1))
    row=dict(zip(m.COLUMNS,result[0].tolist()))
    assert abs(row['wheel_center_longitudinal_slip_mps'])<1e-7
    assert abs(row['root_based_wheel_mismatch_mps']-.2)<1e-6
    assert abs(row['true_com_height_m']-.25)<1e-6
    assert row['min_normal_load_fraction']==.5
    assert abs(row['quasistatic_support_margin_proxy_m']-(.2-.25/9.81))<1e-6
def test_airborne_values_remain_finite_and_are_identified_by_zero_normal_force():
    z=torch.zeros
    result=m.turn_metrics(z(1,3),z(1,3),z(1,3),z(1),z(1),
        torch.tensor([[1.,0.]]),z(1,2,3),torch.ones(1,2),z(1,2,3),z(1,2,3),
        torch.ones(1,2),z(1,2,3),z(1))
    assert torch.isfinite(result).all()
    assert result[0,m.COLUMNS.index('contact_normal_force_n')]==0

def test_world_bank_positive_inward_for_positive_heading_turn():
    # Independent matrix rotation fixture for heading+Y, body up tilted toward world-X.
    angles=torch.tensor([.1,-.1])
    root=torch.zeros(2,4);root[:,0]=angles
    def apply(q,vec):
        bank=q[:,0]
        # Authored -Z -> heading+Y; authored+Y -> tilted world up.
        result=torch.zeros_like(vec)
        result[:,1]=-vec[:,2]
        result[:,0]=-torch.sin(bank)*vec[:,1]+torch.cos(bank)*vec[:,0]
        result[:,2]=torch.cos(bank)*vec[:,1]+torch.sin(bank)*vec[:,0]
        return result
    torch.testing.assert_close(m.world_bank_angle(root,apply),angles)

def test_inward_bank_proxy_handles_reverse_and_does_not_invent_standstill_lean():
    bank=torch.tensor([5.,5.,5.,5.,5.])
    speed=torch.tensor([1.,-1.,1.,-1.,0.])
    yaw=torch.tensor([1.,1.,-1.,-1.,1.])
    torch.testing.assert_close(m.bank_aligned_with_turn_proxy(bank,speed,yaw),
                               torch.tensor([5.,-5.,-5.,5.,0.]))
