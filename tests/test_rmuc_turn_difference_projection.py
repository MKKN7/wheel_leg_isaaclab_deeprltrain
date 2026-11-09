import copy,sys
from pathlib import Path
import torch,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_physics_adaptation import PhysicsAdaptation
from rmuc_command_adapter import install_rmuc_physics_adaptation

def make():
    m=PhysicsAdaptation();m.install_joint_delay_network([.08,.05,.1,.08,.05,.1])
    m.install_joint_delay_scope([-2.,-1.8,-.1,0.,.01,.05]);m.install_joint_delay_reference()
    m.install_joint_turn_reference([.9,1.,1.5,1.6,-1.6,-1.4,-.6,-.45])
    m.install_joint_turn_source_update_scope([.4,.45,.55,.6,.15,.2,.4,.45])
    torch.nn.init.normal_(m.joint_turn_source_network[-1].weight,std=.3)
    return m

def test_projection_preserves_leg_common_and_bounds_under_large_delta():
    torch.manual_seed(11900);m=make();old=copy.deepcopy(m);m.install_joint_turn_source_difference_anchor()
    m.joint_turn_source_network[-1].bias.data.copy_(torch.tensor([5.,-4.,-8.,3.,7.,8.]))
    x=torch.randn(4096,25)*2;x[:,7]=.25;x[:,8]=.3-x[:,2]/.25
    a=old.joint_turn_source_output(x);b=m.joint_turn_source_output(x)
    assert torch.equal(a[:,[0,1,3,4]],b[:,[0,1,3,4]])
    torch.testing.assert_close(a[:,2]+a[:,5],b[:,2]+b[:,5],atol=3e-7,rtol=0)
    assert b.abs().max()<=1.000001
    assert (b[:,2]-a[:,2]).abs().max()>.1
    outside=x.clone();outside[:,7]=.5
    assert torch.equal(old.correction(outside),m.correction(outside))
    inside=torch.zeros(32,25);inside[:,7]=.25;inside[:,8]=.3
    assert (old.correction(inside)-m.correction(inside)).abs().max()>0

def test_projection_checkpoint_roundtrip_and_missing_flag_rejected():
    m=make();m.install_joint_turn_source_difference_anchor();m.joint_turn_source_network[-1].bias.data[5]=.2
    state={'rmuc_physics_adaptation.'+k:v for k,v in m.state_dict().items()}
    a=torch.nn.Module();a.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));a.rmuc_position_authority=torch.nn.Identity()
    restored=install_rmuc_physics_adaptation(a,state);restored.load_state_dict(m.state_dict(),strict=True)
    x=torch.randn(256,25);assert torch.equal(m.correction(x),restored.correction(x))
    del state['rmuc_physics_adaptation.joint_turn_source_difference_only']
    b=torch.nn.Module();b.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));b.rmuc_position_authority=torch.nn.Identity()
    with pytest.raises(ValueError):install_rmuc_physics_adaptation(b,state)

def test_different_wheel_budgets_rejected():
    m=make();m.joint_delay_budget[5]=.11
    with pytest.raises(ValueError):m.install_joint_turn_source_difference_anchor()

def test_leg_difference_projection_restores_common_wheel_and_keeps_leg_delta():
    torch.manual_seed(11901);m=make();anchor=copy.deepcopy(m);m.install_joint_turn_source_leg_difference_anchor()
    m.joint_turn_source_network[-1].bias.data.copy_(torch.tensor([.5,-.4,-.8,.3,.7,.8]))
    x=torch.randn(4096,25)*2;x[:,7]=.25;x[:,8]=.3-x[:,2]/.25
    out=m.joint_turn_source_output(x);raw=torch.tanh(m.joint_turn_source_network(x));base=torch.tanh(m.joint_turn_source_leg_difference_anchor(x))
    assert torch.equal(out[:,[0,1,3,4]],raw[:,[0,1,3,4]])
    torch.testing.assert_close(out[:,2]+out[:,5],base[:,2]+base[:,5],atol=3e-7,rtol=0)
    # The desired differential is preserved whenever the fixed common mode
    # admits it inside the residual envelope; saturated requests stay bounded.
    common=(base[:,2]+base[:,5])*.5
    desired=(raw[:,5]-raw[:,2])*.5
    interior=(common.abs()+desired.abs())<.99999
    assert interior.sum()>100
    torch.testing.assert_close(out[interior,5]-out[interior,2],raw[interior,5]-raw[interior,2],atol=3e-7,rtol=0)
    assert out.abs().max()<=1.000001
    outside=x.clone();outside[:,7]=.5;assert torch.equal(anchor.correction(outside),m.correction(outside))

def test_leg_difference_install_is_identity_and_roundtrip_is_exact():
    m=make();old=copy.deepcopy(m);m.install_joint_turn_source_leg_difference_anchor()
    x=torch.randn(512,25);x[:,7]=.25;x[:,8]=.3-x[:,2]/.25
    assert torch.equal(old.correction(x),m.correction(x))
    m.joint_turn_source_network[-1].bias.data.add_(.1)
    state={'rmuc_physics_adaptation.'+k:v for k,v in m.state_dict().items()}
    a=torch.nn.Module();a.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));a.rmuc_position_authority=torch.nn.Identity()
    restored=install_rmuc_physics_adaptation(a,state);restored.load_state_dict(m.state_dict(),strict=True)
    assert torch.equal(m.correction(x),restored.correction(x))
    with pytest.raises(ValueError):m.install_joint_turn_source_difference_anchor()
    del state['rmuc_physics_adaptation.joint_turn_source_leg_difference_only']
    b=torch.nn.Module();b.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));b.rmuc_position_authority=torch.nn.Identity()
    with pytest.raises(ValueError):install_rmuc_physics_adaptation(b,state)
