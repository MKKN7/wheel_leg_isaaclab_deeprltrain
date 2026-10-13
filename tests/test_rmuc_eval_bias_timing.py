import runpy
from pathlib import Path
import torch
import pytest
f=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/rmuc_eval_commands.py'))['diagnostic_bias_fraction']

def test_bias_waits_then_ramps_without_overshoot_and_resets_per_environment():
    age=torch.tensor([0,100,200,300,400,500])
    torch.testing.assert_close(f(age,.01,2.,2.),torch.tensor([0.,0.,0.,.5,1.,1.]))
    age[4]=0
    assert f(age,.01,2.,2.)[4]==0

def test_zero_timing_reproduces_full_bias_and_delayed_step_is_supported():
    age=torch.tensor([0,1,100,200])
    assert torch.equal(f(age,.01,0.,0.),torch.ones(4))
    assert torch.equal(f(age,.01,1.,0.),torch.tensor([0.,0.,1.,1.]))

@pytest.mark.parametrize('dt,start,ramp',[(0.,0.,1.),(.01,-1.,1.),(.01,0.,float('nan'))])
def test_invalid_times_rejected(dt,start,ramp):
    with pytest.raises(ValueError):f(torch.tensor([0]),dt,start,ramp)

def test_held_legs_preserve_live_wheels_and_release_each_reset():
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/rmuc_eval_commands.py'))
    hold=module['HeldLegReference'](2,'cpu',2.)
    a=torch.tensor([[.1,.2,.3,1.4,.5,.6],[.1,.2,.3,.4,.5,.6]])
    torch.testing.assert_close(hold.apply(a,torch.tensor([0,0]),.01),a)
    captured=hold.apply(a,torch.tensor([200,200]),.01)
    assert captured[0,3]==1.
    changed=a+.2
    later=hold.apply(changed,torch.tensor([201,201]),.01)
    torch.testing.assert_close(later[:,[0,1,3,4]],captured[:,[0,1,3,4]])
    torch.testing.assert_close(later[:,[2,5]],changed[:,[2,5]])
    reset=hold.apply(changed,torch.tensor([0,202]),.01)
    torch.testing.assert_close(reset[0],changed[0])
    torch.testing.assert_close(reset[1,[0,1,3,4]],captured[1,[0,1,3,4]])

def test_target_offset_bypasses_policy_clip_but_keeps_joint_limits():
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/rmuc_eval_commands.py'))
    target=torch.full((2,4),.22);delta=torch.full_like(target,.1)
    limits=torch.tensor([-.5,.3]).expand(2,4,2)
    out=module['offset_leg_targets'](target,delta,torch.tensor([0.,1.]),limits,.02)
    torch.testing.assert_close(out[0],target[0])
    torch.testing.assert_close(out[1],torch.full((4,),.28))
    assert torch.equal(target,torch.full((2,4),.22))
