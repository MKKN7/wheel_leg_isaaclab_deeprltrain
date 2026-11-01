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


def test_finite_pulse_ramps_out_and_resets():
    age=torch.tensor([0,1700,1710,1720,1730,1740,1750])
    torch.testing.assert_close(f(age,.01,17.,.2,17.4),torch.tensor([0.,0.,.5,1.,.5,0.,0.]),atol=2e-5,rtol=0)
    assert f(torch.tensor([1740]),.01,17.,0.,17.4).item()==0

@pytest.mark.parametrize('end',[17.,16.,float('nan'),17.1])
def test_invalid_pulse_rejected(end):
    with pytest.raises(ValueError):f(torch.tensor([0]),.01,17.,.1,end)


def test_cli_default_optional_end_is_not_in_required_time_validation():
    import ast
    from types import SimpleNamespace
    source=(Path(__file__).resolve().parents[1]/'scripts/evaluate_rmuc_grid.py').read_text()
    tree=ast.parse(source)
    # Execute actual top-level timing validation with normal no-probe defaults.
    start=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bias_cases' for t in n.targets))
    class Parser:
        def error(self,message):raise AssertionError(message)
    scope={'args':SimpleNamespace(action_bias_start_seconds=0.,action_bias_ramp_seconds=0.,action_bias_end_seconds=None,action_bias_sweep=None),'parser':Parser(),'math':__import__('math')}
    exec(compile(ast.Module(body=tree.body[start+1:start+4],type_ignores=[]),'<actual-cli-validation>','exec'),scope)


def test_residual_probe_respects_directional_margin_and_zero_gate():
    m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/rmuc_eval_commands.py'))
    r=torch.tensor([[.07,-.19],[0.,0.]])
    b=torch.tensor([.08,.20]);g=torch.tensor([1.,0.]);d=torch.tensor([[.04,-.04],[.04,-.04]])
    actual=m['bounded_residual_probe'](r,b,d,g)
    torch.testing.assert_close(actual,torch.tensor([[.01,-.01],[0.,0.]]))
    assert torch.equal(m['bounded_residual_probe'](r,b,torch.zeros_like(d),g),torch.zeros_like(d))
