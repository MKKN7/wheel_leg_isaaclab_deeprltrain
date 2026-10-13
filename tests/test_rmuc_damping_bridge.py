"""Real environment bridge block plus installed IdealPD compute, CPU only."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace, MethodType
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]


def bridge_code():
    tree=ast.parse((ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RmucWheelLegEnv')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_apply_action')
    start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
               and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='anneal_target')
    return compile(ast.Module(body=fn.body[start:start+2],type_ignores=[]),'<actual bridge>','exec')


def method(path,cls_name,name):
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls_name)
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    for arg in fn.args.args:arg.annotation=None
    fn.returns=None;ns={'torch':torch}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    return ns[name]


def env(fraction,enabled=True):
    actuator=SimpleNamespace(damping=torch.full((3,2),.1),stiffness=torch.zeros(3,2),effort_limit=torch.full((3,2),3.))
    return SimpleNamespace(cfg=SimpleNamespace(wheel_damping_anneal_target=.0425 if enabled else -1.,
        curriculum_stage=1,wheel_damping_anneal_start_level=17,run_mode='sim_training'),
        _command_level=17,_wheel_actuator=actuator,_wheel_actuator_damping=.1,
        _domain_wheel_damping_scale=torch.tensor([.005/.0425,1.,.08/.0425]),
        _wheel_damping_anneal_progress=lambda:fraction,
        _last_wheel_damping_anneal_fraction=torch.zeros(3),_last_effective_wheel_damping=torch.zeros(3))


@pytest.mark.parametrize('fraction',[0.,.25,.5,1.])
def test_written_damping_matches_report_across_DR_and_partial_reset(fraction):
    e=env(fraction);before=torch.tensor([[3.,-3.]]).repeat(3,1)
    scope={'self':e,'torch':torch,'requested_wheel':before.clone()}
    code=bridge_code();exec(code,scope)
    expected=((1-fraction)*.1+fraction*.0425)*e._domain_wheel_damping_scale
    torch.testing.assert_close(e._wheel_actuator.damping,expected[:,None].expand(3,2))
    assert torch.equal(scope['requested_wheel'],before)
    torch.testing.assert_close(e._last_effective_wheel_damping,expected)
    e._wheel_actuator.damping[1]=.1 # reset-time DR may restore base before next action
    exec(code,scope);torch.testing.assert_close(e._wheel_actuator.damping,expected[:,None].expand(3,2))


def test_disabled_bridge_keeps_existing_actuator_and_effort_unchanged():
    e=env(1.,enabled=False);original=e._wheel_actuator.damping.clone();target=torch.randn(3,2)
    scope={'self':e,'torch':torch,'requested_wheel':target.clone()};exec(bridge_code(),scope)
    assert torch.equal(e._wheel_actuator.damping,original)
    assert torch.equal(scope['requested_wheel'],target)


def test_actual_IdealPD_saturation_uses_target_damping_not_clipped_feedforward():
    spec=importlib.util.find_spec('isaaclab')
    if spec is None:pytest.skip('Installed IsaacLab source required for actuator compute audit')
    source=Path(spec.origin).parent/'actuators'
    compute=method(source/'actuator_pd.py','IdealPDActuator','compute')
    clip=method(source/'actuator_base.py','ActuatorBase','_clip_effort')
    e=env(1.);scope={'self':e,'torch':torch,'requested_wheel':torch.zeros(3,2)}
    exec(bridge_code(),scope)
    actuator=e._wheel_actuator;actuator._clip_effort=MethodType(clip,actuator)
    speed=torch.tensor([[50.,-50.]]).repeat(3,1)
    for requested in [torch.zeros(3,2),torch.tensor([[3.,-3.]]).repeat(3,1),torch.tensor([[-3.,3.]]).repeat(3,1)]:
        action=SimpleNamespace(joint_positions=torch.zeros(3,2),joint_velocities=torch.zeros(3,2),joint_efforts=requested.clone())
        result=compute(actuator,action,torch.zeros(3,2),speed)
        expected=(requested-actuator.damping*speed).clamp(-3.,3.)
        torch.testing.assert_close(result.joint_efforts,expected)
        assert result.joint_efforts.abs().max()<=3.
    # Center damping at3m/s: correct3-.0425*50=.875; previous compensation
    # clipped before baseD yielded clamp(3-.1*50)=-2, the opposite sign.
    requested=torch.tensor([3.,-3.]);velocity=torch.tensor([50.,-50.])
    old=((requested+(.1-.0425)*velocity).clamp(-3.,3.)-.1*velocity).clamp(-3.,3.)
    new=(requested-.0425*velocity).clamp(-3.,3.)
    torch.testing.assert_close(old,torch.tensor([-2.,2.]))
    torch.testing.assert_close(new,torch.tensor([.875,-.875]))
