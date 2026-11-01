import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/motor_response.py'
spec=importlib.util.spec_from_file_location('motor_response',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_step_matches_exact_first_order_response_at_physics_rate():
    r=m.EffortResponse(torch.zeros(1,2),.002,.01,.01);r.time_constant.fill_(.01)
    request=torch.tensor([[3.,-3.]])
    for i in range(1,31):
        actual=r.step(request,torch.full_like(request,3.))
        torch.testing.assert_close(actual,request*(1-math.exp(-i*.002/.01)),atol=1e-6,rtol=1e-6)
    assert 0<r.step(torch.zeros_like(request),torch.full_like(request,3.))[0,0]<3


def test_zero_strength_is_bitwise_identity_and_does_not_consume_rng():
    r=m.EffortResponse(torch.zeros(8,6),.002,.002,.03)
    state=torch.random.get_rng_state().clone();r.sample(torch.arange(8),0.)
    assert torch.equal(state,torch.random.get_rng_state())
    target=torch.randn(8,6);r.effort.uniform_(-3,3)
    assert torch.equal(r.step(target,torch.full_like(target,3.)),target.clamp(-3,3))


def test_independent_sampling_partial_reset_and_reduced_effort_limit():
    r=m.EffortResponse(torch.zeros(1000,6),.002,.002,.03)
    r.sample(torch.arange(1000),1.)
    assert .002<=r.time_constant.min() and r.time_constant.max()<=.03
    assert not torch.equal(r.time_constant[:,0],r.time_constant[:,1])
    before=r.time_constant.clone();r.sample(torch.tensor([3]),.05)
    assert torch.equal(r.time_constant[:3],before[:3]) and torch.equal(r.time_constant[4:],before[4:])
    assert r.time_constant[3].max()<=.0015
    r.effort.fill_(5.5);r.reset(torch.tensor([3]))
    assert r.effort[3].count_nonzero()==0 and (r.effort[4]==5.5).all()
    output=r.step(torch.full_like(r.effort,3.),torch.full_like(r.effort,3.))
    assert output.abs().max()<=3. # shrinking migration limits cannot leak old effort


def test_actuator_wrapper_filters_applied_effort_and_preserves_reset_scope():
    actuator=SimpleNamespace(damping=torch.zeros(2,2),effort_limit=torch.full((2,2),3.))
    resets=[]
    def compute(action,joint_pos,joint_vel):
        actuator.computed_effort=action.joint_efforts.clone()
        actuator.applied_effort=action.joint_efforts.clamp(-3,3)
        action.joint_efforts=actuator.applied_effort
        return action
    actuator.compute=compute;actuator.reset=lambda ids:resets.append(ids.clone())
    response=m.install_effort_response(actuator,.002,.01,.01)
    response.sample(torch.arange(2),1.)
    action=actuator.compute(SimpleNamespace(joint_efforts=torch.full((2,2),6.)),None,None)
    torch.testing.assert_close(action.joint_efforts,torch.full((2,2),3*(1-math.exp(-.2))))
    assert torch.equal(action.joint_efforts,actuator.applied_effort)
    actuator.reset(torch.tensor([0]));assert response.effort[0].count_nonzero()==0
    assert response.effort[1].count_nonzero()==2 and len(resets)==1
    with pytest.raises(ValueError,match='already'):m.install_effort_response(actuator,.002,.01,.01)


def test_motor_diagnostic_cannot_become_a_policy_certificate():
    tree=ast.parse((ROOT/'scripts/evaluate_rmuc_grid.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='diagnostic_dr_ablation' for k in n.keys))
    expr=next(v for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant) and k.value=='passed')
    args=SimpleNamespace(world_model_action_noise_std=0.,diagnostic_wheel_peak_nm=None,diagnostic_plane_terrain=False,diagnostic_total_residual_budget=None,
        disable_high_speed_feedback=False,disable_stage_head=None,base_wheel_clip=None,radial_spring_force=None,
        gas_spring_torque=None,dr_ablation='none',audit_critic=False,action_bias_sweep=None,pitch_rate_feedback=None,
        motor_response_time_constants_ms=[2.,30.], diagnostic_disable_outer_feedback=False,
        collect_wheel_teacher=False, diagnostic_hardware_profile=None)
    ns={'args':args,'rows':[{'passed':True}],'sequence_table':None}
    assert not eval(compile(ast.Expression(expr),'<actual report>','eval'),ns)
    args.motor_response_time_constants_ms=None
    assert eval(compile(ast.Expression(expr),'<actual report>','eval'),ns)


@pytest.mark.parametrize('dt,low,high',[(0,.01,.02),(.002,-.001,.02),(.002,.03,.02),(.002,.01,float('nan'))])
def test_invalid_response_parameters_rejected(dt,low,high):
    with pytest.raises(ValueError):m.EffortResponse(torch.zeros(1,2),dt,low,high)


def test_zero_endpoint_and_independent_response_strength():
 r=m.EffortResponse(torch.zeros(4096,2),.002,0.,.005)
 r.sample(torch.arange(4096),m.response_strength(.1,1.))
 assert r.time_constant.max()>.0049 and r.time_constant.min()>=0
 assert m.response_strength(.1)==.1
 z=m.EffortResponse(torch.zeros(2,2),.002,0.,0.);z.sample(torch.arange(2),1.)
 x=torch.tensor([[1.,-2.],[4.,-4.]])
 assert torch.equal(z.step(x,torch.full_like(x,4.89)),x)
 with pytest.raises(ValueError):m.response_strength(.1,2.)


def test_explicit_response_override_follows_joint_profile():
 text=(ROOT/'scripts/evaluate_rmuc_grid.py').read_text()
 assert text.index('cfg.motor_response_strength_override=args.motor_response_strength')>text.index('apply_joint_dr_profile(cfg,')
 assert text.index('cfg.motor_response_time_constant_max_s = args.motor_response_time_constants_ms[1]/1000.')<text.index('env = RslRlVecEnvWrapper(gym.make')
