import ast
from pathlib import Path
import runpy
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg'
select = runpy.run_path(str(TASK/'policy_speed_source.py'))['select_policy_forward_velocity']


def test_default_preserves_reference_and_rng():
    root = torch.tensor([.3, -.2]); qd = torch.zeros(2, 6)
    state = torch.random.get_rng_state()
    assert select('root_heading', root, qd, .06) is root
    assert torch.equal(state, torch.random.get_rng_state())


def test_forward_wheel_sign_and_mirrored_physical_hip():
    qd = torch.tensor([[0., 0., -5., 0., 0., -5.],
                       [1., 1., 0., 1., -1., 0.]])
    torch.testing.assert_close(select('desktop2_proxy', torch.zeros(2), qd, .06), torch.tensor([.3, -.06]))


def test_unknown_source_rejected():
    with pytest.raises(ValueError): select('typo', torch.zeros(1), torch.zeros(1, 6), .06)


def test_firmware_leg_rate_deadband_does_not_zero_wheel_rate():
    qd = torch.tensor([[.021, .021, -.021, .021, -.021, -.021]])
    torch.testing.assert_close(select('desktop2_proxy', torch.zeros(1), qd, .06), torch.tensor([.00126]))


def test_real_observation_builder_changes_only_obs6_before_delay_and_noise():
    tree = ast.parse((TASK/'env.py').read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_instant_policy_observation')
    space = {'torch': torch, 'select_policy_forward_velocity': select}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(TASK/'env.py'), 'exec'), space)
    builder = space['_instant_policy_observation']
    n = 4
    qd = torch.tensor([[0., 0., -5., 0., 0., -5.]]).repeat(n, 1)
    u = NS(cfg=NS(command_observation_mode='tracking_error_with_target', command_observation_error_clip=2., command_yaw_observation_error_clip=2.,wheel_radius_m=.06),
           robot=NS(data=NS(joint_pos=NS(torch=torch.zeros(n,6)), joint_vel=NS(torch=qd),root_lin_vel_w=NS(torch=torch.zeros(n,3)))),
           _control_dof_ids=list(range(6)),_default_control_pos=torch.zeros(n,6),_command_scale=torch.tensor([2.1,.5,1.]),_policy_actions=torch.arange(n*6).reshape(n,6).float()/30)
    command=torch.tensor([[.5,0.,.2]]).repeat(n,1)
    u._current_command=lambda:command
    u._state=lambda:(torch.tensor([[0.,0.,-1.]]).repeat(n,1),torch.zeros(n,3),torch.ones(n,3)*.1)
    u._heading_forward_velocity=lambda _:torch.tensor([.1,.2,.3,.4])
    baseline,_=builder(u)
    u.cfg.diagnostic_policy_speed_source='root_heading'
    explicit,_=builder(u);torch.testing.assert_close(baseline,explicit,rtol=0,atol=0)
    state=torch.random.get_rng_state();u.cfg.diagnostic_policy_speed_source='desktop2_proxy'
    proxy,_=builder(u)
    other=[i for i in range(25) if i!=6]
    torch.testing.assert_close(proxy[:,other],baseline[:,other],rtol=0,atol=0)
    torch.testing.assert_close(proxy[:,6],torch.full((n,),.42))
    assert torch.equal(state,torch.random.get_rng_state())
