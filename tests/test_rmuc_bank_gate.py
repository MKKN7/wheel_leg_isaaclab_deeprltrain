"""Turning certification must reject an unbanked direction independently."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.mark.parametrize("yaw,errors,expected", [(0., [9., 9.], True),
                                               (.3, [2., 2.], True),
                                               (.3, [3.8, .4], False),
                                               (.3, [.4, 3.8], False)])
@pytest.mark.parametrize('require_bank',[True,False])
def test_directional_bank_gate(yaw, errors, expected, require_bank):
    path = Path(__file__).resolve().parents[1] / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RmucWheelLegEnv')
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_active_turn_gate_status')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    cfg = NS(command_require_analytic_bank=True, command_stage_yaw_rate_rad_s=[yaw], command_turn_gate_min_samples_per_direction=300,
             command_rehearsal_gate_min_episodes=64, command_advance_yaw_error_rad_s=.05,
             command_min_stable_env_fraction=.45, command_min_survival_env_fraction=.95,
             command_turn_gate_max_bank_error_deg=3.)
    env = NS(device='cpu', cfg=cfg, _command_level=0, _command_linear_error_limit=lambda: .15)
    for name, value in {'_command_turn_gate_sample_count': 500,
                        '_command_turn_survival_episode_count': 100,
                        '_command_turn_gate_linear_error_ema': .02,
                        '_command_turn_gate_yaw_error_ema': .02,
                        '_command_turn_gate_stable_fraction_ema': .9,
                        '_command_turn_survival_fraction_ema': 1.}.items():
        setattr(env, name, torch.full((2,), value))
    env._command_turn_gate_bank_error_deg_ema = torch.tensor(errors)
    cfg.command_require_analytic_bank = require_bank
    result = namespace[function.name](env)
    assert bool(result['bank']) == (expected if require_bank else True)
    assert all(bool(result[k]) for k in ('ready', 'linear', 'yaw', 'stable', 'survival'))


@pytest.mark.parametrize('errors,turning,expected', [([[2.,2.],[2.,2.]], [True,True], True), ([[3.5,.5],[.5,.5]], [True,True], False), ([[.5,3.5],[.5,.5]], [True,True], False), ([[20.,20.],[.5,.5]], [False,True], True)])
def test_retention_bank_directions_and_straight_exemption(errors, turning, expected):
    path = Path(__file__).resolve().parents[1] / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'RmucWheelLegEnv')
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_rehearsal_gate_status')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    cfg = NS(command_require_analytic_bank=True, command_rehearsal_gate_enforce_start_level=24, command_rehearsal_enabled=True,
             command_rehearsal_gate_min_samples=300, command_rehearsal_gate_min_episodes=64,
             command_advance_yaw_error_rad_s=.05, command_min_stable_env_fraction=.45,
             command_min_survival_env_fraction=.95, command_turn_gate_max_bank_error_deg=3.)
    env = NS(device='cpu', cfg=cfg, _command_level=26, _rehearsal_gate_stage_ids=torch.tensor([18,23]),
             _rehearsal_gate_turning=torch.tensor(turning), _rehearsal_gate_linear_limits=torch.tensor([.2,.15]),
             _rehearsal_gate_bank_error_deg_ema=torch.tensor(errors))
    for name,value in [('sample_count',500),('linear_error_ema',.02),('yaw_error_ema',.02),
                       ('stable_fraction_ema',.9),('survival_episode_count',100),('survival_fraction_ema',1.)]:
        setattr(env,'_rehearsal_gate_'+name,torch.full((2,2),value))
    result=namespace[function.name](env)
    assert bool(result['bank']) == expected
    assert all(bool(result[k]) for k in ('ready','linear','yaw','stable','survival'))
    env._rehearsal_gate_sample_count.zero_()
    assert not bool(namespace[function.name](env)['bank'])
    cfg.command_rehearsal_enabled=False
    assert bool(namespace[function.name](env)['bank'])


def test_retention_bank_accumulates_only_ready_matching_direction():
    path = Path(__file__).resolve().parents[1] / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'RmucWheelLegEnv')
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_update_rehearsal_gate_stats')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    env = NS(cfg=NS(command_rehearsal_enabled=True,command_rehearsal_gate_min_samples=300,command_advance_yaw_error_rad_s=.05),
             _rehearsal_gate_stage_ids=torch.tensor([18]), _rehearsal_stage=torch.tensor([18,18,18,18,18]),
             _rehearsal_mask=torch.tensor([True,True,True,False,True]), _rehearsal_gate_turning=torch.tensor([True]),
             _rehearsal_gate_linear_limits=torch.tensor([.2]))
    for suffix in ['linear_error_ema','yaw_error_ema','stable_fraction_ema','bank_error_deg_ema']:
        setattr(env,'_rehearsal_gate_'+suffix,torch.zeros(1,2))
    env._rehearsal_gate_sample_count=torch.zeros(1,2,dtype=torch.long)
    command=torch.tensor([[2.,0,.3],[2.,0,.3],[2.,0,-.3],[2.,0,-.3],[2.,0,-.3]])
    namespace[function.name](env,safe=torch.ones(5,dtype=torch.bool),window_linear_error=torch.zeros(5),
        window_yaw_error=torch.zeros(5),bank_error_deg=torch.tensor([3.,5.,1.,99.,99.]),command=command,
        window_ready=torch.tensor([True,True,True,True,False]))
    torch.testing.assert_close(env._rehearsal_gate_bank_error_deg_ema,torch.tensor([[4.,1.]]))
    assert env._rehearsal_gate_sample_count.tolist()==[[2,1]]
