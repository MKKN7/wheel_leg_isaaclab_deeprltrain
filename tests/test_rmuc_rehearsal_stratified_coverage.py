"""Retain rare signed commands without starving weighted reverse replay."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

SRC = Path(__file__).resolve().parents[1] / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg'
spec = importlib.util.spec_from_file_location('sampling', SRC / 'command_sampling.py')
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)
tree = ast.parse((SRC / 'env.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == '_refresh_rehearsal_assignments')
ns = {'torch': torch, 'ensure_priority_sign_coverage': sampling.ensure_priority_sign_coverage}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(SRC / 'env.py'), 'exec'), ns)
assign = ns[method.name]


def environment(enabled):
    cfg = SimpleNamespace(command_rehearsal_enabled=True, command_rehearsal_start_level=0,
        command_rehearsal_speed_focus_start_level=1000, command_rehearsal_speed_focus_end_level=1001,
        command_rehearsal_speed_focus_fraction=.2, command_rehearsal_fraction=.5,
        command_rehearsal_anchor_stage_ids=list(range(20)),
        command_rehearsal_priority_stage_ids=list(range(20)),
        command_rehearsal_priority_stage_weights=[.01]*15+[10.]*5,
        command_rehearsal_priority_fraction=1., command_rehearsal_stratified_coverage=enabled)
    return SimpleNamespace(cfg=cfg, num_envs=256, device='cpu', _command_level=20,
        _rehearsal_mask=torch.zeros(256, dtype=torch.bool),
        _rehearsal_stage=torch.zeros(256, dtype=torch.long),
        _turn_fixed_sign_mask=sampling.fixed_turn_sign_cohorts(256, .5, 66, 'cpu'))


def test_actual_sampler_covers_rare_commands_in_both_fixed_signs_without_extra_rng():
    for seed in range(10):
        old, new = environment(False), environment(True)
        torch.manual_seed(seed)
        assign(old)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(seed)
        assign(new)
        assert torch.equal(torch.get_rng_state(), rng)
        assert torch.equal(old._rehearsal_mask, new._rehearsal_mask)
        assert new._rehearsal_mask.sum() == 128
        assert torch.unique(old._rehearsal_stage[old._rehearsal_mask]).numel() < 20
        for sign in [True, False]:
            selected = new._rehearsal_mask & (new._turn_fixed_sign_mask == sign)
            assert set(new._rehearsal_stage[selected].tolist()) == set(range(20))
        # The weighted remainder still heavily favors the requested repair points.
        assert (new._rehearsal_stage[new._rehearsal_mask] >= 15).sum() >= 90


def test_small_asynchronous_reset_does_not_force_first_priority_or_mutate_input():
    stages = torch.tensor([17, 18, 19])
    before = stages.clone()
    result = sampling.ensure_priority_sign_coverage(stages, torch.tensor([True, False, True]),
                                                    torch.arange(20))
    assert torch.equal(result, before)
    assert torch.equal(stages, before)


def test_partial_reset_preserves_nonreset_episodes_and_global_quota():
    env = environment(True)
    torch.manual_seed(5)
    assign(env)
    old_stage, old_mask = env._rehearsal_stage.clone(), env._rehearsal_mask.clone()
    assign(env, torch.tensor([12]))
    retained = torch.arange(256) != 12
    assert torch.equal(env._rehearsal_stage[retained], old_stage[retained])
    assert torch.equal(env._rehearsal_mask[retained], old_mask[retained])
    assert env._rehearsal_mask.sum() == 128
