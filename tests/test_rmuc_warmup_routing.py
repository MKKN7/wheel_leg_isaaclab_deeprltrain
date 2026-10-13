import ast
from pathlib import Path
from types import SimpleNamespace
import torch


def functions():
    path = Path(__file__).resolve().parents[1]/'scripts/rsl_rl/train.py'
    tree = ast.parse(path.read_text())
    names = {'_restore_actor_trainability', '_configure_turn_residual_bank_schedule'}
    namespace = {'RMUC_TURN_DELTA_SPECS': [(26, 'first'), (28, 'second')]}
    def route(actor, level):
        actor._rmuc_turn_delta_active_index = {26: 0, 28: 1}.get(level)
        for i, head in enumerate(actor.rmuc_turn_delta_adapters):
            head.requires_grad_(i == actor._rmuc_turn_delta_active_index)
        return str(level)
    namespace['set_rmuc_turn_delta_trainable'] = route
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_level_transition_during_warmup_stays_frozen_and_releases_current_head():
    ns = functions()
    actor = torch.nn.Module()
    actor.rmuc_turn_delta_adapters = torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])
    actor.domain = torch.nn.Linear(1, 1)
    ns['set_rmuc_turn_delta_trainable'](actor, 26)
    saved = [(p, p.requires_grad) for p in actor.parameters()]
    runner = SimpleNamespace(env=SimpleNamespace(_command_level=26),
        alg=SimpleNamespace(get_policy=lambda: actor,
            update=lambda: {'any_trainable': any(p.requires_grad for p in actor.parameters())}),
        _codex_actor_warmup={'active': True})
    ns['_configure_turn_residual_bank_schedule'](runner)
    assert not runner.alg.update()['any_trainable']
    runner.env._command_level = 28
    assert not runner.alg.update()['any_trainable']
    runner._codex_actor_warmup['active'] = False
    ns['_restore_actor_trainability'](runner, saved)
    assert not any(p.requires_grad for p in actor.rmuc_turn_delta_adapters[0].parameters())
    assert all(p.requires_grad for p in actor.rmuc_turn_delta_adapters[1].parameters())
    assert all(p.requires_grad for p in actor.domain.parameters())
