import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def test_evaluator_keeps_delay_exposure_after_terminal_reset():
    source = Path(__file__).resolve().parents[1]/'scripts/evaluate_rmuc_grid.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'u._sensor_model_enabled')
    env = SimpleNamespace(_sensor_model_enabled=True,
                          _sensor_delay_steps=torch.tensor([1, 0]),
                          _communication_delay_steps=torch.tensor([0, 1]))
    counts = torch.zeros(2, 2, dtype=torch.long)
    maximum = torch.zeros_like(counts)
    scope = dict(torch=torch, u=env, delay_exposure=counts, delay_maximum=maximum)
    code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), 'exec')
    exec(code, scope)
    exec(code, scope)
    # A terminal reset draws zero delays for the next episode; it must not erase
    # evidence of the delayed inputs already evaluated.
    env._sensor_delay_steps.zero_()
    env._communication_delay_steps.zero_()
    exec(code, scope)
    assert torch.equal(counts, torch.tensor([[2, 0], [0, 2]]))
    assert torch.equal(maximum, torch.tensor([[1, 0], [0, 1]]))
