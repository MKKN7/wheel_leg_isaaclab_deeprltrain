"""Reject silently truncated rehearsal weights after command-line overrides."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'
TREE = ast.parse(SOURCE.read_text())
FUNCTION = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                and n.name == 'validate_rehearsal_priority_config')
NAMESPACE = {'math': math}
exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), str(SOURCE), 'exec'), NAMESPACE)
validate = NAMESPACE[FUNCTION.name]


@pytest.mark.parametrize('stages,weights', [
    ([48, 49], [.02, .12, .12]),
    ([48, 48], [1., 1.]),
    ([48, 49], [1., float('nan')]),
    ([48, 49], [1., float('inf')]),
    ([48, 49], [1., 0.]),
])
def test_invalid_overrides_rejected(stages, weights):
    with pytest.raises(ValueError):
        validate(SimpleNamespace(command_rehearsal_priority_stage_ids=stages,
                                 command_rehearsal_priority_stage_weights=weights))


def test_explicit_paired_weights_and_empty_priority_allowed():
    for stages, weights in [([48, 49], [1., 1.]), ([], [])]:
        validate(SimpleNamespace(command_rehearsal_priority_stage_ids=stages,
                                 command_rehearsal_priority_stage_weights=weights))


def test_validation_precedes_simulation_construction():
    cls = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'RmucWheelLegEnv')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    calls = [ast.unparse(node) for node in init.body[:3]]
    assert calls[0] == 'self._separated_wheel_passive = configure_wheel_torque_contract(cfg)'
    assert calls[1] == 'validate_rehearsal_priority_config(cfg)'
    assert calls[2].startswith('super().__init__')
