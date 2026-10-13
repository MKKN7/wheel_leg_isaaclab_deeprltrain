"""Exercise the training std configuration without launching Isaac Sim."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_configuration_function():
    path = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/train.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_configure_action_std")
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("name", ["std_param", "log_std_param"])
def test_wheel_floor_changes_parameter_and_survives_optimizer_update(name):
    values = torch.tensor([.004, .004, .001, .004, .004, .003])
    parameter = torch.nn.Parameter(values.log() if name.startswith("log") else values.clone())
    distribution = torch.nn.Module()
    distribution.register_parameter(name, parameter)
    actor = SimpleNamespace(distribution=distribution)

    def optimizer_update():
        with torch.no_grad():
            parameter[[2, 5]] -= 1.0
        return {"existing_metric": 1.0}

    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: actor, update=optimizer_update))
    load_configuration_function()(runner, freeze=False, freeze_leg=True, override=None,
                                  min_wheel_std=.006, command_level=26)
    actual = lambda: parameter.exp() if name.startswith("log") else parameter
    expected = torch.tensor([.004, .004, .006, .004, .004, .006])
    torch.testing.assert_close(actual(), expected)
    result = runner.alg.update()
    torch.testing.assert_close(actual(), expected)
    assert result["existing_metric"] == 1.0
    assert result["wheel_action_std_min"] == pytest.approx(.006)
    parameter.sum().backward()
    torch.testing.assert_close(parameter.grad, torch.tensor([0., 0., 1., 0., 0., 1.]))
