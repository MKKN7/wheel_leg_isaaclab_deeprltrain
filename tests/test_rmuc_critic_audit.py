import importlib.util
from pathlib import Path
import torch
import ast
import inspect
import textwrap
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('critic_audit', Path(__file__).resolve().parents[1] /
                                            'scripts/rmuc_critic_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_return_tail_distinguishes_timeout_from_true_termination():
    rewards = torch.ones(4, 2)
    terminated = torch.zeros(4, 2, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    terminated[1, 0] = True
    truncated[1, 1] = True
    returns, bounds = audit.returns_and_tail_bounds(rewards, terminated, truncated, .5, 1.)
    torch.testing.assert_close(returns[:2], torch.tensor([[1.5, 1.5], [1., 1.]]))
    torch.testing.assert_close(bounds[:2], torch.tensor([[0., .5], [0., 1.]]))
    # Omitted return for continuing unit rewards is exactly the reported bound.
    torch.testing.assert_close(2 - returns[2:], bounds[2:])


def test_metrics_reports_bias_even_when_explained_variance_is_perfect():
    target = torch.tensor([1., 2., 3.])
    result = audit.value_metrics(target + 10, target)
    assert result['explained_variance'] == 1.
    assert result['mean_bias'] == result['rmse'] == 10.


def test_installed_ppo_value_clipping_erases_large_return_correction():
    from rsl_rl.algorithms import PPO
    tree = ast.parse(textwrap.dedent(inspect.getsource(PPO.update)))
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Attribute) and node.test.attr == 'use_clipped_value_loss')
    fn = ast.parse('def loss(self, values, batch):\n    pass').body[0]
    fn.body = [branch] + ast.parse('return value_loss').body
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<ppo-value-loss>', 'exec'), namespace)
    batch = SimpleNamespace(values=torch.tensor([-1000.]), returns=torch.tensor([-1500.]))
    clipped = torch.tensor([-1001.], requires_grad=True)
    plain = clipped.detach().clone().requires_grad_()
    namespace['loss'](SimpleNamespace(use_clipped_value_loss=True, clip_param=.1), clipped, batch).backward()
    namespace['loss'](SimpleNamespace(use_clipped_value_loss=False, clip_param=.1), plain, batch).backward()
    assert clipped.grad.item() == 0
    assert plain.grad.item() == 998  # Descent moves value toward the lower target.
