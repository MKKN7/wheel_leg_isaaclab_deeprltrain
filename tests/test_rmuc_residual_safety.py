"""Verify the injected gradient equals an explicit PPO + L2 loss and audit stops."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location('residual_safety', Path(__file__).resolve().parents[1] /
                                            'scripts/rsl_rl/rmuc_residual_safety.py')
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)


def test_probe_covers_all_environments_without_stride_aliasing():
    observations = torch.zeros(128, 128, 25)
    observations[:, :, 0] = torch.arange(128)[None, :]
    observations[:, :, 1] = torch.arange(128)[:, None]
    rng = torch.random.get_rng_state().clone()
    probe = safety.rollout_probe(observations)
    assert probe.shape == (512, 25)
    assert probe[:, 0].unique().numel() == 128
    assert probe[:, 1].unique().numel() == 4
    assert not probe.requires_grad
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize('coefficient', [0., .3, 1.])
def test_gradient_matches_explicit_gaussian_loss(coefficient):
    torch.manual_seed(72)
    weight = torch.randn(25, 6, requires_grad=True)
    direct = weight.detach().clone().requires_grad_()
    obs, action, anchor = torch.randn(17, 25), torch.randn(17, 6), torch.randn(17, 6)
    mean = obs @ weight
    loss_value = safety.regularize_mean(mean, anchor, coefficient)
    torch.distributions.Normal(mean, .2).log_prob(action).mean().neg().backward()
    direct_mean = obs @ direct
    explicit_l2 = coefficient * (direct_mean - anchor).square().mean()
    (-torch.distributions.Normal(direct_mean, .2).log_prob(action).mean() + explicit_l2).backward()
    torch.testing.assert_close(weight.grad, direct.grad)
    assert loss_value == pytest.approx(float(explicit_l2.detach()))


class Distribution(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._distribution = None

    def update(self, mean):
        self._distribution = torch.distributions.Normal(mean, .2)


class Actor(torch.nn.Module):
    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Linear(25, 6).requires_grad_(False)
        self.residual = torch.nn.Linear(25, 6)
        self.distribution = Distribution()

    def get_latent(self, obs, *args):
        return obs['policy']

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        x = self.get_latent(obs)
        mean = self.mlp(x) + self.residual(x)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution._distribution.sample()
        return mean


@pytest.mark.parametrize('tamper', [False, True])
@pytest.mark.parametrize('explicit_anchor', [False, True])
def test_audit_and_frozen_violation(tmp_path, tamper, explicit_anchor):
    actor = Actor()
    obs = {'policy': torch.randn(2, 8, 25)}
    optimizer = torch.optim.SGD(actor.parameters(), lr=.001)
    frozen_before = actor.mlp.weight.clone()
    anchor_path = None
    if explicit_anchor:
        anchor_path = str(tmp_path/'anchor.pt')
        torch.save({'actor_state_dict': actor.state_dict()}, anchor_path)
        with torch.no_grad():
            actor.residual.bias.add_(.05)
    output_before = actor(obs).detach().clone()
    saved = []

    def update():
        optimizer.zero_grad()
        actor(obs, stochastic_output=True)
        actor.distribution._distribution.log_prob(torch.zeros(2, 8, 6)).mean().neg().backward()
        optimizer.step()
        if tamper:
            with torch.no_grad():
                actor.mlp.weight[0, 0].add_(.1)
        return {}

    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: actor, update=update,
                             storage=SimpleNamespace(observations=obs)),
                             logger=SimpleNamespace(log_dir=tmp_path),
                             current_learning_iteration=123, save=saved.append)
    safety.configure_residual_safety(runner, 1., anchor_path)
    torch.testing.assert_close(actor(obs), output_before)
    if tamper:
        with pytest.raises(RuntimeError, match='safety violation'):
            runner.alg.update()
        assert saved
    else:
        runner.alg.update()
        runner.alg.update()
        torch.testing.assert_close(actor.mlp.weight, frozen_before, rtol=0, atol=0)
    records = [json.loads(line) for line in (tmp_path/'residual_safety.jsonl').read_text().splitlines()]
    assert records[0]['iteration'] == 123
    assert records[0]['regularized_minibatches'] == 1
    assert records[0]['anchor_checkpoint'] == anchor_path
    if explicit_anchor:
        assert records[0]['anchor_l2_loss'] == pytest.approx(.05 ** 2)
    assert bool(records[-1]['changed_frozen_names']) == tamper
    if not tamper:
        assert records[-1]['iteration'] == 124
        assert records[-1]['anchor_l2_loss'] > 0


def test_routing_buffer_tamper_is_saved_and_stopped(tmp_path):
    actor=Actor();actor.register_buffer('rmuc_test_route',torch.zeros(6))
    observations={'policy':torch.randn(2,8,25)};saved=[]
    def update():
        actor.rmuc_test_route[0]=.12
        return {}
    runner=SimpleNamespace(alg=SimpleNamespace(get_policy=lambda:actor,update=update,
        storage=SimpleNamespace(observations=observations)),logger=SimpleNamespace(log_dir=tmp_path),
        current_learning_iteration=0,save=saved.append)
    safety.configure_residual_safety(runner,0.)
    with pytest.raises(RuntimeError,match='safety violation'):runner.alg.update()
    assert saved
    record=json.loads((tmp_path/'residual_safety.jsonl').read_text().splitlines()[-1])
    assert 'rmuc_test_route' in record['changed_frozen_names']
    assert record['immutable_routing_buffer_abs_max_delta']==pytest.approx(.12)


def test_anchor_gradient_conflict_measurement_does_not_change_update():
    mean=torch.ones(2,6,requires_grad=True);records=[]
    safety.regularize_mean(mean,torch.zeros_like(mean),1.,records)
    (-mean.mean()).backward()
    assert len(records)==1
    assert records[0]['cosine']==pytest.approx(-1.,abs=1e-6)
    assert records[0]['anchor_to_ppo_norm_ratio']==pytest.approx(2.)
    torch.testing.assert_close(mean.grad,torch.full_like(mean,1/12))


def test_anchor_gradient_zero_ppo_is_reported_without_infinite_ratio():
    mean=torch.ones(2,6,requires_grad=True);records=[]
    safety.regularize_mean(mean,torch.zeros_like(mean),1.,records)
    (mean.sum()*0).backward()
    assert records[0]['anchor_to_ppo_norm_ratio'] is None
    assert records[0]['cosine'] is None
