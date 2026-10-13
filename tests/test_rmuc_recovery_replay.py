import copy
import importlib.util
from pathlib import Path
import torch

path=Path(__file__).resolve().parents[1]/'scripts/rsl_rl/rmuc_recovery_replay.py'
spec=importlib.util.spec_from_file_location('recovery_replay',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_auxiliary_gradients_match_joint_loss_before_and_after_clipping():
    torch.manual_seed(9)
    actor=torch.nn.Linear(3,2);expected=copy.deepcopy(actor)
    std=torch.nn.Parameter(torch.ones(2));expected_std=torch.nn.Parameter(std.detach().clone())
    params=list(actor.parameters())+[std];expected_params=list(expected.parameters())+[expected_std]
    helper=m.AuxiliaryGradients(params)
    for _ in range(2):
        for p in params+expected_params:p.grad=None
        replay_obs=torch.randn(7,3);target=torch.randn(7,2);ppo_obs=torch.randn(5,3)
        loss=.7*(actor(replay_obs)-target).square().mean()
        helper.prepare(loss)
        (actor(ppo_obs).square().mean()+std.square().sum()).backward()
        joint=(expected(ppo_obs).square().mean()+expected_std.square().sum()
               +.7*(expected(replay_obs)-target).square().mean())
        joint.backward()
        for a,b in zip(params,expected_params):torch.testing.assert_close(a.grad,b.grad)
        torch.nn.utils.clip_grad_norm_(params,.1)
        torch.nn.utils.clip_grad_norm_(expected_params,.1)
        for a,b in zip(params,expected_params):torch.testing.assert_close(a.grad,b.grad)
        helper.clear()
    assert not helper.gradients


def test_clear_removes_injection_on_subsequent_backward():
    p=torch.nn.Parameter(torch.tensor(2.));helper=m.AuxiliaryGradients([p])
    helper.prepare(p.square());helper.clear()
    (3*p).backward()
    assert p.grad.item()==3.
