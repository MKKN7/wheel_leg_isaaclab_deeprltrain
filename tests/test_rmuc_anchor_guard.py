import importlib.util
from pathlib import Path
import pytest
import torch

spec = importlib.util.spec_from_file_location('anchor_guard', Path(__file__).resolve().parents[1] / 'scripts/rsl_rl/rmuc_anchor_guard.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize('optimizer_type', [torch.optim.Adam, torch.optim.AdamW])
def test_momentum_decay_freeze_and_release(optimizer_type):
    p = torch.nn.Parameter(torch.ones(3, 2, 4))
    opt = optimizer_type([p], lr=.1, weight_decay=.1, amsgrad=True)
    p.sum().backward(); opt.step(); opt.zero_grad()
    assert opt.state[p]['exp_avg'][1, 0].abs().sum() > 0
    enabled = [True]
    handles = m.protect_optimizer_slices(opt, [p], (1, 0), lambda: enabled[0])
    anchor = p[1, 0].detach().clone()
    other = p[0, 0].detach().clone()
    for _ in range(4):
        p.square().sum().backward(); opt.step(); opt.zero_grad()
        assert torch.equal(p[1, 0], anchor)
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            assert torch.count_nonzero(opt.state[p][key][1, 0]) == 0
    assert not torch.equal(p[0, 0], other)
    enabled[0] = False
    p.square().sum().backward(); opt.step()
    assert not torch.equal(p[1, 0], anchor)
    assert opt.state[p]['exp_avg'][1, 0].abs().sum() > 0
    for h in handles:
        h.remove()
