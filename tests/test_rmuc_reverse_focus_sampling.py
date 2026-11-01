import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/rmuc_world_model'))
from domain_sampling import ActiveReverseSampler, DelayDomainSampler


def fixture():
    ctx = torch.zeros(1000, 35)
    ctx[500:, 12:14] = 1
    ctx[:, -3] = -.25
    ctx[980:, -3] = -.6
    obs = torch.zeros(1000, 25)
    obs[:, 7] = ctx[:, -3] * .5
    adaptation = SimpleNamespace(reverse_gate=lambda x: torch.ones(len(x)),
                                 reverse_update_domain=torch.tensor([-.6, -.45]))
    replay = SimpleNamespace(starts=torch.arange(1000), context=ctx)
    return replay, obs, adaptation


def test_focus_reaches_active_windows_without_inventing_missing_domain():
    replay, obs, adaptation = fixture()
    sampler = ActiveReverseSampler(replay, obs, adaptation, .75, DelayDomainSampler(replay))
    ids = sampler.sample(256, torch.Generator().manual_seed(1))
    assert len(ids) == 256
    assert (ids >= 980).sum() >= 192
    assert (ids < 500).sum() == 32
    assert sampler.report['active_delay_domains'] == [3]
    assert sampler.report['available_delay_domains'] == [0, 3]
    assert sampler.last_active_fraction >= .75


def test_reject_no_active_windows_and_invalid_fraction():
    replay, obs, adaptation = fixture()
    obs[:, 7] = -.1
    with pytest.raises(ValueError, match='No active'):
        ActiveReverseSampler(replay, obs, adaptation, .75, None)
    with pytest.raises(ValueError, match='fraction'):
        ActiveReverseSampler(replay, obs, adaptation, 1., None)


def test_anchor_focus_samples_low_command_not_protected_fast_domain():
    replay, obs, adaptation = fixture()
    sampler = ActiveReverseSampler(replay, obs, adaptation, .75,
                                   DelayDomainSampler(replay), anchor=True,
                                   command_range=[-.3, -.2])
    ids = sampler.sample(256, torch.Generator().manual_seed(3))
    assert (ids < 980).sum() >= 192
    assert sampler.report['active_windows'] == 980
    assert sampler.report['active_delay_domains'] == [0, 3]


def test_phase_mask_excludes_other_parts_of_same_command():
    replay, obs, adaptation = fixture()
    eligible = torch.zeros(1000, dtype=torch.bool)
    eligible[700:800] = True
    sampler = ActiveReverseSampler(replay, obs, adaptation, .75, None,
                                   anchor=True, command_range=[-.3, -.2], eligible=eligible)
    ids = sampler.sample(256, torch.Generator().manual_seed(8))
    assert ((ids >= 700) & (ids < 800)).sum() >= 192
    assert sampler.report['active_windows'] == 100
