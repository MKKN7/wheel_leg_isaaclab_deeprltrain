"""Checkpoint compatibility and command-local restoration of a frozen anchor."""
import importlib.util
from pathlib import Path
import pytest
import torch

spec = importlib.util.spec_from_file_location('anchor_adapter', Path(__file__).resolve().parents[1] / 'scripts/rsl_rl/rmuc_command_adapter.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


@pytest.mark.parametrize('explicit_seed', [False, True])
def test_router_checkpoint_seed_migration_preserves_recovery_output(explicit_seed, monkeypatch):
    import sys
    router_path = Path(__file__).resolve().parents[1] / 'scripts/rsl_rl/rmuc_state_router.py'
    router_spec = importlib.util.spec_from_file_location('rmuc_state_router', router_path)
    module = importlib.util.module_from_spec(router_spec)
    router_spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, 'rmuc_state_router', module)
    reference = module.IsolatedTurnStateRouter(recovery_positive_2p0=True)
    with torch.no_grad():
        for p in reference.parameters():
            p.uniform_(-.2, .2)
        if explicit_seed:
            reference.recovery_positive_2p0_wheel_seed.copy_(torch.tensor([0., 0., -.01, 0., 0., .01]))
    state = {'rmuc_state_router.' + k: v.clone() for k, v in reference.state_dict().items()}
    key = 'rmuc_state_router.recovery_positive_2p0_wheel_seed'
    if not explicit_seed:
        del state[key]
    actor = torch.nn.Module()
    actor.mlp = torch.nn.Sequential(torch.nn.Linear(25, 6))
    actor.rmuc_turn_adapter = True
    restored = adapter.install_rmuc_state_router(actor, state)
    restored.load_state_dict({k.removeprefix('rmuc_state_router.'): v for k, v in state.items()}, strict=True)
    x = torch.randn(256, 25) * .2
    x[:, 7] = torch.linspace(.85, 1.35, len(x))
    x[:, 8] = .3 - x[:, 2] / .25
    assert torch.equal(restored(x), reference(x))


def test_legacy_checkpoint_keeps_original_straight_correction():
    bank = adapter.RmucDomainResidualBank(25, 6, 4)
    with torch.no_grad():
        bank.heads['straight'][-1].bias.fill_(.3)
    state = bank.state_dict()
    del state['legacy_straight_anchor_suppression']
    restored = adapter.RmucDomainResidualBank(25, 6, 4)
    restored.load_state_dict(state, strict=True)
    obs = torch.zeros(3, 25)
    obs[:, 7] = torch.tensor([2.8, 2.9, 3.]) * .5
    assert torch.equal(bank(obs), restored(obs))
    assert (restored(obs) != 0).all()


def test_anchor_suppression_is_local_and_leaves_other_yaw_outputs_intact():
    bank = adapter.RmucDomainResidualBank(25, 6, 4)
    with torch.no_grad():
        for head in bank.heads.values():
            head[-1].bias.fill_(.3)
    obs = torch.zeros(7, 25)
    obs[:, 7] = torch.tensor([2.8, 2.9, 3., 2.9, 2.9, 2.9, 2.9]) * .5
    obs[:, 8] = torch.tensor([0., 0., 0., -.3, -.15, .15, .3])
    original = bank(obs)
    bank.legacy_straight_anchor_suppression.fill_(1.)
    repaired = bank(obs)
    assert torch.equal(repaired[1], torch.zeros(6))
    keep = torch.tensor([0, 2, 3, 4, 5, 6])
    assert torch.equal(repaired[keep], original[keep])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1])
def test_reject_invalid_routing_buffers(value):
    bank = adapter.RmucDomainResidualBank(25, 6, 4)
    state = bank.state_dict()
    state['legacy_straight_anchor_suppression'] = torch.tensor(value)
    with pytest.raises(RuntimeError, match='finite scalar'):
        bank.load_state_dict(state, strict=True)
    module = torch.nn.Module()
    module.register_buffer('rmuc_speed_expert_mix_gain', torch.ones(()))
    module.register_load_state_dict_pre_hook(adapter._load_speed_expert_mix_gain)
    with pytest.raises(RuntimeError, match='finite scalar'):
        module.load_state_dict({'rmuc_speed_expert_mix_gain': torch.tensor(value)})


def test_old_speed_expert_checkpoint_defaults_to_full_mix():
    module = torch.nn.Module()
    module.register_buffer('rmuc_speed_expert_mix_gain', torch.zeros(()))
    module.register_load_state_dict_pre_hook(adapter._load_speed_expert_mix_gain)
    module.load_state_dict({}, strict=True)
    assert module.rmuc_speed_expert_mix_gain.item() == 1.


def test_local_bank_seed_checkpoint_compatibility_and_support():
    bank=adapter.RmucDomainResidualBank(25,6,4)
    old=bank.state_dict();old.pop('positive_2p0_bank_seed')
    bank.load_state_dict(old,strict=True)
    assert torch.count_nonzero(bank.positive_2p0_bank_seed)==0
    bank.positive_2p0_bank_seed.copy_(torch.tensor([-.12,0,0,.12,0,0]))
    obs=torch.zeros(7,25);obs[:,7]=torch.tensor([1.75,2.,2.2,2.,2.,2.,2.])*0.5
    obs[:,8]=torch.tensor([.3,.3,.3,-.3,0.,.15,.25])
    y=bank(obs)
    torch.testing.assert_close(y[1],bank.positive_2p0_bank_seed)
    assert torch.count_nonzero(y[torch.tensor([0,2,3,4,5,6])])==0
    bad=bank.state_dict();bad['positive_2p0_bank_seed']=torch.full((6,),.25)
    with pytest.raises(RuntimeError,match='bounded'):bank.load_state_dict(bad)


def test_negative_seed_is_direction_local_and_backward_compatible():
    bank=adapter.RmucDomainResidualBank(25,6,4)
    old=bank.state_dict();old.pop('negative_2p0_yaw_seed');bank.load_state_dict(old,strict=True)
    assert bank.negative_2p0_yaw_seed.count_nonzero()==0
    bank.negative_2p0_yaw_seed.copy_(torch.tensor([0,0,.02,0,0,-.02]))
    x=torch.zeros(6,25);x[:,7]=torch.tensor([2.,2.,2.,2.,1.75,2.2])*.5;x[:,8]=torch.tensor([-.3,.3,-.25,0.,-.3,-.3])
    y=bank(x);torch.testing.assert_close(y[0],bank.negative_2p0_yaw_seed);assert y[1:].count_nonzero()==0


def test_positive_2p4_route_preserves_neighbors_and_legacy_heads():
    head=adapter.RmucTurnDeltaAdapter(25,6,4,speed_start=2.,speed_end=2.2,yaw_start=.15,yaw_end=.3)
    with torch.no_grad(): head.net[-1].bias.fill_(.2)
    state=head.state_dict();state.pop('positive_2p4_suppression')
    head.load_state_dict(state,strict=True)
    bank=adapter.RmucDomainResidualBank(25,6,4)
    old=bank.state_dict();old.pop('positive_2p4_bank_seed');bank.load_state_dict(old,strict=True)
    x=torch.zeros(8,25);x[:,7]=torch.tensor([2.2,2.4,2.6,2.4,2.4,2.4,2.3,2.5])*.5
    x[:,8]=torch.tensor([.3,.3,.3,-.3,.15,0.,.3,.3])
    original=head(x).detach().clone()
    head.positive_2p4_suppression.fill_(.7)
    bank.positive_2p4_bank_seed.copy_(torch.tensor([-.08,0,0,.08,0,0]))
    keep=torch.tensor([0,2,3,4,5]);assert torch.equal(head(x)[keep],original[keep])
    assert bank(x)[keep].count_nonzero()==0
    torch.testing.assert_close(head(x)[1],original[1]*.3)
    torch.testing.assert_close(bank(x)[1],bank.positive_2p4_bank_seed)
    torch.testing.assert_close(head(x)[6:8],original[6:8]*.65)
    for invalid in [float('nan'),-1.,1.1]:
        bad=head.state_dict();bad['positive_2p4_suppression']=torch.tensor(invalid)
        with pytest.raises(RuntimeError,match='finite scalar'):head.load_state_dict(bad)
