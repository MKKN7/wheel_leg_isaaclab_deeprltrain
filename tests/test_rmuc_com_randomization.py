"""CPU test of local-frame COM sampling and actual partial-write protocol."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "com_randomization", ROOT / "source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/com_randomization.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class FakePhysxAsset:
    def __init__(self, n=32):
        self.body_names = ["wheel", "robot", "link"]
        self.com = torch.zeros(n, 3, 7)
        self.com[..., :3] = torch.tensor([.12, -.02, .004])
        # Deliberately nontrivial principal-axis rotation must remain intact.
        self.com[..., 3:] = torch.tensor([.5, -.5, .5, .5])
        self.mass = torch.full((n, 3), 2.)
        self.inertia = torch.arange(9, dtype=torch.float32).repeat(n, 3, 1)
        self.data = SimpleNamespace(body_com_pose_b=SimpleNamespace(torch=self.com))
        self.writes = []

    def set_coms_index(self, *, coms, body_ids, env_ids):
        assert coms.shape == (len(env_ids), len(body_ids), 7)
        self.writes.append((coms.clone(), list(body_ids), env_ids.clone()))
        self.com[env_ids[:, None], torch.tensor(body_ids)[None, :]] = coms


def test_physical_flu_to_body_and_initial_world_directions():
    physical_axes = torch.eye(3)
    body_axes = m.flu_to_body(physical_axes)
    expected_body = torch.tensor([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
    assert torch.equal(body_axes, expected_body)
    # Authored +90deg X body->world pose gives F=world+Y,L=world-X,U=world+Z.
    body_to_world = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    world_axes = body_axes @ body_to_world.T
    assert torch.equal(world_axes, torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]))
    torch.testing.assert_close(torch.linalg.cross(world_axes[0], world_axes[1]), world_axes[2])


def test_disabled_default_does_not_write_or_consume_rng():
    asset = FakePhysxAsset()
    before = asset.com.clone()
    model = m.RootComRandomizer(asset, 1)
    rng = torch.get_rng_state().clone()
    model.reset(None, 1.)
    assert not asset.writes
    assert torch.equal(asset.com, before)
    assert torch.equal(torch.get_rng_state(), rng)


def test_subset_bounds_preserve_other_bodies_and_principal_axes_and_inertia():
    asset = FakePhysxAsset(512)
    original = asset.com.clone()
    inertia, mass = asset.inertia.clone(), asset.mass.clone()
    model = m.RootComRandomizer(asset, 1, enabled=True)
    ids = torch.arange(0, 512, 2)
    model.reset(ids, .1)
    assert torch.all(model.offset_flu_m[ids].abs() <= torch.tensor([.001, .001, .0005]))
    assert model.offset_flu_m[ids].std() > 0
    assert torch.equal(asset.com[1::2], original[1::2])
    assert torch.equal(asset.com[:, [0, 2]], original[:, [0, 2]])
    assert torch.equal(asset.com[..., 3:], original[..., 3:])
    assert torch.equal(asset.inertia, inertia) and torch.equal(asset.mass, mass)
    data = model.readback()
    assert data["actual_matches_requested"]
    assert data["com_quaternion_exactly_preserved"]
    assert data["scope"].startswith("engineering_prior")


def test_repeated_resets_anchor_to_nominal_not_current_random_com():
    asset = FakePhysxAsset()
    model = m.RootComRandomizer(asset, 1, enabled=True)
    torch.manual_seed(714)
    model.reset(None, 1.)
    first = asset.com.clone()
    torch.manual_seed(714)
    model.reset(None, 1.)
    assert torch.equal(asset.com, first)
    # Calling strength0 restores precisely the root nominal COM without RNG.
    rng = torch.get_rng_state().clone()
    model.reset(None, 0.)
    assert torch.equal(asset.com[:, 1:2], model.nominal_com_pose_b)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(model.offset_body_m, torch.zeros_like(model.offset_body_m))


def test_snapshot_is_independent_and_empty_reset_is_noop():
    asset = FakePhysxAsset()
    model = m.RootComRandomizer(asset, 1, enabled=True)
    initial = model.nominal_com_pose_b.clone()
    asset.com[:, 1, :3].add_(1.)
    assert torch.equal(model.nominal_com_pose_b, initial)
    rng = torch.get_rng_state().clone()
    model.reset(torch.empty(0, dtype=torch.long), .5)
    assert not asset.writes and torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("kwargs", [
    {"max_offset_flu_m": [1., 2.]}, {"max_offset_flu_m": [-.01, .01, .005]},
    {"max_offset_flu_m": [.01, float("nan"), .005]}, {"backend": "newton"},
    {"enabled": 1},
])
def test_invalid_config_fails_closed(kwargs):
    with pytest.raises(ValueError):
        m.RootComRandomizer(FakePhysxAsset(), 1, **kwargs)


@pytest.mark.parametrize("strength", [-.1, 1.1, float("nan"), True])
def test_invalid_strength_fails_closed(strength):
    model = m.RootComRandomizer(FakePhysxAsset(), 1, enabled=True)
    with pytest.raises(ValueError):
        model.reset(None, strength)


def test_root_target_and_env_indices_fail_closed():
    with pytest.raises(ValueError):
        m.RootComRandomizer(FakePhysxAsset(), 0, enabled=True)
    model = m.RootComRandomizer(FakePhysxAsset(), 1, enabled=True)
    for ids in ([-1], [32], [0, 0], [0.5]):
        with pytest.raises(ValueError):
            model.reset(ids, .1)
