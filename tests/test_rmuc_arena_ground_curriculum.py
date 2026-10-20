"""Pair interpolation and reset invariants; no simulator import or GPU needed."""
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/arena_ground.py'))
PROFILE = ROOT / 'configs/rmuc_arena_ground.json'
blend = MODULE['curriculum_parameters']
factor = MODULE['factor_wheel_coefficients']
combine = MODULE['combine_coefficient']


def test_progress_zero_preserves_actual_legacy_pair_not_raw_material():
    # Legacy average .5 wheel / .5 ground is effective .5, not .25.
    base = (combine(.5, .5), combine(.5, .5), combine(0., 0.))
    for severity in (0., .3, 1.):
        effective = blend(PROFILE, 0., severity, base)
        wheel = factor(effective, (.5, .5, 0.))
        assert wheel == (1., 1., 0.)
        assert tuple(combine(wheel[i], (.5, .5, 0.)[i], 'multiply' if i < 2 else 'max') for i in range(3)) == base


def test_full_progress_hits_fresh_and_worn_and_intermediate_is_correlated():
    ground = (.5, .5, 0.)
    for severity, expected in [(0., (.65, .55, .02)), (1., (.45, .38, .02))]:
        pair = blend(PROFILE, 1., severity, (.5, .5, 0.))
        assert pair == pytest.approx(expected)
        wheel = factor(pair, ground)
        # Worn dynamic .38 would be impossible with legacy average combine;
        # multiply reaches it without modifying the shared terrain.
        assert wheel[0] * ground[0] == pytest.approx(expected[0])
        assert wheel[1] * ground[1] == pytest.approx(expected[1])
    assert blend(PROFILE, .1, 1., (.5, .5, 0.)) == pytest.approx((.495, .488, .002))


def test_priority_and_unrepresentable_factor_fail_closed():
    assert combine(.2, .5, 'average', 'multiply') == pytest.approx(.1)
    assert combine(.2, .5, 'multiply', 'max') == .5
    with pytest.raises(ValueError): factor((.5, .5, 0.), (.5, .5, .02))
    with pytest.raises(ValueError): factor((.5, .5, .02), (.5, 0., 0.))
    with pytest.raises(ValueError): blend(PROFILE, 1.1, .5, (.5, .5, 0.))
    with pytest.raises(ValueError): blend(PROFILE, .1, float('nan'), (.5, .5, 0.))


def test_disabled_duplicate_and_incompatible_modes_rejected():
    prepare = MODULE['prepare_curriculum_ground']
    cfg = SimpleNamespace(arena_ground_severity=-1., friction_randomization_enabled=False,
        domain_randomization_max_friction_fraction=0., arena_ground_migration_progress=.1,
        arena_ground_severity_range=(0., 1.), arena_ground_randomization_enabled=True,
        arena_ground_profile_path=PROFILE,
        terrain=SimpleNamespace(physics_material=SimpleNamespace(static_friction=.5,
            dynamic_friction=.5, restitution=0., friction_combine_mode='average', restitution_combine_mode='average')))
    original = cfg.terrain.physics_material
    audit = prepare(cfg)
    assert cfg.terrain.physics_material is original
    assert audit['terrain_material_changed'] is False
    cfg.domain_randomization_max_friction_fraction = .2
    with pytest.raises(ValueError): prepare(cfg)
    cfg.domain_randomization_max_friction_fraction = 0.
    cfg.arena_ground_severity = 0.
    with pytest.raises(ValueError): prepare(cfg)
    cfg.arena_ground_severity = -1.
    cfg.terrain.physics_material.friction_combine_mode = 'max'
    with pytest.raises(ValueError): prepare(cfg)


def test_reset_only_selected_envs_wheel_shapes_and_buffer_matches_readback():
    cls = MODULE['ArenaGroundRandomizer']
    dr = cls.__new__(cls)
    dr.torch = torch
    dr.wp = SimpleNamespace(float32=torch.float32, int32=torch.int32,
                            from_torch=lambda value, dtype: value)
    material = torch.full((5, 7, 3), .5)
    material[:, :, 2] = 0.
    dr._materials = lambda: material.clone()
    def setter(value, ids):
        material[ids.long()] = value[ids.long()]
    dr.env = SimpleNamespace(_domain_material_properties=material.clone(),
        robot=SimpleNamespace(root_view=SimpleNamespace(set_material_properties=setter)))
    dr.shape_indices = {'left_wheel': 1, 'right_wheel': 5}
    dr.palette_severity = torch.linspace(0., 1., 64)
    dr.palette = {name: torch.tensor([factor(blend(PROFILE, 1., float(s), (.5, .5, 0.)), (.5, .5, 0.))
                  for s in dr.palette_severity]) for name in dr.shape_indices}
    dr.audit = {'randomization_enabled': True}
    dr.ground = (.5, .5, 0.)
    dr.expected = material.clone()
    dr.severity = torch.zeros(5)
    dr.reset_counts = torch.zeros(5, dtype=torch.long)
    dr.write_readback_max_abs_error = 0.
    before = material.clone()
    torch.manual_seed(11)
    dr.reset(torch.tensor([1, 3]))
    assert torch.equal(material[[0, 2, 4]], before[[0, 2, 4]])
    assert torch.equal(material[:, [0, 2, 3, 4, 6]], before[:, [0, 2, 3, 4, 6]])
    assert torch.equal(material[[1, 3], 1], material[[1, 3], 5])
    assert torch.equal(dr.env._domain_material_properties, material)
    assert dr.reset_counts.tolist() == [0, 1, 0, 1, 0]
    assert dr.readback()['max_abs_error_vs_expected'] == 0.
    assert dr.audit['runtime_verified'] is True
    first = dr.severity.clone()
    dr.reset(torch.tensor([1, 3]))
    assert not torch.equal(first, dr.severity)


def test_unbound_colliders_use_explicit_simulation_default_and_null_modes():
    cfg=SimpleNamespace(sim=SimpleNamespace(physics_material=SimpleNamespace(
        static_friction=.6,dynamic_friction=.4,restitution=.01,
        friction_combine_mode=None,restitution_combine_mode=None)))
    values,modes=MODULE['configured_default_material'](cfg)
    assert values==(.6,.4,.01) and modes==('average','average')


def test_each_cloned_wheel_receives_explicit_material_binding():
    from pxr import Usd,UsdShade
    stage=Usd.Stage.CreateInMemory();material=UsdShade.Material.Define(stage,'/World/WheelMaterial')
    wheels={}
    for env in range(3):
        for name in ['left_wheel','right_wheel']:
            path=f'/World/envs/env_{env}/Robot/{name}/collision'
            stage.DefinePrim(path,'Cube')
            if env==0:wheels[name]={'collider':path,'material_path':'/World/WheelMaterial'}
    audit={'wheels':wheels};MODULE['bind_curriculum_clones'](stage,SimpleNamespace(scene=SimpleNamespace(num_envs=3)),audit)
    assert audit['cloned_colliders_bound']==6
    for env in range(3):
        for name in wheels:
            prim=stage.GetPrimAtPath(f'/World/envs/env_{env}/Robot/{name}/collision')
            resolved,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            assert resolved.GetPath()==material.GetPath()
