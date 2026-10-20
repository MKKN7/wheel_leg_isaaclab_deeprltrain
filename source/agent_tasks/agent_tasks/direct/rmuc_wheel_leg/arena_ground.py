"""Explicit wheel-ground contact-pair prior. Disabled in historical environments."""
import json,math
from pathlib import Path

def arena_parameters(path,severity):
    if not math.isfinite(severity) or not 0<=severity<=1:raise ValueError('Arena severity must be in[0,1]')
    d=json.loads(Path(path).read_text());fresh=d['profiles']['fresh'];worn=d['profiles']['worn']
    values={k:fresh[k]+severity*(worn[k]-fresh[k]) for k in ['effective_static_friction','effective_dynamic_friction','effective_restitution']}
    if not 0<=values['effective_dynamic_friction']<=values['effective_static_friction'] or not 0<=values['effective_restitution']<=1:raise ValueError('Invalid material prior')
    return values

def prepare_ground(cfg):
    from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxRigidBodyMaterialCfg
    if cfg.friction_randomization_enabled or cfg.domain_randomization_max_friction_fraction!=0:
        raise ValueError('Arena material requires legacy ground/friction randomization disabled explicitly')
    values=arena_parameters(cfg.arena_ground_profile_path,cfg.arena_ground_severity)
    cfg.terrain.physics_material=PhysxRigidBodyMaterialCfg(static_friction=values['effective_static_friction'],dynamic_friction=values['effective_dynamic_friction'],restitution=values['effective_restitution'],friction_combine_mode='multiply',restitution_combine_mode='max')
    return values

def bind_wheels(stage,robot_path='/World/envs/env_0/Robot'):
    from pxr import Usd,UsdPhysics,UsdShade,PhysxSchema
    material_path='/World/RmucArenaWheelMaterial'
    material=UsdShade.Material.Define(stage,material_path)
    api=UsdPhysics.MaterialAPI.Apply(material.GetPrim());api.CreateStaticFrictionAttr(1.);api.CreateDynamicFrictionAttr(1.);api.CreateRestitutionAttr(0.)
    physx=PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim());physx.CreateFrictionCombineModeAttr('multiply');physx.CreateRestitutionCombineModeAttr('max')
    root=stage.GetPrimAtPath(robot_path)
    if not root.IsValid():raise ValueError('Missing RMUC source robot prim')
    found={n:[] for n in ['left_wheel','right_wheel']}
    for p in Usd.PrimRange(root):
        if not p.HasAPI(UsdPhysics.CollisionAPI):continue
        parts=str(p.GetPath()).split('/')
        for wheel in found:
            if wheel in parts:
                UsdShade.MaterialBindingAPI.Apply(p).Bind(material,UsdShade.Tokens.strongerThanDescendants,'physics')
                found[wheel].append(str(p.GetPath()))
    if any(len(v)!=1 for v in found.values()):raise ValueError(f'Expected one explicit cylinder per driven wheel: {found}')
    return dict(material_path=material_path,source_colliders=found,friction_combine_mode='multiply',restitution_combine_mode='max',wheel_coefficients=[1.,1.,0.])


# The shared terrain cannot carry a different material for each cloned env.
# This opt-in curriculum therefore factors the desired wheel/ground pair onto
# wheel-only shapes, keeping the actual terrain and other body contacts intact.
_FIELDS = ('effective_static_friction', 'effective_dynamic_friction', 'effective_restitution')
_COMBINE_ORDER = {'average': 0, 'min': 1, 'multiply': 2, 'max': 3}


def combine_coefficient(a, b, mode_a='average', mode_b='average'):
    """PhysX uses the higher-priority combine mode of the two materials."""
    if mode_a not in _COMBINE_ORDER or mode_b not in _COMBINE_ORDER:
        raise ValueError('Unknown PhysX material combine mode')
    mode = max((mode_a, mode_b), key=_COMBINE_ORDER.__getitem__)
    return {'average': lambda: (a+b)/2, 'min': lambda: min(a, b),
            'multiply': lambda: a*b, 'max': lambda: max(a, b)}[mode]()


def curriculum_parameters(path, progress, severity, baseline):
    """Interpolate effective contact coefficients, never raw material factors."""
    if not math.isfinite(progress) or not 0 <= progress <= 1:
        raise ValueError('Arena migration progress must be in [0,1]')
    if len(baseline) != 3 or not all(math.isfinite(float(x)) for x in baseline):
        raise ValueError('Invalid baseline contact coefficients')
    if not 0 <= baseline[1] <= baseline[0] or not 0 <= baseline[2] <= 1:
        raise ValueError('Invalid baseline contact ordering')
    target = arena_parameters(path, severity)
    return tuple(float(baseline[i]) + progress*(target[k]-baseline[i]) for i, k in enumerate(_FIELDS))


def factor_wheel_coefficients(effective, ground):
    """Wheel multiply friction / max restitution with unchanged ground."""
    if len(effective) != 3 or len(ground) != 3:
        raise ValueError('Expected static/dynamic/restitution triplets')
    if not all(math.isfinite(float(x)) for x in (*effective, *ground)):
        raise ValueError('Material values must be finite')
    if ground[0] <= 0 or ground[1] <= 0 or effective[2] < ground[2]-1e-7:
        raise ValueError('Ground cannot represent requested pair coefficients')
    values = (effective[0]/ground[0], effective[1]/ground[1], effective[2])
    if values[1] > values[0]+1e-7:
        raise ValueError('Pair factorization would violate dynamic <= static')
    return values


def prepare_curriculum_ground(cfg):
    if cfg.arena_ground_severity >= 0:
        raise ValueError('Fixed arena severity and per-env curriculum are mutually exclusive')
    if cfg.friction_randomization_enabled or cfg.domain_randomization_max_friction_fraction != 0:
        raise ValueError('Arena curriculum requires duplicate legacy friction DR disabled')
    progress = float(cfg.arena_ground_migration_progress)
    low, high = (float(x) for x in cfg.arena_ground_severity_range)
    curriculum_parameters(cfg.arena_ground_profile_path, progress, low, (.5, .5, 0.))
    arena_parameters(cfg.arena_ground_profile_path, high)
    if low > high:
        raise ValueError('Arena severity range must be ordered')
    if not cfg.arena_ground_randomization_enabled and low != high:
        raise ValueError('Disabled arena sampling requires a fixed severity range')
    material = cfg.terrain.physics_material
    ground = (float(material.static_friction), float(material.dynamic_friction), float(material.restitution))
    ground_modes = (getattr(material, 'friction_combine_mode', None) or 'average',
                    getattr(material, 'restitution_combine_mode', None) or 'average')
    if ground_modes[0] == 'max':
        raise ValueError('Ground max friction overrides wheel multiply factorization')
    factor_wheel_coefficients(curriculum_parameters(cfg.arena_ground_profile_path, progress, high, ground), ground)
    return {'mode': 'per_env_wheel_ground_pair_curriculum', 'migration_progress': progress,
            'severity_range': [low, high], 'randomization_enabled': bool(cfg.arena_ground_randomization_enabled),
            'ground_coefficients': list(ground), 'ground_combine_modes': list(ground_modes),
            'terrain_material_changed': False, 'sample_scope': 'one correlated severity per env at reset; both driven wheels',
            'engineering_prior_only': True, 'behavior_certified': False}


def _read_material(material):
    from pxr import UsdPhysics, PhysxSchema
    if not material or not material.GetPrim().IsValid():
        raise ValueError('Missing explicit collision physics material binding')
    api = UsdPhysics.MaterialAPI(material.GetPrim())
    values = [api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get(), api.GetRestitutionAttr().Get()]
    if any(x is None for x in values):
        raise ValueError(f'Missing material coefficient at {material.GetPath()}')
    physx = PhysxSchema.PhysxMaterialAPI(material.GetPrim())
    modes = [physx.GetFrictionCombineModeAttr().Get() or 'average',
             physx.GetRestitutionCombineModeAttr().Get() or 'average']
    return tuple(float(x) for x in values), tuple(str(x) for x in modes)


def configured_default_material(cfg):
    """Unbound collider uses the simulation's configured default material."""
    material = cfg.sim.physics_material
    values = tuple(float(getattr(material, k)) for k in ('static_friction', 'dynamic_friction', 'restitution'))
    if not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError('Invalid simulation default physics material')
    modes = tuple(getattr(material, k, None) or 'average' for k in ('friction_combine_mode', 'restitution_combine_mode'))
    return values, modes


def bind_curriculum_wheels(stage, cfg, robot_path='/World/envs/env_0/Robot'):
    """Call after source Articulation spawn, before cloning and PhysX start."""
    from pxr import Usd, UsdPhysics, UsdShade, PhysxSchema
    audit = prepare_curriculum_ground(cfg)
    ground, ground_modes = audit['ground_coefficients'], audit['ground_combine_modes']
    root = stage.GetPrimAtPath(robot_path)
    if not root.IsValid():
        raise ValueError('Missing RMUC source robot prim')
    found = {name: [] for name in ('left_wheel', 'right_wheel')}
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            for name in found:
                if name in str(prim.GetPath()).split('/'):
                    found[name].append(prim)
    if any(len(x) != 1 for x in found.values()):
        raise ValueError('Expected exactly one collision shape for each driven wheel')
    audit['wheels'] = {}
    for name, prims in found.items():
        prim = prims[0]
        original, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        if original and original.GetPrim().IsValid():
            old, old_modes = _read_material(original)
            original_path = str(original.GetPath())
        else:
            old, old_modes = configured_default_material(cfg)
            original_path = 'simulation_default:cfg.sim.physics_material'
        baseline = tuple(combine_coefficient(old[i], ground[i], old_modes[i//2], ground_modes[i//2]) for i in range(3))
        initial = factor_wheel_coefficients(baseline, ground)
        path = '/World/RmucArenaCurriculum_' + name
        material = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(initial[0]); api.CreateDynamicFrictionAttr(initial[1]); api.CreateRestitutionAttr(initial[2])
        physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx.CreateFrictionCombineModeAttr('multiply'); physx.CreateRestitutionCombineModeAttr('max')
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
        audit['wheels'][name] = {'collider': str(prim.GetPath()), 'original_material': original_path,
            'original_coefficients': list(old), 'original_combine_modes': list(old_modes),
            'baseline_effective': list(baseline), 'material_path': path, 'initial_coefficients': list(initial)}
    return audit


def bind_curriculum_clones(stage, cfg, audit):
    """Bind every cloned collider before PhysX starts; env0 alone is insufficient."""
    from pxr import UsdShade
    count = 0
    for env_id in range(cfg.scene.num_envs):
        for meta in audit['wheels'].values():
            path = meta['collider'].replace('/env_0/', f'/env_{env_id}/', 1)
            collider = stage.GetPrimAtPath(path)
            if not collider.IsValid():
                raise ValueError(f'Missing cloned wheel collision: {path}')
            material = UsdShade.Material.Get(stage, meta['material_path'])
            UsdShade.MaterialBindingAPI.Apply(collider).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
            resolved, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial('physics')
            if not resolved or str(resolved.GetPath()) != meta['material_path']:
                raise ValueError(f'Clone material binding failed: {path}')
            count += 1
    audit['cloned_colliders_bound'] = count


class ArenaGroundRandomizer:
    """Reset-time pair DR with runtime PhysX shape mapping and readback."""
    def __init__(self, env):
        import torch
        import warp as wp
        from pxr import Usd, UsdPhysics, UsdShade
        from isaaclab.sim.utils import get_current_stage
        self.env, self.cfg, self.wp, self.torch = env, env.cfg, wp, torch
        self.audit = env.arena_ground_audit
        if self.audit is None or self.audit.get('mode') != 'per_env_wheel_ground_pair_curriculum':
            raise ValueError('Arena curriculum must be bound before physics initialization')
        self.ground = tuple(self.audit['ground_coefficients'])
        stage = get_current_stage()
        ground_shapes = [p for p in Usd.PrimRange(stage.GetPrimAtPath(self.cfg.terrain.prim_path))
                         if p.HasAPI(UsdPhysics.CollisionAPI)]
        if not ground_shapes:
            raise ValueError('No terrain collision shape found for material readback')
        for prim in ground_shapes:
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            values, modes = _read_material(material)
            if any(abs(x-y) > 1e-6 for x, y in zip(values, self.ground)) or list(modes) != self.audit['ground_combine_modes']:
                raise ValueError('Actual terrain material disagrees with pair factorization')
        self.shape_indices = {}
        offset = 0
        for path in env.robot.root_view.link_paths[0]:
            count = env.robot._physics_sim_view.create_rigid_body_view(path).max_shapes
            name = str(path).rsplit('/', 1)[-1]
            if name in self.audit['wheels']:
                if count != 1:
                    raise ValueError(f'Expected one runtime wheel shape: {name} has {count}')
                self.shape_indices[name] = offset
            offset += count
        if offset != env.robot.root_view.max_shapes or set(self.shape_indices) != set(self.audit['wheels']):
            raise ValueError('Runtime shape mapping does not cover the two driven wheels exactly')
        self.audit['runtime_shape_indices'] = dict(self.shape_indices)
        # Bound the material palette; repeated resets cannot create unbounded
        # PhysX unique materials. Correlate static/dynamic and the two wheels.
        low, high = self.audit['severity_range']
        self.palette_severity = torch.linspace(low, high, 64 if low != high else 1)
        self.palette = {}
        for name, meta in self.audit['wheels'].items():
            self.palette[name] = torch.tensor([factor_wheel_coefficients(curriculum_parameters(
                self.cfg.arena_ground_profile_path, self.audit['migration_progress'], float(s), meta['baseline_effective']), self.ground)
                for s in self.palette_severity], dtype=torch.float32)
        current = self._materials()
        for name, idx in self.shape_indices.items():
            expected = torch.tensor(self.audit['wheels'][name]['initial_coefficients'])
            if not torch.allclose(current[:, idx], expected.expand_as(current[:, idx]), atol=1e-6, rtol=0.):
                raise ValueError(f'Initial wheel material mismatch: wheel={name},shape={idx},'
                    f'expected={expected.tolist()},actual={current[:,idx].tolist()},'
                    f'env0_shapes={current[0].tolist()},audit={self.audit}')
        self.expected = current.clone()
        self.severity = torch.zeros(env.num_envs)
        self.reset_counts = torch.zeros(env.num_envs, dtype=torch.long)
        self.write_readback_max_abs_error = 0.
        self.audit['runtime_verified'] = False

    def _materials(self):
        return self.wp.to_torch(self.env.robot.root_view.get_material_properties()).detach().cpu().clone()

    def reset(self, env_ids):
        torch, wp = self.torch, self.wp
        ids = torch.as_tensor(env_ids, dtype=torch.long, device='cpu')
        if ids.numel() == 0:
            return
        bucket = torch.randint(len(self.palette_severity), (len(ids),)) if self.audit['randomization_enabled'] else torch.zeros(len(ids), dtype=torch.long)
        material = self._materials()
        before = material.clone()
        for name, idx in self.shape_indices.items():
            material[ids, idx] = self.palette[name][bucket]
        self.env.robot.root_view.set_material_properties(wp.from_torch(material, dtype=wp.float32),
            wp.from_torch(ids.to(dtype=torch.int32), dtype=wp.int32))
        actual = self._materials()
        error = float((actual[ids]-material[ids]).abs().max())
        if error > 1e-6:
            raise RuntimeError(f'Arena material write/readback mismatch: {error}')
        self.write_readback_max_abs_error = max(self.write_readback_max_abs_error, error)
        # Preserve the environment's existing generic material readback audit.
        if self.env._domain_material_properties is not None:
            self.env._domain_material_properties[ids] = actual[ids]
        untouched = [i for i in range(material.shape[1]) if i not in self.shape_indices.values()]
        if not torch.allclose(actual[ids][:, untouched], before[ids][:, untouched], atol=1e-6, rtol=0.):
            raise RuntimeError('Arena DR changed a non-wheel collision shape')
        self.expected[ids] = actual[ids]
        self.severity[ids] = self.palette_severity[bucket]
        self.reset_counts[ids] += 1
        self.audit['runtime_verified'] = True

    def readback(self):
        material = self._materials()
        wheels = {}
        for name, idx in self.shape_indices.items():
            values = material[:, idx]
            effective = values.clone()
            effective[:, :2] *= self.torch.tensor(self.ground[:2])
            effective[:, 2] = self.torch.maximum(effective[:, 2], self.torch.tensor(self.ground[2]))
            wheels[name] = {'shape_index': idx, 'coefficients_per_env': values.tolist(),
                            'effective_pair_per_env': effective.tolist()}
        return {**self.audit, 'wheels_readback': wheels,
                'palette_size': len(self.palette_severity), 'severity_per_env': self.severity.tolist(),
                'reset_counts': self.reset_counts.tolist(),
                'write_readback_max_abs_error': self.write_readback_max_abs_error,
                'max_abs_error_vs_expected': float((material-self.expected).abs().max())}
