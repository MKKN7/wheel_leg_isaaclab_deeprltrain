"""Exercise the actual environment's reset-time mass/inertia write path on CPU."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'


def reset_prefix():
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == '_apply_progressive_domain_randomization')
    end = next(i for i, n in enumerate(fn.body)
               if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Attribute)
               and n.value.func.attr == 'set_inertias_index')
    fn.body = fn.body[:end+1]
    namespace = {'torch': torch, 'math': math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 str(SOURCE), 'exec'), namespace)
    return namespace[fn.name]


def environment(inertia_fraction, mass_fraction=.05):
    n = 32
    writes = {}
    env = NS(device='cpu', _root_body_id=0, _motor_responses={}, _sensor_model_enabled=False,
             _progressive_domain_randomization_strength=lambda: .5,
             _domain_randomization_strength=torch.zeros(n),
             _domain_nominal_root_mass=torch.ones(n)*10,
             _domain_nominal_root_inertia=torch.diag(torch.tensor([2., 3., 4.])).flatten().repeat(n, 1),
             cfg=NS(leg_actuator_dr_enabled=False,
                    domain_randomization_max_base_mass_fraction=mass_fraction,
                    domain_randomization_max_base_inertia_fraction=inertia_fraction,
                    domain_randomization_max_friction_fraction=.15,
                    domain_randomization_max_wheel_damping_fraction=.1,
                    domain_randomization_max_wheel_torque_gain_fraction=.05,
                    domain_randomization_max_observation_noise_std=0.),
             robot=NS(set_masses_index=lambda **kw: writes.update(mass=kw['masses']),
                      set_inertias_index=lambda **kw: writes.update(inertia=kw['inertias'])))
    for name in ('base_mass_scale', 'base_inertia_shape_scale', 'base_inertia_scale',
                 'friction_scale', 'wheel_damping_scale', 'wheel_torque_gain', 'observation_noise_std'):
        setattr(env, '_domain_'+name, torch.ones(n))
    return env, writes


def test_disabled_inertia_preserves_legacy_draws_and_mass_proportional_tensor():
    torch.manual_seed(435)
    expected = [torch.empty(32).uniform_(1-f*.5, 1+f*.5) for f in (.05, .15, .1, .05)]
    rng = torch.get_rng_state().clone()
    torch.manual_seed(435)
    env, writes = environment(0.)
    reset_prefix()(env, torch.arange(32))
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in zip(('base_mass_scale', 'friction_scale', 'wheel_damping_scale', 'wheel_torque_gain'), expected):
        assert torch.equal(getattr(env, '_domain_'+name), value)
    assert torch.equal(writes['inertia'].view(32, 9), env._domain_nominal_root_inertia*expected[0][:, None])


def test_inertia_varies_without_mass_change_and_preserves_physical_tensor():
    torch.manual_seed(435)
    env, writes = environment(.2, mass_fraction=0.)
    reset_prefix()(env, torch.arange(32))
    assert torch.equal(writes['mass'], torch.full((32, 1), 10.))
    scale = env._domain_base_inertia_shape_scale
    assert scale.std() > 0 and ((scale >= .9) & (scale <= 1.1)).all()
    tensor = writes['inertia'].view(32, 3, 3)
    assert torch.equal(tensor, tensor.transpose(1, 2))
    moments = torch.linalg.eigvalsh(tensor)
    assert (moments > 0).all()
    assert (moments[:, 2] <= moments[:, :2].sum(1)).all()


@pytest.mark.parametrize('fraction', [-1., 1., float('nan'), float('inf')])
def test_invalid_inertia_rejected_before_physics_write(fraction):
    env, writes = environment(fraction)
    with pytest.raises(ValueError, match='inertia'):
        reset_prefix()(env, torch.arange(32))
    assert not writes
