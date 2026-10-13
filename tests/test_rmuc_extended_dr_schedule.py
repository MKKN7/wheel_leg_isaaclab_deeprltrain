"""Execute the real environment schedule without launching Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]


def schedule(level, fraction=0., override=-1., play=False, enabled=True):
    config=yaml.safe_load((ROOT/'configs/rmuc_wheel_leg.yaml').read_text())
    dr=config['domain_randomization']
    tree=ast.parse((ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RmucWheelLegEnv')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_progressive_domain_randomization_strength')
    namespace={}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual DR schedule>','exec'),namespace)
    cfg=SimpleNamespace(domain_randomization_enabled=enabled,curriculum_stage=1,play=play,
        domain_randomization_apply_in_play=False,domain_randomization_strength_override=override,
        domain_randomization_start_level=dr['start_command_level'],
        domain_randomization_full_level=dr['full_command_level'],
        domain_randomization_extended_start_level=dr['extended_start_command_level'],
        domain_randomization_extended_full_level=dr['extended_full_command_level'])
    env=SimpleNamespace(cfg=cfg,_command_level=level,_global_step_counter=1000*fraction,
        _command_level_start_step=0,_current_command_transition_steps=lambda:1000)
    return namespace[method.name](env)


@pytest.mark.parametrize('fraction',[0.,.5,1.])
def test_all_legacy_levels_keep_the_original_schedule(fraction):
    for level in range(47):
        expected=0 if level<29 else min((level-29+1+fraction)/19,1.)
        assert schedule(level,fraction)==expected


def test_extension_starts_mild_and_progresses_to_full_without_a_premature_cap():
    cfg=yaml.safe_load((ROOT/'configs/rmuc_wheel_leg.yaml').read_text())
    first=cfg['domain_randomization']['extended_start_command_level']
    last=cfg['domain_randomization']['extended_full_command_level']
    assert first==47 and last==len(cfg['command']['stages'])-1
    values=[schedule(level) for level in range(first,last+1)]
    assert 0<values[0]<=.05
    assert all(a<b for a,b in zip(values,values[1:]))
    assert schedule(last,1.)==1.


def test_explicit_evaluation_strength_and_disabled_paths_override_both_schedules():
    for level in [18,46,47,70]:
        assert schedule(level,override=1.)==1.
        assert schedule(level,override=.05)==.05
        assert schedule(level,override=1.,play=True)==0.
        assert schedule(level,override=1.,enabled=False)==0.
