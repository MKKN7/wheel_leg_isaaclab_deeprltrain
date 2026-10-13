"""Execute the real recovery gate without simulator imports."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

source=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/recovery_env.py'
tree=ast.parse(source.read_text())
fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_progressive_domain_randomization_strength')
namespace={'math':math}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),namespace)
gate=namespace[fn.name]


@pytest.mark.parametrize('strength,expected',[(-1.,0.),(0.,0.),(.2,.2),(1.,1.)])
def test_recovery_strength_is_independent_of_command_stage(strength,expected):
    cfg=NS(curriculum_stage=0,domain_randomization_enabled=True,play=False,
           domain_randomization_strength_override=strength,domain_randomization_apply_in_play=False)
    assert gate(NS(cfg=cfg))==expected
    cfg.curriculum_stage=1
    assert gate(NS(cfg=cfg))==expected
    cfg.domain_randomization_enabled=False
    assert gate(NS(cfg=cfg))==0.


def test_play_requires_explicit_randomization():
    cfg=NS(domain_randomization_enabled=True,play=True,domain_randomization_apply_in_play=False,
           domain_randomization_strength_override=1.)
    assert gate(NS(cfg=cfg))==0.
    cfg.domain_randomization_apply_in_play=True
    assert gate(NS(cfg=cfg))==1.


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-2.,1.01])
def test_invalid_strength_rejected(value):
    with pytest.raises(ValueError):gate(NS(cfg=NS(domain_randomization_strength_override=value)))
