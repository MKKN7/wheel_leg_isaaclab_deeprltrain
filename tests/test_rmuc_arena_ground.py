from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[1]
f=runpy.run_path(str(ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/arena_ground.py'))['arena_parameters']
PROFILE=ROOT/'configs/rmuc_arena_ground.json'

def test_correlated_endpoints_and_midpoint():
 a=f(PROFILE,0);b=f(PROFILE,1);c=f(PROFILE,.5)
 assert a['effective_static_friction']==.65 and a['effective_dynamic_friction']==.55
 assert b['effective_static_friction']==.45 and b['effective_dynamic_friction']==.38
 assert c['effective_static_friction']==pytest.approx(.55)
 assert c['effective_dynamic_friction']==pytest.approx(.465)
 assert c['effective_restitution']==.02

@pytest.mark.parametrize('s',[-.1,1.1,float('nan'),float('inf')])
def test_invalid_severity(s):
 with pytest.raises(ValueError):f(PROFILE,s)
