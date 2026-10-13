"""Execute the production action path without Isaac or any privileged state."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]

def test_e2e_bypass_is_exact_clipped_actor_and_never_reads_state():
    path = ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RmucWheelLegEnv')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_compute_policy_action')
    ns = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), 'exec'), ns)
    env = SimpleNamespace(cfg=SimpleNamespace(action_clip=1., policy_outer_feedback_enabled=False),
                          _last_raw_policy_actions=torch.zeros(2,6))
    names = ['_last_speed_feedback_action', '_last_highspeed_propulsion_action',
             '_last_pitch_feedback_action', '_last_pitch_emergency_feedback_action',
             '_last_yaw_feedback_action', '_last_action_feedback_delta']
    for name in names:
        setattr(env, name, torch.ones(2,6) if name.endswith('delta') else torch.ones(2))
    # No _state(), command, or robot exists: touching privileged state fails.
    actions = torch.tensor([[.1,-.2,2.,-3.,.6,-.7],[float('nan'),float('inf'),float('-inf'),0.,1.,-1.]])
    expected = torch.nan_to_num(actions, nan=0., posinf=1., neginf=-1.).clamp(-1,1)
    actual = ns['_compute_policy_action'](env, actions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(env._last_raw_policy_actions, expected, rtol=0, atol=0)
    assert all(torch.count_nonzero(getattr(env, name)) == 0 for name in names)
