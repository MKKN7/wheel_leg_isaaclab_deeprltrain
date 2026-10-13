import ast
import math
from pathlib import Path
import runpy
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]
module=runpy.run_path(str(ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/bank_geometry.py'))
bank_reference=module['bank_reference']

def test_bank_matches_independent_world_body_up_heading_geometry_with_pitch():
    torch.manual_seed(212)
    rotations,_=torch.linalg.qr(torch.randn(1000,3,3,dtype=torch.float64))
    rotations[:,:,0]*=torch.det(rotations)[:,None]
    world_g=torch.tensor([0.,0.,-1.],dtype=torch.float64)
    source=(rotations.transpose(1,2)@world_g.expand(1000,3).unsqueeze(-1)).squeeze(-1)
    gravity=torch.stack((-source[:,2],source[:,0],source[:,1]),dim=-1)
    forward=-rotations[:,:,2];up=rotations[:,:,1]
    h=forward[:,:2]/torch.linalg.vector_norm(forward[:,:2],dim=-1,keepdim=True)
    left=torch.stack((-h[:,1],h[:,0]),dim=-1)
    expected=torch.atan2((up[:,:2]*left).sum(-1),up[:,2])
    observed,_=bank_reference(gravity,torch.zeros(1000,3,dtype=torch.float64))
    torch.testing.assert_close(observed,expected,atol=1e-9,rtol=1e-9)

def test_actual_reward_prefers_physical_inward_over_outward_both_yaw_signs():
    helpers=runpy.run_path(str(ROOT/'tests/test_rmuc_reward_contract.py'))
    env=helpers['make_env']();reward=helpers['reward_function']()
    env.cfg.command_reward_contract_by_command=True
    g,v,w=env._state();cmd=env._current_command()
    cmd[:]=torch.tensor([1.,0.,1.]);cmd[1::2,2]=-1.
    v[:]=torch.tensor([1.,0.,0.]);w.zero_();w[:,2]=cmd[:,2]
    env._previous_command_aligned_speed.copy_(v[:,0])
    angle=torch.atan(cmd[:,2]/9.81)
    # At the authored +90deg X pose, physical inward for positive yaw is
    # world-X; its canonical projected gravity has negative lateral component.
    g[:,0]=0.;g[:,1]=-torch.sin(angle);g[:,2]=-torch.cos(angle)
    inner=reward(env)[0].clone()
    observed,target=bank_reference(g,cmd)
    torch.testing.assert_close(observed,target)
    g[:,1]*=-1.
    outer=reward(env)[0].clone()
    assert (inner>outer).all()
    observed,target=bank_reference(g,cmd)
    assert (torch.rad2deg((observed-target).abs())>11).all()

def test_actual_curriculum_bank_error_rejects_outward_and_accepts_inward():
    path=ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'
    tree=ast.parse(path.read_text())
    statements=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign):
            names={x.id for t in n.targets for x in ast.walk(t) if isinstance(x,ast.Name)}
            if names & {'command_roll_rad','command_bank_error_deg'}:statements.append(n)
    assert len(statements)==2
    statements.sort(key=lambda n:n.lineno)
    cmd=torch.tensor([[1.,0.,1.],[1.,0.,-1.]])
    angle=torch.atan(cmd[:,2]/9.81)
    g=torch.stack((torch.zeros(2),-torch.sin(angle),-torch.cos(angle)),dim=-1)
    scope={'torch':torch,'bank_reference':bank_reference,'gravity':g,'command':cmd}
    code=compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])),str(path),'exec')
    exec(code,scope);assert (scope['command_bank_error_deg']<1e-4).all()
    g[:,1]*=-1;exec(code,scope);assert (scope['command_bank_error_deg']>11).all()

def test_evaluator_uses_same_reference_as_reward_and_gate():
    path=ROOT/'scripts/evaluate_rmuc_grid.py'
    tree=ast.parse(path.read_text())
    assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='bank_reference']
    assert len(assignments)==1
    cmd=torch.tensor([[2.5,0.,1.]])
    g=torch.tensor([[0.,-.2,-math.sqrt(.96)]])
    scope={'torch':torch,'bank_reference':bank_reference,'gravity':g,'command':cmd}
    exec(compile(ast.fix_missing_locations(ast.Module(body=assignments,type_ignores=[])),str(path),'exec'),scope)
    expected,target=bank_reference(g,cmd)
    torch.testing.assert_close(scope['roll'],expected)
    torch.testing.assert_close(scope['target_roll'],target)


def test_actual_bank_penalty_has_bounded_dense_tail_and_preserves_near_target():
    path = ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'
    tree = ast.parse(path.read_text())
    statements = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and
                          t.id == 'command_bank_error_penalty_score' for t in n.targets)]
    statements.sort(key=lambda n: n.lineno)
    assert len(statements) == 3  # Includes the zero initialization before command mode.
    statements = statements[1:]
    errors = torch.tensor([0., 1., 3., 5., 6., 6.001, 8., 10., 11.9, 12., 18., 1000.])
    scope = {'torch': torch, 'self': SimpleNamespace(cfg=SimpleNamespace(command_bank_error_scale_deg=6.)),
             'command_bank_error_abs_deg': errors}
    exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
                 str(path), 'exec'), scope)
    actual = scope['command_bank_error_penalty_score']
    torch.testing.assert_close(actual[:5], errors[:5]/6., rtol=0, atol=0)
    assert ((actual >= 0) & (actual <= 2)).all()
    assert (actual[1:] > actual[:-1]).all()
    assert float(2*(actual[9]-actual[8])) > .01  # 12 -> 11.9 deg yields signal.
    assert abs(float((actual[5]-actual[4])/.001)-1/6) < .001
