import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('commands',ROOT/'scripts/rmuc_eval_commands.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_ramp_preserves_final_targets_and_restarts_only_reset_environments():
    target=torch.tensor([[2.95,0.,0.],[3.,0.,0.]])
    before=target.clone()
    torch.testing.assert_close(m.ramp_straight_commands(target,torch.tensor([125,125]),.008,2.)[:,0],torch.tensor([1.975,2.]))
    assert torch.equal(m.ramp_straight_commands(target,torch.tensor([250,1000]),.008,2.),target)
    output=m.ramp_straight_commands(target,torch.tensor([0,1000]),.008,2.)
    assert output[0,0]==1 and output[1,0]==3 and torch.equal(target,before)


def test_actual_observation_hook_updates_command_before_single_observation_call():
    tree=ast.parse((ROOT/'scripts/evaluate_rmuc_grid.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    node=next(n for n in main.body if isinstance(n,ast.If) and ast.unparse(n.test)=='args.diagnostic_speed_ramp_seconds')
    count=[];u=SimpleNamespace(_curriculum_step_counter=torch.tensor([0,250]),step_dt=.008)
    u.set_manual_command=lambda c:setattr(u,'manual',c.clone())
    def observe():count.append(1);return {'policy':u.manual.clone()}
    u._get_observations=observe
    scope={'u':u,'command':torch.tensor([[3.,0.,0.],[3.,0.,0.]]),
        'args':SimpleNamespace(diagnostic_speed_ramp_seconds=2.),'ramp_straight_commands':m.ramp_straight_commands}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual ramp hook>','exec'),scope)
    obs=u._get_observations();assert len(count)==1
    assert torch.equal(obs['policy'][:,0],torch.tensor([1.,3.]))


@pytest.mark.parametrize('forward,reverse',[(2.,0.),(0.,.3),(0.,.5)])
def test_ramp_diagnostic_cannot_certify_fixed_command_policy(forward,reverse):
    tree=ast.parse((ROOT/'scripts/evaluate_rmuc_grid.py').read_text())
    d=next(n for n in ast.walk(tree) if isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='diagnostic_dr_ablation' for k in n.keys))
    expr=next(v for k,v in zip(d.keys,d.values) if isinstance(k,ast.Constant) and k.value=='passed')
    args=SimpleNamespace(diagnostic_plane_terrain=False,diagnostic_total_residual_budget=None,disable_high_speed_feedback=False,
        disable_stage_head=None,base_wheel_clip=None,radial_spring_force=None,gas_spring_torque=None,
        dr_ablation='none',audit_critic=False,action_bias_sweep=None,pitch_rate_feedback=None,
        diagnostic_speed_ramp_seconds=forward,diagnostic_reverse_ramp_seconds=reverse,
        world_model_action_noise_std=0.,diagnostic_wheel_peak_nm=None)
    scope={'args':args,'rows':[{'passed':True}],'sequence_table':None}
    assert not eval(compile(ast.Expression(expr),'<report>','eval'),scope)
    args.diagnostic_speed_ramp_seconds=0.;args.diagnostic_reverse_ramp_seconds=0.
    assert eval(compile(ast.Expression(expr),'<report>','eval'),scope)


@pytest.mark.parametrize('target',[[[.5,0,0]],[[3,0,.3]]])
def test_out_of_scope_ramp_rejected(target):
    with pytest.raises(ValueError):m.ramp_straight_commands(torch.tensor(target),torch.tensor([0]),.008,2.)


def test_reverse_ramp_applies_to_both_low_and_high_reverse_without_touching_turns():
    target=torch.tensor([[-.6,0.,0.],[-1.5,0.,0.],[1.5,0.,.8],[0.,0.,0.]])
    before=target.clone()
    out=m.ramp_reverse_commands(target,torch.tensor([0,25,0,0]),.008,.4)
    torch.testing.assert_close(out[:,0],torch.tensor([0.,-.75,1.5,0.]))
    assert torch.equal(out[2:],before[2:]) and torch.equal(target,before)
    assert torch.equal(m.ramp_reverse_commands(target,torch.tensor([50,100,0,0]),.008,.4),target)


def test_reverse_hook_has_single_observation_call_and_seeds_zero_command_at_reset():
    tree=ast.parse((ROOT/'scripts/evaluate_rmuc_grid.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    node=next(n for n in main.body if isinstance(n,ast.If) and ast.unparse(n.test)=='args.diagnostic_reverse_ramp_seconds')
    calls=[];u=SimpleNamespace(_curriculum_step_counter=torch.tensor([0,63]),step_dt=.008)
    u.set_manual_command=lambda c:setattr(u,'manual',c.clone())
    def observe():calls.append(1);return u.manual.clone()
    u._get_observations=observe
    scope={'u':u,'command':torch.tensor([[-.6,0.,0.],[-.6,0.,0.]]),'args':SimpleNamespace(diagnostic_reverse_ramp_seconds=.5)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual reverse ramp>','exec'),scope)
    obs=u._get_observations();assert len(calls)==1
    torch.testing.assert_close(obs[:,0],torch.tensor([0.,-.6]))
    u._curriculum_step_counter=torch.tensor([63,0]);obs=u._get_observations()
    torch.testing.assert_close(obs[:,0],torch.tensor([-.6,0.]))
