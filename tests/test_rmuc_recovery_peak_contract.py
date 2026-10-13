"""Exercise actual saved-config restore and consumer expressions across peak contracts."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]

def main_nodes(relative):
    tree=ast.parse((ROOT/relative).read_text())
    return next((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'),tree)

@pytest.mark.parametrize('peak',[3.,5.5])
def test_saved_wheel_and_leg_mechanics_override_new_defaults(peak):
    main=main_nodes('scripts/evaluate_rmuc_recovery.py')
    selected=[]
    for node in ast.walk(main):
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0]) in ('cfg.wheel_torque_scale_nm','cfg.wheel_authored_effort_limit_nm'):
            selected.append(node)
        if isinstance(node,ast.For) and ast.unparse(node.target)=='group' and ast.unparse(node.iter)=="('wheel_motors', 'leg_motors')":
            selected.append(node)
    assert len(selected)==3
    cfg=NS(wheel_torque_scale_nm=5.5,wheel_authored_effort_limit_nm=5.5,
           robot_cfg=NS(actuators={name:NS(stiffness=0,damping=0,effort_limit=5.5) for name in ('wheel_motors','leg_motors')}))
    saved=dict(wheel_torque_scale_nm=str(peak),wheel_authored_effort_limit_nm=str(peak),robot_cfg=dict(actuators=dict(
        wheel_motors=dict(stiffness='0',damping='.031',effort_limit=str(peak)),
        leg_motors=dict(stiffness='60',damping='2',effort_limit='40'))))
    exec(compile(ast.Module(body=selected,type_ignores=[]),'actual_restore','exec'),dict(cfg=cfg,saved=saved))
    assert cfg.wheel_torque_scale_nm==peak and cfg.robot_cfg.actuators['wheel_motors'].effort_limit==peak
    assert cfg.robot_cfg.actuators['wheel_motors'].damping==.031
    assert cfg.robot_cfg.actuators['leg_motors'].stiffness==60 and cfg.robot_cfg.actuators['leg_motors'].effort_limit==40

@pytest.mark.parametrize('peak',[3.,5.5])
def test_recovery_consumer_uses_recorded_peak_without_hidden_three_clip(peak):
    main=main_nodes('sim2real_RL/sim2sim/run_recovery_onnx.py')
    selected=[n for n in ast.walk(main) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) in ('wheel_target','effort[[2, 5]]')]
    assert len(selected)==2
    scope=dict(np=np,contract=dict(wheel_action_sign=[-1.,-1.],wheel_explicit_damping=.0225),
               wheel_peak=peak,wheel_scale=peak,action=np.array([0.,0.,.8,0.,0.,-.8]),
               data=NS(qvel=np.zeros(6)),vids=np.arange(6),effort=np.zeros(6))
    exec(compile(ast.Module(body=selected,type_ignores=[]),'actual_consumer','exec'),scope)
    np.testing.assert_allclose(scope['effort'][[2,5]],[-.8*peak,.8*peak],atol=1e-12)
    if peak==5.5:assert np.all(abs(scope['effort'][[2,5]])>3.)

@pytest.mark.parametrize('peak,applied,rejected',[(3.,4.,True),(5.5,4.,False),(5.5,5.6,True)])
def test_native_audit_uses_configured_peak_and_rejects_excess(peak,applied,rejected):
    import torch
    tree=ast.parse((ROOT/'scripts/evaluate_rmuc_recovery.py').read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='audit_wheel_motor')
    scope=dict(torch=torch,closed=torch.tensor([False]),wheel_motor_peak=torch.zeros(1,2),
               wheel_motor_samples=torch.zeros(1,dtype=torch.long),
               u=NS(_wheel_control_ids=[2,5],robot=NS(
                   data=NS(applied_torque=NS(torch=torch.tensor([[0.,0.,applied,0.,0.,-applied]]))),
                   actuators={'wheel_motors':NS(effort_limit=torch.full((1,2),peak))})))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_motor_audit','exec'),scope)
    if rejected:
        with pytest.raises(AssertionError):scope['audit_wheel_motor']()
    else:
        scope['audit_wheel_motor']()
        assert scope['wheel_motor_peak'].tolist()==[[applied,applied]]
