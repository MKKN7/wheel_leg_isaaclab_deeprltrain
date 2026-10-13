"""Execute the real read-only observer, including IdealPD's in-place target clearing."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch

def test_observer_captures_target_before_compute_and_uses_actuator_joint_order():
    path=Path(__file__).resolve().parents[1]/'scripts/evaluate_rmuc_grid.py'
    tree=ast.parse(path.read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='audited_leg_compute')
    actuator=SimpleNamespace(stiffness=torch.full((2,4),60.),effort_limit=torch.full((2,4),40.))
    def compute(control,pos,vel):
        actuator.computed_effort=60*(control.joint_positions-pos)-2*vel
        actuator.applied_effort=actuator.computed_effort.clamp(-40,40)
        control.joint_efforts=actuator.applied_effort.clone()
        control.joint_positions=None
        return control
    audit={'physics_calls':0,'max_abs_target_position_rad':torch.zeros(4),'max_abs_speed_rad_s':torch.zeros(4),
           'steady_max_abs_speed_rad_s':torch.zeros(4),'samples_above_17_rad_s':torch.zeros(4),
           'steady_physics_samples_per_env':torch.zeros(2)}
    names=['target_position_rad','actual_position_rad','absolute_position_error_rad',
           'proportional_effort_nm','computed_effort_nm','applied_effort_nm',
           'torque_saturation_fraction','action_saturation_fraction','actual_velocity_rad_s',
           'target_velocity_rad_s','feedforward_effort_nm','position_difference_velocity_rad_s','velocity_consistency_abs_error_rad_s']
    sums={name:torch.zeros(2,4) for name in names}
    env=SimpleNamespace(_curriculum_step_counter=torch.tensor([1200,1199]),
                        _policy_actions=torch.tensor([[0.,0.,0.,1.,0.,0.],[1.,1.,1.,1.,1.,1.]]))
    scope=dict(torch=torch,leg_audit=audit,leg_sums=sums,leg_actuator=actuator,
               original_leg_compute=compute,u=env,leg_policy_slots=[3,4,0,1],cfg=SimpleNamespace(sim=SimpleNamespace(dt=.002)))
    factory=ast.parse('def factory():\n    previous_leg_position=None\n    previous_leg_age=None\n').body[0]
    factory.body += [fn] + ast.parse('return audited_leg_compute').body
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory],type_ignores=[])),str(path),'exec'),scope)
    scope['audited_leg_compute']=scope['factory']()
    control=SimpleNamespace(joint_positions=torch.full((2,4),.2),joint_velocities=torch.zeros(2,4),joint_efforts=torch.zeros(2,4))
    result=scope['audited_leg_compute'](control,torch.full((2,4),.5),torch.zeros(2,4))
    assert result is control and result.joint_positions is None
    torch.testing.assert_close(result.joint_efforts,torch.full((2,4),-18.))
    torch.testing.assert_close(sums['target_position_rad'][0],torch.full((4,),.2))
    torch.testing.assert_close(sums['absolute_position_error_rad'][0],torch.full((4,),.3))
    assert all(torch.count_nonzero(v[1])==0 for v in sums.values())
    torch.testing.assert_close(sums['action_saturation_fraction'][0],torch.tensor([1.,0.,0.,0.]))
    assert torch.count_nonzero(sums['torque_saturation_fraction'])==0
