import ast,importlib.util,math
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/wheel_velocity_control.py'
spec=importlib.util.spec_from_file_location('wheel_controller',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_equivalent_p_matches_installed_ideal_pd_on_actual_signed_effort_and_speed():
    pd=Path('/home/kellen/桌面/IsaacLab-3.0/source/isaaclab/isaaclab/actuators/actuator_pd.py')
    base=pd.with_name('actuator_base.py')
    def method(file,cls_name,name):
        cls=next(n for n in ast.parse(file.read_text()).body if isinstance(n,ast.ClassDef) and n.name==cls_name)
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        ns={'torch':torch,'ArticulationActions':SimpleNamespace}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(file),'exec'),ns)
        return ns[name]
    compute=method(pd,'IdealPDActuator','compute');clip=method(base,'ActuatorBase','_clip_effort')
    torch.manual_seed(214)
    vel=torch.randn(100,2)*60;requested=torch.randn(100,2)*5.5
    actuator=SimpleNamespace(stiffness=torch.zeros(100,2),damping=torch.full((100,2),.1),
        effort_limit=torch.full((100,2),5.5),reset=lambda ids:None)
    actuator._clip_effort=lambda effort:clip(actuator,effort)
    action=lambda:SimpleNamespace(joint_positions=torch.zeros_like(vel),joint_velocities=torch.zeros_like(vel),joint_efforts=requested.clone())
    expected=compute(actuator,action(),torch.zeros_like(vel),vel).joint_efforts.clone()
    controller,audit=m.install_legacy_velocity_equivalence(actuator,.002)
    actual=actuator.compute(action(),torch.zeros_like(vel),vel).joint_efforts
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    assert audit['compute_calls']==1
    actuator.reset(torch.tensor([2]));assert not controller.initialized[2].any()
    assert controller.initialized[3].all()

def test_pi_antiwindup_three_nm_and_reversal():
    c=m.WheelVelocityController(torch.zeros(2,2),.002,.2,ki=2.)
    limit=torch.full((2,2),3.)
    for _ in range(1000):out=c.step(torch.full((2,2),70.),torch.zeros(2,2),limit)
    assert (out==3.).all() and c.integral_effort.abs().max()==0
    out=c.step(torch.full((2,2),-70.),torch.zeros(2,2),limit)
    assert (out==-3.).all()
    c.integral_effort.fill_(2.9)
    out=c.step(torch.zeros(2,2),torch.ones(2,2),limit)
    assert (c.integral_effort<2.9).all() and (out.abs()<=3).all()

def test_partial_reset_disable_filter_and_reduced_limit():
    c=m.WheelVelocityController(torch.zeros(2,2),.002,.2,ki=1.,filter_tau_s=.01,reference_limit_rad_s=70.)
    limit=torch.full((2,2),3.)
    c.step(torch.full((2,2),1.),torch.zeros(2,2),limit)
    c.step(torch.full((2,2),1.),torch.full((2,2),2.),limit)
    torch.testing.assert_close(c.filtered_velocity,torch.full((2,2),2*(1-math.exp(-.2))))
    c.reset(torch.tensor([0]));assert not c.initialized[0].any() and c.initialized[1].all()
    c.integral_effort.fill_(3.)
    out=c.step(torch.full((2,2),100.),torch.zeros(2,2),torch.full((2,2),.5),
        enabled=torch.tensor([[False],[True]]))
    assert out[0].count_nonzero()==0 and c.integral_effort[0].count_nonzero()==0
    assert (out.abs()<=.5).all() and c.reference[1,0]==70

@pytest.mark.parametrize('kp,ki,tau',[(0.,0.,0.),(.2,-1.,0.),(.2,0.,float('nan'))])
def test_bad_parameters_rejected(kp,ki,tau):
    with pytest.raises(ValueError):m.WheelVelocityController(torch.zeros(1,2),.002,kp,ki,tau)

def test_equivalence_diagnostic_cannot_certify_policy_even_if_rows_pass():
    tree=ast.parse((ROOT/'scripts/evaluate_rmuc_grid.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Dict) and any(
        isinstance(k,ast.Constant) and k.value=='diagnostic_dr_ablation' for k in n.keys))
    expression=next(v for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant) and k.value=='passed')
    assert not eval(compile(ast.Expression(expression),'<actual acceptance>','eval'),
        {'args':SimpleNamespace(diagnostic_wheel_velocity_equivalence=True),'rows':[{'passed':True}]})

def test_target_velocity_wrapper_uses_reference_after_delay_and_real3nm_cap():
    a=SimpleNamespace(stiffness=torch.zeros(2,2),damping=torch.zeros(2,2),
        effort_limit=torch.full((2,2),3.),reset=lambda ids:None)
    reference=torch.tensor([[40.,-40.],[1.,1.]])
    enabled=torch.tensor([[True],[False]])
    c,audit=m.install_wheel_velocity_controller(a,.002,lambda:reference,lambda:enabled,.2,.5)
    action=SimpleNamespace(joint_efforts=torch.full((2,2),100.),joint_positions=None,joint_velocities=None)
    output=a.compute(action,None,torch.zeros(2,2)).joint_efforts
    torch.testing.assert_close(output,torch.tensor([[3.,-3.],[0.,0.]]))
    assert torch.equal(output,a.applied_effort)
    reference[0]=torch.tensor([-1.,1.])
    output=a.compute(action,None,torch.zeros(2,2)).joint_efforts
    assert output[0,0]<0 and output[0,1]>0
    a.reset(torch.tensor([0]));assert c.integral_effort[0].count_nonzero()==0
    assert audit['max_applied_effort_nm']==3.
    with pytest.raises(ValueError):m.install_wheel_velocity_controller(a,.002,lambda:reference,lambda:enabled,.2)

def test_target_profile_validation_and_no_old_pd_double_count():
    import yaml,copy
    spec=importlib.util.spec_from_file_location('velocity_profile',ROOT/'scripts/rmuc_velocity_profile.py')
    profile_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile_module)
    profile=yaml.safe_load((ROOT/'configs/rmuc_wheel_velocity_migration.yaml').read_text())
    cfg=SimpleNamespace(sim=SimpleNamespace(dt=.002),wheel_torque_scale_nm=5.5,motor_response_enabled=False,robot_cfg=SimpleNamespace(actuators={
        'wheel_motors':SimpleNamespace(stiffness=0.,damping=.1,effort_limit=5.5)}))
    invalid=copy.deepcopy(profile);invalid['target_controller_proposal']['period_s']=.01
    with pytest.raises(ValueError):profile_module.apply_velocity_profile(cfg,invalid)
    assert cfg.robot_cfg.actuators['wheel_motors'].effort_limit==5.5
    result=profile_module.apply_velocity_profile(cfg,profile)
    assert result['output_effort_limit_nm']==3 and result['legacy_action_reference_scale_rad_s']==55
    assert cfg.robot_cfg.actuators['wheel_motors'].effort_limit==3
    assert cfg.robot_cfg.actuators['wheel_motors'].damping==0
    assert cfg.wheel_damping_compensation_ratio==0 and not cfg.motor_response_enabled

def test_target_velocity_profile_diagnostic_cannot_certify_policy():
    tree=ast.parse((ROOT/'scripts/evaluate_rmuc_grid.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Dict) and any(
        isinstance(k,ast.Constant) and k.value=='diagnostic_dr_ablation' for k in n.keys))
    expression=next(v for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant) and k.value=='passed')
    assert not eval(compile(ast.Expression(expression),'<actual acceptance>','eval'),
        {'args':SimpleNamespace(diagnostic_wheel_velocity_profile='proposal.yaml'),'rows':[{'passed':True}]})

@pytest.mark.parametrize('fraction',[0.,.05,.1,.25,1.])
def test_wheel_only_fraction_preserves_leg_model_and_policy_scale(fraction):
    import yaml,copy
    spec=importlib.util.spec_from_file_location('velocity_profile',ROOT/'scripts/rmuc_velocity_profile.py')
    profile_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile_module)
    profile=yaml.safe_load((ROOT/'configs/rmuc_wheel_velocity_migration.yaml').read_text())
    legs=SimpleNamespace(stiffness=60.,damping=2.,effort_limit=40.)
    cfg=SimpleNamespace(sim=SimpleNamespace(dt=.002),wheel_torque_scale_nm=5.5,
        gas_spring_radial_force_n=10.775,motor_response_enabled=True,robot_cfg=SimpleNamespace(actuators={
        'wheel_motors':SimpleNamespace(stiffness=0.,damping=.1,effort_limit=5.5),'leg_motors':legs}))
    leg_before=copy.deepcopy(vars(legs))
    result=profile_module.apply_velocity_profile(cfg,profile,fraction)
    assert result['proportional_gain_nm_per_rad_s']==pytest.approx(.1+.1*fraction)
    assert result['output_effort_limit_nm']==pytest.approx(5.5-2.5*fraction)
    assert result['legacy_action_reference_scale_rad_s']==pytest.approx(55.)
    assert vars(legs)==leg_before and cfg.wheel_torque_scale_nm==5.5 and cfg.gas_spring_radial_force_n==10.775
    assert result['target_controller']['output_effort_limit_nm']==3. and cfg.motor_response_enabled
