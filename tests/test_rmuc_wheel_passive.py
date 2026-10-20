"""CPU audit of separate motor/passive contracts and actual environment wiring."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace, MethodType
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg'
spec = importlib.util.spec_from_file_location('wheel_passive', SOURCE / 'wheel_passive.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def config(mode='separated_passive_v1'):
    return SimpleNamespace(sim=SimpleNamespace(dt=.002),decimation=4,leg_pd_update_period_s=.002,wheel_torque_contract=mode, wheel_mechanical_viscous_nm_s_rad=.002,
        wheel_coulomb_friction_nm=.001, wheel_coulomb_omega_eps_rad_s=.1,
        wheel_feedback_damping_nm_s_rad=0., wheel_feedback_damping_mode='postclip_loss_v1', wheel_damping_anneal_target=-1.,
        separated_wheel_mechanical_viscous_range=(.001,.005),
        separated_wheel_coulomb_range=(.001,.001),
        separated_wheel_feedback_damping_range=(0.,0.),
        wheel_damping_compensation_ratio=0., wheel_torque_scale_nm=5.5,
        robot_cfg=SimpleNamespace(actuators={'wheel_motors': SimpleNamespace(damping=.1, effort_limit=5.5)}))


def test_legacy_is_exact_noop_including_rng_and_cfg():
    cfg=config('legacy_idealpd'); before=repr(cfg); rng=torch.get_rng_state().clone()
    assert not m.configure_wheel_torque_contract(cfg)
    assert repr(cfg)==before and torch.equal(torch.get_rng_state(),rng)
    assert m.configure_wheel_torque_contract(cfg:=config())
    assert cfg.robot_cfg.actuators['wheel_motors'].damping==0.


@pytest.mark.parametrize('dt,decimation', [(.001,8),(.002,4)])
def test_explicit_period_survives_environment_contract_configuration(dt,decimation):
    cfg=config()
    cfg.separated_wheel_physics_dt_s=dt
    assert m.configure_wheel_torque_contract(cfg)
    assert cfg.sim.dt==dt and cfg.decimation==decimation
    assert cfg.sim.dt*cfg.decimation==.008 and cfg.leg_pd_update_period_s==.002
    assert cfg.robot_cfg.actuators['wheel_motors'].effort_limit==5.5


def test_unsupported_explicit_period_fails_closed():
    cfg=config()
    cfg.separated_wheel_physics_dt_s=.004
    with pytest.raises(ValueError):m.configure_wheel_torque_contract(cfg)


def test_declared_hardware_peak_is_shared_by_training_and_simulator():
    cfg=config();cfg.wheel_authored_effort_limit_nm=4.89
    motor=cfg.robot_cfg.actuators['wheel_motors']
    motor.effort_limit=motor.effort_limit_sim=4.89
    assert m.configure_wheel_torque_contract(cfg)
    assert cfg.wheel_torque_scale_nm==5.5
    motor.effort_limit_sim=5.5
    with pytest.raises(ValueError,match='Simulator'):m.configure_wheel_torque_contract(cfg)
    motor.effort_limit_sim=4.89;motor.effort_limit=5.5
    with pytest.raises(ValueError,match='declared'):m.configure_wheel_torque_contract(cfg)


@pytest.mark.parametrize('field,value', [('wheel_torque_contract','bad'),
    ('wheel_damping_anneal_target',0.),('wheel_damping_compensation_ratio',.1),
    ('wheel_coulomb_omega_eps_rad_s',0.),('wheel_mechanical_viscous_nm_s_rad',-1.),
    ('wheel_feedback_damping_nm_s_rad',float('nan')),('wheel_feedback_damping_mode','bad'),('wheel_torque_scale_nm',3.)])
def test_invalid_contract_fails_closed(field,value):
    cfg=config();setattr(cfg,field,value)
    with pytest.raises(ValueError):m.configure_wheel_torque_contract(cfg)


def test_zero_odd_dissipative_drag_and_body_reaction_power():
    omega=torch.tensor([0.,.01,.1,10.,50.],dtype=torch.float64)
    drag=m.mechanical_torque(omega,.002,.001,.1)
    assert drag[0]==0 and (drag*omega<=0).all()
    torch.testing.assert_close(m.mechanical_torque(-omega,.002,.001,.1),-drag)
    q=torch.tensor([[.5,.5,.5,.5]],dtype=torch.float64).expand(5,-1)
    axis=torch.tensor([1.,0.,0.],dtype=torch.float64)
    pair=m.body_torque_pair(q,axis,drag)
    torch.testing.assert_close(pair.sum(-2),torch.zeros(5,3,dtype=torch.float64))
    world_axis=m.rotate_axis(q,axis)
    parent_omega=torch.ones(5,3,dtype=torch.float64)
    child_omega=parent_omega+world_axis*omega[:,None]
    power=(pair[:,0]*parent_omega+pair[:,1]*child_omega).sum(-1)
    torch.testing.assert_close(power,drag*omega)


def test_separated_wheel_dr_samples_actual_loss_terms_and_restores_nominal():
    cfg=config()
    b, c, d = m.sample_separated_wheel_parameters(cfg, 1024, 1.0, 'cpu')
    assert b.shape == c.shape == d.shape == (1024, 2)
    assert .001 <= b.min() and b.max() <= .005
    assert b.std() > 0 and not torch.equal(b[:, 0], b[:, 1])
    torch.testing.assert_close(c, torch.full_like(c, .001))
    torch.testing.assert_close(d, torch.zeros_like(d))
    b, c, d = m.sample_separated_wheel_parameters(cfg, 4, 0.0, 'cpu')
    torch.testing.assert_close(b, torch.full_like(b, .002))
    cfg.separated_wheel_feedback_damping_range = (0., .08)
    _, _, d = m.sample_separated_wheel_parameters(cfg, 1024, 1.0, 'cpu')
    assert 0 <= d.min() and d.max() <= .08 and d.std() > 0


def extract_method(path,cls_name,name):
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls_name)
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    for arg in fn.args.args:arg.annotation=None
    fn.returns=None;ns={'torch':torch}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    return ns[name]


def test_actual_idealpd_motor_clips_before_external_drag():
    lab=importlib.util.find_spec('isaaclab')
    assert lab is not None, 'Installed IsaacLab source required'
    source=Path(lab.origin).parent/'actuators'
    compute=extract_method(source/'actuator_pd.py','IdealPDActuator','compute')
    clip=extract_method(source/'actuator_base.py','ActuatorBase','_clip_effort')
    omega=torch.tensor([[50.,-50.]])
    actuator=SimpleNamespace(stiffness=torch.zeros_like(omega),damping=torch.zeros_like(omega),effort_limit=torch.full_like(omega,5.5))
    actuator._clip_effort=MethodType(clip,actuator)
    action=SimpleNamespace(joint_positions=torch.zeros_like(omega),joint_velocities=torch.zeros_like(omega),joint_efforts=torch.tensor([[-10.,10.]]))
    motor=compute(actuator,action,torch.zeros_like(omega),omega).joint_efforts
    passive=m.mechanical_torque(omega,.002,.001,.1)
    assert motor.abs().max()<=5.5
    torch.testing.assert_close(motor+passive,torch.tensor([[-5.601,5.601]]))


@pytest.mark.parametrize('active', [False, True])
def test_actual_usd_geometry_and_environment_substep_composition(active):
    geometry=m.read_wheel_joint_geometry(str(ROOT/'source/agent_world/agent_world/assets/usd_files/rmuc_wheel_leg/rmuc_robot_closed_loop.usd'),('right_wheel_spin','left_wheel_spin'))
    assert geometry==[('right_lower_leg','right_wheel',(1.,0.,0.)),('left_lower_leg','left_wheel',(1.,0.,0.))]
    tree=ast.parse((SOURCE/'env.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RmucWheelLegEnv')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_apply_action')
    code=compile(ast.Module(body=[fn.body[0]],type_ignores=[]),'<actual passive physics block>','exec')
    calls=[]
    data=SimpleNamespace(joint_vel=SimpleNamespace(torch=torch.tensor([[50.,-50.]])),body_quat_w=SimpleNamespace(torch=torch.tensor([[[0.,0.,0.,1.]]*4])))
    env=SimpleNamespace(cfg=config(),_separated_wheel_passive=True,_wheel_control_ids=[0,1],
        _wheel_mechanical_viscous_per_env=torch.full((1,2),.002),
        _wheel_coulomb_per_env=torch.full((1,2),.001),
        _wheel_feedback_damping_per_env=torch.zeros(1,2),
        robot=SimpleNamespace(data=data,instantaneous_wrench_composer=SimpleNamespace(add_forces_and_torques_index=lambda **kw:calls.append(kw))),
        _wheel_passive_body_ids=[torch.tensor([0,1]),torch.tensor([2,3])],_wheel_passive_axes_b=torch.tensor([[1.,0.,0.]]*2),
        _last_wheel_mechanical_torque_nm=torch.zeros(1,2),
        _last_wheel_feedback_torque_nm=torch.zeros(1,2),
        _last_wheel_total_passive_torque_nm=torch.zeros(1,2),
        _last_wheel_mechanical_body_torques_w=torch.zeros(1,2,2,3))
    scope={'self':env,'mechanical_torque':m.mechanical_torque,
           'feedback_damping_torque':m.feedback_damping_torque,
           'body_torque_pair':m.body_torque_pair}
    env.cfg.wheel_feedback_damping_mode = 'active_motor_v1' if active else 'postclip_loss_v1'
    env._wheel_feedback_damping_per_env.fill_(.05)
    for _ in range(2):exec(code,scope)
    assert len(calls)==4 and all(c['is_global'] for c in calls)
    torch.testing.assert_close(env._last_wheel_mechanical_torque_nm,torch.tensor([[-.101,.101]]))
    torch.testing.assert_close(env._last_wheel_feedback_torque_nm,torch.tensor([[-2.5,2.5]]))
    expected = [[-.101,.101]] if active else [[-2.601,2.601]]
    torch.testing.assert_close(env._last_wheel_total_passive_torque_nm,torch.tensor(expected))
    for c in calls:torch.testing.assert_close(c['torques'].sum(1),torch.zeros(1,3))
    env._separated_wheel_passive=False;exec(code,scope);assert len(calls)==4


def pd_actuator():
    lab=importlib.util.find_spec('isaaclab')
    source=Path(lab.origin).parent/'actuators'
    compute=extract_method(source/'actuator_pd.py','IdealPDActuator','compute')
    clip=extract_method(source/'actuator_base.py','ActuatorBase','_clip_effort')
    a=SimpleNamespace(is_implicit_model=False,stiffness=torch.full((2,1),10.),
        damping=torch.full((2,1),.5),effort_limit=torch.full((2,1),20.),reset=lambda ids:None)
    a._clip_effort=MethodType(clip,a)
    a.compute=MethodType(compute,a)
    return a


def pd_action():
    return SimpleNamespace(joint_positions=torch.ones(2,1),joint_velocities=torch.zeros(2,1),joint_efforts=torch.zeros(2,1))


def test_true_leg_pd_compute_cadence_hold_changed_state_and_scoped_reset():
    a=pd_actuator();hold=m.install_explicit_pd_sample_hold(a,.001,.002)
    results=[];flags=[]
    for tick in range(8):
        # Joint states change EVERY physics step; PD may read them only at 0,2,4,6.
        out=a.compute(pd_action(),torch.full((2,1),tick*.1),torch.full((2,1),float(tick)))
        results.append(out.joint_efforts.clone());flags.append(hold.last_was_update)
    assert flags==[True,False]*4 and hold.physics_calls==8 and hold.update_count==4
    torch.testing.assert_close(torch.stack(results)[:,0,0],torch.tensor([10.,10.,7.,7.,4.,4.,1.,1.]))
    a.compute(pd_action(),torch.zeros(2,1),torch.zeros(2,1)) # update tick 8
    a.reset(torch.tensor([0]))
    result=a.compute(pd_action(),torch.ones(2,1),torch.ones(2,1)) # hold tick 9
    torch.testing.assert_close(result.joint_efforts,torch.tensor([[0.],[10.]]))
    assert not hold.last_was_update
    cfg=config();m.configure_wheel_torque_contract(cfg)
    assert cfg.sim.dt==.001 and cfg.decimation==8 and cfg.sim.dt*cfg.decimation==.008


def test_pd_hold_inside_motor_response_keeps_1ms_physical_response():
    spec=importlib.util.spec_from_file_location('motor_response',SOURCE/'motor_response.py')
    response_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(response_module)
    a=pd_actuator();hold=m.install_explicit_pd_sample_hold(a,.001,.002)
    response=response_module.install_effort_response(a,.001,.01,.01)
    response.time_constant.fill_(.01)
    first=a.compute(pd_action(),torch.zeros(2,1),torch.zeros(2,1)).joint_efforts.clone()
    second=a.compute(pd_action(),torch.ones(2,1),torch.ones(2,1)).joint_efforts.clone()
    assert hold.update_count==1 and torch.all(second>first)
    torch.testing.assert_close(second,torch.full((2,1),10*(1-__import__('math').exp(-.002/.01))))
    torch.testing.assert_close(hold.held_applied_effort,torch.full((2,1),10.))
    a.reset(torch.tensor([0]))
    assert hold.held_applied_effort[0].item()==response.effort[0].item()==0.
    assert hold.held_applied_effort[1].item()==10.


def test_pd_sample_hold_rejects_implicit_or_wrong_install_order():
    a=pd_actuator();a.is_implicit_model=True
    with pytest.raises(ValueError,match='explicit'):m.install_explicit_pd_sample_hold(a,.001,.002)
    a.is_implicit_model=False;a._rmuc_effort_response=object()
    with pytest.raises(ValueError,match='before'):m.install_explicit_pd_sample_hold(a,.001,.002)
    with pytest.raises(ValueError):m.ExplicitPDSampleHold(torch.zeros(1),.001,.0015)


def test_axis_matches_actual_installed_isaac_api_for_reset_and_rotated_states():
    from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
    roll=torch.tensor([torch.pi/2,0.,.32,-.7],dtype=torch.float64)
    pitch=torch.tensor([0.,.4,-.2,.6],dtype=torch.float64)
    yaw=torch.tensor([0.,-.8,1.2,.2],dtype=torch.float64)
    q=quat_from_euler_xyz(roll,pitch,yaw)
    axis=torch.tensor([1.,0.,0.],dtype=torch.float64).expand(4,-1)
    expected=quat_apply(q,axis)
    torch.testing.assert_close(m.rotate_axis(q,axis),expected)
    torch.testing.assert_close(expected[0],torch.tensor([1.,0.,0.],dtype=torch.float64))
    omega=torch.tensor([10.,-20.,30.,-40.],dtype=torch.float64)
    torque=m.mechanical_torque(omega,.002,.001,.1)
    pairs=m.body_torque_pair(q,axis,torque)
    # Validate power against the independent backend API axis, not helper itself.
    relative_angular_velocity=expected*omega[:,None]
    power=(pairs[:,1]*relative_angular_velocity).sum(-1)
    torch.testing.assert_close(power,torque*omega)
    assert (power<=0).all()


def test_axes_match_actual_usd_parent_body_transform():
    from pxr import Usd,UsdGeom
    from isaaclab.utils.math import quat_apply
    path=ROOT/'source/agent_world/agent_world/assets/usd_files/rmuc_wheel_leg/rmuc_robot_closed_loop.usd'
    stage=Usd.Stage.Open(str(path))
    geometry=m.read_wheel_joint_geometry(str(path),('right_wheel_spin','left_wheel_spin'))
    for parent,child,axis in geometry:
        prim=next(p for p in stage.Traverse() if p.GetName()==parent)
        transform=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rotation=transform.ExtractRotationQuat()
        q=torch.tensor([list(rotation.GetImaginary())+[rotation.GetReal()]],dtype=torch.float64)
        local=torch.tensor([axis],dtype=torch.float64)
        expected=torch.tensor([tuple(transform.TransformDir(tuple(axis)))],dtype=torch.float64)
        torch.testing.assert_close(m.rotate_axis(q,local),expected)
        torch.testing.assert_close(m.rotate_axis(q,local),quat_apply(q,local))


def test_separated_forces_backend_spring_order_legacy_preserves_history():
    cfg=config('legacy_idealpd');cfg.gas_spring_quaternion_order='wxyz'
    m.configure_wheel_torque_contract(cfg);assert cfg.gas_spring_quaternion_order=='wxyz'
    cfg.wheel_torque_contract='separated_passive_v1'
    m.configure_wheel_torque_contract(cfg);assert cfg.gas_spring_quaternion_order=='xyzw'
    assert cfg.robot_cfg.actuators['wheel_motors'].damping==0.


def test_explicit_lower_electromagnetic_peak_keeps_action_scale():
    cfg=config()
    cfg.robot_cfg.actuators['wheel_motors'].effort_limit=2.4469479
    with pytest.raises(ValueError):m.configure_wheel_torque_contract(cfg)
    cfg.diagnostic_wheel_peak_nm=2.4469479
    assert m.configure_wheel_torque_contract(cfg)
    assert cfg.wheel_torque_scale_nm==5.5
    cfg.diagnostic_wheel_peak_nm=float('nan')
    with pytest.raises(ValueError):m.configure_wheel_torque_contract(cfg)
