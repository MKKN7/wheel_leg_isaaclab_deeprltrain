import importlib.util
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import ast
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('noise',ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/observation_noise.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_integer_delays_and_command_reconstruction():
    h=torch.zeros(3,4,25);h[:,:,0]=torch.arange(4);h[:,:,7]=.5;h[:,:,6]=.2
    now=h[:,0].clone();now[:,7]=1.;now[:,19:]=.7
    out=m.delayed_sensor_observation(h,torch.tensor([1,2,3]),now,torch.tensor([.3,0.,-.3]))
    torch.testing.assert_close(out[:,0],torch.tensor([1.,2.,3.]))
    assert torch.equal(out[:,7],now[:,7]) and torch.equal(out[:,19:],now[:,19:])
    torch.testing.assert_close(out[:,2]/.25+out[:,8],torch.tensor([.3,0.,-.3]))


def test_physical_units_bias_and_zero_noise():
    c=yaml.safe_load((ROOT/'configs/rmuc_wheel_leg.yaml').read_text())['domain_randomization']['sensor_model']
    for key in list(c):
        if 'std' in key:c[key]=0.
    x=torch.zeros(2,25);x[:,5]=-1;x[:,7]=1
    gyro=torch.full((2,3),.02);acc=torch.zeros_like(gyro)
    y=m.sensor_noise(x,torch.ones(2),gyro,acc,torch.zeros(2),c)
    torch.testing.assert_close(y[:,:3],gyro*.25)
    torch.testing.assert_close(y[:,2]/.25+y[:,8],torch.zeros(2))
    torch.testing.assert_close(y[:,3:6].norm(dim=-1),torch.ones(2))
    assert torch.equal(y[:,9:],x[:,9:])


def test_action_fifo_real_prephysics_tail():
    tree=ast.parse((ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RmucWheelLegEnv')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_pre_physics_step')
    node=next(n for n in fn.body if isinstance(n,ast.If) and ast.unparse(n.test)=='self._sensor_model_enabled')
    env=SimpleNamespace(_sensor_model_enabled=True,_command_action_history=torch.zeros(3,4,6),num_envs=3,device='cpu',_communication_delay_steps=torch.tensor([1,2,3]))
    for step in range(1,6):
        env._policy_actions=torch.full((3,6),float(step))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual fifo>','exec'),{'torch':torch,'self':env})
        torch.testing.assert_close(env._policy_actions[:,0],torch.tensor([max(step-1,0),max(step-2,0),max(step-3,0)]).float())


def test_sensor_reset_sampling_bounds_and_partial_reset_isolation(monkeypatch):
    tree=ast.parse((ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RmucWheelLegEnv')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_apply_progressive_domain_randomization')
    node=next(n for n in fn.body if isinstance(n,ast.If) and ast.unparse(n.test)=='self._sensor_model_enabled')
    c=yaml.safe_load((ROOT/'configs/rmuc_wheel_leg.yaml').read_text())['domain_randomization']['sensor_model']
    package = ModuleType('_sensor_sampling_fixture')
    package.__path__ = [str(ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg')]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    env=SimpleNamespace(device='cpu',_sensor_cfg=c,_sensor_model_enabled=True,
        cfg=SimpleNamespace(recovery_communication_delay_probability_override=-1.),
        _sensor_delay_steps=torch.full((1025,),9,dtype=torch.long),
        _communication_delay_steps=torch.full((1025,),9,dtype=torch.long),
        _sensor_gyro_bias=torch.full((1025,3),9.),_sensor_accel_bias=torch.full((1025,3),9.))
    ids=torch.arange(1024)
    scope={'torch':torch,'self':env,'env_ids':ids,'strength':1.,
           '__name__':'_sensor_sampling_fixture.env','__package__':'_sensor_sampling_fixture'}
    code=compile(ast.Module(body=[node],type_ignores=[]),'<actual sampling>','exec')
    torch.manual_seed(19);exec(code,scope)
    for delay in [env._sensor_delay_steps,env._communication_delay_steps]:
        assert set(delay[:1024].tolist())=={1,2,3}
        assert delay[-1]==9
    assert not torch.equal(env._sensor_delay_steps[:1024],env._communication_delay_steps[:1024])
    assert env._sensor_gyro_bias[:1024].abs().max()<=c['gyro_bias_max_rad_s']
    assert env._sensor_accel_bias[:1024].abs().max()<=c['accel_bias_max_mps2']
    scope['strength']=0.;exec(code,scope)
    assert env._sensor_delay_steps[:1024].count_nonzero()==0
    assert env._communication_delay_steps[:1024].count_nonzero()==0
    assert env._sensor_gyro_bias[:1024].count_nonzero()==0
    assert env._sensor_accel_bias[:1024].count_nonzero()==0
    reset=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_reset_idx')
    clear=next(n for n in reset.body if isinstance(n,ast.Assign) and any(ast.unparse(t)=='self._command_action_history[env_ids]' for t in n.targets))
    env._command_action_history=torch.ones(1025,4,6)
    exec(compile(ast.Module(body=[clear],type_ignores=[]),'<actual reset>','exec'),scope)
    assert env._command_action_history[:1024].count_nonzero()==0
    assert torch.all(env._command_action_history[-1]==1)


def test_spin_yaw_survives_noise_and_delay_without_widening_speed_clip():
    c=yaml.safe_load((ROOT/'configs/rmuc_wheel_leg.yaml').read_text())['domain_randomization']['sensor_model']
    x=torch.zeros(2,25);x[:,5]=-1;x[:,6]=2.;x[:,7]=0
    targets=torch.tensor([3.,-3.]);bias=torch.zeros(2,3)
    y=m.sensor_noise(x,torch.zeros(2),bias,bias,targets,c,yaw_error_clip=6.)
    torch.testing.assert_close(y[:,2]/.25+y[:,8],targets)
    h=y[:,None,:].repeat(1,4,1);h[:,:,2]=torch.tensor([[-.75],[.75]])
    out=m.delayed_sensor_observation(h,torch.tensor([1,3]),y,targets,yaw_error_clip=6.)
    torch.testing.assert_close(out[:,2]/.25+out[:,8],targets)
    assert out[:,6].abs().max()<=2
