import importlib.util
from pathlib import Path
import torch
import pytest
p=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/leg_transport_delay.py';s=importlib.util.spec_from_file_location('legdelay',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_disabled_identity():
    x=torch.randn(4,4);d=m.LegTransportDelay(x,.002);state=torch.random.get_rng_state();d.reset(torch.arange(4));assert d.step(x) is x;assert torch.equal(state,torch.random.get_rng_state())

def test_delay_order_reset_and_no_double_delay():
    x=torch.zeros(2,4);d=m.LegTransportDelay(x,.002,.002,1);d.reset(torch.arange(2));a=torch.ones_like(x);b=a*2
    assert torch.equal(d.step(a),x)
    assert torch.equal(d.step(b),a)
    d.reset(torch.tensor([0]));y=d.step(b*2);assert torch.equal(y[0],x[0]);assert torch.equal(y[1],b[1])
    assert d.readback()['physics_steps']==1

def test_mixture_and_validation():
    x=torch.zeros(1024,4);d=m.LegTransportDelay(x,.002,.002,.5);d.reset(torch.arange(1024));y=d.step(x+3);assert 400<int(d.active.sum())<624;assert (y[d.active]==0).all();assert (y[~d.active]==3).all()
    for delay in [-.002,.013,.001]:
        with pytest.raises(ValueError):m.LegTransportDelay(x,.002,delay)

def test_host_pd_transport_before_motor_response_order():
    from types import SimpleNamespace
    response_path=p.with_name('motor_response.py')
    spec=importlib.util.spec_from_file_location('response',response_path);response=importlib.util.module_from_spec(spec);spec.loader.exec_module(response)
    class Actuator:
        def __init__(self):
            self.damping=torch.zeros(2,4);self.effort_limit=torch.full((2,4),40.);self.applied_effort=torch.zeros(2,4)
        def compute(self,action,q,dq):
            # Stand-in for already calculated/clipped host PD output.
            self.applied_effort=action.joint_efforts.clamp(-40,40)
            action.joint_efforts=self.applied_effort
            return action
        def reset(self,ids):self.applied_effort[ids]=0
    actuator=Actuator();delay=m.install_leg_transport(actuator,.002,.002,1,1)
    motor=response.install_effort_response(actuator,.002,.002,.003)
    actuator.reset(torch.arange(2));motor.time_constant.fill_(.002)
    def call():return actuator.compute(SimpleNamespace(joint_efforts=torch.full((2,4),10.)),None,None).joint_efforts.clone()
    assert torch.equal(call(),torch.zeros(2,4))
    expected=10*(1-torch.exp(torch.tensor(-1.)))
    assert torch.allclose(call(),torch.full((2,4),expected))
    assert delay.calls==2
    actuator.reset(torch.tensor([0]));y=call();assert (y[0]==0).all();assert (y[1]>expected).all()
