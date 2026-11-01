import importlib.util
import sys
import types
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg'
pkg=types.ModuleType('feedback_age_test');pkg.__path__=[str(p)];sys.modules[pkg.__name__]=pkg
for name in ['joint_delay_dr','wheel_feedback_age']:
 spec=importlib.util.spec_from_file_location('feedback_age_test.'+name,p/(name+'.py'))
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
WheelFeedbackAge=module.WheelFeedbackAge

def test_age_ramp_hold_and_no_double_tick():
 q=WheelFeedbackAge(torch.zeros(2,2),.002,[.004,.004],[.004,.004],7)
 q.phase[:]=0
 for t in range(10):
  result=q.update(torch.full((2,2),float(t)),t).clone()
  if t>=2: assert torch.all(result==max(0,t-t%2-2))
  assert torch.equal(q.update(torch.full((2,2),999.),t),result)
 assert q.readback()['maximum_age_ms']==6

def test_independent_wheels_reset_and_rng():
 torch.manual_seed(11);before=torch.random.get_rng_state().clone()
 q=WheelFeedbackAge(torch.zeros(1024,2),.002,[0,.006],[.002,.004],7)
 assert torch.equal(before,torch.random.get_rng_state())
 assert (q.queues[0].steps!=q.queues[1].steps).any()
 q.update(torch.ones(1024,2),0)
 q.reset(torch.tensor([0]))
 out=q.update(torch.full((1024,2),9.),0)
 assert torch.equal(out[0],torch.tensor([9.,9.]))
 assert torch.all(out[1:]==1)
 assert q.readback()['maximum_age_ms']==8
