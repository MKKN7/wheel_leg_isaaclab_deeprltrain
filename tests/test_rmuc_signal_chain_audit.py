from pathlib import Path
import runpy
import torch
from rmuc_signal_chain_audit import TimestampQueueAudit
Delay=runpy.run_path(str(Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/joint_delay_dr.py'))['SampledDelay']
def test_observer_keeps_output_and_rng_and_handles_duplicate_partial_reset():
 torch.manual_seed(5);a=Delay(torch.zeros(4,2),.002,[0,.008],9,True);b=Delay(torch.zeros(4,2),.002,[0,.008],9,True);stamp=[0];audit=TimestampQueueAudit(a,lambda:stamp[0]);rng=torch.random.get_rng_state().clone()
 for t in range(20):
  stamp[0]=t;x=torch.full((4,2),float(t));a.push(x,t);b.push(x,t)
  if t%4==0:a.push(x+1,t);b.push(x+1,t)
  if t==11:
   ids=torch.tensor([0,2]);a.reset(ids);b.reset(ids);a.push(x+2,t);b.push(x+2,t)
  torch.testing.assert_close(a.read(),b.read(),rtol=0,atol=0)
 assert torch.equal(rng,torch.random.get_rng_state())
 assert audit.result()['max_abs']==0 and audit.result()['scalars']>0

def test_timestamp_checker_detects_extra_actual_step():
 a=Delay(torch.zeros(1,1),.002,[.002,.002],9);stamp=[0];audit=TimestampQueueAudit(a,lambda:stamp[0])
 for t in range(5):stamp[0]=t;a.step(torch.tensor([[float(t)]]))
 # A duplicate actuator update at same physics time shifts the realqueue twice.
 a.step(torch.tensor([[99.]]))
 assert audit.result()['max_abs']>0

def test_observer_attached_after_reset_does_not_treat_ignored_push_as_new_sample():
 a=Delay(torch.zeros(2,1),.002,[0,.008],2,True);a.steps[:]=torch.tensor([0,4]);a.push(torch.zeros(2,1),0)
 stamp=[0];audit=TimestampQueueAudit(a,lambda:stamp[0]);a.push(torch.ones(2,1),0)
 for t in range(1,10):stamp[0]=t;a.push(torch.full((2,1),float(t)),t)
 assert audit.result()['max_abs']==0 and audit.result()['scalars']>0
