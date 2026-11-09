from pathlib import Path
import runpy,torch,pytest
f=runpy.run_path(str(Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/tracking_oscillation.py'))['tracking_error_variation']
def test_exact_steps_ramps_and_constant_offset_do_not_create_oscillation_cost():
 command=torch.tensor([0.,-.25,.5,1.5]);filtered=torch.tensor([0.,-.05,.2,.8]);ready=torch.ones(4,dtype=torch.bool)
 assert not f(command,command,filtered,filtered,ready).any()
 assert not f(command,command-.1,filtered,filtered-.1,ready).any()
def test_variation_and_reset_mask():
 z=torch.zeros(3);cost=f(z,torch.tensor([.1,-.1,.1]),z,z,torch.tensor([True,True,False]))
 assert cost[0]==cost[1] and 0<cost[0]<1 and cost[2]==0
 with pytest.raises(ValueError):f(z,z,z,z,torch.ones(3,dtype=torch.bool),scale=0)

def test_matched_filter_reversal_and_reset():
 command=torch.tensor([.3,.3,-.3,-.3,0.,0.]);speed=command.clone();fs=0.;ft=0.
 for i in range(len(command)):
  if i==0:fs=speed[i];ft=command[i]
  else:fs=fs+.2*(speed[i]-fs);ft=ft+.2*(command[i]-ft)
  assert f(command[i],speed[i],ft,fs,torch.tensor(True))==0
 assert f(torch.tensor(0.),torch.tensor(.3),torch.tensor(0.),torch.tensor(0.),torch.tensor(False))==0

def test_sustained_sine_error_has_nonzero_cost():
 t=torch.arange(1000)*.008;speed=.03*torch.sin(2*torch.pi*5*t);filtered=speed[0];cost=[]
 for v in speed:
  filtered=filtered+.05*(v-filtered)
  cost.append(f(torch.tensor(0.),v,torch.tensor(0.),filtered,torch.tensor(True)))
 assert torch.stack(cost)[100:].mean()>.05
