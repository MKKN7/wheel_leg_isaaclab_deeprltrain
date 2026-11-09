import runpy
from pathlib import Path
import torch
cost=runpy.run_path(str(Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/low_speed_tracking.py'))['low_forward_overspeed_cost']
def test_idle_both_directions_and_no_underspeed_penalty():
 v=torch.tensor([0.,0.,.1,.1,.1]);s=torch.tensor([.02,-.02,.08,.1,.12]);r=cost(v,s)
 torch.testing.assert_close(r,torch.tensor([.36,.36,0.,0.,.32]))
def test_reverse_and_high_speed_excluded_and_bounded():
 v=torch.tensor([-.2,.76,.5,.75]);s=torch.tensor([-1.,2.,5.,.75]);r=cost(v,s)
 torch.testing.assert_close(r,torch.tensor([0.,0.,1.,0.]))
