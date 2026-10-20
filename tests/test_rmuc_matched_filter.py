import importlib.util
from pathlib import Path
import math
import torch
p=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/command_sequence.py'
s=importlib.util.spec_from_file_location('matched_filter',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_perfect_ramp_has_zero_matched_error_and_nonzero_legacy_error():
 alpha=1-math.exp(-.008/.16);v=torch.zeros(2);target=v.clone();valid=torch.zeros(2,dtype=torch.bool);worst=0
 for i in range(101):
  cmd=torch.tensor([.005*i,.5-.005*i]);v=torch.where(valid,v+alpha*(cmd-v),cmd)
  target=m.matched_filtered_target(target,cmd,valid,alpha)
  torch.testing.assert_close(v,target);worst=max(worst,float((v-cmd).abs().max()));valid[:]=True
 assert worst>.09


def test_partial_reset_does_not_keep_old_target():
 out=m.matched_filtered_target(torch.tensor([.5,.5]),torch.zeros(2),torch.tensor([False,True]),.1)
 torch.testing.assert_close(out,torch.tensor([0.,.45]))
