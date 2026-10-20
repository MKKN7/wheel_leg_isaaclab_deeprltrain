"""Exercise actual reset routing without loading Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import pytest
source=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'
tree=ast.parse(source.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='_refresh_continuous_commands' for m in x.body));method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_refresh_continuous_commands')

@pytest.mark.parametrize('lo,history,level,expected_calls',[
    (-.1,[-.1,.3],0,0),
    (0.,[0.,.3],0,1),
    (.5,[-.4,1.6],1,0),  # Forward schedule after reverse curriculum.
    (.5,[.5,1.6,-.4],1,1),  # Unopened negative stage must not change old RNG.
])
def test_actual_reset_uses_signed_schedule_without_forward_sampler(lo,history,level,expected_calls):
    calls=[]
    def sampler(n,*args,**kwargs):calls.append(n);return torch.ones(n,3)
    ns={'torch':torch,'sample_opened_commands':sampler}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),ns)
    reset=[]
    h=NS(num_envs=32,device='cpu',_command_level=level,_continuous_command_mask=torch.zeros(32,dtype=torch.bool),_rehearsal_mask=torch.zeros(32,dtype=torch.bool),_continuous_commands=torch.zeros(32,3),_random_speed_sequence=NS(lo=lo,reset=lambda ids:reset.append(ids.clone())),_training_sequence_table=None,cfg=NS(command_continuous_sampling_fraction=.14,curriculum_stage=1,feedback_contract_by_command=True,command_reward_contract_by_command=True,command_rehearsal_fraction=.75,command_rehearsal_speed_focus_fraction=0,command_stage_linear_x_mps=history,command_stage_yaw_rate_rad_s=[0]*len(history)))
    ns['_refresh_continuous_commands'](h,torch.arange(32))
    assert len(calls)==expected_calls and len(reset)==1
    assert int(h._continuous_command_mask.sum())==round(32*.14)
    assert torch.equal(torch.sort(reset[0]).values,torch.where(h._continuous_command_mask)[0])
