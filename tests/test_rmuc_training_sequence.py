"""Exercise real environment command methods without loading Isaac Sim."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
from rmuc_command_sequence import validate_sequence, sequence_commands

ROOT=Path(__file__).resolve().parents[1]
ENV=ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'
CONTRACT=ENV.with_name('command_contract.py')
spec=importlib.util.spec_from_file_location('training_contract_test',CONTRACT)
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
names={'_current_command','_continuous_command_values','_acquisition_command_age','_current_command_stage_values_tensor'}
cls=next(n for n in ast.parse(ENV.read_text()).body if isinstance(n,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='_current_command' for m in n.body))
ns={'torch':torch,'sequence_commands':sequence_commands,'interpolate_leg_scale':contract.interpolate_leg_scale}
exec(compile(ast.Module(body=[m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name in names],type_ignores=[]),str(ENV),'exec'),ns)
Harness=type('Harness',(),{name:ns[name] for name in names})

def make():
 h=Harness();h.cfg=SimpleNamespace(play=False,curriculum_stage=1,command_reset_acquisition_ramp_s=0)
 h.num_envs=2;h.device='cpu';h._command=torch.zeros(2,3);h._command_turn_sign=torch.ones(2)
 h._rehearsal_mask=torch.tensor([False,False]);h._continuous_command_mask=torch.tensor([True,False]);h._continuous_commands=torch.tensor([[.2,0,0],[.2,0,0]])
 h._manual_leg_speed_knots=torch.tensor([0.,1.5]);h._manual_leg_scale_knots=torch.tensor([.19,.24])
 h._current_command_stage_values=lambda:(1.,0.,.22)
 h._curriculum_step_counter=torch.tensor([750,750]);h._training_sequence_table=validate_sequence([[0,1.5,0],[6,0,0]],.008,12)
 return h

def test_command_scale_and_reward_share_boundary_age():
 h=make();h._command_in_physics_step=True
 assert h._current_command()[0,0]==1.5
 assert h._current_command_stage_values_tensor()[2][0]==torch.tensor(.24)
 h._command_in_physics_step=False
 assert h._current_command()[0,0]==0
 assert h._current_command_stage_values_tensor()[2][0]==torch.tensor(.19)
 assert h._current_command()[1,0]==1 # fixed cohort unaffected

def test_reset_returns_to_initial_command_without_other_env_reset():
 h=make();h._curriculum_step_counter[0]=0
 assert h._current_command()[0,0]==1.5
 assert h._current_command()[1,0]==1

def test_disabled_sequence_preserves_sampled_command():
 h=make();h._training_sequence_table=None
 assert h._continuous_command_values() is h._continuous_commands
 assert h._current_command()[0,0]==torch.tensor(.2)


def test_explicit_leg_scale_profile_matches_fixed_dynamic_and_play_without_recursion():
 h=make();h.cfg.command_leg_scale_profile={'schema_version':1,'speed_abs_mps':[0.,.5],'scale_rad':[.19,.2]}
 h._manual_leg_speed_knots=torch.tensor([0.,.5]);h._manual_leg_scale_knots=torch.tensor([.19,.2])
 h._training_sequence_table=None;h._continuous_commands[:,0]=-.1
 h._current_command_stage_values=lambda:(-.1,0.,.184)
 h._command_turn_sign=torch.ones(2);h.cfg.command_reset_acquisition_ramp_s=0
 command=h._current_command()
 torch.testing.assert_close(command[:,0],torch.full((2,),-.1))
 scale=h._current_command_stage_values_tensor()[2]
 torch.testing.assert_close(scale,torch.full((2,),.192))
 h.cfg.play=True;h._manual_command=command.clone()
 torch.testing.assert_close(h._current_command_stage_values_tensor()[2],scale)


def test_leg_profile_validation_rejects_ambiguous_or_nonfinite_contracts():
 import pytest
 valid={'schema_version':1,'speed_abs_mps':[0.,.5],'scale_rad':[.19,.2]}
 assert contract.validate_leg_scale_profile(valid)==([0.,.5],[.19,.2])
 from omegaconf import OmegaConf
 assert contract.validate_leg_scale_profile(OmegaConf.create(valid))==([0.,.5],[.19,.2])
 for extra in [{'speed_abs_mps':[-.1,.5]},{'scale_rad':[.19,float('nan')]},{'speed_abs_mps':[0.,0.]},{'scale_rad':[.19]}]:
  with pytest.raises(ValueError):contract.validate_leg_scale_profile(dict(valid,**extra))


def test_random_schedule_uses_same_boundary_age_for_command_and_scale():
 h=make();h._training_sequence_table=None
 class Schedule:
  def commands(self,age):
   c=torch.zeros(len(age),3);c[:,0]=torch.where(age<750,1.5,0.);return c
 h._random_speed_sequence=Schedule();h._command_in_physics_step=True
 assert h._current_command()[0,0]==1.5
 assert h._current_command_stage_values_tensor()[2][0]==torch.tensor(.24)
 h._command_in_physics_step=False
 assert h._current_command()[0,0]==0
 assert h._current_command_stage_values_tensor()[2][0]==torch.tensor(.19)
 assert h._current_command()[1,0]==1
