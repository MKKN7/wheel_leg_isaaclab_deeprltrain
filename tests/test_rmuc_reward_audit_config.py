from types import SimpleNamespace as N
from rmuc_reward_audit_config import apply_reward_audit_config

def test_reverse_oscillation_grace_is_copied_to_mc_reward_only():
 cfg=N(command_reverse_oscillation_grace_s=0.)
 changed=apply_reward_audit_config(cfg,{'command_reverse_oscillation_grace_s':'3.0','command_stage_linear_x_mps':[0.,1.]})
 assert cfg.command_reverse_oscillation_grace_s==3.
 assert changed['command_reverse_oscillation_grace_s']==3.

def test_reverse_alive_and_pitch_reward_survive_mc_config_transfer():
 cfg=N(command_reverse_alive_baseline=0.,command_reverse_pitch_oscillation_weight=0.)
 changed=apply_reward_audit_config(cfg,{
  'command_reverse_alive_baseline':'10.0',
  'command_reverse_pitch_oscillation_weight':'2.0',
  'command_stage_linear_x_mps':['0','1']},[[-.975,0],[-1.,0]])
 assert changed['command_reverse_alive_baseline']==cfg.command_reverse_alive_baseline==10.
 assert changed['command_reverse_pitch_oscillation_weight']==cfg.command_reverse_pitch_oscillation_weight==2.

def test_reward_only_copy_keeps_physics_and_command_sampler():
 cfg=N(command_speed_oscillation_start_level=13,command_low_forward_overspeed_shaping_enabled=False,
       command_straight_alignment_weight=14.,reward_clip_min=-20.,command_stage_linear_x_mps=[1.,2.],
       command_rehearsal_fraction=.5,wheel_torque_scale_nm=5.5,command_reset_speed_bootstrap_start_level=2)
 changed=apply_reward_audit_config(cfg,{'command_speed_oscillation_start_level':'27',
  'command_low_forward_overspeed_shaping_enabled':'true','command_straight_alignment_weight':'2',
  'reward_clip_min':'-100','command_stage_linear_x_mps':[0.,3.],
  'command_rehearsal_fraction':'.1','wheel_torque_scale_nm':'2','command_reset_speed_bootstrap_start_level':'9'})
 assert cfg.command_speed_oscillation_start_level==27 and cfg.command_low_forward_overspeed_shaping_enabled
 assert cfg.command_straight_alignment_weight==2 and cfg.reward_clip_min==-100
 assert cfg.command_stage_linear_x_mps==[1.,2.] and cfg.command_rehearsal_fraction==.5
 assert cfg.wheel_torque_scale_nm==5.5 and cfg.command_reset_speed_bootstrap_start_level==2
 assert cfg.command_reward_stage_speed_lookup==(0.,3.)
 assert cfg.command_sequence_overshoot_weight==0
 assert cfg.command_dynamic_tracking_weight==0
 assert len(changed)==9


def test_fixed_reverse_audit_rejects_silent_loss_of_reverse_dynamic_reward():
 import pytest
 with pytest.raises(ValueError, match='Fixed reverse'):
  apply_reward_audit_config(N(),{'command_dynamic_tracking_include_fixed_reverse':'true'},[[-.4,0]])


def test_explicit_fixed_reverse_audit_preserves_tracking_and_runaway_cost():
 cfg=N(command_dynamic_tracking_include_fixed_reverse=False,command_dynamic_tracking_weight=0.,
       command_dynamic_tracking_scale_mps=.2,command_reverse_runaway_weight=0.)
 changed=apply_reward_audit_config(cfg,{
  'command_stage_linear_x_mps':['0','1.8'],
  'command_dynamic_tracking_include_fixed_reverse':'true',
  'command_dynamic_tracking_weight':'4',
  'command_dynamic_tracking_scale_mps':'.5',
  'command_reverse_runaway_weight':'2'},[[-.6,0]],fixed_reverse_equivalent=True)
 assert cfg.command_dynamic_tracking_include_fixed_reverse
 assert cfg.command_dynamic_tracking_weight==4 and cfg.command_dynamic_tracking_scale_mps==.5
 assert cfg.command_reverse_runaway_weight==2 and cfg.critic_audit_fixed_reverse_reward_enabled
 assert cfg.command_sequence_overshoot_weight==0 and cfg.command_dynamic_stop_posture_weight==0


def test_non_weight_named_alive_reward_is_copied_for_future_highspeed_audits():
 import runpy
 from pathlib import Path
 import torch
 cfg=N(command_high_straight_alive_baseline=0.,command_min_height_m=.15,
       command_min_operating_root_height_m=.15)
 changed=apply_reward_audit_config(cfg,{'command_high_straight_alive_baseline':'25',
       'command_min_height_m':'.16','command_min_operating_root_height_m':'.17',
       'command_stage_linear_x_mps':['0','2.5']})
 assert changed['command_high_straight_alive_baseline']==25.
 assert cfg.command_min_height_m==.16 and cfg.command_min_operating_root_height_m==.17
 fn=runpy.run_path(str(Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/survival_shaping.py'))['high_straight_alive_baseline']
 command=torch.tensor([[2.,0,0],[2.1,0,0],[2.5,0,.1],[2.5,0,0]])
 torch.testing.assert_close(fn(command,torch.tensor([False,False,False,True]),cfg.command_high_straight_alive_baseline),torch.tensor([0.,25.,0.,0.]))
