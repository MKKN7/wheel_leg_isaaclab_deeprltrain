"""Copy only reward shaping fields for explicit critic audits, never hardware/command ranges."""
import math
# Command levels use a dedicated reward speed lookup; live commands stay unchanged.
REWARD_REFERENCES={'command_reverse_alive_baseline', 'command_idle_micro_speed_penalty_weight', 'stand_leg_length_weight', 'command_rehearsal_fraction', 'command_yaw_overspeed_relative_to_target', 'command_turn_attitude_scale', 'command_tracking_error_penalty_weight', 'command_min_wheel_contact_fraction', 'command_pitch_excess_tolerance_deg', 'command_dynamic_tracking_include_fixed_reverse', 'command_reverse_runaway_weight', 'command_speed_oscillation_start_level', 'command_bank_error_scale_deg', 'effort_limit_nm', 'stand_geometry_symmetry_weight', 'command_min_ready_env_fraction', 'command_speed_error_scale_mps', 'command_stillness_weight', 'command_speed_oscillation_penalty_weight', 'command_turn_pitch_rate_tolerance_rad_s', 'command_severe_underspeed_grace_steps', 'command_straight_alignment_scale_deg', 'command_height_penalty_weight', 'command_low_speed_turn_yaw_error_scale_rad_s', 'command_pitch_tracking_penalty_weight', 'command_straight_alignment_rate_weight', 'command_pitch_tracking_scale_deg', 'command_pitch_target_max_deg', 'command_high_speed_exploration_weight', 'command_turn_pitch_rate_penalty_weight', 'settled_wheel_smoothness_scale', 'command_rehearsal_speed_focus_end_level', 'spin_underspeed_scale_rad_s', 'command_severe_underspeed_margin_mps', 'command_dynamic_stop_tracking_scale_mps', 'stand_attitude_weight', 'command_dynamic_tracking_weight', 'command_reward_stage_speed_lookup', 'command_large_turn_pitch_tolerance_deg', 'command_overspeed_margin_mps', 'command_rehearsal_gate_stage_ids', 'command_dynamic_matched_filter_enabled', 'command_stage_linear_x_mps', 'command_sequence_overshoot_weight', 'wheel_speed_feedback_damping_anneal_target_scale', 'stillness_weight', 'command_leg_length_preference_scale', 'tilt_weight', 'command_rehearsal_speed_focus_start_level', 'command_pitch_tracking_tolerance_deg', 'command_small_turn_dense_yaw_weight', 'command_require_analytic_bank', 'command_bank_error_penalty_weight', 'command_speed_delta_filter_tau_s', 'command_pitch_excess_penalty_weight', 'stand_height_weight', 'stand_contact_weight', 'sim', 'command_speed_oscillation_tolerance_mps', 'yaw_tolerance_rad_s', 'command_bank_tracking_weight', 'command_straight_alignment_yaw_gate_floor', 'command_turn_yaw_tracking_start_level', 'command_reverse_settled_tracking_weight', 'command_speed_acquisition_weight', 'stand_leg_extension_weight', 'command_reward_contract_by_command', 'command_stage_hold_steps', 'command_advance_yaw_error_rad_s', 'command_tracking_error_tail_gain', 'command_turn_yaw_tracking_penalty_weight', 'command_high_speed_pitch_rate_penalty_weight', 'spin_asymmetric_reward_enabled', 'command_stand_shape_weight', 'command_roll_penalty_weight', 'command_speed_error_margin_mps', 'command_turn_pitch_rate_scale_rad_s', 'command_posture_weight', 'command_reset_speed_bootstrap_fraction', 'command_pitch_rate_penalty_weight', 'command_overspeed_penalty_weight', 'energy_weight', 'command_pitch_rate_scale_rad_s', 'wheel_damping_anneal_target', 'command_dynamic_tracking_scale_mps', 'command_pitch_target_deg_per_mps', 'command_high_speed_pitch_rate_tolerance_rad_s', 'command_speed_regression_penalty_weight', 'yaw_weight', 'command_yaw_weight', 'command_bank_tolerance_deg', 'command_turn_pitch_rate_start_level', 'command_high_speed_pitch_rate_scale_rad_s', 'command_min_stable_env_fraction', 'command_min_posture_env_fraction', 'command_large_turn_pitch_scale_deg', 'upright_weight', 'command_forward_weight', 'command_velocity_tolerance_mps', 'command_filtered_speed_error_penalty_weight', 'decimation', 'command_tracking_variation_scale_mps', 'command_speed_delta_scale_mps', 'command_active_bank_reward_scale', 'command_yaw_tracking_penalty_weight', 'command_termination_penalty_weight', 'command_overspeed_scale_mps', 'command_pitch_excess_scale_deg', 'command_pitch_rate_tolerance_rad_s', 'spin_overspeed_deadband_rad_s', 'curriculum_stage', 'command_exact_low_speed_reward_enabled', 'command_reverse_alignment_scale_deg', 'velocity_tolerance_mps', 'command_tracking_variation_weight', 'stand_symmetry_weight', 'command_small_turn_dense_yaw_max_rad_s', 'stand_no_contact_penalty', 'command_underspeed_penalty_weight', 'command_reset_speed_bootstrap_start_level', 'velocity_weight', 'reward_clip_min', 'command_dynamic_stop_posture_weight', 'command_yaw_tolerance_rad_s', 'command_rehearsal_priority_stage_ids', 'command_severe_underspeed_penalty_weight', 'action_rate_weight', 'command_reverse_penalty_weight', 'command_wheel_sync_weight', 'command_tracking_error_tail_enabled', 'command_progress_weight', 'command_advance_confirmation_steps', 'spin_underspeed_penalty_weight', 'command_leg_sync_weight', 'command_high_straight_alive_baseline', 'command_contact_weight', 'command_pitch_target_underspeed_gain_deg_per_mps', 'command_speed_oscillation_scale_mps', 'command_low_forward_overspeed_shaping_enabled', 'command_straight_alignment_tolerance_deg', 'command_reset_speed_bootstrap_enabled', 'command_min_survival_env_fraction', 'command_min_height_m', 'command_yaw_rate_penalty_weight', 'command_idle_micro_speed_scale_mps', 'command_survival_bonus', 'command_tracking_variation_tolerance_mps', 'command_yaw_stabilization_weight', 'command_min_operating_root_height_m', 'command_straight_alignment_weight', 'command_straight_alignment_rate_scale_rad_s', 'command_lateral_weight', 'command_severe_underspeed_scale_mps', 'command_rehearsal_speed_focus_fraction', 'settled_wheel_smoothness_weight', 'command_speed_error_penalty_weight', 'command_turn_roll_scale', 'reward_clip_max', 'command_pitch_tracking_start_level', 'command_severe_underspeed_start_level'}
REWARD_REFERENCES.add('command_reverse_oscillation_grace_s')
REWARD_REFERENCES.add('command_reverse_pitch_oscillation_weight')
EXTRA={'command_reverse_oscillation_grace_s','command_survival_bonus','command_small_turn_dense_yaw_max_rad_s',
       'command_reverse_alive_baseline',
       'command_low_speed_turn_yaw_error_scale_rad_s','command_turn_attitude_scale',
       'command_turn_roll_scale','spin_overspeed_deadband_rad_s','settled_wheel_smoothness_scale',
       'command_advance_yaw_error_rad_s','command_exact_low_speed_reward_enabled','command_dynamic_matched_filter_enabled',
       'command_dynamic_tracking_include_fixed_reverse','command_low_forward_overspeed_shaping_enabled',
       'command_low_speed_turn_yaw_tracking_enabled','command_tracking_error_tail_enabled',
       'command_tracking_error_tail_gain','command_yaw_overspeed_relative_to_target',
       'command_reward_contract_by_command','command_require_analytic_bank',
       'command_active_bank_reward_scale','command_leg_length_preference_scale',
       'command_straight_alignment_yaw_gate_floor','command_severe_underspeed_grace_steps',
       'command_pitch_target_deg_per_mps','command_pitch_target_underspeed_gain_deg_per_mps',
       'command_pitch_target_max_deg','spin_asymmetric_reward_enabled','stand_no_contact_penalty',
       'command_high_straight_alive_baseline','command_min_height_m',
       'command_min_operating_root_height_m'}
SUFFIXES=('_weight','_scale_mps','_scale_rad_s','_scale_deg','_tolerance_mps','_tolerance_rad_s',
          '_tolerance_deg','_margin_mps','_start_level','_tau_s','_delta_scale_mps')
PREFIXES=('command_','spin_','stand_','settled_wheel_')
BASE={'reward_clip_min','reward_clip_max','velocity_weight','upright_weight','yaw_weight',
      'stillness_weight','tilt_weight','action_rate_weight','energy_weight',
      'velocity_tolerance_mps','yaw_tolerance_rad_s'}
EXCLUDE={'command_rehearsal_start_level','command_reset_speed_bootstrap_start_level',
         'command_rehearsal_speed_focus_start_level','command_gate_steps',
         'command_stand_shape_weight_dummy'}

def apply_reward_audit_config(cfg, values, fixed_commands=(), *, fixed_reverse_equivalent=False):
    reverse_dynamic = (any(float(c[0])<0 for c in fixed_commands)
                       and str(values.get('command_dynamic_tracking_include_fixed_reverse','false')).lower()=='true')
    if reverse_dynamic and not fixed_reverse_equivalent:
        raise ValueError('Fixed reverse critic audit requires explicit dynamic-reward equivalent configuration')
    changed={}
    for name,raw in values.items():
        if name in EXCLUDE or name not in REWARD_REFERENCES or not hasattr(cfg,name):continue
        if name not in EXTRA and name not in BASE and not(name.startswith(PREFIXES) and name.endswith(SUFFIXES)):continue
        old=getattr(cfg,name)
        if isinstance(old,bool):
            if str(raw).lower() not in ('true','false'):raise ValueError(f'Invalid bool {name}')
            value=str(raw).lower()=='true'
        elif isinstance(old,int):value=int(raw)
        elif isinstance(old,float):value=float(raw)
        else:continue
        setattr(cfg,name,value);changed[name]=value
    speeds=tuple(float(v) for v in values.get('command_stage_linear_x_mps',()))
    if not speeds or not all(math.isfinite(v) for v in speeds):raise ValueError('Training reward stage speed lookup required')
    # Fixed manual-command audits have no dynamic cohort: this term is exactly
    # zero there; retain the environment's sequence-required guard.
    for name in ('command_sequence_overshoot_weight','command_dynamic_tracking_weight',
                 'command_dynamic_stop_posture_weight','command_dynamic_stop_tracking_scale_mps'):
        if name=='command_dynamic_tracking_weight' and reverse_dynamic and fixed_reverse_equivalent:
            # This term is also active for fixed reverse commands in training.
            # Only sequence-only terms may be zeroed for the fixed audit.
            continue
        setattr(cfg,name,0.)
        changed[name]=0.
    if reverse_dynamic and fixed_reverse_equivalent:
        cfg.critic_audit_fixed_reverse_reward_enabled=True
        changed['critic_audit_fixed_reverse_reward_enabled']=True
    cfg.command_reward_stage_speed_lookup=speeds
    changed['command_reward_stage_speed_lookup']=list(speeds)
    return changed
