"""Export independent recovery actor with explicit, incompatible action contract."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import yaml


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    params=args.checkpoint.resolve().parent/'params/env.yaml'
    cfg=yaml.load(params.read_text(),Loader=yaml.BaseLoader)
    radial_force=float(cfg['gas_spring_radial_force_n'])
    equivalent_torque=float(cfg.get('gas_spring_assist_torque_nm',0.))
    if (not all(math.isfinite(v) and v>=0 for v in (radial_force,equivalent_torque))
            or (radial_force>0 and equivalent_torque>0)):
        raise ValueError('Invalid or double-counted gas-spring physics in checkpoint params')
    progress_enabled=cfg.get('recovery_progress_observation','false').lower()=='true'
    rate_text=str(cfg.get('recovery_leg_target_slew_rate_rad_s','inf')).lower()
    slew_rate=None if rate_text in ('inf','.inf','+.inf') else float(rate_text)
    if slew_rate is not None and (not math.isfinite(slew_rate) or slew_rate<=0):
        raise ValueError('Invalid target slew rate')
    if float(cfg.get('recovery_joint_limit_tolerance_rad',0.))!=0.:
        raise ValueError('Nonzero recovery joint tolerance requires matching deployment termination guards')
    if cfg.get('curriculum_stage_name')!='recovery_prototype':
        raise ValueError('Only an explicitly recorded recovery run may use this exporter')
    if cfg.get('policy_outer_feedback_enabled')!='false' or cfg.get('command_observation_mode')!='tracking_error_with_target':
        raise ValueError('Recovery requires explicit E2E and matching observation semantics')
    sys.path.insert(0,str(root/'scripts/rsl_rl'))
    from verify_rmuc_export import restore_actor
    actor=restore_actor(args.checkpoint)
    if any(k.startswith('rmuc_') for k in actor.state_dict()):
        raise ValueError('Locomotion residual checkpoint cannot be relabeled as recovery')
    from pxr import Usd,UsdPhysics
    usd=Path(cfg['robot_cfg']['spawn']['usd_path'])
    stage=Usd.Stage.Open(str(usd));names=['right_active_link_pitch','right_hip_pitch','right_wheel_spin',
                                       'left_active_link_pitch','left_hip_pitch','left_wheel_spin']
    bounds=[]
    for name in [names[i] for i in [0,1,3,4]]:
        matches=[p for p in stage.Traverse() if p.GetName()==name]
        if len(matches)!=1:raise ValueError(f'Nonunique joint {name}')
        joint=UsdPhysics.RevoluteJoint(matches[0]);margin=float(cfg['leg_joint_limit_margin_rad'])
        low=math.radians(joint.GetLowerLimitAttr().Get())+margin
        high=math.radians(joint.GetUpperLimitAttr().Get())-margin
        if not all(math.isfinite(v) for v in [low,high]) or low>=high:raise ValueError('Invalid leg range')
        bounds.append([low,high])
    # Use the existing full-forward parity verifier, not the base-only exporter.
    subprocess.run([sys.executable,str(root/'scripts/verify_rmuc_export.py'),'--checkpoint',str(args.checkpoint),
                    '--output',str(args.output)],cwd=root,check=True)
    leg=cfg['robot_cfg']['actuators']['leg_motors'];wheel=cfg['robot_cfg']['actuators']['wheel_motors']
    robot_config=yaml.safe_load((root/'configs/rmuc_wheel_leg.yaml').read_text())['robot']
    safety_joints=[]
    for name in robot_config['dof_names']:
        matches=[p for p in stage.Traverse() if p.GetName()==name]
        if len(matches)!=1:raise ValueError(f'Nonunique safety joint {name}')
        joint=UsdPhysics.RevoluteJoint(matches[0])
        low=joint.GetLowerLimitAttr().Get();high=joint.GetUpperLimitAttr().Get()
        safety_joints.append(dict(name=name,lower_rad=math.radians(low) if math.isfinite(low) else None,
                                 upper_rad=math.radians(high) if math.isfinite(high) else None))
    if equivalent_torque > 0.:
        spring_model='equivalent_assist_joint_torque'
    elif radial_force > 0.:
        spring_model='legacy_radial_force_diagnostic'
    else:
        spring_model='equivalent_assist_joint_torque'
    contract=dict(policy_kind='rmuc_recovery_v1',observation_dim=25,action_dim=6,joint_order=names,
        onnx_sha256=hashlib.sha256((args.output/'policy.onnx').read_bytes()).hexdigest(),
        source_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_params_sha256=hashlib.sha256(params.read_bytes()).hexdigest(),
        source_usd_sha256=hashlib.sha256(usd.read_bytes()).hexdigest(),
        observation_mode=cfg['command_observation_mode'],command=[0.,0.,0.],
        command_observation_scale=list(map(float,cfg['command_observation_scale'])),
        output_semantics='raw normalized recovery actions; clip all channels to [-1,1]',
        uniform_clip_one_compatible=True,leg_action_slots=[0,1,3,4],leg_target_ranges_rad=bounds,
        leg_mapping='midpoint + half_range * clipped_action; NOT locomotion angle scale',
        kp=float(leg['stiffness']),kd=float(leg['damping']),leg_effort_limit_nm=float(leg['effort_limit']),
        target_leg_velocity=0.,leg_feedforward_nm=0.,wheel_action_slots=[2,5],
        wheel_action_sign=list(map(float,cfg['wheel_action_sign'])),
        wheel_action_scale_nm=float(cfg['wheel_torque_scale_nm']),wheel_effort_limit_nm=float(wheel['effort_limit']),
        wheel_limit_semantics='instantaneous_output_shaft_peak',
        wheel_continuous_output_reference_nm=3.0,
        wheel_motor='RM3508',wheel_esc='C620',
        matches_current_wheel_peak_contract=float(wheel['effort_limit'])==5.5 and float(cfg['wheel_torque_scale_nm'])==5.5,
        wheel_explicit_damping=float(wheel['damping']),
        radial_force_n_per_leg=radial_force,
        gas_spring_model=spring_model,
        equivalent_assist_torque_nm=equivalent_torque,
        training_equivalent_spring_dr=dict(
            common_fraction=float(cfg.get('gas_spring_torque_common_dr_fraction',0.)),
            independent_fraction=float(cfg.get('gas_spring_torque_independent_dr_fraction',0.)),
            composition='additive_zero_mean_relative_deviations'),
        equivalent_assist_body_order=['robot','left_active_link','right_active_link'],
        equivalent_assist_joint_order=['left_active_link_pitch','right_active_link_pitch'],
        equivalent_assist_direction='active_link_pitch=-torque; robot=equal_opposite_reaction',
        pd_dt=float(cfg['sim']['dt']),policy_dt=float(cfg['sim']['dt'])*int(cfg['decimation']),
        task_outer_feedback_required=False,locomotion_consumer_compatible=False,
        runtime_validation_complete=False,hardware_validated=False,handover_verified=False)
    contract['recovery_criteria']=dict(hold_seconds=float(cfg['recovery_hold_seconds']),
        upright_degrees=float(cfg['recovery_upright_degrees']),min_height=float(cfg['recovery_min_height']),
        max_planar_speed=float(cfg['recovery_max_planar_speed']),max_angular_speed=float(cfg['recovery_max_angular_speed']),
        wheel_normal_net_force_threshold_n=2.)
    contract['training_ground']=dict(inclination_degrees=float(cfg.get('recovery_slope_degrees',0.)),
        uphill_azimuth_degrees=float(cfg.get('recovery_slope_azimuth_degrees',0.)),
        sampling='fixed_plane_per_run',slope_behavior_certified=False)
    contract['training_communication_delay_cap_steps']=int(cfg.get('recovery_communication_delay_max_steps',-1))
    contract['training_communication_delay_probability_override']=float(cfg.get('recovery_communication_delay_probability_override',-1.))
    contract['termination_limits']=dict(min_root_height=.02,max_root_height=float(cfg['max_height_m']),
        max_xy_distance=float(cfg['max_xy_distance_m']),max_linear_speed=float(cfg['max_velocity_mps']),
        joint_limits=safety_joints)
    contract['episode_length_s']=float(cfg['episode_length_s'])
    contract['training_reward_profile']=cfg.get('recovery_reward_profile','legacy_v0')
    contract['training_safety_reward']={
        'full_scope':str(cfg.get('recovery_full_safety_scope','false')).lower()=='true',
        'joint_limit_weight':float(cfg.get('recovery_joint_limit_penalty',0.)),
        'impact_weight':float(cfg.get('recovery_impact_penalty',0.)),
        'braking_weight':float(cfg.get('recovery_braking_penalty',0.)),
        'wheel_gap_progress_weight':float(cfg.get('recovery_wheel_gap_progress_weight',0.)),
        'contact_soft_limit_n':float(cfg.get('recovery_contact_soft_limit_n',300.))}
    contract['training_refall_parameters']={k:float(v) for k,v in cfg.items() if k.startswith('recovery_refall_')}
    contract['recovery_rise_shaping_height_m']=float(cfg.get('recovery_rise_shaping_height',cfg.get('recovery_min_height',.28)))
    contract['training_strict_min_height_m']=float(cfg.get('recovery_min_height',.28))
    if int(cfg.get('recovery_slew_anneal_steps',0))>0:
        contract['training_slew_anneal']={
            'start_rad_s':float(cfg['recovery_slew_start_rad_s']),
            'end_rad_s':float(cfg['recovery_slew_end_rad_s']),
            'steps':int(cfg['recovery_slew_anneal_steps'])}
        slew_rate=float(cfg['recovery_slew_end_rad_s'])
    contract['training_standing_weights']={k:float(v) for k,v in cfg.get('recovery_standing_weights',{}).items()}
    contract['training_case_standing_weights']=cfg.get('recovery_case_standing_weights',{})
    contract['training_resolved_case_standing_weights']={name:{**contract['training_standing_weights'],
        **{k:float(v) for k,v in cfg.get('recovery_case_standing_weights',{}).get(name,{}).items()}}
        for name in ('front_down','back_down','left_wheel_side_down','right_wheel_side_down')}
    contract['training_contact_event_mode']=cfg.get('recovery_contact_event_mode','legacy_edge')
    contract['training_contact_hold_seconds']=float(cfg.get('recovery_contact_hold_seconds',.04))

    contract['training_reward_weights']={k:float(v) for k,v in cfg.get('recovery_reward_weights',{}).items()}
    contract['first_episode_control_steps']=math.ceil(contract['episode_length_s']/contract['policy_dt'])-1
    contract['policy_kind']='rmuc_recovery_v2' if progress_enabled or slew_rate is not None else 'rmuc_recovery_v1'
    contract['recovery_progress_observation']=progress_enabled
    contract['progress_slot']=7 if progress_enabled else None
    contract['progress_semantics']='measured_orientation_rise_hold_v1' if progress_enabled else None
    contract['leg_target_slew_rate_rad_s']=slew_rate
    contract['slew_reset']='measured_leg_position'
    contract['training_initial_tilt_degrees']=int(cfg.get('recovery_initial_tilt_degrees',90))
    contract['training_reset_cohorts']=[[int(v) for v in pair] for pair in cfg.get('recovery_reset_cohorts',[])]
    (args.output/'action_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    print('[RECOVERY_EXPORT] complete; deployment behavior not certified')


if __name__=='__main__':main()
