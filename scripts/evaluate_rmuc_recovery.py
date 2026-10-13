"""First complete episode per fallen start; capture terminal state before reset."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--replicas',type=int,default=1)
parser.add_argument('--gui-play',action='store_true',help='Real-time looping GUI preview; no acceptance report.')
parser.add_argument('--seed',type=int,default=None)
parser.add_argument('--diagnostic-dr-strength',type=float,default=None)
parser.add_argument('--diagnostic-hardware-profile',type=Path,default=None)
parser.add_argument('--diagnostic-slope-degrees',type=float,default=None)
parser.add_argument('--diagnostic-slope-azimuth-degrees',type=float,default=None)
parser.add_argument('--diagnostic-communication-delay-steps',type=int,choices=[0,1,2,3],default=None)
parser.add_argument('--diagnostic-communication-delay-cap-steps',type=int,choices=[0,1,2,3],default=None)
parser.add_argument('--diagnostic-communication-delay-probability',type=float,default=None)
parser.add_argument('--trace-stride',type=int,default=0,help='Save control-boundary trajectory every N steps plus terminal; 0 disables.')
parser.add_argument('--save-reference-rollout',type=Path,default=None,
    help='Record delivered policy observations and raw deterministic actions for fixed-reference rehearsal.')
parser.add_argument('--diagnostic-zero-action-after',type=float,default=None,
    help='Override normalized actions after this many seconds; mechanical diagnostic, never policy acceptance.')
parser.add_argument('--diagnostic-joint-tolerance',type=float,default=None)
parser.add_argument('--diagnostic-leg-target-slew-rate',type=float,default=None)
parser.add_argument('--diagnostic-initial-tilt',type=int,choices=[0,5,15,30,45,60,75,90],default=None)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.gui_play and args.headless:parser.error('--gui-play requires a visible GUI')
if not 1<=args.replicas<=32:parser.error('replicas must be1..32')
if args.trace_stride<0:parser.error('trace-stride must be nonnegative')
if args.diagnostic_dr_strength is not None and not 0<=args.diagnostic_dr_strength<=1:
    parser.error('diagnostic-dr-strength must be within [0,1]')
if args.diagnostic_zero_action_after is not None and not 0<=args.diagnostic_zero_action_after<float('inf'):
    parser.error('diagnostic-zero-action-after must be finite and nonnegative')
launcher=AppLauncher(args);app=launcher.app
try:
    import hashlib,json,math,sys
    import torch
    import gymnasium as gym
    import agent_tasks
    sys.path.insert(0,str(Path(__file__).resolve().parent/'rsl_rl'))
    from verify_rmuc_export import restore_actor
    from agent_tasks.direct.rmuc_wheel_leg.recovery_env import RmucRecoveryEnvCfg
    import yaml
    saved=yaml.load((args.checkpoint.resolve().parent/'params/env.yaml').read_text(),Loader=yaml.BaseLoader)
    from agent_tasks.direct.rmuc_wheel_leg.recovery_rewards import DEFAULT_WEIGHTS
    weights={k:float(v) for k,v in saved.get('recovery_reward_weights',DEFAULT_WEIGHTS).items()}
    saved_rate=str(saved.get('recovery_leg_target_slew_rate_rad_s','inf')).lower()
    slew_rate=float('inf') if saved_rate in ('inf','.inf','+.inf') else float(saved_rate)
    if args.diagnostic_leg_target_slew_rate is not None:
        slew_rate=args.diagnostic_leg_target_slew_rate
    cfg=RmucRecoveryEnvCfg(recovery_reward_profile=saved.get('recovery_reward_profile','legacy_v0'),
                           recovery_reward_weights=weights,
                           recovery_leg_target_slew_rate_rad_s=slew_rate,
                           recovery_initial_tilt_degrees=int(saved.get('recovery_initial_tilt_degrees',90))
                           if args.diagnostic_initial_tilt is None else args.diagnostic_initial_tilt,
                           recovery_joint_limit_tolerance_rad=float(saved.get('recovery_joint_limit_tolerance_rad',0.))
                           if args.diagnostic_joint_tolerance is None else args.diagnostic_joint_tolerance)
    cfg.scene.num_envs=4*args.replicas;cfg.sim.device='cuda:0'
    # Restore actual checkpoint mechanics before any explicit diagnostic profile.
    # New5.5Nm defaults must not reinterpret historical3Nm recovery weights.
    cfg.wheel_torque_scale_nm=float(saved['wheel_torque_scale_nm'])
    cfg.wheel_authored_effort_limit_nm=float(saved.get('wheel_authored_effort_limit_nm',3.))
    for group in ('wheel_motors','leg_motors'):
        for field in ('stiffness','damping','effort_limit'):
            setattr(cfg.robot_cfg.actuators[group],field,
                    float(saved['robot_cfg']['actuators'][group][field]))
    # Independent evaluation uses the requested fixed angle and all four
    # directions, never the training cohort frequency or promotion state.
    cfg.recovery_reset_cohorts=()
    cfg.recovery_curriculum_enabled=False
    cfg.recovery_slope_degrees=float(saved.get('recovery_slope_degrees',0.))
    cfg.recovery_slope_azimuth_degrees=float(saved.get('recovery_slope_azimuth_degrees',0.))
    if args.diagnostic_slope_degrees is not None:cfg.recovery_slope_degrees=args.diagnostic_slope_degrees
    if args.diagnostic_slope_azimuth_degrees is not None:cfg.recovery_slope_azimuth_degrees=args.diagnostic_slope_azimuth_degrees
    cfg.recovery_progress_observation=saved.get('recovery_progress_observation','false').lower()=='true'
    # Old checkpoints explicitly recorded zero equivalent torque and radial
    # force 10.775 N. Missing fields remain zero to avoid reinterpreting them
    # under the new 10.775 N·m contract.
    cfg.gas_spring_assist_torque_nm=float(saved.get('gas_spring_assist_torque_nm',0.))
    cfg.gas_spring_radial_force_n=float(saved.get('gas_spring_radial_force_n',0.))
    cfg.gas_spring_radial_dr_fraction=float(saved.get('gas_spring_radial_dr_fraction',0.))
    for name in ('gas_spring_torque_common_dr_fraction','gas_spring_torque_independent_dr_fraction',
                 'motor_response_time_constant_min_s','motor_response_time_constant_max_s'):
        if name in saved:setattr(cfg,name,float(saved[name]))
    cfg.motor_response_enabled=saved.get('motor_response_enabled','false').lower()=='true'
    cfg.recovery_full_safety_scope=saved.get('recovery_full_safety_scope','false').lower()=='true'
    # Historical runs before these terms were added had zero safety cost.
    cfg.recovery_joint_limit_penalty=float(saved.get('recovery_joint_limit_penalty',0.))
    cfg.recovery_impact_penalty=float(saved.get('recovery_impact_penalty',0.))
    cfg.recovery_braking_penalty=float(saved.get('recovery_braking_penalty',0.))
    cfg.recovery_wheel_gap_progress_weight=float(saved.get('recovery_wheel_gap_progress_weight',0.))
    cfg.recovery_contact_soft_limit_n=float(saved.get('recovery_contact_soft_limit_n',300.))
    if 'recovery_standing_weights' in saved:
        saved_standing={k:float(v) for k,v in saved['recovery_standing_weights'].items()}
        cfg.recovery_standing_weights.update(saved_standing)
        # Missing terms must retain the historical zero contribution.
        cfg.recovery_standing_weights['stable_pose']=saved_standing.get('stable_pose',0.)
        cfg.recovery_standing_weights['double_contact']=saved_standing.get('double_contact',0.)
    cfg.recovery_case_standing_weights={}  # missing historical bank inherits restored global weights
    cfg.recovery_contact_event_mode=saved.get('recovery_contact_event_mode','legacy_edge')
    cfg.recovery_contact_hold_seconds=float(saved.get('recovery_contact_hold_seconds',.04))
    if 'recovery_case_standing_weights' in saved:
        cfg.recovery_case_standing_weights={k:{n:float(v) for n,v in bank.items()} for k,bank in saved['recovery_case_standing_weights'].items()}
    for name in ('recovery_refall_penalty','recovery_refall_arm_seconds','recovery_refall_dwell_seconds',
                 'recovery_refall_height_m','recovery_refall_tilt_degrees'):
        if name in saved:setattr(cfg,name,float(saved[name]))
    cfg.recovery_rise_shaping_height=float(saved.get(
        'recovery_rise_shaping_height',saved.get('recovery_min_height',.28)))
    if 'recovery_min_height' in saved:
        cfg.recovery_min_height=float(saved['recovery_min_height'])
    for name in ('recovery_hold_seconds','recovery_upright_degrees',
                 'recovery_max_planar_speed','recovery_max_angular_speed','episode_length_s'):
        if name in saved:setattr(cfg,name,float(saved[name]))
    # Evaluation is fixed-rate: use the terminal contract, never replay the
    # training anneal or depend on evaluation batch size.
    if (int(saved.get('recovery_slew_anneal_steps',0))>0
            and args.diagnostic_leg_target_slew_rate is None):
        cfg.recovery_leg_target_slew_rate_rad_s=float(saved['recovery_slew_end_rad_s'])
    slew_rate=cfg.recovery_leg_target_slew_rate_rad_s
    if args.seed is not None:cfg.seed=args.seed
    hardware_override=None
    if args.diagnostic_hardware_profile is not None:
        from rmuc_hardware_profile import apply_hardware_profile
        hardware_override=apply_hardware_profile(cfg,yaml.safe_load(args.diagnostic_hardware_profile.read_text()))
    if args.diagnostic_dr_strength is not None:
        cfg.domain_randomization_enabled=True
        cfg.domain_randomization_strength_override=args.diagnostic_dr_strength
    actor=restore_actor(args.checkpoint).to('cuda:0').eval()
    # Evaluation defaults to the original uncapped distribution, even if the
    # source trained with an easier communication-delay curriculum.
    if args.diagnostic_communication_delay_cap_steps is not None:
        cfg.recovery_communication_delay_max_steps=args.diagnostic_communication_delay_cap_steps
    if args.diagnostic_communication_delay_probability is not None:
        cfg.recovery_communication_delay_probability_override=args.diagnostic_communication_delay_probability
    env=gym.make('Robotics-RMUC-WheelLeg-Recovery-v0',cfg=cfg)
    if args.gui_play:
        import time
        try:
            u=env.unwrapped
            obs,_=env.reset()
            u.sim.set_camera_view(eye=(6.,6.,5.),target=(1.5,1.5,.3))
            print('[RECOVERY_GUI] deterministic policy; four directions; auto reset; not certification',flush=True)
            steps=0
            while app.is_running():
                start=time.perf_counter()
                with torch.inference_mode():
                    action=actor({'policy':obs['policy']},stochastic_output=False)
                    obs,_,_,_,_=env.step(action)
                steps+=1
                if steps%500==0:
                    print(f'[RECOVERY_GUI] steps={steps}',flush=True)
                time.sleep(max(0.,u.step_dt-(time.perf_counter()-start)))
        finally:
            env.close()
        raise SystemExit(0)
    try:
        u=env.unwrapped;obs,_=env.reset();count=u.num_envs
        sampled_communication_delay=u._recovery_sampled_communication_delay_steps.clone()
        if args.diagnostic_communication_delay_steps is not None:
            u._communication_delay_steps.fill_(args.diagnostic_communication_delay_steps)
        assert torch.equal(u._recovery_case,torch.arange(count,device=u.device)%4)
        assert u._recovery_cohorts is None and not cfg.recovery_curriculum_enabled
        initial_dr_readback=dict(strength=u._domain_randomization_strength.tolist(),
            wheel_damping=u.robot.actuators['wheel_motors'].damping.tolist(),
            wheel_effort_limit_nm=u.robot.actuators['wheel_motors'].effort_limit.tolist(),
            equivalent_spring_torque_nm=u._gas_spring_torque_nm.tolist(),
            mass_scale=u._domain_base_mass_scale.tolist(),friction_scale=u._domain_friction_scale.tolist(),
            observation_delay_steps=u._sensor_delay_steps.tolist(),communication_delay_steps=u._communication_delay_steps.tolist())
        hold=torch.zeros(count,dtype=torch.long,device=u.device)
        longest=hold.clone();closed=torch.zeros(count,dtype=torch.bool,device=u.device)
        unsafe_seen=closed.clone();ever_established=closed.clone()
        best_up=torch.full((count,),-1.,device=u.device)
        duration=torch.zeros(count,device=u.device)
        terminal=[None]*count
        traces=[[] for _ in range(count)]
        gate_names=('upright','height','planar_speed','angular_speed','both_wheels',
                    'supported_upright','supported_upright_height','all')
        gate_counts=torch.zeros(count,len(gate_names),device=u.device)
        gate_samples=torch.zeros(count,device=u.device)
        motion_peak=torch.zeros(count,4,device=u.device)
        wheel_motor_peak=torch.zeros(count,2,device=u.device)
        wheel_motor_samples=torch.zeros(count,device=u.device,dtype=torch.long)
        def audit_wheel_motor():
            values=u.robot.data.applied_torque.torch[:,u._wheel_control_ids].abs()
            active=~closed
            assert bool(torch.isfinite(values[active]).all()),'Nonfinite applied wheel torque'
            limits=u.robot.actuators['wheel_motors'].effort_limit
            assert bool((values[active]<=limits[active]+1e-5).all()),'Applied wheel torque exceeds configured peak'
            wheel_motor_peak.copy_(torch.where(active[:,None],torch.maximum(wheel_motor_peak,values),wheel_motor_peak))
            wheel_motor_samples.add_(active.long())
        previous_leg_target=None
        contact_peak_event=[None]*count
        previous_applied=u.robot.data.joint_pos.torch[:,u._leg_control_ids].clone()
        applied_step_peak=torch.zeros(count,device=u.device)
        original_target_writer=u.robot.set_joint_position_target
        def audited_target_writer(target,*positional,**keywords):
            ids=keywords.get('joint_ids')
            if ids is u._leg_control_ids:
                # The base environment also writes a target. Only the final
                # recovery write is sampled below, after _apply_action returns.
                audited_target_writer.last=target.detach().clone()
            return original_target_writer(target,*positional,**keywords)
        u.robot.set_joint_position_target=audited_target_writer
        original_apply=u._apply_action
        def audited_apply():
            audit_wheel_motor()
            original_apply()
            target=audited_target_writer.last
            delta=(target-previous_applied).abs().amax(-1)
            applied_step_peak.copy_(torch.where(~closed,torch.maximum(applied_step_peak,delta),applied_step_peak))
            if math.isfinite(cfg.recovery_leg_target_slew_rate_rad_s):
                assert bool((delta[~closed]<=cfg.recovery_leg_target_slew_rate_rad_s*cfg.sim.dt+1e-6).all()),'Applied target slew limit violated'
            previous_applied.copy_(target)
        u._apply_action=audited_apply
        original=u._get_dones
        def audited_dones():
            audit_wheel_motor()
            done,timeout=original()
            up,height,ready=u.recovery_metrics()
            active=~closed
            _,linear,angular=u._state()
            wheel_forces=u.contact_sensor.data.net_forces_w[:,u._wheel_contact_sensor_ids]
            upright=up>math.cos(math.radians(cfg.recovery_upright_degrees))
            high=height>cfg.recovery_min_height
            slow=torch.linalg.vector_norm(linear[:,:2],dim=-1)<cfg.recovery_max_planar_speed
            calm=torch.linalg.vector_norm(angular,dim=-1)<cfg.recovery_max_angular_speed
            supported=(torch.linalg.vector_norm(wheel_forces,dim=-1)>2.).all(-1)
            joint_gate=upright&high&slow&calm&supported
            assert torch.equal(joint_gate,ready),'Diagnostic gate differs from recovery ready'
            conditions=torch.stack((upright,high,slow,calm,supported,
                supported&upright,supported&upright&high,joint_gate),-1)
            gate_counts.add_((conditions&active[:,None]).float())
            gate_samples.add_(active.float())
            leg_ids=u._leg_control_ids
            leg_speed=u.robot.data.joint_vel.torch[:,leg_ids].abs().amax(-1)
            leg_torque=u.robot.data.applied_torque.torch[:,leg_ids].abs().amax(-1)
            contact_norm=torch.linalg.vector_norm(u.contact_sensor.data.net_forces_w,dim=-1)
            contact_peak,contact_body=contact_norm.max(-1)
            for i in (active&(contact_peak>motion_peak[:,3])).nonzero().flatten().tolist():
                body_id=int(contact_body[i])
                contact_peak_event[i]=dict(
                    body=u.contact_sensor.body_names[body_id],
                    seconds=float(duration[i])+u.step_dt,
                    force_n=float(contact_peak[i]),
                    tilt_deg=float(torch.rad2deg(torch.acos(up[i].clamp(-1,1)))),
                    root_height_m=float(height[i]),
                    leg_speed_rad_s=float(leg_speed[i]),
                    leg_torque_nm=float(leg_torque[i]))
            motion_peak[:,1:].copy_(torch.where(active[:,None],
                torch.maximum(motion_peak[:,1:],torch.stack((leg_speed,leg_torque,contact_peak),-1)),motion_peak[:,1:]))
            hold.copy_(torch.where(active&ready,hold+1,torch.where(active,torch.zeros_like(hold),hold)))
            longest.copy_(torch.maximum(longest,hold))
            established=hold*u.step_dt>=cfg.recovery_hold_seconds
            ever_established.logical_or_(established&active)
            unsafe_seen.logical_or_(done&active)
            best_up.copy_(torch.where(active,torch.maximum(best_up,up),best_up))
            duration.add_(active.float()*u.step_dt)
            if args.trace_stride:
                for i in active.nonzero().flatten().tolist():
                    if int(gate_samples[i])%args.trace_stride and not bool(done[i]|timeout[i]):continue
                    q=u.robot.data.joint_pos.torch[i,u._dof_ids]
                    limits=u._joint_pos_limits[i] if u._joint_pos_limits.ndim==3 else u._joint_pos_limits
                    margin=torch.minimum(q-limits[:,0],limits[:,1]-q)
                    traces[i].append(dict(seconds=float(duration[i]),height_m=float(height[i]),
                        tilt_deg=float(torch.rad2deg(torch.acos(up[i].clamp(-1,1)))),ready=bool(ready[i]),
                        wheel_load_n=torch.linalg.vector_norm(wheel_forces[i],dim=-1).tolist(),
                        wheel_height_m=u.robot.data.body_pos_w.torch[i,u._wheel_body_ids,2].tolist(),
                        body_contacts=[dict(body=u.contact_sensor.body_names[j],net_force_n=float(contact_norm[i,j]))
                            for j in (contact_norm[i]>2.).nonzero().flatten().tolist()],
                        planar_speed_mps=float(torch.linalg.vector_norm(linear[i,:2])),
                        angular_speed_rad_s=float(torch.linalg.vector_norm(angular[i])),
                        ready_gates={name:bool(conditions[i,j]) for j,name in enumerate(gate_names)},
                        continuous_hold_seconds=float(hold[i]*u.step_dt),
                        wheel_velocity_world_mps=u.robot.data.body_lin_vel_w.torch[i,u._wheel_body_ids].tolist(),
                        joint_position_rad=q.tolist(),joint_margin_rad=margin.tolist(),
                        executed_action=u._policy_actions[i].tolist(),
                        applied_leg_target_rad=u._recovery_applied_leg_target[i].tolist(),
                        leg_motor_torque_nm=u.robot.data.applied_torque.torch[i,leg_ids].tolist(),
                        wheel_motor_torque_nm=u.robot.data.applied_torque.torch[i,u._wheel_control_ids].tolist(),
                        contact_peak_n=float(contact_peak[i]),unsafe=bool(done[i])))
            for i in (active&(done|timeout)).nonzero().flatten().tolist():
                q=u.robot.data.joint_pos.torch[i,u._dof_ids]
                qd=u.robot.data.joint_vel.torch[i,u._dof_ids]
                limits=u._joint_pos_limits[i] if u._joint_pos_limits.ndim==3 else u._joint_pos_limits
                violations=[]
                for j in ((q<limits[:,0])|(q>limits[:,1])).nonzero().flatten().tolist():
                    joint_id=int(u._dof_ids[j])
                    violations.append(dict(joint=u.robot.joint_names[joint_id],
                        position_rad=float(q[j]),velocity_rad_s=float(qd[j]),
                        lower_rad=float(limits[j,0]),upper_rad=float(limits[j,1]),
                        excess_rad=float(torch.maximum(limits[j,0]-q[j],q[j]-limits[j,1])),
                        leg_controlled=joint_id in u._leg_control_ids))
                terminal[i]={'unsafe':bool(done[i]),'timeout':bool(timeout[i]),
                    'joint_limit_violations':violations,
                    'final_ready':bool(ready[i]),'final_established':bool(established[i]),
                    'final_height':float(height[i]),'final_tilt_deg':float(torch.rad2deg(torch.acos(up[i].clamp(-1,1)))),
                    'termination_reasons':[k for k,v in u.get_termination_reasons().items() if bool(v[i])]}
            closed.logical_or_(done|timeout)
            return done,timeout
        u._get_dones=audited_dones
        reference_samples=[]
        if args.save_reference_rollout is not None and args.diagnostic_zero_action_after is not None:
            raise ValueError('Reference rollout cannot contain action interventions')
        with torch.inference_mode():
            for step in range(math.ceil(cfg.episode_length_s/u.step_dt)+2):
                action=actor({'policy':obs['policy']},stochastic_output=False)
                if args.save_reference_rollout is not None and step%4==0:
                    active=~closed
                    reference_samples.append(dict(observation=obs['policy'][active].cpu().clone(),
                        action=action[active].cpu().clone(),case=u._recovery_case[active].cpu().clone(),
                        env_id=active.nonzero().flatten().cpu(),
                        control_step=torch.full((int(active.sum()),),step,dtype=torch.long)))
                if args.diagnostic_zero_action_after is not None and step*u.step_dt>=args.diagnostic_zero_action_after:
                    action=torch.zeros_like(action)
                bounds=u.robot.data.joint_pos_limits.torch[:,u._leg_control_ids,:]
                half_range=(bounds[...,1]-bounds[...,0])/2.-cfg.leg_joint_limit_margin_rad
                target=(bounds[...,1]+bounds[...,0])/2.+half_range*action[:,[0,1,3,4]].clamp(-1.,1.)
                # Include the initial target jump from measured reset pose.
                if previous_leg_target is None:
                    previous_leg_target=u.robot.data.joint_pos.torch[:,u._leg_control_ids].clone()
                delta=(target-previous_leg_target).abs().amax(-1)
                motion_peak[:,0].copy_(torch.where(~closed,torch.maximum(motion_peak[:,0],delta),motion_peak[:,0]))
                previous_leg_target=target.clone()
                obs,reward,done,timeout,_=env.step(action)
                if not torch.isfinite(obs['policy']).all() or not torch.isfinite(reward).all():
                    raise RuntimeError('Nonfinite recovery rollout')
                if bool(closed.all()):break
        assert bool(closed.all()),'Incomplete episode coverage'
        names=['front_down','back_down','left_wheel_side_down','right_wheel_side_down']
        rows=[]
        for i in range(count):
            rows.append(dict(case=names[i%4],seconds=float(duration[i]),best_tilt_deg=float(torch.rad2deg(torch.acos(best_up[i].clamp(-1,1)))),
                longest_ready_seconds=float(longest[i]*u.step_dt),ever_established=bool(ever_established[i]),
                terminal=terminal[i],recovery_hold_passed=bool(terminal[i]['final_established'] and not unsafe_seen[i])))
            rows[-1]['motion_audit']=dict(zip(
                ['max_requested_leg_target_step_rad','max_leg_speed_rad_s','max_leg_applied_torque_nm','max_body_net_contact_force_n'],
                motion_peak[i].tolist()))
            rows[-1]['contact_peak_event']=contact_peak_event[i]
            rows[-1]['max_applied_leg_target_substep_rad']=float(applied_step_peak[i])
            rows[-1]['ready_condition_fractions']=dict(zip(gate_names,
                (gate_counts[i]/gate_samples[i].clamp_min(1)).tolist()))
            rows[-1]['ready_condition_samples']=int(gate_samples[i])
            rows[-1]['wheel_motor_peak_nm']=wheel_motor_peak[i].tolist()
            rows[-1]['wheel_motor_samples']=int(wheel_motor_samples[i])
            if args.trace_stride:rows[-1]['trace']=traces[i]
        report=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            source_params_sha256=hashlib.sha256((args.checkpoint.resolve().parent/'params/env.yaml').read_bytes()).hexdigest(),
            training_reset_cohorts=[[int(v) for v in pair] for pair in saved.get('recovery_reset_cohorts',[])],
            evaluation_reset_mode='fixed_angle_four_directions_equal_replicas',
            evaluation_seed=cfg.seed,
            recovery_ground=dict(inclination_degrees=cfg.recovery_slope_degrees,
                uphill_azimuth_degrees=cfg.recovery_slope_azimuth_degrees,
                height_reference='local_ground_normal' if cfg.recovery_slope_degrees else 'flat_world_z',
                sampling='fixed_plane_per_run'),
            diagnostic_slope_degrees=args.diagnostic_slope_degrees,
            diagnostic_slope_azimuth_degrees=args.diagnostic_slope_azimuth_degrees,
            diagnostic_communication_delay_steps=args.diagnostic_communication_delay_steps,
            diagnostic_communication_delay_cap_steps=args.diagnostic_communication_delay_cap_steps,
            diagnostic_communication_delay_probability=args.diagnostic_communication_delay_probability,
            training_communication_delay_probability_override=float(saved.get('recovery_communication_delay_probability_override',-1.)),
            evaluation_communication_delay_probability_override=cfg.recovery_communication_delay_probability_override,
            training_communication_delay_cap_steps=int(saved.get('recovery_communication_delay_max_steps',-1)),
            evaluation_communication_delay_cap_steps=cfg.recovery_communication_delay_max_steps,
            sampled_communication_delay_steps_before_override=sampled_communication_delay.tolist(),
            diagnostic_dr_strength_override=args.diagnostic_dr_strength,
            diagnostic_hardware_profile=hardware_override,
            diagnostic_hardware_profile_sha256=(hashlib.sha256(args.diagnostic_hardware_profile.read_bytes()).hexdigest()
                if args.diagnostic_hardware_profile is not None else None),
            initial_dr_readback=initial_dr_readback,
            recovery_criteria=dict(min_height_m=cfg.recovery_min_height,
                upright_degrees=cfg.recovery_upright_degrees,hold_seconds=cfg.recovery_hold_seconds,
                max_planar_speed_mps=cfg.recovery_max_planar_speed,max_angular_speed_rad_s=cfg.recovery_max_angular_speed,
                wheel_net_force_threshold_n=2.),
            rows=rows,all_recovery_hold_passed=(args.diagnostic_zero_action_after is None and all(r['recovery_hold_passed'] for r in rows)),
            diagnostic_zero_action_after_seconds=args.diagnostic_zero_action_after,
            policy_behavior_evaluation=args.diagnostic_zero_action_after is None,
            wheel_motor_audit_scope='Applied motor torque snapshots before each physics action and at control boundaries; excludes constraint/passive spring forces',
            wheel_motor_limit_nm=float(cfg.robot_cfg.actuators['wheel_motors'].effort_limit),
            wheel_continuous_reference_nm=3.,
            handover_verified=False,maximum_dr_verified=False,certified=False,
            reward_profile=cfg.recovery_reward_profile,
            trace_stride_control_steps=args.trace_stride,
            trace_joint_order=[u.robot.joint_names[int(j)] for j in u._dof_ids],
            trace_leg_order=[u.robot.joint_names[int(j)] for j in u._leg_control_ids],
            gas_spring_assist_torque_nm=cfg.gas_spring_assist_torque_nm,
            gas_spring_radial_force_n=cfg.gas_spring_radial_force_n,
            gas_spring_parameters_source=('explicit diagnostic hardware profile' if hardware_override is not None
                else 'checkpoint params/env.yaml; not current defaults'),
            recovery_progress_observation=cfg.recovery_progress_observation,
            initial_tilt_degrees=cfg.recovery_initial_tilt_degrees,
            diagnostic_initial_tilt_override=args.diagnostic_initial_tilt,
            joint_limit_tolerance_rad=cfg.recovery_joint_limit_tolerance_rad,
            diagnostic_joint_tolerance_override=args.diagnostic_joint_tolerance,
            diagnostic_leg_target_slew_rate=args.diagnostic_leg_target_slew_rate,
            applied_leg_target_slew_rate_rad_s=slew_rate if math.isfinite(slew_rate) else None,
            motion_audit_scope='Control-boundary samples only; includes first target jump. Contact value is max body net force, not individual impact or substep peak.',
            scope=('Fixed starts with explicit DR/physics diagnostic; not complete competition DR or deployment certification. Terminal captured before reset.'
                if (args.diagnostic_dr_strength is not None or hardware_override is not None
                    or cfg.recovery_slope_degrees != 0.
                    or args.diagnostic_slope_degrees is not None) else
                'Fixed fallen starts, nominal dynamics, first complete episode per replica. Terminal metrics captured before automatic reset.'))
        if args.save_reference_rollout is not None:
            data={k:torch.cat([sample[k] for sample in reference_samples]) for k in reference_samples[0]}
            data.update(checkpoint_sha256=report['checkpoint_sha256'],
                evaluation_seed=cfg.seed,initial_tilt_degrees=cfg.recovery_initial_tilt_degrees,
                recovery_ground=report['recovery_ground'],recovery_criteria=report['recovery_criteria'],
                initial_dr_readback=initial_dr_readback,
                evaluation_communication_delay_cap_steps=cfg.recovery_communication_delay_max_steps,
                diagnostic_communication_delay_steps=args.diagnostic_communication_delay_steps,
                diagnostic_hardware_profile_sha256=report['diagnostic_hardware_profile_sha256'],
                terminal_passed=torch.tensor([r['recovery_hold_passed'] for r in rows]),
                scope='Delivered pre-step observations and raw deterministic actions every fourth control step; no policy update')
            args.save_reference_rollout.parent.mkdir(parents=True,exist_ok=True)
            with args.save_reference_rollout.open('xb') as f:torch.save(data,f)
            report['reference_rollout']=str(args.save_reference_rollout.resolve())
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
        print('[RECOVERY_RESULT]',json.dumps(rows),flush=True)
    finally:env.close()
except Exception:
    import traceback
    error=traceback.format_exc();print(error,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.with_suffix('.error.txt').write_text(error)
    raise
finally:app.close()
