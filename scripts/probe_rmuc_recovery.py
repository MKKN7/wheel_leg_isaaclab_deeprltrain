"""Bounded four-environment native recovery initialization smoke, no training."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--reward-profile',choices=['legacy_v0','recovery_v1','recovery_v2','recovery_v3'],default='legacy_v0')
parser.add_argument('--progress-observation',action='store_true')
parser.add_argument('--initial-tilt',type=int,choices=[0,5,15,30,45,60,75,90],default=90)
parser.add_argument('--target-slew-rate',type=float,default=float('inf'))
parser.add_argument('--exercise-slew',action='store_true')
parser.add_argument('--slew-anneal-samples',type=int,default=0,
    help='Anneal duration in aggregate environment control transitions, not PPO updates.')
parser.add_argument('--slew-start-rate',type=float,default=4.)
parser.add_argument('--slew-end-rate',type=float,default=3.5)
parser.add_argument('--refall-penalty',type=float,default=0.)
parser.add_argument('--curriculum',action='store_true')
parser.add_argument('--curriculum-success-episodes',type=int,default=8)
parser.add_argument('--full-safety-scope',action='store_true')
parser.add_argument('--steps',type=int,default=128)
parser.add_argument('--audit-spring',action='store_true')
parser.add_argument('--dr-strength',type=float,default=0.)
parser.add_argument('--target-damping-dr',action='store_true')
parser.add_argument('--spring-common-dr',type=float,default=0.)
parser.add_argument('--spring-independent-dr',type=float,default=0.)
parser.add_argument('--reset-cohorts',type=str,default=None,help='JSON angle/direction pairs for explicit rehearsal layout.')
parser.add_argument('--slope-degrees',type=float,default=0.)
parser.add_argument('--slope-azimuth-degrees',type=float,default=0.)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();app=AppLauncher(args).app
try:
    import json
    import torch
    import gymnasium as gym
    import agent_tasks
    from agent_tasks.direct.rmuc_wheel_leg.recovery_env import RmucRecoveryEnvCfg
    print('[RECOVERY_SMOKE] imported task',flush=True)
    cfg=RmucRecoveryEnvCfg(recovery_reward_profile=args.reward_profile,recovery_refall_penalty=args.refall_penalty,
        recovery_initial_tilt_degrees=args.initial_tilt,recovery_leg_target_slew_rate_rad_s=args.target_slew_rate);cfg.scene.num_envs=4;cfg.sim.device='cuda:0'
    cfg.recovery_curriculum_enabled=args.curriculum
    if args.curriculum:cfg.recovery_curriculum_start_stage=cfg.recovery_curriculum_tilt_stages.index(args.initial_tilt)
    cfg.recovery_curriculum_max_stage=cfg.recovery_curriculum_start_stage
    cfg.recovery_curriculum_success_episodes=args.curriculum_success_episodes
    cfg.recovery_progress_observation=args.progress_observation
    cfg.recovery_slew_anneal_steps=args.slew_anneal_samples
    cfg.recovery_slew_start_rad_s=args.slew_start_rate
    cfg.recovery_slew_end_rad_s=args.slew_end_rate
    if args.reset_cohorts is not None:cfg.recovery_reset_cohorts=json.loads(args.reset_cohorts)
    cfg.recovery_full_safety_scope=args.full_safety_scope
    cfg.recovery_slope_degrees=args.slope_degrees
    cfg.recovery_slope_azimuth_degrees=args.slope_azimuth_degrees
    cfg.domain_randomization_strength_override=args.dr_strength
    cfg.gas_spring_torque_common_dr_fraction=args.spring_common_dr
    cfg.gas_spring_torque_independent_dr_fraction=args.spring_independent_dr
    if args.target_damping_dr:
        import yaml
        hardware=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/rmuc_hardware_target.yaml').read_text())
        damping=hardware['wheel']['damping']
        cfg.robot_cfg.actuators['wheel_motors'].damping=float(damping['proposed_center'])
        cfg.domain_randomization_max_wheel_damping_fraction=float(damping['full_dr_fraction'])
    cfg.sim.physics.enable_external_forces_every_iteration=True
    env=gym.make('Robotics-RMUC-WheelLeg-Recovery-v0',cfg=cfg)
    try:
        obs,_=env.reset();u=env.unwrapped
        initial=obs['policy'].clone()
        assert initial.shape==(4,25) and torch.isfinite(initial).all()
        gravity,_,_=u._state()
        tilt=torch.rad2deg(torch.acos((-gravity[:,2]).clamp(-1.,1.)))
        expected_tilt=(u._recovery_cohorts[:,0] if u._recovery_cohorts is not None else args.initial_tilt)
        if args.slope_degrees:
            assert args.initial_tilt==0 and args.reset_cohorts is None,'Slope smoke currently audits near-standing starts only'
            expected_tilt=abs(args.slope_degrees)
        assert bool(((tilt-expected_tilt).abs()<1.).all()),tilt
        direct_done,_=u._get_dones()
        assert not bool(direct_done.any()),'Fallen start incorrectly terminates immediately'
        terminated=torch.zeros(4,dtype=torch.long,device=u.device)
        target_peak=torch.zeros(4,device=u.device)
        previous_target=u.robot.data.joint_pos.torch[:,u._leg_control_ids].clone()
        substeps=0
        slew_samples=[]
        reset_counter_audit=[]
        double_contact_events=0.
        if args.exercise_slew:
            assert 0<args.target_slew_rate<float('inf')
            original_writer=u.robot.set_joint_position_target
            def writer(target,*pos,**kw):
                if kw.get('joint_ids') is u._leg_control_ids:
                    writer.last=target.detach().clone()
                return original_writer(target,*pos,**kw)
            u.robot.set_joint_position_target=writer
            original_apply=u._apply_action
            def apply():
                global substeps
                original_apply()
                delta=(writer.last-previous_target).abs().amax(-1)
                expected_rate=args.target_slew_rate
                if args.slew_anneal_samples>0:
                    fraction=min(max(u._global_step_counter/args.slew_anneal_samples,0.),1.)
                    expected_rate=args.slew_start_rate+(args.slew_end_rate-args.slew_start_rate)*fraction
                assert bool((delta<=expected_rate*cfg.sim.dt+1e-6).all()),delta
                slew_samples.append(dict(global_samples=u._global_step_counter,
                    expected_rate_rad_s=expected_rate,observed_peak_delta_rad=float(delta.max())))
                target_peak.copy_(torch.maximum(target_peak,delta))
                previous_target.copy_(writer.last);substeps+=1
            u._apply_action=apply
        spring_audit=dict(substeps=0,max_world_torque_error_nm=0.,max_net_world_torque_nm=0.)
        if args.audit_spring:
            from isaaclab.utils.math import quat_apply
            from agent_tasks.direct.rmuc_wheel_leg.gas_spring import spring_body_torques
            prior_apply=u._apply_action
            def audited_spring_apply():
                prior_apply()
                ids=u._gas_spring_body_ids
                assert ids is not None
                composer=u.robot.instantaneous_wrench_composer
                composer.compose_to_body_frame()
                body_torque=composer.out_torque_b.torch[:,ids]
                quats=u.robot.data.body_quat_w.torch[:,ids]
                world=quat_apply(quats.reshape(-1,4),body_torque.reshape(-1,3)).reshape_as(body_torque)
                expected=spring_body_torques(quats[:,0],u._gas_spring_torque_nm)
                error=float((world-expected).abs().max())
                net=float(world.sum(1).abs().max())
                assert error<1e-3 and net<1e-3
                spring_audit['substeps']+=1
                spring_audit['max_world_torque_error_nm']=max(spring_audit['max_world_torque_error_nm'],error)
                spring_audit['max_net_world_torque_nm']=max(spring_audit['max_net_world_torque_nm'],net)
            u._apply_action=audited_spring_apply
        ready_count=torch.zeros(4,device=u.device,dtype=torch.long)
        longest_ready=ready_count.clone()
        for step in range(args.steps):
            if args.exercise_slew and step==64:
                before_reset=u._global_step_counter
                obs,_=env.reset()
                assert u._global_step_counter==before_reset,'Episode reset restarted slew schedule'
                reset_counter_audit.append(dict(before=before_reset,after=u._global_step_counter))
                previous_target.copy_(u.robot.data.joint_pos.torch[:,u._leg_control_ids])
            action=torch.zeros(4,6,device=u.device)
            if args.exercise_slew:
                action[:,[0,1,3,4]]=1. if (step//8)%2==0 else -1.
            obs,reward,done,timeout,info=env.step(action)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            terminated+=done.long()
            ready=u.recovery_metrics()[2] & ~(done|timeout)
            ready_count=torch.where(ready,ready_count+1,torch.zeros_like(ready_count))
            longest_ready=torch.maximum(longest_ready,ready_count)
            double_contact_events+=float(u.extras['episode']['Recovery/double_contact_event_fraction'])*u.num_envs
            if args.exercise_slew and bool((done|timeout).any()):
                ids=(done|timeout).nonzero().flatten()
                previous_target[ids]=u.robot.data.joint_pos.torch[ids][:,u._leg_control_ids]
        leg=u.robot.actuators['leg_motors'];wheel=u.robot.actuators['wheel_motors']
        report=dict(initial_observation=initial.tolist(),initial_tilt_deg=tilt.tolist(),
            initial_done=direct_done.tolist(),termination_counts=terminated.tolist(),steps=args.steps,
            slope_degrees=args.slope_degrees,slope_azimuth_degrees=args.slope_azimuth_degrees,
            ground_normal=(u._recovery_ground_normal.tolist() if u._recovery_ground_normal is not None else None),
            env_origins=u.scene.env_origins.tolist(),
            final_ground_height_m=u._recovery_ground_height(u.robot.data.root_pos_w.torch).tolist(),
            spring_wrench_audit=spring_audit,
            longest_ready_seconds=(longest_ready*u.step_dt).tolist(),
            leg_kp=leg.stiffness.tolist(),leg_kd=leg.damping.tolist(),wheel_effort_limit=wheel.effort_limit.tolist(),
            final_ready=u.recovery_metrics()[2].tolist(),
            reward_profile=cfg.recovery_reward_profile,
            progress_observation=cfg.recovery_progress_observation,
            initial_tilt_degrees=cfg.recovery_initial_tilt_degrees,
            target_slew_rate_rad_s=args.target_slew_rate if args.target_slew_rate<float('inf') else None,
            recovery_curriculum_enabled=cfg.recovery_curriculum_enabled,
            recovery_curriculum_stage=u._recovery_tilt_stage.tolist(),
            reset_cohorts=cfg.recovery_reset_cohorts,
            recovery_cases=u._recovery_case.tolist(),
            full_safety_scope=cfg.recovery_full_safety_scope,
            double_contact_events=double_contact_events,
            gas_spring_assist_torque_nm=cfg.gas_spring_assist_torque_nm,
            gas_spring_radial_force_n=cfg.gas_spring_radial_force_n,
            actual_equivalent_spring_torque_nm=u._gas_spring_torque_nm.tolist(),
            requested_dr_strength=args.dr_strength,
            actual_dr_strength=u._domain_randomization_strength.tolist(),
            actual_wheel_damping=wheel.damping.tolist(),
            actual_mass_scale=u._domain_base_mass_scale.tolist(),
            actual_friction_scale=u._domain_friction_scale.tolist(),
            observation_delay_steps=u._sensor_delay_steps.tolist(),
            communication_delay_steps=u._communication_delay_steps.tolist(),
            maximum_dr_certified=False,
            refall_penalty=cfg.recovery_refall_penalty,
            refall_cost=float(u.extras['episode']['RecoveryReward/refall']),
            exercise_slew=args.exercise_slew,
            audited_substeps=substeps,
            slew_anneal_samples=args.slew_anneal_samples,
            slew_substep_audit=slew_samples,
            manual_reset_counter_audit=reset_counter_audit,
            max_applied_target_substep_rad=target_peak.tolist(),
            weighted_reward_terms={k:float(v) for k,v in u.extras.get('recovery_reward',{}).items()},
            scope='Recovery initialization and specified synthetic actions only; no learned recovery or policy handover',
            recovery_certified=False)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print('[RECOVERY_SMOKE]',json.dumps({'initial_tilt':tilt.tolist(),'terminations':terminated.tolist()}),flush=True)
    finally:
        env.close()
except Exception:
    import traceback
    error=traceback.format_exc()
    print(error,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.error.txt').write_text(error)
    raise
finally:
    app.close()
