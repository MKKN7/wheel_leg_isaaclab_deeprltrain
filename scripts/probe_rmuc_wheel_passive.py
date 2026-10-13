"""Bounded native wheel/leg execution interface probe, never a policy certificate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--num-envs',type=int,default=32)
p.add_argument('--steps',type=int,default=128)
p.add_argument('--seed',type=int,default=67)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
assert a.num_envs==32 and a.steps==128,'Fixed native probe budget'
assert not a.output.exists(),'Refuse overwrite'
expected_sha='5960533edf88c8a08a564ab86ca19e679faa49b44074f6cb3ba9873cfa1ad502'
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==expected_sha
root=Path(__file__).resolve().parents[1]
for package in ('agent_tasks','agent_world','agent_rl'):
    sys.path.insert(0,str(root/'source'/package))
app=AppLauncher(a).app
try:
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.utils.math import quat_apply
    import agent_tasks
    import agent_world
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from verify_rmuc_export import restore_actor
    cfg=parse_env_cfg('Robotics-Wheelbipe-V14-RMUC-Command-v0',device=a.device,num_envs=a.num_envs,use_fabric=True)
    cfg.seed=a.seed;cfg.play=True
    cfg.command_start_level=46;cfg.command_start_at_target=True
    cfg.command_stage_hold_steps=1<<60;cfg.command_rehearsal_enabled=False
    cfg.feedback_contract_by_command=True
    cfg.policy_outer_feedback_enabled=False;cfg.position_authority_enabled=True
    cfg.domain_randomization_enabled=False
    cfg.domain_randomization_strength_override=0.
    cfg.domain_randomization_apply_in_play=True
    cfg.domain_randomization_sensor_model={**cfg.domain_randomization_sensor_model, "enabled": False}
    cfg.motor_response_enabled=False
    cfg.gas_spring_assist_torque_nm=1.0775
    cfg.gas_spring_torque_common_dr_fraction=0.
    cfg.gas_spring_torque_independent_dr_fraction=0.
    cfg.gas_spring_radial_force_n=0.
    cfg.wheel_torque_contract='separated_passive_v1'
    cfg.wheel_mechanical_viscous_nm_s_rad=.002
    cfg.wheel_coulomb_friction_nm=.001
    cfg.wheel_coulomb_omega_eps_rad_s=.1
    cfg.wheel_feedback_damping_nm_s_rad=0.
    cfg.wheel_damping_anneal_target=-1.
    cfg.wheel_damping_compensation_ratio=0.
    cfg.wheel_torque_scale_nm=5.5
    cfg.robot_cfg.actuators['wheel_motors'].effort_limit=5.5
    cfg.sim.dt=.001;cfg.decimation=8;cfg.leg_pd_update_period_s=.002
    if cfg.sim.physics is None:
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
        cfg.sim.physics=PhysxCfg()
    cfg.sim.physics.enable_external_forces_every_iteration=True
    env=RslRlVecEnvWrapper(gym.make('Robotics-Wheelbipe-V14-RMUC-Command-v0',cfg=cfg),clip_actions=None)
    try:
        u=env.unwrapped
        actor=restore_actor(a.checkpoint).to(u.device).eval()
        assert hasattr(actor,'rmuc_position_authority')
        command=torch.tensor([[1.,0.,0.],[1.,0.,1.]],device=u.device).repeat_interleave(16,dim=0)
        u.set_manual_command(command)
        obs,_=env.reset()
        stats=dict(physics_calls=0,wheel_compute_calls=0,leg_hold_calls=0,
            max_passive_wrench_buffer_error_nm=0.,max_pair_reaction_error_nm=0.,
            max_motor_effort_nm=0.,max_passive_effort_nm=0.,max_passive_positive_power_w=0.,
            max_leg_hold_error_nm=0.,max_applied_motor_readback_nm=0.)
        samples=[]
        policy_trace=[]
        representatives=[0,16]
        current_policy_step=-1
        last_leg_targets=None
        prior_position_target=u.robot.set_joint_position_target
        def position_target(target,*args,**kw):
            global last_leg_targets
            if kw.get('joint_ids') is u._leg_control_ids:
                last_leg_targets=target.detach().clone()
            return prior_position_target(target,*args,**kw)
        u.robot.set_joint_position_target=position_target
        stats.update(max_wheel_axis_oracle_error_nm=0.,max_spring_axis_oracle_error_nm=0.,
                     max_physical_passive_positive_power_w=0.,max_relative_axis_speed_error_rad_s=0.)
        # USD frame metadata and installed quat_apply are independent of both
        # spring_body_torques and wheel_passive.rotate_axis under test.
        from pxr import Usd,UsdPhysics,Gf
        stage=Usd.Stage.Open(cfg.robot_cfg.spawn.usd_path)
        axes=[]
        for jid in u._wheel_control_ids:
            name=u.robot.joint_names[jid]
            prim=next(p for p in stage.Traverse() if p.GetName()==name)
            joint=UsdPhysics.RevoluteJoint(prim)
            vector=Gf.Vec3d(*(float(i=='XYZ'.index(joint.GetAxisAttr().Get())) for i in range(3)))
            vector=Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get())).TransformDir(vector)
            axes.append(tuple(vector))
        oracle_local_axes=torch.tensor(axes,device=u.device)
        oracle_spring_local_axis=torch.tensor([1.,0.,0.],device=u.device).expand(u.num_envs,-1)
        last_wheel_projection=None
        last_spring_projection=None
        prior_apply=u._apply_action
        def apply():
            global last_wheel_projection,last_spring_projection
            prior_apply()
            composer=u.robot.instantaneous_wrench_composer
            composer.compose_to_body_frame()
            passive=u._last_wheel_mechanical_torque_nm
            omega=u.robot.data.joint_vel.torch[:,u._wheel_control_ids]
            assert torch.isfinite(passive).all()
            stats['physics_calls']+=1
            stats['max_passive_effort_nm']=max(stats['max_passive_effort_nm'],float(passive.abs().max()))
            stats['max_passive_positive_power_w']=max(stats['max_passive_positive_power_w'],float((passive*omega).max()))
            for side,ids in enumerate(u._wheel_passive_body_ids):
                body=composer.out_torque_b.torch[:,ids]
                quat=u.robot.data.body_quat_w.torch[:,ids]
                world=quat_apply(quat.reshape(-1,4),body.reshape(-1,3)).reshape_as(body)
                expected=u._last_wheel_mechanical_body_torques_w[:,side]
                stats['max_passive_wrench_buffer_error_nm']=max(stats['max_passive_wrench_buffer_error_nm'],float((world-expected).abs().max()))
                stats['max_pair_reaction_error_nm']=max(stats['max_pair_reaction_error_nm'],float(world.sum(1).abs().max()))
                oracle_axis=quat_apply(quat[:,0],oracle_local_axes[side].expand(u.num_envs,-1))
                oracle_child=oracle_axis*passive[:,side,None]
                stats['max_wheel_axis_oracle_error_nm']=max(stats['max_wheel_axis_oracle_error_nm'],float((world[:,1]-oracle_child).abs().max()))
                relative=u.robot.data.body_ang_vel_w.torch[:,ids[1]]-u.robot.data.body_ang_vel_w.torch[:,ids[0]]
                power=(world[:,1]*relative).sum(-1)
                stats['max_physical_passive_positive_power_w']=max(stats['max_physical_passive_positive_power_w'],float(power.max()))
                stats['max_relative_axis_speed_error_rad_s']=max(stats['max_relative_axis_speed_error_rad_s'],float(((relative*oracle_axis).sum(-1)-omega[:,side]).abs().max()))
                if side==0:last_wheel_projection=torch.zeros_like(passive)
                last_wheel_projection[:,side]=(world[:,1]*oracle_axis).sum(-1)
            ids=u._gas_spring_body_ids
            body=composer.out_torque_b.torch[:,ids]
            quat=u.robot.data.body_quat_w.torch[:,ids]
            world=quat_apply(quat.reshape(-1,4),body.reshape(-1,3)).reshape_as(body)
            spring_axis=quat_apply(quat[:,0],oracle_spring_local_axis)
            oracle_children=-cfg.gas_spring_assist_torque_nm*spring_axis[:,None,:].expand(-1,2,-1)
            oracle_spring=torch.cat((-oracle_children.sum(1,keepdim=True),oracle_children),dim=1)
            stats['max_spring_axis_oracle_error_nm']=max(stats['max_spring_axis_oracle_error_nm'],float((world-oracle_spring).abs().max()))
            last_spring_projection=(world[:,1:]*spring_axis[:,None,:]).sum(-1)
        u._apply_action=apply
        wheel=u.robot.actuators['wheel_motors'];prior_wheel=wheel.compute
        def compute_wheel(*args,**kw):
            result=prior_wheel(*args,**kw)
            assert torch.isfinite(result.joint_efforts).all()
            stats['wheel_compute_calls']+=1
            stats['max_motor_effort_nm']=max(stats['max_motor_effort_nm'],float(result.joint_efforts.abs().max()))
            return result
        wheel.compute=compute_wheel
        leg=u.robot.actuators['leg_motors'];prior_leg=leg.compute
        hold=u._leg_pd_sample_hold
        start_count=hold.update_count;start_calls=hold.physics_calls
        def compute_leg(*args,**kw):
            result=prior_leg(*args,**kw)
            assert torch.isfinite(result.joint_efforts).all()
            if not hold.last_was_update:
                stats['leg_hold_calls']+=1
                stats['max_leg_hold_error_nm']=max(stats['max_leg_hold_error_nm'],float((result.joint_efforts-hold.held_applied_effort).abs().max()))
            if len(samples)<16:
                samples.append(dict(physics_call=hold.physics_calls,leg_pd_update=hold.last_was_update,
                    leg_pd_command_nm=hold.held_applied_effort[0].tolist(),leg_motor_output_nm=result.joint_efforts[0].tolist(),
                    wheel_passive_nm=u._last_wheel_mechanical_torque_nm[0].tolist()))
            return result
        leg.compute=compute_leg
        with torch.inference_mode():
            for step in range(a.steps):
                current_policy_step=step
                input_observation=obs['policy'][representatives].detach().clone()
                action=actor(obs,stochastic_output=False)
                raw_action=action[representatives].detach().clone()
                obs,reward,done,extra=env.step(action)
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                torque=u.robot.data.applied_torque.torch[:,u._wheel_control_ids]
                stats['max_applied_motor_readback_nm']=max(stats['max_applied_motor_readback_nm'],float(torque.abs().max()))
                policy_trace.append(dict(policy_step=step,env_ids=representatives,
                    observation=input_observation.tolist(),raw_action=raw_action.tolist(),
                    consumed_action=u._policy_actions[representatives].tolist(),
                    leg_targets_rad=last_leg_targets[representatives].tolist(),
                    motor_output_nm=u.robot.data.applied_torque.torch[representatives][:,u._control_dof_ids].tolist(),
                    wheel_passive_axis_torque_nm=last_wheel_projection[representatives].tolist(),
                    spring_axis_torque_nm=last_spring_projection[representatives].tolist(),
                    done_after_step=done[representatives].tolist(),
                    scope='Input obs/raw before step; targets/axial torques last physics substep; consumed action may reset on done'))
        stats.update(leg_compute_calls=hold.physics_calls-start_calls,leg_pd_updates=hold.update_count-start_count)
        checks=dict(physics_1ms=cfg.sim.dt==.001,policy_8ms=u.step_dt==.008,
            leg_pd_2ms=hold.update_period==.002 and stats['leg_pd_updates']==512,
            physics_calls=stats['physics_calls']==1024,wheel_1ms=stats['wheel_compute_calls']==1024,
            leg_hold=stats['leg_hold_calls']==512 and stats['max_leg_hold_error_nm']==0.,
            wrench_buffer=stats['max_passive_wrench_buffer_error_nm']<1e-5,
            reaction_pair=stats['max_pair_reaction_error_nm']<1e-5,
            dissipative=stats['max_passive_positive_power_w']<=1e-8,
            wheel_axis_oracle=stats['max_wheel_axis_oracle_error_nm']<1e-5,
            spring_axis_oracle=stats['max_spring_axis_oracle_error_nm']<1e-5,
            physical_passive_power=stats['max_physical_passive_positive_power_w']<1e-5,
            motor_limit=stats['max_motor_effort_nm']<=5.50001 and stats['max_applied_motor_readback_nm']<=5.50001,
            dr0=bool((u._domain_randomization_strength==0).all()),
            wheel_feedback_d0=bool((wheel.damping==0).all()))
        source=Path(__file__).resolve().parents[1]
        paths=[Path(__file__),source/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py',
            source/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env_cfg.py',
            source/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/wheel_passive.py',
            source/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/gas_spring.py',Path(cfg.robot_cfg.spawn.usd_path)]
        report=dict(passed=all(checks.values()),diagnostic_only=True,policy_certified=False,
            checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=expected_sha,
            source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            num_envs=a.num_envs,policy_steps=a.steps,simulated_seconds=a.steps*u.step_dt,
            checks=checks,stats=stats,substep_samples=samples,policy_trace=policy_trace,
            runtime=dict(wheel_torque_contract=cfg.wheel_torque_contract,physics_dt=cfg.sim.dt,
                wheel_quaternion_order="xyzw",gas_spring_quaternion_order=cfg.gas_spring_quaternion_order,
                decimation=cfg.decimation,leg_pd_period_s=hold.update_period,policy_dt=u.step_dt,
                mechanical_viscous_nm_s_rad=.002,coulomb_nm=.001,omega_epsilon_rad_s=.1,
                feedback_damping_nm_s_rad=0.,feedback_damping_scope='No-feedback diagnostic; hardware D unidentified',
                coulomb_smoothing_scope='Diagnostic regularization, not measured',
                motor_peak_nm=5.5,continuous_reference_nm=3.,action_scale_nm=5.5,
                spring_nm_per_leg=cfg.gas_spring_assist_torque_nm,spring_is_final_10_775=False,
                motor_response_enabled=cfg.motor_response_enabled,command_per_env=command.tolist(),
                wheel_joint_order=[u.robot.joint_names[i] for i in u._wheel_control_ids],
                leg_kp=leg.stiffness.tolist(),leg_kd=leg.damping.tolist(),
                mechanical_readback_scope='Native GPU wrench composer body-frame buffer transformed to world; not solver inverse dynamics',
                motor_readback_scope='Explicit actuator compute output and robot.data.applied_torque; excludes passive wrench'))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed':report['passed'],'checks':checks,'stats':stats}),flush=True)
        assert report['passed'],checks
    finally:
        env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush();sys.stderr.flush()
    raise
finally:
    app.close()
