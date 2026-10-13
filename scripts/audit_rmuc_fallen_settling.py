"""Native settled-fallen state collection; no policy or recovery claim."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--motor-mode',choices=['hold_zero','legs_off'],default='hold_zero')
parser.add_argument('--seconds',type=float,default=3.)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();app=AppLauncher(args).app
try:
    import json,math,torch,gymnasium as gym
    import agent_tasks
    from agent_tasks.direct.rmuc_wheel_leg.recovery_env import RmucRecoveryEnvCfg
    cfg=RmucRecoveryEnvCfg(recovery_initial_tilt_degrees=90,
        recovery_reward_profile='recovery_v3',recovery_full_safety_scope=True,
        recovery_leg_target_slew_rate_rad_s=3.5)
    cfg.scene.num_envs=4;cfg.sim.device='cuda:0'
    cfg.domain_randomization_strength_override=0.
    env=gym.make('Robotics-RMUC-WheelLeg-Recovery-v0',cfg=cfg)
    try:
        u=env.unwrapped;env.reset()
        if args.motor_mode=='legs_off':
            u.robot.actuators['leg_motors'].stiffness.zero_()
            u.robot.actuators['leg_motors'].damping.zero_()
        # Preparation runs continuously; prevent automatic reset from hiding
        # a bad pose. Numerical validity and joint excursions are audited below.
        u._get_dones=lambda:(torch.zeros(4,device=u.device,dtype=torch.bool),
                             torch.zeros(4,device=u.device,dtype=torch.bool))
        nonwheel=[i for i in range(len(u.contact_sensor.body_names))
                  if i not in u._wheel_contact_sensor_ids]
        dwell=torch.zeros(4,device=u.device,dtype=torch.long)
        captured=[None]*4;trace=[]
        for step in range(math.ceil(args.seconds/u.step_dt)):
            with torch.inference_mode():env.step(torch.zeros(4,6,device=u.device))
            gravity,linear,angular=u._state()
            tilt=torch.rad2deg(torch.acos((-gravity[:,2]).clamp(-1,1)))
            q=u.robot.data.joint_pos.torch
            qd=u.robot.data.joint_vel.torch
            assert torch.isfinite(q).all() and torch.isfinite(u.robot.data.root_state_w.torch).all()
            force=u.contact_sensor.data.net_forces_w.norm(dim=-1)
            support=(force[:,nonwheel]>2.).any(-1)
            monitored=q[:,u._dof_ids];limits=u._joint_pos_limits
            excess=torch.maximum(limits[...,0]-monitored,monitored-limits[...,1]).amax(-1).clamp_min(0.)
            quiet=(linear.norm(dim=-1)<.1)&(angular.norm(dim=-1)<.2)&(qd[:,u._leg_control_ids].abs().amax(-1)<.2)
            valid=support&quiet&(tilt>=45)&(tilt<=135)&(excess<=1e-5)
            dwell=torch.where(valid,dwell+1,torch.zeros_like(dwell))
            records=[]
            for i in range(4):
                contacts=[dict(body=u.contact_sensor.body_names[j],force_n=float(force[i,j]))
                          for j in (force[i]>2.).nonzero().flatten().tolist()]
                record=dict(case=i,time_s=(step+1)*u.step_dt,tilt_degrees=float(tilt[i]),
                    ground_height_m=float(u._recovery_ground_height(u.robot.data.root_pos_w.torch)[i]),
                    linear_speed=float(linear[i].norm()),angular_speed=float(angular[i].norm()),
                    maximum_leg_speed=float(qd[i,u._leg_control_ids].abs().max()),
                    joint_excess_rad=float(excess[i]),contacts=contacts,valid=bool(valid[i]),
                    dwell_seconds=float(dwell[i]*u.step_dt))
                records.append(record)
                if captured[i] is None and dwell[i]*u.step_dt>=.2:
                    root=u.robot.data.root_state_w.torch[i].clone()
                    root[:3]-=u.scene.env_origins[i]
                    captured[i]=dict(record,root_state=root.tolist(),joint_position=q[i].tolist(),
                                     joint_velocity=qd[i].tolist())
            if step%10==0 or step+1==math.ceil(args.seconds/u.step_dt):trace.append(records)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        report=dict(motor_mode=args.motor_mode,initial_tilt_degrees=90,seconds=args.seconds,
            joint_names=u.robot.joint_names,candidates=captured,trace=trace,
            criteria=dict(non_main_wheel_contact_force_n=2.,max_linear_speed=.1,max_angular_speed=.2,
                max_leg_speed=.2,minimum_tilt=45,maximum_tilt=135,hold_seconds=.2,
                joint_numerical_tolerance=1e-5),
            certified=False,scope='Powered or motor-off preparation only. Contact with current simplified colliders; no learned self-righting, real hull coverage, or reset-replay certification.')
        with args.output.open('x') as f:json.dump(report,f,indent=2)
        print('settled candidates',sum(x is not None for x in captured),flush=True)
    finally:env.close()
except Exception:
    import traceback
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.error.txt').write_text(traceback.format_exc())
    raise
finally:app.close()
