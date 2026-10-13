"""Diagnostic locomotion ONNX closed loop; not a certified Isaac-equivalent runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
import onnxruntime as ort

from rmuc_policy_interface import JOINT_NAMES, LEG_INDICES, POLICY_DT, build_observation, consume_action
from radial_spring import RadialSpring
from equivalent_spring import apply as apply_equivalent_spring, validate_model
from rmuc_controller import load_contract, LowLevelEffortScheduler, command_leg_scale
from command_schedule import CommandSchedule
from wheel_passive import (SEPARATED, FIELDS, validate_wheel_contract, passive_components,
                           feedback_damping_torque, feedback_damping_is_active_motor,
                           configure_passive_runtime, apply_passive)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', type=Path, default=Path(__file__).with_name('rmuc_peak55.xml'))
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--controller-contract', type=Path,
                        help='Verify policy/MJCF hashes and explicit actuator settings before replay')
    parser.add_argument('--seconds', type=float, default=2.)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--yaw', type=float, default=0.)
    parser.add_argument('--command-schedule', type=Path,
                        help='JSON list of {policy_step,vx,yaw}; overrides fixed speed/yaw, no resets')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physics-dt', type=float)
    parser.add_argument('--passive-damping', type=float)
    parser.add_argument('--sliding-friction', type=float)
    parser.add_argument('--wheel-action-scale', type=float, default=5.5)
    parser.add_argument('--wheel-action-sign', type=float, nargs=2, default=(-1., -1.),
                        metavar=('RIGHT', 'LEFT'), help='Policy-to-wheel output signs, in JOINT_NAMES wheel order.')
    parser.add_argument('--wheel-effort-limit', type=float, default=5.5,
                        help='Instantaneous output-shaft peak; 3Nm continuous rating is a separate thermal reference')
    parser.add_argument('--radial-force', type=float, default=0., help='Extension N per leg, separate from motor torque')
    parser.add_argument('--gas-spring-torque', type=float, required=True,
                        help='Equivalent passive assistance Nm per active leg joint; excludes radial force')
    parser.add_argument('--wheel-damping', type=float, help='Explicit value matching the selected policy experiment, Nm/(rad/s)')
    parser.add_argument('--wheel-torque-contract', choices=['legacy', SEPARATED])
    parser.add_argument('--wheel-feedback-damping-mode', choices=['postclip_loss_v1', 'active_motor_v1'])
    for key in FIELDS:
        parser.add_argument('--' + key.replace('_', '-'), type=float)
    parser.add_argument('--video', type=Path, default=None,
                        help='Optional MP4 path. Diagnostic visual recording of this MuJoCo rollout.')
    parser.add_argument('--video-fps', type=int, default=30)
    parser.add_argument('--video-width', type=int, default=960)
    parser.add_argument('--video-height', type=int, default=720)
    parser.add_argument('--video-follow', action='store_true', help='Follow chassis with a display-only camera.')
    args = parser.parse_args()
    controller_contract = None
    if args.controller_contract is not None:
        controller_contract = load_contract(args.controller_contract)
        if (args.policy.resolve() != Path(controller_contract['onnx'])
                or args.xml.resolve() != Path(controller_contract['mjcf'])):
            parser.error('Policy or MJCF differs from controller contract')
        actual = dict(wheel_action_scale_nm=args.wheel_action_scale,
                      wheel_peak_nm=args.wheel_effort_limit,
                      wheel_action_sign=list(args.wheel_action_sign),
                      gas_spring_torque_nm=args.gas_spring_torque,
                      radial_force_n=args.radial_force)
        if any(controller_contract[k] != v for k, v in actual.items()):
            parser.error('CLI dynamics differ from controller contract')
        if args.physics_dt is not None and args.physics_dt != controller_contract['physics_dt']:
            parser.error('Physics period differs from controller contract')
        for name, requested in (('passive_damping', args.passive_damping),
                                ('sliding_friction', args.sliding_friction)):
            recorded = controller_contract.get(name)
            if requested is not None and recorded is not None and requested != recorded:
                parser.error(f'{name} differs from controller contract')
    wheel_settings = {'wheel_torque_contract': args.wheel_torque_contract or
                      (controller_contract or {}).get('wheel_torque_contract', 'legacy')}
    feedback_mode = (controller_contract or {}).get('wheel_feedback_damping_mode')
    if args.wheel_feedback_damping_mode is not None:
        if controller_contract and args.wheel_feedback_damping_mode != (feedback_mode or 'postclip_loss_v1'):
            parser.error('CLI feedback damping mode differs from controller contract')
        feedback_mode = args.wheel_feedback_damping_mode
    if feedback_mode is not None:
        wheel_settings['wheel_feedback_damping_mode'] = feedback_mode
    for key in (*FIELDS, 'wheel_damping'):
        cli = getattr(args, key)
        recorded = (controller_contract or {}).get(key)
        if cli is not None and recorded is not None and cli != recorded:
            parser.error(f'CLI {key} differs from controller contract')
        if cli is not None or recorded is not None:
            wheel_settings[key] = cli if cli is not None else recorded
    if (controller_contract and args.wheel_torque_contract is not None and
            args.wheel_torque_contract != controller_contract.get('wheel_torque_contract', 'legacy')):
        parser.error('CLI wheel torque contract differs from controller contract')
    separated = wheel_settings['wheel_torque_contract'] == SEPARATED
    if separated:
        wheel_settings['hardware_sign_verified'] = False
        wheel_settings['leg_pd_dt'] = .002
        from rmuc_controller import separated_physics_dt
        expected_dt = separated_physics_dt(controller_contract or {})
        if args.physics_dt is not None and args.physics_dt != expected_dt:
            parser.error('Separated physics period differs from explicit contract')
        args.physics_dt = expected_dt
        if list(args.wheel_action_sign) != [-1., -1.]:
            parser.error('Separated contract inherits simulation signs [-1,-1]')
        if args.wheel_action_scale != 5.5 or args.wheel_effort_limit != 5.5:
            parser.error('Separated contract requires 5.5Nm scale and peak')
    try:
        validate_wheel_contract(wheel_settings)
    except ValueError as exc:
        parser.error(str(exc))
    args.wheel_damping = wheel_settings.get('wheel_damping')
    if not separated and args.wheel_damping is None:
        parser.error('Legacy mode requires --wheel-damping or a controller contract')
    if args.output.exists():
        parser.error('Refusing to overwrite an existing diagnostic report')
    if not np.isfinite(args.gas_spring_torque) or not 0 <= args.gas_spring_torque <= 10.775:
        parser.error('Equivalent gas spring torque must be finite and within [0,10.775] Nm')
    if args.gas_spring_torque and args.radial_force:
        parser.error('Equivalent torque and historical radial assistance cannot be combined')
    if not np.isfinite([args.seconds,args.speed,args.yaw]).all() or args.seconds <= 0:
        parser.error('Finite commands and positive duration required')
    try:
        schedule = (CommandSchedule.load(args.command_schedule) if args.command_schedule else
                    CommandSchedule([dict(policy_step=0, vx=args.speed, yaw=args.yaw)]))
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    if schedule.steps[-1] * POLICY_DT >= args.seconds:
        parser.error('Every command must begin before the requested duration')
    if not np.isfinite([args.wheel_action_scale,args.wheel_effort_limit]).all() or min(args.wheel_action_scale,args.wheel_effort_limit)<=0:
        parser.error('Wheel scale and output torque limit must be positive and finite')
    if not np.isfinite(args.wheel_action_sign).all() or any(abs(float(v)) != 1. for v in args.wheel_action_sign):
        parser.error('Wheel action signs must be exactly +/-1')
    if not np.isfinite(args.radial_force) or args.radial_force<0:
        parser.error('Radial force must be finite and nonnegative')
    if not separated and (not np.isfinite(args.wheel_damping) or args.wheel_damping<0):
        parser.error('Wheel damping must be finite and nonnegative')
    if args.video is not None:
        if args.video.exists():
            parser.error(f'Refusing to overwrite video {args.video}')
        if args.video_fps <= 0 or args.video_width <= 0 or args.video_height <= 0:
            parser.error('Video fps and dimensions must be positive')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(args.policy),sess_options=options,
                                   providers=['CPUExecutionProvider'])
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    if args.gas_spring_torque:
        validate_model(model)
    if args.sliding_friction is not None:
        if not np.isfinite(args.sliding_friction) or args.sliding_friction < 0:
            parser.error('sliding-friction must be finite and nonnegative')
        model.geom_friction[:,0] = args.sliding_friction
        model.pair_friction[:,:2] = args.sliding_friction
    if args.physics_dt is not None:
        if not np.isfinite(args.physics_dt) or args.physics_dt <= 0:
            parser.error('physics-dt must be positive and finite')
        model.opt.timestep = args.physics_dt
    passive_overrides = []
    if args.passive_damping is not None:
        if not np.isfinite(args.passive_damping) or args.passive_damping < 0:
            parser.error('passive-damping must be finite and nonnegative')
        for side in ('left','right'):
            for suffix in ('long_link_pitch','node_link_pitch','parallel_link_pitch','knee_pitch'):
                name = f'{side}_{suffix}'
                joint = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
                if joint < 0:
                    raise ValueError(f'Missing passive joint {name}')
                index = model.jnt_dofadr[joint]
                passive_overrides.append(dict(joint=name,old=float(model.dof_damping[index]),new=args.passive_damping))
                model.dof_damping[index] = args.passive_damping
    if controller_contract is not None:
        expected_damping = controller_contract.get('passive_damping')
        if expected_damping is not None:
            for side in ('left', 'right'):
                for suffix in ('long_link_pitch', 'node_link_pitch', 'parallel_link_pitch', 'knee_pitch'):
                    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{side}_{suffix}')
                    if joint < 0 or not np.isclose(model.dof_damping[model.jnt_dofadr[joint]], expected_damping,
                                                    rtol=0, atol=1e-9):
                        parser.error('Compiled passive joint damping differs from controller contract')
        expected_friction = controller_contract.get('sliding_friction')
        if expected_friction is not None and (not np.allclose(model.geom_friction[:, 0], expected_friction,
                                                              rtol=0, atol=1e-9)
                                              or not np.allclose(model.pair_friction[:, 0], expected_friction,
                                                                 rtol=0, atol=1e-9)):
            parser.error('Compiled sliding friction differs from controller contract')
    data = mujoco.MjData(model)
    video_proc = None
    renderer = None
    video_frames = 0
    spring = RadialSpring(model)
    dt = float(model.opt.timestep)
    if controller_contract is not None and dt != controller_contract['physics_dt']:
        parser.error('MJCF physics period differs from controller contract')
    stride = round(POLICY_DT/dt)
    if stride < 1 or abs(stride*dt-POLICY_DT) > 1e-9:
        raise ValueError('Physics timestep must divide 8ms policy interval')
    joints = np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in JOINT_NAMES])
    if (joints < 0).any():
        raise ValueError('Missing RMUC joint')
    qids, vids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    actuator_ids = []
    for j in joints:
        candidates = np.flatnonzero(model.actuator_trnid[:,0] == j)
        if len(candidates) != 1:
            raise ValueError('Expected one actuator per controlled joint')
        actuator_ids.append(int(candidates[0]))
    gears = model.actuator_gear[actuator_ids,0]
    if not np.isfinite(gears).all() or (abs(gears)<1e-12).any():
        raise ValueError('Invalid transmission')
    # Reject a declared wheel limit that the loaded model cannot deliver.
    # Source motor units and output-side units may differ by gear.
    for slot in [2,5]:
        i=actuator_ids[slot]
        capacities=[]
        if model.actuator_ctrllimited[i]:
            capacities.append(float(min(abs(model.actuator_ctrlrange[i]))*abs(gears[slot])))
        if model.actuator_forcelimited[i]:
            capacities.append(float(min(abs(model.actuator_forcerange[i]))*abs(gears[slot])))
        if capacities and args.wheel_effort_limit>min(capacities)+1e-8:
            raise ValueError('Requested wheel output limit exceeds loaded MJCF capability')
    # Explicit diagnostic runtime overrides, source XML is never modified.
    # Only actuated damping is removed to avoid doubling the explicit PD below.
    original_damping = model.dof_damping[vids].copy()
    wheel_native_passive = configure_passive_runtime(model, vids[[2, 5]]) if separated else None
    model.dof_damping[vids] = 0.
    model.dof_armature[vids[LEG_INDICES]] = .01
    body = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot')
    video_camera = None
    if args.video is not None:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.video_width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.video_height)
        renderer = mujoco.Renderer(model, height=args.video_height, width=args.video_width)
        if args.video_follow:
            video_camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(video_camera)
            video_camera.distance = 1.7
            video_camera.azimuth = 135.
            video_camera.elevation = -18.
        video_proc = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s',
            f'{args.video_width}x{args.video_height}', '-r', str(args.video_fps), '-i', '-',
            '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(args.video)],
            stdin=subprocess.PIPE)
    mujoco.mj_forward(model,data)
    default = data.qpos[qids].copy()
    history = np.zeros(6,dtype=np.float32)
    action = history.copy()
    effort_contract = controller_contract or dict(leg_scale_rad=.22,
        wheel_peak_nm=args.wheel_effort_limit, wheel_action_scale_nm=args.wheel_action_scale,
        wheel_action_sign=list(args.wheel_action_sign), **wheel_settings)
    effort_scheduler = LowLevelEffortScheduler(effort_contract)
    rows=[]
    torque_rows=[]
    stop_reason='duration_reached'
    max_torque = np.zeros(6)
    max_measured_torque = np.zeros(6)
    wheel_saturated_steps = 0
    max_leg_velocity = 0.
    for step in range(int(np.ceil(args.seconds/dt))):
        if step % stride == 0:
            segment, command = schedule.at(step // stride)
            # mj_step advances qpos/qvel after computing derived fields.
            # Refresh transforms/velocities so obs uses one consistent time.
            mujoco.mj_forward(model,data)
            rotation = data.xmat[body].reshape(3,3)
            spatial = np.zeros(6)
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,body,spatial,0)
            # Match training's authored root axes: canonical=(-z,x,y).
            def canonical(world):
                local = rotation.T@world
                return np.array([-local[2],local[0],local[1]])
            gravity = canonical(np.array([0.,0.,-1.]))
            omega = canonical(spatial[:3])
            forward = -rotation[:,2]
            heading_speed = float(spatial[3:5]@forward[:2]/max(np.linalg.norm(forward[:2]),1e-6))
            obs = build_observation(angular_velocity_body=omega,gravity_body=gravity,
                heading_forward_velocity=heading_speed,command_vx=command['vx'],command_yaw=command['yaw'],
                joint_position=data.qpos[qids],joint_velocity=data.qvel[vids],
                default_joint_position=default,previous_executed_action=history,
                yaw_error_clip=controller_contract.get('command_yaw_observation_error_clip', 2.0)
                    if controller_contract else 2.0)
            raw = session.run([session.get_outputs()[0].name],{session.get_inputs()[0].name:obs[None]})[0][0]
            action = consume_action(obs,raw)
            history = action.copy()
            tilt = float(np.degrees(np.arccos(np.clip(-gravity[2],-1,1))))
            rows.append(dict(time=float(data.time),height=float(data.xpos[body,2]),tilt_deg=tilt,
                policy_step=step // stride,command_segment=segment,
                command=[command['vx'],command['yaw']],leg_scale_rad=command_leg_scale(effort_contract, command['vx']),raw_action=raw.tolist(),
                heading_speed=heading_speed,yaw_rate=float(omega[2]),contacts=int(data.ncon),
                closure_distance_m={f'{side}_{a}_{b}': float(np.linalg.norm(
                    data.site_xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,f'{side}_{a}_site')]
                    - data.site_xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,f'{side}_{b}_site')]))
                    for side in ('left','right') for a,b in [('node','upper'),('parallel','lower')]},
                deepest_contact_distance_m=min((float(data.contact[i].dist) for i in range(data.ncon)),default=None),
                contact_sliding_friction=[float(data.contact[i].friction[0]) for i in range(data.ncon)],
                self_contact_pairs=[[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(data.contact[i].geom[j])) for j in (0,1)] for i in range(data.ncon) if all(model.geom_bodyid[int(data.contact[i].geom[j])] != 0 for j in (0,1))],
                observation=obs.tolist(),action=action.tolist()))
            if tilt>45 or data.xpos[body,2]<.08:
                stop_reason='tilt_or_height_diagnostic_limit'; break
        torque, targets, wheel_unclipped = effort_scheduler.compute(
            action, data.qpos[qids], data.qvel[vids],
            default, model.jnt_range[joints], command_vx=command['vx'])
        wheel_saturated_steps += int((abs(wheel_unclipped)>=args.wheel_effort_limit).any())
        data.ctrl[actuator_ids] = np.clip(torque/gears,model.actuator_ctrlrange[actuator_ids,0],
                                         model.actuator_ctrlrange[actuator_ids,1])
        data.qfrc_applied[:] = 0.
        data.xfrc_applied[:] = 0.
        if args.radial_force:
            mujoco.mj_forward(model,data)
            spring.apply(model,data,args.radial_force)
        if args.gas_spring_torque:
            mujoco.mj_forward(model,data)
            apply_equivalent_spring(model,data,args.gas_spring_torque)
        omega = data.qvel[vids[[2, 5]]]
        viscous, coulomb = passive_components(effort_contract, omega)
        feedback_damping = feedback_damping_torque(effort_contract, omega)
        if separated:
            feedback_is_active_motor = feedback_damping_is_active_motor(effort_contract)
            applied_feedback = np.zeros_like(feedback_damping) if feedback_is_active_motor else feedback_damping
            apply_passive(data, vids[[2, 5]], viscous + applied_feedback, coulomb)
            torque_rows.append(dict(time=float(data.time),
                wheel_velocity_rad_s=data.qvel[vids[[2, 5]]].tolist(),
                motor_torque_nm=torque[[2, 5]].tolist(),
                leg_motor_torque_nm=torque[LEG_INDICES].tolist(),
                leg_pd_updated=step % effort_scheduler.leg_stride == 0,
                feedback_damping_torque_nm=feedback_damping.tolist(),
                feedback_damping_mode=effort_contract.get('wheel_feedback_damping_mode', 'postclip_loss_v1'),
                mechanical_viscous_torque_nm=viscous.tolist(),
                coulomb_torque_nm=coulomb.tolist(),
                passive_torque_nm=(applied_feedback+viscous+coulomb).tolist(),
                net_shaft_torque_nm=(torque[[2, 5]]+applied_feedback+viscous+coulomb).tolist()))
        mujoco.mj_step(model,data)
        if renderer is not None:
            frame_stride=max(1,round(1./(args.video_fps*dt)))
            if step % frame_stride == 0:
                if video_camera is not None:
                    video_camera.lookat[:] = data.xpos[body]
                    renderer.update_scene(data, camera=video_camera)
                else:
                    renderer.update_scene(data)
                video_proc.stdin.write(np.asarray(renderer.render(),dtype=np.uint8).tobytes())
                video_frames += 1
        measured = data.qfrc_actuator[vids]
        if (abs(measured[[2,5]]) > args.wheel_effort_limit+1e-8).any():
            raise RuntimeError('Actual MuJoCo wheel actuator torque exceeds declared output limit')
        max_measured_torque=np.maximum(max_measured_torque,abs(measured))
        max_torque = np.maximum(max_torque,abs(torque))
        max_leg_velocity=max(max_leg_velocity,float(abs(data.qvel[vids[LEG_INDICES]]).max()))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            stop_reason='nonfinite'; break
    if video_proc is not None:
        video_proc.stdin.close()
        video_returncode=video_proc.wait(timeout=30)
        if video_returncode != 0:
            raise RuntimeError(f'ffmpeg video encoder failed with code {video_returncode}')
    report=dict(invocation=[sys.executable, *sys.argv],
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('run_mujoco_probe.py', 'rmuc_controller.py', 'wheel_passive.py',
                                    'rmuc_policy_interface.py', 'leg_pd.py')},
        policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        xml_sha256=hashlib.sha256(args.xml.read_bytes()).hexdigest(),xml=str(args.xml.resolve()),
        simulated_seconds=float(data.time),requested_seconds=args.seconds,stop_reason=stop_reason,
        command=None if args.command_schedule else [args.speed,args.yaw],
        command_schedule=schedule.entries,
        command_schedule_sha256=(hashlib.sha256(args.command_schedule.read_bytes()).hexdigest()
                                 if args.command_schedule else None),
        policy_dt=POLICY_DT,physics_dt=dt,low_level_dt=dt,policy_substeps=stride,leg_pd_dt=.002 if separated else dt,
        wheel_contract=wheel_settings,hardware_sign_verified=False,
        wheel_parameter_status=dict(feedback_damping='Diagnostic assumption; hardware unidentified',
            mechanical_viscous='Prior, not hardware measurement',
            coulomb_smoothing='Diagnostic smoothing, not hardware measurement'),
        wheel_native_passive_removed=wheel_native_passive,
        wheel_torque_trace=torque_rows,
        runtime_overrides=dict(controlled_joint_original_damping=original_damping.tolist(),
            passive_damping_overrides=passive_overrides,
            sliding_friction_override=args.sliding_friction,
            controlled_joint_damping=0.,leg_armature=.01,leg_kp=60.,leg_kd=2.,leg_scale=effort_contract['leg_scale_rad'],
            manual_leg_speed_knots=effort_contract.get('manual_leg_speed_knots'),
            manual_leg_scale_knots=effort_contract.get('manual_leg_scale_knots'),
            wheel_action_scale_nm=args.wheel_action_scale,
            radial_force_n_per_leg=args.radial_force,
            equivalent_spring_torque_nm_per_leg=args.gas_spring_torque,
            wheel_output_limit_nm=args.wheel_effort_limit,wheel_action_sign=list(args.wheel_action_sign),
            wheel_explicit_damping=args.wheel_damping),
        video=dict(path=str(args.video.resolve()),fps=args.video_fps,width=args.video_width,height=args.video_height,frames=video_frames)
            if args.video is not None else None,
        max_measured_actuator_output_nm=max_measured_torque.tolist(),
        wheel_saturated_physics_steps=wheel_saturated_steps,
        max_requested_output_torque_nm=max_torque.tolist(),max_leg_velocity=max_leg_velocity,
        controller_contract=controller_contract,
        certified=False,scope='Diagnostic nominal locomotion probe, not a passed behavior gate',
        unresolved=['Passive damping/contact/frame parity','PhysX leg 17rad/s velocity constraint equivalence',
                    'Matched Isaac reset and estimator validation','Final hardware behavior acceptance','DR'],
        trace=rows)
    tail = [r for r in rows if r['time'] >= args.seconds/2]
    report['second_half_metrics'] = (dict(samples=len(tail),
        mean_heading_speed=float(np.mean([r['heading_speed'] for r in tail])),
        mean_yaw_rate=float(np.mean([r['yaw_rate'] for r in tail])),
        mean_abs_speed_error=float(np.mean([abs(r['heading_speed']-r['command'][0]) for r in tail])),
        mean_abs_yaw_error=float(np.mean([abs(r['yaw_rate']-r['command'][1]) for r in tail]))) if tail else None)
    report['command_segments'] = []
    for index, command in enumerate(schedule.entries):
        samples = [row for row in rows if row['command_segment'] == index]
        report['command_segments'].append(dict(**command, samples=len(samples),
            reached=bool(samples),
            mean_abs_speed_error=(float(np.mean([abs(r['heading_speed']-command['vx'])
                                                 for r in samples])) if samples else None),
            mean_abs_yaw_error=(float(np.mean([abs(r['yaw_rate']-command['yaw'])
                                               for r in samples])) if samples else None)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['simulated_seconds','stop_reason','max_leg_velocity','certified']}))


if __name__ == '__main__':
    main()
