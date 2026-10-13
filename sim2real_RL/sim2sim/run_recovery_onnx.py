"""Independent recovery ONNX / MuJoCo diagnostic with a strict action contract."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
import onnxruntime as ort
from audit_recovery_starts import make_starts
from leg_pd import leg_pd_torque, KP, KD, EFFORT_LIMIT_NM, PD_PERIOD_S
from radial_spring import RadialSpring
from equivalent_spring import apply as apply_equivalent_spring, validate_model, BODY_ORDER, JOINT_ORDER
from rmuc_policy_interface import build_observation, JOINT_NAMES, LEG_INDICES
from recovery_control import phase_progress, slew_target


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--diagnostic-initial-tilt',type=int,choices=[0,5,15,30,45,60,75,90],default=None)
    args=parser.parse_args();here=Path(__file__).resolve().parent
    contract=json.loads((args.export/'action_contract.json').read_text())
    if (contract.get('policy_kind') not in ('rmuc_recovery_v1','rmuc_recovery_v2') or contract['joint_order']!=list(JOINT_NAMES)
        or contract['observation_dim']!=25 or contract['action_dim']!=6
        or contract['observation_mode']!='tracking_error_with_target'
        or contract['command_observation_scale']!=[2.1,.5,1.]
        or contract['command']!=[0.,0.,0.] or contract['task_outer_feedback_required']):
        raise ValueError('Unsupported recovery observation/action contract')
    progress_enabled=contract.get('recovery_progress_observation',False)
    slew_rate=contract.get('leg_target_slew_rate_rad_s')
    if progress_enabled and (contract.get('policy_kind')!='rmuc_recovery_v2' or contract.get('progress_slot')!=7 or contract.get('progress_semantics')!='measured_orientation_rise_hold_v1'):
        raise ValueError('Unknown progress schema')
    if slew_rate is not None and (contract.get('policy_kind')!='rmuc_recovery_v2' or not np.isfinite(slew_rate) or slew_rate<=0 or contract.get('slew_reset')!='measured_leg_position'):
        raise ValueError('Unknown slew schema')
    for key,expected in [('kp',KP),('kd',KD),('leg_effort_limit_nm',EFFORT_LIMIT_NM),
        ('target_leg_velocity',0.),('leg_feedforward_nm',0.),('pd_dt',PD_PERIOD_S),
        ('policy_dt',.008)]:
        if contract[key]!=expected:raise ValueError(f'Unsupported contract {key}')
    wheel_peak=float(contract['wheel_effort_limit_nm'])
    wheel_scale=float(contract['wheel_action_scale_nm'])
    if not np.isfinite([wheel_peak,wheel_scale]).all() or not (0<wheel_peak<=5.5 and 0<wheel_scale<=5.5):
        raise ValueError('Invalid recorded wheel output-shaft peak/scale')
    if contract['leg_action_slots']!=LEG_INDICES or contract['wheel_action_slots']!=[2,5]:
        raise ValueError('Slot order mismatch')
    criteria=contract['recovery_criteria'];safety=contract['termination_limits']
    policy=args.export/'policy.onnx';digest=hashlib.sha256(policy.read_bytes()).hexdigest()
    if digest!=contract.get('onnx_sha256'):raise ValueError('ONNX/contract hash mismatch')
    xml=here/('rmuc_usd_limits.xml' if wheel_peak<=3. else 'rmuc_peak55.xml')
    model=mujoco.MjModel.from_xml_path(str(xml))
    if abs(model.opt.timestep-contract['pd_dt'])>1e-12:raise ValueError('PD period mismatch')
    joints=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in JOINT_NAMES])
    if min(joints)<0:raise ValueError('Missing joints')
    qids=model.jnt_qposadr[joints];vids=model.jnt_dofadr[joints]
    ids=[]
    for j in joints:
        matches=np.flatnonzero(model.actuator_trnid[:,0]==j)
        if len(matches)!=1:raise ValueError('Actuator mapping mismatch')
        ids.append(int(matches[0]))
    if not np.all(model.actuator_gear[ids,0]==1.):raise ValueError('Output-shaft Nm model required')
    for slot in (2,5):
        i=ids[slot]
        if (min(abs(model.actuator_ctrlrange[i]))<wheel_peak
                or min(abs(model.actuator_forcerange[i]))<wheel_peak):
            raise ValueError('MJCF would silently clip below the recorded peak')
    bounds=np.asarray(contract['leg_target_ranges_rad'])
    np.testing.assert_allclose(bounds,model.jnt_range[joints[LEG_INDICES]]+np.array([.02,-.02]),atol=1e-6,rtol=0)
    model.dof_damping[vids]=0.;model.dof_armature[vids[LEG_INDICES]]=.01
    safety_ids=[];safety_bounds=[]
    for entry in safety['joint_limits']:
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,entry['name'])
        if j<0:raise ValueError('Missing monitored joint '+entry['name'])
        safety_ids.append(model.jnt_qposadr[j])
        safety_bounds.append([entry['lower_rad'] if entry['lower_rad'] is not None else -np.inf,
                              entry['upper_rad'] if entry['upper_rad'] is not None else np.inf])
    safety_bounds=np.array(safety_bounds)
    spring=RadialSpring(model);root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot')
    equivalent_torque=float(contract.get('equivalent_assist_torque_nm',0.))
    spring_model=contract.get('gas_spring_model','legacy_radial_force_diagnostic')
    if spring_model not in ('equivalent_assist_joint_torque','legacy_radial_force_diagnostic'):
        raise ValueError('Unsupported gas-spring model')
    radial_force=float(contract.get('radial_force_n_per_leg',0.))
    if not all(np.isfinite(v) and v>=0 for v in (equivalent_torque,radial_force)):
        raise ValueError('Nonfinite or negative spring magnitude')
    if spring_model == 'equivalent_assist_joint_torque':
        if (contract.get('equivalent_assist_body_order')!=list(BODY_ORDER)
                or contract.get('equivalent_assist_joint_order')!=list(JOINT_ORDER)
                or contract.get('equivalent_assist_direction')!='active_link_pitch=-torque; robot=equal_opposite_reaction'):
            raise ValueError('Unknown equivalent spring direction or mapping')
        validate_model(model)
    if spring_model == 'equivalent_assist_joint_torque' and abs(radial_force)>1e-12:
        raise ValueError('Equivalent-torque contract cannot carry legacy radial force')
    if spring_model == 'legacy_radial_force_diagnostic' and abs(equivalent_torque)>1e-12:
        raise ValueError('Legacy radial contract cannot carry equivalent torque')
    wheels=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,s+'_wheel') for s in ['left','right']]
    ground=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'ground')
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(policy),sess_options=options,providers=['CPUExecutionProvider'])
    stride=round(contract['policy_dt']/model.opt.timestep);results=[]
    if abs(stride*model.opt.timestep-contract['policy_dt'])>1e-12:raise ValueError('Noninteger control stride')
    last_step=int(contract['first_episode_control_steps'])*stride
    if last_step<stride:raise ValueError('Invalid episode horizon')
    initial_tilt=(contract.get('training_initial_tilt_degrees',90) if args.diagnostic_initial_tilt is None
                  else args.diagnostic_initial_tilt)
    for start in make_starts(model,initial_tilt):
        data=mujoco.MjData(model);data.qpos[:]=start['qpos'];data.qvel[:]=start['qvel']
        default=np.zeros(6);history=np.zeros(6,dtype=np.float32);action=history.copy()
        hold_steps=0;longest_steps=0;trace=[];reason='duration_reached';peak=np.zeros(6)
        previous_target=data.qpos[qids[LEG_INDICES]].copy();target_step_peak=0.
        for step in range(last_step+1):
            mujoco.mj_forward(model,data)
            if step%stride==0:
                rotation=data.xmat[root].reshape(3,3);spatial=np.zeros(6)
                mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,root,spatial,0)
                def canonical(v):
                    b=rotation.T@v;return np.array([-b[2],b[0],b[1]])
                gravity=canonical(np.array([0.,0.,-1.]));angular=canonical(spatial[:3])
                forward=-rotation[:,2];speed=spatial[3:5]@forward[:2]/max(np.linalg.norm(forward[:2]),1e-6)
                obs=build_observation(angular_velocity_body=angular,gravity_body=gravity,
                    heading_forward_velocity=speed,command_vx=0.,command_yaw=0.,joint_position=data.qpos[qids],
                    joint_velocity=data.qvel[vids],default_joint_position=default,previous_executed_action=history)
                normal_vectors=np.zeros((2,3))
                for ci,c in enumerate(data.contact):
                    pair=[int(c.geom1),int(c.geom2)]
                    wrench=np.zeros(6);mujoco.mj_contactForce(model,data,ci,wrench)
                    normal=np.asarray(c.frame).reshape(3,3)[0]*wrench[0]
                    for side,geom in enumerate(pair):
                        body=int(model.geom_bodyid[geom])
                        if body in wheels:normal_vectors[wheels.index(body)]+=normal*(1. if side else -1.)
                loads=np.linalg.norm(normal_vectors,axis=1)
                tilt=float(np.degrees(np.arccos(np.clip(-gravity[2],-1,1))))
                ready=(tilt<criteria['upright_degrees'] and data.xpos[root,2]>criteria['min_height']
                       and np.linalg.norm(canonical(spatial[3:])[:2])<criteria['max_planar_speed']
                       and np.linalg.norm(angular)<criteria['max_angular_speed']
                       and (loads>criteria['wheel_normal_net_force_threshold_n']).all())
                if step>0:
                    hold_steps=hold_steps+1 if ready else 0
                    longest_steps=max(longest_steps,hold_steps)
                if progress_enabled:
                    obs[7]=phase_progress(-gravity[2],data.xpos[root,2],ready,
                        hold_steps*contract['policy_dt'],criteria['min_height'],
                        np.cos(np.deg2rad(criteria['upright_degrees'])),criteria['hold_seconds'])
                trace.append(dict(time=float(data.time),tilt_deg=tilt,height=float(data.xpos[root,2]),
                    wheel_normal_loads=loads.tolist(),ready=bool(ready),obs=obs.tolist(),
                    previous_executed_action=history.tolist(),next_action=None))
                q=data.qpos[safety_ids]
                failures=dict(joint_limit=bool(((q<safety_bounds[:,0])|(q>safety_bounds[:,1])).any()),
                    root_low=data.xpos[root,2]<safety['min_root_height'],
                    root_high=data.xpos[root,2]>safety['max_root_height'],
                    xy_distance=np.linalg.norm(data.xpos[root,:2])>safety['max_xy_distance'],
                    linear_speed=np.linalg.norm(spatial[3:])>safety['max_linear_speed'])
                if any(failures.values()):
                    reason=','.join(k for k,v in failures.items() if v);break
                if step==last_step:break
                raw=session.run(['actions'],{'obs':obs[None]})[0][0]
                if not np.isfinite(raw).all():raise ValueError('Nonfinite ONNX output')
                action=np.clip(raw,-1.,1.);history=action.copy()
                trace[-1]['next_action']=action.tolist()
            target=bounds.mean(axis=1)+(bounds[:,1]-bounds[:,0])*.5*action[LEG_INDICES]
            if slew_rate is not None:
                target=slew_target(target,previous_target,slew_rate,model.opt.timestep)
            delta=float(np.max(np.abs(target-previous_target)))
            target_step_peak=max(target_step_peak,delta)
            if slew_rate is not None and delta>slew_rate*model.opt.timestep+1e-10:
                raise ValueError('Target slew violation')
            previous_target=target.copy()
            effort=np.zeros(6);effort[LEG_INDICES]=leg_pd_torque(target,data.qpos[qids[LEG_INDICES]],data.qvel[vids[LEG_INDICES]])
            wheel_target=np.clip(np.asarray(contract['wheel_action_sign'])*wheel_scale*action[[2,5]],-wheel_peak,wheel_peak)
            effort[[2,5]]=np.clip(wheel_target-contract['wheel_explicit_damping']*data.qvel[vids[[2,5]]],-wheel_peak,wheel_peak)
            data.ctrl[ids]=effort;data.qfrc_applied[:]=0.
            if spring_model == 'legacy_radial_force_diagnostic':
                spring.apply(model,data,radial_force)
            if spring_model == 'equivalent_assist_joint_torque':
                apply_equivalent_spring(model,data,equivalent_torque)
            mujoco.mj_step(model,data)
            peak=np.maximum(peak,abs(data.qfrc_actuator[vids]))
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():reason='nonfinite';break
        results.append(dict(case=start['name'],seconds=float(data.time),stop_reason=reason,
            longest_ready_seconds=longest_steps*contract['policy_dt'],final_hold_seconds=hold_steps*contract['policy_dt'],
            final_established=hold_steps*contract['policy_dt']>=criteria['hold_seconds'],
            peak_motor_torque_nm=peak.tolist(),max_applied_target_substep_rad=target_step_peak,trace=trace))
    report=dict(onnx_sha256=digest,mjcf_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        initial_tilt_degrees=initial_tilt,diagnostic_initial_tilt_override=args.diagnostic_initial_tilt,
        contract=contract,rows=results,certified=False,handover_verified=False,
        scope='Contract-based recovery diagnostic; equivalent gas-spring torque is applied as passive body torque; solver contact forces and PhysX velocity constraints not proven equivalent')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['case','seconds','longest_ready_seconds','stop_reason']} for r in results],indent=2))


if __name__=='__main__':main()
