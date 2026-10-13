"""CPU contact/closure diagnostic of fallen starts, with motors unpowered."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from audit_recovery_starts import make_starts
from radial_spring import RadialSpring


def main():
    here=Path(__file__).resolve().parent
    xml=here/'rmuc_usd_limits.xml'
    model=mujoco.MjModel.from_xml_path(str(xml));spring=RadialSpring(model)
    root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot')
    starts=make_starts(model);rows=[]
    leg_joint_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'{side}_{part}')
                   for side in ('left','right') for part in ('active_link_pitch','hip_pitch')]
    if min(leg_joint_ids)<0:raise ValueError('Missing active leg joints')
    leg_dofs=model.jnt_dofadr[leg_joint_ids]
    loops=[]
    for side in ('left','right'):
        for a,b in [('node','upper'),('parallel','lower')]:
            ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,f'{side}_{s}_site') for s in (a,b)]
            assert min(ids)>=0;loops.append(ids)
    for force in [0.,10.775]:
        for start in starts:
            data=mujoco.MjData(model);data.qpos[:]=start['qpos'];data.qvel[:]=start['qvel']
            max_gap=0.;max_speed=0.;max_leg_speed=0.;finite=True
            for _ in range(round(2./model.opt.timestep)):
                mujoco.mj_forward(model,data)
                data.ctrl[:]=0.;data.qfrc_applied[:]=0.
                if force:spring.apply(model,data,force)
                mujoco.mj_step(model,data)
                finite=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
                if not finite:break
                max_speed=max(max_speed,float(abs(data.qvel[6:]).max()))
                max_leg_speed=max(max_leg_speed,float(abs(data.qvel[leg_dofs]).max()))
                max_gap=max(max_gap,max(float(np.linalg.norm(data.site_xpos[a]-data.site_xpos[b])) for a,b in loops))
            mujoco.mj_forward(model,data)
            velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,root,velocity,0)
            contacts=[(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(c.geom1)),
                       mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(c.geom2))) for c in data.contact]
            settled=(finite and np.linalg.norm(velocity[:3])<.1 and np.linalg.norm(velocity[3:])<.05
                     and abs(data.qvel[6:]).max()<.2 and data.ncon>0)
            rows.append(dict(name=start['name'],radial_force_n_per_leg=force,simulated_seconds=float(data.time),
                finite=finite,settled_by_diagnostic_threshold=bool(settled),
                closure_peak_m=max_gap,max_joint_speed_rad_s=max_speed,
                max_active_leg_speed_rad_s=max_leg_speed,
                final_root_angular_speed=float(np.linalg.norm(velocity[:3])),
                final_root_linear_speed=float(np.linalg.norm(velocity[3:])),
                final_max_joint_speed_rad_s=float(abs(data.qvel[6:]).max()),
                final_max_active_leg_speed_rad_s=float(abs(data.qvel[leg_dofs]).max()),
                final_root_height=float(data.xpos[root,2]),contacts=contacts,
                qpos=data.qpos.tolist(),qvel=data.qvel.tolist()))
    report=dict(model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        motors='zero torque, no PD hold',topology_modified=False,rows=rows,
        scope='Fallen contact-state preparation only; no recovery action, reachability or policy success claimed')
    (here/'recovery_settle_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['name','radial_force_n_per_leg','finite',
        'settled_by_diagnostic_threshold','closure_peak_m','max_active_leg_speed_rad_s']} for r in rows],indent=2))


if __name__=='__main__':main()
