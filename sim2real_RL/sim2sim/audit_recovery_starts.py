"""Construct floor-safe fallen starts without changing the closed-loop model.

This audits initial geometry only. It does not prove that recovery is reachable
or train/use a recovery policy. Pose names refer to the authored assemblies.
"""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np


def make_starts(model, tilt_degrees=90.):
    if not np.isfinite(tilt_degrees) or not 0.<=tilt_degrees<=90.:
        raise ValueError('Tilt must be finite and within [0,90] degrees')
    root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot')
    ground=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'ground')
    if min(root,ground)<0:raise ValueError('Missing root/ground')
    colliders=[i for i in range(model.ngeom) if i!=ground and model.geom_bodyid[i]!=0
        and ((model.geom_contype[i]&model.geom_conaffinity[ground]) or
             (model.geom_contype[ground]&model.geom_conaffinity[i]))]
    if not colliders:raise ValueError('No floor collision shapes')
    definitions=[('front_down',[1.,0,0],-np.pi/2),('back_down',[1.,0,0],np.pi/2),
                 ('left_wheel_side_down',[0,1.,0],np.pi/2),('right_wheel_side_down',[0,1.,0],-np.pi/2)]
    result=[]
    for name,axis,angle in definitions:
        angle*=tilt_degrees/90.
        data=mujoco.MjData(model)
        if model.jnt_type[0]!=mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('Expected root freejoint as first joint')
        delta=np.r_[np.cos(angle/2),np.asarray(axis)*np.sin(angle/2)]
        quaternion=np.empty(4)
        mujoco.mju_mulQuat(quaternion,delta,data.qpos[3:7].copy())
        data.qpos[3:7]=quaternion
        mujoco.mj_forward(model,data)
        def distances():
            return [(i,float(mujoco.mj_geomDistance(model,data,ground,i,10.,None))) for i in colliders]
        lowest=min(d for i,d in distances())
        data.qpos[2]+=.003-lowest
        mujoco.mj_forward(model,data)
        measured=distances();gap=min(d for i,d in measured)
        assert abs(gap-.003)<1e-6
        closure=[]
        for side in ('left','right'):
            for a,b in [('node','upper'),('parallel','lower')]:
                ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,f'{side}_{part}_site') for part in (a,b)]
                if min(ids)<0:raise ValueError('Missing closed-loop sites')
                closure.append(float(np.linalg.norm(data.site_xpos[ids[0]]-data.site_xpos[ids[1]])))
        assert max(closure)<1e-6 and np.isfinite(data.qacc).all()
        result.append(dict(name=name,tilt_degrees=float(tilt_degrees),qpos=data.qpos.tolist(),qvel=data.qvel.tolist(),
            floor_gap_m=gap,closure_max_m=max(closure),
            nearest_floor_geoms=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i,d in measured if d<gap+.001],
            root_height=float(data.xpos[root,2]),
            scope='Valid pre-contact fallen pose, not a settled physical state or successful self-righting'))
    return result


def main():
    here=Path(__file__).resolve().parent
    xml=here/'rmuc_usd_limits.xml'
    model=mujoco.MjModel.from_xml_path(str(xml))
    starts=make_starts(model)
    report=dict(xml=xml.name,sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        topology_modified=False,actuator_commands_applied=False,starts=starts,
        recovery_certified=False,next='Settle/contact distribution and independent recovery action/termination contract before training')
    (here/'recovery_start_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['name','floor_gap_m','closure_max_m','nearest_floor_geoms','root_height']} for r in starts],indent=2))


if __name__=='__main__':main()
