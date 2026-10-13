"""Virtual hip-to-wheel radial assistance matching training gas_spring.py."""
import mujoco
import numpy as np


def force_pair(hip, wheel, magnitude):
    delta=np.asarray(wheel)-np.asarray(hip)
    length=np.linalg.norm(delta)
    if not np.isfinite(delta).all() or not np.isfinite(magnitude) or magnitude<0 or length<=1e-6:
        raise ValueError('Invalid radial spring geometry or force')
    force=delta/length*magnitude
    return -force,force


class RadialSpring:
    def __init__(self,model):
        self.root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot')
        self.legs=[]
        for side in ('left','right'):
            hip=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'{side}_hip_pitch')
            wheel=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'{side}_wheel_spin')
            body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f'{side}_wheel')
            if min(self.root,hip,wheel,body)<0:raise ValueError('Missing spring attachment')
            self.legs.append((hip,wheel,body))

    def apply(self,model,data,magnitude):
        """Accumulate generalized forces; caller owns clearing qfrc_applied."""
        points=[]
        for hip,wheel,body in self.legs:
            h,w=data.xanchor[hip],data.xanchor[wheel]
            fh,fw=force_pair(h,w,magnitude)
            mujoco.mj_applyFT(model,data,fh,np.zeros(3),h,self.root,data.qfrc_applied)
            mujoco.mj_applyFT(model,data,fw,np.zeros(3),w,body,data.qfrc_applied)
            points.append(dict(length=float(np.linalg.norm(w-h)),force_n=float(np.linalg.norm(fw))))
        return points
