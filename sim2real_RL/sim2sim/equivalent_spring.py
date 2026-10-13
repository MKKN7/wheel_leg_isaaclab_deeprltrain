"""MuJoCo equivalent of the confirmed RMUC gas-spring joint torque model."""
import mujoco
import numpy as np


BODY_ORDER = ("robot", "left_active_link", "right_active_link")
JOINT_ORDER = ("left_active_link_pitch", "right_active_link_pitch")


def validate_model(model):
    """Require the same parent-local +X joints as the validated USD."""
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY_ORDER[0])
    if root < 0:
        raise ValueError('Missing spring chassis')
    for name, body_name in zip(JOINT_ORDER, BODY_ORDER[1:]):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if (min(joint, body)<0 or model.jnt_bodyid[joint]!=body
                or model.body_parentid[body]!=root
                or model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError('Equivalent spring joint/body mapping mismatch')
        rotation=np.empty(9)
        mujoco.mju_quat2Mat(rotation,model.body_quat[body])
        axis=rotation.reshape(3,3)@model.jnt_axis[joint]
        if not np.allclose(axis,[1.,0.,0.],atol=1e-10,rtol=0):
            raise ValueError('Equivalent spring requires parent-local +X joint axes')


def body_torques(model, data, magnitude_nm):
    """Return world torques [robot, left active link, right active link].

    Each active-link pitch joint receives -magnitude_nm at the output side.
    The chassis receives the equal and opposite reaction from both legs. This
    is a passive mechanical term and never enters the policy action or PD.
    """
    magnitude_nm = np.asarray(magnitude_nm,dtype=np.float64)
    if magnitude_nm.shape not in ((),(2,)) or not np.isfinite(magnitude_nm).all() or (magnitude_nm < 0.).any():
        raise ValueError("Equivalent spring torque must be finite and nonnegative")
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY_ORDER[0])
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY_ORDER[1])
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY_ORDER[2])
    if min(root, left, right) < 0:
        raise ValueError("Missing equivalent spring body")
    axis = np.asarray(data.xmat[root], dtype=np.float64).reshape(3, 3)[:, 0]
    if magnitude_nm.ndim==0:
        child = -magnitude_nm * axis
        return np.stack((-2. * child, child, child), axis=0)
    children=-magnitude_nm[:,None]*axis[None,:]
    return np.concatenate((-children.sum(0,keepdims=True),children),axis=0)


def apply(model, data, magnitude_nm):
    """Accumulate body torques into ``data.qfrc_applied``."""
    torques = body_torques(model, data, magnitude_nm)
    for name, torque in zip(BODY_ORDER, torques):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        mujoco.mj_applyFT(model, data, np.zeros(3), torque,
                          np.asarray(data.xipos[body]), body,
                          data.qfrc_applied)
    return torques
