"""Nominal v261 policy interface, independent of ROS and simulation backend.

Inputs must already use the training root/body frame and joint signs. Sensor
noise/delay belongs before inference, actuator delay after action consumption.
This module does not implement an actuator, estimator or recovery controller.
"""
import numpy as np

JOINT_NAMES = (
    "right_active_link_pitch", "right_hip_pitch", "right_wheel_spin",
    "left_active_link_pitch", "left_hip_pitch", "left_wheel_spin",
)
LEG_INDICES = [0, 1, 3, 4]
POLICY_DT = 0.008


def _vector(value, length, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite ({length},) vector")
    return value


def build_observation(*, angular_velocity_body, gravity_body,
                      heading_forward_velocity, command_vx, command_yaw,
                      joint_position, joint_velocity, default_joint_position,
                      previous_executed_action, yaw_error_clip=2.0):
    """25D tracking_error_with_target observation; no simulated noise/delay.

    gravity_body is unit projected world gravity (upright [0,0,-1]), not an
    accelerometer sample. Forward velocity is horizontal heading-frame speed,
    not body-x velocity during pitch/roll. History uses the training consumer's
    executed normalized action, not raw ONNX output or torque in Nm.
    """
    omega = _vector(angular_velocity_body, 3, "angular_velocity_body")
    gravity = _vector(gravity_body, 3, "gravity_body")
    pos = _vector(joint_position, 6, "joint_position")
    vel = _vector(joint_velocity, 6, "joint_velocity")
    default = _vector(default_joint_position, 6, "default_joint_position")
    history = _vector(previous_executed_action, 6, "previous_executed_action")
    scalars = _vector([heading_forward_velocity, command_vx, command_yaw], 3, "commands/state")
    speed, vx, yaw = scalars
    if not np.isfinite(yaw_error_clip) or yaw_error_clip < 2.0:
        raise ValueError('yaw_error_clip must be finite and >=2 rad/s')
    cmd = np.array([vx-speed, vx, yaw-omega[2]], dtype=np.float32) * np.array([2.1, .5, 1.], dtype=np.float32)
    cmd[:2] = np.clip(cmd[:2], -2., 2.)
    cmd[2] = np.clip(cmd[2], -yaw_error_clip, yaw_error_clip)
    return np.concatenate((omega*.25, gravity, cmd,
                           (pos-default)[LEG_INDICES], vel*.05, history)).astype(np.float32)


def consume_action(observation, action):
    """Exact nominal authority clipping; physical joint limits remain required."""
    obs = _vector(observation, 25, "observation")
    action = _vector(action, 6, "action")
    speed = obs[7] / np.float32(.5)
    yaw = obs[2] / np.float32(.25) + obs[8]
    # Mirror env._compute_policy_action, not the smooth actor/helper gates:
    # float rounding immediately next to support boundaries can differ.
    active = (speed > np.float32(.8) and speed < np.float32(1.2)
              and abs(yaw) > np.float32(.4501))
    limits = np.ones(6, dtype=np.float32)
    if active:
        limits[[0, 3]] = 1.+.15/.22
    return np.clip(action, -limits, limits)
