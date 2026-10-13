"""Single-environment RMUC closed-loop smoke test.

This deliberately runs before PPO.  It uses the bundled immutable USD and
checks that reset/zero-action stepping produces finite observations, rewards,
wheel/body contacts and no immediate safety termination.
"""
from __future__ import annotations

import argparse
import copy
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--zero-joint-pose", action="store_true")
parser.add_argument("--root-z", type=float, default=None)
parser.add_argument("--pulse-index", type=int, default=None, help="Policy action channel to pulse for actuation diagnosis.")
parser.add_argument("--pulse-value", type=float, default=1.0)
parser.add_argument("--pulse-steps", type=int, default=20)
parser.add_argument("--constant-wheel-value", type=float, default=None)
parser.add_argument("--constant-right-wheel-value", type=float, default=None)
parser.add_argument("--constant-left-wheel-value", type=float, default=None)
parser.add_argument(
    "--constant-leg-value",
    type=float,
    default=None,
    help="Hold all four V14 leg action channels at this value for pulse-steps.",
)
parser.add_argument("--constant-right-leg-value", type=float, default=None)
parser.add_argument("--constant-left-leg-value", type=float, default=None)
parser.add_argument("--constant-leg-values", type=str, default=None, help="Four comma-separated values: right active,right hip,left active,left hip.")
parser.add_argument("--standing-profile", action="store_true")
parser.add_argument("--leg-scale", type=float, default=None)
parser.add_argument("--trace-bodies", action="store_true", help="Print key rigid-body positions during stepping.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply
import agent_tasks  # noqa: F401
from agent_tasks.direct.rmuc_wheel_leg.adapter import RmucWheelLegCourseEnvCfg
from agent_world.assets.rmuc_wheel_leg import RMUC_WHEEL_LEG_CFG, RMUC_CONTROLLED_DOF_NAMES
from agent_tasks.direct.rmuc_wheel_leg.env_cfg import ACTION_DIM, STANDING_JOINT_POS

_LOOP_POINTS = (
    ("left_node_link", "left_upper_leg", (0.005250, 0.088675, 0.069231), (0.009000, -0.064325, 0.069228)),
    ("left_parallel_link", "left_lower_leg", (0.006625, -0.078620, 0.084618), (-0.001500, 0.059840, 0.025380)),
    ("right_node_link", "right_upper_leg", (-0.005250, 0.088675, 0.069231), (-0.009000, -0.064325, 0.069228)),
    ("right_parallel_link", "right_lower_leg", (-0.006625, -0.078620, 0.084618), (0.001500, 0.059840, 0.025380)),
)


def _loop_residuals(robot):
    names = {name: index for index, name in enumerate(robot.body_names)}
    pos = robot.data.body_pos_w.torch
    quat = robot.data.body_quat_w.torch
    result = []
    for body0, body1, local0, local1 in _LOOP_POINTS:
        p0 = pos[:, names[body0]] + quat_apply(quat[:, names[body0]], torch.tensor(local0, device=pos.device))
        p1 = pos[:, names[body1]] + quat_apply(quat[:, names[body1]], torch.tensor(local1, device=pos.device))
        result.append(torch.linalg.vector_norm(p0 - p1, dim=-1))
    return torch.stack(result, dim=-1)


def main() -> None:
    cfg = RmucWheelLegCourseEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.robot_cfg = copy.deepcopy(RMUC_WHEEL_LEG_CFG)
    cfg.robot_cfg.prim_path = "/World/envs/env_.*/Robot"
    if args.leg_scale is not None:
        cfg.leg_position_scale_rad = float(args.leg_scale)
    if args.standing_profile:
        cfg.robot_cfg.init_state.joint_pos = dict(STANDING_JOINT_POS)
    # Optional diagnostic: pass ``--zero-joint-pose`` to distinguish a loop
    # residual problem from a general PhysX/articulation load problem.
    if getattr(args, "zero_joint_pose", False):
        os.environ["RMUC_RESET_OVERRIDE"] = "1"
        cfg.robot_cfg.init_state.pos = (0.0, 0.0, 0.35)
        cfg.robot_cfg.init_state.joint_pos = {".*": 0.0}
    if args.root_z is not None:
        os.environ["RMUC_RESET_OVERRIDE"] = "1"
        cfg.robot_cfg.init_state.pos = (0.0, 0.0, float(args.root_z))
    env = gym.make("Robotics-Wheelbipe-V14-RMUC-Flat-v0", cfg=cfg)
    obs, _ = env.reset()
    unwrapped = env.unwrapped
    wheel_ids = [unwrapped.robot.body_names.index("left_wheel"), unwrapped.robot.body_names.index("right_wheel")]
    trace_body_names = (
        "robot", "left_upper_leg", "left_lower_leg", "left_wheel",
        "right_upper_leg", "right_lower_leg", "right_wheel",
    )
    trace_body_ids = [unwrapped.robot.body_names.index(name) for name in trace_body_names]
    print("[RMUC_SMOKE] reset_loop_residual_m={} reset_wheel_heights={}".format(
        _loop_residuals(unwrapped.robot),
        unwrapped.robot.data.body_pos_w.torch[:, wheel_ids, 2],
    ))
    if args.trace_bodies:
        print("[RMUC_BODY_TRACE] step=reset names={} positions={}".format(
            trace_body_names,
            unwrapped.robot.data.body_pos_w.torch[0, trace_body_ids].detach().cpu().tolist(),
        ))
    original_reset_idx = unwrapped._reset_idx

    def _trace_reset(env_ids):
        gravity, vel, ang_vel = unwrapped._state()
        pos = unwrapped.robot.data.root_pos_w.torch
        quat = unwrapped.robot.data.root_quat_w.torch
        joint_pos = unwrapped.robot.data.joint_pos.torch[:, unwrapped._dof_ids]
        joint_vel = unwrapped.robot.data.joint_vel.torch[:, unwrapped._dof_ids]
        tilt = torch.rad2deg(torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0)))
        limits = unwrapped._joint_pos_limits
        violation = torch.any((joint_pos < limits[..., 0]) | (joint_pos > limits[..., 1]), dim=-1)
        print(
            "[RMUC_SMOKE] pre_reset ids={} pos={} vel={} tilt={} "
            "limit_violation={} estop={}".format(
                env_ids, pos, vel, tilt, violation, unwrapped._estop
            )
        )
        print("[RMUC_SMOKE] pre_reset_loop_residual_m={}".format(_loop_residuals(unwrapped.robot)))
        return original_reset_idx(env_ids)

    unwrapped._reset_idx = _trace_reset
    if not torch.isfinite(obs["policy"]).all():
        raise RuntimeError("non-finite policy observation immediately after reset")
    if args.pulse_index is not None:
        if args.pulse_index < 0 or args.pulse_index >= ACTION_DIM:
            raise ValueError(f"pulse-index must be in [0, {ACTION_DIM - 1}]")
        pulse = torch.zeros((1, ACTION_DIM), device=cfg.sim.device)
        pulse[0, args.pulse_index] = float(args.pulse_value)
        print(
            "[RMUC_ACTUATION] policy_order={} control_dof_ids={} before_joint_pos={}".format(
                list(RMUC_CONTROLLED_DOF_NAMES),
                getattr(unwrapped, "_control_dof_ids", None),
                unwrapped.robot.data.joint_pos.torch[0].detach().cpu().tolist(),
            )
        )
        for pulse_step in range(max(int(args.pulse_steps), 1)):
            obs, reward, terminated, truncated, _ = env.step(pulse)
            if pulse_step in (0, max(int(args.pulse_steps), 1) - 1):
                print(
                    "[RMUC_ACTUATION] step={} joint_pos={} joint_vel={} applied_effort={}".format(
                        pulse_step + 1,
                        unwrapped.robot.data.joint_pos.torch[0].detach().cpu().tolist(),
                        unwrapped.robot.data.joint_vel.torch[0].detach().cpu().tolist(),
                        unwrapped.robot.data.applied_torque.torch[0].detach().cpu().tolist(),
                    )
                )
            if bool(terminated.any()) or bool(truncated.any()):
                break
        env.reset()
    if args.constant_leg_value is not None or args.constant_right_leg_value is not None or args.constant_left_leg_value is not None or args.constant_leg_values is not None:
        constant_action = torch.zeros((1, ACTION_DIM), device=cfg.sim.device)
        if args.constant_leg_values is not None:
            values = [float(item.strip()) for item in args.constant_leg_values.split(",")]
            if len(values) != 4:
                raise ValueError("constant-leg-values requires four comma-separated values")
            constant_action[0, [0, 1, 3, 4]] = torch.tensor(values, device=cfg.sim.device)
            right_value, left_value = values[0], values[2]
        else:
            right_value = args.constant_right_leg_value if args.constant_right_leg_value is not None else args.constant_leg_value
            left_value = args.constant_left_leg_value if args.constant_left_leg_value is not None else args.constant_leg_value
            constant_action[0, [0, 1]] = float(right_value)
            constant_action[0, [3, 4]] = float(left_value)
        print(
            "[RMUC_STAND_TEST] right_value={} left_value={} steps={} before_root={} before_wheels={}".format(
                right_value,
                left_value,
                args.pulse_steps,
                unwrapped.robot.data.root_pos_w.torch,
                unwrapped.robot.data.body_pos_w.torch[:, wheel_ids, 2],
            )
        )
        for hold_step in range(max(int(args.pulse_steps), 1)):
            obs, reward, terminated, truncated, _ = env.step(constant_action)
            if bool(terminated.any()) or bool(truncated.any()):
                break
        print(
            "[RMUC_STAND_TEST] after_root={} after_wheels={} gravity={} joint_pos={}".format(
                unwrapped.robot.data.root_pos_w.torch,
                unwrapped.robot.data.body_pos_w.torch[:, wheel_ids, 2],
                unwrapped._state()[0],
                unwrapped.robot.data.joint_pos.torch[0, unwrapped._control_dof_ids],
            )
        )
        env.reset()
    if (
        args.constant_wheel_value is not None
        or args.constant_right_wheel_value is not None
        or args.constant_left_wheel_value is not None
    ):
        wheel_action = torch.zeros((1, ACTION_DIM), device=cfg.sim.device)
        shared = float(args.constant_wheel_value or 0.0)
        right_value = float(args.constant_right_wheel_value) if args.constant_right_wheel_value is not None else shared
        left_value = float(args.constant_left_wheel_value) if args.constant_left_wheel_value is not None else shared
        wheel_action[0, 2] = right_value
        wheel_action[0, 5] = left_value
        print("[RMUC_WHEEL_TEST] right={} left={} before_root={} before_wheel_joint_pos={}".format(
            right_value, left_value, unwrapped.robot.data.root_pos_w.torch,
            unwrapped.robot.data.joint_pos.torch[0, unwrapped._wheel_control_ids],
        ))
        for _ in range(max(int(args.pulse_steps), 1)):
            obs, reward, terminated, truncated, _ = env.step(wheel_action)
            if bool(terminated.any()) or bool(truncated.any()):
                break
        print("[RMUC_WHEEL_TEST] after_root={} root_vel_world={} canonical_vel={} canonical_ang_vel={} wheel_joint_pos={} wheel_joint_vel={} applied_effort={}".format(
            unwrapped.robot.data.root_pos_w.torch,
            unwrapped.robot.data.root_lin_vel_w.torch,
            unwrapped._state()[1],
            unwrapped._state()[2],
            unwrapped.robot.data.joint_pos.torch[0, unwrapped._wheel_control_ids],
            unwrapped.robot.data.joint_vel.torch[0, unwrapped._wheel_control_ids],
            unwrapped.robot.data.applied_torque.torch[0, unwrapped._wheel_control_ids],
        ))
        env.reset()
    terminated_count = 0
    for step in range(args.steps):
        action = torch.zeros((1, ACTION_DIM), device=cfg.sim.device)
        obs, reward, terminated, truncated, _ = env.step(action)
        if args.trace_bodies and step < 5:
            print("[RMUC_BODY_TRACE] step={} root={} positions={}".format(
                step + 1,
                unwrapped.robot.data.root_pos_w.torch[0].detach().cpu().tolist(),
                unwrapped.robot.data.body_pos_w.torch[0, trace_body_ids].detach().cpu().tolist(),
            ))
        if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
            raise RuntimeError(f"non-finite state at step {step}")
        terminated_count += int(terminated.sum().item())
        if bool(terminated.any()) and step == 0:
            unwrapped = env.unwrapped
            gravity, _, _ = unwrapped._state()
            tilt = torch.rad2deg(torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0)))
            print(
                "[RMUC_SMOKE] immediate_done terminated={} truncated={} estop={} root_pos={} "
                "root_vel={} joint_pos_finite={} joint_pos={} limits={}".format(
                    terminated,
                    truncated,
                    getattr(unwrapped, "_estop", None),
                    unwrapped.robot.data.root_pos_w.torch,
                    unwrapped.robot.data.root_lin_vel_w.torch,
                    torch.isfinite(unwrapped.robot.data.joint_pos).all().item(),
                    unwrapped.robot.data.joint_pos.torch,
                    unwrapped.robot.data.joint_pos_limits.torch,
                )
            )
            print("[RMUC_SMOKE] root_quat={} gravity={} tilt_deg={} episode_length={} max_episode_length={}".format(
                unwrapped.robot.data.root_quat_w.torch, gravity, tilt,
                unwrapped.episode_length_buf, unwrapped.max_episode_length,
            ))
        if bool(truncated.any()):
            env.reset()
    unwrapped = env.unwrapped
    print(
        "[RMUC_SMOKE] steps={} terminated={} controlled_dofs={} "
        "dof_count={} body_count={}".format(
            args.steps,
            terminated_count,
            list(RMUC_CONTROLLED_DOF_NAMES),
            unwrapped.robot.num_joints,
            len(unwrapped.robot.body_names),
        )
    )
    print(
        "[RMUC_SMOKE] final_root_pos={} wheel_heights={} contact_sensor_ready={}".format(
            unwrapped.robot.data.root_pos_w.torch,
            unwrapped.robot.data.body_pos_w.torch[:, wheel_ids, 2],
            getattr(unwrapped, "contact_sensor", None) is not None,
        )
    )
    print("[RMUC_SMOKE] final_gravity={} final_tilt_deg={}".format(
        unwrapped._state()[0],
        torch.rad2deg(torch.acos(torch.clamp(-unwrapped._state()[0][:, 2], -1.0, 1.0))),
    ))
    if getattr(unwrapped, "contact_sensor", None) is not None and getattr(unwrapped, "_contact_sensor_ids", None) is not None:
        forces = unwrapped.contact_sensor.data.net_forces_w[:, unwrapped._contact_sensor_ids]
        print("[RMUC_SMOKE] wheel_contact_force_norms={}".format(torch.linalg.vector_norm(forces, dim=-1)))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
