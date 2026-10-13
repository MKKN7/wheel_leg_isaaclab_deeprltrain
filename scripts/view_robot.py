# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""
import sys
import argparse
import math
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_balls", type=int, default=36, help="Number of physics balls to spawn.")
parser.add_argument("--ball_radius_min", type=float, default=0.03, help="Min ball radius (m).")
parser.add_argument("--ball_radius_max", type=float, default=0.18, help="Max ball radius (m).")
parser.add_argument("--ball_spawn_height", type=float, default=1.0, help="Ball spawn Z height (m).")
parser.add_argument("--ball_grid_spacing", type=float, default=0.45, help="Grid spacing between ball centers (m).")
parser.add_argument("--ball_density", type=float, default=500.0, help="Ball material density (kg/m^3).")
parser.add_argument("--ball_friction", type=float, default=1.0, help="Ball friction for physics material.")
parser.add_argument("--ball_restitution", type=float, default=0.2, help="Ball restitution for physics material.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)

import agent_world  # noqa: F401
import agent_tasks  # noqa: F401


def _spawn_physics_balls_in_envs(env: gym.Env, *, num_balls: int, radius_min: float, radius_max: float,
                                  spawn_height: float, grid_spacing: float, density: float,
                                  friction: float, restitution: float) -> None:
    """Spawn many different-radius physics balls into each vectorized sub-environment.

    Notes:
    - Objects are spawned into USD with rigid body + collision APIs, so they will physically interact with
      the simulation.
    - We intentionally only need physics, not scene/tensor handles for these balls.
    """
    # Import inside function: this script is typically executed under Isaac Sim's python runtime.
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import use_stage

    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "scene") or not hasattr(unwrapped, "sim"):
        raise AttributeError(
            "Current environment does not expose 'scene'/'sim' on env.unwrapped, "
            "cannot spawn physics balls dynamically."
        )
    scene = unwrapped.scene
    env_regex_ns = scene.env_regex_ns  # e.g. "/World/envs/env_.*"

    # Use a unique id to avoid "prim already exists" errors on repeated script runs.
    ball_root_id = int(time.time())

    # Deterministic radius distribution: interpolate linearly across the index.
    if num_balls <= 0:
        return
    radius_min = float(radius_min)
    radius_max = float(radius_max)
    if radius_max <= 0.0:
        raise ValueError("ball_radius_max must be > 0.")

    # Grid layout in each environment (local coordinates).
    grid_n = int(math.ceil(math.sqrt(num_balls)))
    # Slight bias so balls are more likely to be in the robot's "forward" half.
    # (We don't know the robot forward axis here; this is a visualization-friendly default.)
    y_bias = grid_spacing * 0.5

    # Prebuild materials (same for all balls, only radius/mass changes).
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=float(friction),
        dynamic_friction=float(friction),
        restitution=float(restitution),
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        kinematic_enabled=False,
        disable_gravity=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    collision_props = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.0,
        rest_offset=0.0,
    )

    # Spawn: one sphere per index, cloned across all env instances by regex in prim_path.
    # spawn_sphere will create physics + collision on the geometry and apply rigid/mass properties.
    with use_stage(unwrapped.sim.get_initial_stage()):
        for k in range(num_balls):
            r = radius_min + (radius_max - radius_min) * (k / max(num_balls - 1, 1))
            mass = float(density) * (4.0 / 3.0) * math.pi * (r ** 3)

            row = k // grid_n
            col = k % grid_n
            x = (col - (grid_n - 1) / 2.0) * float(grid_spacing)
            y = (row - (grid_n - 1) / 2.0) * float(grid_spacing) + y_bias
            z = float(spawn_height)

            prim_path = f"{env_regex_ns}/PhysBall_{ball_root_id}_{k}"

            # Visual colors: map radius to a simple blue->red gradient.
            t = (r - radius_min) / max(radius_max - radius_min, 1e-8)
            diffuse_color = (float(t), 0.0, float(1.0 - t))
            visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=diffuse_color,
                metallic=0.0,
                roughness=0.5,
            )

            sphere_cfg = sim_utils.SphereCfg(
                radius=float(r),
                visual_material=visual_material,
                physics_material=physics_material,
                collision_props=collision_props,
                rigid_props=rigid_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            )

            # Spawn USD prims into the current stage context.
            sphere_cfg.func(prim_path, sphere_cfg, translation=(x, y, z))


def main():
    """Random actions agent with Isaac Lab environment."""
    # 清理任务名称，去除可能的前导 '=' 符号
    if args_cli.task and args_cli.task.startswith('='):
        args_cli.task = args_cli.task[1:]
        print(f"[INFO] 已清理任务名称，去除前导 '=' 符号: {args_cli.task}")
    
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Add custom physics objects before first reset.
    # _spawn_physics_balls_in_envs(
    #     env,
    #     num_balls=int(args_cli.num_balls),
    #     radius_min=float(args_cli.ball_radius_min),
    #     radius_max=float(args_cli.ball_radius_max),
    #     spawn_height=float(args_cli.ball_spawn_height),
    #     grid_spacing=float(args_cli.ball_grid_spacing),
    #     density=float(args_cli.ball_density),
    #     friction=float(args_cli.ball_friction),
    #     restitution=float(args_cli.ball_restitution),
    # )

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
