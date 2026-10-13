"""Direct Isaac Lab audit of RMUC actuator channels and link inertias."""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset", choices=("rmuc", "v14"), default="rmuc")
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--pulse_index", type=int, default=1)
parser.add_argument("--pulse_effort", type=float, default=20.0)
parser.add_argument(
    "--differential_pair",
    action="store_true",
    help="Apply +effort to right active-link and -effort to right hip to expose redundant/locked leg DOFs.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.configclass import configclass

from agent_world.assets.rmuc_wheel_leg import RMUC_WHEEL_LEG_CFG, RMUC_CONTROLLED_DOF_NAMES
from agent_world.assets.wheelbipe_V14_2 import Wheelbipe_V14_2_CFG


RMUC_USD_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "agent_world"
    / "agent_world"
    / "assets"
    / "usd_files"
    / "rmuc_wheel_leg"
    / "rmuc_robot_closed_loop.usd"
)
V14_USD_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "agent_world"
    / "agent_world"
    / "assets"
    / "usd_files"
    / "wheelbipeV14_2_1"
    / "wheelbipeV14_2.usd"
)
V14_CONTROLLED_DOF_NAMES = (
    "right_front1_joint",
    "right_rear1_joint",
    "right_wheel_joint",
    "left_front1_joint",
    "left_rear1_joint",
    "left_wheel_joint",
)


@configclass
class AuditSceneCfg(InteractiveSceneCfg):
    # No terrain is needed for an actuator pulse audit; omitting it also keeps
    # the diagnostic independent of optional Isaac asset downloads.
    terrain = None
    robot = RMUC_WHEEL_LEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.002, device=args.device, render_interval=1))
    sim.set_camera_view(eye=[2.5, 2.5, 1.5], target=[0.0, 0.0, 0.2])
    scene_cfg = AuditSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)
    if args.asset == "v14":
        scene_cfg.robot = Wheelbipe_V14_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        scene_cfg.robot.spawn.usd_path = str(V14_USD_PATH)
        controlled_names = V14_CONTROLLED_DOF_NAMES
    else:
        scene_cfg.robot.spawn.usd_path = str(RMUC_USD_PATH)
        controlled_names = RMUC_CONTROLLED_DOF_NAMES
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot: Articulation = scene.articulations["robot"]
    print("[RMUC_AUDIT] joint_names=", robot.joint_names)
    print("[RMUC_AUDIT] asset=", args.asset)
    print("[RMUC_AUDIT] controlled_names=", list(controlled_names))
    control_ids = []
    for name in controlled_names:
        ids, _ = robot.find_joints(name)
        control_ids.append(int(ids[0]))
    print("[RMUC_AUDIT] control_ids=", control_ids)
    print("[RMUC_AUDIT] default_joint_pos=", robot.data.default_joint_pos.torch[0].cpu().tolist())
    print("[RMUC_AUDIT] body_names=", robot.body_names)
    if hasattr(robot.data, "body_mass"):
        print("[RMUC_AUDIT] body_mass=", robot.data.body_mass.torch[0].cpu().tolist())
    if hasattr(robot.data, "body_inertia"):
        print("[RMUC_AUDIT] body_inertia_shape=", tuple(robot.data.body_inertia.shape))

    effort = robot.data.default_joint_pos.torch.new_zeros((1, len(control_ids)))
    if args.pulse_index < 0 or args.pulse_index >= len(control_ids):
        raise ValueError(f"pulse_index must be in [0, {len(control_ids) - 1}]")
    effort[0, args.pulse_index] = float(args.pulse_effort)
    if args.differential_pair:
        effort.zero_()
        effort[0, 0] = float(args.pulse_effort)
        effort[0, 1] = -float(args.pulse_effort)
    before = robot.data.joint_pos.torch[0].clone()
    for step in range(max(args.steps, 1)):
        robot.set_joint_effort_target_index(target=effort, joint_ids=control_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    after = robot.data.joint_pos.torch[0].clone()
    print(
        "[RMUC_AUDIT] pulse_name=",
        "right_active_minus_right_hip" if args.differential_pair and args.asset == "rmuc"
        else "right_front1_minus_right_rear1" if args.differential_pair
        else controlled_names[args.pulse_index],
    )
    print("[RMUC_AUDIT] before=", before.cpu().tolist())
    print("[RMUC_AUDIT] after=", after.cpu().tolist())
    print("[RMUC_AUDIT] delta=", (after - before).cpu().tolist())
    print("[RMUC_AUDIT] applied_effort=", robot.data.applied_torque.torch[0].cpu().tolist())
    print("[RMUC_AUDIT] root_pos=", robot.data.root_pos_w.torch[0].cpu().tolist())
    sim.clear_instance()


if __name__ == "__main__":
    main()
    simulation_app.close()
