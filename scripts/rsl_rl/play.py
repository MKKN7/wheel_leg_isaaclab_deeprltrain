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

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from rmuc_command_adapter import (
    checkpoint_has_actuator_bridge_adapter,
    checkpoint_has_domain_residual_bank,
    checkpoint_has_speed_isolated_domain_bank,
    checkpoint_has_directional_turn_delta_bank,
    checkpoint_has_speed_expert,
    checkpoint_has_turn_delta_bank,
    checkpoint_turn_adapter_hidden_dim,
    checkpoint_turn_delta_bank_size,
    install_rmuc_actuator_bridge_adapter,
    install_rmuc_domain_residual_bank,
    install_rmuc_speed_expert,
    install_rmuc_turn_adapter,
    install_rmuc_state_router,
)
from rmuc_policy_export import export_rmuc_policy

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--feedback_contract_by_command", action="store_true")
parser.add_argument("--low_speed_yaw_feedback_clip", type=float, default=None)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--command_level",
    type=int,
    default=None,
    help="RMUC play-only command curriculum level; applies the selected level immediately and holds it.",
)
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--keyboard_episode_length_s",
    type=float,
    default=None,
    help="Override episode length in keyboard play mode. If unset, keep the task play config value.",
)
parser.add_argument("--plot", action="store_true", default=False, help="Enable real-time data visualization (matplotlib).")
parser.add_argument("--expand_obs_dims", action="store_true", default=False, help="Whether to pad observation to four dims.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop play after this many steps; 0 runs until closed.")
parser.add_argument("--slip_debug", action="store_true", default=False, help="Print wheel-ground slip diagnostics.")
parser.add_argument("--slip_debug_interval", type=int, default=50, help="Steps between slip diagnostic prints.")
parser.add_argument("--slip_wheel_radius", type=float, default=None, help="Wheel radius used by slip diagnostics.")
parser.add_argument("--slip_speed_threshold", type=float, default=0.25, help="Horizontal contact-point speed treated as slip.")
parser.add_argument(
    "--barlow_twins_jit",
    type=str,
    default=None,
    help="TorchScript 路径（默认与导出一致：exported/barlow_twins_actor.pt）。"
    " 仅推理 MlpBarlowTwinsActor，需先 --checkpoint 加载同一任务以匹配 num_prop/num_hist。",
)
parser.add_argument(
    "--dreamwaq_print_code_vel",
    action="store_true",
    default=False,
    help="DreamWaq 任务下每步（或按间隔）打印 cenet 的 code_vel（需 OnPolicyDreamWaqRunner）。",
)
parser.add_argument(
    "--dreamwaq_code_vel_interval",
    type=int,
    default=1,
    help="与 --dreamwaq_print_code_vel 配合：每隔多少仿真步打印一次（默认 1=每步）。",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print(
    f"[PLAY] visualizer={getattr(app_launcher, '_cli_visualizer_types', None)} "
    f"headless={getattr(app_launcher, '_headless', None)} "
    f"experience={getattr(app_launcher, '_sim_experience_file', None)}",
    flush=True,
)

"""Rest everything follows."""

import warnings
import logging
# 抑制quat_rotate弃用警告，避免卡顿
# 这个警告来自isaaclab.utils.math，通过warnings和logging都可能输出
warnings.filterwarnings("ignore", message=".*quat_rotate.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="isaaclab.utils.math")
# 设置math模块的logging级别为ERROR，抑制WARNING
logging.getLogger("isaaclab.utils.math").setLevel(logging.ERROR)
# 添加自定义过滤器，过滤包含quat_rotate的警告消息
class QuatRotateFilter(logging.Filter):
    def filter(self, record):
        return "quat_rotate" not in str(record.getMessage())
logging.getLogger().addFilter(QuatRotateFilter())

import gymnasium as gym
import os
import time
import torch
import copy

import isaaclab.utils.math as math_utils
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
# 导入自定义键盘控制器（同目录下的文件）
import sys
# 添加当前目录到路径，以便导入同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keyboard_controller import Se3KeyboardMobile, Se3KeyboardMobileCfg

try:
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("[WARNING] RealtimePlotter not available. Install matplotlib to use --plot")
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import parse_env_cfg

from rsl_rl.runners import *  # noqa: F401
from agent_rl.rsl_rl.runners import *  # noqa: F401
import agent_world  # noqa: F401
import agent_tasks  # noqa: F401
try:
    from agent_rl.rsl_rl.modules import ActorCriticExt
except ImportError:
    ActorCriticExt = None
try:
    from agent_rl.rsl_rl.runners.on_constraint_policy_runner import OnConstraintPolicyRunner
except ImportError:
    OnConstraintPolicyRunner = None
try:
    from agent_rl.rsl_rl.runners.on_policy_runner_dreamwaq import OnPolicyDreamWaqRunner
except ImportError:
    OnPolicyDreamWaqRunner = None
try:
    from agent_rl.rsl_rl.runners.on_policy_runner_him import OnPolicyHIMRunner
except ImportError:
    OnPolicyHIMRunner = None


def _obs_tensor_for_barlow_twins(obs, device):
    """与训练一致：优先 ``on_constraint`` 展平向量（NP3O / BarlowTwins）。"""
    if isinstance(obs, dict):
        if "on_constraint" in obs:
            x = obs["on_constraint"]
        elif "policy" in obs:
            x = obs["policy"]
        else:
            raise KeyError(
                "BarlowTwins 推理需要观测含 'on_constraint' 或 'policy'，当前: " + str(list(obs.keys()))
            )
    else:
        x = obs
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    dev = torch.device(device) if isinstance(device, str) else device
    return x.to(dev, dtype=torch.float32)


class _BarlowTwinsActTeacherDictWrapper:
    """``act_teacher(flat_obs)`` → ``callable(obs_dict)``，适配 RslRlVecEnvWrapper 字典观测。"""

    def __init__(self, act_teacher, num_prop: int, num_hist: int, device):
        self._act_teacher = act_teacher
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.device = device

    def __call__(self, obs):
        x = _obs_tensor_for_barlow_twins(obs, self.device)
        return self._act_teacher(x)


class _BarlowTwinsTorchScriptPolicy:
    """加载 ``exporter_normal`` 导出的 TorchScript（两输入与 ``MlpBarlowTwinsActor.forward`` 一致）。"""

    def __init__(self, jit_model, num_prop: int, num_hist: int, device):
        self.model = jit_model
        self.model.eval()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.device = device

    def __call__(self, obs):
        x = _obs_tensor_for_barlow_twins(obs, self.device)
        obs_prop = x[:, : self.num_prop]
        obs_hist = x[:, -self.num_hist * self.num_prop :].view(-1, self.num_hist, self.num_prop)
        return self.model(obs_prop, obs_hist)


def _prepare_stock_rsl_cfg(agent_cfg) -> dict:
    """Normalize legacy Isaac Lab policy configs for RSL-RL 5.x playback."""
    cfg = agent_cfg.to_dict()
    policy = cfg.get("policy") or {}
    if cfg.get("actor") == {} and cfg.get("critic") == {} and policy:
        actor_norm = policy.get("actor_obs_normalization", False)
        critic_norm = policy.get("critic_obs_normalization", False)
        cfg["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": list(policy.get("actor_hidden_dims", [256, 256, 256])),
            "activation": str(policy.get("activation", "elu")),
            "obs_normalization": bool(actor_norm) if not isinstance(actor_norm, dict) else False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": float(policy.get("init_noise_std", 1.0)),
                "std_type": str(policy.get("noise_std_type", "scalar")),
            },
        }
        cfg["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": list(policy.get("critic_hidden_dims", [256, 256, 256])),
            "activation": str(policy.get("activation", "elu")),
            "obs_normalization": bool(critic_norm) if not isinstance(critic_norm, dict) else False,
            "distribution_cfg": None,
        }
    return cfg


def camera_follow(env):
    """Keep the camera on the robot's authored-frame rear heading.

    RMUC's forward direction is the authored local ``-Z`` axis (the same axis
    used by ``_heading_forward_velocity``). The previous camera used a local
    ``-X`` offset, so the view was lateral to the vehicle and made forward
    motion look like a backward slide.
    """
    if not hasattr(camera_follow, "smooth_camera_positions"):
        camera_follow.smooth_camera_positions = []
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]

    authored_forward = torch.zeros_like(robot_pos)
    authored_forward[2] = -1.0
    forward_w = math_utils.quat_apply(robot_quat, authored_forward)
    authored_side = torch.zeros_like(robot_pos)
    authored_side[0] = 1.0
    side_w = math_utils.quat_apply(robot_quat, authored_side)
    forward_xy = forward_w[:2]
    forward_xy = forward_xy / torch.linalg.vector_norm(forward_xy).clamp_min(1.0e-6)
    side_xy = side_w[:2]
    side_xy = side_xy / torch.linalg.vector_norm(side_xy).clamp_min(1.0e-6)

    if hasattr(env.unwrapped, "_heading_forward_velocity"):
        heading_speed = torch.abs(
            env.unwrapped._heading_forward_velocity(
                env.unwrapped.scene["robot"].data.root_lin_vel_w.torch
            )[0]
        )
    else:
        heading_speed = torch.linalg.vector_norm(
            env.unwrapped.scene["robot"].data.root_lin_vel_w[0, :2]
        )
    speed_ratio = torch.clamp(heading_speed / 3.0, 0.0, 1.0)
    side_distance = 2.8 + 0.8 * speed_ratio
    chase_height = 1.0 + 0.35 * speed_ratio
    rear_bias = 0.70 + 0.30 * speed_ratio
    lookahead_distance = 0.35 + 0.65 * speed_ratio
    camera_offset = torch.stack(
        (
            side_xy[0] * side_distance - forward_xy[0] * rear_bias,
            side_xy[1] * side_distance - forward_xy[1] * rear_bias,
            chase_height,
        )
    )
    camera_pos = robot_pos + camera_offset
    lookat_offset = torch.stack(
        (
            forward_xy[0] * lookahead_distance,
            forward_xy[1] * lookahead_distance,
            torch.as_tensor(0.30, device=env.device, dtype=robot_pos.dtype),
        )
    )
    lookat = robot_pos + lookat_offset
    if not hasattr(camera_follow, "_logged_frame"):
        print(
            "[RMUC_CAMERA] authored_forward_w="
            f"{forward_w.detach().cpu().tolist()} "
            "authored_side_w="
            f"{side_w.detach().cpu().tolist()} "
            "eye="
            f"{camera_pos.detach().cpu().tolist()} "
            "lookat="
            f"{lookat.detach().cpu().tolist()} "
            f"side_distance={float(side_distance):.2f} height={float(chase_height):.2f}",
            flush=True,
        )
        camera_follow._logged_frame = True

    # A short filter removes render jitter without leaving the robot a full
    # second behind during acceleration or a turn.
    window_size = 8
    camera_follow.smooth_camera_positions.append(camera_pos)
    if len(camera_follow.smooth_camera_positions) > window_size:
        camera_follow.smooth_camera_positions.pop(0)
    smooth_camera_pos = torch.mean(torch.stack(camera_follow.smooth_camera_positions), dim=0)
    vcc = env.unwrapped.viewport_camera_controller
    # Keep the camera in world-origin mode. Asset-root tracking callbacks can
    # otherwise overwrite the pose after this function returns.
    vcc.cfg.origin_type = "world"
    vcc.viewer_origin = torch.zeros(3)
    eye_np = smooth_camera_pos.cpu().numpy()
    lookat_np = lookat.cpu().numpy()
    vcc.update_view_location(eye=eye_np, lookat=lookat_np)
    # Also update the simulation camera directly so Kit and the viewport
    # controller use the same world-space pose.
    env.unwrapped.sim.set_camera_view(
        eye=tuple(float(value) for value in eye_np),
        target=tuple(float(value) for value in lookat_np),
    )


def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def compute_wheel_slip_diagnostics(env_u, wheel_radius: float | None, contact_threshold: float):
    """Estimate wheel slip from horizontal velocity of the wheel bottom contact point."""
    if not hasattr(env_u, "robot") or not hasattr(env_u, "_wheel_link_idx"):
        return None
    wheel_link_idx = list(getattr(env_u, "_wheel_link_idx", []))
    if len(wheel_link_idx) == 0:
        return None

    robot_data = env_u.robot.data
    if not hasattr(robot_data, "body_lin_vel_w") or not hasattr(robot_data, "body_ang_vel_w"):
        return None

    radius = wheel_radius
    if radius is None:
        radius = float(getattr(env_u.cfg, "height_reward_airborne_wheel_radius", 0.06))

    wheel_lin_vel_w = robot_data.body_lin_vel_w[:, wheel_link_idx]
    wheel_ang_vel_w = robot_data.body_ang_vel_w[:, wheel_link_idx]
    bottom_offset_w = torch.zeros_like(wheel_lin_vel_w)
    bottom_offset_w[..., 2] = -float(radius)
    contact_vel_w = wheel_lin_vel_w + torch.cross(wheel_ang_vel_w, bottom_offset_w, dim=-1)
    slip_speed = torch.linalg.norm(contact_vel_w[..., :2], dim=-1)

    contact_mask = torch.ones_like(slip_speed, dtype=torch.bool)
    contact_peaks = None
    if hasattr(env_u, "contact_sensor") and hasattr(env_u, "_get_wheel_contact_force_peaks"):
        net_forces_hist = getattr(getattr(env_u.contact_sensor, "data", None), "net_forces_w_history", None)
        if net_forces_hist is not None:
            contact_peaks = env_u._get_wheel_contact_force_peaks(net_forces_hist)
            wheel_count = min(contact_peaks.shape[-1], slip_speed.shape[-1])
            contact_mask = torch.zeros_like(slip_speed, dtype=torch.bool)
            if wheel_count > 0:
                contact_mask[:, :wheel_count] = contact_peaks[:, :wheel_count] > float(contact_threshold)

    if hasattr(env_u, "_wheel_idx"):
        wheel_joint_idx = list(getattr(env_u, "_wheel_idx", []))
        wheel_surface_speed = torch.abs(robot_data.joint_vel[:, wheel_joint_idx]) * float(radius)
    else:
        wheel_surface_speed = None
    wheel_center_speed = torch.linalg.norm(wheel_lin_vel_w[..., :2], dim=-1)

    return {
        "radius": float(radius),
        "slip_speed": slip_speed,
        "contact_mask": contact_mask,
        "contact_peaks": contact_peaks,
        "wheel_center_speed": wheel_center_speed,
        "wheel_surface_speed": wheel_surface_speed,
    }



def export_policy_as_onnx_exp(
        policy: object, obs: torch.Tensor, path: str, filename="policy.onnx", verbose=False,
        device="cpu"
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        obs_dict: input observation.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    os.makedirs(path, exist_ok=True)
    # copy policy parameters
    if hasattr(policy, "actor"):
        actor = copy.deepcopy(policy.actor).to(device)
    else:
        raise ValueError("Policy does not have an actor/student module.")
    torch.onnx.export(
                actor,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes=None,
            )

def main():
    """Play with RSL-RL agent."""
    # 清理任务名称，去除可能的前导 '=' 符号
    if args_cli.task and args_cli.task.startswith('='):
        args_cli.task = args_cli.task[1:]
        print(f"[INFO] 已清理任务名称，去除前导 '=' 符号: {args_cli.task}")
    
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.play = True
    if args_cli.feedback_contract_by_command:
        env_cfg.feedback_contract_by_command = True
    if args_cli.low_speed_yaw_feedback_clip is not None:
        if not 0.0 <= args_cli.low_speed_yaw_feedback_clip <= 1.0:
            raise ValueError("--low_speed_yaw_feedback_clip must be in [0,1]")
        env_cfg.wheel_yaw_feedback_low_speed_action_clip = args_cli.low_speed_yaw_feedback_clip
    if args_cli.command_level is not None:
        stage_names = getattr(env_cfg, "command_stage_names", ())
        command_level = int(args_cli.command_level)
        if not stage_names:
            raise ValueError("--command_level is only supported by staged RMUC command tasks")
        if not 0 <= command_level < len(stage_names):
            raise ValueError(
                f"--command_level must be in [0, {len(stage_names) - 1}], got {command_level}"
            )
        env_cfg.command_start_level = command_level
        env_cfg.command_transition_steps = 1
        env_cfg.command_stage_hold_steps = 1 << 60
        env_cfg.command_rehearsal_enabled = False
        print(
            f"[PLAY] Holding RMUC command level={command_level} name={stage_names[command_level]}",
            flush=True,
        )
    terrain_cfg = getattr(env_cfg, "terrain", None)
    if getattr(terrain_cfg, "terrain_type", None) == "generator" and hasattr(env_cfg, "play_terrain_debug_vis"):
        env_cfg.play_terrain_debug_vis = True
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # 配置键盘控制
    controller = None
    keyboard_config = None
    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        if args_cli.keyboard_episode_length_s is not None:
            env_cfg.episode_length_s = float(args_cli.keyboard_episode_length_s)
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "debug_vis"):
            env_cfg.commands.debug_vis = False
        # 键盘模式：关闭周期性跳跃，改为手动触发
        env_cfg.keyboard_manual_jump = True
        # 硬编码跳跃高度（手动跳跃时使用）
        env_cfg.manual_jump_height = 0.70
        # 硬编码手动跳跃窗口（步数）
        env_cfg.manual_jump_window = 25
        env_cfg.constant_spring_force = 250.0  # 测试高弹簧力
        # 硬编码速度和角速度灵敏度
        keyboard_config = Se3KeyboardMobileCfg(
            pos_sensitivity=2.0,      # 前后移动速度 (m/s)
            rot_sensitivity=2.0,     # 旋转角速度 (rad/s，约180度/秒)
            height_sensitivity=0.2,   # 高度增量灵敏度
        )
        controller = Se3KeyboardMobile(keyboard_config)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = os.path.abspath(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    # 键盘模式需要渲染窗口（render_mode=None会显示窗口）
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    
    # 键盘模式：在环境创建后初始化控制器
    if args_cli.keyboard and controller is not None:
        # 设置环境引用
        controller.set_env(env)
        
        # 初始化高度命令：从环境配置读取默认值，确保与环境一致
        default_height = getattr(env_cfg, "default_height_cmd", 0.24)
        if hasattr(env.unwrapped, "height_cmd"):
            # 使用环境当前的 height_cmd 值（可能已经在环境初始化时设置好了）
            current_env_height = float(env.unwrapped.height_cmd[0].item())
            # 优先使用环境当前值，其次使用配置值
            initial_height = current_env_height if current_env_height > 0.0 else default_height
            
            controller.set_default_height(initial_height)
            controller.set_current_height(initial_height)
            controller.set_height_range([0.15, 0.55])  # 高度范围限制
            print(f"[INFO] 初始高度命令: {initial_height:.3f} (从环境读取: {current_env_height:.3f}, 配置默认: {default_height:.3f})")
        else:
            print("[WARNING] 环境没有height_cmd属性，无法设置高度控制")
        
    
    # 键盘模式提示
    if args_cli.keyboard:
        print("[INFO] ========================================")
        print("[INFO] 键盘控制模式已启用")
        if controller is not None and keyboard_config is not None:
            print(f"[INFO] 键盘控制器已初始化: {controller}")
            print(f"[INFO] 速度灵敏度配置: pos={keyboard_config.pos_sensitivity:.3f}, rot={keyboard_config.rot_sensitivity:.3f}")
            print(f"[INFO] 高度灵敏度配置: height={keyboard_config.height_sensitivity:.4f}")
        else:
            print("[WARNING] 键盘控制器未初始化！")
        print("[INFO] 请确保Isaac Sim窗口有焦点才能接收键盘输入")
        print("[INFO] W键：加速（前进），S键：减速（后退）")
        print("[INFO] A键：左转，D键：右转")
        print("[INFO] Z键：增加高度，X键：降低高度（增量式）")
        print("[INFO] Q键：触发一次手动跳跃窗口（保持JUMP onehot）")
        print("[INFO] 松开ws/ad时速度归零，松开zx时高度保持")
        print("[INFO] ========================================")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    runner_class = eval(getattr(agent_cfg, "runner_class", "OnPolicyRunner"))
    # wrap around environment for rsl-rl
    if runner_class is OnPolicyRunner:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    else:
        from agent_rl.rsl_rl.env import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner_cfg = _prepare_stock_rsl_cfg(agent_cfg) if runner_class is OnPolicyRunner else agent_cfg.to_dict()
    runner = runner_class(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    adapter_hidden_dim = checkpoint_turn_adapter_hidden_dim(resume_path)
    if adapter_hidden_dim is not None:
        has_turn_delta_bank = checkpoint_has_turn_delta_bank(resume_path)
        has_directional_turn_bank = checkpoint_has_directional_turn_delta_bank(
            resume_path
        )
        install_rmuc_turn_adapter(
            runner.alg.get_policy(),
            hidden_dim=adapter_hidden_dim,
            freeze_base=False,
            enable_delta_bank=has_turn_delta_bank,
            enable_directional_bank=has_directional_turn_bank,
            bank_size=checkpoint_turn_delta_bank_size(resume_path),
        )
        print(
            "[INFO]: Loaded RMUC turn residual adapter "
            f"hidden_dim={adapter_hidden_dim}, delta_bank={has_turn_delta_bank}, "
            f"directional_bank={has_directional_turn_bank}",
            flush=True,
        )
    if checkpoint_has_actuator_bridge_adapter(resume_path):
        install_rmuc_actuator_bridge_adapter(
            runner.alg.get_policy(),
            hidden_dim=adapter_hidden_dim or 64,
        )
        print("[INFO]: Loaded RMUC actuator bridge residual", flush=True)
    if checkpoint_has_domain_residual_bank(resume_path):
        install_rmuc_domain_residual_bank(runner.alg.get_policy(),
            speed_isolated=checkpoint_has_speed_isolated_domain_bank(resume_path))
        print("[INFO]: Loaded RMUC bounded domain residual bank", flush=True)
    if checkpoint_has_speed_expert(resume_path):
        install_rmuc_speed_expert(runner.alg.get_policy())
        print("[INFO]: Loaded RMUC frozen 2.9 m/s straight expert", flush=True)
    router_payload = torch.load(resume_path, map_location='cpu', weights_only=False)
    router_state = router_payload.get('actor_state_dict', router_payload.get('model_state_dict', {}))
    if any(k.startswith('rmuc_state_router.') for k in router_state):
        install_rmuc_state_router(runner.alg.get_policy(), router_state)
        print('[INFO]: Loaded RMUC isolated state router', flush=True)
    del router_payload, router_state
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if adapter_hidden_dim is not None:
        runner.load(
            resume_path,
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
        )
    else:
        runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    play_device = env.unwrapped.device
    if runner_class is OnConstraintPolicyRunner:
        pol = runner.alg.policy
        if args_cli.barlow_twins_jit:
            jit_path = os.path.abspath(args_cli.barlow_twins_jit)
            print(f"[INFO] BarlowTwins TorchScript 推理: {jit_path}")
            jit_m = torch.jit.load(jit_path, map_location=play_device)
            policy = _BarlowTwinsTorchScriptPolicy(
                jit_m, int(pol.num_prop), int(pol.num_hist), play_device
            )
        else:
            policy = _BarlowTwinsActTeacherDictWrapper(
                pol.act_teacher, int(pol.num_prop), int(pol.num_hist), play_device
            )

    # Initialize the manual command before reset so the first observation is
    # generated with the same command that the keyboard controller exposes.
    # Resetting after observation creation used to leave a stale Level-18
    # command in the first policy input and caused an apparent reverse kick.
    if args_cli.keyboard and controller is not None:
        controller.reset()
    # reset environment
    obs, extras = env.reset()
    if args_cli.keyboard and controller is not None:
        print("[INFO] 初始环境重置完成，所有键盘命令已设置为默认值")

    # # export policy to onnx/jit.  RSL-RL 5.x stores the actor on ``alg.actor``
    # instead of the legacy ``alg.policy`` attribute.
    policy_module = getattr(runner.alg, "policy", None)
    if policy_module is None:
        policy_module = getattr(runner.alg, "actor", None)
    if policy_module is not None:
        policy_module.eval()
    if args_cli.keyboard:
        policy_obs = obs.get("policy") if hasattr(obs, "get") else obs
        actor_input_dim = None
        actor_module = getattr(policy_module, "actor", policy_module)
        first_layer = getattr(getattr(actor_module, "mlp", None), "__getitem__", lambda _: None)(0)
        if first_layer is not None:
            actor_input_dim = getattr(first_layer, "in_features", None)
        if not torch.is_tensor(policy_obs) or policy_obs.ndim != 2:
            raise RuntimeError(
                f"RMUC keyboard expects a [num_envs, obs_dim] tensor, got {type(policy_obs).__name__}"
            )
        if actor_input_dim is not None and policy_obs.shape[-1] != actor_input_dim:
            raise RuntimeError(
                f"RMUC observation/policy mismatch: env={policy_obs.shape[-1]} actor={actor_input_dim}"
            )
        print(
            "[RMUC_OBS_CONTRACT] "
            f"shape={tuple(policy_obs.shape)} "
            "layout=ang_vel[0:3],gravity[3:6],command[6:9],"
            "leg_pos[9:13],joint_vel[13:19],previous_action[19:25] "
            f"initial_command_slots={policy_obs[0, 6:9].detach().cpu().tolist()}",
            flush=True,
        )
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if runner_class is OnPolicyRunner:
        try:
            if hasattr(policy_module, "rmuc_turn_adapter"):
                export_rmuc_policy(policy_module, export_model_dir)
            elif args_cli.expand_obs_dims:
                exp_obs = torch.zeros(1, 1, 1, 28, device=play_device)
                export_policy_as_onnx_exp(
                    policy_module, exp_obs, path=export_model_dir, filename="policy.onnx"
                )
            elif hasattr(runner, "export_policy_to_jit"):
                runner.export_policy_to_jit(export_model_dir, filename="policy.pt")
                runner.export_policy_to_onnx(export_model_dir, filename="policy.onnx")
            else:
                from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
                normalizer = getattr(runner, "obs_normalizer", None)
                export_policy_as_jit(policy_module, normalizer, path=export_model_dir, filename="policy.pt")
                export_policy_as_onnx(
                    policy_module, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
                )
        except Exception as exc:
            print(f"[WARNING] Policy export skipped during playback: {exc}")
    elif runner_class is OnPolicyDreamWaqRunner:
        from agent_rl.rsl_rl.utils.exporter_normal import export_policy_as_jit, export_policy_as_onnx
        export_policy_as_jit(
            runner.alg.policy, obs, path=export_model_dir, filename="policy.pt"
        )
        export_policy_as_onnx(
            runner.alg.policy, obs, path=export_model_dir, filename="policy.onnx"
        )
    elif runner_class is OnPolicyHIMRunner:
        from agent_rl.rsl_rl.utils.exporter_normal import export_policy_as_jit, export_policy_as_onnx
        export_policy_as_jit(
            runner.alg.policy, obs, path=export_model_dir, filename="policy.pt"
        )
        export_policy_as_onnx(
            runner.alg.policy, obs, path=export_model_dir, filename="policy.onnx"
        )
    elif runner_class is OnConstraintPolicyRunner:
        from agent_rl.rsl_rl.utils.exporter_normal import export_barlow_twins_actor_from_policy

        try:
            pt_path, onnx_path = export_barlow_twins_actor_from_policy(
                runner.alg.policy,
                export_model_dir,
                device=str(agent_cfg.device),
            )
            print(f"[INFO] BarlowTwins actor TorchScript: {pt_path}")
            print(f"[INFO] BarlowTwins actor ONNX: {onnx_path}")
        except Exception as e:
            print(f"[WARNING] BarlowTwins actor 导出失败: {e}")
    else:
        print("No exporter provided.")
    # export_policy_as_onnx(runner,agent_cfg,args_cli,export_model_dir, obs)

    timestep = 0
    
    # 初始化实时绘图器
    plotter = None
    if args_cli.plot and PLOT_AVAILABLE:
        if env_cfg.scene.num_envs == 1:
            # 获取腿部关节数量
            num_leg_joints = 6  # 默认6个主动腿部关节
            # 获取特定的主动关节索引 (front1, rear1)
            active_joint_indices = []
            if hasattr(env.unwrapped, '_front1_joint_idx') and hasattr(env.unwrapped, '_rear1_joint_idx'):
                # 按照 L_Front1, L_Rear1, R_Front1, R_Rear1 的顺序尝试排列
                f1_idx = env.unwrapped._front1_joint_idx
                r1_idx = env.unwrapped._rear1_joint_idx
                # 假设索引 0 是左，1 是右 (根据 articulation 加载顺序)
                if len(f1_idx) >= 2 and len(r1_idx) >= 2:
                    active_joint_indices = [f1_idx[0], r1_idx[0], f1_idx[1], r1_idx[1]]
                else:
                    active_joint_indices = f1_idx + r1_idx
            
            num_plot_joints = len(active_joint_indices) if active_joint_indices else 4
            try:
                # 延迟导入以避免在 SimulationApp 启动前竞争图形上下文
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../utils'))
                from realtime_plotter import RealtimePlotter
                plotter = RealtimePlotter(max_points=50, num_leg_joints=num_plot_joints, update_interval=10)
                print(f"[INFO] Real-time plotter initialized (update_interval=10, max_points=50, joints={num_plot_joints}, wheel_power=True)")
            except Exception as e:
                print(f"[WARNING] 启动绘图器失败: {e}")
                plotter = None
            # 保存索引供后续使用
            if plotter is not None:
                plotter.active_joint_indices = active_joint_indices
        else:
            print("[WARNING] --plot only works with single environment")
    
    # 构建地形类型实时检测所需的数据
    # 通过机器人世界坐标 + terrain_origins 网格反推当前所在格子和地形类型
    # curriculum=True 保证同列所有行类型一致，col → 地形类型名称
    terrain_debug_info = None  # (x_start, y_start, cell_size, num_rows, num_cols, col_to_type)
    if (hasattr(env.unwrapped, 'terrain')
            and hasattr(env.unwrapped.terrain, 'terrain_origins')
            and env.unwrapped.terrain.terrain_origins is not None):
        _terrain = env.unwrapped.terrain
        if (hasattr(_terrain, 'cfg') and hasattr(_terrain.cfg, 'terrain_generator')
                and _terrain.cfg.terrain_generator is not None):
            import numpy as _np
            _sub_terrain_keys = list(_terrain.cfg.terrain_generator.sub_terrains.keys())
            _proportions = _np.array([v.proportion for v in _terrain.cfg.terrain_generator.sub_terrains.values()], dtype=float)
            _proportions /= _proportions.sum()
            _num_cols = _terrain.cfg.terrain_generator.num_cols
            _num_rows = _terrain.cfg.terrain_generator.num_rows
            _cell_size = float(_terrain.cfg.terrain_generator.size[0])
            _cumsum = _np.cumsum(_proportions)
            _col_to_type = []
            for _col in range(_num_cols):
                _sub_idx = int(_np.min(_np.where(_col / _num_cols + 0.001 < _cumsum)[0]))
                _col_to_type.append(_sub_terrain_keys[_sub_idx])
            # terrain_origins shape: (num_rows, num_cols, 3)，存储每个格子中心的世界坐标
            # 格子左下角 = 中心坐标 - cell_size/2
            _t_origins = _terrain.terrain_origins  # torch.Tensor
            _x_start = float(_t_origins[0, 0, 0].item()) - _cell_size / 2.0
            _y_start = float(_t_origins[0, 0, 1].item()) - _cell_size / 2.0
            terrain_debug_info = (_x_start, _y_start, _cell_size, _num_rows, _num_cols, _col_to_type)
            print(f"[地形] 列→类型映射: {dict(enumerate(_col_to_type))}")
            print(f"[地形] 格子起点: x={_x_start:.1f}, y={_y_start:.1f}, 单格={_cell_size}m ({_num_rows}行×{_num_cols}列)")

    # simulate environment
    play_step = 0
    slip_sum = 0.0
    slip_count = 0
    slip_over_count = 0
    slip_max = 0.0
    while simulation_app.is_running():
        start_time = time.time()
        # print(env.unwrapped.robot.data.applied_torque[0, env.unwrapped._spring_idx])
        # 键盘控制：ws加减速（非增量式），ad转向（非增量式），zx高度（增量式）
        cmd_info = None
        if args_cli.keyboard and controller is not None:
            # 执行一步并应用命令到环境
            dt = env.unwrapped.step_dt
            cmd_info = controller.advance_step(dt)
            
            # 输出键盘输入信息（只在有输入时输出）
            if cmd_info["has_input"]:
                height_info = f", height: {cmd_info['height']:.3f}" if hasattr(env.unwrapped, "height_cmd") else ""
                print(f"[键盘输入] v_x: {cmd_info['v_x']:.3f}, omega_z: {cmd_info['omega_z']:.3f}{height_info}")
                # Direct RMUC exposes the effective command through the same
                # method used by rewards and observations. Reading a legacy
                # ``command`` attribute here used to report stale values.
                if hasattr(env.unwrapped, "_current_command"):
                    actual_cmd = env.unwrapped._current_command()[0].detach().cpu().numpy()
                    print(
                        f"[调试] 环境实际命令: v_x={actual_cmd[0]:+.3f}, "
                        f"v_y={actual_cmd[1]:+.3f}, omega_z={actual_cmd[2]:+.3f}"
                    )
        
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # DreamWaq：打印 code_vel、仿真线速度/高度及误差
            if args_cli.dreamwaq_print_code_vel and runner_class is OnPolicyDreamWaqRunner:
                if (
                    args_cli.dreamwaq_code_vel_interval > 0
                    and play_step % args_cli.dreamwaq_code_vel_interval == 0
                ):
                    pol = runner.alg.policy
                    cv = getattr(pol, "last_code_vel", None)
                    if cv is not None:
                        code_vel = cv[0].detach().cpu()
                        code_vel[3] = code_vel[3] / 5.0  # 将 code_vel 的最后一个元素（高度）缩放回实际范围
                        code_vel[-2:] = code_vel[-2:] * 100.0 + 250.  # 将 code_vel 的最后两个元素（弹簧力）缩放回实际范围
                        if hasattr(env.unwrapped, "robot"):
                            sim_vx = float(env.unwrapped.robot.data.root_lin_vel_b[0, 0].item())
                            sim_vz = float(env.unwrapped.robot.data.root_lin_vel_b[0, 2].item())
                            sim_h = float(env.unwrapped.robot.data.root_pos_w[0, 2].item()-env.unwrapped.ground_z_est.item())
                            sim_s = env.unwrapped.spring_force[0].detach().cpu()
                            est_vx = float(code_vel[0].item())
                            est_vz = float(code_vel[2].item())
                            est_h = float(code_vel[3].item())
                            est_s = code_vel[-2:]
                            err_vx = est_vx - sim_vx
                            err_vz = est_vz - sim_vz
                            err_h = est_h - sim_h
                            err_s = est_s - sim_s
                            print(
                                f"[DreamWaq] step={play_step} "
                                f"code_vel={code_vel.numpy()} | "
                                f"sim=({sim_vx:.4f},{sim_vz:.4f},{sim_h:.4f},{sim_s}) | "
                                f"est=({est_vx:.4f},{est_vz:.4f},{est_h:.4f},{est_s}) | "
                                f"err=({err_vx:.4f},{err_vz:.4f},{err_h:.4f},{err_s})"
                            )
                        else:
                            print(f"[DreamWaq] step={play_step} code_vel={code_vel.numpy()} (shape={tuple(cv.shape)})")
            
            # 调试：检查策略网络输出的动作和实际角速度（如果有omega输入时）
            if args_cli.keyboard and controller is not None and cmd_info is not None:
                # 跳跃状态：只要窗口/flag 激活就打印
                jump_active = False
                jpm = None
                if hasattr(env.unwrapped, "jump_phase_manager") and hasattr(env.unwrapped, "manual_jump_steps"):
                    jpm = env.unwrapped.jump_phase_manager
                    if jpm is not None:
                        jump_active = bool(torch.any(env.unwrapped.manual_jump_steps > 0) or torch.any(jpm.jump_flag))

                if cmd_info.get("has_input", False) or jump_active:
                    # 检查策略网络接收到的命令值（从环境读取）
                    policy_obs = obs.get("policy") if hasattr(obs, "get") else obs
                    if torch.is_tensor(policy_obs) and policy_obs.ndim == 2 and policy_obs.shape[-1] >= 9:
                        command_obs = policy_obs[0, 6:9].detach().cpu().tolist()
                        print(
                            f"[调试] policy obs shape={tuple(policy_obs.shape)}, "
                            f"command slots[6:9]={['%.3f' % value for value in command_obs]}"
                        )
                    
                    action_np = actions[0].cpu().numpy()
                    # 检查轮子动作（通常是动作的后几个维度）
                    if hasattr(env.unwrapped, "_wheel_idx") or hasattr(env.unwrapped, "wheel_name"):
                        # RMUC action layout is
                        # [right_active, right_hip, right_wheel,
                        #  left_active, left_hip, left_wheel].
                        wheel_actions = action_np[[2, 5]] if len(action_np) >= 6 else action_np
                        print(f"[调试] 策略输出轮子动作: {wheel_actions}")
                    # 检查实际状态：角速度 / 线速度 / 高度
                    if hasattr(env.unwrapped, "robot"):
                        if hasattr(env.unwrapped, "_state"):
                            _, _, canonical_angular_velocity = env.unwrapped._state()
                            actual_omega = canonical_angular_velocity[0, 2].item()
                        else:
                            actual_omega = env.unwrapped.robot.data.root_ang_vel_b[0, 2].item()
                        print(f"[调试] 实际角速度: {actual_omega:.3f} rad/s, 目标: {cmd_info.get('omega_z', 0):.3f} rad/s")
                        # RMUC forward is the authored local -Z heading. Body
                        # frame X is not the task's forward metric and can
                        # have the opposite sign after the asset frame tilt.
                        if hasattr(env.unwrapped, "_heading_forward_velocity"):
                            actual_vx = env.unwrapped._heading_forward_velocity(
                                env.unwrapped.robot.data.root_lin_vel_w.torch
                            )[0].item()
                        else:
                            actual_vx = env.unwrapped.robot.data.root_lin_vel_b[0, 0].item()
                        print(f"[调试] 实际前向速度: {actual_vx:.3f} m/s, 目标: {cmd_info.get('v_x', 0):.3f} m/s")
                        if hasattr(env.unwrapped, "height_cmd"):
                            target_h = float(env.unwrapped.height_cmd[0].item())
                            actual_h = env.unwrapped.robot.data.root_pos_w[0, 2].item()
                            print(f"[调试] 实际高度: {actual_h:.3f} m, 目标: {target_h:.3f} m")
                    # 跳跃调试：窗口内最高高度 vs 目标跳跃高度
                    if jpm is not None and jump_active:
                        target_jump_h = float(jpm.jump_height[0].item())
                        max_jump_h = float(jpm.max_height[0].item())
                        remaining = int(env.unwrapped.manual_jump_steps[0].item())
                        print(f"[调试] 跳跃高度: 当前最高={max_jump_h:.3f} m, 目标={target_jump_h:.3f} m")
                        print(f"[调试] 跳跃窗口: 剩余步数={remaining}, 最高高度={max_jump_h:.3f} m, 目标={target_jump_h:.3f} m")
            # env stepping
            # RslRlVecEnvWrapper返回4个值: obs, rewards, dones, extras
            step_result = env.step(actions)
            if len(step_result) == 4:
                obs, rewards, dones, extras = step_result
                # 从extras中获取terminated和truncated信息
                terminated = dones.bool()  # dones是terminated和truncated的组合
                truncated = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool)) if isinstance(extras, dict) else torch.zeros_like(dones, dtype=torch.bool)
            else:
                # 兼容新版本gymnasium API（5个返回值）
                obs, rewards, terminated, truncated, extras = step_result

            play_step += 1

            if args_cli.slip_debug:
                slip_info = compute_wheel_slip_diagnostics(
                    env.unwrapped,
                    args_cli.slip_wheel_radius,
                    contact_threshold=1.0,
                )
                if slip_info is not None:
                    slip_speed = slip_info["slip_speed"]
                    contact_mask = slip_info["contact_mask"]
                    active_slip = slip_speed[contact_mask]
                    if active_slip.numel() > 0:
                        slip_sum += float(torch.sum(active_slip).item())
                        slip_count += int(active_slip.numel())
                        slip_over_count += int((active_slip > args_cli.slip_speed_threshold).sum().item())
                        slip_max = max(slip_max, float(torch.max(active_slip).item()))
                    interval = max(int(args_cli.slip_debug_interval), 1)
                    if play_step % interval == 0:
                        mean_slip = slip_sum / max(slip_count, 1)
                        over_ratio = slip_over_count / max(slip_count, 1)
                        env0_slip = slip_speed[0].detach().cpu().tolist()
                        env0_contact = contact_mask[0].detach().cpu().tolist()
                        env0_center = slip_info["wheel_center_speed"][0].detach().cpu().tolist()
                        surface = slip_info["wheel_surface_speed"]
                        env0_surface = surface[0].detach().cpu().tolist() if surface is not None else None
                        yaw_rate = float(env.unwrapped.robot.data.root_ang_vel_b[0, 2].item())
                        cmd = getattr(env.unwrapped, "command", None)
                        cmd_yaw = float(cmd[0, 2].item()) if torch.is_tensor(cmd) and cmd.shape[-1] >= 3 else float("nan")
                        print(
                            f"[SlipDebug] step={play_step} R={slip_info['radius']:.3f} "
                            f"cmd_yaw={cmd_yaw:+.3f} yaw={yaw_rate:+.3f} "
                            f"mean={mean_slip:.3f} max={slip_max:.3f} "
                            f"over{args_cli.slip_speed_threshold:.2f}={over_ratio:.2%} "
                            f"env0_slip={['%.3f' % v for v in env0_slip]} "
                            f"env0_contact={env0_contact} "
                            f"env0_center={['%.3f' % v for v in env0_center]} "
                            f"env0_wheel_rqdot={None if env0_surface is None else ['%.3f' % v for v in env0_surface]}"
                        )
            
            # # 打印 link 受力并标识惩罚部位 (用户需求: 分析碰撞惩罚来源)
            # if args_cli.keyboard and env_cfg.scene.num_envs == 1:
            #     if hasattr(env.unwrapped, 'contact_sensor'):
            #         contact_sensor = env.unwrapped.contact_sensor
            #         if hasattr(contact_sensor.data, 'net_forces_w_history') and contact_sensor.data.net_forces_w_history is not None:
            #             body_names = contact_sensor.body_names
            #             net_forces = contact_sensor.data.net_forces_w_history[0, 0]
                        
            #             # 获取环境配置中的索引
            #             undesired_idx = getattr(env.unwrapped, '_undesired_contact_link_idx', [])
            #             desired_idx = getattr(env.unwrapped, '_desired_contact_link_idx', [])
                        
            #             penalty_strings = []
            #             other_strings = []
            #             for i, name in enumerate(body_names):
            #                 force_vec = net_forces[i]
            #                 force_norm = torch.norm(force_vec).item()
                            
            #                 # 判定是否触发碰撞
            #                 is_contact = force_norm > 1.0
                            
            #                 if is_contact:
            #                     force_info = f"{name}: {force_norm:.1f}N"
            #                     # 检查是否是有害接触
            #                     if i in undesired_idx:
            #                         penalty_strings.append(f"[有害碰撞!惩罚] {force_info}")
            #                     # 检查是否是必要的轮子接触缺失 (这里逻辑是 net_force[desired_idx] < 1.0 会扣分)
            #                     # 我们这里只打印正在接触的部位，所以轮子接触是好事
            #                     elif i in desired_idx:
            #                         other_strings.append(f"[正常接触] {force_info}")
            #                     else:
            #                         other_strings.append(f"[其他接触] {force_info}")
                        
            #             # 检查是否有导致惩罚的轮子缺失接触 (rew_desired_contact 惩罚项)
            #             for idx in desired_idx:
            #                 if torch.norm(net_forces[idx]).item() < 1.0:
            #                     penalty_strings.append(f"[轮子悬空!惩罚] {body_names[idx]}")

            #             if penalty_strings:
            #                 print("\033[91m" + " | ".join(penalty_strings) + "\033[0m") # 红色打印惩罚项
            #             if other_strings:
            #                 print(" | ".join(other_strings))
            
            # 检测环境reset（episode结束）
            # RslRlVecEnvWrapper返回4个值: obs, rewards, dones, extras
            # dones是terminated和truncated的组合
            if len(step_result) == 4:
                has_reset = dones.any() if hasattr(dones, 'any') else bool(dones)
            else:
                # 兼容新版本gymnasium API（5个返回值）
                has_reset = (terminated.any() if hasattr(terminated, 'any') else bool(terminated)) or (truncated.any() if hasattr(truncated, 'any') else bool(truncated))
            
            if has_reset:
                # 重置所有键盘控制命令到默认值
                if args_cli.keyboard and controller is not None:
                    controller.reset()
                    # The vector wrapper has already reset the physics while
                    # returning ``obs``. Reset once more after applying the
                    # zero manual command so the next policy input cannot
                    # contain the previous staged command.
                    obs, extras = env.reset()
                    print("[INFO] 环境已重置，所有键盘命令已恢复到默认值")
        
        # 实时数据可视化
        if plotter is not None:
            try:
                # 获取目标高度和实际高度
                target_h = 0.0
                actual_h = 0.0
                jump_phase = 0.0
                
                # 获取策略实际看到的目标高度
                # 这个值在 env_jump.py 的 _get_observations 中被计算
                # 跳跃时会用 target_height_trajectory 替换 height_cmd
                if hasattr(env.unwrapped, 'height_cmd'):
                    # 默认使用 height_cmd
                    target_h = float(env.unwrapped.height_cmd[0].item())
                    
                    # 如果在跳跃窗口内，需要使用轨迹值（与观测逻辑一致）
                    if hasattr(env.unwrapped, 'jump_phase_manager'):
                        jpm = env.unwrapped.jump_phase_manager
                        if jpm is not None and hasattr(jpm, 'jump_flag') and jpm.jump_flag is not None:
                            # 跳跃时直接使用jump_height(阶跃值,不再使用曲线)
                            is_jumping = bool(jpm.jump_flag[0].item())
                            if is_jumping and hasattr(jpm, 'jump_height'):
                                # 使用目标跳跃高度(阶跃)
                                target_h = float(jpm.jump_height[0].item())
                
                if hasattr(env.unwrapped, 'relative_body_h_now'):
                    actual_h = float(env.unwrapped.relative_body_h_now[0].item())
                elif hasattr(env.unwrapped, 'robot'):
                    actual_h = float(env.unwrapped.robot.data.root_pos_w[0, 2].item())
                
                if hasattr(env.unwrapped, 'jump_phase_manager'):
                    jpm = env.unwrapped.jump_phase_manager
                    if jpm is not None:
                        jump_phase = float(jpm.jump_phase_sin[0].item())
                
                # 获取轮电机功率：P = tau * qdot，保留符号以区分驱动/制动
                power_left = 0.0
                power_right = 0.0
                if hasattr(env.unwrapped, 'robot') and hasattr(env.unwrapped.robot.data, 'applied_torque'):
                    if hasattr(env.unwrapped, '_wheel_idx') and hasattr(env.unwrapped.robot.data, 'joint_vel'):
                        wheel_idx = env.unwrapped._wheel_idx
                        torques = env.unwrapped.robot.data.applied_torque[0, wheel_idx]
                        wheel_vel = env.unwrapped.robot.data.joint_vel[0, wheel_idx]
                        wheel_power = torques * wheel_vel
                        if len(wheel_power) >= 2:
                            power_left = float(wheel_power[0].item())
                            power_right = float(wheel_power[1].item())
                
                # 获取腿部主动关节力矩 (Front1, Rear1)
                leg_torques = []
                if hasattr(env.unwrapped, 'robot') and hasattr(env.unwrapped.robot.data, 'applied_torque'):
                    indices = getattr(plotter, 'active_joint_indices', [])
                    if indices:
                        torques = env.unwrapped.robot.data.applied_torque[0, indices]
                        leg_torques = [float(t.item()) for t in torques]
                    elif hasattr(env.unwrapped, '_legs_act_idx'):
                        leg_idx = env.unwrapped._legs_act_idx
                        torques = env.unwrapped.robot.data.applied_torque[0, leg_idx]
                        leg_torques = [float(t.item()) for t in torques]
                
                # 获取弹簧力
                spring_left = 0.0
                spring_right = 0.0
                if hasattr(env.unwrapped, 'spring_force') and env.unwrapped.spring_force is not None:
                    s_force = env.unwrapped.spring_force[0]
                    if len(s_force) >= 2:
                        spring_left = float(s_force[0].item())
                        spring_right = float(s_force[1].item())
                    elif len(s_force) == 1:
                        spring_left = float(s_force[0].item())

                # 更新图表
                plotter.update(target_h, actual_h, jump_phase, 
                             power_left, power_right, leg_torques,
                             spring_left, spring_right)
            except Exception as e:
                print(f"[WARNING] Plotter update failed: {e}")
        
        # 相机跟随
        if args_cli.keyboard:
            camera_follow(env)
        
        timestep += 1

        # 实时通过世界坐标计算机器人（env 0）当前所在地形格子和类型，每 50 步打印一次
        if terrain_debug_info is not None and timestep % 50 == 0:
            _x0, _y0, _cs, _nr, _nc, _c2t = terrain_debug_info
            _pos = env.unwrapped.robot.data.root_pos_w[0]
            _px, _py = float(_pos[0].item()), float(_pos[1].item())
            _row = int((_px - _x0) / _cs)
            _col = int((_py - _y0) / _cs)
            _row = max(0, min(_row, _nr - 1))
            _col = max(0, min(_col, _nc - 1))
            _cur_type = _c2t[_col]
            print(f"[地形] step={timestep} pos=({_px:.1f},{_py:.1f}) 格[{_row},{_col}] 地形: {_cur_type}")

        if args_cli.video and timestep == args_cli.video_length:
            # Exit the play loop after recording one video
            break

        if args_cli.max_steps > 0 and play_step >= args_cli.max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.slip_debug:
        mean_slip = slip_sum / max(slip_count, 1)
        over_ratio = slip_over_count / max(slip_count, 1)
        print(
            f"[SlipDebugSummary] samples={slip_count} mean={mean_slip:.4f} "
            f"max={slip_max:.4f} over{args_cli.slip_speed_threshold:.2f}={over_ratio:.2%}"
        )

    # 关闭绘图器
    if plotter is not None:
        try:
            plotter.save_figure(os.path.join(log_dir, 'jump_visualization.png'))
            plotter.close()
        except Exception as e:
            print(f"[WARNING] Failed to save/close plotter: {e}")
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
