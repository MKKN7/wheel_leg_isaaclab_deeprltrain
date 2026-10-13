"""Evaluate a RMUC checkpoint against the staged-training gate.

Exit code 0 means the checkpoint passed the configured gate. Exit code 2
means it must be retrained at the same stage; no stage advancement is allowed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent / "rsl_rl"))
import cli_args  # noqa: E402
from rmuc_command_adapter import (  # noqa: E402
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
)

parser = argparse.ArgumentParser(description="Evaluate an RMUC staged-training checkpoint.")
parser.add_argument("--task", required=True)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument(
    "--speed-expert-checkpoint",
    type=str,
    default=None,
    help="Embed a frozen 2.9 m/s straight expert before loading the student checkpoint.",
)
parser.add_argument("--no-write", action="store_true", help="Do not overwrite curriculum_gate.json.")
parser.add_argument("--dr-strength", type=float, default=None,
                    help="Evaluate a fixed flat-domain strength in [0,1], including retained low-speed stages.")
parser.add_argument("--low-speed-yaw-feedback-clip", type=float, default=None)
parser.add_argument("--strict-survival", action="store_true",
                    help="Also require full-episode survival by yaw sign and minimum physical root height.")
parser.add_argument("--min-survival-episodes", type=int, default=64)
parser.add_argument("--command-consistent-dr-noise", action="store_true")
parser.add_argument("--feedback-contract-by-command", action="store_true")
parser.add_argument(
    "--save-teacher-dataset",
    type=str,
    default=None,
    help="Save deterministic policy observation/action samples from this fixed-stage evaluation.",
)
parser.add_argument(
    "--teacher-dataset-kind",
    choices=("quality", "trajectory"),
    default="quality",
    help=(
        "Mark saved samples as strict steady-state quality replay or as a full "
        "launch/recovery trajectory from an independently passing teacher."
    ),
)
parser.add_argument(
    "--teacher-dataset-max-speed-error-mps",
    type=float,
    default=None,
    help="Keep teacher samples only within this speed error; defaults to the stage gate.",
)
parser.add_argument(
    "--teacher-dataset-max-yaw-error-rad-s",
    type=float,
    default=None,
    help="Keep teacher samples only within this yaw error; defaults to the stage gate.",
)
parser.add_argument(
    "--augment-turn-mirror",
    action="store_true",
    default=False,
    help="Add a left/right mirrored copy of each accepted RMUC turn-teacher sample.",
)
parser.add_argument("--action-noise-std", type=float, default=0.0, help="Add Gaussian action noise for robustness diagnosis.")
parser.add_argument(
    "--command_level",
    type=int,
    default=None,
    help="For RMUC command evaluation, hold this staged command level instead of level 0.",
)
parser.add_argument("--yaw_feedback_gain", type=float, default=None)
parser.add_argument("--yaw_feedback_clip", type=float, default=None)
parser.add_argument("--speed_feedback_gain", type=float, default=None)
parser.add_argument("--speed_feedback_clip", type=float, default=None)
parser.add_argument("--pitch_feedback_gain", type=float, default=None)
parser.add_argument("--pitch_rate_feedback_gain", type=float, default=None)
parser.add_argument("--pitch_feedback_clip", type=float, default=None)
parser.add_argument(
    "--wheel_action_bias",
    type=float,
    default=0.0,
    help="Add a constant common-mode bias to both policy wheel actions for actuation diagnosis.",
)
parser.add_argument(
    "--wheel_torque_scale_nm",
    type=float,
    default=None,
    help="Override normalized wheel-action torque scale for actuation diagnosis.",
)
parser.add_argument(
    "--wheel_damping",
    type=float,
    default=None,
    help="Override the explicit wheel actuator damping for actuation diagnosis.",
)
parser.add_argument(
    "--wheel_damping_compensation_ratio",
    type=float,
    default=None,
    help="Override feed-forward compensation for explicit wheel actuator damping.",
)
parser.add_argument(
    "--wheel_damping_anneal_target",
    type=float,
    default=None,
    help="Override the simulation-only effective damping bridge target.",
)
parser.add_argument(
    "--wheel_damping_anneal_steps",
    type=int,
    default=None,
    help="Environment steps used by the damping bridge during fixed evaluation.",
)
parser.add_argument(
    "--command_vx_observation_scale",
    type=float,
    default=None,
    help="Override only the RMUC vx command/error observation scale for sensitivity diagnosis.",
)
parser.add_argument(
    "--command_observation_mode",
    choices=("tracking_error", "tracking_error_with_target", "target_with_forward_velocity", "target"),
    default=None,
    help="Override the command observation semantics when auditing a legacy checkpoint.",
)
parser.add_argument(
    "--policy_command_observation_mode",
    choices=("tracking_error", "tracking_error_with_target", "target_with_forward_velocity", "target"),
    default=None,
    help=(
        "Transform only the command slots passed to a legacy policy while retaining "
        "the current environment observation contract for a saved teacher dataset."
    ),
)
parser.add_argument(
    "--policy_command_observation_scale",
    type=str,
    default=None,
    help="Three comma-separated command scales used by --policy_command_observation_mode.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import agent_tasks  # noqa: F401,E402
import agent_world  # noqa: F401,E402


REPO_ROOT = Path(__file__).resolve().parents[1]


def _prepare_stock_rsl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg) -> dict:
    cfg = agent_cfg.to_dict()
    policy = cfg.get("policy") or {}
    if cfg.get("actor") == {} and cfg.get("critic") == {} and policy:
        cfg["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": list(policy.get("actor_hidden_dims", [256, 256, 256])),
            "activation": str(policy.get("activation", "elu")),
            "obs_normalization": False,
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
            "obs_normalization": False,
            "distribution_cfg": None,
        }
    return cfg


def _stage_name(task: str) -> str:
    if "Rough" in task:
        return "rough_robust"
    if "Command" in task:
        return "command"
    return "stand"


def _mirror_rmuc_actions(actions: torch.Tensor) -> torch.Tensor:
    """Mirror [right active, right hip, right wheel, left active, left hip, left wheel]."""

    mirrored = actions.clone()
    mirrored[:, 0] = actions[:, 3]
    mirrored[:, 1] = -actions[:, 4]
    mirrored[:, 2] = actions[:, 5]
    mirrored[:, 3] = actions[:, 0]
    mirrored[:, 4] = -actions[:, 1]
    mirrored[:, 5] = actions[:, 2]
    return mirrored


def _mirror_rmuc_observations(observations: torch.Tensor) -> torch.Tensor:
    """Mirror the fixed 25-D RMUC observation contract across the sagittal plane."""

    if observations.shape[-1] != 25:
        raise RuntimeError(f"RMUC mirror expects 25-D observations, got {observations.shape[-1]}")
    mirrored = observations.clone()
    mirrored[:, 0] = -observations[:, 0]
    mirrored[:, 2] = -observations[:, 2]
    mirrored[:, 4] = -observations[:, 4]
    # Slot 7 is absolute forward target speed and is mirror invariant.
    mirrored[:, 8] = -observations[:, 8]
    leg_pos = observations[:, 9:13]
    mirrored[:, 9] = leg_pos[:, 2]
    mirrored[:, 10] = -leg_pos[:, 3]
    mirrored[:, 11] = leg_pos[:, 0]
    mirrored[:, 12] = -leg_pos[:, 1]
    mirrored[:, 13:19] = _mirror_rmuc_actions(observations[:, 13:19])
    mirrored[:, 19:25] = _mirror_rmuc_actions(observations[:, 19:25])
    return mirrored


def _load_gate(stage: str) -> tuple[dict, int]:
    try:
        import yaml

        config = yaml.safe_load((REPO_ROOT / "configs" / "rmuc_wheel_leg.yaml").read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - Isaac runtime normally bundles PyYAML
        raise RuntimeError(f"Unable to load curriculum gate config: {exc}") from exc
    curriculum = dict(config.get("curriculum", {}))
    gate_root = dict(curriculum.get("gate", {}))
    return dict(gate_root.get(stage, {})), int(gate_root.get("evaluation_steps", 2048))


def main() -> int:
    stage = _stage_name(args_cli.task)
    gate, configured_steps = _load_gate(stage)
    eval_steps = max(int(args_cli.steps or configured_steps), 1)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env_cfg.play = True
    if args_cli.feedback_contract_by_command:
        env_cfg.feedback_contract_by_command = True
    if args_cli.command_consistent_dr_noise:
        env_cfg.domain_randomization_command_consistent_noise = True
    if args_cli.dr_strength is not None:
        if not 0.0 <= args_cli.dr_strength <= 1.0:
            raise ValueError("--dr-strength must be in [0,1]")
        env_cfg.domain_randomization_enabled = True
        env_cfg.domain_randomization_apply_in_play = True
        env_cfg.domain_randomization_strength_override = args_cli.dr_strength
    if args_cli.low_speed_yaw_feedback_clip is not None:
        if not 0.0 <= args_cli.low_speed_yaw_feedback_clip <= 1.0:
            raise ValueError("--low-speed-yaw-feedback-clip must be in [0,1]")
        env_cfg.wheel_yaw_feedback_low_speed_action_clip = args_cli.low_speed_yaw_feedback_clip
    if args_cli.command_level is not None:
        stage_names = getattr(env_cfg, "command_stage_names", ())
        if not stage_names or not 0 <= args_cli.command_level < len(stage_names):
            raise ValueError(
                f"--command_level must be in [0, {len(stage_names) - 1}], got {args_cli.command_level}"
            )
        env_cfg.command_start_level = int(args_cli.command_level)
        # The staged environment reads the per-level tuple, not the legacy
        # scalar ``command_transition_steps`` field.  Updating only the scalar
        # silently left a fixed-checkpoint evaluation ramping Level 15 from
        # zero to its target over one million steps, so its reported speed and
        # yaw metrics did not describe the requested command.  Hold the
        # selected level at its final target for evaluation.
        transition_steps = list(env_cfg.command_stage_transition_steps)
        transition_steps[args_cli.command_level] = 1
        env_cfg.command_stage_transition_steps = tuple(transition_steps)
        env_cfg.command_start_at_target = True
        env_cfg.command_stage_hold_steps = 1 << 60
        env_cfg.command_rehearsal_enabled = False
        if args_cli.command_level >= env_cfg.command_high_speed_gate_start_level:
            gate["max_mean_linear_velocity_error_mps"] = min(
                float(gate.get("max_mean_linear_velocity_error_mps", float("inf"))),
                float(
                    env_cfg.command_stage_advance_linear_error_mps[
                        args_cli.command_level
                    ]
                ),
            )
        print(
            f"[RMUC_GATE] Holding command level={args_cli.command_level} "
            f"name={stage_names[args_cli.command_level]}"
        )
    if args_cli.command_vx_observation_scale is not None:
        command_scale = list(env_cfg.command_observation_scale)
        command_scale[0] = float(args_cli.command_vx_observation_scale)
        env_cfg.command_observation_scale = tuple(command_scale)
        print(
            "[RMUC_GATE] Override command vx observation scale="
            f"{command_scale[0]}"
        )
    if args_cli.command_observation_mode is not None:
        env_cfg.command_observation_mode = args_cli.command_observation_mode
        print(
            "[RMUC_GATE] Override command observation mode="
            f"{env_cfg.command_observation_mode}"
        )
    if args_cli.wheel_torque_scale_nm is not None:
        if args_cli.wheel_torque_scale_nm <= 0.0:
            raise ValueError("--wheel_torque_scale_nm must be positive")
        env_cfg.wheel_torque_scale_nm = float(args_cli.wheel_torque_scale_nm)
        # A fixed endpoint evaluation must change both the normalized action
        # scale and the explicit actuator cap. Changing only the former leaves
        # an unrealizable 5.5 Nm simulation actuator behind a 3 Nm policy map.
        env_cfg.robot_cfg.actuators["wheel_motors"].effort_limit = float(
            args_cli.wheel_torque_scale_nm
        )
        print(
            "[RMUC_GATE] Override wheel torque scale/effort limit="
            f"{env_cfg.wheel_torque_scale_nm} Nm"
        )
    if args_cli.wheel_damping is not None:
        if args_cli.wheel_damping < 0.0:
            raise ValueError("--wheel_damping must be non-negative")
        env_cfg.robot_cfg.actuators["wheel_motors"].damping = float(
            args_cli.wheel_damping
        )
        print(f"[RMUC_GATE] Override wheel damping={args_cli.wheel_damping}")
    if args_cli.wheel_damping_compensation_ratio is not None:
        if not 0.0 <= args_cli.wheel_damping_compensation_ratio <= 1.0:
            raise ValueError("--wheel_damping_compensation_ratio must be in [0, 1]")
        env_cfg.wheel_damping_compensation_ratio = float(
            args_cli.wheel_damping_compensation_ratio
        )
        print(
            "[RMUC_GATE] Override wheel damping compensation ratio="
            f"{env_cfg.wheel_damping_compensation_ratio}"
        )
    if args_cli.wheel_damping_anneal_target is not None:
        if args_cli.wheel_damping_anneal_target < 0.0:
            raise ValueError("--wheel_damping_anneal_target must be non-negative")
        if (
            args_cli.wheel_damping_anneal_steps is not None
            and args_cli.wheel_damping_anneal_steps < 1
        ):
            raise ValueError("--wheel_damping_anneal_steps must be >= 1")
        env_cfg.wheel_damping_anneal_target = float(
            args_cli.wheel_damping_anneal_target
        )
        if args_cli.wheel_damping_anneal_steps is not None:
            env_cfg.wheel_damping_anneal_steps = int(
                args_cli.wheel_damping_anneal_steps
            )
        print(
            "[RMUC_GATE] Override wheel damping anneal target="
            f"{env_cfg.wheel_damping_anneal_target}, "
            f"steps={env_cfg.wheel_damping_anneal_steps}"
        )
    elif args_cli.wheel_damping_anneal_steps is not None:
        raise ValueError(
            "--wheel_damping_anneal_steps requires --wheel_damping_anneal_target"
        )
    feedback_overrides = {
        "wheel_yaw_feedback_action_per_rad_s": args_cli.yaw_feedback_gain,
        "wheel_yaw_feedback_action_clip": args_cli.yaw_feedback_clip,
        "wheel_speed_feedback_action_per_mps": args_cli.speed_feedback_gain,
        "wheel_speed_feedback_action_clip": args_cli.speed_feedback_clip,
        "wheel_pitch_feedback_action_per_rad": args_cli.pitch_feedback_gain,
        "wheel_pitch_rate_feedback_action_per_rad_s": args_cli.pitch_rate_feedback_gain,
        "wheel_pitch_feedback_action_clip": args_cli.pitch_feedback_clip,
    }
    for name, value in feedback_overrides.items():
        if value is not None:
            setattr(env_cfg, name, float(value))
            print(f"[RMUC_GATE] Override {name}={float(value)}")
    if args_cli.yaw_feedback_clip is not None and args_cli.low_speed_yaw_feedback_clip is None:
        env_cfg.wheel_yaw_feedback_low_speed_action_clip = float(args_cli.yaw_feedback_clip)
    if args_cli.speed_feedback_gain is not None:
        env_cfg.wheel_speed_feedback_high_speed_action_per_mps = float(
            args_cli.speed_feedback_gain
        )
        print(
            "[RMUC_GATE] Override wheel_speed_feedback_high_speed_action_per_mps="
            f"{float(args_cli.speed_feedback_gain)}"
        )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, _prepare_stock_rsl_cfg(agent_cfg), log_dir=None, device=agent_cfg.device)
    checkpoint = str(Path(args_cli.checkpoint).resolve())
    print(f"[RMUC_GATE] stage={stage} checkpoint={checkpoint} steps={eval_steps}")
    adapter_hidden_dim = checkpoint_turn_adapter_hidden_dim(checkpoint)
    if adapter_hidden_dim is not None:
        has_turn_delta_bank = checkpoint_has_turn_delta_bank(checkpoint)
        has_directional_turn_bank = checkpoint_has_directional_turn_delta_bank(
            checkpoint
        )
        install_rmuc_turn_adapter(
            runner.alg.get_policy(),
            hidden_dim=adapter_hidden_dim,
            freeze_base=False,
            enable_delta_bank=has_turn_delta_bank,
            enable_directional_bank=has_directional_turn_bank,
            bank_size=checkpoint_turn_delta_bank_size(checkpoint),
        )
        print(
            "[RMUC_GATE] Loaded RMUC turn residual adapter "
            f"hidden_dim={adapter_hidden_dim}, delta_bank={has_turn_delta_bank}, "
            f"directional_bank={has_directional_turn_bank}"
        )
    if checkpoint_has_actuator_bridge_adapter(checkpoint):
        install_rmuc_actuator_bridge_adapter(
            runner.alg.get_policy(),
            hidden_dim=adapter_hidden_dim or 64,
        )
        print("[RMUC_GATE] Loaded RMUC actuator bridge residual")
    if checkpoint_has_domain_residual_bank(checkpoint):
        install_rmuc_domain_residual_bank(runner.alg.get_policy(),
            speed_isolated=checkpoint_has_speed_isolated_domain_bank(checkpoint))
        print("[RMUC_GATE] Loaded RMUC bounded domain residual bank")
    embedded_speed_expert = checkpoint_has_speed_expert(checkpoint)
    if args_cli.speed_expert_checkpoint or embedded_speed_expert:
        install_rmuc_speed_expert(
            runner.alg.get_policy(),
            checkpoint_path=(
                args_cli.speed_expert_checkpoint
                if not embedded_speed_expert
                else None
            ),
        )
        print(
            "[RMUC_GATE] Loaded RMUC frozen 2.9 m/s straight expert: "
            f"{args_cli.speed_expert_checkpoint or 'checkpoint-embedded'}"
        )
    runner.load(
        checkpoint,
        load_cfg={
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        },
        strict=not bool(args_cli.speed_expert_checkpoint and not embedded_speed_expert),
    )
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs, _ = env.reset()
    # Observe terminal physics before DirectRLEnv automatically resets bodies.
    # Post-step root positions otherwise hide the last falling state.
    audit = {"min_root_height_m": float("inf"), "completed": [0, 0, 0],
             "survived": [0, 0, 0]}
    original_get_dones = env.unwrapped._get_dones

    def audited_get_dones():
        terminated, truncated = original_get_dones()
        robot_env = env.unwrapped
        audit["min_root_height_m"] = min(audit["min_root_height_m"],
            float(robot_env.robot.data.root_pos_w.torch[:, 2].min()))
        completed = terminated | truncated
        command_yaw = robot_env._current_command()[:, 2]
        horizon = robot_env._current_command_survival_gate_steps()
        survived = (~terminated) & (robot_env._curriculum_step_counter >= horizon)
        for index, mask in enumerate((command_yaw.abs() <= 1e-4,
                                      command_yaw > 1e-4, command_yaw < -1e-4)):
            audit["completed"][index] += int((completed & mask).sum())
            audit["survived"][index] += int((completed & mask & survived).sum())
        return terminated, truncated

    env.unwrapped._get_dones = audited_get_dones
    dataset_observations: list[torch.Tensor] = []
    dataset_actions: list[torch.Tensor] = []
    dataset_commands: list[torch.Tensor] = []
    policy_command_scale = None
    if args_cli.policy_command_observation_mode is not None:
        if not args_cli.policy_command_observation_scale:
            raise ValueError(
                "--policy_command_observation_mode requires "
                "--policy_command_observation_scale"
            )
        policy_command_scale = tuple(
            float(value.strip())
            for value in args_cli.policy_command_observation_scale.split(",")
            if value.strip()
        )
        if len(policy_command_scale) != 3 or any(value <= 0.0 for value in policy_command_scale):
            raise ValueError("--policy_command_observation_scale must contain three positive values")
        print(
            "[RMUC_GATE] Legacy policy command observation="
            f"{args_cli.policy_command_observation_mode} scale={policy_command_scale}; "
            f"saved dataset observation remains {env.unwrapped.cfg.command_observation_mode} "
            f"scale={tuple(env.unwrapped.cfg.command_observation_scale)}"
        )

    def encoded_command_slots(mode: str, scale_values) -> torch.Tensor:
        command = env.unwrapped._current_command()
        _, linear_velocity, angular_velocity = env.unwrapped._state()
        heading_velocity = env.unwrapped._heading_forward_velocity(
            env.unwrapped.robot.data.root_lin_vel_w.torch
        )
        if mode == "tracking_error":
            command_slots = torch.stack(
                (
                    command[:, 0] - heading_velocity,
                    command[:, 1] - linear_velocity[:, 1],
                    command[:, 2] - angular_velocity[:, 2],
                ),
                dim=-1,
            )
        elif mode == "tracking_error_with_target":
            command_slots = torch.stack(
                (
                    command[:, 0] - heading_velocity,
                    command[:, 0],
                    command[:, 2] - angular_velocity[:, 2],
                ),
                dim=-1,
            )
        elif mode == "target_with_forward_velocity":
            command_slots = torch.stack(
                (command[:, 0], heading_velocity, command[:, 2]), dim=-1
            )
        else:
            command_slots = command
        scale = torch.as_tensor(
            scale_values, device=command_slots.device, dtype=command_slots.dtype
        )
        return torch.clamp(
            command_slots * scale,
            -float(env.unwrapped.cfg.command_observation_error_clip),
            float(env.unwrapped.cfg.command_observation_error_clip),
        )

    def policy_observation(current_obs):
        if args_cli.policy_command_observation_mode is None:
            return current_obs
        policy_obs = current_obs["policy"].clone()
        policy_obs[:, 6:9] = encoded_command_slots(
            args_cli.policy_command_observation_mode, policy_command_scale
        )
        return {"policy": policy_obs}

    dataset_speed_error_limit = float(
        args_cli.teacher_dataset_max_speed_error_mps
        if args_cli.teacher_dataset_max_speed_error_mps is not None
        else (
            env.unwrapped.cfg.command_stage_advance_linear_error_mps[
                args_cli.command_level
            ]
            if args_cli.command_level is not None
            else env.unwrapped.cfg.command_advance_linear_error_mps
        )
    )
    dataset_yaw_error_limit = float(
        args_cli.teacher_dataset_max_yaw_error_rad_s
        if args_cli.teacher_dataset_max_yaw_error_rad_s is not None
        else env.unwrapped.cfg.command_advance_yaw_error_rad_s
    )
    if dataset_speed_error_limit <= 0.0 or dataset_yaw_error_limit <= 0.0:
        raise ValueError("Teacher dataset error limits must be positive")
    termination_counts: dict[str, int] = {}
    terminated_total = 0
    direction_warmup_steps = min(100, eval_steps // 4)
    direction = {
        "forward_correct": 0.0,
        "forward_reverse": 0.0,
        "forward_total": 0,
        "actual_vx_sum": 0.0,
        "root_heading_error_sum": 0.0,
        "axle_heading_error_sum": 0.0,
        "positive_yaw_correct": 0.0,
        "positive_yaw_total": 0,
        "positive_yaw_rate_sum": 0.0,
        "negative_yaw_correct": 0.0,
        "negative_yaw_total": 0,
        "negative_yaw_rate_sum": 0.0,
        "max_root_xy_distance_m": 0.0,
    }
    filter_time_constants_s = (0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.40)
    control_dt = float(env.unwrapped.step_dt)
    filtered_root_vx: dict[float, torch.Tensor] = {}
    filtered_root_error_sum = {tau: 0.0 for tau in filter_time_constants_s}
    temporal_root_vx_sum = torch.zeros(env.num_envs, device=env.unwrapped.device)
    temporal_target_vx_sum = torch.zeros_like(temporal_root_vx_sum)
    temporal_target_yaw_sum = torch.zeros_like(temporal_root_vx_sum)
    temporal_sample_count = torch.zeros_like(temporal_root_vx_sum)
    rolling_window_steps = max(int(env.unwrapped.cfg.command_gate_steps), 1)
    steady_state_warmup_steps = max(
        rolling_window_steps,
        int(env.unwrapped.cfg.command_survival_gate_steps),
    )
    rolling_root_vx = torch.zeros(
        (env.num_envs, rolling_window_steps), device=env.unwrapped.device
    )
    rolling_target_vx = torch.zeros_like(rolling_root_vx)
    rolling_yaw_error = torch.zeros_like(rolling_root_vx)
    rolling_count = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.unwrapped.device
    )
    rolling_index = 0
    rolling_linear_error_sum = torch.zeros_like(temporal_root_vx_sum)
    rolling_yaw_error_sum = torch.zeros_like(temporal_root_vx_sum)
    rolling_sample_count = torch.zeros_like(temporal_root_vx_sum)
    directional_steady_state = {
        "positive": {
            "actual_vx_sum": 0.0,
            "linear_error_sum": 0.0,
            "yaw_error_sum": 0.0,
            "sample_count": 0,
        },
        "negative": {
            "actual_vx_sum": 0.0,
            "linear_error_sum": 0.0,
            "yaw_error_sum": 0.0,
            "sample_count": 0,
        },
    }
    steady_state_window_count = 0
    signed_steady_state_window_count = 0
    steps_since_reset = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.unwrapped.device
    )

    def env_vector(value: torch.Tensor, name: str, *, dtype=None) -> torch.Tensor:
        """Normalize one per-environment value and reject silent broadcasting."""

        tensor = torch.as_tensor(value, device=env.unwrapped.device)
        if tensor.numel() != env.num_envs:
            raise RuntimeError(
                f"{name} must contain one value per environment; "
                f"expected {env.num_envs}, got shape={tuple(tensor.shape)}"
            )
        tensor = tensor.reshape(env.num_envs)
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    samples = {key: [] for key in (
        "tilt_deg",
        "upright",
        "wheel_contact_fraction",
        "loop_residual_m",
        "linear_velocity_error_mps",
        "yaw_velocity_error_rad_s",
        "root_height_m",
        "stand_height_error_m",
        "leg_extension_score",
        "leg_symmetry_error_rad",
        "attitude_score",
        "wheel_force_imbalance",
        "standing_fraction",
        "leg_length_m",
        "leg_length_score",
        "wheel_height_error_m",
        "wheel_height_score",
        "roll_error",
        "roll_score",
        "geometry_symmetry_score",
        "v_actual_over_v_cmd",
        "roll_deg",
        "roll_abs_deg",
        "pitch_deg",
        "turn_fraction",
        "turn_roll_abs_deg",
        "turn_roll_aligned_deg",
        "leg_length_diff_m",
        "turn_leg_length_diff_m",
        "command_rehearsal_priority_fraction",
        "wheel_torque_abs_mean_nm",
        "wheel_torque_ratio_mean",
        "wheel_torque_saturation_fraction",
        "wheel_slip_speed_mps",
        "wheel_slip_ratio",
    )}
    for step in range(eval_steps):
        dataset_step_metrics = None
        dataset_step_valid = None
        if args_cli.save_teacher_dataset:
            # Observation/action labels describe the pre-step state. Filtering
            # them with post-step metrics can cross an asynchronous reset and
            # falsely mark a low-speed launch state as a high-quality sample.
            # Clone the matching pre-step metrics; post-step ``dones`` below is
            # used only to reject transitions which terminate.
            dataset_step_metrics = {
                key: value.detach().clone()
                for key, value in env.unwrapped.get_curriculum_metrics().items()
            }
            dataset_step_valid = steps_since_reset > direction_warmup_steps
        with torch.inference_mode():
            actions = policy(policy_observation(obs))
            if args_cli.action_noise_std > 0.0:
                actions = actions + float(args_cli.action_noise_std) * torch.randn_like(actions)
            if args_cli.wheel_action_bias != 0.0:
                actions[:, [2, 5]] += float(args_cli.wheel_action_bias)
        if args_cli.save_teacher_dataset:
            dataset_step_observations = obs["policy"].detach().clone()
            # The wrapper observation can straddle an asynchronous reset. Make
            # the saved command slots explicitly match the pre-step sim state
            # used by both the quality filter and the legacy-teacher adapter.
            dataset_step_observations[:, 6:9] = encoded_command_slots(
                str(env.unwrapped.cfg.command_observation_mode),
                tuple(env.unwrapped.cfg.command_observation_scale),
            )
            dataset_step_actions = actions.detach().clamp(
                -float(agent_cfg.clip_actions), float(agent_cfg.clip_actions)
            ).clone()
            dataset_step_commands = env.unwrapped._current_command().detach().clone()
        if step == 0:
            print(
                "[RMUC_GATE_ACTION] first_action_mean={} first_action_abs_max={}".format(
                    actions.mean(dim=0).detach().cpu().tolist(),
                    float(actions.abs().max().item()),
                )
            )
        result = env.step(actions)
        if len(result) == 4:
            obs, _, dones, _ = result
            dones = env_vector(dones, "dones", dtype=torch.bool)
        else:
            obs, _, terminated, truncated, _ = result
            terminated = env_vector(terminated, "terminated", dtype=torch.bool)
            truncated = env_vector(truncated, "truncated", dtype=torch.bool)
            dones = terminated | truncated
        steps_since_reset += 1
        metrics = env.unwrapped.get_curriculum_metrics()
        direction["max_root_xy_distance_m"] = max(
            direction["max_root_xy_distance_m"],
            float(metrics["root_xy_distance_m"].max().item()),
        )
        root_heading_vx = env_vector(
            metrics["root_heading_vx_mps"], "root_heading_vx_mps"
        )
        command_target_vx = env_vector(
            metrics["command_target_vx_mps"], "command_target_vx_mps"
        )
        current_target_yaw = env_vector(
            metrics["command_target_yaw_rad_s"], "command_target_yaw_rad_s"
        )
        current_yaw_error = env_vector(
            metrics["yaw_velocity_error_rad_s"], "yaw_velocity_error_rad_s"
        )
        rolling_root_vx[:, rolling_index] = root_heading_vx
        rolling_target_vx[:, rolling_index] = command_target_vx
        rolling_yaw_error[:, rolling_index] = current_yaw_error
        rolling_index = (rolling_index + 1) % rolling_window_steps
        rolling_count.clamp_max_(rolling_window_steps - 1).add_(1)
        rolling_denominator = rolling_count.clamp_min(1).to(root_heading_vx.dtype)
        rolling_actual_vx = rolling_root_vx.sum(dim=-1) / rolling_denominator
        # Average the instantaneous absolute error inside each window. Using
        # abs(mean(v) - mean(cmd)) lets overspeed and underspeed cancel and can
        # make an oscillating policy look accurately tracked.
        rolling_linear_error = torch.abs(
            rolling_root_vx - rolling_target_vx
        ).sum(dim=-1) / rolling_denominator
        rolling_mean_yaw_error = rolling_yaw_error.sum(dim=-1) / rolling_denominator
        rolling_valid = (
            (rolling_count >= rolling_window_steps)
            & (steps_since_reset >= steady_state_warmup_steps)
            & ~dones
        )
        rolling_linear_error_sum[rolling_valid] += rolling_linear_error[rolling_valid]
        rolling_yaw_error_sum[rolling_valid] += rolling_mean_yaw_error[rolling_valid]
        rolling_sample_count[rolling_valid] += 1.0
        steady_state_window_count += int(rolling_valid.sum().item())
        signed_steady_state_window_count += int(
            (rolling_valid & (torch.abs(current_target_yaw) > 1.0e-4)).sum().item()
        )
        for label, sign_mask in (
            ("positive", current_target_yaw > 1.0e-4),
            ("negative", current_target_yaw < -1.0e-4),
        ):
            # Classify every valid steady-state window by its current command.
            # This keeps resets and later episodes from changing the sign of
            # an already accumulated sample through a lifetime average.
            directional_mask = rolling_valid & sign_mask
            directional_count = int(directional_mask.sum().item())
            if directional_count > 0:
                directional_steady_state[label]["actual_vx_sum"] += float(
                    rolling_actual_vx[directional_mask].sum().item()
                )
                directional_steady_state[label]["linear_error_sum"] += float(
                    rolling_linear_error[directional_mask].sum().item()
                )
                directional_steady_state[label]["yaw_error_sum"] += float(
                    rolling_mean_yaw_error[directional_mask].sum().item()
                )
                directional_steady_state[label]["sample_count"] += directional_count
        for tau in filter_time_constants_s:
            alpha = control_dt / (tau + control_dt)
            previous = filtered_root_vx.get(tau)
            if previous is None:
                filtered_root_vx[tau] = root_heading_vx.clone()
            else:
                filtered_root_vx[tau].lerp_(root_heading_vx, alpha)
        valid_after_reset = (steps_since_reset > direction_warmup_steps) & ~dones
        if args_cli.save_teacher_dataset:
            current_command_scale = tuple(
                float(value) for value in env.unwrapped.cfg.command_observation_scale
            )
            observation_speed_error = torch.abs(
                dataset_step_observations[:, 6] / current_command_scale[0]
            )
            observation_yaw_error = torch.abs(
                dataset_step_observations[:, 8] / current_command_scale[2]
            )
            quality_mask = (
                dataset_step_valid
                & ~dones
                & (
                    dataset_step_metrics["linear_velocity_error_mps"]
                    <= dataset_speed_error_limit
                )
                & (
                    dataset_step_metrics["yaw_velocity_error_rad_s"]
                    <= dataset_yaw_error_limit
                )
                & (
                    dataset_step_metrics["tilt_deg"]
                    <= float(env.unwrapped.cfg.curriculum_max_tilt_deg)
                )
                & (
                    dataset_step_metrics["root_height_m"]
                    >= float(env.unwrapped.cfg.command_min_operating_root_height_m)
                )
                & (
                    dataset_step_metrics["wheel_contact_fraction"]
                    >= float(env.unwrapped.cfg.command_min_wheel_contact_fraction)
                )
                & (observation_speed_error <= dataset_speed_error_limit)
                & (observation_yaw_error <= dataset_yaw_error_limit)
            )
            if torch.any(quality_mask):
                accepted_observations = dataset_step_observations[quality_mask]
                accepted_actions = dataset_step_actions[quality_mask]
                accepted_commands = dataset_step_commands[quality_mask]
                dataset_observations.append(accepted_observations.cpu())
                dataset_actions.append(accepted_actions.cpu())
                dataset_commands.append(accepted_commands.cpu())
                if args_cli.augment_turn_mirror:
                    dataset_observations.append(
                        _mirror_rmuc_observations(accepted_observations).cpu()
                    )
                    dataset_actions.append(_mirror_rmuc_actions(accepted_actions).cpu())
                    mirrored_commands = accepted_commands.clone()
                    mirrored_commands[:, 1:] *= -1.0
                    dataset_commands.append(mirrored_commands.cpu())
        if torch.any(valid_after_reset):
            target_vx = command_target_vx
            actual_vx = env_vector(metrics["actual_vx_mps"], "actual_vx_mps")
            moving_forward = (torch.abs(target_vx) > 1.0e-4) & valid_after_reset
            aligned_vx = actual_vx * torch.sign(target_vx)
            direction["forward_correct"] += float(((aligned_vx > 0.02) & moving_forward).sum().item())
            direction["forward_reverse"] += float(((aligned_vx < -0.02) & moving_forward).sum().item())
            direction["forward_total"] += int(moving_forward.sum().item())
            direction["actual_vx_sum"] += float(actual_vx[moving_forward].sum().item())
            temporal_root_vx_sum[moving_forward] += metrics["root_heading_vx_mps"][moving_forward]
            temporal_target_vx_sum[moving_forward] += target_vx[moving_forward]
            temporal_target_yaw_sum[moving_forward] += metrics[
                "command_target_yaw_rad_s"
            ][moving_forward]
            temporal_sample_count[moving_forward] += 1.0
            direction["root_heading_error_sum"] += float(
                torch.abs(metrics["root_heading_vx_mps"] - target_vx)[moving_forward].sum().item()
            )
            direction["axle_heading_error_sum"] += float(
                torch.abs(metrics["axle_heading_vx_mps"] - target_vx)[moving_forward].sum().item()
            )
            for tau in filter_time_constants_s:
                filtered_root_error_sum[tau] += float(
                    torch.abs(filtered_root_vx[tau] - target_vx)[moving_forward].sum().item()
                )

            target_yaw = current_target_yaw
            actual_yaw = env_vector(
                metrics["actual_yaw_rate_rad_s"], "actual_yaw_rate_rad_s"
            )
            positive_yaw = (target_yaw > 1.0e-4) & valid_after_reset
            negative_yaw = (target_yaw < -1.0e-4) & valid_after_reset
            direction["positive_yaw_correct"] += float(((actual_yaw > 0.01) & positive_yaw).sum().item())
            direction["positive_yaw_total"] += int(positive_yaw.sum().item())
            direction["positive_yaw_rate_sum"] += float(actual_yaw[positive_yaw].sum().item())
            direction["negative_yaw_correct"] += float(((actual_yaw < -0.01) & negative_yaw).sum().item())
            direction["negative_yaw_total"] += int(negative_yaw.sum().item())
            direction["negative_yaw_rate_sum"] += float(actual_yaw[negative_yaw].sum().item())
        if bool(dones.any()):
            terminated_total += int(dones.sum().item())
            for key, mask in env.unwrapped.get_termination_reasons().items():
                termination_counts[key] = termination_counts.get(key, 0) + int(mask.sum().item())
        sample_valid = ~dones
        if torch.any(sample_valid):
            for key in samples:
                sample_values = metrics[key].detach().float()
                if sample_values.ndim == 0:
                    # A small number of diagnostics are already reduced by
                    # the environment and therefore cannot be masked again.
                    samples[key].append(sample_values.item())
                else:
                    samples[key].append(sample_values[sample_valid].mean().item())
        if bool(dones.any()):
            for tau in filter_time_constants_s:
                filtered_root_vx[tau][dones] = 0.0
            rolling_root_vx[dones] = 0.0
            rolling_target_vx[dones] = 0.0
            rolling_yaw_error[dones] = 0.0
            rolling_count[dones] = 0
            # DirectRLEnv already resets only the completed environments during
            # step(). A global reset here used to restart every healthy peer
            # whenever one environment ended, flooding long evaluations with
            # acceleration transients and making high-speed policies appear
            # artificially slow.
            steps_since_reset[dones] = 0
    if not samples["tilt_deg"]:
        raise RuntimeError("Evaluation produced no non-terminal metric samples")
    aggregate = {
        "mean_tilt_deg": sum(samples["tilt_deg"]) / len(samples["tilt_deg"]),
        "upright_fraction": sum(samples["upright"]) / len(samples["upright"]),
        "wheel_contact_fraction": sum(samples["wheel_contact_fraction"]) / len(samples["wheel_contact_fraction"]),
        "max_loop_residual_m": max(samples["loop_residual_m"]),
        "mean_linear_velocity_error_mps": sum(samples["linear_velocity_error_mps"]) / len(samples["linear_velocity_error_mps"]),
        "mean_yaw_velocity_error_rad_s": sum(samples["yaw_velocity_error_rad_s"]) / len(samples["yaw_velocity_error_rad_s"]),
        "mean_root_height_m": sum(samples["root_height_m"]) / len(samples["root_height_m"]),
        "max_mean_height_error_m": sum(samples["stand_height_error_m"]) / len(samples["stand_height_error_m"]),
        "mean_leg_extension_score": sum(samples["leg_extension_score"]) / len(samples["leg_extension_score"]),
        "max_mean_leg_symmetry_error_rad": sum(samples["leg_symmetry_error_rad"]) / len(samples["leg_symmetry_error_rad"]),
        "mean_attitude_score": sum(samples["attitude_score"]) / len(samples["attitude_score"]),
        "mean_wheel_force_imbalance": sum(samples["wheel_force_imbalance"]) / len(samples["wheel_force_imbalance"]),
        "standing_fraction": sum(samples["standing_fraction"]) / len(samples["standing_fraction"]),
        "mean_leg_length_m": sum(samples["leg_length_m"]) / len(samples["leg_length_m"]),
        "mean_leg_length_score": sum(samples["leg_length_score"]) / len(samples["leg_length_score"]),
        "mean_wheel_height_error_m": sum(samples["wheel_height_error_m"]) / len(samples["wheel_height_error_m"]),
        "mean_wheel_height_score": sum(samples["wheel_height_score"]) / len(samples["wheel_height_score"]),
        "mean_roll_error": sum(samples["roll_error"]) / len(samples["roll_error"]),
        "mean_roll_score": sum(samples["roll_score"]) / len(samples["roll_score"]),
        "mean_geometry_symmetry_score": sum(samples["geometry_symmetry_score"]) / len(samples["geometry_symmetry_score"]),
        "mean_v_actual_over_v_cmd": sum(samples["v_actual_over_v_cmd"]) / len(samples["v_actual_over_v_cmd"]),
        "mean_roll_deg": sum(samples["roll_deg"]) / len(samples["roll_deg"]),
        "mean_roll_abs_deg": sum(samples["roll_abs_deg"]) / len(samples["roll_abs_deg"]),
        "mean_pitch_deg": sum(samples["pitch_deg"]) / len(samples["pitch_deg"]),
        "turn_fraction": sum(samples["turn_fraction"]) / len(samples["turn_fraction"]),
        "mean_turn_roll_abs_deg": sum(samples["turn_roll_abs_deg"]) / len(samples["turn_roll_abs_deg"]),
        "mean_turn_roll_aligned_deg": sum(samples["turn_roll_aligned_deg"]) / len(samples["turn_roll_aligned_deg"]),
        "mean_leg_length_diff_m": sum(samples["leg_length_diff_m"]) / len(samples["leg_length_diff_m"]),
        "mean_turn_leg_length_diff_m": sum(samples["turn_leg_length_diff_m"]) / len(samples["turn_leg_length_diff_m"]),
        "mean_command_rehearsal_priority_fraction": sum(samples["command_rehearsal_priority_fraction"]) / len(samples["command_rehearsal_priority_fraction"]),
        "mean_wheel_torque_abs_nm": sum(samples["wheel_torque_abs_mean_nm"]) / len(samples["wheel_torque_abs_mean_nm"]),
        "mean_wheel_torque_ratio": sum(samples["wheel_torque_ratio_mean"]) / len(samples["wheel_torque_ratio_mean"]),
        "wheel_torque_saturation_fraction": sum(samples["wheel_torque_saturation_fraction"]) / len(samples["wheel_torque_saturation_fraction"]),
        "mean_wheel_slip_speed_mps": sum(samples["wheel_slip_speed_mps"]) / len(samples["wheel_slip_speed_mps"]),
        "mean_wheel_slip_ratio": sum(samples["wheel_slip_ratio"]) / len(samples["wheel_slip_ratio"]),
    }
    forward_total = max(direction["forward_total"], 1)
    positive_yaw_total = max(direction["positive_yaw_total"], 1)
    negative_yaw_total = max(direction["negative_yaw_total"], 1)
    aggregate.update({
        "direction_warmup_steps": direction_warmup_steps,
        "mean_actual_vx_mps": direction["actual_vx_sum"] / forward_total,
        "mean_root_heading_velocity_error_mps": direction["root_heading_error_sum"] / forward_total,
        "mean_axle_heading_velocity_error_mps": direction["axle_heading_error_sum"] / forward_total,
        "forward_direction_fraction": direction["forward_correct"] / forward_total,
        "reverse_motion_fraction": direction["forward_reverse"] / forward_total,
        "mean_positive_command_yaw_rate_rad_s": (
            direction["positive_yaw_rate_sum"] / positive_yaw_total
        ),
        "positive_yaw_direction_fraction": direction["positive_yaw_correct"] / positive_yaw_total,
        "mean_negative_command_yaw_rate_rad_s": (
            direction["negative_yaw_rate_sum"] / negative_yaw_total
        ),
        "negative_yaw_direction_fraction": direction["negative_yaw_correct"] / negative_yaw_total,
        "max_root_xy_distance_m": direction["max_root_xy_distance_m"],
        "forward_command_sample_count": direction["forward_total"],
        "positive_yaw_command_sample_count": direction["positive_yaw_total"],
        "negative_yaw_command_sample_count": direction["negative_yaw_total"],
    })
    aggregate["filtered_root_heading_velocity_error_mps"] = {
        f"tau_{tau:.2f}": filtered_root_error_sum[tau] / forward_total
        for tau in filter_time_constants_s
    }
    temporal_valid = temporal_sample_count > 0
    temporal_mean_actual_vx = temporal_root_vx_sum[temporal_valid] / temporal_sample_count[
        temporal_valid
    ]
    temporal_mean_target_vx = temporal_target_vx_sum[temporal_valid] / temporal_sample_count[
        temporal_valid
    ]
    temporal_mean_target_yaw = temporal_target_yaw_sum[temporal_valid] / temporal_sample_count[
        temporal_valid
    ]
    cancelled_temporal_error = torch.abs(
        temporal_mean_actual_vx - temporal_mean_target_vx
    )
    aggregate["legacy_cancelled_temporal_velocity_error_mps"] = float(
        cancelled_temporal_error.mean().item()
    )
    rolling_valid_env = rolling_sample_count > 0
    accumulated_window_count = int(rolling_sample_count.sum().item())
    if accumulated_window_count != steady_state_window_count:
        raise RuntimeError(
            "Steady-state rolling sample bookkeeping mismatch: "
            f"per-env={accumulated_window_count}, observed={steady_state_window_count}"
        )
    directional_window_count = sum(
        int(stats["sample_count"]) for stats in directional_steady_state.values()
    )
    if directional_window_count != signed_steady_state_window_count:
        raise RuntimeError(
            "Directional rolling sample bookkeeping mismatch: "
            f"directions={directional_window_count}, signed={signed_steady_state_window_count}"
        )
    if torch.any(rolling_valid_env):
        temporal_mean_error = (
            rolling_linear_error_sum[rolling_valid_env]
            / rolling_sample_count[rolling_valid_env]
        )
        temporal_mean_yaw_error = (
            rolling_yaw_error_sum[rolling_valid_env]
            / rolling_sample_count[rolling_valid_env]
        )
        aggregate["mean_per_env_temporal_velocity_error_mps"] = float(
            temporal_mean_error.mean().item()
        )
        aggregate["max_per_env_temporal_velocity_error_mps"] = float(
            temporal_mean_error.max().item()
        )
        aggregate["mean_per_env_temporal_yaw_error_rad_s"] = float(
            temporal_mean_yaw_error.mean().item()
        )
        aggregate["steady_state_rolling_sample_count"] = int(
            steady_state_window_count
        )
    else:
        aggregate["mean_per_env_temporal_velocity_error_mps"] = float("inf")
        aggregate["max_per_env_temporal_velocity_error_mps"] = float("inf")
        aggregate["mean_per_env_temporal_yaw_error_rad_s"] = float("inf")
        aggregate["steady_state_rolling_sample_count"] = 0
    aggregate["signed_steady_state_rolling_sample_count"] = int(
        signed_steady_state_window_count
    )
    episode_exposure = max(
        float(env.num_envs * eval_steps)
        / max(float(env.unwrapped.max_episode_length), 1.0),
        1.0,
    )
    aggregate["tilt_termination_rate"] = (
        float(termination_counts.get("tilt", 0)) / episode_exposure
    )
    for label, stats in directional_steady_state.items():
        sample_count = int(stats["sample_count"])
        aggregate[f"temporal_{label}_yaw_sample_count"] = sample_count
        if sample_count > 0:
            actual_vx = float(stats["actual_vx_sum"]) / sample_count
            linear_error = float(stats["linear_error_sum"]) / sample_count
            yaw_error = float(stats["yaw_error_sum"]) / sample_count
        else:
            actual_vx = 0.0
            linear_error = float("inf")
            yaw_error = float("inf")
        # Preserve the existing vx keys and add the corresponding yaw error
        # and explicit rolling-window sample count.
        aggregate[f"temporal_{label}_yaw_actual_vx_mps"] = actual_vx
        aggregate[f"temporal_{label}_yaw_vx_error_mps"] = linear_error
        aggregate[f"temporal_{label}_yaw_error_rad_s"] = yaw_error
    aggregate["worst_direction_temporal_velocity_error_mps"] = max(
        aggregate["temporal_positive_yaw_vx_error_mps"],
        aggregate["temporal_negative_yaw_vx_error_mps"],
    )
    aggregate["worst_direction_temporal_yaw_error_rad_s"] = max(
        aggregate["temporal_positive_yaw_error_rad_s"],
        aggregate["temporal_negative_yaw_error_rad_s"],
    )
    checks = {
        "upright_fraction": aggregate["upright_fraction"] >= float(gate.get("min_upright_fraction", 0.95)),
        "mean_tilt_deg": aggregate["mean_tilt_deg"] <= float(gate.get("max_mean_tilt_deg", 12.0)),
        "wheel_contact_fraction": aggregate["wheel_contact_fraction"] >= float(gate.get("min_wheel_contact_fraction", 0.6)),
        "max_loop_residual_m": aggregate["max_loop_residual_m"] <= float(gate.get("max_loop_residual_m", 0.01)),
        "mean_linear_velocity_error_mps": aggregate["mean_linear_velocity_error_mps"] <= float(gate.get("max_mean_linear_velocity_error_mps", 0.5)),
        "mean_yaw_velocity_error_rad_s": aggregate["mean_yaw_velocity_error_rad_s"] <= float(gate.get("max_mean_yaw_velocity_error_rad_s", 0.5)),
        "min_mean_root_height_m": aggregate["mean_root_height_m"] >= float(gate.get("min_mean_root_height_m", env.unwrapped.cfg.command_min_operating_root_height_m)),
    }
    # Height/extension are stand objectives; do not reject command/rough
    # locomotion merely because the body moves away from the reset height.
    if stage == "stand":
        checks.update({
            "max_mean_height_error_m": aggregate["max_mean_height_error_m"] <= float(gate.get("max_mean_height_error_m", 0.12)),
            "min_mean_leg_extension_score": aggregate["mean_leg_extension_score"] >= float(gate.get("min_mean_leg_extension_score", 0.7)),
            "max_mean_leg_symmetry_error_rad": aggregate["max_mean_leg_symmetry_error_rad"] <= float(gate.get("max_mean_leg_symmetry_error_rad", 0.15)),
            "min_mean_attitude_score": aggregate["mean_attitude_score"] >= float(gate.get("min_mean_attitude_score", 0.95)),
            "min_standing_fraction": aggregate["standing_fraction"] >= float(gate.get("min_standing_fraction", 0.95)),
            "max_wheel_force_imbalance": aggregate["mean_wheel_force_imbalance"] <= float(gate.get("max_wheel_force_imbalance", 0.35)),
            "min_mean_leg_length_score": aggregate["mean_leg_length_score"] >= float(gate.get("min_mean_leg_length_score", 0.70)),
            "min_mean_wheel_height_score": aggregate["mean_wheel_height_score"] >= float(gate.get("min_mean_wheel_height_score", 0.70)),
            "min_mean_roll_score": aggregate["mean_roll_score"] >= float(gate.get("min_mean_roll_score", 0.70)),
        })
    elif stage == "command":
        checks.pop("mean_linear_velocity_error_mps")
        checks.pop("mean_yaw_velocity_error_rad_s")
        checks.update({
            "mean_per_env_temporal_velocity_error_mps": (
                aggregate["mean_per_env_temporal_velocity_error_mps"]
                <= float(gate.get("max_mean_linear_velocity_error_mps", 0.5))
            ),
            "mean_per_env_temporal_yaw_error_rad_s": (
                aggregate["mean_per_env_temporal_yaw_error_rad_s"]
                <= float(gate.get("max_mean_yaw_velocity_error_rad_s", 0.5))
            ),
            "forward_direction_fraction": aggregate["forward_direction_fraction"]
            >= float(gate.get("min_forward_direction_fraction", 0.90)),
            "reverse_motion_fraction": aggregate["reverse_motion_fraction"]
            <= float(gate.get("max_reverse_motion_fraction", 0.02)),
            "tilt_termination_rate": aggregate["tilt_termination_rate"]
            <= float(gate.get("max_tilt_termination_rate", 0.10)),
        })
        if direction["positive_yaw_total"] > 0:
            checks["positive_yaw_direction_fraction"] = (
                aggregate["positive_yaw_direction_fraction"]
                >= float(gate.get("min_yaw_direction_fraction", 0.90))
            )
        if direction["negative_yaw_total"] > 0:
            checks["negative_yaw_direction_fraction"] = (
                aggregate["negative_yaw_direction_fraction"]
                >= float(gate.get("min_yaw_direction_fraction", 0.90))
            )
        has_turn_commands = (
            direction["positive_yaw_total"] > 0
            or direction["negative_yaw_total"] > 0
        )
        if has_turn_commands:
            linear_error_limit = float(
                gate.get("max_mean_linear_velocity_error_mps", 0.5)
            )
            yaw_error_limit = float(
                gate.get("max_mean_yaw_velocity_error_rad_s", 0.5)
            )
            for label in ("positive", "negative"):
                checks[f"{label}_yaw_steady_state_samples"] = (
                    aggregate[f"temporal_{label}_yaw_sample_count"] > 0
                )
                checks[f"{label}_yaw_temporal_velocity_error_mps"] = (
                    aggregate[f"temporal_{label}_yaw_vx_error_mps"]
                    <= linear_error_limit
                )
                checks[f"{label}_yaw_temporal_yaw_error_rad_s"] = (
                    aggregate[f"temporal_{label}_yaw_error_rad_s"]
                    <= yaw_error_limit
                )
    aggregate["physical_min_root_height_m"] = audit["min_root_height_m"]
    for index, label in enumerate(("straight", "positive", "negative")):
        count = audit["completed"][index]
        aggregate[f"{label}_completed_episodes"] = count
        aggregate[f"{label}_full_episode_survival"] = audit["survived"][index] / max(count, 1)
    if args_cli.strict_survival:
        selected_yaw = env_cfg.command_stage_yaw_rate_rad_s[env.unwrapped._command_level]
        buckets = ("positive", "negative") if abs(selected_yaw) > 1e-4 else ("straight",)
        for label in buckets:
            checks[f"{label}_completed_episodes"] = (
                aggregate[f"{label}_completed_episodes"] >= args_cli.min_survival_episodes)
            checks[f"{label}_full_episode_survival"] = (
                aggregate[f"{label}_full_episode_survival"] >= env_cfg.command_min_survival_env_fraction)
        checks["physical_min_root_height_m"] = (
            audit["min_root_height_m"] > env_cfg.command_min_operating_root_height_m)
    passed = all(checks.values())
    result = {
        "gate_schema_version": 2,
        "strict_survival": args_cli.strict_survival,
        "dr_strength_override": args_cli.dr_strength,
        "stage": stage,
        "checkpoint": checkpoint,
        "steps": eval_steps,
        "metrics": aggregate,
        "termination_total": terminated_total,
        "termination_counts": termination_counts,
        "checks": checks,
        "passed": passed,
    }
    if args_cli.save_teacher_dataset:
        dataset_path = Path(args_cli.save_teacher_dataset).resolve()
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if not dataset_observations:
            raise RuntimeError(
                "No teacher samples met the speed/yaw/posture/contact quality filters"
            )
        dataset_obs = torch.cat(dataset_observations, dim=0)
        dataset_act = torch.cat(dataset_actions, dim=0)
        dataset_cmd = torch.cat(dataset_commands, dim=0)
        if (
            dataset_obs.shape[-1] != 25
            or dataset_act.shape[-1] != 6
            or dataset_cmd.shape != (dataset_obs.shape[0], 3)
        ):
            raise RuntimeError(
                "RMUC teacher dataset contract mismatch: "
                f"observations={tuple(dataset_obs.shape)}, actions={tuple(dataset_act.shape)}, "
                f"commands={tuple(dataset_cmd.shape)}"
            )
        if str(env.unwrapped.cfg.command_observation_mode) in (
            "tracking_error",
            "tracking_error_with_target",
        ):
            dataset_scale = tuple(
                float(value) for value in env.unwrapped.cfg.command_observation_scale
            )
            saved_speed_error = torch.abs(dataset_obs[:, 6] / dataset_scale[0])
            saved_yaw_error = torch.abs(dataset_obs[:, 8] / dataset_scale[2])
            post_valid = (
                (saved_speed_error <= dataset_speed_error_limit + 1.0e-5)
                & (saved_yaw_error <= dataset_yaw_error_limit + 1.0e-5)
            )
            if not torch.all(post_valid):
                original_count = int(dataset_obs.shape[0])
                dataset_obs = dataset_obs[post_valid]
                dataset_act = dataset_act[post_valid]
                dataset_cmd = dataset_cmd[post_valid]
                print(
                    "[RMUC_TEACHER_DATASET] post-filter removed "
                    f"{original_count - int(dataset_obs.shape[0])}/{original_count} "
                    "samples whose saved command-error slots missed the declared thresholds",
                    flush=True,
                )
        if dataset_obs.shape[0] == 0:
            raise RuntimeError("Teacher dataset post-filter removed every sample")
        positive_turn_samples = int((dataset_cmd[:, 2] > 1.0e-4).sum().item())
        negative_turn_samples = int((dataset_cmd[:, 2] < -1.0e-4).sum().item())
        torch.save(
            {
                "dataset_schema_version": 2,
                "observations": dataset_obs,
                "actions": dataset_act,
                "commands": dataset_cmd,
                "checkpoint": checkpoint,
                "command_level": args_cli.command_level,
                "stage": stage,
                "steps": eval_steps,
                "num_envs": int(args_cli.num_envs),
                "gate_passed": bool(passed),
                "quality_filtered": True,
                "replay_kind": str(args_cli.teacher_dataset_kind),
                "max_speed_error_mps": dataset_speed_error_limit,
                "max_yaw_error_rad_s": dataset_yaw_error_limit,
                "evaluation_metrics": aggregate,
                "evaluation_checks": checks,
                "observation_mode": str(env.unwrapped.cfg.command_observation_mode),
                "observation_scale": tuple(
                    float(value) for value in env.unwrapped.cfg.command_observation_scale
                ),
                "policy_source_observation_mode": (
                    args_cli.policy_command_observation_mode
                    or str(env.unwrapped.cfg.command_observation_mode)
                ),
                "policy_source_observation_scale": (
                    policy_command_scale
                    or tuple(
                        float(value)
                        for value in env.unwrapped.cfg.command_observation_scale
                    )
                ),
                "turn_mirror_augmented": bool(args_cli.augment_turn_mirror),
                "positive_turn_sample_count": positive_turn_samples,
                "negative_turn_sample_count": negative_turn_samples,
                "turn_direction_covered": (
                    positive_turn_samples > 0 and negative_turn_samples > 0
                ),
            },
            dataset_path,
        )
        print(
            f"[RMUC_TEACHER_DATASET] saved={dataset_path} samples={dataset_obs.shape[0]} "
            f"obs_dim={dataset_obs.shape[-1]} action_dim={dataset_act.shape[-1]} "
            f"positive_turn={positive_turn_samples} negative_turn={negative_turn_samples} "
            f"gate_passed={passed}"
        )
    print("[RMUC_GATE] " + json.dumps(result, ensure_ascii=False, sort_keys=True))
    output_path = Path(checkpoint).with_name("curriculum_gate.json")
    if not args_cli.no_write:
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    env.close()
    return 0 if passed else 2


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
