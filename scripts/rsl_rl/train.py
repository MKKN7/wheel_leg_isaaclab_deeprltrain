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

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
from collections import deque
import inspect
import json
import math
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument('--train_position_authority', action='store_true')
parser.add_argument('--router_speed_direction', choices=['positive', 'negative'], default=None,
                    help='Train only isolated speed-extension copies; requires fresh authority optimizer.')
parser.add_argument('--train_physics_adaptation', action='store_true',
                    help='With train_position_authority, optimize only the explicit physical head.')
parser.add_argument('--physics_train_modules', nargs='+', default=None,
                    help='Explicit existing physics network paths; requires physics adaptation and reset optimizer.')
parser.add_argument('--require_command_coverage', action='store_true',
                    help='Reject missing rehearsal gate commands before simulation and in the first three rollouts.')
parser.add_argument('--allow_observation_contract_migration', action='store_true',
                    help='Explicitly allow a checkpoint yaw observation clip change; requires fresh optimizer.')
parser.add_argument('--train_local_authority', action='store_true',
                    help='With train_position_authority, optimize only local experts and freeze the source head.')
parser.add_argument('--local_authority_train_indices', type=int, nargs='+', default=None,
                    help='Train only selected local expert indices; omitted means all local experts.')
parser.add_argument('--rmuc_tgs_external_forces_every_iteration', action='store_true',
                    help='Explicit RMUC physics migration for position/velocity consistency.')
parser.add_argument(
    "--num_steps_per_env_override",
    type=int,
    default=None,
    help=(
        "Override PPO rollout steps per environment. Longer rollouts improve temporal credit "
        "for RMUC high-speed oscillation without changing the policy/checkpoint contract."
    ),
)
parser.add_argument(
    "--command_level",
    type=int,
    default=None,
    help="Start a staged RMUC command task at this level instead of the YAML default.",
)
parser.add_argument(
    "--start_command_at_target",
    action="store_true",
    default=False,
    help="Start the requested RMUC command level at its final target instead of repeating the previous-level transition.",
)
parser.add_argument(
    "--stop_at_command_level",
    type=int,
    default=None,
    help="Save the entry checkpoint and stop when the RMUC curriculum reaches this level.",
)
parser.add_argument(
    "--teacher_speed_checkpoint",
    type=str,
    default=None,
    help="Checkpoint whose deterministic actor supervises straight/high-speed rollout samples.",
)
parser.add_argument(
    "--teacher_turn_checkpoint",
    type=str,
    default=None,
    help="Checkpoint whose deterministic actor supervises non-zero-yaw rollout samples.",
)
parser.add_argument('--recovery_anchor_checkpoint',type=str,default=None,
    help='Fixed recovery actor reference for <=15 degree rehearsal cohorts only.')
parser.add_argument('--recovery_anchor_coefficient',type=float,default=0.)
parser.add_argument('--recovery_replay_manifest',type=str,default=None)
parser.add_argument('--recovery_replay_coefficient',type=float,default=0.)
parser.add_argument('--recovery_curriculum',action='store_true',
    help='Enable progressive recovery reset angles for the Recovery task.')
parser.add_argument('--recovery_curriculum_start_tilt',type=int,choices=[0,5,15,30,45,60,75,90],default=None)
parser.add_argument('--recovery_curriculum_max_tilt',type=int,choices=[0,5,15,30,45,60,75,90],default=None)
parser.add_argument('--recovery_curriculum_success_episodes',type=int,default=None)
parser.add_argument('--recovery_slew_start_rate',type=float,default=None)
parser.add_argument('--recovery_slew_end_rate',type=float,default=None)
parser.add_argument('--recovery_slew_anneal_steps',type=int,default=None,
    help='Recovery target-slew anneal duration in aggregate control transitions.')
parser.add_argument('--recovery_slope_degrees',type=float,default=None)
parser.add_argument('--recovery_slope_azimuth_degrees',type=float,default=None)
parser.add_argument(
    "--teacher_distill_weight",
    type=float,
    default=0.20,
    help="Initial teacher-distillation strength used to schedule the auxiliary actor step.",
)
parser.add_argument(
    "--teacher_distill_final_weight",
    type=float,
    default=None,
    help="Final teacher-distillation strength; defaults to the initial fixed weight.",
)
parser.add_argument(
    "--teacher_distill_anneal_iterations",
    type=int,
    default=0,
    help="Cosine-anneal teacher strength over this many trainable actor updates.",
)
parser.add_argument(
    "--teacher_distill_learning_rate",
    type=float,
    default=None,
    help="Optional learning rate for the teacher-distillation actor step; defaults to PPO LR.",
)
parser.add_argument(
    "--teacher_distill_updates",
    type=int,
    default=3,
    help="Number of balanced speed/turn teacher updates before each PPO update.",
)
parser.add_argument(
    "--teacher_distill_only_iterations",
    type=int,
    default=0,
    help=(
        "For the first N trainable updates, update the critic with PPO but update "
        "the actor only from teacher replay. PPO actor gradients resume afterward."
    ),
)
parser.add_argument(
    "--teacher_speed_anchor_weight",
    type=float,
    default=1.0,
    help=(
        "Relative weight of online and replay speed-teacher losses versus turn-skill losses. "
        "Use a value above one when preserving a validated high-speed actor."
    ),
)
parser.add_argument(
    "--teacher_turn_common_mode_weight",
    type=float,
    default=0.0,
    help=(
        "Wheel common-mode retention weight for fixed turn replay datasets. Zero is fully "
        "decoupled; a small value can preserve direction-specific forward speed when replay "
        "observations carry an explicit target-speed slot. Online turn supervision remains "
        "differential-only."
    ),
)
parser.add_argument(
    "--teacher_turn_leg_weight",
    type=float,
    default=0.05,
    help=(
        "Leg-action retention weight for fixed turn replay. Keep it small so "
        "the student can still adapt posture at a higher speed."
    ),
)
parser.add_argument(
    "--teacher_speed_dataset",
    type=str,
    default=None,
    help="Saved fixed-stage speed-teacher observation/action replay dataset.",
)
parser.add_argument(
    "--teacher_speed_replay_only",
    action="store_true",
    default=False,
    help=(
        "Use the quality-filtered speed dataset without online supervision from "
        "the full teacher actor. Use when the teacher has good local samples but fails long-horizon validation."
    ),
)
parser.add_argument(
    "--teacher_turn_replay_only",
    action="store_true",
    default=False,
    help=(
        "Use fixed mirrored turn datasets without evaluating the legacy turn teacher "
        "on current observations. Required when command observation semantics changed."
    ),
)
parser.add_argument(
    "--allow_online_highspeed_teacher",
    action="store_true",
    default=False,
    help=(
        "Explicitly allow online supervision from the full speed teacher at high-speed levels. "
        "By default, high-speed training with a speed replay dataset is replay-only because "
        "the legacy teacher can contain long-horizon acceleration failures."
    ),
)
parser.add_argument(
    "--teacher_turn_dataset",
    type=str,
    default=None,
    help="Comma-separated fixed-stage turn-teacher replay datasets.",
)
parser.add_argument(
    "--allow_unanchored_highspeed_distillation",
    action="store_true",
    default=False,
    help=(
        "Allow high-speed teacher distillation without a fixed speed replay dataset. "
        "Use only for an explicit ablation; anchored speed replay is required by default."
    ),
)
parser.add_argument(
    "--teacher_replay_batch_size",
    type=int,
    default=2048,
    help="Samples drawn from each teacher dataset per distillation update.",
)
parser.add_argument(
    "--teacher_gradient_projection",
    action="store_true",
    default=False,
    help=(
        "Project conflicting speed/turn teacher gradients before the auxiliary actor step. "
        "Aligned teacher gradients are left unchanged."
    ),
)
parser.add_argument(
    "--turn_residual_adapter",
    action="store_true",
    default=False,
    help=(
        "Freeze the checkpoint MLP and train a command-yaw-gated residual adapter. "
        "Straight-command outputs remain exactly checkpoint-compatible."
    ),
)
parser.add_argument(
    "--turn_residual_adapter_hidden_dim",
    type=int,
    default=64,
    help="Hidden width of the RMUC turn residual adapter.",
)
parser.add_argument(
    "--turn_residual_bank",
    action="store_true",
    default=False,
    help=(
        "Use speed/yaw-conditioned cumulative residual heads. Only the active "
        "curriculum head is trainable; certified lower-speed heads stay frozen."
    ),
)
parser.add_argument(
    "--turn_directional_residual_bank",
    action="store_true",
    default=False,
    help=(
        "Split every residual-bank head into positive- and negative-yaw "
        "corrections so opposing turn gradients cannot cancel."
    ),
)
parser.add_argument(
    "--actuator_bridge_adapter",
    action="store_true",
    default=False,
    help=(
        "Train a zero-initialized all-command residual while torque, effort "
        "limit and damping are annealed to the hardware contract."
    ),
)
parser.add_argument(
    "--freeze_turn_residual_bank_during_bridge",
    action="store_true",
    default=False,
    help=(
        "Keep every speed/yaw residual head frozen while the all-command "
        "actuator bridge residual adapts to the hardware contract."
    ),
)
parser.add_argument(
    "--freeze_actuator_bridge_adapter",
    action="store_true",
    default=False,
    help=(
        "Freeze a checkpoint-embedded all-command adapter after its retained "
        "skills have been certified."
    ),
)
parser.add_argument(
    "--turn_expert_checkpoint",
    type=str,
    default=None,
    help=(
        "Certified low/medium-speed turn checkpoint embedded as a frozen, "
        "yaw-gated expert alongside the frozen straight-speed base actor."
    ),
)
parser.add_argument(
    "--speed_expert_checkpoint",
    type=str,
    default=None,
    help=(
        "Embed a frozen MLP expert around the certified 2.9 m/s zero-yaw "
        "command. The expert is saved inside subsequent checkpoints."
    ),
)
parser.add_argument(
    "--highspeed_reset_bootstrap",
    action="store_true",
    default=False,
    help="Use partial high-speed rolling resets for RMUC training; assisted episodes are excluded from the gate.",
)
parser.add_argument(
    "--highspeed_regression_guard",
    action="store_true",
    default=False,
    help="Save the best high-speed checkpoint and stop after sustained within-level speed regression.",
)
parser.add_argument(
    "--highspeed_regression_guard_warn_only",
    action="store_true",
    default=False,
    help=(
        "Keep saving/reporting sustained high-speed regression without stopping the trainer. "
        "Use with --highspeed_regression_guard when curriculum advancement is the only desired stop."
    ),
)
parser.add_argument(
    "--wheel_noise_hold_steps",
    type=int,
    default=1,
    help="Hold a shared wheel common-mode Gaussian sample for this many control steps (training only).",
)
parser.add_argument(
    "--resume_training",
    action="store_true",
    default=False,
    help="Resume optimizer state and iteration counter from --checkpoint, then continue from the next iteration.",
)
parser.add_argument(
    "--preserve_checkpoint_std",
    action="store_true",
    default=False,
    help="For actor-only finetuning, retain the checkpoint's learned action standard deviations.",
)
parser.add_argument(
    "--load_critic_for_finetune",
    action="store_true",
    default=False,
    help="Load and zero-pad compatible critic tensors during actor finetuning.",
)
parser.add_argument(
    "--zero_observation_indices_for_finetune",
    type=str,
    default=None,
    help="Comma-separated zero-based input columns to clear in actor/critic first layers after loading.",
)
parser.add_argument(
    "--reset_optimizer",
    action="store_true",
    default=False,
    help="With --resume_training, restore actor/critic/iteration but initialize a fresh optimizer.",
)
parser.add_argument(
    "--freeze_action_std",
    action="store_true",
    default=False,
    help="Keep the checkpoint action standard deviations fixed while training the policy mean.",
)
parser.add_argument(
    "--freeze_leg_action_std",
    action="store_true",
    default=False,
    help="Freeze only the four RMUC leg action std entries while keeping both wheel std entries trainable.",
)
parser.add_argument(
    "--min_wheel_action_std",
    type=float,
    default=None,
    help=(
        "Optional lower bound for both trainable RMUC wheel-action standard deviations. "
        "Applied after every policy update so high-speed common-mode exploration cannot collapse."
    ),
)
parser.add_argument(
    "--allow_fresh_critic_highspeed",
    action="store_true",
    default=False,
    help="Explicitly allow actor-only high-speed finetuning with a freshly initialized critic.",
)
parser.add_argument(
    "--critic_warmup_iterations",
    type=int,
    default=0,
    help="Train only the critic for this many initial iterations before releasing actor updates.",
)
parser.add_argument(
    "--freeze_actor_until_command_level",
    type=int,
    default=None,
    help=(
        "Keep the loaded actor and action std frozen until the RMUC command curriculum "
        "reaches this level. Use this to certify an already validated stage before "
        "starting the next harder stage."
    ),
)
parser.add_argument(
    "--action_std_override",
    type=str,
    default=None,
    help="Comma-separated per-action Gaussian std values applied after loading a checkpoint.",
)
parser.add_argument(
    "--wheel_damping_compensation_ratio",
    type=float,
    default=None,
    help=(
        "Override simulation-only wheel damping compensation in [0, 1]. "
        "The default YAML value remains checkpoint-compatible (0)."
    ),
)
parser.add_argument(
    "--wheel_damping_override",
    type=float,
    default=None,
    help=(
        "Override the explicit RMUC wheel actuator damping for a staged actuator-model "
        "migration. Existing checkpoints were trained at 0.1; do not jump directly to "
        "the XML value 0.01 without an intermediate bridge."
    ),
)
parser.add_argument(
    "--wheel_damping_anneal_target",
    type=float,
    default=None,
    help=(
        "Simulation-only effective wheel damping target. The configured actuator "
        "damping is preserved and feed-forward compensation is cosine-annealed."
    ),
)
parser.add_argument(
    "--wheel_damping_anneal_steps",
    type=int,
    default=None,
    help="Environment steps over which effective wheel damping reaches its target.",
)
parser.add_argument(
    "--wheel_torque_scale_override_nm",
    type=float,
    default=None,
    help="Override the RMUC normalized wheel-action torque scale during a staged actuator-model migration.",
)
parser.add_argument(
    "--disable_progressive_flat_dr",
    action="store_true",
    default=False,
    help=(
        "Disable the proficiency-gated flat-terrain domain-randomization curriculum. "
        "Use only for nominal ablations; play/evaluation disable it automatically."
    ),
)
parser.add_argument(
    "--rehearsal_feedback_parity", action="store_true",
    help="Use each retained stage's active speed/pitch feedback contract during rehearsal.",
)
parser.add_argument(
    "--low_speed_yaw_feedback_clip", type=float, default=None,
    help="Override only low-speed turn feedback clipping for teacher-contract ablations.",
)
parser.add_argument("--command_consistent_dr_noise", action="store_true",
                    help="Randomize sensor channels while preserving command/expert-routing consistency.")
parser.add_argument("--rehearsal_straight_feedback_only", action="store_true",
                    help="Restrict feedback parity to retained high-speed straight commands.")
parser.add_argument('--isolated_state_router', action='store_true', help='Train only explicit checkpoint state router and critic.')
parser.add_argument('--state_router_train_branches', type=lambda value: value.split(','), default=None,
                    help='Comma-separated exact router branch names; other branches remain frozen even on eligible commands. Changing this set requires matching optimizer state or reset_optimizer.')
parser.add_argument('--state_router_learning_rate', type=float, default=1e-4)
parser.add_argument('--state_router_balance_coef', type=float, default=0.01)
parser.add_argument("--domain_residual_learning_rate", type=float, default=None,
                    help="Separate LR for the bounded domain bank; requires a fixed PPO LR schedule.")
parser.add_argument("--stage_residual_learning_rate", type=float, default=None,
                    help="Separate fixed LR for the current stage residual; certified heads stay frozen.")
parser.add_argument("--protect_nominal_straight_anchor", action="store_true",
                    help="Protect restored 2.9 straight local head and Adam state until scheduled DR becomes nonzero.")
parser.add_argument("--residual_anchor_l2", type=float, default=None,
                    help="Enable per-update frozen-weight/action audit and fixed-checkpoint action L2; 0 audits only.")
parser.add_argument("--residual_anchor_checkpoint", type=str, default=None,
                    help="Keep the same action-regularization anchor across training restarts.")
parser.add_argument("--critic_learning_rate", type=float, default=None,
                    help="Let the critic adapt without increasing the frozen-skill actor update rate.")
parser.add_argument("--speed_isolated_domain_bank", action="store_true",
                    help="Freeze shared domain heads and add bounded per-speed/yaw adaptation heads.")
parser.add_argument("--feedback_contract_by_command", action="store_true",
                    help="Use one command-based outer-loop control contract in active and retained environments.")
parser.add_argument(
    "--learning_rate_override",
    type=float,
    default=None,
    help="Override PPO learning rate for this stage without changing shared YAML defaults.",
)
parser.add_argument(
    "--num_learning_epochs_override",
    type=int,
    default=None,
    help="Override PPO update epochs for this stage without changing shared YAML defaults.",
)
parser.add_argument(
    "--gamma_override",
    type=float,
    default=None,
    help="Override PPO gamma for long-horizon actuator adaptation.",
)
parser.add_argument(
    "--lam_override",
    type=float,
    default=None,
    help="Override PPO GAE lambda for long-horizon actuator adaptation.",
)
parser.add_argument(
    "--allow_aggressive_highspeed",
    action="store_true",
    default=False,
    help="Acknowledge high-speed settings above the validated anti-regression envelope.",
)
parser.add_argument(
    "--allow_curriculum_override",
    action="store_true",
    default=False,
    help="Explicitly allow advancing despite a failed previous-stage gate.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.reset_optimizer and not args_cli.resume_training:
    parser.error("--reset_optimizer requires --resume_training")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

from rsl_rl.runners import *  # noqa: F401
from agent_rl.rsl_rl.runners import *  # noqa: F401
import agent_world  # noqa: F401
import agent_tasks  # noqa: F401
from rmuc_command_adapter import (
    RMUC_TURN_DELTA_SPECS,
    checkpoint_has_actuator_bridge_adapter,
    checkpoint_has_domain_residual_bank,
    checkpoint_has_speed_isolated_domain_bank,
    checkpoint_has_directional_turn_delta_bank,
    checkpoint_has_speed_expert,
    checkpoint_has_turn_delta_bank,
    checkpoint_turn_adapter_hidden_dim,
    checkpoint_turn_delta_bank_size,
    disable_rmuc_turn_adapter,
    install_rmuc_actuator_bridge_adapter,
    install_rmuc_domain_residual_bank,
    install_rmuc_speed_expert,
    install_rmuc_turn_adapter,
    set_rmuc_turn_delta_trainable,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# 清理任务名称，去除可能的前导 '=' 符号
if args_cli.task and args_cli.task.startswith('='):
    args_cli.task = args_cli.task[1:]
    print(f"[INFO] 已清理任务名称，去除前导 '=' 符号: {args_cli.task}")


def _load_policy_for_finetune(
    runner,
    checkpoint_path: str,
    *,
    preserve_checkpoint_std: bool = False,
    load_critic: bool = False,
    zero_observation_indices: tuple[int, ...] = (),
) -> None:
    """Load actor weights for finetuning, with an optional critic warm start."""
    loaded_dict = torch.load(checkpoint_path, weights_only=False, map_location=runner.device)
    # RSL-RL checkpoints use actor_state_dict; older local exports used
    # model_state_dict.  Both contain the policy module tensors needed for
    # cross-stage finetuning.
    source_state = loaded_dict.get("model_state_dict") or loaded_dict.get("actor_state_dict")
    if source_state is None:
        raise KeyError("checkpoint has neither model_state_dict nor actor_state_dict")
    actor = getattr(runner.alg, "policy", None) or getattr(runner.alg, "actor", None)
    if actor is None:
        raise AttributeError("RSL-RL algorithm exposes neither policy nor actor")

    def load_module(label: str, target_module, source_module_state: dict, *, skip_std: bool) -> None:
        target_state = target_module.state_dict()
        compatible_state = {}
        expanded_state = {}
        skipped = []
        for key, value in source_module_state.items():
            if skip_std and "distribution.std" in key:
                skipped.append((key, tuple(value.shape), tuple(target_state[key].shape) if key in target_state else None))
                continue
            target_value = target_state.get(key)
            if target_value is not None and tuple(target_value.shape) == tuple(value.shape):
                compatible_state[key] = value
            elif (
                target_value is not None
                and value.ndim == 2
                and target_value.ndim == 2
                and target_value.shape[0] == value.shape[0]
                and target_value.shape[1] > value.shape[1]
            ):
                expanded_value = torch.zeros_like(target_value)
                expanded_value[:, : value.shape[1]] = value
                expanded_state[key] = expanded_value
            else:
                skipped.append((key, tuple(value.shape), tuple(target_value.shape) if target_value is not None else None))
        target_state.update(compatible_state)
        target_state.update(expanded_state)
        input_weight_keys = [key for key in target_state if key.endswith("mlp.0.weight")]
        if zero_observation_indices and len(input_weight_keys) != 1:
            raise RuntimeError(
                f"Expected one {label} input weight for observation-column reset, got {input_weight_keys}"
            )
        for key in input_weight_keys:
            input_width = target_state[key].shape[1]
            invalid = [index for index in zero_observation_indices if not 0 <= index < input_width]
            if invalid:
                raise ValueError(
                    f"Observation indices {invalid} are outside {label} input width {input_width}"
                )
            target_state[key][:, list(zero_observation_indices)] = 0.0
        target_module.load_state_dict(target_state, strict=True)
        print(
            f"[INFO]: Finetune {label} load: loaded {len(compatible_state)} tensors, "
            f"zero-padded {len(expanded_state)} input tensors, skipped {len(skipped)} tensors."
        )
        for key, value in expanded_state.items():
            source_shape = tuple(source_module_state[key].shape)
            target_shape = tuple(value.shape)
            print(f"[INFO]:   zero-padded {label}.{key}: checkpoint{source_shape} -> model{target_shape}")
        for key, source_shape, target_shape in skipped[:20]:
            print(f"[INFO]:   skipped {label}.{key}: checkpoint{source_shape} -> model{target_shape}")
        if zero_observation_indices:
            print(
                f"[INFO]:   zeroed {label} observation columns: {list(zero_observation_indices)}"
            )

    load_module(
        "actor",
        actor,
        source_state,
        skip_std=not preserve_checkpoint_std,
    )
    if load_critic:
        source_critic_state = loaded_dict.get("critic_state_dict")
        critic = getattr(runner.alg, "critic", None)
        if source_critic_state is None or critic is None:
            raise KeyError("--load_critic_for_finetune requires checkpoint and runner critic state")
        load_module("critic", critic, source_critic_state, skip_std=False)
    else:
        print(
            "[INFO]: Finetune critic initialized fresh; actor skills restored from checkpoint. "
            "Learning iteration counter reset to 0.",
            flush=True,
        )
    runner.current_learning_iteration = 0


def _load_frozen_teacher(actor_template, checkpoint_path: str, label: str):
    """Create a frozen deterministic teacher with the exact 25-D/6-action contract."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    loaded_dict = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    source_state = loaded_dict.get("actor_state_dict") or loaded_dict.get("model_state_dict")
    if source_state is None:
        raise KeyError(f"{label} checkpoint has neither actor_state_dict nor model_state_dict")
    teacher = copy.deepcopy(actor_template).cpu()
    disable_rmuc_turn_adapter(teacher)
    try:
        incompatible = teacher.load_state_dict(source_state, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{label} teacher checkpoint is incompatible with the current actor contract: {exc}"
        ) from exc
    allowed_missing = {
        key
        for key in teacher.state_dict()
        if key.startswith(
            (
                "rmuc_turn_adapter.",
                "rmuc_turn_expert.",
                "rmuc_turn_delta_adapters.",
                "rmuc_actuator_bridge_adapter.",
                "rmuc_speed_expert.",
                "rmuc_domain_residual_bank.",
            )
        )
    }
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected or missing:
        raise RuntimeError(
            f"{label} teacher checkpoint is incompatible: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    input_width = int(teacher.mlp[0].weight.shape[1])
    output_width = int(teacher.mlp[-1].weight.shape[0])
    if input_width != 25 or output_width != 6:
        raise RuntimeError(
            f"{label} teacher must be 25-D observation/6-D action, got {input_width}/{output_width}"
        )
    print(
        f"[INFO] Loaded frozen {label} teacher: {checkpoint_path} (obs={input_width}, actions={output_width})",
        flush=True,
    )
    return teacher


def _mirror_rmuc_actions(actions: torch.Tensor) -> torch.Tensor:
    """Mirror the six RMUC policy actions across the sagittal plane."""

    mirrored = actions.clone()
    mirrored[:, 0] = actions[:, 3]
    mirrored[:, 1] = -actions[:, 4]
    mirrored[:, 2] = actions[:, 5]
    mirrored[:, 3] = actions[:, 0]
    mirrored[:, 4] = -actions[:, 1]
    mirrored[:, 5] = actions[:, 2]
    return mirrored


def _mirror_rmuc_observations(observations: torch.Tensor) -> torch.Tensor:
    """Mirror the fixed 25-D RMUC observation contract."""

    if observations.shape[-1] != 25:
        raise RuntimeError(f"RMUC mirror expects 25-D observations, got {observations.shape[-1]}")
    mirrored = observations.clone()
    mirrored[:, 0] = -observations[:, 0]
    mirrored[:, 2] = -observations[:, 2]
    mirrored[:, 4] = -observations[:, 4]
    # Slot 7 is absolute forward target speed in the current 25-D contract and
    # therefore remains unchanged under a left/right mirror.
    mirrored[:, 8] = -observations[:, 8]
    leg_pos = observations[:, 9:13]
    mirrored[:, 9] = leg_pos[:, 2]
    mirrored[:, 10] = -leg_pos[:, 3]
    mirrored[:, 11] = leg_pos[:, 0]
    mirrored[:, 12] = -leg_pos[:, 1]
    mirrored[:, 13:19] = _mirror_rmuc_actions(observations[:, 13:19])
    mirrored[:, 19:25] = _mirror_rmuc_actions(observations[:, 19:25])
    return mirrored


def _rmuc_turn_skill_loss(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    *,
    common_mode_weight: float = 0.0,
    leg_weight: float = 0.0,
) -> torch.Tensor:
    """Preserve differential steering without imposing teacher speed/posture.

    Low/medium-speed turn teachers have a smaller common wheel command than a
    high-speed student. Matching both wheel actions independently therefore
    distills the old forward speed along with yaw and conflicts with the speed
    teacher. Their absolute leg targets also conflict with high-speed balance.
    The wheel difference is the directly transferable yaw control.  Fixed
    replay may add small common-mode and leg terms so retained turns do not
    lose their forward-driving posture, while online supervision stays
    differential-only.
    """

    if student_actions.shape[-1] != 6 or teacher_actions.shape != student_actions.shape:
        raise RuntimeError(
            "RMUC turn distillation expects matching [N, 6] actions, got "
            f"student={tuple(student_actions.shape)} teacher={tuple(teacher_actions.shape)}"
        )
    student_wheel_difference = 0.5 * (
        student_actions[:, 2] - student_actions[:, 5]
    )
    teacher_wheel_difference = 0.5 * (
        teacher_actions[:, 2] - teacher_actions[:, 5]
    )
    differential_loss = torch.nn.functional.smooth_l1_loss(
        student_wheel_difference, teacher_wheel_difference
    )
    loss = differential_loss
    student_wheel_common = 0.5 * (
        student_actions[:, 2] + student_actions[:, 5]
    )
    teacher_wheel_common = 0.5 * (
        teacher_actions[:, 2] + teacher_actions[:, 5]
    )
    if common_mode_weight > 0.0:
        common_mode_loss = torch.nn.functional.smooth_l1_loss(
            student_wheel_common, teacher_wheel_common
        )
        loss = loss + float(common_mode_weight) * common_mode_loss
    if leg_weight > 0.0:
        leg_loss = torch.nn.functional.smooth_l1_loss(
            student_actions[:, [0, 1, 3, 4]], teacher_actions[:, [0, 1, 3, 4]]
        )
        loss = loss + float(leg_weight) * leg_loss
    return loss


def _configure_teacher_distillation(
    runner,
    *,
    speed_checkpoint: str | None,
    turn_checkpoint: str | None,
    weight: float,
    final_weight: float | None,
    anneal_iterations: int,
    learning_rate: float | None,
    updates: int,
    distill_only_iterations: int,
    speed_anchor_weight: float,
    turn_common_mode_weight: float,
    turn_leg_weight: float,
    speed_replay_only: bool,
    turn_replay_only: bool,
    speed_dataset_path: str | None,
    turn_dataset_path: str | None,
    replay_batch_size: int,
    gradient_projection: bool,
) -> None:
    """Distill any configured fixed speed/turn teachers into one student actor.

    Rollout labels are captured before ``env.step`` so asynchronous resets cannot
    associate a transition with the next episode's command. Straight samples use
    the speed teacher; non-zero-yaw samples use the turn teacher. The final
    policy remains a single stock MLP and keeps the original 25-D deployment API.
    """
    if not speed_checkpoint and not turn_checkpoint and not turn_dataset_path:
        raise ValueError(
            "Teacher distillation requires an online teacher checkpoint or turn replay dataset"
        )
    if speed_dataset_path and not speed_checkpoint:
        raise ValueError("--teacher_speed_dataset requires --teacher_speed_checkpoint")
    if speed_replay_only and not speed_dataset_path:
        raise ValueError("--teacher_speed_replay_only requires --teacher_speed_dataset")
    if turn_replay_only and not turn_dataset_path:
        raise ValueError("--teacher_turn_replay_only requires --teacher_turn_dataset")
    if turn_dataset_path and not turn_checkpoint and not turn_replay_only:
        raise ValueError(
            "--teacher_turn_dataset without --teacher_turn_checkpoint requires "
            "--teacher_turn_replay_only"
        )
    if not 0.0 < float(weight) <= 1.0:
        raise ValueError("--teacher_distill_weight must be in (0, 1]")
    final_weight = float(weight) if final_weight is None else float(final_weight)
    if not 0.0 <= final_weight <= float(weight):
        raise ValueError(
            "--teacher_distill_final_weight must be in [0, teacher_distill_weight]"
        )
    if int(anneal_iterations) < 0:
        raise ValueError("--teacher_distill_anneal_iterations must be >= 0")
    if final_weight != float(weight) and int(anneal_iterations) == 0:
        raise ValueError(
            "A different --teacher_distill_final_weight requires "
            "--teacher_distill_anneal_iterations > 0"
        )
    if int(updates) < 1:
        raise ValueError("--teacher_distill_updates must be >= 1")
    if int(distill_only_iterations) < 0:
        raise ValueError("--teacher_distill_only_iterations must be >= 0")
    if float(speed_anchor_weight) <= 0.0:
        raise ValueError("--teacher_speed_anchor_weight must be positive")
    if not 0.0 <= float(turn_common_mode_weight) <= 2.0:
        raise ValueError("--teacher_turn_common_mode_weight must be in [0, 2]")
    if not 0.0 <= float(turn_leg_weight) <= 2.0:
        raise ValueError("--teacher_turn_leg_weight must be in [0, 2]")
    if int(replay_batch_size) < 1:
        raise ValueError("--teacher_replay_batch_size must be >= 1")
    actor = runner.alg.get_policy()
    speed_teacher = (
        _load_frozen_teacher(actor, speed_checkpoint, "speed")
        if speed_checkpoint and not speed_replay_only
        else None
    )
    # Replay datasets already contain fixed action labels and may come from
    # several independently validated teachers. Do not require or load one
    # arbitrary online turn actor when replay-only mode is selected.
    turn_teacher = (
        _load_frozen_teacher(actor, turn_checkpoint, "turn")
        if turn_checkpoint and not turn_replay_only
        else None
    )
    if speed_teacher is not None:
        speed_teacher.to(runner.device)
    if turn_teacher is not None:
        turn_teacher.to(runner.device)

    unwrapped_env = getattr(runner.env, "unwrapped", runner.env)
    current_command_level = int(getattr(unwrapped_env, "_command_level", 0))
    highspeed_level = int(
        getattr(unwrapped_env.cfg, "command_high_speed_gate_start_level", 9)
    )
    teacher_dataset_metadata: dict[str, float] = {}

    def _load_single_teacher_dataset(path: str, label: str):
        path = os.path.abspath(path)
        payload = torch.load(path, weights_only=False, map_location="cpu")
        trajectory_replay = payload.get("replay_kind") == "trajectory"
        if trajectory_replay and (
            label != "turn"
            or not bool(payload.get("gate_passed", False))
            or bool(payload.get("turn_mirror_augmented", False))
            or not bool(payload.get("turn_direction_covered", False))
            or not payload.get("checkpoint")
        ):
            raise RuntimeError(
                "Trajectory replay requires a passing real-bidirectional turn "
                "evaluation with no mirror augmentation and a source checkpoint"
            )
        dataset_level = payload.get("command_level") if isinstance(payload, dict) else None
        if label == "turn":
            has_real_bidirectional_data = bool(payload.get("turn_direction_covered", False))
            teacher_dataset_metadata["turn_real_bidirectional"] = bool(
                teacher_dataset_metadata.get("turn_real_bidirectional", True)
                and has_real_bidirectional_data
            )
            if dataset_level is not None:
                dataset_level_index = int(dataset_level)
                stage_speeds = tuple(
                    float(value) for value in unwrapped_env.cfg.command_stage_linear_x_mps
                )
                stage_yaws = tuple(
                    float(value) for value in unwrapped_env.cfg.command_stage_yaw_rate_rad_s
                )
                if not 0 <= dataset_level_index < len(stage_speeds):
                    raise RuntimeError(
                        f"Turn teacher dataset command_level={dataset_level_index} is outside "
                        f"the configured stage range [0, {len(stage_speeds) - 1}]"
                    )
                certified_targets = teacher_dataset_metadata.setdefault(
                    "turn_certified_targets", []
                )
                target = (
                    stage_speeds[dataset_level_index],
                    abs(stage_yaws[dataset_level_index]),
                )
                if target not in certified_targets:
                    certified_targets.append(target)
        if label == "speed" and dataset_level is not None:
            stage_speeds = tuple(
                float(value)
                for value in unwrapped_env.cfg.command_stage_linear_x_mps
            )
            dataset_level_index = int(dataset_level)
            if not 0 <= dataset_level_index < len(stage_speeds):
                raise RuntimeError(
                    f"Speed teacher dataset command_level={dataset_level_index} is outside "
                    f"the configured stage range [0, {len(stage_speeds) - 1}]"
                )
            teacher_dataset_metadata["speed_target_limit_mps"] = abs(
                stage_speeds[dataset_level_index]
            )
        if (
            label == "speed"
            and current_command_level >= highspeed_level
            and dataset_level is not None
            and int(dataset_level) < current_command_level
        ):
            print(
                f"[INFO] {label} teacher replay from command_level={int(dataset_level)} "
                f"will be used as retention-only data at command_level={current_command_level}; "
                "online supervision is limited to the teacher's target speed.",
                flush=True,
            )
        if current_command_level >= highspeed_level:
            if not bool(payload.get("quality_filtered", False)):
                raise RuntimeError(
                    f"High-speed {label}-teacher replay must be quality-filtered; regenerate it "
                    "with scripts/evaluate_rmuc_stage.py --save-teacher-dataset"
                )
            if label == "turn" and not (
                bool(payload.get("turn_mirror_augmented", False))
                or bool(payload.get("turn_direction_covered", False))
            ):
                raise RuntimeError(
                    "High-speed turn-teacher replay must contain real positive/negative "
                    "command samples or explicit mirror augmentation"
                )
        if (
            label == "turn"
            and "gate_passed" in payload
            and not bool(payload["gate_passed"])
        ):
            if float(turn_common_mode_weight) > 0.0 or float(turn_leg_weight) > 0.0:
                raise RuntimeError(
                    "Turn-teacher replay from a non-passing checkpoint may supervise only "
                    "the wheel differential. Set --teacher_turn_common_mode_weight 0 and "
                    "--teacher_turn_leg_weight 0 so a failed teacher cannot cap student "
                    "forward speed or posture."
                )
            print(
                "[WARNING] Accepted non-passing turn-teacher replay for differential-only "
                "distillation; wheel common-mode and leg supervision are disabled.",
                flush=True,
            )
        if (
            label == "turn"
            and turn_teacher is not None
            and (float(turn_common_mode_weight) > 0.0 or float(turn_leg_weight) > 0.0)
        ):
            dataset_checkpoint = payload.get("checkpoint")
            if (
                not bool(payload.get("gate_passed", False))
                or not dataset_checkpoint
                or not turn_checkpoint
                or os.path.realpath(str(dataset_checkpoint))
                != os.path.realpath(turn_checkpoint)
            ):
                raise RuntimeError(
                    "Online turn common-mode/leg distillation requires every turn replay "
                    "dataset to pass its gate and originate from the selected "
                    "--teacher_turn_checkpoint. Use differential-only weights for an "
                    "uncertified teacher."
                )
        if label == "speed" and current_command_level >= highspeed_level:
            dataset_checkpoint = payload.get("checkpoint")
            if not dataset_checkpoint or os.path.realpath(str(dataset_checkpoint)) != os.path.realpath(
                speed_checkpoint
            ):
                raise RuntimeError(
                    "High-speed speed-teacher replay must be generated from the selected "
                    "--teacher_speed_checkpoint"
                )
            if not bool(payload.get("gate_passed", False)):
                # A quality-filtered replay is specifically useful for closing
                # the last speed gap. Requiring the source policy to have
                # already passed that same speed gate creates a circular
                # dependency. Permit only a stable source whose sole failed
                # fixed-stage check is temporal speed tracking; all posture,
                # contact, direction, yaw, and termination checks remain hard.
                evaluation_checks = payload.get("evaluation_checks")
                failed_checks = (
                    {
                        str(name)
                        for name, passed in evaluation_checks.items()
                        if not bool(passed)
                    }
                    if isinstance(evaluation_checks, dict)
                    else set()
                )
                if failed_checks != {"mean_per_env_temporal_velocity_error_mps"}:
                    raise RuntimeError(
                        "High-speed speed-teacher replay from a non-passing checkpoint is "
                        "allowed only when temporal speed tracking is its sole failed gate; "
                        f"failed checks: {sorted(failed_checks)}"
                    )
                print(
                    "[INFO] Accepted stable speed-acquisition replay whose sole source "
                    "gate miss is temporal velocity tracking.",
                    flush=True,
                )
            dataset_speed_limit = float(payload.get("max_speed_error_mps", float("inf")))
            allowed_speed_limit = float(
                unwrapped_env.cfg.command_stage_advance_linear_error_mps[
                    int(dataset_level)
                ]
            )
            if dataset_speed_limit > allowed_speed_limit:
                raise RuntimeError(
                    f"High-speed speed-teacher replay error limit {dataset_speed_limit:.3f} m/s "
                    f"exceeds the stage gate {allowed_speed_limit:.3f} m/s"
                )
            evaluation_metrics = payload.get("evaluation_metrics")
            if isinstance(evaluation_metrics, dict):
                evaluated_linear_error = float(
                    evaluation_metrics.get(
                        "mean_per_env_temporal_velocity_error_mps", float("inf")
                    )
                )
                evaluated_yaw_error = float(
                    evaluation_metrics.get("mean_yaw_velocity_error_rad_s", float("inf"))
                )
                if math.isfinite(evaluated_linear_error) and math.isfinite(
                    evaluated_yaw_error
                ):
                    runner._codex_speed_teacher_reference = (
                        evaluated_linear_error,
                        evaluated_yaw_error,
                    )
        observations = payload.get("observations")
        actions = payload.get("actions")
        commands = payload.get("commands")
        if not isinstance(observations, torch.Tensor) or not isinstance(actions, torch.Tensor):
            raise TypeError(f"{label} teacher dataset must contain tensor observations/actions")
        if observations.ndim != 2 or observations.shape[-1] != 25:
            raise RuntimeError(
                f"{label} teacher dataset observations must be [N, 25], got {tuple(observations.shape)}"
            )
        if actions.ndim != 2 or actions.shape != (observations.shape[0], 6):
            raise RuntimeError(
                f"{label} teacher dataset actions must be [N, 6], got {tuple(actions.shape)}"
            )
        if label == "turn" and current_command_level >= highspeed_level:
            dataset_schema_version = int(payload.get("dataset_schema_version", 0))
            dataset_observation_mode = payload.get("observation_mode")
            dataset_observation_scale = payload.get("observation_scale")
            current_observation_mode = str(unwrapped_env.cfg.command_observation_mode)
            current_observation_scale = tuple(
                float(value) for value in unwrapped_env.cfg.command_observation_scale
            )
            if dataset_schema_version < 2:
                raise RuntimeError(
                    f"High-speed turn-teacher replay {path} has no observation schema. "
                    "Legacy v3 replay mixed incompatible command-slot meanings; regenerate "
                    "it with the current scripts/evaluate_rmuc_stage.py."
                )
            if dataset_observation_mode != current_observation_mode:
                raise RuntimeError(
                    f"Turn replay observation_mode={dataset_observation_mode!r} does not "
                    f"match current mode={current_observation_mode!r}"
                )
            if (
                not isinstance(dataset_observation_scale, (tuple, list))
                or len(dataset_observation_scale) != 3
                or any(
                    abs(float(actual) - expected) > 1.0e-6
                    for actual, expected in zip(
                        dataset_observation_scale, current_observation_scale
                    )
                )
            ):
                raise RuntimeError(
                    f"Turn replay observation_scale={dataset_observation_scale!r} does not "
                    f"match current scale={current_observation_scale!r}"
                )
            if not isinstance(commands, torch.Tensor) or commands.shape != (
                observations.shape[0],
                3,
            ):
                raise RuntimeError(
                    "Schema-v2 turn replay must contain commands shaped [N, 3]"
                )
            positive_count = int((commands[:, 2] > 1.0e-4).sum().item())
            negative_count = int((commands[:, 2] < -1.0e-4).sum().item())
            if positive_count < 1 or negative_count < 1:
                raise RuntimeError(
                    "Schema-v2 turn replay must contain both positive and negative yaw commands"
                )
            declared_speed_limit = float(
                payload.get("max_speed_error_mps", float("inf"))
            )
            declared_yaw_limit = float(
                payload.get("max_yaw_error_rad_s", float("inf"))
            )
            stage_linear_limit = float(
                unwrapped_env.cfg.command_stage_advance_linear_error_mps[
                    int(dataset_level)
                ]
            )
            if (
                not trajectory_replay
                and declared_speed_limit > stage_linear_limit + 1.0e-6
            ):
                raise RuntimeError(
                    f"Turn replay declares {declared_speed_limit:.3f} m/s speed error, "
                    f"above stage {int(dataset_level)} gate {stage_linear_limit:.3f} m/s"
                )
            if not trajectory_replay and declared_yaw_limit > float(
                unwrapped_env.cfg.command_advance_yaw_error_rad_s
            ) + 1.0e-6:
                raise RuntimeError(
                    f"Turn replay declares {declared_yaw_limit:.3f} rad/s yaw error, "
                    "above the current yaw gate"
                )
            encoded_speed_error = torch.abs(
                observations[:, 6] / current_observation_scale[0]
            )
            encoded_yaw_error = torch.abs(
                observations[:, 8] / current_observation_scale[2]
            )
            if not trajectory_replay and (
                torch.any(encoded_speed_error > declared_speed_limit + 1.0e-5)
                or torch.any(encoded_yaw_error > declared_yaw_limit + 1.0e-5)
            ):
                raise RuntimeError(
                    "Schema-v2 turn replay contains command-error slots outside its "
                    "declared quality limits"
                )
        if (
            unwrapped_env.cfg.command_observation_mode
            == "tracking_error_with_target"
        ):
            if dataset_level is None:
                raise RuntimeError(
                    f"{label} teacher dataset needs command_level for target-speed migration"
                )
            dataset_level_index = int(dataset_level)
            stage_speeds = tuple(
                float(value) for value in unwrapped_env.cfg.command_stage_linear_x_mps
            )
            if not 0 <= dataset_level_index < len(stage_speeds):
                raise RuntimeError(
                    f"{label} teacher dataset command_level={dataset_level_index} is invalid"
                )
            observations = observations.clone()
            observations[:, 7] = (
                stage_speeds[dataset_level_index]
                * float(unwrapped_env.cfg.command_observation_scale[1])
            )
            print(
                f"[INFO] Migrated {label} replay slot 7 to target_vx="
                f"{stage_speeds[dataset_level_index]:.2f} m/s",
                flush=True,
            )
        if observations.shape[0] < int(replay_batch_size):
            raise RuntimeError(
                f"{label} teacher dataset has {observations.shape[0]} samples, fewer than "
                f"--teacher_replay_batch_size={int(replay_batch_size)}"
            )
        observations = observations.to(runner.device)
        actions = actions.to(runner.device)
        commands = commands.to(runner.device) if isinstance(commands, torch.Tensor) else None
        print(
            f"[INFO] Loaded {label} teacher replay: {path} samples={observations.shape[0]}"
            + (
                f" positive_turn={int((commands[:, 2] > 1.0e-4).sum())}"
                f" negative_turn={int((commands[:, 2] < -1.0e-4).sum())}"
                if commands is not None
                else ""
            ),
            flush=True,
        )
        return observations, actions, commands

    def _load_teacher_dataset(path_spec: str | None, label: str):
        if not path_spec:
            return None
        paths = [part.strip() for part in path_spec.split(",") if part.strip()]
        if label == "speed" and len(paths) != 1:
            raise ValueError("High-speed training requires exactly one speed-teacher dataset")
        datasets = [_load_single_teacher_dataset(path, label) for path in paths]
        if label == "turn":
            if len(datasets) > 1:
                print(
                    f"[INFO] Loaded {len(datasets)} equally weighted turn skill datasets: "
                    f"samples={[dataset[0].shape[0] for dataset in datasets]}",
                    flush=True,
                )
            return datasets
        if len(datasets) == 1:
            return datasets[0]
        observations = torch.cat([dataset[0] for dataset in datasets], dim=0)
        actions = torch.cat([dataset[1] for dataset in datasets], dim=0)
        commands = (
            torch.cat([dataset[2] for dataset in datasets], dim=0)
            if all(dataset[2] is not None for dataset in datasets)
            else None
        )
        print(
            f"[INFO] Combined {len(datasets)} {label} teacher datasets: "
            f"samples={observations.shape[0]}",
            flush=True,
        )
        return observations, actions, commands

    speed_dataset = _load_teacher_dataset(speed_dataset_path, "speed")
    turn_dataset = _load_teacher_dataset(turn_dataset_path, "turn")
    distill_lr = float(learning_rate) if learning_rate is not None else float(runner.alg.learning_rate)
    if distill_lr <= 0.0:
        raise ValueError("--teacher_distill_learning_rate must be positive")
    distill_optimizer = torch.optim.Adam(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=distill_lr,
    )
    state = {"command_vx": [], "command_yaw": [], "trainable_update_count": 0}
    original_act = runner.alg.act
    original_update = runner.alg.update

    def _capture_command_labels():
        command_fn = getattr(unwrapped_env, "_current_command", None)
        if not callable(command_fn):
            raise RuntimeError("Teacher distillation requires RMUC _current_command()")
        command = command_fn()
        if command.ndim != 2 or command.shape[-1] < 3:
            raise RuntimeError(
                f"Teacher distillation expected command shape [num_envs, 3], got {tuple(command.shape)}"
            )
        return command[:, 0].detach().clone(), command[:, 2].detach().clone()

    def _act_with_teacher_label(obs):
        command_vx, command_yaw = _capture_command_labels()
        state["command_vx"].append(command_vx)
        state["command_yaw"].append(command_yaw)
        return original_act(obs)

    def _update_with_teacher_distillation(*args, **kwargs):
        expected_steps = int(runner.alg.storage.num_transitions_per_env)
        if (
            len(state["command_vx"]) != expected_steps
            or len(state["command_yaw"]) != expected_steps
        ):
            raise RuntimeError(
                "Teacher distillation rollout label count mismatch: "
                f"vx={len(state['command_vx'])}, yaw={len(state['command_yaw'])}, "
                f"expected={expected_steps}"
            )
        command_vx = torch.stack(state["command_vx"], dim=0).reshape(-1)
        command_yaw = torch.stack(state["command_yaw"], dim=0).reshape(-1)
        positive_turn_mask = command_yaw > 1.0e-4
        negative_turn_mask = command_yaw < -1.0e-4
        turn_mask = positive_turn_mask | negative_turn_mask
        online_speed_mask = (
            ~turn_mask
            if speed_teacher is not None and not speed_replay_only
            else torch.zeros_like(turn_mask)
        )
        speed_target_limit = teacher_dataset_metadata.get("speed_target_limit_mps")
        if speed_target_limit is not None:
            # A Level-13 speed teacher must preserve its 1.75 m/s skill without
            # supervising Level-14 active samples back toward the teacher's
            # lower common-mode wheel action. Replay still protects the old
            # states while PPO is free to acquire the new 2.0 m/s command.
            online_speed_mask &= torch.abs(command_vx) <= speed_target_limit + 1.0e-4
        online_turn_mask = (
            turn_mask
            if turn_teacher is not None and not turn_replay_only
            else torch.zeros_like(turn_mask)
        )
        certified_turn_targets = teacher_dataset_metadata.get(
            "turn_certified_targets", []
        )
        if torch.any(online_turn_mask) and certified_turn_targets:
            certified_turn_mask = torch.zeros_like(online_turn_mask)
            for certified_vx, certified_yaw_abs in certified_turn_targets:
                certified_turn_mask |= (
                    torch.abs(command_vx - float(certified_vx)) <= 1.0e-4
                ) & (
                    torch.abs(torch.abs(command_yaw) - float(certified_yaw_abs))
                    <= 1.0e-4
                )
            online_turn_mask &= certified_turn_mask
        # The stock PPO update clears/reuses rollout storage.  Clone the
        # observations before that update so the post-PPO teacher step trains
        # on the same labeled rollout instead of a zeroed/reused buffer.
        observations = runner.alg.storage.observations.flatten(0, 1).detach().clone()
        try:
            policy_observations = observations["policy"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "Teacher distillation requires a 'policy' observation group in rollout storage"
            ) from exc
        if policy_observations.ndim != 2 or policy_observations.shape[-1] != 25:
            raise RuntimeError(
                "Teacher distillation expected flattened policy observations [N, 25], "
                f"got {tuple(policy_observations.shape)}"
            )
        if policy_observations.shape[0] != command_yaw.numel():
            raise RuntimeError(
                "Teacher distillation rollout observation/label count mismatch: "
                f"observations={policy_observations.shape[0]}, labels={command_yaw.numel()}"
            )
        action_shape = runner.alg.storage.actions.shape[-1]
        actor_trainable = any(parameter.requires_grad for parameter in actor.parameters())
        distill_only_active = (
            actor_trainable
            and state["trainable_update_count"] < int(distill_only_iterations)
        )
        ppo_actor_states = None
        if distill_only_active:
            ppo_actor_states = [
                (parameter, parameter.requires_grad) for parameter in actor.parameters()
            ]
            for parameter, _ in ppo_actor_states:
                parameter.requires_grad_(False)
        try:
            result = original_update(*args, **kwargs)
        finally:
            if ppo_actor_states is not None:
                for parameter, requires_grad in ppo_actor_states:
                    parameter.requires_grad_(requires_grad)
        if int(anneal_iterations) > 0:
            anneal_fraction = min(
                state["trainable_update_count"] / float(anneal_iterations), 1.0
            )
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * anneal_fraction))
            effective_weight = final_weight + (float(weight) - final_weight) * cosine_factor
        else:
            anneal_fraction = 0.0
            effective_weight = float(weight)
        # Scaling a loss by a constant is largely cancelled by Adam's moment
        # normalization. Scale the optimizer LR relative to the initial
        # strength so the annealing changes the actual parameter step while
        # retaining the configured initial distillation learning rate.
        effective_distill_lr = distill_lr * effective_weight / float(weight)
        for parameter_group in distill_optimizer.param_groups:
            parameter_group["lr"] = effective_distill_lr
        teacher_gradient_cosine = torch.ones((), device=runner.device)
        teacher_gradient_conflict = torch.zeros((), device=runner.device)
        if actor_trainable:
            with torch.no_grad():
                teacher_actions = torch.empty(
                    (turn_mask.numel(), action_shape), device=runner.device
                )
                if torch.any(online_speed_mask):
                    teacher_actions[online_speed_mask] = speed_teacher(observations[online_speed_mask])
                if torch.any(online_turn_mask):
                    turn_observations = observations[online_turn_mask]
                    direct_turn_actions = turn_teacher(turn_observations)
                    if teacher_dataset_metadata.get("turn_real_bidirectional", False):
                        # Preserve learned direction-specific compensation when
                        # the teacher has real passing samples for both yaw signs.
                        # The RMUC mass/contact model is not perfectly mirrored;
                        # averaging direct and reflected actions erased this
                        # compensation and made the weak direction underspeed.
                        teacher_actions[online_turn_mask] = direct_turn_actions
                    else:
                        mirrored_turn_observations = turn_observations.clone()
                        mirrored_turn_observations["policy"] = _mirror_rmuc_observations(
                            turn_observations["policy"]
                        )
                        mirrored_turn_actions = _mirror_rmuc_actions(
                            turn_teacher(mirrored_turn_observations)
                        )
                        teacher_actions[online_turn_mask] = 0.5 * (
                            direct_turn_actions + mirrored_turn_actions
                        )
            # Normalize group contributions so a 30% rehearsal draw cannot make
            # the turn teacher's gradient disappear from a small rollout.
            for _ in range(int(updates)):
                student_actions = actor(observations)
                weighted_group_losses = []
                group_weights = []
                speed_group_losses = []
                speed_group_weights = []
                turn_group_objectives = []
                turn_group_weights = []
                speed_loss = torch.zeros((), device=runner.device)
                turn_loss = torch.zeros((), device=runner.device)
                if torch.any(online_speed_mask):
                    speed_loss = torch.nn.functional.smooth_l1_loss(
                        student_actions[online_speed_mask], teacher_actions[online_speed_mask]
                    )
                    weighted_group_losses.append(float(speed_anchor_weight) * speed_loss)
                    group_weights.append(float(speed_anchor_weight))
                    speed_group_losses.append(speed_loss)
                    speed_group_weights.append(float(speed_anchor_weight))
                online_turn_direction_losses = []
                for direction_mask in (
                    positive_turn_mask & online_turn_mask,
                    negative_turn_mask & online_turn_mask,
                ):
                    if torch.any(direction_mask):
                        direction_loss = _rmuc_turn_skill_loss(
                            student_actions[direction_mask],
                            teacher_actions[direction_mask],
                            # Certified same-checkpoint replay proves that the
                            # online teacher's forward common mode and posture
                            # are safe. Supervise them on student-visited states
                            # so turn underspeed is corrected along with yaw.
                            common_mode_weight=turn_common_mode_weight,
                            leg_weight=turn_leg_weight,
                        )
                        online_turn_direction_losses.append(direction_loss)
                if online_turn_direction_losses:
                    turn_loss = torch.stack(online_turn_direction_losses).mean()
                    weighted_group_losses.append(turn_loss)
                    group_weights.append(1.0)
                    turn_group_objectives.append(turn_loss)
                    turn_group_weights.append(1.0)
                replay_speed_loss = torch.zeros((), device=runner.device)
                replay_turn_loss = torch.zeros((), device=runner.device)
                if speed_dataset is not None:
                    speed_obs, speed_actions, _ = speed_dataset
                    speed_indices = torch.randint(
                        speed_obs.shape[0], (int(replay_batch_size),), device=runner.device
                    )
                    replay_speed_loss = torch.nn.functional.smooth_l1_loss(
                        actor({"policy": speed_obs[speed_indices]}), speed_actions[speed_indices]
                    )
                    weighted_group_losses.append(
                        float(speed_anchor_weight) * replay_speed_loss
                    )
                    group_weights.append(float(speed_anchor_weight))
                    speed_group_losses.append(replay_speed_loss)
                    speed_group_weights.append(float(speed_anchor_weight))
                if turn_dataset is not None:
                    replay_turn_losses = []
                    for turn_obs, turn_actions, turn_commands in turn_dataset:
                        if turn_commands is not None:
                            positive_pool = (turn_commands[:, 2] > 1.0e-4).nonzero(
                                as_tuple=False
                            ).flatten()
                            negative_pool = (turn_commands[:, 2] < -1.0e-4).nonzero(
                                as_tuple=False
                            ).flatten()
                            positive_batch = int(replay_batch_size) // 2
                            negative_batch = int(replay_batch_size) - positive_batch
                            turn_indices = torch.cat(
                                (
                                    positive_pool[
                                        torch.randint(
                                            positive_pool.numel(),
                                            (positive_batch,),
                                            device=runner.device,
                                        )
                                    ],
                                    negative_pool[
                                        torch.randint(
                                            negative_pool.numel(),
                                            (negative_batch,),
                                            device=runner.device,
                                        )
                                    ],
                                )
                            )
                            turn_indices = turn_indices[
                                torch.randperm(turn_indices.numel(), device=runner.device)
                            ]
                        else:
                            turn_indices = torch.randint(
                                turn_obs.shape[0],
                                (int(replay_batch_size),),
                                device=runner.device,
                            )
                        dataset_loss = _rmuc_turn_skill_loss(
                            actor({"policy": turn_obs[turn_indices]}),
                            turn_actions[turn_indices],
                            common_mode_weight=turn_common_mode_weight,
                            leg_weight=turn_leg_weight,
                        )
                        replay_turn_losses.append(dataset_loss)
                    replay_turn_loss = torch.stack(replay_turn_losses).mean()
                    weighted_group_losses.append(replay_turn_loss)
                    group_weights.append(1.0)
                    turn_group_objectives.append(replay_turn_loss)
                    turn_group_weights.append(1.0)
                if not weighted_group_losses:
                    raise RuntimeError("Teacher distillation rollout contained no labeled samples")
                total_group_weight = sum(group_weights)
                distill_loss = torch.stack(weighted_group_losses).sum() / total_group_weight
                distill_optimizer.zero_grad(set_to_none=True)
                if gradient_projection and speed_group_losses and turn_group_objectives:
                    trainable_parameters = [
                        parameter for parameter in actor.parameters() if parameter.requires_grad
                    ]
                    speed_component = sum(
                        group_weight * group_loss
                        for group_weight, group_loss in zip(
                            speed_group_weights, speed_group_losses
                        )
                    ) / total_group_weight
                    turn_component = sum(
                        group_weight * group_loss
                        for group_weight, group_loss in zip(
                            turn_group_weights, turn_group_objectives
                        )
                    ) / total_group_weight
                    speed_gradients = torch.autograd.grad(
                        speed_component,
                        trainable_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    turn_gradients = torch.autograd.grad(
                        turn_component,
                        trainable_parameters,
                        allow_unused=True,
                    )
                    gradient_dot = torch.zeros((), device=runner.device)
                    speed_norm_sq = torch.zeros((), device=runner.device)
                    turn_norm_sq = torch.zeros((), device=runner.device)
                    for speed_gradient, turn_gradient in zip(
                        speed_gradients, turn_gradients
                    ):
                        if speed_gradient is not None:
                            speed_norm_sq += torch.sum(speed_gradient.square())
                        if turn_gradient is not None:
                            turn_norm_sq += torch.sum(turn_gradient.square())
                        if speed_gradient is not None and turn_gradient is not None:
                            gradient_dot += torch.sum(speed_gradient * turn_gradient)
                    teacher_gradient_cosine = gradient_dot / torch.sqrt(
                        speed_norm_sq.clamp_min(1.0e-12)
                        * turn_norm_sq.clamp_min(1.0e-12)
                    )
                    conflicting_dot = torch.minimum(
                        gradient_dot, torch.zeros_like(gradient_dot)
                    )
                    teacher_gradient_conflict = (gradient_dot < 0.0).float()
                    speed_projection_scale = (
                        conflicting_dot / turn_norm_sq.clamp_min(1.0e-12)
                    )
                    turn_projection_scale = (
                        conflicting_dot / speed_norm_sq.clamp_min(1.0e-12)
                    )
                    for parameter, speed_gradient, turn_gradient in zip(
                        trainable_parameters, speed_gradients, turn_gradients
                    ):
                        if speed_gradient is None and turn_gradient is None:
                            continue
                        if speed_gradient is None:
                            parameter.grad = turn_gradient
                        elif turn_gradient is None:
                            parameter.grad = speed_gradient
                        else:
                            parameter.grad = (
                                speed_gradient
                                - speed_projection_scale * turn_gradient
                                + turn_gradient
                                - turn_projection_scale * speed_gradient
                            )
                else:
                    distill_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                distill_optimizer.step()
            state["trainable_update_count"] += 1
        else:
            # Critic-only warmup intentionally freezes the student actor.
            distill_loss = torch.zeros((), device=runner.device)
            speed_loss = torch.zeros((), device=runner.device)
            turn_loss = torch.zeros((), device=runner.device)
            replay_speed_loss = torch.zeros((), device=runner.device)
            replay_turn_loss = torch.zeros((), device=runner.device)
        result["teacher_distill"] = float(distill_loss.detach().cpu())
        result["teacher_speed_action"] = float(speed_loss.detach().cpu())
        result["teacher_turn_action"] = float(turn_loss.detach().cpu())
        result["teacher_replay_speed_action"] = float(replay_speed_loss.detach().cpu())
        result["teacher_replay_turn_action"] = float(replay_turn_loss.detach().cpu())
        result["teacher_turn_fraction"] = float(online_turn_mask.float().mean().detach().cpu())
        result["teacher_speed_fraction"] = float(online_speed_mask.float().mean().detach().cpu())
        result["teacher_distill_effective_weight"] = effective_weight
        result["teacher_distill_effective_lr"] = effective_distill_lr
        # The stock logger prints scalar values to four decimals, which would
        # render a 1e-5 LR as 0.0000. Keep an explicitly readable diagnostic.
        result["teacher_distill_effective_lr_x1e5"] = effective_distill_lr * 1.0e5
        result["teacher_distill_anneal_fraction"] = anneal_fraction
        result["teacher_speed_anchor_weight"] = float(speed_anchor_weight)
        result["teacher_distill_only_active"] = float(distill_only_active)
        result["teacher_gradient_cosine"] = float(
            teacher_gradient_cosine.detach().cpu()
        )
        result["teacher_gradient_conflict"] = float(
            teacher_gradient_conflict.detach().cpu()
        )
        state["command_vx"].clear()
        state["command_yaw"].clear()
        return result

    runner.alg.act = _act_with_teacher_label
    runner.alg.update = _update_with_teacher_distillation
    runner._codex_teacher_distillation = {
        "speed_checkpoint": os.path.abspath(speed_checkpoint) if speed_checkpoint else None,
        "turn_checkpoint": os.path.abspath(turn_checkpoint) if turn_checkpoint else None,
        "weight": float(weight),
        "final_weight": final_weight,
        "anneal_iterations": int(anneal_iterations),
        "distill_only_iterations": int(distill_only_iterations),
        "gradient_projection": bool(gradient_projection),
        "learning_rate": distill_lr,
        "updates": int(updates),
        "speed_anchor_weight": float(speed_anchor_weight),
        "turn_common_mode_weight": float(turn_common_mode_weight),
        "turn_leg_weight": float(turn_leg_weight),
        "speed_replay_only": bool(speed_replay_only),
        "turn_replay_only": bool(turn_replay_only),
        "speed_dataset": (
            [os.path.abspath(path.strip()) for path in speed_dataset_path.split(",") if path.strip()]
            if speed_dataset_path else []
        ),
        "turn_dataset": (
            [os.path.abspath(path.strip()) for path in turn_dataset_path.split(",") if path.strip()]
            if turn_dataset_path else []
        ),
        "replay_batch_size": int(replay_batch_size),
        "speed_target_limit_mps": teacher_dataset_metadata.get(
            "speed_target_limit_mps"
        ),
    }
    print(
        f"[INFO] Teacher distillation enabled: weight={float(weight):.3f}->"
        f"{final_weight:.3f} over {int(anneal_iterations)} actor updates, "
        f"initial_lr={distill_lr:.3g}, updates={int(updates)}; "
        f"speed_anchor_weight={float(speed_anchor_weight):.2f}; "
        f"turn_replay_common_mode_weight={float(turn_common_mode_weight):.2f}; "
        f"turn_replay_leg_weight={float(turn_leg_weight):.2f}; "
        f"gradient_projection={bool(gradient_projection)}; "
        f"replay_batch={int(replay_batch_size) if (speed_dataset is not None or turn_dataset is not None) else 0}; "
        f"online_speed={'replay-only' if speed_replay_only else ('enabled' if speed_teacher is not None else 'disabled')}, "
        f"speed_teacher_target_limit={teacher_dataset_metadata.get('speed_target_limit_mps')}, "
        f"online_turn={'replay-only' if turn_replay_only else ('enabled' if turn_teacher is not None else 'disabled')}; "
        "online turn teacher uses certified current-state wheel/leg supervision; "
        f"real_bidirectional={bool(teacher_dataset_metadata.get('turn_real_bidirectional', False))}; "
        f"certified_targets={teacher_dataset_metadata.get('turn_certified_targets', [])}; "
        f"turn common-mode weight={float(turn_common_mode_weight):.2f}, leg weight="
        f"{float(turn_leg_weight):.2f}.",
        flush=True,
    )


def _notify_env_training_progress(env, iteration: int) -> None:
    """Seed env-side step curricula with the runner iteration before learn()."""

    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "set_training_progress"):
        unwrapped.set_training_progress(iteration=iteration)


def _configure_action_std(
    runner,
    *,
    freeze: bool,
    freeze_leg: bool,
    override: str | None,
    min_wheel_std: float | None,
    command_level: int | None = None,
) -> None:
    """Configure state-independent Gaussian exploration parameters.

    A frozen ``0.004`` std is useful for protecting the standing basin, but it
    is not a valid high-speed adaptation setting: the policy cannot explore
    the additional wheel torque/lean combinations needed above 1.5 m/s.  Fail
    early for that specific unsafe combination instead of spending thousands
    of iterations on a gate that cannot be reached.
    """

    if not freeze and not freeze_leg and override is None and min_wheel_std is None:
        return
    if min_wheel_std is not None and float(min_wheel_std) <= 0.0:
        raise ValueError("--min_wheel_action_std must be positive")
    if freeze and min_wheel_std is not None:
        raise ValueError(
            "--min_wheel_action_std requires trainable wheel std; do not combine it with "
            "--freeze_action_std"
        )
    policy = runner.alg.get_policy()
    distribution = getattr(policy, "distribution", None)
    named_std_parameters = [
        (name, parameter)
        for name, parameter in distribution.named_parameters()
        if name in {"std_param", "log_std_param"}
    ] if distribution is not None else []
    if not named_std_parameters:
        raise RuntimeError("action-std configuration requested, but policy exposes no action-std parameter")
    if override is not None:
        override_values = [float(value.strip()) for value in override.split(",") if value.strip()]
        if len(named_std_parameters) != 1:
            raise RuntimeError("per-action std override requires exactly one Gaussian std parameter")
        name, parameter = named_std_parameters[0]
        if len(override_values) != parameter.numel() or any(value <= 0.0 for value in override_values):
            raise ValueError(
                f"--action_std_override requires {parameter.numel()} positive values, got {override_values}"
            )
        values = torch.tensor(override_values, device=parameter.device, dtype=parameter.dtype)
        if name == "log_std_param":
            values = torch.log(values)
        with torch.no_grad():
            parameter.copy_(values)
    std_values = []
    for name, parameter in named_std_parameters:
        values = torch.exp(parameter) if name == "log_std_param" else parameter
        std_values.extend(values.detach().cpu().tolist())
        if freeze:
            parameter.requires_grad_(False)
        elif freeze_leg:
            if parameter.numel() != 6:
                raise RuntimeError(
                    "--freeze_leg_action_std requires the six-action RMUC policy layout"
                )
            gradient_mask = torch.tensor(
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.register_hook(lambda gradient, mask=gradient_mask: gradient * mask)
    if freeze and command_level is not None and command_level >= 5:
        # The RMUC command stages are zero-based; level 5 is the 1.5 m/s
        # straight stage.  Keep the four leg channels conservative, but do
        # not freeze both wheel channels at the standing-stage std.
        wheel_std = [std_values[index] for index in (2, 5) if index < len(std_values)]
        if wheel_std and max(wheel_std) <= 0.0045:
            raise RuntimeError(
                "High-speed RMUC command stage refuses frozen wheel action std "
                f"{wheel_std}; use a trainable std, e.g. "
                "--action_std_override 0.004,0.004,0.02,0.004,0.004,0.02 "
                "without --freeze_action_std."
            )
    if freeze:
        state = "frozen"
    elif freeze_leg:
        state = "leg-frozen/wheel-trainable"
    else:
        state = "trainable"
    print(
        f"[INFO]: Action std parameters ({state}): {std_values}",
        flush=True,
    )
    if min_wheel_std is not None:
        if len(named_std_parameters) != 1 or named_std_parameters[0][1].numel() != 6:
            raise RuntimeError(
                "--min_wheel_action_std requires the six-action RMUC Gaussian policy layout"
            )
        std_name, std_parameter = named_std_parameters[0]
        wheel_indices = (2, 5)
        lower_bound = (
            math.log(float(min_wheel_std))
            if std_name == "log_std_param"
            else float(min_wheel_std)
        )

        def _clamp_wheel_std() -> float:
            with torch.no_grad():
                # Advanced indexing returns a copy: in-place clamp on that
                # temporary never updated the actual Gaussian parameter.
                std_parameter[list(wheel_indices)] = std_parameter[list(wheel_indices)].clamp(min=lower_bound)
                current = (
                    torch.exp(std_parameter[list(wheel_indices)])
                    if std_name == "log_std_param"
                    else std_parameter[list(wheel_indices)]
                )
            return float(current.min().detach().cpu())

        initial_wheel_std = _clamp_wheel_std()
        original_update = runner.alg.update

        def _update_with_wheel_std_floor(*args, **kwargs):
            result = original_update(*args, **kwargs)
            current_wheel_std = _clamp_wheel_std()
            if isinstance(result, dict):
                result["wheel_action_std_floor"] = float(min_wheel_std)
                result["wheel_action_std_min"] = current_wheel_std
            return result

        runner.alg.update = _update_with_wheel_std_floor
        print(
            f"[INFO]: Wheel action std floor enabled: {float(min_wheel_std):.4f} "
            f"(initial minimum={initial_wheel_std:.4f}).",
            flush=True,
        )


def _configure_temporally_correlated_wheel_noise(runner, hold_steps: int) -> None:
    """Hold wheel Gaussian samples long enough to affect inverted-pendulum speed."""

    hold_steps = int(hold_steps)
    if hold_steps < 1:
        raise ValueError("--wheel_noise_hold_steps must be >= 1")
    if hold_steps == 1:
        return
    policy = runner.alg.get_policy()
    distribution = getattr(policy, "distribution", None)
    sample_fn = getattr(distribution, "sample", None)
    if distribution is None or not callable(sample_fn):
        raise RuntimeError("Temporal wheel noise requires a Gaussian policy distribution")
    state = {"wheel_noise": None, "age": None}
    wheel_indices = (2, 5)
    unwrapped_env = getattr(runner.env, "unwrapped", runner.env)
    hold_mask_fn = getattr(unwrapped_env, "get_wheel_exploration_hold_mask", None)
    if not callable(hold_mask_fn):
        raise RuntimeError(
            "Temporal wheel noise requires env.get_wheel_exploration_hold_mask()"
        )

    def _sample_with_held_wheel_noise():
        actions = sample_fn()
        # PPO calls actor(..., stochastic_output=True) while recomputing
        # log-probabilities, but ignores that sampled value. Do not advance the
        # rollout noise state during gradient updates.
        if torch.is_grad_enabled():
            return actions
        mean = distribution.mean
        std = distribution.std
        num_envs = actions.shape[0]
        if state["wheel_noise"] is None or state["wheel_noise"].shape[0] != num_envs:
            state["wheel_noise"] = torch.zeros(
                (num_envs, len(wheel_indices)), device=actions.device, dtype=actions.dtype
            )
            state["age"] = torch.full(
                (num_envs,), hold_steps, device=actions.device, dtype=torch.long
            )
        hold_mask = hold_mask_fn()
        if hold_mask.shape != state["age"].shape:
            raise RuntimeError(
                "Wheel exploration hold mask shape does not match the rollout batch"
            )
        # Outside the underspeed region, keep the original independent sample
        # and invalidate any held noise. This prevents temporal exploration
        # from continuing to accelerate after v_actual reaches v_cmd.
        state["age"][~hold_mask] = hold_steps
        refresh = hold_mask & (state["age"] >= hold_steps)
        if torch.any(refresh):
            # Straight-line speed is controlled by wheel common mode. Two
            # independent held samples mostly inject differential yaw/roll,
            # conflict with wheel-sync shaping, and provide almost no clean
            # evidence for increasing propulsion. Use one shared standard-
            # normal sample; per-wheel std scaling below remains intact.
            common_noise = torch.randn(
                (int(refresh.sum()), 1),
                device=actions.device,
                dtype=actions.dtype,
            )
            state["wheel_noise"][refresh] = common_noise.expand(
                -1, len(wheel_indices)
            )
            state["age"][refresh] = 0
        if torch.any(hold_mask):
            held_actions = actions[hold_mask]
            held_actions[:, wheel_indices] = (
                mean[hold_mask][:, wheel_indices]
                + std[hold_mask][:, wheel_indices] * state["wheel_noise"][hold_mask]
            )
            actions[hold_mask] = held_actions
            state["age"][hold_mask] += 1
        return actions

    process_env_step_fn = runner.alg.process_env_step

    def _process_env_step_with_noise_reset(obs, rewards, dones, extras):
        result = process_env_step_fn(obs, rewards, dones, extras)
        if state["age"] is not None and torch.any(dones):
            state["age"][dones] = hold_steps
        return result

    distribution.sample = _sample_with_held_wheel_noise
    runner.alg.process_env_step = _process_env_step_with_noise_reset
    print(
        f"[INFO] Common-mode wheel exploration noise hold enabled: {hold_steps} steps "
        f"({hold_steps * float(runner.env.unwrapped.step_dt):.3f} s).",
        flush=True,
    )


def _configure_env_action_credit(runner) -> None:
    """Execute environment-side residuals exactly once during rollouts.

    RMUC applies bounded speed/pitch/yaw residuals in ``_pre_physics_step``.
    The residual is a deterministic outer-loop controller, not a sample from
    the actor's Gaussian distribution.  It must therefore be cached for
    PhysX, while PPO keeps the *raw* actor sample and its original log-prob.
    Treating the residual action as a Gaussian sample is numerically unsafe at
    the tiny high-speed action std (the likelihood ratio can overflow and
    corrupt the actor with NaNs).  ``prepare_action_for_policy`` handles the
    one-time cache; this wrapper deliberately does not mutate ``transition``.
    """

    unwrapped = getattr(runner.env, "unwrapped", runner.env)
    prepare = getattr(unwrapped, "prepare_action_for_policy", None)
    if not callable(prepare):
        return
    algorithm = runner.alg
    original_act = algorithm.act

    def act_with_env_credit(obs):
        raw_actions = original_act(obs)
        applied_actions = prepare(raw_actions)
        # Keep transition.actions/actions_log_prob from ``original_act``.  The
        # residual is outside the actor distribution; replacing these fields
        # makes PPO evaluate a clipped, distribution-out-of-support action and
        # can produce exp(log_prob_delta)=inf/NaN after a few updates.
        return applied_actions

    algorithm.act = act_with_env_credit
    algorithm._codex_env_action_credit = True
    print(
        "[INFO] RMUC residual execution alignment enabled: residual is applied once "
        "for PhysX; PPO keeps the raw actor action/log-probability.",
        flush=True,
    )


def _configure_critic_warmup(runner, iterations: int) -> None:
    """Temporarily freeze the actor while a changed reward/observation critic adapts."""

    iterations = max(int(iterations), 0)
    if iterations == 0:
        return
    actor = runner.alg.get_policy()
    parameter_states = [(parameter, parameter.requires_grad) for parameter in actor.parameters()]
    for parameter, _ in parameter_states:
        parameter.requires_grad_(False)
    runner._codex_actor_warmup = {
        "release_iteration": int(runner.current_learning_iteration) + iterations,
        "parameter_states": parameter_states,
        "active": True,
    }
    print(
        f"[INFO]: Critic-only warmup for {iterations} iterations; actor temporarily frozen.",
        flush=True,
    )


def _configure_actor_level_freeze(runner, target_level: int | None) -> None:
    """Freeze a validated actor until the environment advances to ``target_level``."""

    if target_level is None:
        return
    actor = runner.alg.get_policy()
    parameter_states = [(parameter, parameter.requires_grad) for parameter in actor.parameters()]
    for parameter, _ in parameter_states:
        parameter.requires_grad_(False)
    runner._codex_actor_level_freeze = {
        "target_level": int(target_level),
        "parameter_states": parameter_states,
        "active": True,
    }
    print(
        f"[INFO]: Actor and action std frozen until command level {int(target_level)}; "
        "critic remains trainable.",
        flush=True,
    )


def _restore_actor_trainability(runner, parameter_states) -> None:
    """Restore non-bank flags, then route heads for the current (not saved) level."""
    for parameter, requires_grad in parameter_states:
        parameter.requires_grad_(requires_grad)
    if getattr(runner, "_codex_turn_residual_bank_schedule", False):
        env = getattr(runner.env, "unwrapped", runner.env)
        set_rmuc_turn_delta_trainable(runner.alg.get_policy(), int(env._command_level))


def _configure_turn_residual_bank_schedule(runner) -> None:
    """Route PPO actor gradients only to the residual head for the active level."""

    actor = runner.alg.get_policy()
    if not hasattr(actor, "rmuc_turn_delta_adapters"):
        return
    env = getattr(runner.env, "unwrapped", runner.env)
    original_update = runner.alg.update
    active_level = None

    def _select_active_head() -> tuple[int, str | None]:
        nonlocal active_level
        level = int(getattr(env, "_command_level", -1))
        if level != active_level:
            label = set_rmuc_turn_delta_trainable(actor, level)
            active_level = level
            print(
                "[INFO] RMUC turn residual bank route: "
                f"command_level={level}, trainable_head={label or 'none'}, "
                "certified_heads_frozen=True.",
                flush=True,
            )
        else:
            label = None
            index = getattr(actor, "_rmuc_turn_delta_active_index", None)
            if index is not None:
                label = RMUC_TURN_DELTA_SPECS[index][-1]
        # A level boundary during critic calibration must not unfreeze the new
        # actor head early. On release, the current-level route is reapplied.
        if any(getattr(runner, key, {}).get("active", False)
               for key in ("_codex_actor_warmup", "_codex_actor_level_freeze")):
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
        return level, label

    _select_active_head()

    def _update_with_turn_residual_bank(*args, **kwargs):
        level, label = _select_active_head()
        result = original_update(*args, **kwargs)
        if isinstance(result, dict):
            result["turn_delta_command_level"] = float(level)
            result["turn_delta_head_active"] = float(label is not None)
        return result

    runner.alg.update = _update_with_turn_residual_bank
    runner._codex_turn_residual_bank_schedule = True


class _CommandLevelReachedStop(RuntimeError):
    """Internal clean-stop signal emitted after a requested level is reached."""


def _enable_terminal_iteration_progress(runner, stop_at_command_level: int | None = None) -> None:
    """Ensure every training iteration is visible immediately in the terminal.

    Newer stock RSL-RL already prints a full metrics block from ``Logger.log``;
    this small wrapper adds an unbuffered one-line marker before that block so
    progress remains visible even when the detailed logger output is delayed or
    redirected by a launcher.
    """

    logger = getattr(runner, "logger", None)
    log_fn = getattr(logger, "log", None)
    if logger is None or not callable(log_fn) or getattr(logger, "_codex_iteration_log", False):
        return
    initial_level = getattr(getattr(runner.env, "unwrapped", runner.env), "_command_level", None)
    last_command_level = initial_level

    def _log_with_iteration(*args, **kwargs):
        nonlocal last_command_level
        iteration = kwargs.get("it")
        total = kwargs.get("total_it")
        if iteration is not None:
            suffix = f"/{total}" if total is not None else ""
            print(f"[INFO] Learning iteration {iteration}{suffix}", flush=True)
        result = log_fn(*args, **kwargs)
        # Kit may consume SIGINT before Python's KeyboardInterrupt handler.
        # A run-local STOP marker provides checkpointed shutdown at an update
        # boundary without saving halfway through a PPO optimizer step.
        if (iteration is not None and logger.log_dir is not None
                and os.path.isfile(os.path.join(logger.log_dir, "STOP"))):
            stop_path = os.path.join(logger.log_dir, f"model_requested_stop_{iteration}.pt")
            runner.save(stop_path, infos={"stop_reason": "stop_marker", "iteration": int(iteration)})
            raise _CommandLevelReachedStop(f"STOP marker consumed; checkpoint retained at {stop_path}")
        actor_warmup = getattr(runner, "_codex_actor_warmup", None)
        if (
            actor_warmup
            and actor_warmup["active"]
            and iteration is not None
            and int(iteration) + 1 >= actor_warmup["release_iteration"]
        ):
            _restore_actor_trainability(runner, actor_warmup["parameter_states"])
            actor_warmup["active"] = False
            print(
                f"[INFO] Critic warmup complete; actor updates enabled after iteration {iteration}.",
                flush=True,
            )
        current_level = getattr(
            getattr(runner.env, "unwrapped", runner.env), "_command_level", None
        )
        actor_level_freeze = getattr(runner, "_codex_actor_level_freeze", None)
        if (
            actor_level_freeze
            and actor_level_freeze["active"]
            and current_level is not None
            and int(current_level) >= actor_level_freeze["target_level"]
        ):
            _restore_actor_trainability(runner, actor_level_freeze["parameter_states"])
            actor_level_freeze["active"] = False
            print(
                f"[INFO] Command level {current_level} reached; actor updates enabled.",
                flush=True,
            )
        level_advanced = (
            iteration is not None
            and current_level is not None
            and last_command_level is not None
            and current_level != last_command_level
        )
        if level_advanced and logger.log_dir is not None:
            milestone_path = os.path.join(
                logger.log_dir, f"model_level_{current_level}_entry_{iteration}.pt"
            )
            runner.save(
                milestone_path,
                infos={"command_level": int(current_level), "entry_iteration": int(iteration)},
            )
            print(
                f"[INFO] Saved command-level entry checkpoint: {milestone_path}",
                flush=True,
            )
        last_command_level = current_level
        if (
            level_advanced
            and stop_at_command_level is not None
            and current_level >= stop_at_command_level
        ):
            raise _CommandLevelReachedStop(
                f"reached requested command level {current_level}; "
                f"entry checkpoint retained in {logger.log_dir}"
            )
        return result

    logger.log = _log_with_iteration
    logger._codex_iteration_log = True


def _configure_domain_action_diagnostics(runner) -> None:
    """Measure adaptation on saved rollout states without advancing sensors/RNG."""
    actor = runner.alg.get_policy()
    bank = getattr(actor, "rmuc_domain_residual_bank", None)
    if bank is None:
        return
    original_update = runner.alg.update

    def update_with_domain_diagnostics(*args, **kwargs):
        from rmuc_residual_safety import rollout_probe
        probe = rollout_probe(runner.alg.storage.observations["policy"], limit=256)
        with torch.no_grad():
            before = bank(actor.get_latent({"policy": probe})).clone()
        result = original_update(*args, **kwargs)
        with torch.no_grad():
            after = bank(actor.get_latent({"policy": probe}))
        if isinstance(result, dict):
            result["domain_residual_action_abs_max"] = float(after.abs().max())
            result["domain_residual_action_abs_mean"] = float(after.abs().mean())
            result["domain_residual_action_update_abs_max"] = float((after - before).abs().max())
        return result

    runner.alg.update = update_with_domain_diagnostics


def _enable_actor_update_diagnostics(runner) -> None:
    """Log the actor's real parameter movement after every PPO update."""

    logger = getattr(runner, "logger", None)
    log_fn = getattr(logger, "log", None)
    if logger is None or not callable(log_fn) or getattr(logger, "_codex_actor_update_log", False):
        return
    actor = runner.alg.get_policy()
    tracked = [
        (name, parameter)
        for name, parameter in actor.named_parameters()
        if not name.startswith("distribution.")
    ]
    if not tracked:
        raise RuntimeError("Actor update diagnostics found no policy-mean parameters")
    output_weight_names = [
        name
        for name, parameter in tracked
        if name.endswith(".weight") and parameter.requires_grad
    ]
    if not output_weight_names:
        output_weight_names = [
            name
            for name, _ in tracked
            if name.startswith("rmuc_turn_adapter.") and name.endswith(".weight")
        ]
    if not output_weight_names:
        output_weight_names = [name for name, _ in tracked if name.endswith(".weight")]
    output_prefix = output_weight_names[-1].rsplit(".", 1)[0] + "."
    optimizer_parameter_ids = {
        id(parameter)
        for group in runner.alg.optimizer.param_groups
        for parameter in group["params"]
    }
    missing = [
        name
        for name, parameter in tracked
        if parameter.requires_grad and id(parameter) not in optimizer_parameter_ids
    ]
    if missing:
        raise RuntimeError(f"Actor parameters are missing from the PPO optimizer: {missing}")
    previous = {name: parameter.detach().clone() for name, parameter in tracked}

    def _log_with_actor_update(*args, **kwargs):
        loss_dict = kwargs.get("loss_dict")
        if loss_dict is not None:
            delta_sq = 0.0
            parameter_sq = 0.0
            output_delta_sq = 0.0
            output_parameter_sq = 0.0
            trainable = 0
            total = 0
            for name, parameter in tracked:
                current = parameter.detach()
                delta_sq += float(torch.sum((current - previous[name]).square()))
                parameter_sq += float(torch.sum(previous[name].square()))
                if name.startswith(output_prefix):
                    output_delta_sq += float(torch.sum((current - previous[name]).square()))
                    output_parameter_sq += float(torch.sum(previous[name].square()))
                trainable += parameter.numel() if parameter.requires_grad else 0
                total += parameter.numel()
            loss_dict["actor_update_l2"] = math.sqrt(delta_sq)
            loss_dict["actor_relative_update"] = math.sqrt(delta_sq) / max(
                math.sqrt(parameter_sq), 1.0e-12
            )
            loss_dict["actor_output_relative_update"] = math.sqrt(output_delta_sq) / max(
                math.sqrt(output_parameter_sq), 1.0e-12
            )
            loss_dict["actor_trainable_fraction"] = trainable / max(total, 1)
        result = log_fn(*args, **kwargs)
        for name, parameter in tracked:
            previous[name].copy_(parameter.detach())
        return result

    logger.log = _log_with_actor_update
    logger._codex_actor_update_log = True
    print(
        f"[INFO] Actor update diagnostics enabled for {sum(p.numel() for _, p in tracked)} "
        "policy-mean parameters; optimizer registration verified.",
        flush=True,
    )


class _HighSpeedRegressionStop(RuntimeError):
    """Internal clean-stop signal emitted after saving a regressed run."""


def _enable_highspeed_regression_guard(
    runner,
    env_cfg,
    checkpoint_path: str | None = None,
    *,
    stop_on_regression: bool = True,
) -> None:
    """Protect command tracking from sustained PPO regression.

    Unqualified ready windows are saved as diagnostic candidates, but cannot
    arm the clean-stop guard. Otherwise an actor which has never passed the
    current gate can be stopped for regressing from an arbitrary, still-failing
    window instead of being allowed to learn.
    """

    logger = getattr(runner, "logger", None)
    log_fn = getattr(logger, "log", None)
    if logger is None or not callable(log_fn) or logger.log_dir is None:
        raise RuntimeError("High-speed regression guard requires an active training logger")
    # Historical checkpoints used several incompatible gate definitions. In
    # particular, older metadata stored abs(mean(v)-mean(v_cmd)), which can
    # cancel an acceleration overshoot against an earlier underspeed period.
    # Never arm a current-run stop decision from checkpoint/dataset metadata;
    # wait until the complete live gate below qualifies under the current env.
    _ = checkpoint_path

    state = {
        "level": None,
        "best_linear_error": float("inf"),
        "best_yaw_error": float("inf"),
        "saved_score": float("inf"),
        "reference_linear_error": float("inf"),
        "reference_yaw_error": float("inf"),
        "acquired": False,
        "has_saved_candidate": False,
        "seen_ready_window": False,
        "bad_iterations": 0,
        "regression_window": deque(maxlen=25),
    }
    patience = 25
    # A 0.15 m/s allowance was as large as the high-speed gate itself and let
    # a certified 1.75 m/s actor drift to the 1.4-1.6 m/s basin before stopping.
    linear_regression_margin_mps = 0.08
    yaw_regression_margin_rad_s = 0.03

    def _log_with_regression_guard(*args, **kwargs):
        result = log_fn(*args, **kwargs)
        unwrapped = getattr(runner.env, "unwrapped", runner.env)
        level = int(getattr(unwrapped, "_command_level", 0))
        iteration = kwargs.get("it")
        if level < int(env_cfg.command_high_speed_gate_start_level) or iteration is None:
            return result
        if state["level"] != level:
            state.update(
                level=level,
                best_linear_error=float("inf"),
                best_yaw_error=float("inf"),
                saved_score=float("inf"),
                reference_linear_error=float("inf"),
                reference_yaw_error=float("inf"),
                acquired=False,
                has_saved_candidate=False,
                seen_ready_window=False,
                bad_iterations=0,
                regression_window=deque(maxlen=25),
            )
        stage_elapsed = int(unwrapped._global_step_counter - unwrapped._command_level_start_step)
        # ``--start_command_at_target`` skips the command ramp. Guard such a
        # direct-entry stage from its first ready window instead of waiting
        # through a nonexistent transition and allowing early collapse.
        required_transition_steps = (
            0
            if bool(getattr(env_cfg, "command_start_at_target", False))
            else int(unwrapped._current_command_transition_steps())
        )
        if stage_elapsed < required_transition_steps:
            return result
        ready_fraction = float(unwrapped._command_ready_fraction)
        ready_window_available = ready_fraction >= float(env_cfg.command_min_ready_env_fraction)
        if ready_window_available:
            state["seen_ready_window"] = True
        if not state["seen_ready_window"]:
            return result
        # Before a baseline exists, wait for a valid ready window. Once a
        # checkpoint baseline has been seeded (or a best window was saved), a
        # falling ready fraction is itself evidence of regression and must not
        # disable the guard.
        if not ready_window_available and not state["acquired"]:
            return result
        linear_error = float(unwrapped._command_linear_error_ema)
        yaw_error = float(unwrapped._command_yaw_error_ema)
        completed_linear_error = float(unwrapped._command_completed_linear_error_mean)
        completed_yaw_error = float(unwrapped._command_completed_yaw_error_mean)
        completed_samples = int(unwrapped._command_completed_error_sample_count)
        survival_fraction = float(unwrapped._command_survival_fraction_ema)
        survival_samples = int(unwrapped._command_survival_sample_count)
        stable_fraction = float(unwrapped._command_stable_fraction_ema)
        if not math.isfinite(linear_error) or not math.isfinite(yaw_error):
            return result
        linear_limit = float(env_cfg.command_stage_advance_linear_error_mps[level])
        yaw_limit = float(env_cfg.command_advance_yaw_error_rad_s)
        active_turn_gate = unwrapped._active_turn_gate_status()
        retention_gate = unwrapped._rehearsal_gate_status()
        retention_enforced = bool(retention_gate["enforced"])
        active_turn_linear_error = float(active_turn_gate["max_linear_error"])
        active_turn_yaw_error = float(active_turn_gate["max_yaw_error"])
        retention_linear_error = float(retention_gate["max_linear_error"])
        retention_yaw_error = float(retention_gate["max_yaw_error"])
        worst_linear_error = max(linear_error, active_turn_linear_error)
        worst_yaw_error = max(yaw_error, active_turn_yaw_error)
        if retention_enforced:
            worst_linear_error = max(worst_linear_error, retention_linear_error)
            worst_yaw_error = max(worst_yaw_error, retention_yaw_error)
        if not math.isfinite(worst_linear_error) or not math.isfinite(worst_yaw_error):
            return result
        state["best_linear_error"] = min(
            state["best_linear_error"], worst_linear_error
        )
        state["best_yaw_error"] = min(state["best_yaw_error"], worst_yaw_error)
        tracking_score = worst_linear_error / max(
            linear_limit, 1.0e-6
        ) + worst_yaw_error / max(
            yaw_limit, 1.0e-6
        )
        gate_mask = ~unwrapped._rehearsal_mask & ~unwrapped._reset_speed_bootstrap_mask
        gate_sample_target = max(int(gate_mask.sum().item()), 1)
        completed_metrics_ready = (
            completed_samples >= gate_sample_target
            and survival_samples >= gate_sample_target
        )
        damping_anneal_complete = (
            bool(unwrapped._wheel_damping_anneal_gate_complete())
            if hasattr(unwrapped, "_wheel_damping_anneal_gate_complete")
            else True
        )
        qualified = (
            ready_window_available
            and damping_anneal_complete
            and linear_error <= linear_limit
            and yaw_error <= yaw_limit
            and completed_metrics_ready
            and completed_linear_error <= linear_limit
            and completed_yaw_error <= yaw_limit
            and float(unwrapped._command_safe_fraction_ema)
            >= float(env_cfg.command_min_posture_env_fraction)
            and float(unwrapped._command_wheel_contact_fraction_ema)
            >= float(env_cfg.command_min_wheel_contact_fraction)
            and survival_fraction >= float(env_cfg.command_min_survival_env_fraction)
            and stable_fraction >= float(env_cfg.command_min_stable_env_fraction)
            and bool(active_turn_gate["ready"])
            and bool(active_turn_gate["linear"])
            and bool(active_turn_gate["yaw"])
            and bool(active_turn_gate["stable"])
            and bool(active_turn_gate["survival"])
            and bool(active_turn_gate["bank"])
            and (
                not retention_enforced
                or (
                    bool(retention_gate["ready"])
                    and bool(retention_gate["linear"])
                    and bool(retention_gate["yaw"])
                    and bool(retention_gate["stable"])
                    and bool(retention_gate["survival"])
                )
            )
        )
        # Dataset quality limits are sample filters, not measured policy error.
        # Establish the anti-regression baseline from a comparable live ready
        # window, then refresh it whenever tracking genuinely improves. The
        # strict completed-episode gate remains authoritative for advancement.
        baseline_eligible = (
            ready_window_available
            and damping_anneal_complete
            and completed_metrics_ready
            and float(unwrapped._command_safe_fraction_ema)
            >= float(env_cfg.command_min_posture_env_fraction)
            and float(unwrapped._command_wheel_contact_fraction_ema)
            >= float(env_cfg.command_min_wheel_contact_fraction)
            and bool(active_turn_gate["ready"])
            and bool(active_turn_gate["stable"])
            and (
                not retention_enforced
                or (
                    bool(retention_gate["ready"])
                    and bool(retention_gate["stable"])
                )
            )
        )
        improved_tracking = (
            not state["has_saved_candidate"]
            or tracking_score <= state["saved_score"] - 0.02
        )
        if baseline_eligible and improved_tracking:
            state["saved_score"] = tracking_score
            state["has_saved_candidate"] = True
            best_path = os.path.join(logger.log_dir, f"model_level_{level}_best_candidate.pt")
            runner.save(
                best_path,
                infos={
                    "command_level": level,
                    "iteration": int(iteration),
                    "linear_error_ema_mps": linear_error,
                    "yaw_error_ema_rad_s": yaw_error,
                    "completed_linear_error_mps": completed_linear_error,
                    "completed_yaw_error_rad_s": completed_yaw_error,
                    "worst_linear_error_mps": worst_linear_error,
                    "worst_yaw_error_rad_s": worst_yaw_error,
                    "active_turn_gate_ready": bool(active_turn_gate["ready"]),
                    "active_turn_gate_stable": bool(active_turn_gate["stable"]),
                    "active_turn_gate_survival": bool(active_turn_gate["survival"]),
                    "active_turn_gate_bank": bool(active_turn_gate["bank"]),
                    "active_turn_max_bank_error_deg": float(active_turn_gate["max_bank_error_deg"]),
                    "active_turn_max_linear_error_mps": active_turn_linear_error,
                    "active_turn_max_yaw_error_rad_s": active_turn_yaw_error,
                    "retention_gate_enforced": retention_enforced,
                    "retention_gate_ready": bool(retention_gate["ready"]),
                    "retention_gate_stable": bool(retention_gate["stable"]),
                    "retention_gate_survival": bool(retention_gate["survival"]),
                    "retention_max_linear_error_mps": retention_linear_error,
                    "retention_max_yaw_error_rad_s": retention_yaw_error,
                    "survival_fraction": survival_fraction,
                    "stable_fraction": stable_fraction,
                    "tracking_score": tracking_score,
                    "qualified_gate": bool(qualified),
                },
            )
            print(
                f"[RMUC_REGRESSION_GUARD] new unqualified level {level} candidate: "
                f"worst_linear={worst_linear_error:.3f} m/s "
                f"worst_yaw={worst_yaw_error:.3f} rad/s "
                f"qualified={qualified}",
                flush=True,
            )
        if qualified:
            qualified_score = tracking_score
            reference_score = state["reference_linear_error"] / max(
                linear_limit, 1.0e-6
            ) + state["reference_yaw_error"] / max(yaw_limit, 1.0e-6)
            if not state["acquired"] or qualified_score < reference_score:
                state["reference_linear_error"] = worst_linear_error
                state["reference_yaw_error"] = worst_yaw_error
                state["acquired"] = True
                state["regression_window"].clear()
                state["bad_iterations"] = 0
                qualified_path = os.path.join(
                    logger.log_dir, f"model_level_{level}_best_tracking.pt"
                )
                runner.save(
                    qualified_path,
                    infos={
                        "command_level": level,
                        "iteration": int(iteration),
                        "linear_error_ema_mps": linear_error,
                        "yaw_error_ema_rad_s": yaw_error,
                        "completed_linear_error_mps": completed_linear_error,
                        "completed_yaw_error_rad_s": completed_yaw_error,
                        "worst_linear_error_mps": worst_linear_error,
                        "worst_yaw_error_rad_s": worst_yaw_error,
                        "active_turn_max_linear_error_mps": active_turn_linear_error,
                        "active_turn_max_yaw_error_rad_s": active_turn_yaw_error,
                        "retention_gate_enforced": retention_enforced,
                        "retention_max_linear_error_mps": retention_linear_error,
                        "retention_max_yaw_error_rad_s": retention_yaw_error,
                        "survival_fraction": survival_fraction,
                        "stable_fraction": stable_fraction,
                        "tracking_score": qualified_score,
                        "qualified_gate": True,
                    },
                )
                print(
                    f"[RMUC_REGRESSION_GUARD] armed at qualified level {level}: "
                    f"worst_linear={worst_linear_error:.3f} m/s "
                    f"worst_yaw={worst_yaw_error:.3f} rad/s; "
                    f"saved={qualified_path}",
                    flush=True,
                )
        linear_regressed = (
            worst_linear_error
            > state["reference_linear_error"] + linear_regression_margin_mps
        )
        yaw_regressed = (
            worst_yaw_error
            > state["reference_yaw_error"] + yaw_regression_margin_rad_s
        )
        if state["acquired"]:
            state["regression_window"].append(bool(linear_regressed or yaw_regressed))
            # High-speed tracking can oscillate above/below the threshold. A
            # consecutive counter misses that pattern and lets the run finish
            # with a degraded final policy. Stop when most of the recent
            # window is regressed, while retaining the best checkpoint.
            state["bad_iterations"] = sum(state["regression_window"])
        else:
            state["regression_window"].clear()
            state["bad_iterations"] = 0
        if (
            len(state["regression_window"]) >= patience
            and state["bad_iterations"] >= int(0.60 * patience)
            and sum(list(state["regression_window"])[-5:]) >= 4
        ):
            stop_path = os.path.join(
                logger.log_dir, f"model_level_{level}_regression_stop_{iteration}.pt"
            )
            runner.save(
                stop_path,
                infos={
                    "command_level": level,
                    "iteration": int(iteration),
                    "reference_linear_error_ema_mps": state["reference_linear_error"],
                    "current_linear_error_ema_mps": worst_linear_error,
                    "reference_yaw_error_ema_rad_s": state["reference_yaw_error"],
                    "current_yaw_error_ema_rad_s": worst_yaw_error,
                    "aggregate_linear_error_ema_mps": linear_error,
                    "aggregate_yaw_error_ema_rad_s": yaw_error,
                    "active_turn_max_linear_error_mps": active_turn_linear_error,
                    "active_turn_max_yaw_error_rad_s": active_turn_yaw_error,
                    "retention_gate_enforced": retention_enforced,
                    "retention_max_linear_error_mps": retention_linear_error,
                    "retention_max_yaw_error_rad_s": retention_yaw_error,
                },
            )
            regression_message = (
                f"level {level} tracking regressed: linear "
                f"{state['reference_linear_error']:.3f}->{worst_linear_error:.3f} m/s, yaw "
                f"{state['reference_yaw_error']:.3f}->{worst_yaw_error:.3f} rad/s for "
                f"{patience} iterations; best checkpoint retained in {logger.log_dir}"
            )
            if stop_on_regression:
                raise _HighSpeedRegressionStop(regression_message)
            print(f"[RMUC_REGRESSION_WARNING] {regression_message}", flush=True)
            state["regression_window"].clear()
            state["bad_iterations"] = 0
        return result

    logger.log = _log_with_regression_guard
    print(
        f"[INFO] Command tracking regression guard enabled: linear_margin="
        f"{linear_regression_margin_mps:.2f} m/s, yaw_margin="
        f"{yaw_regression_margin_rad_s:.2f} rad/s, patience={patience} iterations.",
        flush=True,
    )


def _attach_checkpoint_metadata(env_cfg, agent_cfg, checkpoint_path: str | None) -> None:
    """Record CLI checkpoint provenance in dumped training YAML files."""

    if checkpoint_path is None:
        return

    checkpoint_path = os.path.abspath(checkpoint_path)
    checkpoint_metadata = {
        "training_bank_contract": getattr(env_cfg, "command_bank_contract", "legacy_unspecified"),
        "training_require_analytic_bank": getattr(env_cfg, "command_require_analytic_bank", False),
        "training_bank_reward_contract": getattr(env_cfg, "command_bank_reward_contract", "legacy_hard_cap"),
        "training_policy_outer_feedback_enabled": getattr(env_cfg, "policy_outer_feedback_enabled", True),
        "training_position_authority_enabled": getattr(env_cfg, "position_authority_enabled", False),
        "training_tgs_external_forces_every_iteration": getattr(env_cfg.sim.physics, "enable_external_forces_every_iteration", False),
        "training_low_speed_turn_yaw_tracking_enabled": getattr(env_cfg, "command_low_speed_turn_yaw_tracking_enabled", False),
        "path": checkpoint_path,
        "dir": os.path.dirname(checkpoint_path),
        "file": os.path.basename(checkpoint_path),
        "resume_training": bool(args_cli.resume_training),
        "reset_optimizer": bool(args_cli.reset_optimizer),
        "freeze_action_std": bool(args_cli.freeze_action_std),
        "freeze_leg_action_std": bool(args_cli.freeze_leg_action_std),
        "action_std_override": args_cli.action_std_override,
        "min_wheel_action_std": args_cli.min_wheel_action_std,
        "start_command_at_target": bool(args_cli.start_command_at_target),
        "stop_at_command_level": args_cli.stop_at_command_level,
        "teacher_speed_checkpoint": args_cli.teacher_speed_checkpoint,
        "teacher_turn_checkpoint": args_cli.teacher_turn_checkpoint,
        "teacher_distill_weight": float(args_cli.teacher_distill_weight),
        "teacher_distill_final_weight": args_cli.teacher_distill_final_weight,
        "teacher_distill_anneal_iterations": int(args_cli.teacher_distill_anneal_iterations),
        "teacher_distill_learning_rate": args_cli.teacher_distill_learning_rate,
        "teacher_distill_updates": int(args_cli.teacher_distill_updates),
        "teacher_distill_only_iterations": int(
            args_cli.teacher_distill_only_iterations
        ),
        "teacher_speed_anchor_weight": float(args_cli.teacher_speed_anchor_weight),
        "teacher_turn_common_mode_weight": float(
            args_cli.teacher_turn_common_mode_weight
        ),
        "teacher_turn_leg_weight": float(args_cli.teacher_turn_leg_weight),
        "teacher_speed_dataset": args_cli.teacher_speed_dataset,
        "teacher_speed_replay_only": bool(args_cli.teacher_speed_replay_only),
        "teacher_turn_replay_only": bool(args_cli.teacher_turn_replay_only),
        "allow_online_highspeed_teacher": bool(args_cli.allow_online_highspeed_teacher),
        "teacher_turn_dataset": args_cli.teacher_turn_dataset,
        "allow_unanchored_highspeed_distillation": bool(
            args_cli.allow_unanchored_highspeed_distillation
        ),
        "teacher_replay_batch_size": int(args_cli.teacher_replay_batch_size),
        "teacher_gradient_projection": bool(args_cli.teacher_gradient_projection),
        "turn_residual_adapter": bool(args_cli.turn_residual_adapter),
        "turn_residual_adapter_hidden_dim": int(
            args_cli.turn_residual_adapter_hidden_dim
        ),
        "turn_residual_bank": bool(args_cli.turn_residual_bank),
        "turn_directional_residual_bank": bool(
            args_cli.turn_directional_residual_bank
        ),
        "actuator_bridge_adapter": bool(args_cli.actuator_bridge_adapter),
        "freeze_turn_residual_bank_during_bridge": bool(
            args_cli.freeze_turn_residual_bank_during_bridge
        ),
        "freeze_actuator_bridge_adapter": bool(
            args_cli.freeze_actuator_bridge_adapter
        ),
        "turn_expert_checkpoint": args_cli.turn_expert_checkpoint,
        "speed_expert_checkpoint": args_cli.speed_expert_checkpoint,
        "highspeed_reset_bootstrap": bool(args_cli.highspeed_reset_bootstrap),
        "highspeed_regression_guard": bool(args_cli.highspeed_regression_guard),
        "highspeed_regression_guard_warn_only": bool(
            args_cli.highspeed_regression_guard_warn_only
        ),
        "wheel_noise_hold_steps": int(args_cli.wheel_noise_hold_steps),
        "wheel_damping_compensation_ratio": args_cli.wheel_damping_compensation_ratio,
        "wheel_damping_override": args_cli.wheel_damping_override,
        "wheel_damping_anneal_target": args_cli.wheel_damping_anneal_target,
        "wheel_damping_anneal_steps": args_cli.wheel_damping_anneal_steps,
        "wheel_torque_scale_override_nm": args_cli.wheel_torque_scale_override_nm,
        "progressive_flat_domain_randomization": not bool(
            args_cli.disable_progressive_flat_dr
        ),
        "rehearsal_feedback_parity": bool(args_cli.rehearsal_feedback_parity),
        "rehearsal_straight_feedback_only": bool(args_cli.rehearsal_straight_feedback_only),
        "low_speed_yaw_feedback_clip": args_cli.low_speed_yaw_feedback_clip,
        "command_consistent_dr_noise": bool(args_cli.command_consistent_dr_noise),
        "domain_residual_learning_rate": args_cli.domain_residual_learning_rate,
        "stage_residual_learning_rate": args_cli.stage_residual_learning_rate,
        "protect_nominal_straight_anchor": args_cli.protect_nominal_straight_anchor,
        "residual_anchor_l2": args_cli.residual_anchor_l2,
        "residual_anchor_checkpoint": args_cli.residual_anchor_checkpoint,
        "train_local_authority": args_cli.train_local_authority,
        "train_physics_adaptation": args_cli.train_physics_adaptation,
        "physics_train_modules": args_cli.physics_train_modules,
        "local_authority_train_indices": args_cli.local_authority_train_indices,
        "critic_learning_rate": args_cli.critic_learning_rate,
        "speed_isolated_domain_bank": bool(args_cli.speed_isolated_domain_bank),
        "feedback_contract_by_command": bool(args_cli.feedback_contract_by_command),
        "allow_fresh_critic_highspeed": bool(args_cli.allow_fresh_critic_highspeed),
        "critic_warmup_iterations": int(args_cli.critic_warmup_iterations),
        "freeze_actor_until_command_level": args_cli.freeze_actor_until_command_level,
        "load_critic_for_finetune": bool(args_cli.load_critic_for_finetune),
        "zero_observation_indices_for_finetune": args_cli.zero_observation_indices_for_finetune,
        "num_steps_per_env_override": args_cli.num_steps_per_env_override,
        "gamma_override": args_cli.gamma_override,
        "lam_override": args_cli.lam_override,
    }
    setattr(env_cfg, "launch_checkpoint", checkpoint_metadata)
    setattr(agent_cfg, "launch_checkpoint", checkpoint_metadata)


def _enforce_curriculum_gate(task_name: str | None, checkpoint_path: str | None) -> None:
    """Refuse Stage 1/2 training unless the previous stage passed its gate."""
    task_name = task_name or ""
    required_previous_stage = None
    if "Command" in task_name:
        required_previous_stage = "stand"
    elif "Rough" in task_name:
        required_previous_stage = "command"
    if required_previous_stage is None:
        return
    if args_cli.allow_curriculum_override:
        print(f"[WARNING] Curriculum gate override enabled for previous stage: {required_previous_stage}")
        return
    if checkpoint_path is None:
        raise RuntimeError(
            f"Curriculum gate: {task_name} requires a passing {required_previous_stage} checkpoint; "
            "start it with --checkpoint <path> --resume_training."
        )
    checkpoint = Path(checkpoint_path).resolve()
    gate_path = checkpoint.with_name("curriculum_gate.json")
    if not gate_path.is_file():
        raise RuntimeError(
            f"Curriculum gate: missing {gate_path}. Evaluate the previous-stage checkpoint before advancing."
        )
    try:
        gate_result = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Curriculum gate: cannot read {gate_path}: {exc}") from exc
    gate_passed = bool(gate_result.get("passed", False))
    gate_stage = gate_result.get("stage")
    recorded_checkpoint = gate_result.get("checkpoint")
    if recorded_checkpoint and os.path.realpath(str(recorded_checkpoint)) != os.path.realpath(
        str(checkpoint)
    ):
        raise RuntimeError(
            "Curriculum gate sidecar belongs to a different checkpoint: "
            f"requested={checkpoint}, recorded={recorded_checkpoint}"
        )
    # A command checkpoint which has already passed the command gate is a
    # valid continuation baseline for another command level. Requiring a
    # separate stand-only gate here rejected legitimate staged command
    # checkpoints because their sidecar records the stronger command result.
    same_stage_command_checkpoint = (
        required_previous_stage == "stand"
        and gate_stage == "command"
    )
    if not gate_passed or (
        gate_stage != required_previous_stage and not same_stage_command_checkpoint
    ):
        raise RuntimeError(
            f"Curriculum gate: checkpoint {checkpoint} has not passed required stage {required_previous_stage}; "
            f"gate={gate_result}"
        )
    if same_stage_command_checkpoint:
        schema_version = int(gate_result.get("gate_schema_version", 1))
        if schema_version < 2:
            print(
                "[WARNING] Historical command gate sidecar accepted only as checkpoint "
                "provenance (legacy schema; speed averaging could cancel over/underspeed). "
                "The current level must pass the live rolling-error, survival and retention gates.",
                flush=True,
            )
        else:
            print(
                "[INFO] Command checkpoint provenance accepted from gate schema "
                f"v{schema_version}; the current level will still be revalidated live.",
                flush=True,
            )
    else:
        print(f"[INFO] Curriculum gate passed for previous stage: {required_previous_stage}")


def _prepare_stock_rsl_cfg(agent_cfg) -> dict:
    """Normalize Isaac Lab 2.x policy configs to the RSL-RL 5.x model schema."""
    cfg = agent_cfg.to_dict()
    policy = cfg.get("policy") or {}
    # Isaac Lab 5/6's runner constructs actor/critic MLPModel instances.  The
    # legacy project configs populate the older policy ActorCritic block; map
    # only the schema here and leave the environment/asset untouched.
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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    _enforce_curriculum_gate(args_cli.task, args_cli.checkpoint)
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    recovery_flags = any(getattr(args_cli, name) is not None for name in (
        'recovery_curriculum_start_tilt', 'recovery_curriculum_max_tilt',
        'recovery_curriculum_success_episodes', 'recovery_slew_start_rate',
        'recovery_slew_end_rate', 'recovery_slew_anneal_steps',
        'recovery_slope_degrees', 'recovery_slope_azimuth_degrees')) or args_cli.recovery_curriculum
    if recovery_flags:
        if 'Recovery' not in (args_cli.task or ''):
            raise ValueError('Recovery curriculum/slew/slope flags require the RMUC Recovery task')
        if args_cli.recovery_curriculum:
            env_cfg.recovery_curriculum_enabled = True
        stages = tuple(int(v) for v in env_cfg.recovery_curriculum_tilt_stages)
        if args_cli.recovery_curriculum_start_tilt is not None:
            if args_cli.recovery_curriculum_start_tilt not in stages:
                raise ValueError('Recovery curriculum start tilt is not in the configured stage list')
            env_cfg.recovery_curriculum_start_stage = stages.index(args_cli.recovery_curriculum_start_tilt)
        if args_cli.recovery_curriculum_max_tilt is not None:
            if args_cli.recovery_curriculum_max_tilt not in stages:
                raise ValueError('Recovery curriculum max tilt is not in the configured stage list')
            env_cfg.recovery_curriculum_max_stage = stages.index(args_cli.recovery_curriculum_max_tilt)
        if env_cfg.recovery_curriculum_start_stage > env_cfg.recovery_curriculum_max_stage:
            raise ValueError('Recovery curriculum start tilt cannot exceed max tilt')
        if args_cli.recovery_curriculum_success_episodes is not None:
            if args_cli.recovery_curriculum_success_episodes < 1:
                raise ValueError('Recovery curriculum success episodes must be positive')
            env_cfg.recovery_curriculum_success_episodes = args_cli.recovery_curriculum_success_episodes
        for arg_name, cfg_name in (
            ('recovery_slew_start_rate', 'recovery_slew_start_rad_s'),
            ('recovery_slew_end_rate', 'recovery_slew_end_rad_s'),
            ('recovery_slew_anneal_steps', 'recovery_slew_anneal_steps'),
            ('recovery_slope_degrees', 'recovery_slope_degrees'),
            ('recovery_slope_azimuth_degrees', 'recovery_slope_azimuth_degrees')):
            value = getattr(args_cli, arg_name)
            if value is not None:
                setattr(env_cfg, cfg_name, value)
        print('[INFO] Recovery course overrides:', {
            'curriculum_enabled': getattr(env_cfg, 'recovery_curriculum_enabled', False),
            'start_tilt': stages[env_cfg.recovery_curriculum_start_stage],
            'max_tilt': stages[env_cfg.recovery_curriculum_max_stage],
            'success_episodes': getattr(env_cfg, 'recovery_curriculum_success_episodes', None),
            'slew_start': getattr(env_cfg, 'recovery_slew_start_rad_s', None),
            'slew_end': getattr(env_cfg, 'recovery_slew_end_rad_s', None),
            'slew_steps': getattr(env_cfg, 'recovery_slew_anneal_steps', None),
            'slope_degrees': getattr(env_cfg, 'recovery_slope_degrees', None),
            'slope_azimuth_degrees': getattr(env_cfg, 'recovery_slope_azimuth_degrees', None),
        }, flush=True)
    if args_cli.learning_rate_override is not None:
        if args_cli.learning_rate_override <= 0.0:
            raise ValueError("--learning_rate_override must be positive")
        agent_cfg.algorithm.learning_rate = float(args_cli.learning_rate_override)
        print(
            f"[INFO] Stage learning-rate override: {agent_cfg.algorithm.learning_rate:.6g}",
            flush=True,
        )
    if args_cli.num_learning_epochs_override is not None:
        if args_cli.num_learning_epochs_override < 1:
            raise ValueError("--num_learning_epochs_override must be >= 1")
        agent_cfg.algorithm.num_learning_epochs = int(args_cli.num_learning_epochs_override)
        print(
            f"[INFO] Stage PPO epoch override: {agent_cfg.algorithm.num_learning_epochs}",
            flush=True,
        )
    if args_cli.gamma_override is not None:
        if not 0.0 < args_cli.gamma_override <= 1.0:
            raise ValueError("--gamma_override must be in (0, 1]")
        agent_cfg.algorithm.gamma = float(args_cli.gamma_override)
        print(f"[INFO] Stage PPO gamma override: {agent_cfg.algorithm.gamma}", flush=True)
    if args_cli.lam_override is not None:
        if not 0.0 < args_cli.lam_override <= 1.0:
            raise ValueError("--lam_override must be in (0, 1]")
        agent_cfg.algorithm.lam = float(args_cli.lam_override)
        print(f"[INFO] Stage PPO lambda override: {agent_cfg.algorithm.lam}", flush=True)
    if args_cli.num_steps_per_env_override is not None:
        if args_cli.num_steps_per_env_override < 1:
            raise ValueError("--num_steps_per_env_override must be >= 1")
        agent_cfg.num_steps_per_env = int(args_cli.num_steps_per_env_override)
        print(
            f"[INFO] Stage rollout-length override: {agent_cfg.num_steps_per_env} steps/env",
            flush=True,
        )
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # YAML validates its default env count, but CLI can request a larger grid.
    # Keep every RMUC flat environment on ground for the full travel radius.
    if ("RMUC" in (args_cli.task or "")
            and not getattr(env_cfg, "rough_terrain_enabled", False)):
        generator = getattr(env_cfg.terrain, "terrain_generator", None)
        if generator is not None:
            count = int(env_cfg.scene.num_envs)
            rows = max(int(math.sqrt(count)), 1)
            cols = math.ceil(count / rows)
            half_grid = .5 * max(rows - 1, cols - 1) * env_cfg.scene.env_spacing
            required_size = 2.0 * (half_grid + env_cfg.max_xy_distance_m + 5.0)
            if min(generator.size) < required_size:
                size = math.ceil(required_size / 32.0) * 32.0
                generator.size = (size, size)
                print(f"[INFO] Expanded flat terrain to {size} m for {count} environments.", flush=True)
    if args_cli.command_level is not None:
        stage_names = getattr(env_cfg, "command_stage_names", ())
        if "Command" not in (args_cli.task or "") or not stage_names:
            raise ValueError("--command_level is only supported by staged RMUC command tasks")
        if not 0 <= args_cli.command_level < len(stage_names):
            raise ValueError(
                f"--command_level must be in [0, {len(stage_names) - 1}], got {args_cli.command_level}"
            )
        env_cfg.command_start_level = int(args_cli.command_level)
        print(
            f"[INFO] Starting RMUC command curriculum at level={args_cli.command_level} "
            f"name={stage_names[args_cli.command_level]}",
            flush=True,
        )
    if args_cli.stop_at_command_level is not None:
        stage_names = getattr(env_cfg, "command_stage_names", ())
        if "Command" not in (args_cli.task or "") or args_cli.command_level is None:
            raise ValueError("--stop_at_command_level requires an RMUC Command task and --command_level")
        if not args_cli.command_level < args_cli.stop_at_command_level < len(stage_names):
            raise ValueError(
                "--stop_at_command_level must be above --command_level and within "
                f"[0, {len(stage_names) - 1}], got {args_cli.stop_at_command_level}"
            )
        print(
            f"[INFO] Training will stop after entering RMUC command level "
            f"{args_cli.stop_at_command_level} name={stage_names[args_cli.stop_at_command_level]}",
            flush=True,
        )
    if args_cli.freeze_actor_until_command_level is not None:
        stage_names = getattr(env_cfg, "command_stage_names", ())
        if "Command" not in (args_cli.task or "") or args_cli.command_level is None:
            raise ValueError(
                "--freeze_actor_until_command_level requires an RMUC Command task and --command_level"
            )
        if not (
            args_cli.command_level
            < args_cli.freeze_actor_until_command_level
            < len(stage_names)
        ):
            raise ValueError(
                "--freeze_actor_until_command_level must be above --command_level and within "
                f"[0, {len(stage_names) - 1}], got {args_cli.freeze_actor_until_command_level}"
            )
        if args_cli.critic_warmup_iterations:
            raise ValueError(
                "Use either --freeze_actor_until_command_level or --critic_warmup_iterations, not both"
            )
    if args_cli.start_command_at_target:
        if "Command" not in (args_cli.task or "") or args_cli.command_level is None:
            raise ValueError("--start_command_at_target requires an RMUC Command task and --command_level")
        env_cfg.command_start_at_target = True
        print(
            f"[INFO] Starting RMUC command level {env_cfg.command_start_level} at its final target",
            flush=True,
        )
    if args_cli.rehearsal_feedback_parity:
        env_cfg.rehearsal_feedback_parity = True
        print("[INFO] Stage-local active/rehearsal feedback parity enabled.", flush=True)
    if args_cli.feedback_contract_by_command:
        env_cfg.feedback_contract_by_command = True
        print("[INFO] Command-based feedback contract: active/rehearsal/inference share control.", flush=True)
    if args_cli.rehearsal_straight_feedback_only:
        if not args_cli.rehearsal_feedback_parity:
            raise ValueError("--rehearsal_straight_feedback_only requires --rehearsal_feedback_parity")
        env_cfg.rehearsal_feedback_straight_only = True
    if args_cli.command_consistent_dr_noise:
        env_cfg.domain_randomization_command_consistent_noise = True
    if args_cli.low_speed_yaw_feedback_clip is not None:
        if not 0.0 <= args_cli.low_speed_yaw_feedback_clip <= 1.0:
            raise ValueError("--low_speed_yaw_feedback_clip must be in [0, 1]")
        env_cfg.wheel_yaw_feedback_low_speed_action_clip = args_cli.low_speed_yaw_feedback_clip
        print(f"[INFO] Low-speed yaw feedback clip={args_cli.low_speed_yaw_feedback_clip}", flush=True)
    if args_cli.disable_progressive_flat_dr:
        env_cfg.domain_randomization_enabled = False
        print(
            "[INFO] RMUC progressive flat domain randomization disabled for this ablation.",
            flush=True,
        )
    elif getattr(env_cfg, "domain_randomization_enabled", False):
        print(
            "[INFO] RMUC progressive flat domain randomization enabled: "
            f"levels {env_cfg.domain_randomization_start_level}->"
            f"{env_cfg.domain_randomization_full_level}; terrain remains flat.",
            flush=True,
        )
    if args_cli.highspeed_reset_bootstrap:
        if "Command" not in (args_cli.task or "") or args_cli.command_level is None:
            raise ValueError(
                "--highspeed_reset_bootstrap requires an RMUC Command task and --command_level"
            )
        env_cfg.command_reset_speed_bootstrap_enabled = True
        print(
            f"[INFO] High-speed staged reset enabled from level "
            f"{env_cfg.command_reset_speed_bootstrap_start_level}; assisted episodes are gate-excluded.",
            flush=True,
        )
    if args_cli.wheel_damping_compensation_ratio is not None:
        damping_ratio = float(args_cli.wheel_damping_compensation_ratio)
        if not 0.0 <= damping_ratio <= 1.0:
            raise ValueError("--wheel_damping_compensation_ratio must be in [0, 1]")
        env_cfg.wheel_damping_compensation_ratio = damping_ratio
        print(
            "[INFO] RMUC wheel damping compensation override="
            f"{damping_ratio:.3f} (simulation-only; authored USD unchanged)",
            flush=True,
        )
    if args_cli.wheel_damping_override is not None:
        damping = float(args_cli.wheel_damping_override)
        if damping < 0.0:
            raise ValueError("--wheel_damping_override must be non-negative")
        env_cfg.robot_cfg.actuators["wheel_motors"].damping = damping
        print(
            f"[INFO] RMUC explicit wheel damping override={damping:.4f}",
            flush=True,
        )
    if args_cli.wheel_damping_anneal_target is not None:
        target = float(args_cli.wheel_damping_anneal_target)
        if target < 0.0:
            raise ValueError("--wheel_damping_anneal_target must be non-negative")
        if (
            args_cli.wheel_damping_anneal_steps is not None
            and args_cli.wheel_damping_anneal_steps < 1
        ):
            raise ValueError("--wheel_damping_anneal_steps must be >= 1")
        env_cfg.wheel_damping_anneal_target = target
        if args_cli.wheel_damping_anneal_steps is not None:
            env_cfg.wheel_damping_anneal_steps = int(args_cli.wheel_damping_anneal_steps)
        print(
            "[INFO] RMUC wheel damping anneal target="
            f"{target:.4f}, steps={env_cfg.wheel_damping_anneal_steps} "
            "(simulation-only; authored USD unchanged)",
            flush=True,
        )
    elif args_cli.wheel_damping_anneal_steps is not None:
        raise ValueError(
            "--wheel_damping_anneal_steps requires --wheel_damping_anneal_target"
        )
    if args_cli.wheel_torque_scale_override_nm is not None:
        torque_scale = float(args_cli.wheel_torque_scale_override_nm)
        if torque_scale <= 0.0:
            raise ValueError("--wheel_torque_scale_override_nm must be positive")
        env_cfg.wheel_torque_scale_nm = torque_scale
        env_cfg.robot_cfg.actuators["wheel_motors"].effort_limit = torque_scale
        print(
            "[INFO] RMUC wheel torque scale/effort-limit override="
            f"{torque_scale:.3f} Nm/action",
            flush=True,
        )
    if (
        args_cli.highspeed_regression_guard_warn_only
        and not args_cli.highspeed_regression_guard
    ):
        raise ValueError(
            "--highspeed_regression_guard_warn_only requires --highspeed_regression_guard"
        )
    from rmuc_training_contract import highspeed_guard_applies
    if (
        hasattr(env_cfg, 'command_stage_linear_x_mps')
        and highspeed_guard_applies(env_cfg)
        and not args_cli.allow_aggressive_highspeed
    ):
        if (
            args_cli.teacher_speed_checkpoint
            and args_cli.teacher_speed_dataset
            and not args_cli.teacher_speed_replay_only
            and not args_cli.allow_online_highspeed_teacher
        ):
            raise RuntimeError(
                "High-speed RMUC distillation with a speed replay dataset must use "
                "--teacher_speed_replay_only. The full legacy speed teacher may contain "
                "long-horizon acceleration/fall cycles and can reintroduce the underspeed "
                "or regression failure. Use --allow_online_highspeed_teacher only for an "
                "explicit ablation."
            )
        stage_transition_steps = int(
            env_cfg.command_stage_transition_steps[env_cfg.command_start_level]
        )
        transition_iterations = max(
            int(
                math.ceil(
                    float(stage_transition_steps)
                    / max(
                        int(env_cfg.scene.num_envs) * int(agent_cfg.num_steps_per_env),
                        1,
                    )
                )
            ),
            1,
        )
        max_warmup_iterations = max(int(math.floor(0.10 * transition_iterations)), 1)
        if (
            not args_cli.start_command_at_target
            and args_cli.critic_warmup_iterations > max_warmup_iterations
        ):
            raise RuntimeError(
                "High-speed anti-regression guard rejected critic warmup "
                f"{args_cli.critic_warmup_iterations}: transition lasts about "
                f"{transition_iterations} iterations, so actor warmup must be <= "
                f"{max_warmup_iterations}. A longer freeze makes the actor miss "
                "the staged speed transition."
            )
        if (
            args_cli.learning_rate_override is not None
            and args_cli.learning_rate_override > 2.0e-5
        ) or (
            args_cli.num_learning_epochs_override is not None
            and args_cli.num_learning_epochs_override > 1
        ):
            raise RuntimeError(
                "High-speed anti-regression guard rejected the requested PPO update size. "
                "Use the validated 2e-5/1-epoch settings, or explicitly acknowledge "
                "the risk with --allow_aggressive_highspeed."
            )
        if args_cli.action_std_override:
            std_values = [
                float(value.strip())
                for value in args_cli.action_std_override.split(",")
                if value.strip()
            ]
            if len(std_values) == 6:
                max_wheel_std = max(std_values[index] for index in (2, 5))
                bounded_wheel_exploration = (
                    args_cli.freeze_leg_action_std
                    and not args_cli.freeze_action_std
                    and args_cli.min_wheel_action_std is not None
                    and 0.012 < max_wheel_std <= 0.05
                    and 0.0 < float(args_cli.min_wheel_action_std) <= max_wheel_std
                )
                if max_wheel_std > 0.012 and not bounded_wheel_exploration:
                    raise RuntimeError(
                        "High-speed anti-regression guard rejected unbounded wheel action "
                        "std > 0.012. For controlled speed acquisition, freeze the four "
                        "leg std entries and provide --min_wheel_action_std with wheel std "
                        "<= 0.05; otherwise explicitly use --allow_aggressive_highspeed."
                    )
    if (
        args_cli.checkpoint is not None
        and getattr(env_cfg, "command_start_level", 0) >= 5
        and not args_cli.resume_training
        and not args_cli.load_critic_for_finetune
        and not args_cli.allow_fresh_critic_highspeed
    ):
        raise RuntimeError(
            "High-speed RMUC finetuning requires the same-stage critic. Use "
            "--load_critic_for_finetune (recommended), --resume_training, or "
            "explicitly acknowledge an incompatible reward with "
            "--allow_fresh_critic_highspeed."
        )
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    if args_cli.local_authority_train_indices is not None and not args_cli.train_local_authority:
        raise ValueError('local_authority_train_indices requires train_local_authority')
    if args_cli.physics_train_modules is not None and (not args_cli.train_physics_adaptation or not args_cli.reset_optimizer):
        raise ValueError('Explicit physics selection requires train_physics_adaptation and reset_optimizer')
    if args_cli.train_physics_adaptation and (not args_cli.train_position_authority or args_cli.train_local_authority):
        raise ValueError('Physics adaptation requires position authority and excludes local expert training')
    if args_cli.train_local_authority and not args_cli.train_position_authority:
        raise ValueError('train_local_authority requires train_position_authority consumer and isolation path')
    if args_cli.train_position_authority:
        if args_cli.isolated_state_router or not args_cli.checkpoint or not args_cli.resume_training or not args_cli.freeze_action_std:
            raise ValueError('Authority training needs explicit checkpoint/resume/frozen std and no router optimizer')
        if env_cfg.policy_outer_feedback_enabled:
            raise ValueError('Authority training requires E2E without outer feedback')
        if args_cli.teacher_speed_checkpoint or args_cli.teacher_turn_checkpoint or args_cli.teacher_turn_dataset:
            raise ValueError('Teacher hooks not supported in authority training')
        env_cfg.position_authority_enabled = True
        from rmuc_position_authority import checkpoint_authority_speed_plateau_end
        authority_source_state = torch.load(args_cli.checkpoint, map_location='cpu', weights_only=False)['actor_state_dict']
        isolated_physics_selection = (args_cli.train_physics_adaptation
                                      and bool(args_cli.physics_train_modules)
                                      and args_cli.reset_optimizer)
        isolated_local_selection = (args_cli.train_local_authority
                                    and bool(args_cli.local_authority_train_indices)
                                    and args_cli.reset_optimizer)
        if 'rmuc_state_router.turn_speed_plateau_end' in authority_source_state and not (args_cli.router_speed_direction or isolated_physics_selection or isolated_local_selection):
            raise ValueError('Router support migration is diagnostic-only until isolated extension training is implemented')
        env_cfg.position_authority_speed_plateau_end = checkpoint_authority_speed_plateau_end(authority_source_state)
        if env_cfg.position_authority_speed_plateau_end > 1.0 and not (args_cli.router_speed_direction or isolated_physics_selection or isolated_local_selection):
            raise ValueError('Expanded authority checkpoint is diagnostic-only until isolated extension training is implemented')
        agent_cfg.clip_actions = None
    if args_cli.router_speed_direction:
        if (not args_cli.train_position_authority or not args_cli.reset_optimizer
                or args_cli.train_physics_adaptation or args_cli.train_local_authority):
            raise ValueError('Router speed extension needs authority consumer, reset optimizer, and exclusive selection')
        if not any(k.startswith('rmuc_state_router.speed_extensions.') for k in authority_source_state):
            raise ValueError('Router speed copies missing from source checkpoint')
    if args_cli.rmuc_tgs_external_forces_every_iteration:
        if not hasattr(env_cfg, 'policy_outer_feedback_enabled'):
            raise ValueError('RMUC physics migration flag requires an RMUC environment')
        if env_cfg.sim.physics is None:
            from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
            env_cfg.sim.physics = PhysxCfg()
        env_cfg.sim.physics.enable_external_forces_every_iteration = True
    required_commands = None
    if args_cli.require_command_coverage:
        if not args_cli.train_position_authority or args_cli.critic_warmup_iterations < 3:
            raise ValueError('Coverage guard requires authority training and at least3 critic warmup updates')
        from rmuc_training_contract import required_training_commands
        required_commands = required_training_commands(env_cfg)
        print(f'[RMUC_REQUIRED_COMMANDS] {required_commands}', flush=True)
    if args_cli.train_position_authority:
        from rmuc_training_contract import observation_contract, checkpoint_observation_contract
        current_observation_contract = observation_contract(env_cfg)
        source_observation_contract = checkpoint_observation_contract(
            torch.load(args_cli.checkpoint, map_location='cpu', weights_only=False))
        if current_observation_contract != source_observation_contract:
            if not args_cli.allow_observation_contract_migration or not args_cli.reset_optimizer:
                raise ValueError('Observation contract differs from checkpoint; explicit migration and reset_optimizer required')
        print(f'[RMUC_OBSERVATION_CONTRACT] {current_observation_contract}', flush=True)
        from rmuc_training_contract import validate_trainable_target_gate
        gate_payload = torch.load(args_cli.checkpoint, map_location='cpu', weights_only=False)
        from rmuc_training_contract import validate_checkpoint_leg_scale_contract
        action_mapping_contract = validate_checkpoint_leg_scale_contract(gate_payload, env_cfg)
        print(f'[RMUC_ACTION_MAPPING_CONTRACT] {action_mapping_contract}', flush=True)
        gate_level = env_cfg.command_start_level
        target_gate = validate_trainable_target_gate(
            gate_payload['actor_state_dict'],
            env_cfg.command_stage_linear_x_mps[gate_level],
            env_cfg.command_stage_yaw_rate_rad_s[gate_level],
            physics_only=args_cli.train_physics_adaptation,
            local_indices=args_cli.local_authority_train_indices,
            physics_modules=args_cli.physics_train_modules,
            router_speed_direction=args_cli.router_speed_direction)
        from rmuc_training_contract import validate_trainable_sequence_gates
        sequence_gates = validate_trainable_sequence_gates(
            gate_payload['actor_state_dict'], getattr(env_cfg, 'command_training_sequence', ()),
            physics_only=args_cli.train_physics_adaptation,
            local_indices=args_cli.local_authority_train_indices,
            physics_modules=args_cli.physics_train_modules,
            router_speed_direction=args_cli.router_speed_direction)
        if sequence_gates:
            print(f'[RMUC_TRAINING_SEQUENCE_GATES] {sequence_gates}', flush=True)
        random_range = getattr(env_cfg, 'command_random_speed_range', ())
        if random_range:
            random_gates = [validate_trainable_target_gate(
                gate_payload['actor_state_dict'], float(v), 0.,
                physics_only=args_cli.train_physics_adaptation,
                local_indices=args_cli.local_authority_train_indices,
                physics_modules=args_cli.physics_train_modules,
                router_speed_direction=args_cli.router_speed_direction)
                for v in torch.linspace(*random_range, 65)]
            print(f'[RMUC_RANDOM_SPEED_GATES] {random_gates}', flush=True)
        del gate_payload
        print(f'[RMUC_TRAINABLE_TARGET_GATE] {target_gate}', flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
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
    runner = runner_class(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
    _enable_terminal_iteration_progress(runner, args_cli.stop_at_command_level)
    adapter_hidden_dim = (
        checkpoint_turn_adapter_hidden_dim(args_cli.checkpoint)
        if args_cli.checkpoint is not None
        else None
    )
    delta_bank_in_checkpoint = (
        checkpoint_has_turn_delta_bank(args_cli.checkpoint)
        if args_cli.checkpoint is not None
        else False
    )
    enable_turn_delta_bank = bool(
        args_cli.turn_residual_bank
        or args_cli.turn_directional_residual_bank
        or delta_bank_in_checkpoint
    )
    directional_bank_in_checkpoint = (
        checkpoint_has_directional_turn_delta_bank(args_cli.checkpoint)
        if args_cli.checkpoint is not None
        else False
    )
    enable_directional_turn_bank = bool(
        args_cli.turn_directional_residual_bank
        or directional_bank_in_checkpoint
    )
    actuator_bridge_in_checkpoint = (
        checkpoint_has_actuator_bridge_adapter(args_cli.checkpoint)
        if args_cli.checkpoint is not None
        else False
    )
    speed_expert_in_checkpoint = (
        checkpoint_has_speed_expert(args_cli.checkpoint)
        if args_cli.checkpoint is not None
        else False
    )
    domain_bank_in_checkpoint = (
        checkpoint_has_domain_residual_bank(args_cli.checkpoint)
        if args_cli.checkpoint is not None
        else False
    )
    if args_cli.turn_residual_adapter or enable_turn_delta_bank or adapter_hidden_dim is not None:
        if args_cli.turn_residual_adapter_hidden_dim < 1:
            raise ValueError("--turn_residual_adapter_hidden_dim must be >= 1")
        actor = runner.alg.get_policy()
        adapter = install_rmuc_turn_adapter(
            actor,
            hidden_dim=(
                adapter_hidden_dim
                if adapter_hidden_dim is not None
                else int(args_cli.turn_residual_adapter_hidden_dim)
            ),
            freeze_base=True,
            optimizer=runner.alg.optimizer,
            turn_expert_checkpoint=args_cli.turn_expert_checkpoint,
            enable_delta_bank=enable_turn_delta_bank,
            enable_directional_bank=enable_directional_turn_bank,
            bank_size=(checkpoint_turn_delta_bank_size(args_cli.checkpoint)
                       if args_cli.checkpoint and args_cli.resume_training and not args_cli.reset_optimizer
                       else None),
        )
        print(
            "[INFO] RMUC turn residual adapter enabled: "
            f"hidden_dim={adapter.net[0].out_features}, base_actor_frozen=True, "
            "straight_gate=0, residual_gate_vx>2.0; "
            f"delta_bank={enable_turn_delta_bank}; "
            f"directional_bank={enable_directional_turn_bank}; "
            f"turn_expert={args_cli.turn_expert_checkpoint or 'checkpoint-embedded'}.",
            flush=True,
        )
    if args_cli.actuator_bridge_adapter or actuator_bridge_in_checkpoint:
        actor = runner.alg.get_policy()
        bridge_adapter = install_rmuc_actuator_bridge_adapter(
            actor,
            hidden_dim=int(args_cli.turn_residual_adapter_hidden_dim),
            optimizer=runner.alg.optimizer,
        )
        print(
            "[INFO] RMUC actuator bridge residual enabled: "
            f"hidden_dim={bridge_adapter.net[0].out_features}, "
            "all_command_support=True.",
            flush=True,
        )
        if args_cli.freeze_actuator_bridge_adapter:
            for parameter in bridge_adapter.parameters():
                parameter.requires_grad_(False)
            print(
                "[INFO] RMUC actuator bridge residual frozen as a certified skill.",
                flush=True,
            )
    elif args_cli.freeze_actuator_bridge_adapter:
        raise RuntimeError(
            "--freeze_actuator_bridge_adapter requires an installed or checkpoint-embedded adapter"
        )
    enable_domain_bank = bool(
        args_cli.speed_isolated_domain_bank or domain_bank_in_checkpoint
        or (
            getattr(env_cfg, "domain_randomization_enabled", False)
            and getattr(env_cfg, "command_start_level", 0)
            >= getattr(env_cfg, "domain_randomization_start_level", 29)
        )
    )
    if enable_domain_bank:
        domain_bank = install_rmuc_domain_residual_bank(
            runner.alg.get_policy(), optimizer=runner.alg.optimizer,
            speed_isolated=(args_cli.speed_isolated_domain_bank or
                            (args_cli.checkpoint is not None and
                             checkpoint_has_speed_isolated_domain_bank(args_cli.checkpoint))),
        )
        print(
            "[INFO] RMUC bounded domain residual bank enabled: "
            f"heads={','.join(domain_bank.HEAD_NAMES)}, max_delta=0.04 action.",
            flush=True,
        )
    if args_cli.speed_expert_checkpoint or speed_expert_in_checkpoint:
        actor = runner.alg.get_policy()
        install_rmuc_speed_expert(
            actor,
            checkpoint_path=(
                args_cli.speed_expert_checkpoint
                if not speed_expert_in_checkpoint
                else None
            ),
            optimizer=runner.alg.optimizer,
        )
        print(
            "[INFO] RMUC frozen 2.9 m/s straight expert enabled: "
            f"source={args_cli.speed_expert_checkpoint or 'checkpoint-embedded'}.",
            flush=True,
        )
    if args_cli.freeze_turn_residual_bank_during_bridge:
        actor = runner.alg.get_policy()
        bank = getattr(actor, "rmuc_turn_delta_adapters", None)
        if bank is None:
            raise RuntimeError(
                "--freeze_turn_residual_bank_during_bridge requires the turn residual bank"
            )
        for parameter in bank.parameters():
            parameter.requires_grad_(False)
        print(
            "[INFO] RMUC turn residual bank fully frozen during actuator bridge.",
            flush=True,
        )
    # write git state to logs
    if hasattr(runner, "add_git_repo_to_log"):
        runner.add_git_repo_to_log(__file__)
    critic_rate = args_cli.critic_learning_rate
    if args_cli.state_router_train_branches is not None and not args_cli.isolated_state_router:
        raise ValueError('state_router_train_branches requires isolated_state_router')
    if critic_rate is None and args_cli.resume_training and args_cli.checkpoint:
        source_optimizer = torch.load(args_cli.checkpoint, weights_only=False, map_location="cpu").get("optimizer_state_dict", {})
        for group in source_optimizer.get("param_groups", []):
            if group.get("rmuc_group") == "critic":
                critic_rate = float(group["lr"])
                args_cli.critic_learning_rate = critic_rate
    if critic_rate is not None:
        if not 0.0 < critic_rate <= 1e-3 or runner.alg.schedule != "fixed":
            raise ValueError("Critic LR must be in (0,1e-3] with a fixed PPO schedule")
        critic_parameters = list(runner.alg.critic.parameters())
        critic_ids = {id(p) for p in critic_parameters}
        for group in runner.alg.optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in critic_ids]
        runner.alg.optimizer.add_param_group({"params": critic_parameters, "lr": critic_rate, "rmuc_group": "critic"})
    if args_cli.isolated_state_router:
        if not args_cli.checkpoint or not args_cli.resume_training or not args_cli.freeze_action_std:
            raise ValueError('Router training requires checkpoint, resume_training and freeze_action_std')
        if args_cli.stage_residual_learning_rate is not None or args_cli.domain_residual_learning_rate is not None:
            raise ValueError('Router mode freezes stage/domain banks; remove their LR overrides')
        if args_cli.teacher_speed_checkpoint or args_cli.teacher_turn_checkpoint or args_cli.teacher_turn_dataset:
            raise ValueError('Teacher hooks are not validated with isolated router')
        from rmuc_state_router_training import prepare_router_optimizer
        router_payload = torch.load(args_cli.checkpoint, map_location='cpu', weights_only=False)
        if args_cli.state_router_train_branches is not None and not args_cli.reset_optimizer:
            groups = router_payload.get('optimizer_state_dict', {}).get('param_groups', [])
            saved_branches = next((group.get('rmuc_branches') for group in groups
                                   if group.get('rmuc_group') == 'state_router'), None)
            if saved_branches != args_cli.state_router_train_branches:
                raise ValueError('Selected router branches lack matching saved optimizer provenance; use reset_optimizer')
        prepare_router_optimizer(runner, router_payload['actor_state_dict'],
                                 args_cli.state_router_learning_rate, critic_rate or 1e-4,
                                 args_cli.state_router_train_branches)
        del router_payload
    if args_cli.train_position_authority:
        from rmuc_position_authority_training import prepare_authority_optimizer
        authority_payload = torch.load(args_cli.checkpoint, map_location='cpu', weights_only=False)
        if not args_cli.reset_optimizer:
            groups = authority_payload.get('optimizer_state_dict', {}).get('param_groups', [])
            expected_group = ('physics_adaptation' if args_cli.train_physics_adaptation else
                              ('local_authority' if args_cli.train_local_authority else 'position_authority'))
            if [g.get('rmuc_group') for g in groups] != [expected_group, 'critic']:
                raise ValueError('Authority optimizer provenance missing; reset optimizer required')
            if args_cli.train_local_authority:
                count = int(authority_payload['actor_state_dict']['rmuc_position_authority.local_correction.centers'].numel())
                expected_indices = args_cli.local_authority_train_indices or list(range(count))
                if groups[0].get('rmuc_local_indices', list(range(count))) != expected_indices:
                    raise ValueError('Local expert selection differs from optimizer provenance; reset optimizer required')
        prepare_authority_optimizer(runner, authority_payload['actor_state_dict'],
                                    args_cli.state_router_learning_rate, critic_rate or 1e-4,
                                    local_only=args_cli.train_local_authority,
                                    local_indices=args_cli.local_authority_train_indices,
                                    physics_only=args_cli.train_physics_adaptation,
                                    physics_modules=args_cli.physics_train_modules,
                                    router_speed_direction=args_cli.router_speed_direction)
        del authority_payload
    # load the checkpoint
    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if args_cli.resume_training:
            # True continuation: restore optimizer/iteration and continue from the next iteration index.
            load_signature = inspect.signature(runner.load)
            load_parameters = load_signature.parameters
            load_kwargs = {}
            if args_cli.reset_optimizer:
                if "load_cfg" in load_parameters:
                    load_kwargs["load_cfg"] = {
                        "actor": True,
                        "critic": True,
                        "optimizer": False,
                        "iteration": True,
                        "rnd": True,
                    }
                elif "load_optimizer" in load_parameters:
                    load_kwargs["load_optimizer"] = False
                else:
                    raise TypeError("runner.load does not support resetting optimizer state")
            elif "load_optimizer" in load_parameters:
                load_kwargs["load_optimizer"] = True
            if "load_iteration" in load_parameters:
                load_kwargs["load_iteration"] = True
            if "map_location" in load_parameters:
                load_kwargs["map_location"] = runner.device
            runner.load(resume_path, **load_kwargs)
            runner.current_learning_iteration += 1
            print(
                f"[INFO]: Resuming training from next iteration: {runner.current_learning_iteration}"
            )
            if args_cli.reset_optimizer:
                print("[INFO]: Restored actor/critic/iteration with a fresh optimizer.", flush=True)
        else:
            # Weight initialization / finetune-from-checkpoint workflow.
            zero_observation_indices = tuple(
                int(value.strip())
                for value in (args_cli.zero_observation_indices_for_finetune or "").split(",")
                if value.strip()
            )
            _load_policy_for_finetune(
                runner,
                resume_path,
                preserve_checkpoint_std=args_cli.preserve_checkpoint_std,
                load_critic=args_cli.load_critic_for_finetune,
                zero_observation_indices=zero_observation_indices,
            )

    if critic_rate is not None:
        for group in runner.alg.optimizer.param_groups:
            if group.get("rmuc_group") == "critic":
                group["lr"] = critic_rate
        print(f"[INFO] Critic learning rate={critic_rate}; policy base LR={runner.alg.learning_rate}.", flush=True)
    if args_cli.stage_residual_learning_rate is not None:
        rate = float(args_cli.stage_residual_learning_rate)
        if not 0.0 < rate <= 1e-4 or runner.alg.schedule != "fixed":
            raise ValueError("Stage residual LR must be in (0,1e-4] with fixed PPO schedule")
        bank = getattr(runner.alg.get_policy(), "rmuc_turn_delta_adapters", None)
        if bank is None:
            raise RuntimeError("Stage residual LR requires the stage residual bank")
        bank_ids = {id(p) for p in bank.parameters()}
        changed = 0
        for group in runner.alg.optimizer.param_groups:
            ids = {id(p) for p in group["params"]}
            if ids & bank_ids:
                if not ids <= bank_ids:
                    raise RuntimeError("Stage residual bank must have an isolated optimizer group")
                group["lr"] = rate
                changed += 1
        if changed != 1:
            raise RuntimeError(f"Expected one stage residual optimizer group, got {changed}")
        print(f"[INFO] Stage residual LR={rate}; only the current stage's routed heads are trainable.", flush=True)
    if args_cli.domain_residual_learning_rate is not None:
        rate = float(args_cli.domain_residual_learning_rate)
        if not 0.0 < rate <= 1e-3 or runner.alg.schedule != "fixed":
            raise ValueError("Domain residual LR must be in (0,1e-3] with fixed PPO schedule")
        bank = getattr(runner.alg.get_policy(), "rmuc_domain_residual_bank", None)
        if bank is None:
            raise RuntimeError("--domain_residual_learning_rate requires a domain residual bank")
        bank_ids = {id(p) for p in bank.parameters()}
        changed_groups = 0
        for group in runner.alg.optimizer.param_groups:
            group_ids = {id(p) for p in group["params"]}
            if group_ids & bank_ids:
                if not group_ids <= bank_ids:
                    raise RuntimeError("Domain bank must have its own optimizer parameter group")
                group["lr"] = rate
                changed_groups += 1
        if changed_groups != 1:
            raise RuntimeError(f"Expected one domain optimizer group, got {changed_groups}")
        print(f"[INFO] Bounded domain residual learning rate={rate}; base/experts remain frozen.", flush=True)
    _configure_action_std(
        runner,
        freeze=args_cli.freeze_action_std,
        freeze_leg=args_cli.freeze_leg_action_std,
        override=args_cli.action_std_override,
        min_wheel_std=args_cli.min_wheel_action_std,
        command_level=getattr(env_cfg, "command_start_level", None),
    )
    if (
        args_cli.teacher_speed_checkpoint
        or args_cli.teacher_turn_checkpoint
        or args_cli.teacher_turn_dataset
    ):
        if "Command" not in (args_cli.task or ""):
            raise ValueError("Teacher distillation is only supported by the staged RMUC Command task")
        highspeed_level = int(getattr(env_cfg, "command_start_level", 0)) >= int(
            getattr(env_cfg, "command_high_speed_gate_start_level", 9)
        )
        if (
            highspeed_level
            and args_cli.teacher_speed_checkpoint
            and not args_cli.teacher_speed_dataset
            and not args_cli.allow_unanchored_highspeed_distillation
        ):
            raise RuntimeError(
                "High-speed teacher distillation requires --teacher_speed_dataset to "
                "anchor the acquired straight/high-speed skill. The previous run "
                "omitted this replay and regressed from ~1.7 m/s to sub-1 m/s. "
                "Use --allow_unanchored_highspeed_distillation only for an explicit ablation."
            )
        _configure_teacher_distillation(
            runner,
            speed_checkpoint=args_cli.teacher_speed_checkpoint,
            turn_checkpoint=args_cli.teacher_turn_checkpoint,
            weight=args_cli.teacher_distill_weight,
            final_weight=args_cli.teacher_distill_final_weight,
            anneal_iterations=args_cli.teacher_distill_anneal_iterations,
            learning_rate=args_cli.teacher_distill_learning_rate,
            updates=args_cli.teacher_distill_updates,
            distill_only_iterations=args_cli.teacher_distill_only_iterations,
            speed_anchor_weight=args_cli.teacher_speed_anchor_weight,
            turn_common_mode_weight=args_cli.teacher_turn_common_mode_weight,
            turn_leg_weight=args_cli.teacher_turn_leg_weight,
            speed_replay_only=args_cli.teacher_speed_replay_only,
            turn_replay_only=args_cli.teacher_turn_replay_only,
            speed_dataset_path=args_cli.teacher_speed_dataset,
            turn_dataset_path=args_cli.teacher_turn_dataset,
            replay_batch_size=args_cli.teacher_replay_batch_size,
            gradient_projection=args_cli.teacher_gradient_projection,
        )
    if args_cli.isolated_state_router:
        from rmuc_state_router_training import configure_router_training
        configure_router_training(runner, args_cli.state_router_balance_coef)
    if args_cli.train_position_authority:
        from rmuc_position_authority_training import configure_authority_audit
        configure_authority_audit(runner, required_commands=required_commands)
        from rmuc_training_contract import configure_checkpoint_observation_contract
        configure_checkpoint_observation_contract(runner, env_cfg)
    if not args_cli.freeze_turn_residual_bank_during_bridge and not args_cli.isolated_state_router and not args_cli.train_position_authority:
        _configure_turn_residual_bank_schedule(runner)
    _configure_temporally_correlated_wheel_noise(runner, args_cli.wheel_noise_hold_steps)
    if args_cli.protect_nominal_straight_anchor:
        from rmuc_anchor_guard import configure_nominal_straight_guard
        configure_nominal_straight_guard(runner)
    _configure_critic_warmup(runner, args_cli.critic_warmup_iterations)
    _configure_actor_level_freeze(runner, args_cli.freeze_actor_until_command_level)
    _configure_env_action_credit(runner)
    _configure_domain_action_diagnostics(runner)
    if args_cli.residual_anchor_l2 is not None:
        from rmuc_residual_safety import configure_residual_safety
        configure_residual_safety(runner, args_cli.residual_anchor_l2, args_cli.residual_anchor_checkpoint)
    _enable_actor_update_diagnostics(runner)
    if args_cli.highspeed_regression_guard:
        if getattr(env_cfg, "command_start_level", 0) < getattr(
            env_cfg, "command_high_speed_gate_start_level", 9
        ):
            raise ValueError(
                "--highspeed_regression_guard requires a high-speed RMUC command level"
            )
        _enable_highspeed_regression_guard(
            runner,
            env_cfg,
            args_cli.checkpoint,
            stop_on_regression=not args_cli.highspeed_regression_guard_warn_only,
        )

    from rmuc_recovery_checkpoint import configure_recovery_slew_checkpoint
    configure_recovery_slew_checkpoint(runner, env, args_cli.checkpoint, args_cli.resume_training)
    if args_cli.recovery_anchor_checkpoint is not None or args_cli.recovery_anchor_coefficient != 0.:
        if args_cli.recovery_anchor_checkpoint is None:
            raise ValueError('Recovery anchor coefficient requires an explicit reference checkpoint')
        from rmuc_recovery_anchor import configure_recovery_anchor
        configure_recovery_anchor(runner, env, args_cli.recovery_anchor_checkpoint,
                                  args_cli.recovery_anchor_coefficient)
    if args_cli.recovery_replay_manifest is not None or args_cli.recovery_replay_coefficient != 0.:
        if args_cli.recovery_replay_manifest is None:
            raise ValueError('Recovery replay requires a verified reference manifest')
        from rmuc_recovery_replay import configure_recovery_replay
        configure_recovery_replay(runner,args_cli.recovery_replay_manifest,
                                  args_cli.recovery_replay_coefficient)
    _notify_env_training_progress(env, int(getattr(runner, "current_learning_iteration", 0)))
    _attach_checkpoint_metadata(env_cfg, agent_cfg, args_cli.checkpoint)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    # Stagger environment phases for every training run. Synchronized 1499-step
    # episodes reset all 128 environments every ~23 PPO iterations and fill a
    # complete rollout with acceleration transients, producing periodic zero-vx
    # batches and biased actor updates. The first partial episodes are excluded
    # from the command survival gate below.
    randomize_initial_episode_length = True
    print(
        f"[INFO]: Randomize initial episode length: {randomize_initial_episode_length}",
        flush=True,
    )
    # Stock RSL-RL reads observations immediately and does not reset here.
    # Ensure custom closed-loop root/link poses are applied before rollout.
    env.reset()
    unwrapped_env = getattr(env, "unwrapped", env)
    if args_cli.train_position_authority and getattr(unwrapped_env, '_separated_wheel_passive', False):
        u = unwrapped_env
        readback = dict(
            wheel_torque_contract=u.cfg.wheel_torque_contract,
            feedback_mode=u.cfg.wheel_feedback_damping_mode,
            wheel_actuator_damping=u._wheel_actuator.damping.detach().cpu().tolist(),
            wheel_actuator_effort_limit=u._wheel_actuator.effort_limit.detach().cpu().tolist(),
            mechanical_b=u._wheel_mechanical_viscous_per_env.detach().cpu().tolist(),
            coulomb=u._wheel_coulomb_per_env.detach().cpu().tolist(),
            feedback_D=u._wheel_feedback_damping_per_env.detach().cpu().tolist(),
            physics_dt=u.cfg.sim.dt, policy_dt=u.cfg.sim.dt*u.cfg.decimation,
            leg_pd_interval_steps=u._leg_pd_sample_hold.interval_steps,
            outer_feedback=u.cfg.policy_outer_feedback_enabled,
            spring_torque_nm=u.cfg.gas_spring_assist_torque_nm,
            radial_force_n=u.cfg.gas_spring_radial_force_n,
            firmware_spring_enabled=u.cfg.firmware_spring_enabled,
            command_training_sequence=getattr(u.cfg, 'command_training_sequence', ()),
            command_random_speed_range=getattr(u.cfg, 'command_random_speed_range', ()),
            command_random_hold_s=getattr(u.cfg, 'command_random_hold_s', ()),
            command_random_ramp_s=getattr(u.cfg, 'command_random_ramp_s', ()),
            command_random_step_fraction=getattr(u.cfg, 'command_random_step_fraction', 0),
            command_random_stop_probability=u._random_speed_sequence.stop_probability if u._random_speed_sequence is not None else None,
            command_random_max_delta_mps=getattr(u.cfg, 'command_random_max_delta_mps', 0),
            command_random_isolated_rng=getattr(u.cfg, 'command_random_isolated_rng', False),
            command_training_sequence_policy_steps=u._training_sequence_table,
            continuous_cohort_count=int(u._continuous_command_mask.sum().item()),
            command_leg_scale_profile=u.cfg.command_leg_scale_profile,
            actual_leg_scale_knots={'speed_abs_mps':u._manual_leg_speed_knots.detach().cpu().tolist(), 'scale_rad':u._manual_leg_scale_knots.detach().cpu().tolist()},
            optimizer_parameter_groups=[dict(
                group=g.get('rmuc_group', 'unnamed'),
                learning_rate=float(g['lr']),
                parameter_count=sum(v.numel() for v in g['params']),
                physics_modules=g.get('rmuc_physics_modules'))
                for g in runner.alg.optimizer.param_groups],
            scope='Post-reset actual actuator tensors and sampled wheel parameters; not behavior certification')
        if getattr(u, '_arena_ground_randomizer', None) is not None:
            readback['arena_ground_curriculum'] = u._arena_ground_randomizer.readback()
            readback['domain_friction_scale'] = u._domain_friction_scale.detach().cpu().tolist()
        if getattr(u.cfg, 'base_com_randomization_enabled', False):
            readback['base_com_randomization'] = u._base_com_randomizer.readback()
        # Keep pure waiting, sampled data age and actuator response distinct.
        # This post-reset snapshot does not certify measured end-to-end latency.
        readback['policy_sampling_std'] = [float(v) for name, parameter in runner.alg.get_policy().distribution.named_parameters() if name in {'std_param', 'log_std_param'} for v in (torch.exp(parameter) if name == 'log_std_param' else parameter).detach().cpu().flatten()]
        readback['leg_transport'] = u._leg_transport.readback()
        readback['gravity_direction_dr'] = u._gravity_direction_dr.readback()
        readback['timing_readback'] = dict(
            scope='Post-reset configuration and buffers; not hardware timing certification',
            physics_dt_s=float(u.cfg.sim.dt),
            policy_period_s=float(u.step_dt),
            leg_pd_period_s=float(u.cfg.sim.dt*u._leg_pd_sample_hold.interval_steps),
            wheel_compute_period_s=float(u.cfg.sim.dt),
            legacy_fixed_observation_delay_steps=int(u._obs_delay_steps),
            sensor_observation_buffer_steps_per_env=u._sensor_delay_steps.detach().cpu().tolist(),
            domain_observation_buffer_steps_per_env=u._domain_observation_delay_steps.detach().cpu().tolist(),
            observation_buffer_semantics='Sensor model uses max(sensor_steps,fixed_steps); otherwise max(domain_steps,fixed_steps); not a sum',
            communication_buffer_steps_per_env=u._communication_delay_steps.detach().cpu().tolist(),
            sensor_model=dict(u.cfg.domain_randomization_sensor_model),
            motor_response_time_constants_s={name: response.time_constant.detach().cpu().tolist()
                for name, response in u._motor_responses.items()},
            motor_response_scope='First-order time constant, not pure command latency',
            wheel_transport=(u._real2sim_wheel.readback() if u._real2sim_wheel is not None else None),
            inference_duration_modeled=False,
            sensor_filter_latency_identified=False)
        Path(log_dir, 'physical_runtime_readback.json').write_text(json.dumps(readback, indent=2)+'\n')
        schedule = getattr(u, '_random_speed_sequence', None)
        if schedule is not None:
            torch.save(dict(schedule=schedule.snapshot(),
                continuous_cohort_mask=u._continuous_command_mask.detach().cpu().clone(),
                policy_dt=u.step_dt,
                scope='Initial post-reset schedule only; later resets resample selected environments'),
                Path(log_dir, 'initial_random_speed_schedule.pt'))
    if hasattr(unwrapped_env, "ignore_current_episodes_for_survival_gate"):
        # This must run after env.reset(), which marks all episodes as started.
        # RSL-RL randomizes their ages inside learn(); the first completion is
        # therefore only a partial episode and must not enter the survival gate.
        unwrapped_env.ignore_current_episodes_for_survival_gate()
    try:
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=randomize_initial_episode_length,
        )
    except _CommandLevelReachedStop as exc:
        print(f"[RMUC_LEVEL_STOP] {exc}", flush=True)
        if getattr(runner.logger, "writer", None) is not None:
            runner.logger.stop_logging_writer()
    except _HighSpeedRegressionStop as exc:
        print(f"[RMUC_REGRESSION_GUARD] {exc}", flush=True)
        if getattr(runner.logger, "writer", None) is not None:
            runner.logger.stop_logging_writer()
    except KeyboardInterrupt:
        iteration = int(getattr(runner, "current_learning_iteration", 0))
        manual_path = os.path.join(log_dir, f"model_manual_stop_{iteration}.pt")
        try:
            runner.save(
                manual_path,
                infos={
                    "stop_reason": "keyboard_interrupt",
                    "iteration": iteration,
                    "command_level": int(getattr(unwrapped_env, "_command_level", -1)),
                },
            )
            print(f"[TRAIN_STOP] Manual interrupt; checkpoint saved to {manual_path}", flush=True)
        except Exception as exc:  # keep shutdown reliable even if the runner is half-closed
            print(f"[TRAIN_STOP] Manual interrupt; checkpoint save failed: {exc}", flush=True)
        if getattr(runner.logger, "writer", None) is not None:
            runner.logger.stop_logging_writer()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
