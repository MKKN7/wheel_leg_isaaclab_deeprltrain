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

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    # NOTE: `--resume` and `--load_run` are placeholders for compatibility with
    # external orchestration scripts. They are intentionally not wired to any
    # resume logic here; training currently uses `--checkpoint`.
    arg_group.add_argument("--resume", action="store_true", default=False, help="(Placeholder) Resume from a previous run.")
    arg_group.add_argument("--load_run", type=str, default=None, help="(Placeholder) Run id/name to load for resuming.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    arg_group.add_argument(
        "--cmoe_router_temperature",
        type=float,
        default=None,
        help=(
            "[CMoE] Override policy moe_router_temperature (e.g. 1.10). "
            "Use with Robotics-Wheelbipe-V13-Flat-MoE-CMoE-v0 sweep without new Gym ids."
        ),
    )
    arg_group.add_argument(
        "--cmoe_aux",
        type=float,
        default=None,
        help="[CMoE] Override algorithm cmoe_expert_value_loss_coef (per-expert value aux; try 0.0 .. 0.35).",
    )
    arg_group.add_argument(
        "--moe_load_balancing_coef",
        type=float,
        default=None,
        help="[MoE/CMoE] Override policy moe_load_balancing_coef.",
    )
    arg_group.add_argument(
        "--clip_actions",
        type=float,
        default=None,
        help="[RSL-RL] Override agent clip_actions before environment action scaling.",
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def _set_nested(container: object, key: str, value: object) -> None:
    try:
        container[key] = value
        return
    except Exception:
        pass
    setattr(container, key, value)


def apply_cmoe_cli_overrides(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace) -> None:
    tau = getattr(args_cli, "cmoe_router_temperature", None)
    lb = getattr(args_cli, "moe_load_balancing_coef", None)
    aux = getattr(args_cli, "cmoe_aux", None)
    policy = getattr(agent_cfg, "policy", None)
    if policy is not None:
        if tau is not None:
            _set_nested(policy, "moe_router_temperature", tau)
        if lb is not None:
            _set_nested(policy, "moe_load_balancing_coef", lb)
    algo = getattr(agent_cfg, "algorithm", None)
    if algo is not None and aux is not None:
        _set_nested(algo, "cmoe_expert_value_loss_coef", aux)


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    exp_name = getattr(args_cli, "experiment_name", None)
    if exp_name is not None and exp_name != "":
        agent_cfg.experiment_name = exp_name

    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    clip_actions = getattr(args_cli, "clip_actions", None)
    if clip_actions is not None:
        agent_cfg.clip_actions = clip_actions
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"}:
        task_suffix = getattr(args_cli, "task", "") or ""
        if args_cli.log_project_name is not None:
            agent_cfg.wandb_project = args_cli.log_project_name
            agent_cfg.neptune_project = args_cli.log_project_name
        else:
            agent_cfg.wandb_project += "." + task_suffix
            agent_cfg.neptune_project += "." + task_suffix

    apply_cmoe_cli_overrides(agent_cfg, args_cli)
    return agent_cfg
