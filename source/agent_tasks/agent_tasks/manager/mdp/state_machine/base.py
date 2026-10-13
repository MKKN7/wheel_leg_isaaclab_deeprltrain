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

import torch


class StateMachineBase:
    """Shared hook surface for runtime state machines."""

    name: str = "base"

    def on_observation_step(self, env) -> None:
        """Update cached state using the latest observations/sensors."""

    def on_command_updated(self, env) -> None:
        """Update state after terrain/base command processing has completed."""

    def apply_command_overrides(self, env) -> None:
        """Apply final command overrides after env-level command post-processing."""

    def get_effective_height_cmd(self, env, height_cmd: torch.Tensor) -> torch.Tensor:
        """Return the runtime height command after machine-specific holds/biases."""
        return height_cmd

    def get_height_reward_reference_height(
        self,
        env,
        relative_obs_height: torch.Tensor,
        wheel_height_w: torch.Tensor,
    ) -> torch.Tensor:
        """Return the reward-side observed height reference."""
        return relative_obs_height

    def get_height_reward_target_height(
        self, env, target_height: torch.Tensor
    ) -> torch.Tensor:
        """Return the reward-side target height."""
        return target_height

    def apply_reward_term_scales(
        self, env, reward_terms: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Scale or bias reward terms in-place and return the mapping."""
        return reward_terms

    def apply_done_masks(
        self,
        env,
        terminate: torch.Tensor,
        time_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allow machines to modify terminate/time_out masks."""
        return terminate, time_out

    def append_reset_logs(
        self, env, extras: dict[str, float], env_ids: torch.Tensor
    ) -> None:
        """Append per-reset logging scalars."""

    def apply_visual_marker_state(
        self,
        env,
        marker_indices: torch.Tensor,
        priorities: torch.Tensor,
        state_name_to_index: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update shared state-machine marker indices with machine-specific priorities."""
        return marker_indices, priorities

    def on_reset(self, env, env_ids: torch.Tensor) -> None:
        """Reset internal state for the provided environments."""
