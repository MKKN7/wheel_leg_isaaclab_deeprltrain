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

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .base import StateMachineBase


class StateMachineManager:
    """Composable runtime manager for environment state machines.

    This class owns only the generic lifecycle fanout. Robot-specific machines,
    default stacks, and marker styles should live with the robot/task package.
    """

    def __init__(
        self,
        machines: list[StateMachineBase] | tuple[StateMachineBase, ...],
        *,
        marker_cfg: VisualizationMarkersCfg | None = None,
        marker_name_to_index: dict[str, int] | None = None,
        marker_height_offset: float = 1.15,
    ):
        self.machines = list(machines)
        self._marker_cfg = marker_cfg
        self._marker_name_to_index = dict(marker_name_to_index or {})
        self._marker_height_offset = float(marker_height_offset)
        self._marker: VisualizationMarkers | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.machines)

    def on_observation_step(self, env) -> None:
        for machine in self.machines:
            machine.on_observation_step(env)

    def on_command_updated(self, env) -> None:
        for machine in self.machines:
            machine.on_command_updated(env)

    def apply_command_overrides(self, env) -> None:
        for machine in self.machines:
            machine.apply_command_overrides(env)

    def get_effective_height_cmd(self, env, height_cmd: torch.Tensor) -> torch.Tensor:
        effective_height_cmd = height_cmd
        for machine in self.machines:
            effective_height_cmd = machine.get_effective_height_cmd(env, effective_height_cmd)
        return effective_height_cmd

    def get_height_reward_reference_height(
        self,
        env,
        relative_obs_height: torch.Tensor,
        wheel_height_w: torch.Tensor,
    ) -> torch.Tensor:
        reference_height = relative_obs_height
        for machine in self.machines:
            reference_height = machine.get_height_reward_reference_height(
                env, reference_height, wheel_height_w
            )
        return reference_height

    def get_height_reward_target_height(self, env, target_height: torch.Tensor) -> torch.Tensor:
        resolved_target_height = target_height
        for machine in self.machines:
            resolved_target_height = machine.get_height_reward_target_height(
                env, resolved_target_height
            )
        return resolved_target_height

    def apply_reward_term_scales(
        self, env, reward_terms: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        scaled_reward_terms = reward_terms
        for machine in self.machines:
            scaled_reward_terms = machine.apply_reward_term_scales(env, scaled_reward_terms)
        return scaled_reward_terms

    def apply_done_masks(
        self,
        env,
        terminate: torch.Tensor,
        time_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        terminate_mask = terminate
        time_out_mask = time_out
        for machine in self.machines:
            terminate_mask, time_out_mask = machine.apply_done_masks(
                env, terminate_mask, time_out_mask
            )
        return terminate_mask, time_out_mask

    def append_reset_logs(self, env, extras: dict[str, float], env_ids: torch.Tensor) -> None:
        for machine in self.machines:
            machine.append_reset_logs(env, extras, env_ids)

    def apply_visual_marker_states(
        self,
        env,
        marker_indices: torch.Tensor,
        priorities: torch.Tensor,
        state_name_to_index: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resolved_indices = marker_indices
        resolved_priorities = priorities
        for machine in self.machines:
            resolved_indices, resolved_priorities = machine.apply_visual_marker_state(
                env,
                resolved_indices,
                resolved_priorities,
                state_name_to_index,
            )
        return resolved_indices, resolved_priorities

    def on_reset(self, env, env_ids: torch.Tensor) -> None:
        for machine in self.machines:
            machine.on_reset(env, env_ids)

    def setup_visual_marker(self, env) -> None:
        """Create the optional shared play-mode marker."""
        if self._marker_cfg is None or not self.enabled:
            self._marker = None
            return

        self._marker = VisualizationMarkers(self._marker_cfg)
        self._marker.set_visibility(True)

    def update_visual_marker(self, env) -> None:
        """Show the highest-priority state-machine marker above each robot."""
        if self._marker is None or not self._marker_name_to_index:
            return

        positions = env.robot.data.root_pos_w.clone()
        positions[:, 2] += self._marker_height_offset
        marker_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        priorities = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        marker_indices, _ = self.apply_visual_marker_states(
            env,
            marker_indices,
            priorities,
            self._marker_name_to_index,
        )
        self._marker.set_visibility(True)
        self._marker.visualize(
            translations=positions.detach().cpu(),
            marker_indices=marker_indices.detach().cpu(),
        )
