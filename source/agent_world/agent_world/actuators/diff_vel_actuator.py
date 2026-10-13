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

"""PD actuator that estimates joint velocity from finite differences."""

from __future__ import annotations

import math

import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class DiffVelPDActuator(IdealPDActuator):
    """Ideal PD actuator using finite-difference joint velocity feedback.

    IsaacLab passes simulator joint velocity into :meth:`compute`. This actuator
    keeps the standard ``IdealPDActuator`` position/velocity PD law, but replaces
    the velocity feedback with ``(q_t - q_{t-1}) / dt``. This is useful when the
    policy or sim2real setup should see/control the same velocity estimator used
    on hardware.
    """

    cfg: "DiffVelPDActuatorCfg"

    def __init__(self, cfg: "DiffVelPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._prev_joint_pos: torch.Tensor | None = None
        self._filtered_joint_vel: torch.Tensor | None = None

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """Compute actuator efforts using finite-difference joint velocity."""
        diff_joint_vel = self._estimate_joint_vel(joint_pos, joint_vel)
        return super().compute(control_action, joint_pos, diff_joint_vel)

    def _estimate_joint_vel(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        if self._prev_joint_pos is None or self._prev_joint_pos.shape != joint_pos.shape:
            self._prev_joint_pos = joint_pos.detach().clone()
            initial_vel = joint_vel if self.cfg.use_input_vel_on_first_step else torch.zeros_like(joint_vel)
            self._filtered_joint_vel = initial_vel.detach().clone()
            return initial_vel

        delta_pos = joint_pos - self._prev_joint_pos
        if self.cfg.wrap_to_pi:
            delta_pos = torch.atan2(torch.sin(delta_pos), torch.cos(delta_pos))

        reset_mask = self._get_reset_mask(delta_pos)
        diff_joint_vel = delta_pos / max(float(self.cfg.diff_dt), 1.0e-8)
        if reset_mask is not None:
            fallback_vel = joint_vel if self.cfg.use_input_vel_on_reset_jump else torch.zeros_like(joint_vel)
            diff_joint_vel = torch.where(reset_mask, fallback_vel, diff_joint_vel)

        diff_joint_vel = self._filter_joint_vel(diff_joint_vel, reset_mask)
        self._prev_joint_pos.copy_(joint_pos.detach())
        return diff_joint_vel

    def _get_reset_mask(self, delta_pos: torch.Tensor) -> torch.Tensor | None:
        threshold = float(self.cfg.reset_position_jump_threshold)
        if not math.isfinite(threshold) or threshold <= 0.0:
            return None
        reset_env = torch.any(torch.abs(delta_pos) > threshold, dim=-1, keepdim=True)
        return reset_env.expand_as(delta_pos)

    def _filter_joint_vel(self, diff_joint_vel: torch.Tensor, reset_mask: torch.Tensor | None) -> torch.Tensor:
        alpha = float(self.cfg.velocity_filter_alpha)
        if alpha >= 1.0:
            self._filtered_joint_vel = diff_joint_vel.detach().clone()
            return diff_joint_vel
        if alpha <= 0.0:
            if self._filtered_joint_vel is None or self._filtered_joint_vel.shape != diff_joint_vel.shape:
                self._filtered_joint_vel = diff_joint_vel.detach().clone()
            return self._filtered_joint_vel

        if self._filtered_joint_vel is None or self._filtered_joint_vel.shape != diff_joint_vel.shape:
            self._filtered_joint_vel = diff_joint_vel.detach().clone()
            return diff_joint_vel

        filtered = alpha * diff_joint_vel + (1.0 - alpha) * self._filtered_joint_vel
        if reset_mask is not None:
            filtered = torch.where(reset_mask, diff_joint_vel, filtered)
        self._filtered_joint_vel.copy_(filtered.detach())
        return filtered


@configclass
class DiffVelPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for :class:`DiffVelPDActuator`."""

    class_type: type = DiffVelPDActuator

    diff_dt: float = 0.005
    """Time interval used by the finite-difference velocity estimator."""

    use_input_vel_on_first_step: bool = True
    """Use simulator-provided velocity before a previous position sample exists."""

    use_input_vel_on_reset_jump: bool = True
    """Use simulator-provided velocity when a large position jump suggests reset."""

    wrap_to_pi: bool = False
    """Wrap position differences to [-pi, pi] before dividing by ``diff_dt``."""

    velocity_filter_alpha: float = 1.0
    """EMA coefficient for finite-difference velocity; 1.0 disables filtering."""

    reset_position_jump_threshold: float = 1.0
    """Position jump threshold treated as reset/teleport; <=0 disables detection."""
