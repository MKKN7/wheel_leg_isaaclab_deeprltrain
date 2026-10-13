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

"""Configuration for ``LearnedVelocityActuator``."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.utils import configclass

from .learned_velocity_actuator import LearnedVelocityActuator


@configclass
class LearnedVelocityActuatorCfg(IdealPDActuatorCfg):
    """Configuration for a learned velocity-controlled actuator.

    Attributes:
        model_path: Path to the exported TorchScript (``.pt``) model file.
        norm_stats_path: Path to the ``norm_stats.json`` file containing
            per-feature ``mean`` and ``std`` arrays.
        history_len: Number of history steps expected by the model (H).
        kd: Velocity damping coefficient used during training.
            This value is passed as a constant input feature to the model.
        effort_limit: Maximum absolute joint torque (Nm).  Torques predicted
            by the model are clamped to ``[-effort_limit, effort_limit]``.
        effort_smoothing: Exponential moving-average factor for effort
            (0 = no smoothing, 0 < s < 1 = smooth).  Helps suppress
            high-frequency jitter from the neural network.
    """

    class_type: type = LearnedVelocityActuator

    model_path: str = MISSING
    """Path to the TorchScript model file."""

    norm_stats_path: str = MISSING
    """Path to the normalization statistics JSON file."""

    history_len: int = 10
    """Number of history steps (H) the model was trained with."""

    kd: float = 0.175
    """Velocity damping coefficient (hardware setting used during training)."""

    effort_limit: float = 5.0
    """Maximum absolute joint torque (Nm)."""

    effort_smoothing: float = 0.0
    """EMA smoothing factor for predicted effort (0 = disabled)."""
