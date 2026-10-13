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

"""Custom actuator models for agent_world assets."""

from .m3508_actuator import M3508Actuator, M3508ActuatorCfg
from .learned_velocity_actuator import LearnedVelocityActuator
from .learned_velocity_actuator_cfg import LearnedVelocityActuatorCfg
from .diff_vel_actuator import DiffVelPDActuator, DiffVelPDActuatorCfg

__all__ = [
    "M3508Actuator",
    "M3508ActuatorCfg",
    "LearnedVelocityActuator",
    "LearnedVelocityActuatorCfg",
    "DiffVelPDActuator",
    "DiffVelPDActuatorCfg",
]
