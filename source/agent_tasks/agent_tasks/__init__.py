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

"""Package containing task implementations for various robotic environments."""

import os

# 导入子模块，触发 gymnasium 环境注册
import agent_tasks.direct.wheelbipe.wheelbipe_V14  # noqa: F401
# Local RMUC closed-loop task registration.  The USD and environment are
# resolved entirely from this repository.
import agent_tasks.direct.rmuc_wheel_leg  # noqa: F401

# Parent Path
RootPath = os.path.dirname(os.path.abspath(__file__))
