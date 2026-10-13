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

"""Package containing robot assets, actuators and terrain generators for the wheelbipe platform."""

import os

# Parent Path
RootPath = os.path.dirname(os.path.abspath(__file__))
AssetPath = os.path.join(RootPath, "assets")

ROBOT_UTILS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
