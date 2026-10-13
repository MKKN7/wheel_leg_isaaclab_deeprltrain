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

"""Root package for custom manager MDP groups."""

import sys as _sys

from .isaaclab import *  # noqa: F401, F403
from .isaaclab import curriculums, events, observations, rewards, terrains

_sys.modules.setdefault(__name__ + ".curriculums", curriculums)
_sys.modules.setdefault(__name__ + ".events", events)
_sys.modules.setdefault(__name__ + ".observations", observations)
_sys.modules.setdefault(__name__ + ".rewards", rewards)
_sys.modules.setdefault(__name__ + ".terrains", terrains)
