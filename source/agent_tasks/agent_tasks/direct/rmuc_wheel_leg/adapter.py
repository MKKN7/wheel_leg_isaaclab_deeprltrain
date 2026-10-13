"""Local entry-point exports for the RMUC closed-loop wheel-leg task.

The task is intentionally self-contained in ``wheeled-legged_RL``. Keeping
these names in a small adapter preserves the existing Gymnasium entry points
without importing or depending on the former sibling project.
"""

from __future__ import annotations

from .env import RmucWheelLegEnv
from .env_cfg import (
    RmucWheelLegCommandEnvCfg,
    RmucWheelLegCourseEnvCfg,
    RmucWheelLegRmucEnvCfg,
    RmucWheelLegRoughEnvCfg,
    RmucWheelLegRoughPlayEnvCfg,
)
from .rsl_rl_ppo_cfg import RmucWheelLegPPORunnerCfg

__all__ = [
    "RmucWheelLegEnv",
    "RmucWheelLegCourseEnvCfg",
    "RmucWheelLegRmucEnvCfg",
    "RmucWheelLegCommandEnvCfg",
    "RmucWheelLegRoughEnvCfg",
    "RmucWheelLegRoughPlayEnvCfg",
    "RmucWheelLegPPORunnerCfg",
]
