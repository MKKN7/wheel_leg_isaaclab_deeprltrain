"""Gymnasium registration for the local RMUC closed-loop wheel-leg task."""

from __future__ import annotations

import gymnasium as gym


_COMMON_KWARGS = {
    "entry_point": "agent_tasks.direct.rmuc_wheel_leg.adapter:RmucWheelLegEnv",
    "disable_env_checker": True,
    "kwargs": {
        "rsl_rl_cfg_entry_point": (
            "agent_tasks.direct.rmuc_wheel_leg.rsl_rl_ppo_cfg:RmucWheelLegPPORunnerCfg"
        )
    },
}


def _register_once(task_id: str, env_cfg_name: str) -> None:
    """Register one RMUC task without importing Isaac/PhysX at module load."""

    if task_id in gym.registry:
        return
    kwargs = dict(_COMMON_KWARGS["kwargs"])
    kwargs["env_cfg_entry_point"] = (
        "agent_tasks.direct.rmuc_wheel_leg.adapter:" + env_cfg_name
    )
    gym.register(
        id=task_id,
        entry_point=_COMMON_KWARGS["entry_point"],
        disable_env_checker=True,
        kwargs=kwargs,
    )


_register_once("Robotics-RMUC-WheelLeg-Course-v0", "RmucWheelLegCourseEnvCfg")
_register_once("Robotics-RMUC-WheelLeg-Course-Play-v0", "RmucWheelLegCourseEnvCfg")
_register_once("Robotics-RMUC-WheelLeg-RMUC-Play-v0", "RmucWheelLegRmucEnvCfg")
_register_once("Robotics-RMUC-WheelLeg-Command-v0", "RmucWheelLegCommandEnvCfg")
_register_once("Robotics-RMUC-WheelLeg-Command-Play-v0", "RmucWheelLegCommandEnvCfg")
_register_once("Robotics-RMUC-WheelLeg-Rough-v0", "RmucWheelLegRoughEnvCfg")

# V14-style aliases.  They intentionally point to the RMUC adapter instead of
# the legacy WheelbipeV14Env, whose fixed front/rear/spring bookkeeping cannot
# represent this closed-loop asset without inventing joints or changing mesh
# semantics.
_register_once("Robotics-Wheelbipe-V14-RMUC-Flat-v0", "RmucWheelLegCourseEnvCfg")
_register_once("Robotics-Wheelbipe-V14-RMUC-Flat-Play-v0", "RmucWheelLegCourseEnvCfg")
_register_once("Robotics-Wheelbipe-V14-RMUC-Play-v0", "RmucWheelLegRmucEnvCfg")
_register_once("Robotics-Wheelbipe-V14-RMUC-Command-v0", "RmucWheelLegCommandEnvCfg")
_register_once("Robotics-Wheelbipe-V14-RMUC-Command-Play-v0", "RmucWheelLegCommandEnvCfg")
_register_once("Robotics-Wheelbipe-V14-RMUC-Rough-v0", "RmucWheelLegRoughPlayEnvCfg")

if 'Robotics-RMUC-WheelLeg-Recovery-v0' not in gym.registry:
    gym.register(id='Robotics-RMUC-WheelLeg-Recovery-v0',
        entry_point='agent_tasks.direct.rmuc_wheel_leg.recovery_env:RmucRecoveryEnv',
        disable_env_checker=True,
        kwargs={'env_cfg_entry_point':'agent_tasks.direct.rmuc_wheel_leg.recovery_env:RmucRecoveryEnvCfg',
                'rsl_rl_cfg_entry_point':'agent_tasks.direct.rmuc_wheel_leg.rsl_rl_ppo_cfg:RmucRecoveryPPORunnerCfg'})
