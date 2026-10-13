"""RMUC wheel-leg USD assets for the generic wheeled-legged RL stack.

The imported USD is deliberately treated as immutable source data.  This
module supplies an Isaac Lab :class:`ArticulationCfg` and command-layer
actuator groups for the six authored motor joints plus an explicit low-gain
passive group. The latter prevents importer-inherited high drives from
locking the closed loop, without changing mesh, mass, inertia, or hardware
motor values in the USD.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import yaml
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from agent_world import AssetPath

_LOGGER = logging.getLogger(__name__)
# ``assets/rmuc_wheel_leg.py`` lives under
# ``wheeled-legged_RL/source/agent_world/agent_world/assets``; parents[4] is
# the wheeled-legged_RL project root where its YAML configuration resides.
_PACKAGE_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_PATH = _PACKAGE_ROOT / "configs" / "rmuc_wheel_leg.yaml"


def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        _LOGGER.warning("Missing %s; using conservative RMUC asset defaults.", _CONFIG_PATH)
        return {}
    try:
        loaded = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        _LOGGER.warning("Unable to read %s (%s); using defaults.", _CONFIG_PATH, exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


_CONFIG = _load_config()
_ROBOT = dict(_CONFIG.get("robot", {}))
_ARTICULATION = dict(_ROBOT.get("articulation", {}))
_ACTUATORS = dict(_ROBOT.get("actuators", {}))


def _asset_file(name: str) -> str:
    path = Path(AssetPath) / "usd_files" / "rmuc_wheel_leg" / name
    if not path.exists():
        raise FileNotFoundError(
            f"RMUC USD asset is missing: {path}. Copy the immutable source USD "
            "into agent_world/assets/usd_files/rmuc_wheel_leg."
        )
    return str(path)


def _actuator_cfg(section: str) -> IdealPDActuatorCfg:
    values = dict(_ACTUATORS.get(section, {}))
    names = list(values.get("joint_names", []))
    if not names:
        raise ValueError(f"robot.actuators.{section}.joint_names must not be empty")
    return IdealPDActuatorCfg(
        joint_names_expr=names,
        stiffness=float(values.get("stiffness", 0.0)),
        damping=float(values.get("damping", 0.0)),
        effort_limit=float(values.get("effort_limit_nm", 0.0)),
        velocity_limit=float(values.get("velocity_limit_rad_s", 0.0)),
        velocity_limit_sim=(float(values["velocity_limit_sim_rad_s"])
                            if "velocity_limit_sim_rad_s" in values else None),
        armature=(float(values["armature_kg_m2"])
                  if "armature_kg_m2" in values else None),
    )


def _initial_root_quat_xyzw() -> tuple[float, float, float, float]:
    roll, pitch, yaw = (
        math.radians(float(value)) for value in _ROBOT.get("asset_frame_rpy_deg", (90.0, 0.0, 0.0))
    )
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
        cy * cr * cp + sy * sr * sp,
    )


def _make_cfg(usd_name: str) -> ArticulationCfg:
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=_asset_file(usd_name),
            activate_contact_sensors=True,
            copy_from_source=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=bool(_ARTICULATION.get("fix_root_link", False)),
                enabled_self_collisions=bool(_ARTICULATION.get("enabled_self_collisions", False)),
                solver_position_iteration_count=int(_ARTICULATION.get("solver_position_iteration_count", 8)),
                solver_velocity_iteration_count=int(_ARTICULATION.get("solver_velocity_iteration_count", 4)),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(_ROBOT.get("initial_pos_z_m", 0.50))),
            rot=_initial_root_quat_xyzw(),
            joint_pos={
                str(name): float(value)
                for name, value in dict(_ROBOT.get("initial_joint_pos", {})).items()
            }
            or {str(name): 0.0 for name in _ROBOT.get("dof_names", [])},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "leg_motors": _actuator_cfg("leg"),
            "wheel_motors": _actuator_cfg("wheel"),
            "passive_linkage": _actuator_cfg("passive"),
            "guide_wheels": _actuator_cfg("guide_wheels"),
        },
    )


RMUC_WHEEL_LEG_CFG = _make_cfg("rmuc_robot_closed_loop.usd")
RMUC_WHEEL_LEG_OPEN_CHAIN_CFG = _make_cfg("rmuc_robot_only.usd")

# Stable names consumed by task configs and downstream scripts.
RMUC_DOF_NAMES = tuple(str(name) for name in _ROBOT.get("dof_names", []))
RMUC_CONTROLLED_DOF_NAMES = tuple(str(name) for name in _ROBOT.get("controlled_dof_names", []))
RMUC_PASSIVE_DOF_NAMES = tuple(str(name) for name in _ROBOT.get("passive_dof_names", []))
RMUC_WHEEL_BODY_NAMES = tuple(str(name) for name in _ROBOT.get("body_names", {}).get("wheels", []))
