"""Validate the immutable RMUC closed-loop USD before PPO training.

Run with Isaac Sim's Python (for example ``isaaclab.bat -p``), because the
``pxr`` USD bindings are not part of a normal CPython installation.  The
validator never edits or flattens the stage.  It checks the articulation root,
rigid-body/mass/collision coverage, joint limits and authored drives, and the
four excluded spherical loop constraints.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - Isaac Sim normally bundles PyYAML
    yaml = None
from pxr import Gf, Usd, UsdGeom, UsdPhysics


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "rmuc_wheel_leg.yaml"
DEFAULT_USD = (
    REPO_ROOT
    / "source"
    / "agent_world"
    / "agent_world"
    / "assets"
    / "usd_files"
    / "rmuc_wheel_leg"
    / "rmuc_robot_closed_loop.usd"
)


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the RMUC validation config")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _target_paths(prim) -> tuple[str, ...]:
    joint = UsdPhysics.Joint(prim)
    body0 = tuple(str(path) for path in (joint.GetBody0Rel().GetTargets() or []))
    body1 = tuple(str(path) for path in (joint.GetBody1Rel().GetTargets() or []))
    return body0 + body1


def validate(stage: Usd.Stage, cfg: dict, *, allow_zero_mass: bool = False) -> dict:
    robot_cfg = dict(cfg.get("robot", {}))
    expected_dofs = list(robot_cfg.get("dof_names", []))
    controlled_dofs = set(robot_cfg.get("controlled_dof_names", []))
    expected_loops = int(cfg.get("validation", {}).get("expected_loop_constraints", 4))

    articulation_roots = [
        prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    rigid_bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    collisions = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI) or "collision" in prim.GetName().lower()
    ]

    if len(articulation_roots) != 1:
        raise RuntimeError(
            "Expected exactly one ArticulationRootAPI, found "
            f"{len(articulation_roots)}: {[str(p.GetPath()) for p in articulation_roots]}"
        )
    if not rigid_bodies:
        raise RuntimeError("USD contains no PhysicsRigidBodyAPI prims")
    if not collisions:
        raise RuntimeError("USD contains no collision prim/API")

    masses = []
    invalid_mass_paths = []
    invalid_inertia_paths = []
    for prim in rigid_bodies:
        mass_attr = prim.GetAttribute("physics:mass")
        mass = mass_attr.Get() if mass_attr else None
        if mass is not None:
            masses.append(float(mass))
            if (not _finite(mass) or float(mass) <= 0.0) and not allow_zero_mass:
                invalid_mass_paths.append(str(prim.GetPath()))
        inertia_attr = prim.GetAttribute("physics:diagonalInertia")
        inertia = inertia_attr.Get() if inertia_attr else None
        if inertia is not None:
            try:
                inertia_values = tuple(float(value) for value in inertia)
            except (TypeError, ValueError):
                inertia_values = ()
            if len(inertia_values) != 3 or not all(math.isfinite(value) for value in inertia_values):
                invalid_inertia_paths.append(str(prim.GetPath()))
    mass_warning = None
    if invalid_mass_paths and not allow_zero_mass:
        # Imported RMUC source data intentionally carries zero mass/inertia on
        # massless linkage shells while the chassis owns the measured mass.
        # Do not rewrite those hardware values here; report them and let the
        # single-environment PhysX smoke test decide whether the build is fit.
        mass_warning = (
            f"{len(invalid_mass_paths)} rigid bodies have non-positive authored mass; "
            "values were preserved. Verify PhysX stability in one environment."
        )
    if invalid_mass_paths and not masses:
        raise RuntimeError(
            "USD has no positive authored mass at all; a dynamic articulation "
            "cannot be validated without measured hardware inertias."
        )
    if invalid_inertia_paths:
        raise RuntimeError("USD has non-finite diagonal inertia on: " + ", ".join(invalid_inertia_paths[:12]))
    inertia_warning = None
    zero_inertia_paths = []
    for prim in rigid_bodies:
        attr = prim.GetAttribute("physics:diagonalInertia")
        value = attr.Get() if attr else None
        try:
            if value is not None and all(float(component) <= 0.0 for component in value):
                zero_inertia_paths.append(str(prim.GetPath()))
        except (TypeError, ValueError):
            pass
    if zero_inertia_paths and not allow_zero_mass:
        inertia_warning = (
            f"{len(zero_inertia_paths)} rigid bodies have zero authored diagonal inertia; "
            "values were preserved. Verify the PhysX single-environment smoke test."
        )

    joints = [
        prim
        for prim in stage.Traverse()
        if prim.GetTypeName().endswith("Joint") and prim.GetTypeName() != "PhysicsJoint"
    ]
    joint_names = {prim.GetName() for prim in joints}
    missing_dofs = [name for name in expected_dofs if name not in joint_names]
    if missing_dofs:
        raise RuntimeError(f"Missing configured joints: {missing_dofs}")

    finite_limit_failures = []
    motor_joints = []
    for prim in joints:
        name = prim.GetName()
        for attr_name in ("physics:lowerLimit", "physics:upperLimit"):
            attr = prim.GetAttribute(attr_name)
            if attr and attr.HasAuthoredValueOpinion():
                value = attr.Get()
                # Continuous wheel joints may use very large sentinel limits;
                # they still must be finite.
                if not _finite(value):
                    finite_limit_failures.append(f"{prim.GetPath()}:{attr_name}={value!r}")
        force_attr = prim.GetAttribute("drive:angular:physics:maxForce")
        force = force_attr.Get() if force_attr else None
        if name in controlled_dofs and force is not None and _finite(force) and float(force) > 0.0:
            motor_joints.append((name, float(force)))

    if finite_limit_failures:
        raise RuntimeError("Non-finite joint limits: " + ", ".join(finite_limit_failures))
    missing_motors = sorted(controlled_dofs - {name for name, _ in motor_joints})
    if missing_motors:
        raise RuntimeError(
            "Configured policy joints do not have finite authored angular drives: "
            + ", ".join(missing_motors)
        )

    loops = [
        prim
        for prim in stage.Traverse()
        if prim.GetTypeName() == "PhysicsSphericalJoint"
    ]
    if len(loops) != expected_loops:
        raise RuntimeError(f"Expected {expected_loops} spherical loop constraints, found {len(loops)}")
    excluded = []
    invalid_targets = []
    for prim in loops:
        joint = UsdPhysics.Joint(prim)
        excluded_attr = joint.GetExcludeFromArticulationAttr()
        excluded_value = bool(excluded_attr.Get()) if excluded_attr else False
        excluded.append(excluded_value)
        targets = _target_paths(prim)
        if len(targets) != 2 or any(
            not stage.GetPrimAtPath(path).IsValid()
            or not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI)
            for path in targets
        ):
            invalid_targets.append((str(prim.GetPath()), targets))
    if not all(excluded):
        raise RuntimeError("Every spherical loop constraint must set physics:excludeFromArticulation=true")
    if invalid_targets:
        raise RuntimeError(f"Loop constraints reference invalid body paths: {invalid_targets}")

    sensor_cfg = dict(cfg.get("sensors", {}))
    if not sensor_cfg.get("contact_sensor_prim_regex") or not sensor_cfg.get("wheel_sensor_prim_regex"):
        raise RuntimeError("Contact sensor configuration must cover chassis and both wheels")

    body_names = {prim.GetName() for prim in rigid_bodies}
    xform_cache = UsdGeom.XformCache()
    initial_loop_residuals = {}
    for prim in loops:
        joint = UsdPhysics.SphericalJoint(prim)
        body0_targets = joint.GetBody0Rel().GetTargets() or []
        body1_targets = joint.GetBody1Rel().GetTargets() or []
        if len(body0_targets) == 1 and len(body1_targets) == 1:
            local0 = joint.GetLocalPos0Attr().Get() or Gf.Vec3f(0.0, 0.0, 0.0)
            local1 = joint.GetLocalPos1Attr().Get() or Gf.Vec3f(0.0, 0.0, 0.0)
            p0 = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body0_targets[0])).Transform(Gf.Vec3d(*local0))
            p1 = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body1_targets[0])).Transform(Gf.Vec3d(*local1))
            initial_loop_residuals[str(prim.GetName())] = float((p0 - p1).GetLength())
    compat_cfg = dict(cfg.get("v14_compatibility", {}))
    legacy_joint_names = set(compat_cfg.get("legacy_required_joint_names", []))
    legacy_body_patterns = list(compat_cfg.get("legacy_required_body_patterns", []))
    semantic = dict(compat_cfg.get("expected_semantic_counts", {}))
    leg_body_names = list(compat_cfg.get("leg_body_patterns", []))
    actual_wheels = sum(1 for name in controlled_dofs if "wheel" in name and name in joint_names)
    actual_active = len(controlled_dofs.intersection(joint_names)) - actual_wheels
    actual_leg = len(expected_dofs) - actual_wheels
    legacy_missing = sorted(legacy_joint_names - joint_names)
    legacy_body_missing = [pattern for pattern in legacy_body_patterns if not any(pattern == name for name in body_names)]
    adapter_ready = (
        actual_active == int(semantic.get("active_leg_joints", 4))
        and actual_wheels == int(semantic.get("wheel_joints", 2))
        and actual_leg == int(semantic.get("leg_joints", 12))
        and len(set(leg_body_names).intersection(body_names)) == int(semantic.get("leg_links", 12))
    )

    return {
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "articulation_root": str(articulation_roots[0].GetPath()),
        "rigid_body_count": len(rigid_bodies),
        "collision_count": len(collisions),
        "joint_count": len(joints),
        "configured_dof_count": len(expected_dofs),
        "motor_joints": sorted(motor_joints),
        "loop_constraints": [str(prim.GetPath()) for prim in loops],
        "initial_loop_residual_m": initial_loop_residuals,
        "authored_mass_count": len(masses),
        "authored_mass_warning": mass_warning,
        "authored_inertia_warning": inertia_warning,
        "contact_sensor_prim_regex": sensor_cfg["contact_sensor_prim_regex"],
        "wheel_sensor_prim_regex": sensor_cfg["wheel_sensor_prim_regex"],
        "v14_direct_reuse": not legacy_missing and not legacy_body_missing,
        "v14_missing_legacy_joints": legacy_missing,
        "v14_missing_legacy_bodies": legacy_body_missing,
        "v14_adapter_ready": adapter_ready,
        "v14_semantic_counts": {
            "active_leg_joints": actual_active,
            "wheel_joints": actual_wheels,
            "leg_joints": actual_leg,
            "leg_links": len(set(leg_body_names).intersection(body_names)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-authored-zero-mass",
        action="store_true",
        help="Only for diagnosis; never use for training without a PhysX single-env check.",
    )
    args = parser.parse_args()
    stage = Usd.Stage.Open(str(args.usd))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {args.usd}")
    summary = validate(stage, _load_config(args.config), allow_zero_mass=args.allow_authored_zero_mass)
    for key, value in summary.items():
        print(f"[RMUC_USD] {key}={value}")


if __name__ == "__main__":
    main()
