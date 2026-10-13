"""Build the RMUC closed-loop USD without touching mesh or inertial values.

The original importer places the physical robot below
``/rm_train_ground/Geometry/world/robot`` and may put an articulation root on
the zero-mass ``world`` helper.  This script discovers the actual ``robot``
prim, moves only the articulation-root API to that prim, and adds the four
MuJoCo loop closures with valid body targets.  Run it with Isaac Sim's Python.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdPhysics


LOOPS = {
    "left_loop1": ("left_node_link", "left_upper_leg", (0.005250, 0.088675, 0.069231), (0.009000, -0.064325, 0.069228)),
    "left_loop2": ("left_parallel_link", "left_lower_leg", (0.006625, -0.078620, 0.084618), (-0.001500, 0.059840, 0.025380)),
    "right_loop1": ("right_node_link", "right_upper_leg", (-0.005250, 0.088675, 0.069231), (-0.009000, -0.064325, 0.069228)),
    "right_loop2": ("right_parallel_link", "right_lower_leg", (-0.006625, -0.078620, 0.084618), (0.001500, 0.059840, 0.025380)),
}


def _find_robot(stage: Usd.Stage):
    candidates = [prim for prim in stage.Traverse() if prim.GetName() == "robot"]
    if not candidates:
        raise RuntimeError("Could not find a prim named 'robot' in the imported USD")
    # Prefer the deepest physical robot prim; the imported stage can contain
    # an IsaacRobotAPI overlay with a shallower helper of the same name.
    candidates.sort(key=lambda prim: len(str(prim.GetPath())), reverse=True)
    return candidates[0]


def _root_prim(stage: Usd.Stage, robot_prim):
    path = robot_prim.GetPath()
    while path != Sdf.Path.absoluteRootPath:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.GetName() != "robot":
            return prim
        path = path.GetParentPath()
    return stage.GetDefaultPrim()


def _remove_articulation_api(prim) -> None:
    if prim is not None and prim.IsValid() and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)


def _find_body(stage: Usd.Stage, name: str, robot_path: Sdf.Path):
    """Find a physical body by basename in flattened or nested imports."""
    nested = stage.GetPrimAtPath(robot_path.AppendChild(name))
    if nested.IsValid() and nested.HasAPI(UsdPhysics.RigidBodyAPI):
        return nested
    candidates = [
        prim
        for prim in stage.Traverse()
        if prim.GetName() == name and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    return candidates[0] if candidates else None


def prepare(input_path: Path, output_path: Path) -> None:
    stage = Usd.Stage.Open(str(input_path))
    if stage is None:
        raise RuntimeError(f"Unable to open {input_path}")
    robot = _find_robot(stage)
    robot_path = robot.GetPath()
    # Keep exactly one articulation root.  Only schema placement is changed;
    # meshes, masses, inertias, drives and joint limits are not rewritten.
    for prim in list(stage.Traverse()):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            _remove_articulation_api(prim)
    UsdPhysics.ArticulationRootAPI.Apply(robot)

    stage_root = stage.GetDefaultPrim()
    if not stage_root or not stage_root.IsValid():
        stage_root = _root_prim(stage, robot)
    constraints_path = stage_root.GetPath().AppendChild("constraints")
    if stage.GetPrimAtPath(constraints_path).IsValid():
        stage.RemovePrim(constraints_path)
    constraints = stage.DefinePrim(constraints_path, "Scope")
    del constraints

    for name, (body0, body1, pos0, pos1) in LOOPS.items():
        body0_prim = _find_body(stage, body0, robot_path)
        body1_prim = _find_body(stage, body1, robot_path)
        if body0_prim is None or body1_prim is None:
            raise RuntimeError(
                f"Loop {name} has invalid rigid-body targets: {body0}, {body1}"
            )
        body0_path = body0_prim.GetPath()
        body1_path = body1_prim.GetPath()
        prim = stage.DefinePrim(constraints_path.AppendChild(name), "PhysicsSphericalJoint")
        joint = UsdPhysics.SphericalJoint(prim)
        joint.CreateBody0Rel().SetTargets([body0_path])
        joint.CreateBody1Rel().SetTargets([body1_path])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*pos0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*pos1))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        joint.CreateJointEnabledAttr().Set(True)
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateExcludeFromArticulationAttr().Set(True)

    stage.SetDefaultPrim(stage_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage.Export(str(output_path))
    print(f"[RMUC_CLOSED_LOOP] wrote {output_path}")
    print(f"[RMUC_CLOSED_LOOP] articulation_root={robot_path}")
    print(f"[RMUC_CLOSED_LOOP] constraints={len(LOOPS)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.input, args.output)


if __name__ == "__main__":
    main()
