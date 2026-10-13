"""Fill zero USD rigid-body mass properties from the compiled MuJoCo XML.

The source XML uses ``inertiafromgeom=auto``: child bodies have no explicit
``<inertial>`` tag, but MuJoCo computes their mass and principal inertia from
their geoms.  This script transfers those compiled values by body name.
It never changes mesh, joint, drive, collision, or root-body properties.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from pxr import Gf, Usd


def _mujoco_bodies(xml_path: Path) -> dict[str, tuple[float, tuple[float, ...], tuple[float, ...], tuple[float, ...]]]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    result = {}
    for body_id in range(1, model.nbody):  # skip world
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name:
            raise RuntimeError(f"MuJoCo body {body_id} has no name")
        result[name] = (
            float(model.body_mass[body_id]),
            tuple(float(x) for x in model.body_ipos[body_id]),
            tuple(float(x) for x in model.body_inertia[body_id]),
            tuple(float(x) for x in model.body_iquat[body_id]),
        )
    return result


def fill(xml_path: Path, usd_path: Path, *, allow_root: bool = False) -> None:
    source = _mujoco_bodies(xml_path)
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {usd_path}")
    rigid_bodies = []
    for prim in stage.Traverse():
        mass_attr = prim.GetAttribute("physics:mass")
        if mass_attr and mass_attr.HasAuthoredValueOpinion():
            rigid_bodies.append(prim)
    usd_names = {prim.GetName() for prim in rigid_bodies}
    missing = sorted(usd_names - source.keys())
    extra = sorted(source.keys() - usd_names)
    if missing or extra:
        raise RuntimeError(f"Body mapping mismatch: missing_in_xml={missing}, missing_in_usd={extra}")

    changed = []
    for prim in rigid_bodies:
        name = prim.GetName()
        if name == "robot" and not allow_root:
            continue
        mass, com, inertia, quat = source[name]
        if mass <= 0.0 or not all(value > 0.0 for value in inertia):
            raise RuntimeError(f"MuJoCo produced invalid inertial data for {name}: {mass}, {inertia}")
        prim.GetAttribute("physics:mass").Set(mass)
        prim.GetAttribute("physics:centerOfMass").Set(Gf.Vec3f(*com))
        prim.GetAttribute("physics:diagonalInertia").Set(Gf.Vec3f(*inertia))
        prim.GetAttribute("physics:principalAxes").Set(Gf.Quatf(quat[0], Gf.Vec3f(*quat[1:])))
        changed.append((name, mass))
    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"Unable to save USD: {usd_path}")
    print(f"[RMUC_MASS] usd={usd_path} bodies_updated={len(changed)}")
    print(f"[RMUC_MASS] total_compiled_mass={sum(v[0] for v in source.values()):.9f}")
    for name, mass in changed:
        print(f"[RMUC_MASS] {name} mass={mass:.9f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--usd", type=Path, action="append", required=True)
    parser.add_argument("--allow-root", action="store_true", help="Also overwrite the existing USD root inertial data.")
    args = parser.parse_args()
    for usd_path in args.usd:
        fill(args.xml, usd_path, allow_root=args.allow_root)


if __name__ == "__main__":
    main()
