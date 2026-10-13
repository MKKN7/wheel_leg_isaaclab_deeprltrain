"""Read-only authored USD / compiled MuJoCo inventory. Does not certify parity."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def usd_inventory(path):
    from pxr import Usd, UsdPhysics, UsdGeom, Gf
    import numpy as np
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    root_prim = next(p for p in stage.Traverse() if p.GetName() == 'robot'
                     and p.HasAPI(UsdPhysics.RigidBodyAPI))
    root_inverse = cache.GetLocalToWorldTransform(root_prim).GetInverse()
    bodies, joints, loops = {}, {}, []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass = UsdPhysics.MassAPI(prim)
            transform = cache.GetLocalToWorldTransform(prim)*root_inverse
            rotation = np.column_stack([transform.TransformDir(Gf.Vec3d(*v)) for v in np.eye(3)])
            principal = np.asarray(Gf.Matrix3d(Gf.Quatd(mass.GetPrincipalAxesAttr().Get()))).T
            inertia = rotation@principal@np.diag(mass.GetDiagonalInertiaAttr().Get())@principal.T@rotation.T
            bodies[prim.GetName()] = dict(mass=mass.GetMassAttr().Get(),
                diagonal_inertia=list(mass.GetDiagonalInertiaAttr().Get()),
                position_root=list(transform.Transform(Gf.Vec3d(0))),
                com_root=list(transform.Transform(Gf.Vec3d(mass.GetCenterOfMassAttr().Get()))),
                inertia_root=inertia.tolist())
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(prim)
            parent = stage.GetPrimAtPath(j.GetBody0Rel().GetTargets()[0])
            parent_to_root = cache.GetLocalToWorldTransform(parent)*root_inverse
            local_axis = Gf.Vec3d(*np.eye(3)['XYZ'.index(j.GetAxisAttr().Get())])
            axis_parent = Gf.Rotation(Gf.Quatd(j.GetLocalRot0Attr().Get())).TransformDir(local_axis)
            joints[prim.GetName()] = dict(axis=j.GetAxisAttr().Get(),
                authored_drive_max_force=prim.GetAttribute('drive:angular:physics:maxForce').Get(),
                axis_root=list(parent_to_root.TransformDir(axis_parent)),
                anchor_root=list(parent_to_root.Transform(Gf.Vec3d(j.GetLocalPos0Attr().Get()))),
                lower_deg=j.GetLowerLimitAttr().Get(), upper_deg=j.GetUpperLimitAttr().Get())
        if prim.IsA(UsdPhysics.SphericalJoint):
            j = UsdPhysics.SphericalJoint(prim)
            loops.append(dict(name=prim.GetName(), body0=[str(p) for p in j.GetBody0Rel().GetTargets()],
                              body1=[str(p) for p in j.GetBody1Rel().GetTargets()]))
    return dict(bodies=bodies, joints=joints, loops=loops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--usd-only', type=Path)
    parser.add_argument('--xml', type=Path, default=Path(__file__).with_name('rmuc_usd_limits.xml'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('model_audit.json'))
    parser.add_argument('--expected-physics-timestep', type=float, default=.002)
    args = parser.parse_args()
    if args.usd_only:
        print(json.dumps(usd_inventory(args.usd_only)))
        return
    if args.xml is None:
        parser.error('--xml is required')
    import mujoco
    import numpy as np
    root = Path(__file__).resolve().parents[2]
    usd = root/'source/agent_world/agent_world/assets/usd_files/rmuc_wheel_leg/rmuc_robot_closed_loop.usd'
    source = json.loads(subprocess.check_output([
        '/home/kellen/anaconda3/envs/issaclab/bin/python', str(Path(__file__).resolve()),
        '--usd-only', str(usd)], text=True))
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bodies = {}
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
    root_rotation = data.xmat[root_id].reshape(3,3)
    for i in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        principal_rotation = root_rotation.T@data.ximat[i].reshape(3,3)
        inertia = principal_rotation@np.diag(model.body_inertia[i])@principal_rotation.T
        bodies[name] = dict(mass=float(model.body_mass[i]), diagonal_inertia=model.body_inertia[i].tolist(),
            position_root=(root_rotation.T@(data.xpos[i]-data.xpos[root_id])).tolist(),
            com_root=(root_rotation.T@(data.xipos[i]-data.xpos[root_id])).tolist(),
            inertia_root=inertia.tolist())
    shared = sorted(set(source['bodies']) & set(bodies))
    mass_errors = {name: abs(source['bodies'][name]['mass']-bodies[name]['mass']) for name in shared}
    inertia_errors = {name: float(np.max(np.abs(np.sort(source['bodies'][name]['diagonal_inertia'])
        - np.sort(bodies[name]['diagonal_inertia'])))) for name in shared}
    frame_errors = {key: {name: float(np.max(np.abs(np.asarray(source['bodies'][name][key])
                    -np.asarray(bodies[name][key])))) for name in shared}
                    for key in ['position_root','com_root','inertia_root']}
    joint_frame_errors = {}
    for name, values in source['joints'].items():
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if index < 0:
            joint_frame_errors[name] = {'missing': True}
            continue
        axis = root_rotation.T@data.xaxis[index]
        anchor = root_rotation.T@(data.xanchor[index]-data.xpos[root_id])
        joint_frame_errors[name] = dict(axis_max_abs_error=float(abs(axis-values['axis_root']).max()),
            anchor_max_abs_error_m=float(abs(anchor-values['anchor_root']).max()))
    closure = {}
    for side in ('left', 'right'):
        for a, b in [('node', 'upper'), ('parallel', 'lower')]:
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f'{side}_{s}_site') for s in (a,b)]
            if min(ids) < 0:
                raise ValueError(f'Missing closure sites {side} {a} {b}')
            closure[f'{side}_{a}_{b}'] = float(np.linalg.norm(data.site_xpos[ids[0]]-data.site_xpos[ids[1]]))
    torque_probes = []
    for i in range(model.nu):
        joint = int(model.actuator_trnid[i, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        # Probe the compiled model rather than assuming ctrl already means
        # output-shaft Nm. No integration and no writes to the source model.
        requested = 3. if 'wheel' in name else 1.
        gear = float(model.actuator_gear[i, 0])
        if not np.isfinite(gear) or abs(gear) < 1e-12:
            raise ValueError(f'Invalid gear: {name}')
        data.ctrl[:] = 0
        data.ctrl[i] = requested/gear
        mujoco.mj_forward(model, data)
        measured = float(data.qfrc_actuator[model.jnt_dofadr[joint]])
        np.testing.assert_allclose(measured, requested, rtol=0, atol=1e-8)
        torque_probes.append(dict(joint=name, requested_output_nm=requested,
            motor_ctrl=float(data.ctrl[i]), measured_output_nm=measured,
            unconverted_ctrl3_output_nm=3.*gear if 'wheel' in name else None))
    report = dict(xml=str(args.xml.resolve()), xml_sha256=hashlib.sha256(args.xml.read_bytes()).hexdigest(),
        usd=str(usd), usd_sha256=hashlib.sha256(usd.read_bytes()).hexdigest(),
        mujoco_version=mujoco.__version__, usd_body_count=len(source['bodies']), mujoco_body_count=len(bodies),
        usd_total_mass=sum(b['mass'] for b in source['bodies'].values()), mujoco_total_mass=float(model.body_mass.sum()),
        usd_only_bodies=sorted(set(source['bodies'])-set(bodies)), mujoco_only_bodies=sorted(set(bodies)-set(source['bodies'])),
        per_body_mass_abs_error=mass_errors, per_body_principal_inertia_abs_error=inertia_errors,
        root_frame_comparison_max_component_errors=frame_errors,
        joint_root_frame_errors=joint_frame_errors,
        usd_spherical_joints=source['loops'], mujoco_equality_count=model.neq,
        initial_closure_distance_m=closure, initial_acceleration_finite=bool(np.isfinite(data.qacc).all()),
        xml_timestep=float(model.opt.timestep), training_physics_timestep=args.expected_physics_timestep,
        runtime_timestep_override_required=abs(float(model.opt.timestep)-args.expected_physics_timestep)>1e-12,
        training_policy_timestep=.008,
        output_torque_conversion_probes=torque_probes,
        actuators=[dict(name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i),
                        joint=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,int(model.actuator_trnid[i,0])),
                        gear=model.actuator_gear[i].tolist(), ctrlrange=model.actuator_ctrlrange[i].tolist())
                   for i in range(model.nu)],
        status='inventory_only_not_dynamics_parity',
        remaining=['Runtime articulation properties versus authored USD',
                   'collision/contact geometry and materials','runtime actuator parameters and joint limits',
                   'root reset and gravity/body frame mapping','closed-loop trajectory validation'])
    out = args.output
    if out.exists():
        parser.error(f'Refusing to overwrite an existing audit: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['usd_total_mass','mujoco_total_mass','usd_only_bodies',
          'mujoco_only_bodies','initial_closure_distance_m','status']}, indent=2))


if __name__ == '__main__':
    main()
