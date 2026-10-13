"""Compare actual Isaac torque law and MuJoCo generalized forces, without training."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from equivalent_spring import apply, validate_model, JOINT_ORDER


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--asymmetric',action='store_true')
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    source=here.parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/gas_spring.py'
    spec=importlib.util.spec_from_file_location('training_spring',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    xml=here/'rmuc_usd_limits.xml'
    model=mujoco.MjModel.from_xml_path(str(xml));validate_model(model)
    root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot')
    joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in JOINT_ORDER]
    dofs=model.jnt_dofadr[joints]
    rng=np.random.default_rng(341);errors=[];power_errors=[];force_errors=[]
    for _ in range(64):
        data=mujoco.MjData(model)
        quaternion=rng.normal(size=4);quaternion/=np.linalg.norm(quaternion)
        data.qpos[3:7]=quaternion
        data.qvel[:]=rng.normal(size=model.nv)
        mujoco.mj_forward(model,data)
        # Existing external loads must survive spring accumulation.
        baseline=rng.normal(size=model.nv);data.qfrc_applied[:]=baseline
        magnitudes=rng.uniform(8.62,12.93,2) if args.asymmetric else np.array([10.775,10.775])
        actual=apply(model,data,magnitudes if args.asymmetric else 10.775)
        reference=module.spring_body_torques(torch.from_numpy(data.xquat[root:root+1].copy()),
            torch.from_numpy(magnitudes[None]) if args.asymmetric else 10.775)[0].numpy()
        np.testing.assert_allclose(actual,reference,atol=1e-12,rtol=0)
        np.testing.assert_allclose(actual.sum(0),0.,atol=1e-12,rtol=0)
        expected=np.zeros(model.nv);expected[dofs]=-magnitudes
        delta=data.qfrc_applied-baseline
        np.testing.assert_allclose(delta,expected,atol=1e-10,rtol=0)
        power=float(delta@data.qvel)
        expected_power=float(-magnitudes@data.qvel[dofs])
        np.testing.assert_allclose(power,expected_power,atol=1e-10,rtol=0)
        errors.append(float(abs(actual-reference).max()))
        force_errors.append(float(abs(delta-expected).max()))
        power_errors.append(abs(power-expected_power))
    axis=model.jnt_axis[joints[0]].copy()
    model.jnt_axis[joints[0]]=-axis
    try:
        validate_model(model)
    except ValueError:
        rejected_wrong_axis=True
    else:
        raise AssertionError('Wrong spring axis accepted')
    finally:
        model.jnt_axis[joints[0]]=axis
    report=dict(passed=True,samples=64,asymmetric=args.asymmetric,
        per_leg_joint_torque_nm=None if args.asymmetric else -10.775,
        body_torque_max_abs_error=max(errors),generalized_force_max_abs_error=max(force_errors),
        virtual_work_max_abs_error=max(power_errors),wrong_axis_rejected=rejected_wrong_axis,
        training_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        mjcf_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        scope='Force-law parity, internal reaction, generalized force and virtual work only; no policy recovery or hardware certification')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
