"""Force-law parity and engine virtual-work check, without policy training."""
import importlib.util
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from radial_spring import RadialSpring, force_pair


def main():
    here=Path(__file__).resolve().parent
    source=here.parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/gas_spring.py'
    spec=importlib.util.spec_from_file_location('training_spring',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(237)
    hip=rng.normal(size=(512,3));wheel=hip+rng.normal(size=(512,3))
    reference,_=module.radial_spring_force_pair(torch.from_numpy(hip),torch.from_numpy(wheel),torch.full((512,),10.775,dtype=torch.float64))
    actual=np.array([force_pair(h,w,10.775) for h,w in zip(hip,wheel)])
    np.testing.assert_allclose(actual,reference.numpy(),atol=1e-12,rtol=0)
    model=mujoco.MjModel.from_xml_path(str(here/'rmuc_usd_limits.xml'))
    data=mujoco.MjData(model); spring=RadialSpring(model)
    data.qvel[:]=rng.normal(size=model.nv)*.1
    mujoco.mj_forward(model,data)
    info=spring.apply(model,data,10.775)
    power=float(data.qfrc_applied@data.qvel)
    free_root_force=float(abs(data.qfrc_applied[:6]).max())
    assert free_root_force<1e-9
    epsilon=1e-6; lengths=[]
    for sign in [-1,1]:
        shifted=mujoco.MjData(model);shifted.qpos[:]=data.qpos
        mujoco.mj_integratePos(model,shifted.qpos,data.qvel,sign*epsilon)
        mujoco.mj_forward(model,shifted)
        lengths.append(sum(np.linalg.norm(shifted.xanchor[w]-shifted.xanchor[h]) for h,w,b in spring.legs))
    expected_power=10.775*(lengths[1]-lengths[0])/(2*epsilon)
    np.testing.assert_allclose(power,expected_power,atol=1e-7,rtol=0)
    report=dict(passed=True,force_law_max_abs_error=float(abs(actual-reference.numpy()).max()),
        per_leg=info,net_free_root_generalized_force_max=free_root_force,
        generalized_power=power,finite_difference_power=expected_power,
        scope='Virtual radial force law and application-point mechanics; not an identified real gas-spring force/stroke law or behavior acceptance.')
    (here/'radial_spring_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
