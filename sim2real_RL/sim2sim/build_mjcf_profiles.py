"""Generate explicit output-shaft torque profiles, preserving source geometry."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml


def topology(element):
    """Protect body hierarchy, joint geometry and closure constraints."""
    entries=[]
    def visit(node,path):
        path=path+((node.tag,node.get('name')),)
        if node.tag in ['body','joint','freejoint']:
            entries.append((path,{k:v for k,v in node.attrib.items()
                                 if k not in ['damping','armature']}))
        for child in node:visit(child,path)
    visit(element.find('worldbody'),())
    return entries,[dict(x.attrib) for x in element.findall('./equality/*')]


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    source = here/'wheeled_infantry.xml'
    usd = root/'source/agent_world/agent_world/assets/usd_files/rmuc_wheel_leg/rmuc_robot_closed_loop.usd'
    params = root/'logs/rsl_rl/rmuc_wheel_leg_course/2026-09-29_13-03-23_negative_yaw_v261/params/env.yaml'
    cfg = yaml.load(params.read_text(), Loader=yaml.BaseLoader)
    inventory = json.loads(subprocess.check_output(['/home/kellen/anaconda3/envs/issaclab/bin/python',
        str(here/'audit_model.py'),'--usd-only',str(usd)],text=True))
    original = mujoco.MjModel.from_xml_path(str(source))
    original_topology = topology(ET.parse(source).getroot())
    reports=[]
    for profile in ['training_v261','usd_limits']:
        tree=ET.parse(source)
        element=tree.getroot()
        element.insert(0,ET.Comment(' Generated output-shaft Nm profile; geometry preserved; see mjcf_profiles.json. '))
        element.find('option').set('timestep',cfg['sim']['dt'])
        motors=element.findall('./actuator/motor')
        controlled={m.get('joint') for m in motors}
        for j in element.findall('.//worldbody//joint'):
            name=j.get('name')
            if name in controlled:
                j.set('damping','0')  # PD/damping is explicit in the controller.
            elif name.startswith('guide_wheel'):
                j.set('damping','0')
            else:
                j.set('damping',cfg['robot_cfg']['actuators']['passive_linkage']['damping'])
        friction=cfg['sim']['physics_material']['dynamic_friction']
        for geom in element.findall('.//geom'):
            values=geom.get('friction','1 .005 .0001').split()
            geom.set('friction',' '.join([friction,*values[1:]]))
        # Explicit self-contact pairs use the same sliding coefficients.
        for pair in element.findall('./contact/pair'):
            values=pair.get('friction','1 1 .005 .0001 .0001').split()
            pair.set('friction',' '.join([friction,friction,*values[2:]]))
        limits={}
        for motor in motors:
            joint=motor.get('joint')
            limit=inventory['joints'][joint]['authored_drive_max_force']
            if profile=='training_v261' and 'wheel' in joint:
                limit=float(cfg['robot_cfg']['actuators']['wheel_motors']['effort_limit'])
            if not np.isfinite(limit) or limit<=0:
                raise ValueError(f'Invalid source actuator limit {joint}')
            limits[joint]=float(limit)
            motor.set('gear','1')
            for key in ['ctrlrange','forcerange']: motor.set(key,f'{-limit:g} {limit:g}')
        output=here/f'rmuc_{profile}.xml'
        if topology(element) != original_topology:
            raise RuntimeError('Refusing profile generation: source topology or joint geometry changed')
        tree.write(output,encoding='utf-8',xml_declaration=True)
        model=mujoco.MjModel.from_xml_path(str(output))
        # Altering transmission units must not change inertial/kinematic data.
        for attr in ['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat',
                     'jnt_pos','jnt_axis','geom_pos','geom_quat','geom_size']:
            np.testing.assert_array_equal(getattr(model,attr),getattr(original,attr))
        data=mujoco.MjData(model)
        for sign in [-1,1]:
            data.ctrl[:]=sign*100
            mujoco.mj_forward(model,data)
            for i in range(model.nu):
                joint=int(model.actuator_trnid[i,0])
                name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,joint)
                np.testing.assert_allclose(data.qfrc_actuator[model.jnt_dofadr[joint]],
                                           sign*limits[name],atol=1e-8,rtol=0)
        reports.append(dict(profile=profile,file=output.name,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                            joint_output_limits_nm=limits,geometry_inertia_unchanged=True,
                            positive_negative_saturation_probe_passed=True))
    report=dict(source_xml_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_usd_sha256=hashlib.sha256(usd.read_bytes()).hexdigest(),
        source_training_params_sha256=hashlib.sha256(params.read_bytes()).hexdigest(),profiles=reports,
        scope='Explicit torque units/limits and nominal parameters only; not complete physics parity or final hardware certification.',
        remaining=['Contact solver equivalence','PhysX velocity limits','Radial spring and measured damping',
                   'DR and whole behavior acceptance'])
    (here/'mjcf_profiles.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
