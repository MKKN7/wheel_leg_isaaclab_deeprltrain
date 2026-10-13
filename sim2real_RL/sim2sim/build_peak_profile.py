"""Explicit diagnostic peak profile; preserve geometry, inertia and constraints."""
import json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np,yaml
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
source=ROOT/'sim2real_RL/sim2sim/rmuc_usd_limits.xml'
config=ROOT/'configs/rmuc_hardware_target.yaml';cfg=yaml.safe_load(config.read_text());peak=cfg['wheel']['max_peak_output_torque_nm']
assert peak==5.5 and cfg['wheel']['max_continuous_output_torque_nm']==3.
tree=ET.parse(source);element=tree.getroot();element.find('compiler').set('meshdir','meshes_mj')
changes=[]
for motor in element.findall('./actuator/motor'):
 if motor.get('joint') in ('left_wheel_spin','right_wheel_spin'):
  assert float(motor.get('gear'))==1.
  changes.append(dict(name=motor.get('name'),old_ctrlrange=motor.get('ctrlrange'),old_forcerange=motor.get('forcerange'),new_output_peak_nm=peak))
  motor.set('ctrlrange',f'-{peak} {peak}');motor.set('forcerange',f'-{peak} {peak}')
assert len(changes)==2
path=P/'rmuc_peak55.xml'
with path.open('xb') as f:tree.write(f,encoding='utf-8',xml_declaration=True)
a=mujoco.MjModel.from_xml_path(str(source));b=mujoco.MjModel.from_xml_path(str(path))
fields=['body_parentid','body_pos','body_quat','body_mass','body_inertia','jnt_type','jnt_axis','jnt_pos','jnt_range','qpos0','dof_damping','dof_armature','geom_type','geom_size','geom_pos','geom_quat','geom_friction','mesh_vert','mesh_face','eq_data','eq_type','eq_obj1id','eq_obj2id','actuator_gear','actuator_trnid']
for key in fields:np.testing.assert_array_equal(getattr(a,key),getattr(b,key),err_msg=key)
for c in changes:
 i=mujoco.mj_name2id(b,mujoco.mjtObj.mjOBJ_ACTUATOR,c['name']);np.testing.assert_array_equal(b.actuator_ctrlrange[i],[-peak,peak]);np.testing.assert_array_equal(b.actuator_forcerange[i],[-peak,peak])
data=mujoco.MjData(b)
for sign in (-1,1):
 data.ctrl[:]=sign*100.;mujoco.mj_forward(b,data)
 for i in range(b.nu):
  joint=int(b.actuator_trnid[i,0]);name=mujoco.mj_id2name(b,mujoco.mjtObj.mjOBJ_JOINT,joint)
  expected=peak if name in ('left_wheel_spin','right_wheel_spin') else 40.
  np.testing.assert_allclose(data.qfrc_actuator[b.jnt_dofadr[joint]],sign*expected,atol=1e-10,rtol=0)
report=dict(passed=True,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),profile_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),output=str(path),output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),continuous_reference_nm=3.0,peak_limit_nm=peak,changes=changes,unchanged_compiled_fields=fields,scope='MuJoCo peak-capacity profile only; no behavior, thermal or torque-speed-envelope certification')
(P/'peak_mjcf_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('compiled profile checked:',path)
