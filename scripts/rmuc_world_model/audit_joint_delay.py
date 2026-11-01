"""Verify measured channel exposure before interpreting joint-delay behavior."""
from pathlib import Path
import argparse,json,numpy as np,hashlib,torch
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();d=a.directory
s=json.loads((d/'command.json').read_text());r=json.loads((d/'evaluation.json').read_text());base=json.loads(Path(s['baseline']).read_text());z=np.load(r['delay_trace']['path']);f=z['frames'];checks={k:r[k]==base[k] for k in ['checkpoint_sha256','seed','initial_state_audit','initial_material_readback','initial_mass_inertia_readback']}
checks.update(sensor8=bool((f[:,:,1]==1).all()),action0=bool((f[:,:,2]==0).all()),leg2=bool((f[:,:,3]==1).all()),policy8=float(z['policy_dt'])==.008,physics2=float(z['physics_dt'])==.002,peak489=all(abs(v-4.89)<1e-5 for v in r['actual_wheel_actuator_effort_limit_range_nm']),spring431=r['gas_spring_assist_torque_nm']==4.31,outer_off=r['policy_outer_feedback_enabled'] is False)
manifest=json.loads((d/'replay/manifest.json').read_text());assert manifest['complete'];torch.set_num_threads(2);wheel_all=True;leg_all=True
for shard in manifest['shards']:
 path=Path(shard['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==shard['sha256'];x=torch.load(path,weights_only=True)
 wheel_all &= bool(x['pre']['wheel_transport_active'].all());leg_all &= bool(x['pre']['leg_delay_active'].all())
checks.update(wheel_transport_active_all=wheel_all,leg_transport_active_all=leg_all)
profile=json.loads(Path(s['command'][s['command'].index('--real2sim-wheel-profile')+1]).read_text());checks['wheel_profile2']=profile['delay_mode']=='motor_transport' and profile['delay_s']==[.002,.002]
out=dict(passed=all(checks.values()),checks=checks,pure_transport_budget_ms={'sensor':8,'action':0,'wheel':2,'leg':2,'longest_branch':10},motor_response_separate=True,scope='Actual per-step sensor/action/leg settings and per-step wheel-active readback plus configured wheel delay; not hardware end-to-end timestamp certification.',promoted=False)
(d/'runtime_verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));assert out['passed']
