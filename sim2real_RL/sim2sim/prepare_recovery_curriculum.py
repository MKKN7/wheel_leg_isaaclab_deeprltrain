"""Prepare audited partial-tilt reset candidates; does not change live training."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from audit_recovery_starts import make_starts


def main():
    here=Path(__file__).resolve().parent;xml=here/'rmuc_usd_limits.xml'
    model=mujoco.MjModel.from_xml_path(str(xml));root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot')
    rows=[]
    for angle in [30.,45.,60.,75.,90.]:
        for start in make_starts(model,angle):
            data=mujoco.MjData(model);data.qpos[:]=start['qpos'];data.qvel[:]=start['qvel']
            mujoco.mj_forward(model,data)
            local=data.xmat[root].reshape(3,3).T@np.array([0.,0.,-1.])
            gravity=np.array([-local[2],local[0],local[1]])
            measured=np.degrees(np.arccos(np.clip(-gravity[2],-1.,1.)))
            np.testing.assert_allclose(measured,angle,atol=1e-8,rtol=0)
            assert np.allclose(data.qpos[7:],model.qpos0[7:],atol=0.,rtol=0.)
            start.update(measured_policy_tilt_deg=float(measured),policy_gravity=gravity.tolist())
            rows.append(start)
    # Existing 90-degree callers must retain all physical initial states.
    old=json.loads((here/'recovery_start_audit.json').read_text())['starts']
    for previous,current in zip(old,rows[-4:]):
        np.testing.assert_array_equal(previous['qpos'],current['qpos'])
        np.testing.assert_array_equal(previous['qvel'],current['qvel'])
    report=dict(model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        source_script_sha256=hashlib.sha256((here/'audit_recovery_starts.py').read_bytes()).hexdigest(),
        stages_deg=[30,45,60,75,90],starts=rows,old_90degree_initial_states_exact=True,
        applied_to_training=False,topology_modified=False,
        scope='Reset candidates and geometric feasibility only; no proven recoverability, skill advancement or randomized DR coverage',
        advancement_requirement='Independent completed-episode recovery and previous-stage retention; current 90degree PPO comparison remains unchanged')
    (here/'recovery_curriculum_candidates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'starts':len(rows),'max_closure_m':max(r['closure_max_m'] for r in rows),
                      'max_floor_gap_error_m':max(abs(r['floor_gap_m']-.003) for r in rows),
                      'old_90degree_exact':True,'applied_to_training':False},indent=2))


if __name__=='__main__':main()
