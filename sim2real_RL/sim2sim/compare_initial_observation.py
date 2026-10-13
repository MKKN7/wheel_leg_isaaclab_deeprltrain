"""Check recorded Isaac reset observations against the real MuJoCo probe."""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    isaac_path = root/'artifacts/continuation_v261/final_turns.json'
    probe_path = here/'probes/v261_friction05.json'
    isaac = json.loads(isaac_path.read_text())
    probe = json.loads(probe_path.read_text())
    observations = np.asarray(isaac['initial_state_audit']['policy_observations'],dtype=np.float32)
    vx, yaw = probe['command']
    selected = np.flatnonzero(np.isclose(observations[:,7]/.5,vx)
        & np.isclose(observations[:,2]/.25+observations[:,8],yaw))
    if len(selected) == 0:
        raise ValueError('No matching Isaac initial command')
    observed = np.asarray(probe['trace'][0]['observation'],dtype=np.float32)
    reference = observations[selected]
    delta = abs(reference-observed)
    np.testing.assert_allclose(reference,np.broadcast_to(observed,reference.shape),atol=2e-6,rtol=0)
    options=ort.SessionOptions(); options.intra_op_num_threads=1
    session=ort.InferenceSession(str(root/'artifacts/continuation_v261/export_final/policy.onnx'),
        sess_options=options,providers=['CPUExecutionProvider'])
    actions = session.run(['actions'],{'obs':reference})[0]
    actual = session.run(['actions'],{'obs':observed[None]})[0][0]
    np.testing.assert_allclose(actions,np.broadcast_to(actual,actions.shape),atol=2e-6,rtol=0)
    report=dict(passed=True,matched_isaac_indices=selected.tolist(),
        per_slot_max_abs_error=delta.max(axis=0).tolist(),observation_max_abs_error=float(delta.max()),
        onnx_action_max_abs_error=float(abs(actions-actual).max()),
        isaac_source=str(isaac_path),mujoco_source=str(probe_path),
        scope='Recorded reset 25D observation and deterministic actor only; not complete physical state, later trajectory, estimator or DR parity.')
    (here/'probes/initial_observation_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
