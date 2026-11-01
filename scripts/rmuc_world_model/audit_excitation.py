"""Verify actual raw actions equal frozen actor plus bounded recorded excitation."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_rmuc_export import restore_actor
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);a=p.parse_args();torch.set_num_threads(2)
r=json.loads((a.capture/'evaluation.json').read_text());spec=json.loads((a.capture/'command.json').read_text());cp=Path(spec['source_checkpoint']);sha=hashlib.sha256(cp.read_bytes()).hexdigest();assert r['checkpoint_sha256']==sha==spec['source_sha256']
std=r['world_model_action_noise_std'];assert std>0 and r['acceptance_eligible'] is False and not r['passed']
trace=r['policy_trace'];path=Path(trace['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==trace['sha256']
z=np.load(path);f=z['frames'];cols={str(k):i for i,k in enumerate(z['columns'])}
o=torch.tensor(f[:,:,[cols[f'action_input_observation_{i}'] for i in range(25)]].reshape(-1,25))
x=torch.tensor(f[:,:,[cols[f'raw_action_{i}'] for i in range(6)]].reshape(-1,6));actor=restore_actor(cp).eval().requires_grad_(False)
with torch.no_grad():base=torch.cat([actor({'policy':b},stochastic_output=False) for b in o.split(2048)])
delta=x-base;limits=delta.abs().amax(0);emp=delta.std(0);checks=dict(actor_hash=True,diagnostic_only=True,finite=bool(torch.isfinite(delta).all()),bounded=bool((limits<=3*std+2e-5).all()),nonzero=bool((emp>std*.5).all()))
assert all(checks.values()),checks
out=dict(passed=True,checks=checks,source_sha256=sha,samples=len(x),configured_std=std,observed_std=emp.tolist(),observed_mean=delta.mean(0).tolist(),observed_max_abs=limits.tolist(),scope='Raw action excitation verification only. No deterministic skill acceptance.',promoted=False)
(a.capture/'excitation_verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
