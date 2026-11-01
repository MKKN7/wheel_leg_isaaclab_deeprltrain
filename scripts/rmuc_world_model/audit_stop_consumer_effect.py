"""Measure actual rebound effect after all actor wrappers and the deployed consumer."""
import argparse,copy,json,hashlib,sys
from pathlib import Path
import torch,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from verify_rmuc_export import restore_actor
from rmuc_position_authority import execute_authority_action
p=argparse.ArgumentParser()
for k in ['checkpoint','capture','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
actor=restore_actor(a.checkpoint).eval();without=copy.deepcopy(actor);del without.rmuc_physics_adaptation.stop_rebound_network
report=json.loads((a.capture/'evaluation.json').read_text());assert report['checkpoint_sha256']==hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
z=np.load(report['policy_trace']['path']);f=z['frames'];cols=list(z['columns']);o=torch.tensor(f[...,[cols.index('action_input_observation_'+str(i)) for i in range(25)]].reshape(-1,25).copy());t=f[...,0].reshape(-1)*float(z['dt']);rows=[]
with torch.no_grad():
 for lo,hi in [(9,9.256),(9.256,9.5),(9.5,10),(10,11),(11,13),(17,20)]:
  mask=(t>=lo)&(t<hi);x=o[mask];raw=actor({'policy':x},stochastic_output=False);base=without({'policy':x},stochastic_output=False)
  applied=execute_authority_action(x,raw);applied_base=execute_authority_action(x,base)
  recorded=torch.tensor(f[...,[cols.index('raw_action_'+str(i)) for i in range(6)]].reshape(-1,6).copy())[mask]
  err=float((raw-recorded).abs().max());assert err<2e-5
  delta=raw-base;ad=applied-applied_base
  rows.append(dict(interval=[lo,hi],samples=len(x),recorded_raw_max_difference=err,raw_delta_mean=delta.mean(0).tolist(),raw_delta_abs_mean=delta.abs().mean(0).tolist(),applied_delta_mean=ad.mean(0).tolist(),applied_delta_abs_mean=ad.abs().mean(0).tolist(),raw_at_clip_fraction=(raw.abs()>=1).float().mean(0).tolist(),effective_delta_ratio=(ad.abs().sum(0)/delta.abs().sum(0).clamp_min(1e-9)).tolist()))
a.output.write_text(json.dumps(dict(rows=rows,checkpoint_sha256=report['checkpoint_sha256'],scope='Same recorded observations, final actor and action consumer with versus without rebound. Excludes state-feedback counterfactual and motor torque saturation.',promoted=False),indent=2));print(json.dumps(rows,indent=2))
