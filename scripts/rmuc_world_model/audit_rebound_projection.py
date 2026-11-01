"""Read back rebound network, gate and shared-envelope projection on recorded inputs."""
import argparse,json,sys
from pathlib import Path
import torch,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from verify_rmuc_export import restore_actor
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2);actor=restore_actor(a.checkpoint).eval();z=np.load(a.capture/'evaluation.policy_trace.npz');cols=list(z['columns']);o=torch.from_numpy(z['frames'][...,[cols.index('action_input_observation_'+str(i)) for i in range(25)]].reshape(-1,25).copy());t=z['frames'][...,0].reshape(-1)*float(z['dt']);h=actor.rmuc_physics_adaptation;rows=[]
with torch.no_grad():
 for lo,hi in [(9,9.256),(9.256,9.5),(9.5,10),(10,11),(11,13),(17,20)]:
  x=o[(t>=lo)&(t<hi)];value=h._stop_bounded_value(x,h.stop_network(x))
  if hasattr(h,'stop_update_anchor'):
   anchor=h._stop_bounded_value(x,h.stop_update_anchor(x))
   value=anchor+h.stop_update_gate(x)[:,None]*(value-anchor)
  raw=h.stop_budget*torch.tanh(h.stop_rebound_network(x));gate=h.stop_rebound_gate(x);requested=raw*gate[:,None];projected=torch.maximum(torch.minimum(requested,h.stop_budget-value),-h.stop_budget-value);final=value+projected
  parity=float((final*h.stop_gate(x)[:,None]-h.stop_residual(x)).abs().max());assert parity<1e-6
  rows.append(dict(interval=[lo,hi],samples=len(x),stop_residual_parity_max=parity,gate_mean=float(gate.mean()),gate_nonzero_fraction=float((gate>0).float().mean()),raw_abs_mean=raw.abs().mean(0).tolist(),requested_abs_mean=requested.abs().mean(0).tolist(),projected_abs_mean=projected.abs().mean(0).tolist(),projection_ratio=(projected.abs().sum()/(requested.abs().sum()+1e-9)).item(),value_abs_mean=value.abs().mean(0).tolist(),final_abs_mean=final.abs().mean(0).tolist()))
out={'checkpoint_sha256':__import__('hashlib').sha256(a.checkpoint.read_bytes()).hexdigest(),'rows':rows,'scope':'Recorded observation readback; no physical counterfactual claim','promoted':False};a.output.write_text(json.dumps(out,indent=2));print(json.dumps(rows,indent=2))
