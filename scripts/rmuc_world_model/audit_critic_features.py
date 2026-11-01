"""Find normalized imagined-state errors visible to the critic on heldout windows."""
import argparse,json,torch
from pathlib import Path
from rich_replay import load_capture
from gpu_replay import WindowReplay
from dynamics import Dynamics
p=argparse.ArgumentParser()
for k in ['model','critic','capture','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
ck=torch.load(a.model,map_location='cpu',weights_only=False);sd=ck['model'];m=Dynamics(sd['mean'],sd['scale'],sd['delta_scale'],sd['context_mean'],sd['context_scale'],ck['history'],ck['width'],ck.get('depth',2));m.load_state_dict(sd);m.eval()
vc=torch.load(a.critic,map_location='cpu',weights_only=False)['model'];scale=vc['scale'][:len(sd['mean'])];mean=vc['mean'][:len(sd['mean'])]
seq,meta=load_capture(a.capture,4,32,True,True);seq=[q for q in seq if q['env']%4==3];r=WindowReplay(seq,4,32,'cpu');s,ah,act,y,c=r.sample(512,torch.Generator().manual_seed(943))
with torch.no_grad():pred=m.rollout(s,ah,act,c)
err=((pred[:,-1]-y[:,-1])/scale).abs();gain=vc['net.0.weight'][:,:len(scale)].square().mean(0).sqrt();rank=(err.mean(0)*gain).argsort(descending=True)
rows=[dict(name=ck['state_columns'][i],normalized_prediction_mae=float(err[:,i].mean()),model_abs_z_p95=float(torch.quantile(((pred[:,-1,i]-mean[i])/scale[i]).abs(),.95)),real_abs_z_p95=float(torch.quantile(((y[:,-1,i]-mean[i])/scale[i]).abs(),.95)),first_layer_weight_rms=float(gain[i])) for i in rank[:30]]
out=dict(rows=rows,scope='Recorded-action H32 heldout windows; first-layer weighted discrepancy is a diagnostic ranking, not causal feature attribution or value-gradient proof.')
a.output.write_text(json.dumps(out,indent=2));print(json.dumps(rows[:10],indent=2))
