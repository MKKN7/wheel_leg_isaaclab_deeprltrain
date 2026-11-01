"""Held-out intervention-family check. Recorded action streams, not policy rollout certification."""
from pathlib import Path
import argparse,json,torch,numpy as np
from replay import load_report,STATE_COLUMNS
from dynamics import Dynamics
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--capture-inputs',action='store_true');p.add_argument('--physical-root-metrics',action='store_true');args=p.parse_args();torch.set_num_threads(2)
ck=torch.load(args.model,map_location='cpu',weights_only=False);sd=ck['model'];m=Dynamics(sd['mean'],sd['scale'],sd['delta_scale'],sd['context_mean'],sd['context_scale'],ck['history'],ck['width'],ck.get('depth',2));m.load_state_dict(sd);m.eval();m.requires_grad_(False)
if args.capture_inputs:
 from rich_replay import load_capture
 rich=len(sd['mean'])>16
 bs,bmeta=load_capture(args.baseline,4,100,rich,ck.get('condition_commands',False));ps,pmeta=load_capture(args.probe,4,100,rich,ck.get('condition_commands',False))
 args.baseline=args.baseline/'evaluation.json';args.probe=args.probe/'evaluation.json'
else:
 bs,bmeta=load_report(args.baseline,4,100);ps,pmeta=load_report(args.probe,4,100)
bd={x['env']:x for x in bs};pd={x['env']:x for x in ps};br=json.loads(args.baseline.read_text());pr=json.loads(args.probe.read_text());raw=np.load(br['policy_trace']['path']);age=raw['frames'][:,:,0];probe_start=float(pr['diagnostic_action_bias_start_seconds']);t0=int(np.flatnonzero(age[:,0]*float(raw['dt'])>=probe_start-1e-6)[0]);assert pr['initial_state_audit']==br['initial_state_audit'] and pr['initial_mass_inertia_readback']==br['initial_mass_inertia_readback'];rows=[]
for env in bd:
 b=bd[env];p=pd[env];t=t0
 assert torch.equal(b['state'][:t-1],p['state'][:t-1])
 state=b['state'][t-4:t][None];ah=b['action'][t-4:t][None];ctx=b['context'][t:t+100][None] if b['context'].ndim==2 else b['context'][None]
 ab=b['action'][t:t+100][None];ap=p['action'][t:t+100][None]
 # Existing post-step state row t-1 is pre-action state for row t.
 with torch.no_grad():yb=m.rollout(state,ah,ab,ctx)[0];yp=m.rollout(state,ah,ap,ctx)[0]
 actual_b=b['state'][t:t+100];actual_p=p['state'][t:t+100]
 if args.physical_root_metrics:
  assert args.capture_inputs and rich
  from observation import rotate,canonical
  ids=[16+i for i,n in enumerate(bmeta['extra_state_names']) if n.startswith('root_state_w:')]
  def physical(x):
   out=x.clone();r=x[:,ids];q=torch.nn.functional.normalize(r[:,[3,0,1,2]],dim=-1)
   v=torch.zeros_like(r[:,:3]);v[:,2]=-1;f=rotate(q,v)[:,:2];f=f/f.norm(dim=-1,keepdim=True).clamp_min(1e-6)
   out[:,1]=(r[:,4:6]*f).sum(-1);g=canonical(q,v);out[:,4]=torch.atan2(g[:,0],-g[:,2]);return out
  yb,yp=physical(yb),physical(yp)
  pb,pp=physical(actual_b),physical(actual_p)
  assert (pb[:,1]-actual_b[:,1]).abs().max()<1e-4 and (pp[:,4]-actual_p[:,4]).abs().max()<1e-4
  actual_b,actual_p=pb,pp
 if not torch.isfinite(yb).all() or not torch.isfinite(yp).all():raise RuntimeError('Nonfinite rollout')
 horizons={}
 for h in [16,32,64,100]:
  actual_delta=float(actual_p[h-1,1]-actual_b[h-1,1]);pred_delta=float(yp[h-1,1]-yb[h-1,1])
  horizons[str(h)]=dict(actual_vx_delta=actual_delta,predicted_vx_delta=pred_delta,delta_error=abs(pred_delta-actual_delta),probe_vx_mae=float((yp[:h,1]-actual_p[:h,1]).abs().mean()),probe_pitch_mae_deg=float(torch.rad2deg(yp[:h,4]-actual_p[:h,4]).abs().mean()))
 rows.append(dict(env=env,case=pr['environment_bucket_ids'][env],horizons=horizons,actual_rebound_improvement=float(actual_p[:,1].min()-actual_b[:,1].min()),predicted_rebound_improvement=float(yp[:,1].min()-yb[:,1].min())))
summary={}
for h in [16,32,64,100]:
 eligible=[x for x in rows if abs(x['horizons'][str(h)]['actual_vx_delta'])>.0005]
 summary[str(h)]=dict(n=len(eligible),direction_agreement=float(np.mean([np.sign(x['horizons'][str(h)]['actual_vx_delta'])==np.sign(x['horizons'][str(h)]['predicted_vx_delta']) for x in eligible])) if eligible else None,median_probe_vx_mae=float(np.median([x['horizons'][str(h)]['probe_vx_mae'] for x in rows])),median_delta_error=float(np.median([x['horizons'][str(h)]['delta_error'] for x in rows])))
heldout_summary={}
for h in [16,32,64,100]:
 eligible=[x for x in rows if x['env']%4==3 and abs(x['horizons'][str(h)]['actual_vx_delta'])>.0005]
 heldout_summary[str(h)]=dict(n=len(eligible),direction_agreement=float(np.mean([np.sign(x['horizons'][str(h)]['actual_vx_delta'])==np.sign(x['horizons'][str(h)]['predicted_vx_delta']) for x in eligible])) if eligible else None,median_delta_error=float(np.median([x['horizons'][str(h)]['delta_error'] for x in eligible])) if eligible else None)
# Confirm action gradient is finite; magnitude/sign accuracy requires actual perturbation validation above.
a=ab.clone().requires_grad_(True);gradient_prediction=m.rollout(state,ah,a,ctx)
(physical(gradient_prediction[0])[-1,1] if args.physical_root_metrics else gradient_prediction[:,-1,1].sum()).backward()
out=dict(metric_source='root_state_xyzw_decoded' if args.physical_root_metrics else 'auxiliary_prediction_heads',summary=summary,heldout_environment_summary=heldout_summary,intervention_family_held_out=False,rows=rows,action_gradient_finite=bool(torch.isfinite(a.grad).all()),source_baseline=bmeta,source_probe=pmeta,qualified_for_policy_update=False,scope='Same-seed recorded intervention replay; family may be in training, heldout envs reported separately; open-loop recorded action streams include actor feedback differences. Not on-policy imagination nor proof of true simulator Jacobian.')
assert not args.output.exists();args.output.write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))
