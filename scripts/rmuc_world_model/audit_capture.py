from pathlib import Path
import argparse,json,hashlib,torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--alignment-only',action='store_true');args=p.parse_args();P=args.directory;torch.set_num_threads(2)
spec=json.loads((P/'experiment_contract.json').read_text());r=json.loads((P/'evaluation.json').read_text());manifest=json.loads((P/'replay/manifest.json').read_text());assert manifest['complete'];pf=np.load(r['policy_trace']['path']);checks={'checkpoint_matches_contract':r['checkpoint_sha256']==spec['source_sha256']}
if not args.alignment_only:
 b=json.loads(Path(spec['baseline']).read_text());checks.update({k:r[k]==b[k] for k in ['checkpoint_sha256','seed','initial_state_audit','initial_mass_inertia_readback','initial_material_readback']});bf=np.load(b['policy_trace']['path']);checks['policy_trace_exact']=np.array_equal(pf['frames'],bf['frames'])
count=0;donecount=0;previous=None;max_boundary=0.;max_action_diff=0.;max_root_diff=0.;keys=set()
for shard in manifest['shards']:
 path=Path(shard['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==shard['sha256'];s=torch.load(path,map_location='cpu',weights_only=True);n=len(s['reward']);pre=s['pre'];post=s['post'];keys.update(pre)
 assert pre.keys()==post.keys()
 assert torch.equal(post['episode_step'],pre['episode_step']+1)
 assert s['observation'].shape[-1]==25 and s['raw_action'].shape[-1]==6
 for group in [pre,post]:
  assert all(torch.isfinite(v).all() for v in group.values())
 done=s['terminated']|s['truncated'];donecount+=int(done.sum())
 if previous is not None:
  prevpost,prevdone=previous;mask=~prevdone
  for k in ['root_state_w','joint_pos','joint_vel']:
   if mask.any():max_boundary=max(max_boundary,float((pre[k][0,mask]-prevpost[k][mask]).abs().max()))
 mask=~done[:-1]
 for k in ['root_state_w','joint_pos','joint_vel']:
  if mask.any():max_boundary=max(max_boundary,float((pre[k][1:][mask]-post[k][:-1][mask]).abs().max()))
 previous=({k:v[-1].clone() for k,v in post.items()},done[-1].clone())
 cols={str(v):i for i,v in enumerate(pf['columns'])};trace=pf['frames'][count:count+n]
 max_action_diff=max(max_action_diff,float(np.max(abs(s['raw_action'].numpy()-trace[:,:,[cols[f'raw_action_{i}'] for i in range(6)]]))))
 max_root_diff=max(max_root_diff,float(np.max(abs(post['root_state_w'][:,:,2].numpy()-trace[:,:,cols['root_z']]))))
 count+=n
checks.update(replay_steps=count==len(pf['frames']),nonterminal_boundaries_exact=max_boundary==0.,actions_exact=max_action_diff==0.,terminal_root_exact=max_root_diff==0.,terminated_or_truncated_observed=donecount>0,history_recorded=all(k in keys for k in ['obs_history','action_history','leg_history','wheel_delay_history','motor_leg_motors_effort','motor_wheel_motors_effort']))
out=dict(unlogged_run_parity_verified=not args.alignment_only,passed=all(checks.values()),checks=checks,steps=count,transitions=count*pf['frames'].shape[1],done_transitions=donecount,max_boundary_difference=max_boundary,max_action_difference=max_action_diff,max_root_difference=max_root_diff,scope='Internal replay/trace/reset alignment only; no unlogged paired run' if args.alignment_only else 'Read-only collector parity and reset alignment, not world-model quality or controller certification')
(P/'capture_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));assert out['passed']
