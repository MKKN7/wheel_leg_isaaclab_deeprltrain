"""Frozen-teacher imagination diagnostic, conditional on recorded sensor residuals.

This is NOT independent simulator validation. Recorded commands/noise are replayed;
physical predictions and subsequent actor actions are autoregressive. No actor updates.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import torch
from dynamics import Dynamics
from rich_replay import load_capture
from observation import observe, rotate, canonical, native_root_to_wxyz
from sensor import push_sensor, push_action
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_rmuc_export import restore_actor
from rmuc_position_authority import execute_authority_action


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','capture','teacher','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--start-seconds',type=float,nargs='+',default=None)
    args=p.parse_args();assert not args.output.exists();torch.set_num_threads(2)
    ck=torch.load(args.model,map_location='cpu',weights_only=False);sd=ck['model']
    m=Dynamics(sd['mean'],sd['scale'],sd['delta_scale'],sd['context_mean'],sd['context_scale'],ck['history'],ck['width'],ck.get('depth',2))
    m.load_state_dict(sd);m.eval().requires_grad_(False)
    actor=restore_actor(args.teacher).eval().requires_grad_(False)
    seq,meta=load_capture(args.capture,ck['history'],100,True,ck.get('condition_commands',False))
    assert hashlib.sha256(args.teacher.read_bytes()).hexdigest()==meta['checkpoint_sha256']
    manifest=json.loads((args.capture/'replay/manifest.json').read_text())
    shards=[torch.load(s['path'],map_location='cpu',weights_only=True) for s in manifest['shards']]
    pre={k:torch.cat([s['pre'][k] for s in shards]) for k in ['root_state_w','joint_pos','joint_vel','_default_control_pos','policy_actions','obs_history','sensor_delay_cycles','action_delay_cycles','action_history']}
    raw=torch.cat([s['raw_action'] for s in shards]);obs=torch.cat([s['observation'] for s in shards])
    post_action=torch.cat([s['post']['policy_actions'] for s in shards]);del shards
    # Audit only the common first episode. Reset queues are initialized rather
    # than advanced normally and cannot be checked with the ordinary push rule.
    common_end=min(len(q['state']) for q in seq)
    pre={k:v[:common_end] for k,v in pre.items()}
    raw,obs,post_action=raw[:common_end],obs[:common_end],post_action[:common_end]
    assert pre['sensor_delay_cycles'].unique().tolist()==[1]
    assert pre['action_delay_cycles'].unique().tolist()==[0]
    # This capture is straight commands only: no recovery of clipped yaw target.
    # Read verified CLI schedule instead of inferring yaw from clipped observations.
    command_file=json.loads((args.capture/'command.json').read_text())
    argv=command_file if isinstance(command_file,list) else command_file.get('command',command_file.get('argv'))
    if not isinstance(argv,list):raise ValueError('Missing exact evaluation argv')
    schedule=json.loads(Path(argv[argv.index('--command-sequence')+1]).read_text());assert all(x[2]==0 for x in schedule)
    commands=torch.zeros(*obs.shape[:2],3);commands[...,0]=obs[...,7]/.5
    ids=[2,3,35,0,1,33];scale=torch.tensor([2.1,.5,1.])
    clean=observe(native_root_to_wxyz(pre['root_state_w']),pre['joint_pos'][...,ids],pre['joint_vel'][...,ids],pre['_default_control_pos'],pre['policy_actions'],commands,scale,2.,2.)
    residual=pre['obs_history'][:,:,0]-clean
    # Verify queue timing on real states, excluding initial/reset boundaries.
    actual_current=(clean+residual)[1:-1].flatten(0,1)
    replay_obs,_=push_sensor(pre['obs_history'][:-2].flatten(0,1),actual_current,torch.ones(len(actual_current),dtype=torch.bool),pre['sensor_delay_cycles'][1:-1].flatten(),commands[1:-1,:,2].flatten())
    queue_error=float((replay_obs-obs[1:-1].flatten(0,1)).abs().max())
    if queue_error>5.e-6:raise RuntimeError(f'Sensor queue timing mismatch {queue_error}')
    # Actor reconstruction and consumer must match recorded deterministic rollout.
    with torch.no_grad():
        parity=max(float((actor({'policy':x},stochastic_output=False)-y).abs().max()) for x,y in zip(obs.unbind(),raw.unbind()))
        consumer=float((execute_authority_action(obs,raw)-post_action).abs().max())
    if parity>2.e-5 or consumer>2.e-6:raise RuntimeError(f'Actor/consumer mismatch {parity}/{consumer}')
    names=meta['extra_state_names']
    def indices(key):return [16+i for i,n in enumerate(names) if n.startswith(key+':')]
    root_ids=indices('root_state_w');q_ids=indices('joint_pos')
    def model_observation(x,previous,cmd,default):
        root=x.new_zeros(len(x),13);root[:,2]=x[:,0];root[:,3:]=x[:,root_ids]
        root=native_root_to_wxyz(root)
        root[:,3:7]=torch.nn.functional.normalize(root[:,3:7],dim=-1)
        qpack=x[:,q_ids];q=torch.atan2(qpack[:,:38],qpack[:,38:])
        dq=x[:,indices('joint_vel')]
        return observe(root,q[:,ids],dq[:,ids],default,previous,cmd,scale,2.,2.)
    def physical_metrics(x):
        root=x[...,root_ids];q=torch.nn.functional.normalize(root[...,[3,0,1,2]],dim=-1)
        forward=torch.zeros_like(root[...,:3]);forward[...,2]=-1
        forward=rotate(q,forward)[...,:2];forward=forward/forward.norm(dim=-1,keepdim=True).clamp_min(1.e-6)
        vx=(root[...,4:6]*forward).sum(-1)
        gravity=torch.zeros_like(root[...,:3]);gravity[...,2]=-1
        gravity=canonical(q,gravity);pitch=torch.atan2(gravity[...,0],-gravity[...,2])
        return vx,pitch
    rows=[];envs=torch.tensor([q['env'] for q in seq]);ctx=torch.stack([q['context'][:common_end] if q['context'].ndim==2 else q['context'] for q in seq])
    with torch.no_grad():
      for seconds in (args.start_seconds if args.start_seconds is not None else [row[0] for row in schedule[1:]]):
        t=round(seconds/.008);h=100
        if t+h+1>common_end:
            raise ValueError(f'Requested window {seconds}s crosses first-episode boundary')
        states=torch.stack([q['state'][t-ck['history']:t] for q in seq]);ah=torch.stack([q['action'][t-ck['history']:t] for q in seq])
        hist=pre['obs_history'][t,envs].clone();action_hist=pre['action_history'][t,envs].clone();o=obs[t,envs].clone()
        predicted=[];actions=[];observation_errors=[]
        for j in range(h):
            a=actor({'policy':o},stochastic_output=False)
            applied,action_hist=push_action(action_hist,execute_authority_action(o,a),pre['action_delay_cycles'][t+j,envs])
            nxt=m(states,ah,a,ctx[:,t+j] if ctx.ndim==3 else ctx)
            # Post-state snapshot precedes next observation assembly. Controller queues are exact.
            nxt[:,indices('policy_actions')]=applied
            nxt[:,indices('action_history')]=action_hist.flatten(1)
            nxt[:,indices('obs_history')]=hist.flatten(1)
            predicted.append(nxt.clone());actions.append(a)
            current=model_observation(nxt,applied,commands[t+j+1,envs],pre['_default_control_pos'][t,envs])+residual[t+j+1,envs]
            # Current action history is deterministic, never borrowed from future recording.
            current[:,19:25]=applied
            current[:,6:9]=current[:,6:9].clamp(-2,2)
            o,hist=push_sensor(hist,current,torch.ones(len(envs),dtype=torch.bool),pre['sensor_delay_cycles'][t+j+1,envs],commands[t+j+1,envs,2])
            observation_errors.append((o-obs[t+j+1,envs]).abs())
            states=torch.cat([states[:,1:],nxt[:,None]],1);ah=torch.cat([ah[:,1:],a[:,None]],1)
        pred=torch.stack(predicted);a=torch.stack(actions);truth=torch.stack([q['state'][t:t+h] for q in seq],1)
        if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite imagination')
        pv,pp=physical_metrics(pred);tv,tp=physical_metrics(truth)
        if (tv-truth[...,1]).abs().max()>1.e-4 or (tp-truth[...,4]).abs().max()>1.e-4:raise RuntimeError('Physical metric reconstruction mismatch')
        for horizon in [16,32,64,100]:
          for split,mask in [('all',torch.ones(len(envs),dtype=torch.bool)),('held_out_envs',envs%4==3)]:
            ae=(a[:horizon,mask]-raw[t:t+horizon,envs[mask]]).abs();oe=torch.stack(observation_errors[:horizon])[:,mask]
            rows.append(dict(start_seconds=seconds,horizon=horizon,split=split,env_count=int(mask.sum()),vx_mae=float((pv[:horizon,mask]-tv[:horizon,mask]).abs().mean()),pitch_mae_deg=float(torch.rad2deg(pp[:horizon,mask]-tp[:horizon,mask]).abs().mean()),auxiliary_vx_mae=float((pred[:horizon,mask,1]-truth[:horizon,mask,1]).abs().mean()),vx_internal_inconsistency=float((pv[:horizon,mask]-pred[:horizon,mask,1]).abs().mean()),action_mae=float(ae.mean()),action_mae_by_dimension=ae.mean((0,1)).tolist(),observation_mae=float(oe.mean()),observation_mae_by_dimension=oe.mean((0,1)).tolist()))
    result=dict(captured_quaternion_order='xyzw',actor_math_quaternion_order='wxyz',sensor_queue_parity_max=queue_error,actor_parity_max=parity,consumer_parity_max=consumer,rows=rows,model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),teacher_sha256=meta['checkpoint_sha256'],source=meta,qualified_for_policy_update=False,scope='Conditional frozen-actor closed loop on development seed. Recorded additive sensor residuals/commands, learned physical and actuator states; analytic action/observation queues. Residuals include state-dependent noise/clipping effects, so not independent noise or simulator validation. No actor update.')
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(actor_parity=parity,consumer_parity=consumer,sensor_queue_parity=queue_error,held_out_h32=[{k:r[k] for k in ['start_seconds','vx_mae','pitch_mae_deg','action_mae']} for r in rows if r['split']=='held_out_envs' and r['horizon']==32]),indent=2))

if __name__=='__main__':main()
