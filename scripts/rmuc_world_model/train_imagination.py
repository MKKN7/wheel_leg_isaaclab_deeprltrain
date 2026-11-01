"""On-policy imagined actor/GAE-critic updates; candidates require physical gates.

The frozen dynamics is updated between invocations with fresh simulator captures.
Only selected complete adaptation modules change; frozen experts stay immutable.
"""
import argparse, copy, hashlib, json, sys, time
from pathlib import Path
import torch
from torch.nn import functional as F
from dynamics import Dynamics
from gpu_replay import WindowReplay
from domain_sampling import DelayDomainSampler, ActiveReverseSampler
from stop_phase_sampling import StopPhaseSampler, stop_phase_labels
from queue_schema import align_delay_queues
from rich_replay import load_capture
from gae import ReturnCritic, lambda_returns
from constraint_update import project_update, late_phase, upper_quartile
from tracking_objective import speed_cost, LEGACY, PHYSICAL
from critic_context import features as value_features, validate_reward_schema
from transition_reward import step_metadata, overshoot_cost, robust_overshoot_cost
from observation import observe, native_root_to_wxyz, rotate, canonical
from sensor import push_action, push_sensor, sensor_noise
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_rmuc_export import restore_actor
from rmuc_position_authority import execute_authority_action


def physical(x, ix):
    packed=x[:,ix('root_state_w')]
    root=native_root_to_wxyz(torch.cat([x.new_zeros(len(x),2),x[:,:1],packed],-1))
    quat=F.normalize(root[:,3:7],dim=-1)
    root=torch.cat([root[:,:3],quat,root[:,7:]],-1)
    qp=x[:,ix('joint_pos')];q=torch.atan2(qp[:,:38],qp[:,38:]);dq=x[:,ix('joint_vel')]
    down=torch.zeros_like(root[:,:3]);down[:,2]=-1
    fw=rotate(quat,down)[:,:2];fw=F.normalize(fw,dim=-1)
    vx=(root[:,7:9]*fw).sum(-1);gravity=canonical(quat,down)
    yaw=canonical(quat,root[:,10:13])[:,2]
    pitch=torch.atan2(gravity[:,0],-gravity[:,2]);roll=torch.atan2(gravity[:,1],-gravity[:,2])
    return root,q,dq,vx,yaw,pitch,roll


def stage_objective(x, context, ix, tracking_version=LEGACY):
    root,q,dq,vx,yaw,pitch,roll=physical(x,ix)
    target_v,target_y=context[:,-3],context[:,-2]
    # Monotone errors avoid flat exponential reward far from target.
    huber=lambda z:torch.sqrt(1+z.square())-1
    vt=torch.where(target_v.abs()<.025,.01,.1*target_v.abs()).clamp_min(.01)
    straight=target_y.abs()<.05
    active=q[:,2]-q[:,0];hip=q[:,3]+q[:,1]
    angle_limit=torch.deg2rad(x.new_tensor(2.))
    symmetry=(huber(active/angle_limit)+huber(hip/angle_limit))*straight
    tracking=speed_cost(vx-target_v,target_v,tracking_version)+huber((yaw-target_y)/.05)
    pitch_cost=huber(F.relu(pitch.abs()-torch.deg2rad(x.new_tensor(5.)))/.05)
    # During bank turns roll must follow lateral acceleration; no straight-leg gate there.
    roll_cost=huber(F.relu(roll.abs()-torch.deg2rad(x.new_tensor(5.)))/.05)*straight
    # Smooth survival is an imagined termination proxy, NOT the simulator's acceptance gate.
    survival=torch.sigmoid((root[:,2]-.12)/.015)*torch.sigmoid((1.0-pitch.abs())/.05)*torch.sigmoid((1.1-roll.abs())/.05)
    reward=.008*(2-tracking-.25*symmetry-.1*(pitch_cost+roll_cost))-10*(1-survival)
    return reward,survival,dict(vx=vx,yaw=yaw,active=active,hip=hip)


def main():
    p=argparse.ArgumentParser()
    for k in ['model','teacher','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--captures',type=Path,nargs='+',required=True)
    p.add_argument('--balance-delay-domains',action='store_true');p.add_argument('--pad-unused-delay-queues',action='store_true');p.add_argument('--retention-teacher',type=Path);p.add_argument('--overshoot-weight',type=float,default=0.);p.add_argument('--critic',type=Path);p.add_argument('--actor-optimizer',type=Path)
    p.add_argument('--modules',nargs='+',default=['startup_network','stop_network'])
    p.add_argument('--focus-stop-phases',action='store_true')
    p.add_argument('--focus-reverse-fraction',type=float,default=0.)
    p.add_argument('--focus-reverse-command-range',type=float,nargs=2)
    p.add_argument('--focus-reverse-deceleration',action='store_true')
    p.add_argument('--capture-prefix-config',type=Path)
    p.add_argument('--target-critic-tau',type=float,default=0.)
    p.add_argument('--finite-horizon-only',action='store_true')
    p.add_argument('--fixed-real-tail',type=Path)
    p.add_argument('--overshoot-loss',choices=['quadratic','smooth_linear_tail'],default='quadratic')
    p.add_argument('--warmup-max-critic-loss',type=float,default=100.)
    p.add_argument('--warmup-max-bootstrap',type=float,default=50.)
    p.add_argument('--frozen-comparison-batches',type=int,default=0);p.add_argument('--project-gate-gradients',action='store_true');p.add_argument('--tracking-version',choices=[LEGACY,PHYSICAL],default=LEGACY);p.add_argument('--mask-constant-critic-inputs',action='store_true');p.add_argument('--critic-warmup-updates',type=int,default=0);p.add_argument('--updates',type=int,default=20);p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--horizon',type=int,default=32);p.add_argument('--device',default='cuda:0')
    p.add_argument('--actor-lr',type=float,default=2e-5);p.add_argument('--critic-lr',type=float,default=2e-4)
    p.add_argument('--gamma',type=float,default=.998);p.add_argument('--lam',type=float,default=.95)
    p.add_argument('--budget-seconds',type=float,default=600);p.add_argument('--rehearsal-weight',type=float,default=50.)
    args=p.parse_args()
    overshoot_fn=robust_overshoot_cost if args.overshoot_loss=='smooth_linear_tail' else overshoot_cost
    if args.finite_horizon_only and (args.critic or args.critic_warmup_updates or args.target_critic_tau):raise ValueError('Finite-horizon mode has no critic bootstrap or warmup')
    if args.fixed_real_tail and (args.finite_horizon_only or args.critic or args.critic_warmup_updates or args.target_critic_tau or args.modules not in [['reverse_network'],['reverse_update_anchor']]):raise ValueError('Fixed real tail requires reverse-only, no other critic mode')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(413);device=torch.device(args.device)
    ck=torch.load(args.model,map_location='cpu',weights_only=False);sd=ck['model']
    wm=Dynamics(sd['mean'],sd['scale'],sd['delta_scale'],sd['context_mean'],sd['context_scale'],ck['history'],ck['width'],ck.get('depth',2)).to(device)
    wm.load_state_dict(sd);wm.eval().requires_grad_(False)
    assert ck.get('condition_commands') and ck['history']==4
    actor=restore_actor(args.teacher).to(device).eval().requires_grad_(False)
    source_actor=copy.deepcopy(actor).requires_grad_(False)
    teacher=restore_actor(args.retention_teacher).to(device).eval().requires_grad_(False) if args.retention_teacher else source_actor
    selected=[];allowed=[]
    for name in args.modules:
        module=getattr(actor.rmuc_physics_adaptation,name)
        module.requires_grad_(True);selected.extend(module.parameters());allowed.append('rmuc_physics_adaptation.'+name+'.')
    seq=[];provenance=[];histories=[]
    prefixes={str(Path(k).resolve()):v for k,v in json.loads(args.capture_prefix_config.read_text()).items()} if args.capture_prefix_config else {}
    for directory in args.captures:
        trajectories,meta=load_capture(directory,4,args.horizon+1,True,True)
        if str(directory.resolve()) in prefixes:
            from replay_window import prefix_sequences
            cutoff=prefixes[str(directory.resolve())]
            trajectories=prefix_sequences(trajectories,cutoff,horizon=args.horizon+1)
            meta['explicit_prefix_seconds']=cutoff
        if args.pad_unused_delay_queues:
            trajectories,meta=align_delay_queues(trajectories,meta,ck['state_columns'][16:])
        assert ck['state_columns']==meta['state_columns']+meta['extra_state_names']
        manifest=json.loads((directory/'replay/manifest.json').read_text())
        hist=torch.cat([torch.load(s['path'],map_location='cpu',weights_only=True)['pre']['obs_history'] for s in manifest['shards']])
        for q in trajectories:
            if q['env']%4==3:continue
            q['sensor_history']=hist[:len(q['state']),q['env']]
            seq.append(q)
        provenance.append(meta)
    replay=WindowReplay(seq,4,args.horizon+1,device)
    domain_sampler=DelayDomainSampler(replay) if args.balance_delay_domains else None
    if args.focus_stop_phases:
        if args.modules not in [['stop_network'], ['stop_rebound_network']]:raise ValueError('Stop-phase curriculum requires stop-only update')
        labels=torch.cat([stop_phase_labels(q['context'][:,-3]) for q in seq]).to(device)
        domain_sampler=StopPhaseSampler(replay,labels)
    observation=torch.cat([q['observations'] for q in seq]).to(device)
    reverse_sampler_report = None
    if args.focus_reverse_fraction:
        if args.modules not in [['reverse_network'], ['reverse_update_anchor']] or args.focus_stop_phases:
            raise ValueError('Reverse focus requires reverse-only updates')
        eligible=None
        if args.focus_reverse_deceleration:
            event_labels=torch.cat([step_metadata(q['context'][:,-3]) for q in seq]).to(device)
            eligible=(event_labels[:,0]>0)&(replay.context[:,-3]<0)
        domain_sampler = ActiveReverseSampler(replay, observation, actor.rmuc_physics_adaptation,
                                             args.focus_reverse_fraction, domain_sampler,
                                             anchor=args.modules==['reverse_update_anchor'],
                                             command_range=args.focus_reverse_command_range, eligible=eligible)
        reverse_sampler_report = domain_sampler.report
        (args.output/'reverse_sampling_audit.json').write_text(json.dumps(reverse_sampler_report,indent=2))
    hist=torch.cat([q['sensor_history'] for q in seq]).to(device)
    transition=torch.cat([step_metadata(q['context'][:,-3]) for q in seq]).to(device)
    late=torch.cat([late_phase(q['context'][:,-3]) for q in seq]).to(device)
    # Loader keeps first episode and validates windows; buffers share concatenation order.
    assert len(observation)==len(replay.state)==len(hist)
    names=provenance[0]['extra_state_names'];ix=lambda k:[16+i for i,n in enumerate(names) if n.startswith(k+':')]
    event_conditioned=args.overshoot_weight>0
    critic_mean=torch.cat([wm.mean,wm.context_mean])
    critic_scale=torch.cat([wm.scale,wm.context_scale])
    if event_conditioned:
        critic_mean=torch.cat([critic_mean,torch.zeros(2,device=device)])
        critic_scale=torch.cat([critic_scale,torch.ones(2,device=device)])
    input_mask=None
    if args.mask_constant_critic_inputs:
        input_mask=torch.cat([(replay.state.amax(0)!=replay.state.amin(0)),(replay.context.amax(0)!=replay.context.amin(0))]).float()
        if event_conditioned:input_mask=torch.cat([input_mask,torch.ones(2,device=device)])
    critic=ReturnCritic(critic_mean,critic_scale,input_mask).to(device)
    if args.critic:
        vc=torch.load(args.critic,map_location='cpu',weights_only=False)
        assert vc['state_columns']==ck['state_columns']
        validate_reward_schema(vc,event_conditioned,args.overshoot_weight)
        if vc.get('overshoot_loss','quadratic')!=args.overshoot_loss:raise ValueError('Overshoot loss changed; fresh critic required')
        if vc.get('tracking_version',LEGACY)!=args.tracking_version:raise ValueError('Critic tracking reward changed; initialize fresh critic')
        if args.mask_constant_critic_inputs:
            if 'input_mask' not in vc['model'] or not torch.equal(vc['model']['input_mask'].to(device),input_mask):raise ValueError('Critic mask changed; initialize fresh critic')
        critic.load_state_dict(vc['model'])
    if args.fixed_real_tail:
        tail=torch.load(args.fixed_real_tail,map_location='cpu',weights_only=False)
        if tail.get('target')!='real_policy_finite2s' or args.overshoot_loss!='smooth_linear_tail' or args.overshoot_weight!=4 or args.tracking_version!=LEGACY:raise ValueError('Tail objective mismatch')
        ts=tail['model'];critic=ReturnCritic(ts['mean'],ts['scale'],ts.get('input_mask')).to(device);critic.load_state_dict(ts);critic.eval().requires_grad_(False)
        (args.output/'fixed_tail_contract.json').write_text(json.dumps(dict(sha256=hashlib.sha256(args.fixed_real_tail.read_bytes()).hexdigest(),scope='Frozen source-policy finite2s continuation; local small-step surrogate, not current-policy value certificate',actor_promotion_requires_physics=True),indent=2))
    if not 0<=args.target_critic_tau<=1:raise ValueError('Invalid target tau')
    target_critic=copy.deepcopy(critic).eval().requires_grad_(False) if args.target_critic_tau else None
    actor_opt=torch.optim.Adam(selected,lr=args.actor_lr,betas=(.9,.95))
    critic_opt=torch.optim.Adam(critic.parameters(),lr=args.critic_lr)
    if args.critic and 'optimizer' in vc:
        critic_opt.load_state_dict(vc['optimizer'])
        for group in critic_opt.param_groups:group['lr']=args.critic_lr
    if args.actor_optimizer:
        actor_opt.load_state_dict(torch.load(args.actor_optimizer,map_location='cpu',weights_only=False))
        for group in actor_opt.param_groups:group['lr']=args.actor_lr
    generator=torch.Generator(device=device).manual_seed(414)
    sensor_cfg=json.loads((Path(__file__).resolve().parents[2]/'artifacts/continuation_v552/joint_dr_profile.json').read_text())['overrides']['domain_randomization_sensor_model']
    controls=[2,3,35,0,1,33];cmdscale=torch.tensor([2.1,.5,1.],device=device)
    H=args.horizon;B=args.batch_size
    # Context prefix is mass/inertia(10), followed by sensor and action cycle counts.
    assert (replay.context[:,10]==1).all() and (replay.context[:,11]==0).all()
    source_sha=hashlib.sha256(args.teacher.read_bytes()).hexdigest()
    (args.output/'command.json').write_text(json.dumps(dict(argv=sys.argv,configuration=vars(args)),default=str,indent=2))
    (args.output/'experiment_contract.json').write_text(json.dumps(dict(finite_horizon_only=args.finite_horizon_only,overshoot_loss=args.overshoot_loss,focus_stop_phases=args.focus_stop_phases,source_sha256=source_sha,world_model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),obs=25,action=6,actor_history=1,trainable_modules=args.modules,gamma=args.gamma,lam=args.lam,fresh_sensor_noise=True,project_gate_gradients=args.project_gate_gradients,tracking_version=args.tracking_version,dr_strength=.1,constant_critic_inputs_masked=int((input_mask==0).sum()) if input_mask is not None else 0,delay_domain_counts=domain_sampler.counts if domain_sampler else None,delay_domain_balanced=args.balance_delay_domains,overshoot_weight=args.overshoot_weight,overshoot_margin=.18,retention_teacher=str(args.retention_teacher),critic=('frozen source-policy finite2s continuation' if args.fixed_real_tail else ('none: finiteH32 objective' if args.finite_horizon_only else 'current imagined policy GAE, five epochs, four minibatches')),survival='smooth proxy; physical gates required',provenance=provenance,promoted=False),indent=2))
    start=time.monotonic();history=[]
    frozen_critic={k:v.detach().clone() for k,v in critic.state_dict().items()} if args.frozen_comparison_batches else None
    if args.frozen_comparison_batches:
        assert args.updates==0 and args.critic_warmup_updates==0 and args.critic is not None
        assert not args.project_gate_gradients
    for update in range(args.frozen_comparison_batches or args.critic_warmup_updates+args.updates):
        if not args.frozen_comparison_batches and update==args.critic_warmup_updates and args.critic_warmup_updates>0 and args.updates>0:
            from warmup_guard import check_warmup
            guard=check_warmup(history,args.warmup_max_critic_loss,args.warmup_max_bootstrap)
            (args.output/'warmup_guard.json').write_text(json.dumps(guard,indent=2))
            if not guard['passed']:
                (args.output/'final_report.json').write_text(json.dumps(dict(state='rejected_before_actor_update',actor_updates=0,promoted=False,guard=guard),indent=2))
                raise RuntimeError('Critic warmup rejected before any actor update')
        warmup=bool(args.frozen_comparison_batches) or update<args.critic_warmup_updates
        torch.set_grad_enabled(not warmup)
        # Uniform valid starts across the complete captured command trajectories.
        seqid=torch.randint(len(replay.offsets)-1,(B,),generator=generator,device=device)
        index=replay.offsets[seqid]+(torch.rand(B,generator=generator,device=device)*(replay.offsets[seqid+1]-replay.offsets[seqid])).long()
        t=domain_sampler.sample(B,generator) if domain_sampler else replay.starts[index];past=t[:,None]+replay.past
        states=replay.state[past];ah=replay.action[past];o=observation[t];sensor_hist=hist[t]
        acthist=states[:,-1,ix('action_history')].reshape(B,-1,6)
        ctx=replay.context[t[:,None]+torch.arange(H+1,device=device)]
        default=ctx[:,0,-9:-3]
        gyro_bias=(torch.rand(B,3,device=device)*2-1)*sensor_cfg['gyro_bias_max_rad_s']*.1
        accel_bias=(torch.rand(B,3,device=device)*2-1)*sensor_cfg['accel_bias_max_mps2']*.1
        def critic_features(state,context,event):
            return value_features(state,context,event,event_conditioned)
        rewards=[];continuations=[];features=[critic_features(states[:,-1],ctx[:,0],transition[t])];metrics=[]
        for j in range(H):
            raw=actor({'policy':o},stochastic_output=False)
            applied,acthist=push_action(acthist,execute_authority_action(o,raw),ctx[:,j,11].long())
            nxt=wm(states,ah,raw,ctx[:,j]).clone()
            nxt[:,ix('policy_actions')]=applied;nxt[:,ix('action_history')]=acthist.flatten(1);nxt[:,ix('obs_history')]=sensor_hist.flatten(1)
            r,c,metric=stage_objective(nxt,ctx[:,j],ix,args.tracking_version)
            r=r-.008*args.overshoot_weight*overshoot_fn(metric['vx'],ctx[:,j,-3],transition[t+j])
            rewards.append(r);continuations.append(c);metrics.append(metric)
            root,q,dq,*_=physical(nxt,ix)
            command=torch.stack([ctx[:,j+1,-3],torch.zeros(B,device=device),ctx[:,j+1,-2]],-1)
            fresh=observe(root,q[:,controls],dq[:,controls],default,applied,command,cmdscale,2.,2.)
            fresh=sensor_noise(fresh,torch.full((B,),.1,device=device),gyro_bias,accel_bias,command[:,2],sensor_cfg)
            o,sensor_hist=push_sensor(sensor_hist,fresh,torch.ones(B,dtype=torch.bool,device=device),ctx[:,j+1,10].long(),command[:,2])
            states=torch.cat([states[:,1:],nxt[:,None]],1);ah=torch.cat([ah[:,1:],raw[:,None]],1)
            features.append(critic_features(nxt,ctx[:,j+1],transition[t+j+1]))
        r=torch.stack(rewards);cont=torch.stack(continuations);feat=torch.stack(features)
        critic.requires_grad_(False)
        values=feat.new_zeros(feat.shape[:2]) if args.finite_horizon_only else critic(feat)
        with torch.no_grad():bootstrap_values=target_critic(feat.detach()) if target_critic is not None else values.detach()
        targets=lambda_returns(r,bootstrap_values,cont,args.gamma,args.lam)
        survival=torch.cat([torch.ones_like(cont[:1]),cont.cumprod(0)],0)
        discount=args.gamma**torch.arange(H+1,device=device)
        objective=((r*survival[:-1]*discount[:-1,None]).sum(0)+discount[-1]*survival[-1]*values[-1]).mean()
        ids=torch.randint(len(observation),(2048,),generator=generator,device=device)
        with torch.no_grad():baseline=teacher({'policy':observation[ids]},stochastic_output=False)
        delta=actor({'policy':observation[ids]},stochastic_output=False)-baseline
        retention=delta.square().mean();loss=-objective+args.rehearsal_weight*retention
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite actor objective')
        grad=loss.new_zeros(())
        constraint_report={};constraint_gradients=[];projection_report=None
        if args.project_gate_gradients and not warmup:
            vx=torch.stack([item['vx'] for item in metrics])
            yaw=torch.stack([item['yaw'] for item in metrics])
            active=torch.stack([item['active'] for item in metrics]);hip=torch.stack([item['hip'] for item in metrics])
            desired=ctx[:,:H,-3].T;desired_yaw=ctx[:,:H,-2].T
            event=transition[t[:,None]+torch.arange(H,device=device)].transpose(0,1)
            is_late=late[t[:,None]+torch.arange(H,device=device)].T
            peak=F.relu((vx-desired)*event[...,0]/event[...,1].clamp_min(.025)-.18)*(event[...,1]>0)
            speed_tol=torch.where(desired.abs()<.025,.01,.1*desired.abs()).clamp_min(.01)
            components={'overshoot':peak,'late_speed':F.relu((vx-desired).abs()/speed_tol-1)*is_late,
                'late_yaw':F.relu((yaw-desired_yaw).abs()/.05-1)*is_late,
                'late_active':F.relu(active.abs()/torch.deg2rad(vx.new_tensor(2.))-1)*is_late*(desired_yaw.abs()<.05),
                'late_hip':F.relu(hip.abs()/torch.deg2rad(vx.new_tensor(2.))-1)*is_late*(desired_yaw.abs()<.05)}
            domain=(ctx[:,0,12]>0).long()+2*(ctx[:,0,13]>0).long()
            for group in domain.unique().tolist():
                for name,violations in components.items():
                    cost=upper_quartile(violations[:,domain==group].amax(0))
                    constraint_report[f'{group}:{name}']=float(cost.detach())
                    if float(cost.detach())<=1e-8:continue
                    derivatives=torch.autograd.grad(cost,selected,retain_graph=True,allow_unused=True)
                    constraint_gradients.append(torch.cat([(torch.zeros_like(param) if der is None else der).flatten() for param,der in zip(selected,derivatives)]).detach())
        if not warmup:
            actor_opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(selected,1.)
            for param in selected:
                if param.grad is not None:param.grad.clamp_(-.1,.1)
            if not torch.isfinite(grad):raise RuntimeError('Nonfinite actor gradient')
            if args.project_gate_gradients:
                before_step=torch.cat([p.detach().flatten() for p in selected]).clone()
                optimizer_before=copy.deepcopy(actor_opt.state_dict())
            actor_opt.step()
            if args.project_gate_gradients:
                proposal=torch.cat([p.detach().flatten() for p in selected])-before_step
                gradients=torch.stack(constraint_gradients) if constraint_gradients else proposal.new_empty((0,len(proposal)))
                applied,projection_report=project_update(proposal,gradients)
                with torch.no_grad():
                    offset=0
                    for parameter in selected:
                        n=parameter.numel();parameter.copy_((before_step[offset:offset+n]+applied[offset:offset+n]).reshape_as(parameter));offset+=n
                if not applied.count_nonzero():actor_opt.load_state_dict(optimizer_before)
        torch.set_grad_enabled(True)
        # Targets are frozen from the same pre-update imagined policy. No stale MC labels.
        train_x=feat[:-1].detach().flatten(0,1);train_y=targets.flatten().detach()
        critic.requires_grad_(not bool(args.fixed_real_tail))
        closs=loss.new_zeros(())
        for epoch in range(0 if args.frozen_comparison_batches or args.finite_horizon_only or args.fixed_real_tail else 5):
            perm=torch.randperm(len(train_x),generator=generator,device=device)
            for idx in perm.tensor_split(4):
                prediction=critic(train_x[idx]);closs=(prediction-train_y[idx]).square().mean()
                if not torch.isfinite(closs):raise RuntimeError('Nonfinite critic loss')
                critic_opt.zero_grad();closs.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);critic_opt.step()
        if target_critic is not None and not args.frozen_comparison_batches:
            from target_critic import update_target
            update_target(target_critic,critic,args.target_critic_tau)
        item=dict(update=update+1,local_constraint_values=constraint_report,optimizer_step_projection=projection_report,critic_only_warmup=warmup,actor_loss=float(loss.detach()),return_mean=float(objective.detach()),discounted_stage_return=float((r*survival[:-1]*discount[:-1,None]).sum(0).mean().detach()),bootstrap_return=float((discount[-1]*survival[-1]*values[-1]).mean().detach()),critic_loss=float(closs.detach()),retention_mse=float(retention.detach()),sample_max_action_delta=float(delta.detach().abs().max()),actor_grad_norm=float(grad),survival_mean=float(survival[-1].detach().mean()),seconds=time.monotonic()-start)
        if reverse_sampler_report is not None:
            item['reverse_active_start_fraction'] = domain_sampler.last_active_fraction
        history.append(item);temp=args.output/'progress.tmp';temp.write_text(json.dumps(item));temp.replace(args.output/'progress.json');print(json.dumps(item),flush=True)
        if time.monotonic()-start>args.budget_seconds:break
    torch.set_grad_enabled(True)
    before=source_actor.state_dict();changed=[k for k,v in actor.state_dict().items() if not torch.equal(v,before[k])]
    bad=[k for k in changed if not k.startswith(tuple(allowed))];assert not bad
    if args.updates==0:assert not changed,'Critic-only run modified actor'
    if args.frozen_comparison_batches:
        assert not changed
        assert all(torch.equal(v,frozen_critic[k]) for k,v in critic.state_dict().items())
        result=dict(state='frozen_comparison_complete',actor_changed=False,critic_changed=False,batches=len(history),history=history,source_sha256=source_sha,scope='Common-random-number H32 imagined return; no training, no certification')
        (args.output/'frozen_comparison.json').write_text(json.dumps(result,indent=2));return
    payload=torch.load(args.teacher,map_location='cpu',weights_only=False);payload['actor_state_dict']={k:v.detach().cpu() for k,v in actor.state_dict().items()};payload.pop('optimizer_state_dict',None)
    payload['rmuc_world_model_update']=dict(source_sha256=source_sha,backend='fixed_real2s_tail_bptt' if args.fixed_real_tail else ('finite_horizon_bptt' if args.finite_horizon_only else 'model_based_bptt_gae'),physical_validation_passed=False)
    torch.save(payload,args.output/'candidate.pt')
    torch.save(dict(overshoot_loss=args.overshoot_loss,model=critic.state_dict(),optimizer=critic_opt.state_dict(),gamma=args.gamma,lam=args.lam,event_conditioned=event_conditioned,overshoot_weight=args.overshoot_weight,tracking_version=args.tracking_version,state_columns=ck['state_columns']),args.output/'critic.pt')
    if target_critic is not None:torch.save(dict(model=target_critic.state_dict(),tau=args.target_critic_tau),args.output/'target_critic.pt')
    torch.save(actor_opt.state_dict(),args.output/'actor_optimizer.pt')
    (args.output/'freeze_audit.json').write_text(json.dumps(dict(passed=True,changed_keys=changed,unexpected=bad),indent=2))
    report=dict(state='imagined_updates_complete_physics_pending',updates=len(history),actor_updates=sum(not x['critic_only_warmup'] for x in history),critic_warmup_updates=sum(x['critic_only_warmup'] for x in history),seconds=time.monotonic()-start,history=history,source_sha256=source_sha,candidate_sha256=hashlib.sha256((args.output/'candidate.pt').read_bytes()).hexdigest(),trainable_parameters=sum(p.numel() for p in selected),replay_observations=len(observation),horizon=H,batch_size=B,promoted=False,retention='Recorded-action regularizer only; physical retention pending')
    (args.output/'final_report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
