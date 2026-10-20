"""Isolated PPO optimization for the post-clip position authority head."""
import json
import os
from pathlib import Path
import torch

def wheel_advantage_audit(storage):
    """Old-policy Gaussian score for +right/-left wheel mean perturbation.

    This is a local surrogate ascent derivative, not an optimizer/network
    gradient or a counterfactual physical return. Samples are correlated.
    """
    obs=storage.observations['policy'].reshape(-1,25)
    action=storage.actions.reshape(-1,6)
    mean,std=(p.reshape(-1,6) for p in storage.distribution_params)
    advantage=storage.advantages.reshape(-1)
    score=(action-mean)/std.square().clamp_min(1e-12)
    direction_score=score[:,2]-score[:,5]
    speed=obs[:,7]/.5; yaw=obs[:,2]/.25+obs[:,8]
    rows=[]
    targets = [(1., w) for w in (.3,-.3,.8,-.8,1.,-1.,1.2,-1.2,1.4,-1.4)]
    targets += [(.25,w) for w in (.1,-.1,.3,-.3)]
    for target_speed, target in targets:
        mask=((speed-target_speed).abs()<1e-3)&((yaw-target).abs()<1e-3)
        if bool(mask.any()):
            a=advantage[mask];s=direction_score[mask]
            rows.append(dict(vx=target_speed,yaw=target,samples=int(mask.sum()),
                advantage_mean=float(a.mean()),
                wheel_diff_surrogate_ascent=float((a*s).mean()),
                centered_wheel_diff_surrogate_ascent=float(((a-a.mean())*s).mean())))
            # obs[8] is commanded minus measured yaw rate. Classify by
            # magnitude relative to the commanded direction, not world sign.
            shortfall=obs[:,8]*(1. if target>0 else -1.)
            rows[-1]['error_groups']=[]
            for label,subset in (
                ('under_turn',shortfall>.05),
                ('near_target',shortfall.abs()<=.05),
                ('over_turn',shortfall<-.05)):
                selected=mask&subset
                if bool(selected.any()):
                    aa=advantage[selected];ss=direction_score[selected]
                    rows[-1]['error_groups'].append(dict(group=label,samples=int(selected.sum()),
                        mean_shortfall_rad_s=float(shortfall[selected].mean()),
                        wheel_diff_surrogate_ascent=float((aa*ss).mean()),
                        centered_wheel_diff_surrogate_ascent=float(((aa-aa.mean())*ss).mean())))
    return rows

def delivered_command_counts(observations):
    """Count delivered 25D command reconstructions, not sampler stage labels."""
    obs = observations.reshape(-1, 25)
    speed = obs[:, 7] / .5
    yaw = obs[:, 2] / .25 + obs[:, 8]
    selected = (speed - 1.).abs() < 1e-3
    counts = {}
    matched = torch.zeros_like(selected)
    for target in (.3, -.3, .8, -.8, 1., -1., 1.2, -1.2, 1.4, -1.4):
        mask = selected & ((yaw - target).abs() < 1e-3)
        counts[f'1mps_{target:+.1f}radps'] = int(mask.sum())
        matched |= mask
    counts['other'] = int((~matched).sum())
    counts['total'] = obs.shape[0]
    return counts

def prepare_authority_optimizer(runner, state, learning_rate=1e-4, critic_rate=1e-4,
                                local_only=False, local_indices=None, physics_only=False, physics_modules=None,
                                router_speed_direction=None):
    from rmuc_command_adapter import install_rmuc_state_router, install_rmuc_position_authority
    if not 0 < learning_rate <= 1e-3 or not 0 < critic_rate <= 1e-3:
        raise ValueError('Positive bounded learning rates required')
    if runner.alg.schedule != 'fixed' or getattr(runner.alg,'is_multi_gpu',False):
        raise ValueError('Authority training supports single-device fixed schedule')
    actor=runner.alg.get_policy()
    if actor.is_recurrent:raise ValueError('Feedforward authority contract required')
    install_rmuc_state_router(actor,state)
    head=install_rmuc_position_authority(actor,state)
    if head is None:raise ValueError('Explicit authority checkpoint required')
    physics=getattr(actor,'rmuc_physics_adaptation',None)
    if physics_only and (local_only or local_indices is not None or physics is None):
        raise ValueError('Physics optimizer requires an explicit physical head and no local expert selection')
    if local_only and head.local_correction is None:
        raise ValueError('Local optimizer requires an explicit local correction checkpoint')
    if not physics_only and not local_only and router_speed_direction is None and head.local_correction is not None:
        raise ValueError('Local correction requires an isolated local optimizer; shared authority training is forbidden')
    if local_indices is not None and not local_only:
        raise ValueError('Local expert selection requires local_only training')
    if physics_modules is not None and not physics_only:
        raise ValueError('Explicit physics modules require physics_only')
    if physics_modules is not None and os.environ.get('RMUC_PHYSICS_RETENTION')=='1':
        raise ValueError('Legacy outer-head-only retention guard cannot protect selected nested modules')
    indices = None
    if router_speed_direction is not None:
        if physics_only or local_only or physics_modules is not None or local_indices is not None:
            raise ValueError('Speed extension optimizer cannot mix with other selections')
        router=actor.rmuc_state_router
        probe=router.budget.new_zeros(1,25)
        modules=[m for _,m,_ in router.speed_extension_branches(probe,router_speed_direction)]
        actor.rmuc_router_speed_direction=router_speed_direction
    elif physics_only:
        from rmuc_physics_selection import selected_modules
        modules=selected_modules(actor,physics_modules) if physics_modules is not None else [physics.network]
        actor.rmuc_physics_train_modules=tuple(physics_modules or ())
    elif local_only:
        count = len(head.local_correction.experts)
        indices = list(range(count)) if local_indices is None else list(local_indices)
        if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < count for i in indices):
            raise ValueError('Local expert indices must be nonempty, unique, and in range')
        modules = [head.local_correction.experts[i] for i in indices]
    else:
        modules = [head]
    actor.requires_grad_(False)
    for module in modules: module.requires_grad_(True)
    runner.alg.optimizer=torch.optim.Adam([
        {'params':[p for module in modules for p in module.parameters()],'lr':learning_rate,
         'rmuc_local_indices':indices, 'rmuc_physics_modules':physics_modules,
         'rmuc_router_speed_direction':router_speed_direction,
         'rmuc_group':'router_speed_extension' if router_speed_direction else ('physics_adaptation' if physics_only else ('local_authority' if local_only else 'position_authority'))},
        {'params':list(runner.alg.critic.parameters()),'lr':critic_rate,'rmuc_group':'critic'}])

def configure_authority_audit(runner, required_commands=None):
    actor=runner.alg.get_policy();head=actor.rmuc_position_authority
    physics=getattr(actor,'rmuc_physics_adaptation',None)
    physics_training=physics is not None and any(p.requires_grad for p in physics.parameters())
    speed_direction=getattr(actor,'rmuc_router_speed_direction',None)
    frozen={k:p.detach().clone() for k,p in actor.named_parameters() if not p.requires_grad}
    buffers={k:b.detach().clone() for k,b in actor.named_buffers()}
    pending={'observation':None}
    def capture(module,inputs):
        if torch.is_grad_enabled():pending['observation']=inputs[0].detach()
    (actor.rmuc_state_router if speed_direction else (physics if physics_training else head)).register_forward_pre_hook(capture)
    def before_step(optimizer,args,kwargs):
        obs=pending['observation']
        if obs is None:return
        if speed_direction:
            branches=[(m,g) for _,m,g in actor.rmuc_state_router.speed_extension_branches(obs,speed_direction)]
        elif physics_training:
            paths=getattr(actor,'rmuc_physics_train_modules',())
            if paths:
                from rmuc_physics_selection import network_gate
                branches=[(actor.get_submodule(path),network_gate(
                    actor.get_submodule(path.rsplit('.',1)[0]),path.rsplit('.',1)[1],obs)) for path in paths]
            else:
                branches=[(physics,physics.update_gate(obs))]
        elif head.local_correction is not None:
            gates=head.local_correction.gates(obs)
            branches=[(branch,gates[:,i]) for i,branch in enumerate(head.local_correction.experts)]
        else:
            pos,neg=head.gates(obs)
            branches=[(head.positive,pos),(head.negative,neg)]
        for branch,gate in branches:
            if not bool((gate>0).any()):
                for parameter in branch.parameters():parameter.grad=None
    def after_step(optimizer,args,kwargs):
        pending['observation']=None
    runner.alg.optimizer.register_step_pre_hook(before_step)
    runner.alg.optimizer.register_step_post_hook(after_step)
    original=runner.alg.update
    path=Path(runner.logger.log_dir)/'position_authority_training.jsonl'
    from rmuc_training_contract import RolloutCoverage
    coverage = (RolloutCoverage(required_commands, path.with_name('command_coverage.json'))
                if required_commands else None)
    start=int(runner.current_learning_iteration);count=0
    capture_update=int(os.environ.get('RMUC_CAPTURE_PPO_ROLLOUT_UPDATE','1'))
    if capture_update<1:
        raise ValueError('RMUC_CAPTURE_PPO_ROLLOUT_UPDATE must be >= 1')
    capture_labels=[]
    capture_enabled=os.environ.get('RMUC_CAPTURE_PPO_ROLLOUT')=='1'
    tail_start=int(os.environ.get('RMUC_TRAINING_TAIL_START_UPDATE','0'))
    if tail_start < 0:
        raise ValueError('Training tail start must be nonnegative')
    tail_frames=[]
    if tail_start:
        tail_environment=runner.env.unwrapped
        tail_original_act=runner.alg.act
        def record_training_tail(observations):
            if count+1 >= tail_start:
                q=tail_environment.robot.data.joint_pos.torch[:,tail_environment._control_dof_ids]
                cmd=tail_environment._current_command()
                vx=tail_environment._heading_forward_velocity(tail_environment.robot.data.root_lin_vel_w.torch)
                tail_frames.append(torch.cat((
                    tail_environment._curriculum_step_counter.to(q.dtype)[:,None],
                    cmd[:,[0,2]],vx[:,None],q),dim=-1).detach().clone())
            return tail_original_act(observations)
        runner.alg.act=record_training_tail
    capture_only=os.environ.get('RMUC_CAPTURE_PPO_ROLLOUT_ONLY')=='1'
    if capture_only and not capture_enabled:
        raise ValueError('Capture-only requires physical rollout capture enabled')
    if capture_enabled:
        environment=runner.env.unwrapped
        random_schedule_frames=[]
        original_act=runner.alg.act
        previous_command=None
        command_age=None
        def labelled_act(observations):
            nonlocal previous_command,command_age
            if count<capture_update:
                obs=observations['policy']
                episode_age=environment._curriculum_step_counter.detach().clone()
                command=torch.stack((obs[:,7]/.5,obs[:,2]/.25+obs[:,8]),-1)
                if previous_command is None:
                    command_age=torch.zeros_like(episode_age)
                else:
                    same=(command-previous_command).abs().amax(-1)<1e-3
                    command_age=torch.where(same&(episode_age>0),command_age+1,
                                            torch.zeros_like(command_age))
                previous_command=command.detach().clone()
                if count+1==capture_update:
                    schedule=getattr(environment, '_random_speed_sequence', None)
                    if schedule is not None:
                        random_schedule_frames.append(schedule.snapshot())
                    capture_labels.append((obs.detach().cpu().clone(),episode_age.cpu(),
                                           command_age.detach().cpu().clone(),
                                           environment._continuous_command_mask.detach().cpu().clone(),
                                           environment._current_command().detach().cpu().clone()))
            return original_act(observations)
        runner.alg.act=labelled_act
    def update(*args,**kwargs):
        nonlocal count
        if tail_frames:
            import numpy as np
            trace=path.parent/f'training_tail_trace_{start+count}.npz'
            if trace.exists():
                raise RuntimeError(f'Refusing to overwrite training trace: {trace}')
            np.savez_compressed(trace, frames=torch.stack(tail_frames).cpu().numpy(),
                dt=float(tail_environment.step_dt),iteration=start+count,
                columns=['episode_step','command_vx','command_yaw','heading_vx',
                         'q0','q1','q2','q3','q4','q5'],
                scope='Actual stochastic training rollout, pre-action state; no deterministic acceptance implied')
            tail_frames.clear()
        if coverage is not None:
            coverage.observe(runner.alg.storage.observations['policy'])
        # Capture before PPO clears/overwrites the rollout buffer.
        advantage_audit=wheel_advantage_audit(runner.alg.storage) if physics_training or capture_enabled else None
        retention=None
        if physics_training and os.environ.get('RMUC_PHYSICS_RETENTION')=='1':
            from rmuc_physics_retention import PhysicsRetentionStep
            retention=PhysicsRetentionStep(runner)
        if capture_enabled and count+1==capture_update:
            storage=runner.alg.storage
            capture=path.parent/'ppo_gradient_snapshot.pt'
            if capture.exists():
                raise RuntimeError(f'Refusing to overwrite rollout diagnostic: {capture}')
            assert len(capture_labels)==storage.actions.shape[0]
            labelled_obs,episode_age,command_age_labels,continuous_labels,physical_commands=(torch.stack(x) for x in zip(*capture_labels))
            assert torch.equal(labelled_obs,storage.observations['policy'].detach().cpu()),'Rollout labels are misaligned'
            torch.save({
                'observations':storage.observations['policy'].detach().cpu(),
                'actions':storage.actions.detach().cpu(),
                'advantages':storage.advantages.detach().cpu(),
                'distribution_params':tuple(x.detach().cpu() for x in storage.distribution_params),
                'actor_state_dict':{k:v.detach().cpu().clone() for k,v in actor.state_dict().items()},
                'iteration':start+count,
                'capture_update_one_based':capture_update,
                'rollout_control_steps':int(storage.actions.shape[0]),
                'preceding_rollout_control_steps':count*int(storage.actions.shape[0]),
                'steady_samples_certified':False,
                'optimizer_updates_disabled':capture_only,
                'physical_episode_age_steps':episode_age,
                'delivered_command_age_steps':command_age_labels,
                'continuous_cohort_mask':continuous_labels,
                'physical_commands':physical_commands,
                'random_speed_schedule_per_step':random_schedule_frames,
                'training_sequence_enabled':getattr(environment, '_training_sequence_table', None) is not None or getattr(environment, '_random_speed_sequence', None) is not None,
                'age_label_scope':'Pre-action physical episode age and observed command continuity; not a stability certificate',
                'scope':'Pre-update rollout; optional diagnostic only, no optimizer mutation',
            },capture)
            capture_labels.clear()
            random_schedule_frames.clear()
        if capture_only:
            # Keep actor, critic and optimizer fixed while collecting later
            # on-policy data. clear() resets the storage cursor, not tensors.
            runner.alg.storage.clear()
            result={}
        else:
            result=original(*args,**kwargs)
        retention_result=retention.finish() if retention is not None else None
        changed=[k for k,p in actor.named_parameters() if k in frozen and not torch.equal(p,frozen[k])]
        changed_buffers=[k for k,b in actor.named_buffers() if not torch.equal(b,buffers[k])]
        if changed or changed_buffers:raise RuntimeError(f'Frozen authority source changed:{changed+changed_buffers}')
        from rmuc_residual_safety import rollout_probe
        obs=rollout_probe(runner.alg.storage.observations['policy'])
        with torch.no_grad():
            pos,neg=head.gates(obs)
            correction=head.correction(obs)
            if not torch.isfinite(correction).all() or not (correction.abs()<=head.budget+1e-6).all():
                raise RuntimeError('Position authority residual bound violated')
        record={'iteration':start+count,'frozen_changed':changed,'buffers_changed':changed_buffers,
                'positive_samples':int((pos>0).sum()),'negative_samples':int((neg>0).sum()),
                'delivered_command_counts':delivered_command_counts(runner.alg.storage.observations['policy']),
                'residual_abs_max':correction.abs().amax(0).tolist()}
        if advantage_audit is not None:
            record['wheel_advantage_audit']=advantage_audit
        if speed_direction:
            record['speed_extension_samples']={n:int((g>0).sum()) for n,_,g in actor.rmuc_state_router.speed_extension_branches(obs,speed_direction)}
            with torch.no_grad():
                router_delta=actor.rmuc_state_router(obs)
                bound=actor.rmuc_state_router.correction_bound(obs)
                if not torch.isfinite(router_delta).all() or not (router_delta.abs()<=bound+1e-6).all():
                    raise RuntimeError('Speed router correction bound violated')
                if hasattr(actor.rmuc_state_router,'post_authority_anchors'):
                    post=actor.rmuc_state_router.post_authority_correction(obs)
                    limit=actor.rmuc_state_router.post_authority_budget
                    if not torch.isfinite(post).all() or not (post.abs()<=limit+1e-6).all():
                        raise RuntimeError('Post-authority correction bound violated')
                    record['post_authority_residual_abs_max']=post.abs().amax(0).tolist()
        if retention_result is not None:
            record['physics_retention']=retention_result
        if head.local_correction is not None:
            record['local_samples']=(head.local_correction.gates(obs)>0).sum(0).tolist()
        if physics is not None:
            with torch.no_grad():delta=physics.correction(obs)
            if not torch.isfinite(delta).all() or not (delta.abs()<=physics.correction_bound()+1e-6).all():
                raise RuntimeError('Physical adaptation residual bound violated')
            record['physics_samples']=int((physics.update_gate(obs)>0).sum())
            record['physics_residual_abs_max']=delta.abs().amax(0).tolist()
            if hasattr(physics, 'reverse_network'):
                with torch.no_grad():
                    gate = physics.reverse_gate(obs)
                    delta = gate[:, None] * physics.reverse_budget * torch.tanh(physics.reverse_network(obs))
                if not torch.isfinite(delta).all() or not (delta.abs() <= physics.reverse_budget+1e-6).all():
                    raise RuntimeError('Reverse residual bound violated')
                record['reverse_samples'] = int((gate > 0).sum())
                record['reverse_residual_abs_max'] = delta.abs().amax(0).tolist()
            if hasattr(physics, 'alignment_network'):
                with torch.no_grad():
                    gate = physics.alignment_gate(obs)
                    alignment = gate[:, None] * physics.alignment_budget * torch.tanh(physics.alignment_network(obs))
                if not torch.isfinite(alignment).all() or not (alignment.abs() <= physics.alignment_budget+1e-6).all():
                    raise RuntimeError('Alignment residual bound violated')
                record['alignment_samples'] = int((gate > 0).sum())
                record['alignment_residual_abs_max'] = alignment.abs().amax(0).tolist()
            if hasattr(physics, 'straight_extension_network'):
                with torch.no_grad():
                    gate = physics.straight_extension_gate(obs)
                    delta = gate[:, None] * physics.straight_extension_budget * torch.tanh(physics.straight_extension_network(obs))
                if not torch.isfinite(delta).all() or not (delta.abs() <= physics.straight_extension_budget+1e-6).all():
                    raise RuntimeError('Straight extension bound violated')
                record['straight_extension_samples'] = int((gate > 0).sum())
                record['straight_extension_residual_abs_max'] = delta.abs().amax(0).tolist()
        with path.open('a') as f:f.write(json.dumps(record)+'\n')
        count+=1
        return result
    runner.alg.update=update
