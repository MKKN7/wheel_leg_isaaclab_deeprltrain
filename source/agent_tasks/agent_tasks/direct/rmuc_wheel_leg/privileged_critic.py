"""Opt-in training-only critic context. Never alters the deployed 25D actor."""
import torch


def reward_state_context(env, like):
    """Read current reward/control memory; no stepping, mutation, or future labels."""
    data = env.robot.data
    position = data.root_pos_w.torch - env.scene.env_origins
    position = position / like.new_tensor([env.cfg.max_xy_distance_m,
                                          env.cfg.max_xy_distance_m, .3])
    command = env._current_command() / like.new_tensor([3.2, 1., 8.])
    memory = torch.stack((env._previous_command_aligned_speed / 3.2,
                          env._previous_filtered_command_target / 3.2,
                          env._previous_heading_speed_for_reward / 3.2,
                          env._previous_reward_direction,
                          env._previous_command_aligned_speed_valid.to(like.dtype)), -1)
    ages = torch.stack((env._curriculum_step_counter,
                        env._acquisition_command_age()), -1).to(like.dtype) * env.step_dt / 20.
    randomization = torch.stack((env._domain_base_mass_scale - 1.,
                                 env._domain_base_inertia_scale - 1.,
                                 env._domain_friction_scale - 1.,
                                 env._domain_wheel_torque_gain - 1.), -1)
    return torch.cat((position, data.root_quat_w.torch, command, memory, ages,
                      env._continuous_command_mask[:, None].to(like.dtype), randomization,
                      env._wheel_mechanical_viscous_per_env / .005,
                      env._wheel_coulomb_per_env / .06,
                      env._wheel_feedback_damping_per_env / .06,
                      env._wheel_feedback_age.value * .05, env._previous_actions,
                      env._base_com_randomizer.offset_flu_m / .01), -1)

def critic_observation(env, delivered, clean):
    q=env._joint_delay_dr
    if q is None:raise ValueError('Privileged delay diagnostic requires joint-delay queues')
    blocks=[delivered,clean]
    ids=env._control_dof_ids;data=env.robot.data
    blocks.extend([data.joint_pos.torch[:,ids],data.joint_vel.torch[:,ids]*.05,
                   data.root_lin_vel_w.torch,data.root_ang_vel_w.torch*.25,
                   data.applied_torque.torch[:,ids]/40.])
    delays=[];valid=[]
    for name in ['observation','action','wheel','leg']:
        x=getattr(q,name);history=x.history
        if history.shape[1]>7:raise ValueError('Context supports at most12ms at2ms ticks')
        padded=history.new_zeros((len(history),7,history.shape[2]));padded[:,:history.shape[1]]=history
        # Wheel/leg histories carry motor effort, not raw actions or leg angles.
        # Wheel is scaled by5.5; leg history remains in Nm in this diagnostic.
        if name=='wheel':padded=padded/5.5
        blocks.append(padded.flatten(1));delays.append(x.steps.to(delivered.dtype)/6.);valid.append(x.valid.to(delivered.dtype))
    blocks.extend([torch.stack(delays,1),torch.stack(valid,1),q.pending])
    for name in ['leg_motors','wheel_motors']:
        response=env._motor_responses[name]
        blocks.extend([response.effort/40.,response.time_constant/.005])
    age=env._wheel_feedback_age
    if age is None:raise ValueError('Wheel feedback age required for this diagnostic')
    tick=int(env._sim_step_counter)
    blocks.append((tick-age.sample_tick).to(delivered.dtype)/6.)
    result=torch.cat(blocks,dim=1)
    mode = getattr(getattr(env, 'cfg', None), 'training_privileged_critic_context', 'queues_v1')
    if mode == 'reward_state_v2':
        # v1 retains historical units. v2 scales leg effort histories to O(1).
        result = result.clone()
        result[:, 305:333] /= 40.
        result = torch.cat((result, reward_state_context(env, delivered)), -1)
        if result.shape[1] != 400:
            raise ValueError('reward_state_v2 requires exactly400 critic inputs')
    elif mode != 'queues_v1':
        raise ValueError(f'Unknown privileged critic context: {mode}')
    if not torch.isfinite(result).all():raise ValueError('Nonfinite privileged critic input')
    return result
