import torch,pytest
from rmuc_privileged_critic import expand_critic_state,value_diagnostics

def test_expansion_preserves_original_predictions_and_does_not_mutate_source():
 torch.manual_seed(17);w=torch.randn(8,25);b=torch.randn(8);s={'mlp.0.weight':w,'mlp.0.bias':b};new=expand_critic_state(s,361)
 x=torch.randn(32,361)
 torch.testing.assert_close(x@new['mlp.0.weight'].T+b,x[:,:25]@w.T+b)
 assert w.shape==(8,25) and torch.count_nonzero(new['mlp.0.weight'][:,25:])==0
 with pytest.raises(ValueError):expand_critic_state(s,25)


def test_critic_resume_rejects_unlabelled_or_wrong_semantics():
 from rmuc_privileged_critic import prepare_critic_state
 state={'mlp.0.weight':torch.randn(8,400),'mlp.0.bias':torch.randn(8)}
 payload={'critic_state_dict':state}
 with pytest.raises(ValueError):prepare_critic_state(payload,400,'reward_state_v2')
 payload['infos']={'rmuc_critic_context_contract':{'dimension':400,'version':'reward_state_v2'}}
 new=prepare_critic_state(payload,400,'reward_state_v2')
 assert torch.equal(new['mlp.0.weight'],state['mlp.0.weight'])
 assert new['mlp.0.weight'].data_ptr()!=state['mlp.0.weight'].data_ptr()
 with pytest.raises(ValueError):prepare_critic_state(payload,400,'queues_v1')

def test_delay_diagnostics_constant_returns_not_false_ev():
 v=torch.tensor([0.,1.,2.,3.]);y=v.clone();a=torch.tensor([-.5,.5,-1.,1.]);s=torch.tensor([0,1,3,4]);r=value_diagnostics(v,y,a,s)
 assert r['groups']['all']['explained_variance']==1
 assert r['groups']['high']['n']==2
 assert value_diagnostics(v,torch.ones_like(v),a,s)['groups']['all']['explained_variance'] is None
 with pytest.raises(ValueError):value_diagnostics(v,y,a,s[:2])

def test_context_has_no_actor_mutation_or_randomness():
 import runpy
 from types import SimpleNamespace as N
 from pathlib import Path
 fn=runpy.run_path(str(Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/privileged_critic.py'))['critic_observation']
 n=3;wrap=lambda x:N(torch=x)
 def q(width):return N(history=torch.zeros(n,5,width),steps=torch.ones(n,dtype=torch.long),valid=torch.ones(n,dtype=torch.bool))
 env=N(_joint_delay_dr=N(observation=q(25),action=q(6),wheel=q(2),leg=q(4),pending=torch.zeros(n,6)),_control_dof_ids=list(range(6)),robot=N(data=N(joint_pos=wrap(torch.zeros(n,6)),joint_vel=wrap(torch.zeros(n,6)),root_lin_vel_w=wrap(torch.zeros(n,3)),root_ang_vel_w=wrap(torch.zeros(n,3)),applied_torque=wrap(torch.zeros(n,6)))),_motor_responses={name:N(effort=torch.zeros(n,d),time_constant=torch.ones(n,d)*.002) for name,d in [('leg_motors',4),('wheel_motors',2)]},_wheel_feedback_age=N(sample_tick=torch.ones(n,2,dtype=torch.long)),_sim_step_counter=3)
 actor=torch.randn(n,25);clean=torch.randn(n,25);before=actor.clone();rng=torch.get_rng_state().clone();out=fn(env,actor,clean)
 assert out.shape==(n,361);assert torch.equal(actor,before);assert torch.equal(out[:,:25],actor);assert torch.equal(rng,torch.get_rng_state())

def test_per_command_diagnostics_do_not_hide_bad_local_fit_in_global_variance():
 y=torch.tensor([0.,1.,2.,3.,100.,101.,102.,103.]);v=y+torch.tensor([3.,-3.,3.,-3.,3.,-3.,3.,-3.]);obs=torch.zeros(8,25);obs[4:,7]=.25
 r=value_diagnostics(v,y,y-v,torch.zeros(8,dtype=torch.long),obs,[[0.,0.],[.5,0.]])
 assert r['groups']['all']['explained_variance']>.99
 assert r['groups']['command_vx0_yaw0']['explained_variance']<0
 assert r['groups']['command_vx0.5_yaw0']['value_residual_rmse']==3
 assert r['groups']['command_vx0.5_yaw0_low']['n']==4


def test_reward_memory_and_position_distinguish_aliases_without_mutation():
 import runpy
 from pathlib import Path
 from types import SimpleNamespace as N
 fn=runpy.run_path(str(Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/privileged_critic.py'))['reward_state_context']
 n=2;z=torch.zeros(n);wrap=lambda x:N(torch=x)
 env=N(robot=N(data=N(root_pos_w=wrap(torch.zeros(n,3)),root_quat_w=wrap(torch.tensor([[1.,0,0,0]]).repeat(n,1)))),scene=N(env_origins=torch.zeros(n,3)),cfg=N(max_xy_distance_m=40),_current_command=lambda:torch.zeros(n,3),_previous_command_aligned_speed=z.clone(),_previous_filtered_command_target=z.clone(),_previous_heading_speed_for_reward=z.clone(),_previous_reward_direction=torch.ones(n),_previous_command_aligned_speed_valid=torch.ones(n,dtype=torch.bool),_curriculum_step_counter=torch.zeros(n,dtype=torch.long),_acquisition_command_age=lambda:torch.zeros(n,dtype=torch.long),step_dt=.008,_continuous_command_mask=torch.zeros(n,dtype=torch.bool),_domain_base_mass_scale=torch.ones(n),_domain_base_inertia_scale=torch.ones(n),_domain_friction_scale=torch.ones(n),_domain_wheel_torque_gain=torch.ones(n),_wheel_mechanical_viscous_per_env=torch.ones(n,2)*.002,_wheel_coulomb_per_env=torch.ones(n,2)*.05,_wheel_feedback_damping_per_env=torch.ones(n,2)*.05,_wheel_feedback_age=N(value=torch.zeros(n,2)),_previous_actions=torch.zeros(n,6),_base_com_randomizer=N(offset_flu_m=torch.zeros(n,3)))
 actor=torch.zeros(n,25);before=actor.clone();rng=torch.get_rng_state().clone()
 a=fn(env,actor);assert a.shape==(n,39);assert torch.equal(a[0],a[1])
 env._previous_command_aligned_speed[1]=.2;env.robot.data.root_pos_w.torch[1,0]=39.
 env._previous_actions[1,3]=.7
 b=fn(env,actor)
 assert torch.equal(a[0],b[0]);assert not torch.equal(a[1],b[1])
 assert b[1,0]==.975
 assert torch.equal(actor,before) and torch.equal(torch.get_rng_state(),rng)
 assert env._previous_command_aligned_speed[1]==.2
