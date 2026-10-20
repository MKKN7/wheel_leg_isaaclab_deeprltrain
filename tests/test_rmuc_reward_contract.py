"""Execute the actual reward computation on identical states across cohorts."""
import ast
from collections import defaultdict
import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
bank_reference = runpy.run_path(str(ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/bank_geometry.py'))['bank_reference']
tracking_targets = runpy.run_path(str(ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/low_speed_tracking.py'))['tracking_targets']


def reward_function():
    path = ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RmucWheelLegEnv')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_rewards')
    stop = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'clipped_reward' for t in n.targets))
    fn.body = fn.body[:stop + 1] + ast.parse('return reward, command_pitch_tracking_score, command_filtered_speed_error_score').body
    namespace = {'torch': torch, 'math': math, 'bank_reference': bank_reference,
                 'tracking_targets': tracking_targets, 'rebase_aligned_history': runpy.run_path(str(ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/low_speed_tracking.py'))['rebase_aligned_history']}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['_get_rewards']


def make_env():
    # Read current source config, without importing/starting Isaac Sim.
    tree = ast.parse((ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env_cfg.py').read_text())
    config = yaml.safe_load((ROOT/'configs/rmuc_wheel_leg.yaml').read_text())
    namespace = {'float': float, 'int': int, 'bool': bool, 'tuple': tuple,
                 '_REWARD': config['reward'], '_COMMAND': config['command'],
                 '_CONTROL': config['control'],
                 '_EXPLOSION': config['safety']['estop']['position_explosion'],
                 '_TORQUE': config['safety']['torque'], '_ESTOP': config['safety']['estop'],
                 '_ENV': config['environment'], '_SAFETY': config['safety'], '_ROBOT': config['robot']}
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RmucWheelLegCourseEnvCfg')
    for statement in cls.body:
        if isinstance(statement, ast.Assign):
            try:
                exec(compile(ast.Module(body=[statement], type_ignores=[]), '<cfg>', 'exec'), namespace)
            except (NameError, TypeError, KeyError, AttributeError):
                pass
    cfg = SimpleNamespace(**{k: v for k, v in namespace.items() if not k.startswith('_')})
    cfg.command_require_analytic_bank = True  # Historical reward tests explicitly opt in.
    cfg.curriculum_stage = 1
    cfg.command_stage_linear_x_mps = tuple(x['linear_x_mps'] for x in config['command']['stages'])
    cfg.sim = SimpleNamespace(dt=.002)
    cfg.decimation = 4
    cfg.command_active_bank_reward_scale = 2.
    cfg.command_turn_yaw_tracking_start_level = 26
    count = 6
    command = torch.tensor([[1.5, 0, .3], [2., 0, -.3], [2.2, 0, .3],
                            [2.2, 0, -.3], [2.9, 0, 0], [2.5, 0, -.15]])
    velocity = command.clone()
    velocity[:, 0] -= .17
    velocity[:, 1:] = 0
    angular = torch.zeros(count, 3)
    angular[:, 1] = .7
    angular[:, 2] = command[:, 2] + .06
    gravity = torch.tensor([-math.sin(math.radians(10)), 0., -math.cos(math.radians(10))]).repeat(count, 1)
    zeros = torch.zeros(count)
    data = SimpleNamespace(root_lin_vel_w=SimpleNamespace(torch=velocity),
                           root_pos_w=SimpleNamespace(torch=torch.tensor([0., 0., .3]).repeat(count, 1)),
                           applied_torque=SimpleNamespace(torch=torch.zeros(count, 6)))
    env = SimpleNamespace(cfg=cfg, _command_level=26, _state=lambda: (gravity, velocity, angular),
        _current_command=lambda: command, _heading_forward_velocity=lambda value: value[:, 0],
        _stand_metrics=lambda _: defaultdict(lambda: torch.ones(count)), robot=SimpleNamespace(data=data),
        _policy_actions=torch.zeros(count, 6), _previous_actions=torch.zeros(count, 6),
        _control_dof_ids=list(range(6)), _rehearsal_mask=torch.zeros(count, dtype=torch.bool),
        _previous_command_aligned_speed=velocity[:, 0] - .03,
        _previous_filtered_command_target=torch.zeros(count),
        _previous_reward_direction=torch.ones(count),
        _previous_command_aligned_speed_valid=torch.ones(count, dtype=torch.bool),
        _curriculum_step_counter=torch.full((count,), 1300),
        _last_termination_reasons={'tilt': zeros.bool()})
    return env


def test_same_command_state_has_same_reward_across_cohort_and_global_level():
    env, reward = make_env(), reward_function()
    env.cfg.command_reward_contract_by_command = True
    active = reward(env)
    env._rehearsal_mask.fill_(True)
    rehearsal = reward(env)
    env._command_level = 31
    later_level = reward(env)
    for expected, replay, later in zip(active, rehearsal, later_level):
        torch.testing.assert_close(expected, replay, rtol=0, atol=0)
        torch.testing.assert_close(expected, later, rtol=0, atol=0)
    assert active[1][0] == 0  # No high-speed pitch penalty at 1.5 m/s.
    assert active[1][-2] > 0  # 2.9 m/s replay retains pitch shaping.
    assert active[2][-2] > 0  # Retained straight speed error is no longer masked.


def test_legacy_reward_mismatch_is_reproduced():
    env, reward = make_env(), reward_function()
    env.cfg.command_reward_contract_by_command = False
    active = reward(env)[0]
    env._rehearsal_mask.fill_(True)
    rehearsal = reward(env)[0]
    assert not torch.allclose(active, rehearsal)


def test_tracking_error_tail_ranks_large_speed_and_yaw_misses():
    env, reward = make_env(), reward_function()
    env.cfg.command_reward_contract_by_command = True
    env.cfg.command_require_analytic_bank = False
    env.cfg.command_tracking_error_tail_enabled = True
    command = env._current_command()
    velocity = env.robot.data.root_lin_vel_w.torch
    angular = env._state()[2]
    command[:, 0] = 1.0
    command[:, 2] = 0.8
    velocity[:, 0] = 0.2
    angular[:, 2] = -0.4
    far = reward(env)[0].clone()
    velocity[:, 0] = 0.4
    angular[:, 2] = 0.0
    near = reward(env)[0].clone()
    assert torch.all(near > far)
    env.cfg.command_tracking_error_tail_enabled = False
    # The new term is opt-in; the legacy score has a flat tail on both axes.
    legacy_near = reward(env)[0].clone()
    velocity[:, 0] = 0.2
    angular[:, 2] = -0.4
    legacy_far = reward(env)[0].clone()
    assert torch.all((near - far) > (legacy_near - legacy_far))


def test_reward_ranking_survives_near_yaw_gate():
    env, reward = make_env(), reward_function()
    env.cfg.command_reward_contract_by_command = True
    gravity, velocity, angular = env._state()
    command = env._current_command()
    pitch = torch.deg2rad(command[:, 0] * env.cfg.command_pitch_target_deg_per_mps)
    roll = torch.atan(command[:, 0] * command[:, 2] / 9.81)
    gravity[:, 0] = torch.sin(pitch)
    gravity[:, 1] = -torch.sin(roll) * torch.cos(pitch)
    gravity[:, 2] = -torch.cos(roll) * torch.cos(pitch)
    angular[:, 1] = 0
    velocity[:, 0] = command[:, 0] - .04
    env._previous_command_aligned_speed.copy_(velocity[:, 0])
    angular[:, 2] = command[:, 2]
    accurate = reward(env)[0][3]
    angular[:, 2] = command[:, 2] + .06
    outside_gate = reward(env)[0][3]
    assert accurate > outside_gate > 10
    assert accurate.clamp(max=10) == outside_gate.clamp(max=10)
    assert accurate.clamp(max=env.cfg.reward_clip_max) > outside_gate.clamp(max=env.cfg.reward_clip_max)
    c = env.cfg
    upper_bound = (c.upright_weight + c.velocity_weight + c.yaw_weight + c.stillness_weight
                   + c.command_forward_weight + c.command_progress_weight
                   + c.command_high_speed_exploration_weight + c.command_speed_acquisition_weight
                   + c.command_yaw_weight + max(1., c.command_active_bank_reward_scale) * c.command_bank_tracking_weight
                   + c.command_survival_bonus + c.command_posture_weight * (
                       c.command_stand_shape_weight * (1 + max(1., c.command_turn_attitude_scale))
                       + c.command_contact_weight))
    assert c.reward_clip_max >= upper_bound


def test_spin_overspeed_penalty_respects_command_when_opted_in():
    path=ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RmucWheelLegEnv')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_get_rewards')
    assignments=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'yaw_overspeed_limit','excessive_yaw'} for t in n.targets)]
    assignments.sort(key=lambda n:n.lineno)
    code=compile(ast.Module(body=assignments,type_ignores=[]),'<actual yaw penalty>','exec')
    command=torch.tensor([[0.,0.,3.],[0.,0.,-3.],[2.,0.,.3]])
    angular=torch.tensor([[0.,0.,3.],[0.,0.,-3.],[0.,0.,1.2]])
    env=SimpleNamespace(cfg=SimpleNamespace(command_yaw_overspeed_relative_to_target=False))
    scope={'torch':torch,'self':env,'command':command,'angular_velocity':angular,
           'spin_mask':torch.zeros(3,dtype=torch.bool),'spin_excess':torch.zeros(3)}
    exec(code,scope);legacy=scope['excessive_yaw'].clone()
    env.cfg.command_yaw_overspeed_relative_to_target=True;exec(code,scope)
    torch.testing.assert_close(legacy,torch.tensor([2.,2.,.2]))
    torch.testing.assert_close(scope['excessive_yaw'],torch.tensor([0.,0.,.2]))


def test_spin_deadband_and_strong_underspeed_in_actual_reward():
    env=make_env();fn=reward_function();env.cfg.spin_asymmetric_reward_enabled=True
    g,v,w=env._state();cmd=env._current_command();cmd[:,:2]=0;v[:]=0;w[:]=0
    cmd[:,2]=torch.tensor([3.,-3.,10.,-10.,3.,-3.])
    outputs=[]
    for error in [0.,.25,.5,.6,-.5]:
        w[:,2]=torch.sign(cmd[:,2])*(cmd[:,2].abs()+error)
        outputs.append(fn(env)[0].clone())
    torch.testing.assert_close(outputs[0],outputs[1],atol=0,rtol=0)
    torch.testing.assert_close(outputs[0],outputs[2],atol=0,rtol=0)
    assert torch.all(outputs[3]<outputs[0])
    assert torch.all(outputs[0]-outputs[4]>=env.cfg.spin_underspeed_penalty_weight)
    w[:,2]=-cmd[:,2]
    assert torch.all(fn(env)[0]<outputs[0])


def test_spin_option_preserves_moving_turn_rewards():
    env=make_env();fn=reward_function();before=fn(env)[0]
    env.cfg.spin_asymmetric_reward_enabled=True
    torch.testing.assert_close(fn(env)[0],before,atol=0,rtol=0)


def test_early_large_turn_pitch_signal_preserves_straight_and_normal_lean():
    env=make_env();fn=reward_function();env.cfg.command_reward_contract_by_command=True
    gravity,velocity,angular=env._state();cmd=env._current_command()
    cmd[:,0]=2.4;cmd[:,2]=torch.tensor([.3,-.3,0.,.15,.3,-.3])
    velocity[:,0]=2.4;env._previous_command_aligned_speed.fill_(2.4)
    pitch=torch.deg2rad(torch.tensor([10.,10.,10.,10.,6.,6.]))
    gravity[:,0]=torch.sin(pitch);gravity[:,1]=0;gravity[:,2]=-torch.cos(pitch)
    angular[:,1]=0;angular[:,2]=cmd[:,2]
    before=fn(env)
    env.cfg.command_large_turn_pitch_tolerance_deg=4.
    env.cfg.command_large_turn_pitch_scale_deg=8.
    after=fn(env)
    assert torch.all(before[1][:2]==0)
    torch.testing.assert_close(after[1][:2],torch.full((2,),.3))
    torch.testing.assert_close(before[0][2:],after[0][2:],atol=0,rtol=0)
    torch.testing.assert_close(before[0][:2]-after[0][:2],torch.full((2,),.6),atol=2e-6,rtol=0)
    env._rehearsal_mask.fill_(True);env._command_level=31
    torch.testing.assert_close(fn(env)[0],after[0],atol=0,rtol=0)


def test_locomotion_has_no_standing_height_attraction_but_keeps_height_floor():
    env=make_env();fn=reward_function();env.cfg.command_reward_contract_by_command=True
    scores={'height_score':torch.zeros(6),'leg_length_score':torch.full((6,),.7),
            'attitude_score':torch.ones(6),'contact_score':torch.ones(6)}
    env._stand_metrics=lambda _:defaultdict(lambda:torch.ones(6),scores)
    reference=fn(env)[0].clone()
    # Change only the obsolete standing-height preference, holding physics fixed.
    scores['height_score']=torch.ones(6)
    torch.testing.assert_close(fn(env)[0],reference,atol=0,rtol=0)
    # Across allowed heights the locomotion reward has no target-height optimum.
    for height in [.15,.2,.3,.4]:
        env.robot.data.root_pos_w.torch[:,2]=height
        torch.testing.assert_close(fn(env)[0],reference,atol=0,rtol=0)
    env.robot.data.root_pos_w.torch[:,2]=.10
    below=fn(env)[0].clone()
    weight=env.cfg.command_height_penalty_weight
    env.cfg.command_height_penalty_weight=0.
    without_floor_penalty=fn(env)[0]
    expected=weight*((env.cfg.command_min_height_m-.10)/.12)**2
    torch.testing.assert_close(without_floor_penalty-below,torch.full((6,),expected),atol=2e-6,rtol=0)


def test_disable_locomotion_leg_preference_keeps_contact_and_attitude():
    env=make_env();fn=reward_function();env.cfg.command_reward_contract_by_command=True
    assert env.cfg.command_leg_length_preference_scale == 0.0
    env.cfg.command_leg_length_preference_scale=1.0  # Explicit legacy comparison.
    scores={'height_score':torch.ones(6),'leg_length_score':torch.full((6,),.5),
            'attitude_score':torch.ones(6),'contact_score':torch.ones(6)}
    env._stand_metrics=lambda _:defaultdict(lambda:torch.ones(6),scores)
    original=fn(env)[0].clone();env.cfg.command_leg_length_preference_scale=0.
    freed=fn(env)[0].clone()
    expected=.5*.5*env.cfg.command_stand_shape_weight*env.cfg.command_posture_weight
    torch.testing.assert_close(original-freed,torch.full((6,),expected),atol=2e-6,rtol=0)
    scores['leg_length_score']=torch.zeros(6)
    torch.testing.assert_close(fn(env)[0],freed,atol=0,rtol=0)
    scores['contact_score']=torch.zeros(6)
    without_contact=fn(env)[0]
    torch.testing.assert_close(freed-without_contact,torch.full((6,),env.cfg.command_contact_weight*env.cfg.command_posture_weight),atol=2e-6,rtol=0)
    scores['attitude_score']=torch.zeros(6)
    assert torch.all(fn(env)[0]<without_contact)
    env._rehearsal_mask.fill_(True);env._command_level=31
    replay=fn(env)[0];env._rehearsal_mask.fill_(False)
    torch.testing.assert_close(fn(env)[0],replay,atol=0,rtol=0)


def test_low_speed_turn_constraint_preserves_protected_commands_and_cohort_parity():
    env, reward = make_env(), reward_function()
    env.cfg.command_reward_contract_by_command = True
    cmd = env._current_command()
    g, v, w = env._state()
    cmd[:] = torch.tensor([[1.,0.,.45],[1.,0.,-.45],[.8,0.,1.],
                           [1.2,0.,1.],[2.5,0.,1.],[1.,0.,0.]])
    v.zero_(); v[:,0] = cmd[:,0]-.1
    w.zero_(); w[:,2] = cmd[:,2]-.2
    env.cfg.command_low_speed_turn_yaw_tracking_enabled = False
    before = reward(env)[0].clone()
    env.cfg.command_low_speed_turn_yaw_tracking_enabled = True
    torch.testing.assert_close(reward(env)[0], before, rtol=0, atol=0)
    cmd[:] = torch.tensor([1.,0.,1.2]); cmd[1::2,2] *= -1
    active = reward(env)[0].clone()
    env._rehearsal_mask.fill_(True)
    torch.testing.assert_close(reward(env)[0], active, rtol=0, atol=0)


def test_low_speed_turn_rejects_speed_gain_by_losing_yaw_both_directions():
    env, reward = make_env(), reward_function()
    env.cfg.command_reward_contract_by_command = True
    env.cfg.command_low_speed_turn_yaw_tracking_enabled = True
    cmd = env._current_command(); g,v,w = env._state()
    cmd[:] = torch.tensor([1.,0.,1.2]); cmd[1::2,2] *= -1
    sign = torch.sign(cmd[:,2])
    angle = -torch.atan(cmd[:,2]/9.81)
    g[:,0]=0.; g[:,1]=-torch.sin(angle); g[:,2]=-torch.cos(angle)
    w.zero_(); v.zero_()
    v[:,0]=.8; w[:,2]=cmd[:,2]-.05*sign
    env._previous_command_aligned_speed.copy_(v[:,0])
    tracking = reward(env)[0].clone()
    v[:,0]=1.; w[:,2]=cmd[:,2]-.45*sign
    env._previous_command_aligned_speed.copy_(v[:,0])
    sacrificed = reward(env)[0].clone()
    assert (tracking > sacrificed).all()


def test_operator_tracking_reward_has_no_analytic_bank_shaping():
    env,reward=make_env(),reward_function()
    env.cfg.command_require_analytic_bank=False
    original=reward(env)[0].clone()
    env.cfg.command_bank_tracking_weight=12345.
    env.cfg.command_bank_error_penalty_weight=98765.
    torch.testing.assert_close(reward(env)[0],original,rtol=0,atol=0)
    env.cfg.command_require_analytic_bank=True
    assert not torch.equal(reward(env)[0],original)


def test_exact_low_speed_fix_preserves_actual_regular_reward():
    env, reward = make_env(), reward_function()
    env.cfg.command_reward_contract_by_command = True
    before = reward(env)
    env.cfg.command_exact_low_speed_reward_enabled = True
    after = reward(env)
    assert all(torch.equal(a,b) for a,b in zip(before,after))


def test_actual_filtered_reward_sees_idle_drift_and_creep_target():
    env, reward = make_env(), reward_function()
    env.cfg.command_reward_contract_by_command = True
    env.cfg.command_speed_error_margin_mps = .02
    command=env._current_command();command.zero_()
    command[:,0]=torch.tensor([0.,0.,0.,.05,-.05,.1])
    gravity,velocity,angular=env._state();angular.zero_()
    velocity.zero_();velocity[:,0]=torch.tensor([0.,-.1,.1,.05,-.05,.1])
    env._previous_command_aligned_speed_valid.zero_()
    legacy=reward(env)[2]
    env.cfg.command_exact_low_speed_reward_enabled = True
    fixed=reward(env)[2]
    assert legacy[0]>0 and legacy[3]>0  # old .1 target leaked into reference
    assert fixed[0]==0 and (fixed[3:]==0).all()
    assert fixed[1]==fixed[2] and fixed[1]>.9
    assert torch.isfinite(reward(env)[0]).all()


def test_idle_micro_speed_penalty_actual_reward_is_symmetric_and_local():
    env, reward = make_env(), reward_function()
    env.cfg.command_reward_contract_by_command = True
    env.cfg.command_exact_low_speed_reward_enabled = True
    command = env._current_command(); command.zero_()
    command[4, 0] = .05
    command[5, 2] = 1.
    _, velocity, _ = env._state()
    velocity.zero_(); velocity[:, 0] = torch.tensor([0., .0025, .005, -.005, .005, .005])
    baseline = reward(env)[0]
    env.cfg.command_idle_micro_speed_penalty_weight = 1.
    changed = reward(env)[0]
    torch.testing.assert_close(baseline-changed, torch.tensor([0., .2, .5, .5, 0., 0.]), atol=1e-5, rtol=0)
    assert torch.equal(baseline[4:], changed[4:])
    env._rehearsal_mask.fill_(True)
    assert torch.equal(changed, reward(env)[0])


def test_idle_micro_speed_opt_in_preserves_regular_reward_and_validates_scale():
    import pytest
    env, reward = make_env(), reward_function()
    baseline = reward(env)[0]
    env.cfg.command_idle_micro_speed_penalty_weight = 1.
    assert torch.equal(baseline, reward(env)[0])
    for scale in (0., -1., float('nan'), float('inf')):
        env.cfg.command_idle_micro_speed_scale_mps = scale
        with pytest.raises(ValueError, match='Idle micro-speed'):
            reward(env)
