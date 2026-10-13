from types import SimpleNamespace as NS
import pytest
import torch


def test_scoped_physics_preflight_matches_gradient_support_without_rng_change():
    from rmuc_physics_adaptation import PhysicsAdaptation
    from rmuc_training_contract import validate_trainable_target_gate
    head = PhysicsAdaptation(PhysicsAdaptation(), update_domain=[1.5,1.7,.45,.6],
        scoped_network_domain=[.3,.45,.55,.75,.02,.1],
        positive_update_domain=[.3,.5,1.5,1.7,.45,.6])
    state = {'rmuc_physics_adaptation.'+k:v for k,v in head.state_dict().items()}
    rng = torch.get_rng_state().clone()
    for vx, yaw in [(1.,0.), (1.,.5), (1.,.8), (.5,0.), (0.,0.)]:
        obs = torch.zeros(2,25)
        obs[:,7] = vx*.5
        obs[:,8] = torch.tensor([yaw,-yaw])
        expected = head.update_gate(obs)*head.positive_gate(obs)*(1-head.scoped_gate(obs))
        if (expected > 0).any():
            result = validate_trainable_target_gate(state,vx,yaw,physics_only=True)
            torch.testing.assert_close(torch.tensor(result['gates']), expected[:len(result['gates'])])
        else:
            with pytest.raises(ValueError, match='gate is zero'):
                validate_trainable_target_gate(state,vx,yaw,physics_only=True)
    assert torch.equal(rng,torch.get_rng_state())


def cfg(priority=(48, 49, 50, 59), weights=(1., 1., 1., 1.)):
    yaws = [0.] * 61
    for index, value in ((48, .6), (49, .8), (50, 1.), (59, .8), (60, 1.)):
        yaws[index] = value
    return NS(command_start_level=60, command_rehearsal_enabled=True,
              command_rehearsal_fraction=.75, command_rehearsal_priority_fraction=.85,
              command_rehearsal_anchor_stage_ids=(), command_rehearsal_priority_stage_ids=priority,
              command_rehearsal_priority_stage_weights=weights,
              command_rehearsal_gate_stage_ids=priority,
              command_stage_linear_x_mps=[1.] * 61,
              command_stage_yaw_rate_rad_s=yaws,
              command_yaw_observation_error_clip=6.)


def test_required_training_commands_rejects_mismatched_priority_lists():
    from rmuc_training_contract import required_training_commands
    with pytest.raises(ValueError, match='match'):
        required_training_commands(cfg(priority=(48,), weights=(1., 1.)))


def test_highspeed_guard_preserves_highspeed_but_excludes_extended_slow_turn():
    from rmuc_training_contract import highspeed_guard_applies
    speeds = [1.] * 60
    speeds[9] = 1.2
    speeds[20] = 2.6
    c = NS(command_start_level=49, command_high_speed_gate_start_level=9,
           command_stage_linear_x_mps=speeds)
    assert not highspeed_guard_applies(c)
    for level in (9, 20):
        c.command_start_level = level
        assert highspeed_guard_applies(c)


def test_required_training_commands_returns_both_turn_signs():
    from rmuc_training_contract import required_training_commands
    commands = required_training_commands(cfg())
    assert (1., .6) in commands and (1., -.6) in commands
    assert (1., 1.) in commands and (1., -1.) in commands


def test_rehearsal_below_start_level_is_not_reachable():
    from rmuc_training_contract import required_training_commands
    c=cfg();c.command_rehearsal_start_level=61
    with pytest.raises(ValueError, match='absent'):
        required_training_commands(c)


def test_v469_inactive_local_target_is_rejected_but_physics_target_is_supported():
    from rmuc_training_contract import validate_trainable_target_gate
    state={'rmuc_position_authority.local_correction.centers':torch.tensor([-.8,-1.,1.2,1.4]),
           'rmuc_position_authority.local_correction.half_width':torch.tensor(.15),
           'rmuc_physics_adaptation.network.0.weight':torch.zeros(64,25)}
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='zero'):
        validate_trainable_target_gate(state,1.,.3,local_indices=[3])
    assert validate_trainable_target_gate(state,1.,.3,physics_only=True)['gates']==[1.,1.]
    assert validate_trainable_target_gate(state,0.,4.,physics_only=True)['gates']==[1.,1.]
    assert validate_trainable_target_gate(state,1.,1.4,local_indices=[3])['passed']
    assert torch.equal(rng,torch.get_rng_state())


def test_rollout_coverage_fails_closed_after_grace(tmp_path):
    from rmuc_training_contract import RolloutCoverage
    coverage = RolloutCoverage([(1., .8)], tmp_path / 'coverage.json', grace_updates=2)
    obs = torch.zeros(4, 25)
    coverage.observe(obs)
    with pytest.raises(RuntimeError, match='missing'):
        coverage.observe(obs)


def test_observation_contract_requires_explicit_yaw_capacity():
    from rmuc_training_contract import validate_observation_contract
    value = dict(schema_version=1, observation_dim=25,
                 command_observation_mode='tracking_error_with_target',
                 command_observation_scale=[2.1, .5, 1.],
                 command_observation_error_clip=2., command_yaw_observation_error_clip=6.)
    assert validate_observation_contract(value)['command_yaw_observation_error_clip'] == 6.
    value['command_yaw_observation_error_clip'] = 1.9
    with pytest.raises(ValueError, match='>=2'):
        validate_observation_contract(value)
