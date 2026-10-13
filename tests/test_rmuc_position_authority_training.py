"""Optimizer selection excludes frozen source and keeps both new directions."""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_position_authority import PositionAuthorityHead
from rmuc_position_authority_training import prepare_authority_optimizer

def test_slow_turn_audit_reconstructs_both_commands_and_score():
    from rmuc_position_authority_training import wheel_advantage_audit, delivered_command_counts
    obs = torch.zeros(1, 4, 25)
    obs[..., 7] = .5
    obs[..., 2] = .025  # Measured yaw .1; observation8 is the error, not command.
    obs[..., 8] = torch.tensor([.2, .2, -.4, -.4])
    action = torch.zeros(1, 4, 6)
    action[..., 2] = torch.tensor([1., -1., -1., 1.])
    storage = SimpleNamespace(observations={'policy': obs}, actions=action,
        distribution_params=(torch.zeros_like(action), torch.ones_like(action)),
        advantages=torch.tensor([1., -1., 1., -1.]).reshape(1, 4, 1))
    rows = wheel_advantage_audit(storage)
    assert len(rows) == 2
    assert [r['samples'] for r in rows] == [2, 2]
    assert [r['wheel_diff_surrogate_ascent'] for r in rows] == [1., -1.]
    counts = delivered_command_counts(obs)
    assert counts['1mps_+0.3radps'] == counts['1mps_-0.3radps'] == 2
    assert counts['other'] == 0

def test_crawl_advantage_audit_does_not_mix_speed_domains():
    from rmuc_position_authority_training import wheel_advantage_audit
    obs=torch.zeros(1,4,25)
    obs[...,7]=torch.tensor([.125,.125,.5,.5])
    obs[...,8]=.3
    actions=torch.zeros(1,4,6);actions[...,2]=torch.tensor([-1.,-1.,1.,1.])
    storage=SimpleNamespace(observations={'policy':obs},actions=actions,
        distribution_params=(torch.zeros_like(actions),torch.ones_like(actions)),
        advantages=torch.ones(1,4,1))
    rows=wheel_advantage_audit(storage)
    assert [(r['vx'],r['samples'],r['wheel_diff_surrogate_ascent']) for r in rows]==[(1.,2,1.),(.25,2,-1.)]
    assert all(r['error_groups'][0]['group']=='under_turn' for r in rows)

def test_physics_optimizer_updates_only_new_head_and_stops_inactive_momentum(tmp_path):
    import json
    from rmuc_physics_adaptation import PhysicsAdaptation
    from rmuc_local_authority import LocalAuthorityCorrection
    from rmuc_position_authority_training import configure_authority_audit
    actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6))
    actor.rmuc_turn_adapter=torch.nn.Linear(25,6)
    actor.rmuc_position_authority=PositionAuthorityHead()
    actor.rmuc_position_authority.local_correction=LocalAuthorityCorrection()
    actor.rmuc_physics_adaptation=PhysicsAdaptation();actor.is_recurrent=False
    critic=torch.nn.Linear(25,1);x=torch.zeros(2,25);x[:,7]=.5;x[:,8]=1.
    alg=SimpleNamespace(get_policy=lambda:actor,schedule='fixed',critic=critic,
                        storage=SimpleNamespace(observations={'policy':x[None]}, actions=torch.zeros(1,2,6),
                            distribution_params=(torch.zeros(1,2,6),torch.ones(1,2,6)), advantages=torch.ones(1,2,1)))
    runner=SimpleNamespace(alg=alg,logger=SimpleNamespace(log_dir=tmp_path),current_learning_iteration=0)
    prepare_authority_optimizer(runner,actor.state_dict(),physics_only=True)
    assert alg.optimizer.param_groups[0]['rmuc_group']=='physics_adaptation'
    assert {id(p) for p in alg.optimizer.param_groups[0]['params']}=={id(p) for p in actor.rmuc_physics_adaptation.parameters()}
    before={k:v.clone() for k,v in actor.state_dict().items()}
    def step():
        alg.optimizer.zero_grad(set_to_none=False)
        old=actor.rmuc_position_authority(x,actor.mlp(x))
        output=actor.rmuc_physics_adaptation(x,old)
        ((output-.5).square().mean()+critic(x).square().mean()).backward()
        alg.optimizer.step();return {}
    alg.update=step;configure_authority_audit(runner);alg.update()
    changed=[k for k,v in actor.state_dict().items() if not torch.equal(v,before[k])]
    assert changed and all(k.startswith('rmuc_physics_adaptation.network.') for k in changed)
    after={k:v.clone() for k,v in actor.state_dict().items()}
    x[:,7]=1. # speed2 is outside physical head support
    alg.update()
    assert all(torch.equal(v,after[k]) for k,v in actor.state_dict().items())
    records=[json.loads(s) for s in (tmp_path/'position_authority_training.jsonl').read_text().splitlines()]
    assert [r['physics_samples'] for r in records]==[2,0]

def test_selected_local_expert_learns_while_other_active_experts_remain_frozen():
    import pytest
    from rmuc_local_authority import LocalAuthorityCorrection
    actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6))
    actor.rmuc_turn_adapter=torch.nn.Linear(25,6)
    actor.rmuc_position_authority=PositionAuthorityHead()
    actor.rmuc_position_authority.local_correction=LocalAuthorityCorrection()
    actor.is_recurrent=False
    head=actor.rmuc_position_authority;critic=torch.nn.Linear(25,1)
    runner=SimpleNamespace(alg=SimpleNamespace(get_policy=lambda:actor,schedule='fixed',critic=critic))
    for invalid in ([], [0,0], [-1], [3]):
        with pytest.raises(ValueError,match='indices'):
            prepare_authority_optimizer(runner,actor.state_dict(),local_only=True,local_indices=invalid)
    prepare_authority_optimizer(runner,actor.state_dict(),local_only=True,local_indices=[0])
    assert runner.alg.optimizer.param_groups[0]['rmuc_local_indices']==[0]
    before={k:v.clone() for k,v in actor.state_dict().items()}
    x=torch.zeros(3,25);x[:,7]=.5;x[:,8]=torch.tensor([-.8,-1.,1.2])
    output=head(x,torch.zeros(3,6));loss=(output-.3).square().mean()+critic(x).square().mean()
    loss.backward();runner.alg.optimizer.step()
    changed=[k for k,v in actor.state_dict().items() if not torch.equal(v,before[k])]
    assert changed and all(k.startswith('rmuc_position_authority.local_correction.experts.0.') for k in changed)

def test_local_optimizer_freezes_source_and_clears_inactive_adam_momentum(tmp_path):
    import json
    from rmuc_local_authority import LocalAuthorityCorrection
    from rmuc_position_authority_training import configure_authority_audit
    actor=torch.nn.Module()
    actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6))
    actor.rmuc_turn_adapter=torch.nn.Linear(25,6)
    actor.rmuc_position_authority=PositionAuthorityHead()
    actor.rmuc_position_authority.local_correction=LocalAuthorityCorrection()
    actor.is_recurrent=False
    head=actor.rmuc_position_authority
    critic=torch.nn.Linear(25,1)
    x=torch.zeros(2,25);x[:,7]=.5;x[:,8]=-.8
    alg=SimpleNamespace(get_policy=lambda:actor,schedule='fixed',critic=critic,
                        storage=SimpleNamespace(observations={'policy':x[None]}))
    runner=SimpleNamespace(alg=alg,logger=SimpleNamespace(log_dir=tmp_path),current_learning_iteration=0)
    prepare_authority_optimizer(runner,actor.state_dict(),local_only=True)
    expected={id(p) for p in head.local_correction.parameters()}|{id(p) for p in critic.parameters()}
    assert {id(p) for g in alg.optimizer.param_groups for p in g['params']}==expected
    def step():
        alg.optimizer.zero_grad(set_to_none=False)
        loss=(head(x,torch.zeros(2,6))-.3).square().mean()+critic(x).square().mean()
        loss.backward();alg.optimizer.step()
        return {}
    alg.update=step
    configure_authority_audit(runner)
    frozen={n:p.detach().clone() for n,p in actor.named_parameters() if not p.requires_grad}
    branch=head.local_correction.experts[0]
    before={n:p.detach().clone() for n,p in branch.named_parameters()}
    alg.update()
    after={n:p.detach().clone() for n,p in branch.named_parameters()}
    assert any(not torch.equal(before[n],p) for n,p in after.items())
    x[:,8]=-1.  # Previously active expert now has zero gradients but nonzero Adam moments.
    alg.update()
    assert all(torch.equal(after[n],p) for n,p in branch.named_parameters())
    assert all(torch.equal(frozen[n],p) for n,p in actor.named_parameters() if n in frozen)
    records=[json.loads(line) for line in (tmp_path/'position_authority_training.jsonl').read_text().splitlines()]
    assert [r['local_samples'] for r in records]==[[2,0,0],[0,2,0]]

def test_shared_authority_optimizer_rejects_local_candidate():
    import pytest
    from rmuc_local_authority import LocalAuthorityCorrection
    actor = torch.nn.Module()
    actor.mlp = torch.nn.Sequential(torch.nn.Linear(25, 6))
    actor.rmuc_turn_adapter = torch.nn.Linear(25, 6)
    actor.rmuc_position_authority = PositionAuthorityHead()
    actor.rmuc_position_authority.local_correction = LocalAuthorityCorrection()
    actor.is_recurrent = False
    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: actor,
                             schedule='fixed', critic=torch.nn.Linear(25, 1)))
    with pytest.raises(ValueError, match='isolated local optimizer'):
        prepare_authority_optimizer(runner, actor.state_dict())
    assert not hasattr(runner.alg, 'optimizer')

def test_delivered_commands_reconstruct_yaw_and_cover_full_rollout():
    from rmuc_position_authority_training import delivered_command_counts
    obs = torch.zeros(2, 4, 25)
    obs[:, :, 7] = .5
    obs[:, :, 2] = .25 * .3
    obs[:, :, 8] = torch.tensor([[.8, -.8, 1., -1.], [1.2, -1.2, .93, .8]]) - .3
    obs[1, 3, 7] = 1.  # Same yaw at a different speed is outside these buckets.
    result = delivered_command_counts(obs)
    assert result['total'] == 8 and result['other'] == 2
    assert all(result[f'1mps_{y:+.1f}radps'] == 1 for y in (.8, -.8, 1., -1., 1.2, -1.2))

def test_optimizer_updates_only_authority_and_critic():
    actor=torch.nn.Module()
    actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6))
    actor.rmuc_turn_adapter=torch.nn.Linear(25,6)
    actor.rmuc_state_router=torch.nn.Linear(25,6)
    actor.rmuc_position_authority=PositionAuthorityHead()
    actor.is_recurrent=False
    critic=torch.nn.Linear(25,1)
    alg=SimpleNamespace(get_policy=lambda:actor,schedule='fixed',critic=critic)
    runner=SimpleNamespace(alg=alg)
    prepare_authority_optimizer(runner,actor.state_dict())
    expected={id(p) for p in actor.rmuc_position_authority.parameters()}|{id(p) for p in critic.parameters()}
    actual={id(p) for g in alg.optimizer.param_groups for p in g['params']}
    assert expected==actual
    assert all(not p.requires_grad for p in actor.mlp.parameters())
    original=actor.mlp[0].weight.detach().clone()
    x=torch.zeros(8,25);x[:,7]=.5;x[:4,8]=1.;x[4:,8]=-1.
    raw=actor.mlp(x).detach()
    before=actor.rmuc_position_authority(x,raw).detach()
    loss=(actor.rmuc_position_authority(x,raw)-.3).square().mean()+critic(x).square().mean()
    alg.optimizer.zero_grad();loss.backward();alg.optimizer.step()
    after=actor.rmuc_position_authority(x,raw).detach()
    assert not torch.equal(before[:4],after[:4]) and not torch.equal(before[4:],after[4:])
    assert torch.equal(original,actor.mlp[0].weight)


def test_migrated_physics_optimizer_freezes_complete_inherited_actor():
    from rmuc_physics_adaptation import PhysicsAdaptation
    actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6))
    actor.rmuc_turn_adapter=torch.nn.Linear(25,6)
    actor.rmuc_position_authority=PositionAuthorityHead();actor.is_recurrent=False
    old=PhysicsAdaptation()
    with torch.no_grad():old.network[-1].bias.fill_(.2)
    actor.rmuc_physics_adaptation=PhysicsAdaptation(old)
    critic=torch.nn.Linear(25,1)
    runner=SimpleNamespace(alg=SimpleNamespace(get_policy=lambda:actor,schedule='fixed',critic=critic))
    prepare_authority_optimizer(runner,actor.state_dict(),physics_only=True)
    assert {id(p) for p in runner.alg.optimizer.param_groups[0]['params']}=={
        id(p) for p in actor.rmuc_physics_adaptation.network.parameters()}
    before={k:v.clone() for k,v in actor.state_dict().items()}
    obs=torch.zeros(16,25);obs[:,7]=.5;obs[:,8]=1.
    loss=(actor.rmuc_physics_adaptation(obs,actor.mlp(obs))-.7).square().mean()
    loss.backward();runner.alg.optimizer.step()
    changed=[k for k,v in actor.state_dict().items() if not torch.equal(v,before[k])]
    assert changed and all(k.startswith('rmuc_physics_adaptation.network.') for k in changed)

def test_router_speed_optimizer_selects_only_positive_copies():
    from rmuc_state_router import IsolatedTurnStateRouter
    from rmuc_local_authority import LocalAuthorityCorrection
    actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6))
    actor.rmuc_turn_adapter=torch.nn.Linear(25,6);actor.is_recurrent=False
    actor.rmuc_state_router=IsolatedTurnStateRouter(extended_turn_1p0=True,e2e_turn_1p0=True)
    actor.rmuc_state_router.register_buffer('turn_speed_plateau_end',torch.tensor(1.5))
    actor.rmuc_state_router.install_speed_extensions()
    actor.rmuc_position_authority=PositionAuthorityHead();actor.rmuc_position_authority.local_correction=LocalAuthorityCorrection()
    alg=SimpleNamespace(get_policy=lambda:actor,schedule='fixed',critic=torch.nn.Linear(25,1))
    runner=SimpleNamespace(alg=alg)
    prepare_authority_optimizer(runner,actor.state_dict(),router_speed_direction='positive')
    names=[n for n,p in actor.named_parameters() if p.requires_grad]
    assert names and all(n.startswith('rmuc_state_router.speed_extensions.') and 'positive' in n for n in names)
    assert alg.optimizer.param_groups[0]['rmuc_group']=='router_speed_extension'
    assert {id(p) for p in alg.optimizer.param_groups[0]['params']}=={id(p) for p in actor.parameters() if p.requires_grad}

def test_router_speed_audit_stops_adam_momentum_on_protected_domain(tmp_path):
    from rmuc_state_router import IsolatedTurnStateRouter
    from rmuc_position_authority_training import configure_authority_audit
    actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_turn_adapter=torch.nn.Identity();actor.is_recurrent=False
    actor.rmuc_state_router=IsolatedTurnStateRouter(extended_turn_1p0=True,e2e_turn_1p0=True)
    actor.rmuc_state_router.register_buffer('turn_speed_plateau_end',torch.tensor(1.5));actor.rmuc_state_router.install_speed_extensions()
    actor.rmuc_position_authority=PositionAuthorityHead()
    x=torch.zeros(2,25);x[:,7]=.625;x[:,8]=1.
    critic=torch.nn.Linear(25,1)
    alg=SimpleNamespace(get_policy=lambda:actor,schedule='fixed',critic=critic,storage=SimpleNamespace(observations={'policy':x[None]}))
    runner=SimpleNamespace(alg=alg,logger=SimpleNamespace(log_dir=tmp_path),current_learning_iteration=0)
    prepare_authority_optimizer(runner,actor.state_dict(),router_speed_direction='positive')
    def step():
        alg.optimizer.zero_grad(set_to_none=False)
        y=actor.rmuc_state_router(x)
        ((y-.1).square().mean()+critic(x).square().mean()).backward()
        alg.optimizer.step();return {}
    alg.update=step;configure_authority_audit(runner);alg.update()
    before={k:v.clone() for k,v in actor.state_dict().items()}
    x[:,7]=.5
    alg.update()
    assert all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())
