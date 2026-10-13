import sys,copy,json
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_state_router import IsolatedTurnStateRouter
from rmuc_state_router_training import configure_router_training, select_router_branches


@pytest.mark.parametrize('recovery', [False, True])
def test_real_aux_hooks_match_explicit_loss_and_preserve_frozen_graph(tmp_path, recovery):
    torch.manual_seed(33)
    actor=torch.nn.Module();actor.add_module('mlp',torch.nn.Linear(25,6));actor.add_module('rmuc_state_router',IsolatedTurnStateRouter(recovery_positive_2p0=recovery))
    with torch.no_grad():actor.rmuc_state_router.positive.router[-1].bias[0]=2
    reference=copy.deepcopy(actor);reference.mlp.requires_grad_(False)
    x=torch.randn(16,25)*.1;x[:,7]=1.2;x[:,8]=.3-x[:,2]/.25
    if recovery:
        x[:8,7]=1.0
    actual_opt=torch.optim.Adam(actor.rmuc_state_router.parameters(),lr=.001)
    expected_opt=torch.optim.Adam(reference.rmuc_state_router.parameters(),lr=.001)
    def update():
        actual_opt.zero_grad();out=actor.mlp(x)+actor.rmuc_state_router(x)
        (out-.4).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
        actual_opt.step();return {}
    alg=SimpleNamespace(get_policy=lambda:actor,optimizer=actual_opt,update=update,
                        storage=SimpleNamespace(observations={'policy':x.unsqueeze(0)}))
    runner=SimpleNamespace(alg=alg,current_learning_iteration=1,logger=SimpleNamespace(log_dir=str(tmp_path)))
    frozen=actor.mlp.weight.detach().clone();configure_router_training(runner,.01)
    for i in range(3):
        expected_opt.zero_grad();out=reference.mlp(x)+reference.rmuc_state_router(x)
        loss=(out-.4).square().mean()+.01*reference.rmuc_state_router.balance_loss(x)
        loss.backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
        reference.rmuc_state_router.isolate_optimizer_step(x);expected_opt.step()
        runner.alg.update()
        for a,b in zip(actor.parameters(),reference.parameters()):torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-6)
    assert torch.equal(actor.mlp.weight,frozen)
    records=[json.loads(l) for l in (tmp_path/'state_router_training.jsonl').read_text().splitlines()]
    assert len(records)==3 and all(r['aux_minibatches']==1 for r in records)
    assert all(r['router_loads']['negative']['samples']==0 for r in records)
    if recovery:
        assert all(r['router_loads']['recovery_positive_2p0']['samples']>0 for r in records)


def test_explicit_branch_freeze_survives_eligible_old_commands_and_aux_loss(tmp_path):
    torch.manual_seed(18)
    actor=torch.nn.Module()
    actor.add_module('mlp',torch.nn.Linear(25,6))
    actor.add_module('rmuc_state_router',IsolatedTurnStateRouter(recovery_positive_2p0=True))
    # Warm all branches with AdamW before selecting only the recovery branch.
    optimizer=torch.optim.AdamW(actor.parameters(),lr=.01,weight_decay=.1)
    x=torch.randn(48,25)*.1;x[:,7]=1.2
    x[:16,7]=1.;x[:,8]=.3-x[:,2]/.25;x[32:,8]=-.3-x[32:,2]/.25
    for _ in range(3):
        optimizer.zero_grad();(actor.mlp(x)+actor.rmuc_state_router(x)-.5).square().mean().backward();optimizer.step()
    before={n:p.detach().clone() for n,p in actor.named_parameters()}
    def update():
        optimizer.zero_grad()
        (actor.mlp(x)+actor.rmuc_state_router(x)-.5).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
        optimizer.step()
        return {}
    alg=SimpleNamespace(get_policy=lambda:actor,optimizer=optimizer,update=update,
        storage=SimpleNamespace(observations={'policy':x.unsqueeze(0)}))
    runner=SimpleNamespace(alg=alg,current_learning_iteration=1,logger=SimpleNamespace(log_dir=str(tmp_path)),
        rmuc_trainable_router_branches=('recovery_positive_2p0',))
    configure_router_training(runner,.01)
    for _ in range(3):runner.alg.update()
    changed=[n for n,p in actor.named_parameters() if not torch.equal(p,before[n])]
    assert changed and all(n.startswith('rmuc_state_router.recovery_positive_2p0.') for n in changed)
    record=json.loads((tmp_path/'state_router_training.jsonl').read_text().splitlines()[-1])
    assert record['router_loads']['positive']['samples']>0 and record['router_loads']['negative']['samples']>0
    assert record['trainable_branches']==['recovery_positive_2p0']
    # Frozen router tensors must be part of the real runtime integrity audit.
    with torch.no_grad():actor.rmuc_state_router.positive.router[0].weight.add_(1)
    with pytest.raises(RuntimeError,match='changed frozen graph'):runner.alg.update()


@pytest.mark.parametrize('names',[[],['missing'],['positive','positive']])
def test_branch_selection_rejects_empty_unknown_and_duplicate_names(names):
    actor=torch.nn.Module();actor.add_module('rmuc_state_router',IsolatedTurnStateRouter())
    with pytest.raises(ValueError):select_router_branches(actor,names)


def test_resume_checks_branch_identity_not_only_optimizer_tensor_shapes():
    import ast
    path=Path(__file__).resolve().parents[1]/'scripts/rsl_rl/train.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
              and ast.unparse(n.test)=='args_cli.state_router_train_branches is not None and (not args_cli.reset_optimizer)')
    code=compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec')
    args=SimpleNamespace(state_router_train_branches=['positive'],reset_optimizer=False)
    for saved in [None,['negative']]:
        scope={'args_cli':args,'router_payload':{'optimizer_state_dict':{'param_groups':[
            {'rmuc_group':'state_router','rmuc_branches':saved}]}}}
        with pytest.raises(ValueError,match='provenance'):exec(code,scope)
        args.reset_optimizer=True;exec(code,scope);args.reset_optimizer=False
    exec(code,{'args_cli':args,'router_payload':{'optimizer_state_dict':{'param_groups':[
        {'rmuc_group':'state_router','rmuc_branches':['positive']}]}}})


def test_branch_cli_does_not_consume_hydra_overrides():
    import ast,argparse
    path=Path(__file__).resolve().parents[1]/'scripts/rsl_rl/train.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
              and n.value.args and isinstance(n.value.args[0],ast.Constant)
              and n.value.args[0].value=='--state_router_train_branches')
    parser=argparse.ArgumentParser()
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),{'parser':parser})
    args,remaining=parser.parse_known_args(['--state_router_train_branches','positive,negative',
        'env.domain_randomization_strength_override=0','agent.run_name=test'])
    assert args.state_router_train_branches==['positive','negative']
    assert remaining==['env.domain_randomization_strength_override=0','agent.run_name=test']
