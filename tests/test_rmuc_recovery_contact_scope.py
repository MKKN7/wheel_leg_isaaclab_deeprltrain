"""Execute the actual environment contact block without launching Isaac."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg'
spec=importlib.util.spec_from_file_location('scope_rewards',TASK/'recovery_rewards.py')
rewards=importlib.util.module_from_spec(spec);spec.loader.exec_module(rewards)
tree=ast.parse((TASK/'recovery_env.py').read_text())
method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_get_rewards')
start=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='wheel_forces' for t in n.targets))
end=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='braking_cost' for t in n.targets))
block=compile(ast.fix_missing_locations(ast.Module(body=method.body[start:end],type_ignores=[])),'<actual contact scope>','exec')


@pytest.mark.parametrize('full_scope',[False,True])
def test_wheel_transition_independent_of_body_impact_scope(full_scope):
    # Noncontiguous wheel indices, with an unrelated unloaded body and a
    # chassis impact. Only env0 acquires two-wheel support; env1 keeps it.
    forces=torch.zeros(4,5,3)
    forces[0,1,2]=forces[0,3,2]=5.
    forces[1,1,2]=forces[1,3,2]=5.
    forces[2,1,2]=5.
    forces[:,0,2]=600.
    previous=torch.tensor([False,True,False,False])
    env=NS(contact_sensor=NS(data=NS(net_forces_w=forces)),
        _wheel_contact_sensor_ids=[1,3],_recovery_previous_both=previous,
        cfg=NS(recovery_full_safety_scope=full_scope,recovery_contact_soft_limit_n=300.,recovery_contact_event_mode='legacy_edge'))
    ns=dict(self=env,torch=torch,contact_impact_penalty=rewards.contact_impact_penalty,
        double_contact_acquired=rewards.double_contact_acquired)
    exec(block,ns)
    assert ns['double_contact'].tolist()==[1.,0.,0.,0.]
    assert ns['wheel_contact'].tolist()==[True,True,True,False]
    assert previous.tolist()==[True,True,False,False]
    assert bool((ns['impact_cost']>0).all())==full_scope
    exec(block,ns)
    assert ns['double_contact'].sum()==0  # no persistent reward while supported


def test_global_weight_four_inherited_and_explicit_historical_bank_two_preserved():
    init=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assignment=next(n for n in init.body if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Attribute) and t.attr=='_recovery_resolved_banks' for t in n.targets))
    code=compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),'<actual bank resolution>','exec')
    global_weights={'supported_height':4.,'stable_hold':3.,'double_contact':0.}
    cfg=NS(recovery_standing_weights=global_weights,recovery_case_standing_weights={})
    env=NS();exec(code,dict(self=env,cfg=cfg))
    assert all(bank['supported_height']==4. for bank in env._recovery_resolved_banks.values())
    cfg.recovery_case_standing_weights={'front_down':{'supported_height':2.},'back_down':{'double_contact':1.}}
    exec(code,dict(self=env,cfg=cfg))
    assert env._recovery_resolved_banks['front_down']['supported_height']==2.
    assert env._recovery_resolved_banks['back_down']['supported_height']==4.
    assert env._recovery_resolved_banks['back_down']['double_contact']==1.
    assert global_weights['supported_height']==4.
