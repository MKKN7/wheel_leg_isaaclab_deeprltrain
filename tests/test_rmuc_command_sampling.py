import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest
import yaml
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg'
spec=importlib.util.spec_from_file_location('sampling',SRC/'command_sampling.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_continuous_envelope_reproducible_and_no_future_commands():
    def sample():return m.sample_opened_commands(10000,[0.,1.,1.,2.,3.],[0.,.15,.3,.2,3.],3,generator=torch.Generator().manual_seed(4))
    x=sample();assert torch.equal(x,sample())
    assert x[:,0].min()>=0 and x[:,0].max()<=2 and torch.unique(x[:,0]).numel()>9900
    assert x[x[:,0]<1,2].count_nonzero()==0
    turning=x[x[:,0]>=1,2];assert turning.abs().max()<=.2
    assert (turning>0).sum()>2000 and (turning<0).sum()>2000


def method(name):
    cls=next(n for n in ast.parse((SRC/'env.py').read_text()).body if isinstance(n,ast.ClassDef))
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    ns={'torch':torch,'sample_opened_commands':m.sample_opened_commands}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_env','exec'),ns)
    return ns[name]


def test_reset_cohort_quota_and_partial_reset_preserve_existing_commands():
    cfg=SimpleNamespace(command_continuous_sampling_fraction=.15,curriculum_stage=1,play=False,
        feedback_contract_by_command=True,command_reward_contract_by_command=True,
        command_rehearsal_fraction=.6,command_rehearsal_speed_focus_fraction=.3,
        command_stage_linear_x_mps=[0.,1.,2.],command_stage_yaw_rate_rad_s=[0.,.3,.3])
    env=SimpleNamespace(cfg=cfg,num_envs=100,device='cpu',_command_level=2,
        _continuous_command_mask=torch.zeros(100,dtype=torch.bool),_continuous_commands=torch.zeros(100,3),
        _rehearsal_mask=torch.arange(100)<60)
    fn=method('_refresh_continuous_commands');fn(env,torch.arange(100))
    assert env._continuous_command_mask.sum()==15
    assert not (env._continuous_command_mask&env._rehearsal_mask).any()
    old=env._continuous_commands.clone();mask=env._continuous_command_mask.clone()
    fn(env,torch.tensor([99]));assert torch.equal(env._continuous_commands[:99],old[:99])
    assert env._continuous_command_mask.sum()==15
    env.cfg.command_continuous_sampling_fraction=0.;fn(env,torch.arange(100));assert not env._continuous_command_mask.any()


def test_actual_fixed_gate_masks_exclude_random_cohort():
    cls=next(n for n in ast.parse((SRC/'env.py').read_text()).body if isinstance(n,ast.ClassDef))
    checks={'gate_mask':0,'episode_tracking_mask':0,'completed_mask':0}
    env=SimpleNamespace(_rehearsal_mask=torch.tensor([False,False,True]),
      _acquisition_ramp_tracking_ready=lambda:torch.ones(3,dtype=torch.bool),
      _continuous_command_mask=torch.tensor([True,False,False]),_reset_speed_bootstrap_mask=torch.zeros(3,dtype=torch.bool),
      _command_episode_started=torch.ones(3,dtype=torch.bool),_curriculum_step_counter=torch.full((3,),1500))
    for n in ast.walk(cls):
      if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in checks:
        ns={'self':env,'env_ids':torch.arange(3),'window_ready':torch.ones(3,dtype=torch.bool),'steady_state_episode':torch.ones(3,dtype=torch.bool)}
        exec(compile(ast.Module(body=[n],type_ignores=[]),'actual_gate','exec'),ns)
        assert torch.equal(ns[n.targets[0].id],torch.tensor([False,True,False]));checks[n.targets[0].id]+=1
    assert all(checks.values())


def test_straight_only_knots_do_not_erase_opened_turn_intervals():
    x=m.sample_opened_commands(10000,[1.,1.2,1.5,2.,3.],[.3,0.,.3,.2,0.],4,
        generator=torch.Generator().manual_seed(7))
    mid=x[(x[:,0]>1.)&(x[:,0]<1.5),2]
    assert (mid>0).sum()>1000 and (mid<0).sum()>1000
    assert mid.abs().max()<=.3
    assert x[x[:,0]>2.,2].count_nonzero()==0


def test_no_turn_anchor_and_single_turn_anchor_do_not_extrapolate():
    for yaws in [[0.,0.],[.3,0.]]:
        x=m.sample_opened_commands(1000,[1.,2.],yaws,1,generator=torch.Generator().manual_seed(4))
        assert x[:,2].count_nonzero()==0
    x=m.sample_opened_commands(100,[1.],[.3],0)
    assert x[:,2].abs().max()<=.3 and x[:,2].count_nonzero()>0


def extension_config():
    cfg=yaml.safe_load((ROOT/'configs/rmuc_wheel_leg.yaml').read_text())['command']
    return cfg, [s['linear_x_mps'] for s in cfg['stages']], [s['yaw_rate_rad_s'] for s in cfg['stages']]


def test_extension_preserves_legacy_distribution_and_rng_exactly():
    cfg,speeds,yaws=extension_config()
    for level in [18,28,46]:
        old=m.sample_opened_commands(3000,speeds,yaws,level,generator=torch.Generator().manual_seed(2))
        new=m.sample_opened_commands(3000,speeds,yaws,level,generator=torch.Generator().manual_seed(2),
            final_turn_envelope=cfg['final_turn_envelope'],extended_start_level=47)
        assert torch.equal(old,new)


def test_final_extension_covers_intermediate_speeds_without_legacy_yaw_bottleneck():
    cfg,speeds,yaws=extension_config()
    x=m.sample_opened_commands(50000,speeds,yaws,len(speeds)-1,generator=torch.Generator().manual_seed(12),
        final_turn_envelope=cfg['final_turn_envelope'],extended_start_level=47)
    # Old 2.2/2.4/2.6/2.8/2.9 knots must not keep the final domain at .3.
    for low,high in [(1.,1.1),(1.2,1.4),(1.8,2.),(2.1,2.4),(2.6,2.99)]:
        yaw=x[(x[:,0]>low)&(x[:,0]<high),2]
        assert yaw.max()>.95 and yaw.min()<-.95
        assert (yaw.abs()<.05).any()  # target-band minimum is not a sampling floor
    assert x[x[:,0]>3,2].abs().max()<=.3
    envelope=cfg['final_turn_envelope']
    cap=m._interpolate_inside(x[:,0],torch.tensor([v['speed_mps'] for v in envelope]),
        torch.tensor([v['yaw_max_rad_s'] for v in envelope]))
    assert (x[x[:,0]<=3,2].abs()<=cap[x[:,0]<=3]+1e-6).all()


def test_future_extension_anchors_are_not_opened_by_final_specification():
    cfg,speeds,yaws=extension_config()
    level=next(i for i,s in enumerate(cfg['stages']) if s['name']=='ext_1p5_turn_0p8')
    x=m.sample_opened_commands(20000,speeds,yaws,level,generator=torch.Generator().manual_seed(3),
        final_turn_envelope=cfg['final_turn_envelope'],extended_start_level=47)
    assert x[x[:,0]>1.5,2].abs().max()<=.3
    # At 1.5 only .8 is opened, not the final 1.8.
    near=x[(x[:,0]>1.49)&(x[:,0]<1.5),2]
    assert near.abs().max()<.85
    assert x[(x[:,0]>1)&(x[:,0]<1.1),2].abs().max()>2


def test_actual_env_reset_consumes_extension_configuration():
    cfg,speeds,yaws=extension_config()
    runtime=SimpleNamespace(command_continuous_sampling_fraction=.2,curriculum_stage=1,play=False,
        feedback_contract_by_command=True,command_reward_contract_by_command=True,
        command_rehearsal_fraction=.6,command_rehearsal_speed_focus_fraction=.3,
        command_stage_linear_x_mps=speeds,command_stage_yaw_rate_rad_s=yaws,
        command_final_turn_envelope=cfg['final_turn_envelope'],command_extended_sampling_start_level=47)
    env=SimpleNamespace(cfg=runtime,num_envs=2000,device='cpu',_command_level=len(speeds)-1,
        _continuous_command_mask=torch.zeros(2000,dtype=torch.bool),_continuous_commands=torch.zeros(2000,3),
        _rehearsal_mask=torch.arange(2000)<1200)
    method('_refresh_continuous_commands')(env,torch.arange(2000))
    assert env._continuous_command_mask.sum()==400
    x=env._continuous_commands[env._continuous_command_mask]
    assert (x[:,2]>1).any() and (x[:,2]<-1).any()


def test_invalid_extension_envelope_rejected():
    with pytest.raises(ValueError):
        m.sample_opened_commands(10,[1.,2.],[.3,.3],1,extended_start_level=0,
            final_turn_envelope=[{'speed_mps':2.,'yaw_max_rad_s':1.},{'speed_mps':1.,'yaw_max_rad_s':2.}])
