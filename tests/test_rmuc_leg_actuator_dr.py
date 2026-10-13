from pathlib import Path
import runpy
from types import SimpleNamespace
import torch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sample=runpy.run_path(str(ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/leg_actuator_dr.py'))['sample_leg_parameters']
apply=runpy.run_path(str(ROOT/'scripts/rmuc_leg_actuator_profile.py'))['apply_leg_profile']

def test_dr_zero_is_exact_and_consumes_no_randomness_and_full_stays_in_range():
    v=torch.full((1000,4),17.);a=torch.full_like(v,.015795)
    state=torch.random.get_rng_state()
    v0,a0=sample(v,a,0.,(12.,17.),(.008,.024))
    assert torch.equal(v,v0) and torch.equal(a,a0) and torch.equal(state,torch.random.get_rng_state())
    vs,ars=sample(v,a,1.,(12.,17.),(.008,.024))
    assert ((vs>=12)&(vs<=17)).all() and ((ars>=.008)&(ars<=.024)).all()
    assert vs.std()>0 and ars.std()>0
    torch.manual_seed(18);vf,af=sample(v,a,1.,(12.,17.),(.008,.024))
    torch.manual_seed(18);vh,ah=sample(v,a,.5,(12.,17.),(.008,.024))
    torch.testing.assert_close(vh,(v+vf)/2);torch.testing.assert_close(ah,(a+af)/2)

def test_profile_sets_actual_solver_limit_and_keeps_wheel_and_leg_pd():
    import yaml
    p=yaml.safe_load((ROOT/'configs/rmuc_leg_actuator_target.yaml').read_text())
    leg=SimpleNamespace(stiffness=60.,damping=2.,effort_limit=40.)
    wheel=SimpleNamespace(effort_limit=5.5)
    cfg=SimpleNamespace(robot_cfg=SimpleNamespace(actuators={'leg_motors':leg,'wheel_motors':wheel}))
    apply(cfg,p,True)
    assert leg.velocity_limit_sim==17 and leg.velocity_limit==17 and leg.armature==.015795
    assert leg.stiffness==60 and leg.damping==2 and leg.effort_limit==40
    assert wheel.effort_limit==5.5 and cfg.leg_actuator_dr_enabled

def test_invalid_range_rejected():
    with pytest.raises(ValueError):sample(torch.ones(2,4),torch.ones(2,4),1.,(17.,12.),(.008,.024))
