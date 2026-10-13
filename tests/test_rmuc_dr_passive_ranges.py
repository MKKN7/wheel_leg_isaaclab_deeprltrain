import importlib.util
from pathlib import Path
import pytest
import torch
path=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/dr_passive_ranges.py'
spec=importlib.util.spec_from_file_location('passive_ranges',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_zero_strength_preserves_rng_and_nominal_exactly():
    template=torch.zeros(17,2);g=torch.Generator().manual_seed(445);before=g.get_state().clone()
    b=m.sample_mechanical_viscous(template,0,generator=g)
    mass,inertia=m.sample_mass_and_total_inertia_scales(template,0,generator=g)
    assert torch.equal(b,torch.full_like(template,.002))
    assert torch.equal(mass,torch.ones_like(template)) and torch.equal(mass,inertia)
    assert torch.equal(before,g.get_state())


@pytest.mark.parametrize('s,expected',[(0,(.002,.002)),(.02,(.00198,.00206)),(.5,(.0015,.0035)),(1,(.001,.005))])
def test_absolute_asymmetric_b_bounds(s,expected):
    assert m.mechanical_viscous_bounds(s)==pytest.approx(expected)
    b=m.sample_mechanical_viscous(torch.zeros(10000,dtype=torch.float64),s)
    assert float(b.min())>=expected[0]-1e-15 and float(b.max())<=expected[1]+1e-15


def test_mass_and_total_inertia_are_independent_and_not_multiplied():
    mass,inertia=m.sample_mass_and_total_inertia_scales(torch.zeros(100000),1,generator=torch.Generator().manual_seed(445))
    assert (mass>=.8).all() and (mass<=1.2).all()
    assert (inertia>=.8).all() and (inertia<=1.2).all()
    assert abs(float(torch.corrcoef(torch.stack((mass,inertia)))[0,1]))<.02
    assert ((mass*inertia<.8)|(mass*inertia>1.2)).any()  # catches the forbidden double scaling


@pytest.mark.parametrize('s',[float('nan'),float('inf'),-.1,1.1,True])
def test_invalid_strength_rejected_before_rng(s):
    g=torch.Generator().manual_seed(1);before=g.get_state().clone()
    with pytest.raises(ValueError):m.sample_mechanical_viscous(torch.zeros(2),s,generator=g)
    with pytest.raises(ValueError):m.sample_mass_and_total_inertia_scales(torch.zeros(2),s,generator=g)
    assert torch.equal(before,g.get_state())


@pytest.mark.parametrize('bounds',[(-.1,.005),(.003,.005),(.001,.0005),(.001,float('nan'))])
def test_invalid_b_bounds(bounds):
    with pytest.raises(ValueError):m.sample_mechanical_viscous(torch.zeros(2),.1,bounds=bounds)


def test_invalid_inertia_fraction_does_not_draw_mass_first():
    g=torch.Generator().manual_seed(1);before=g.get_state().clone()
    with pytest.raises(ValueError):m.sample_mass_and_total_inertia_scales(torch.zeros(2),.1,inertia_fraction=1.,generator=g)
    assert torch.equal(before,g.get_state())
