import importlib.util
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[1] / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/reverse_stability.py'
spec = importlib.util.spec_from_file_location('reverse_stability', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_only_fast_straight_reverse_after_acquisition_is_penalized():
    c = torch.tensor([[-.975,0,0],[-1.8,0,0],[-.925,0,0],[1,0,0],[-1,0,.2],[-1.9,0,0],[-1,0,0]])
    age = torch.tensor([375,375,375,375,375,375,374])
    r = module.reverse_pitch_oscillation_cost(c, torch.full((7,),.24), age,.008)
    torch.testing.assert_close(r,torch.tensor([1.,1.,0.,0.,0.,0.,0.]))


def test_low_frequency_pitch_at_gate_has_dense_signal_and_sign_symmetry():
    t = torch.arange(750)*.008
    # Three-degree p2p at the observed .833 Hz is below the old .15rad/s rate threshold.
    rate = torch.deg2rad(torch.tensor(1.5))*2*torch.pi*(5/6)*torch.cos(2*torch.pi*(5/6)*t)
    c = torch.zeros(750,3);c[:,0]=-.975;age=torch.arange(750)+750
    cost=module.reverse_pitch_oscillation_cost(c,rate,age,.008)
    assert rate.abs().max()<.15 and float(cost.mean())>.05
    torch.testing.assert_close(cost,module.reverse_pitch_oscillation_cost(c,-rate,age,.008))
    assert not module.reverse_pitch_oscillation_cost(c,torch.zeros_like(rate),age,.008).any()
    severe=module.reverse_pitch_oscillation_cost(c,torch.full_like(rate,10),age,.008)
    assert torch.equal(severe,torch.ones_like(severe))


@pytest.mark.parametrize('dt',[0,-.008,float('nan'),float('inf')])
def test_invalid_step(dt):
    with pytest.raises(ValueError):
        module.reverse_pitch_oscillation_cost(torch.zeros(1,3),torch.zeros(1),torch.zeros(1),dt)
