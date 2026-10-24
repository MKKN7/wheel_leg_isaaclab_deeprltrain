import importlib.util
from pathlib import Path
import pytest
import torch

path = Path(__file__).resolve().parents[1] / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/survival_shaping.py'
spec = importlib.util.spec_from_file_location('survival_shaping', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unsafe_and_protected_skills_receive_no_baseline():
    c = torch.tensor([[2.5,0,0],[2.5,0,0],[1.9,0,0],[-.4,0,0],[2.5,0,.3]])
    unsafe = torch.tensor([False,True,False,False,False])
    assert torch.equal(module.high_straight_alive_baseline(c,unsafe,25),torch.tensor([25.,0,0,0,0]))
    assert torch.equal(module.high_straight_alive_baseline(c,unsafe,0),torch.zeros(5))


def test_recorded_negative_step_regime_no_longer_rewards_early_exit():
    c=torch.tensor([[2.5,0,0],[2.5,0,0]])
    raw=torch.tensor([-20.,-25.])
    rewards=(raw+module.high_straight_alive_baseline(c,torch.tensor([False,True]),25)).clamp(-25,20)
    assert rewards.tolist()==[5.,-25.]
    assert float(rewards[0]+.997*rewards[1])>float(rewards[1])


@pytest.mark.parametrize('amount',[-1,26,float('nan'),float('inf')])
def test_invalid_amount(amount):
    with pytest.raises(ValueError):module.high_straight_alive_baseline(torch.zeros(1,3),torch.zeros(1,dtype=torch.bool),amount)
