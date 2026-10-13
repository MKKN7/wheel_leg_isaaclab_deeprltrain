import importlib.util
from pathlib import Path
import pytest
import torch

path=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/recovery_delay.py'
spec=importlib.util.spec_from_file_location('recovery_delay',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_probability_override_is_independent_but_nominal_retention_stays_zero():
    assert m.communication_delay_probability(.02)==.02
    assert m.communication_delay_probability(.02,.1)==.1
    assert m.communication_delay_probability(.02,0.)==0.
    assert m.communication_delay_probability(.02,1.)==1.
    assert m.communication_delay_probability(0.,1.)==0.
    for value in [-.2,1.1,float('nan')]:
        with pytest.raises(ValueError):m.communication_delay_probability(.02,value)


def test_cap_preserves_zero_samples_source_and_rng():
    source=torch.tensor([0,3,1,0,2]);rng=torch.get_rng_state().clone()
    assert m.cap_communication_delay(source,-1) is source
    assert m.cap_communication_delay(source,1).tolist()==[0,1,1,0,1]
    assert m.cap_communication_delay(source,2).tolist()==[0,2,1,0,2]
    assert torch.equal(m.cap_communication_delay(source,3),source)
    assert source.tolist()==[0,3,1,0,2] and torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('cap',[-2,4,1.5,True])
def test_invalid_cap_rejected(cap):
    with pytest.raises(ValueError):m.cap_communication_delay(torch.zeros(1),cap)
