import torch
import pytest
from rmuc_student_runtime import StudentActor

@pytest.mark.parametrize('shape,count', [([25,128,128,6],20614),([25,160,152,6],29550)])
def test_shapes_and_normalizer_fold(shape,count):
    _,h1,h2,_=shape
    net=torch.nn.Sequential(torch.nn.Linear(25,h1),torch.nn.ReLU(),torch.nn.Linear(h1,h2),torch.nn.ReLU(),torch.nn.Linear(h2,6))
    payload=dict(architecture=shape,student_state_dict=net.state_dict(),mean=torch.randn(25),std=torch.rand(25)+.01)
    rng_before = torch.random.get_rng_state().clone()
    actor=StudentActor(payload).eval()
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert sum(v.numel() for v in actor.parameters())==count
    x=torch.randn(100,25)
    with torch.no_grad():
        expected=actor({'policy':x})
        w=net[0].weight.clone();net[0].weight.copy_(w/payload['std']);net[0].bias.sub_((w*(payload['mean']/payload['std'])).sum(1))
        torch.testing.assert_close(net(x),expected,atol=2e-5,rtol=2e-5)
    with pytest.raises(ValueError):actor({'policy':x},stochastic_output=True)
