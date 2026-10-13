import torch
from rmuc_startup_adaptation import StartupAdaptation

def observations():
    obs=torch.zeros(8,25)
    obs[:,7]=.5;obs[:,8]=1.4;obs[:,6]=2.
    return obs

def test_zero_init_and_scope_boundaries():
    m=StartupAdaptation();obs=observations();source=torch.randn(8,6)
    assert torch.equal(m(obs,source),source)
    obs[1,8]=1.2;obs[2,8]=-1.4;obs[3,7]=.75;obs[4,8]=.8
    obs[5,6]=0.;obs[6,8]=0.;obs[7,7]=0.
    assert torch.equal(m.gate(obs),torch.tensor([1.,0.,0.,0.,0.,0.,0.,0.]))
    with torch.no_grad():m.network[-1].bias.fill_(100.)
    actual=m(obs,source)
    torch.testing.assert_close(actual[0]-source[0],m.budget)
    assert torch.equal(actual[1:],source[1:])
