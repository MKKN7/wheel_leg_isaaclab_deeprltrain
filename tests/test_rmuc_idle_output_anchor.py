import torch
from torch import nn
from rmuc_startup_anchor import StartupAnchor

class Actor(nn.Module):
    def __init__(self):super().__init__();self.layer=nn.Linear(25,6)
    def forward(self,obs,stochastic_output=False):return self.layer(obs['policy'])

def test_anchor_limits_fixed_outputs_and_restores_optimizer_on_rollback(tmp_path):
    a=Actor();o=torch.randn(16,25);y=a({'policy':o}).detach();f=tmp_path/'anchor.pt';torch.save(dict(obs=o,target=y),f)
    opt=torch.optim.Adam(a.parameters(),lr=.1);guard=StartupAnchor(a,opt,f)
    before=[p.detach().clone() for p in a.parameters()];guard.begin();a({'policy':o}).sum().backward();opt.step();r=guard.finish()
    assert r['accepted_scale']<1
    assert guard.error()<=.010001
    assert all(p not in opt.state for p in a.parameters())
    assert all(torch.isfinite(p).all() for p in a.parameters())
