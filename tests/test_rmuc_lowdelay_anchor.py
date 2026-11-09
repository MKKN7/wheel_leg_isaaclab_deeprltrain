import torch
from rmuc_startup_anchor import StartupAnchor
class Actor(torch.nn.Module):
 def __init__(self):super().__init__();self.linear=torch.nn.Linear(25,6)
 def forward(self,obs,stochastic_output=False):return self.linear(obs['policy'])
def test_tight_guard_scales_update_and_restores_optimizer(tmp_path):
 torch.manual_seed(4);a=Actor();o=torch.randn(64,25);f=tmp_path/'anchor.pt';torch.save({'obs':o,'target':a({'policy':o}).detach()},f);opt=torch.optim.Adam(a.parameters());guard=StartupAnchor(a,opt,f,limit=.002);guard.begin()
 with torch.no_grad():a.linear.bias.add_(.008)
 result=guard.finish();assert result['accepted_scale']==.25;assert guard.error()<=.002001
