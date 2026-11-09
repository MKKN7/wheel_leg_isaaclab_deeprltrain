import torch
from rmuc_consumer_boundary import boundary_loss

def test_bound_gradient_preserves_feasible_actions_and_excludes_other_commands():
 obs=torch.zeros(3,25);obs[:,7]=torch.tensor([.25,.25,1.]);obs[:,8]=.3
 x=torch.tensor([[0.,0.,0.,1.1,0.,0.],[0.,0.,0.,.9,0.,0.],[0.,0.,0.,1.5,0.,0.]],requires_grad=True)
 before=x.detach().clone();loss=boundary_loss(x,obs,[[.5,.3]],2.);loss.backward()
 expected=torch.zeros_like(x);expected[0,3]=2*.1/12
 torch.testing.assert_close(x.grad,expected);torch.testing.assert_close(x.detach(),before)

def test_negative_limit_and_expanded_authority_domain():
 obs=torch.zeros(2,25);obs[:,7]=torch.tensor([.25,.5]);obs[:,8]=torch.tensor([.3,.8])
 x=torch.zeros(2,6);x[0,3]=-1.1;x[1,3]=1.5;x.requires_grad_()
 boundary_loss(x,obs,[[.5,.3],[1.,.8]],1.).backward()
 assert x.grad[0,3]<0
 assert x.grad[1,3]==0

def test_hook_keeps_forward_and_adds_exact_gradient_and_allows_warmup(tmp_path):
 from types import SimpleNamespace
 from rmuc_consumer_boundary import configure_consumer_boundary
 class Actor(torch.nn.Module):
  is_recurrent=False
  def __init__(self):
   super().__init__();self.mean=torch.nn.Parameter(torch.tensor([[0.,0.,0.,1.1,0.,0.]]));self.distribution=SimpleNamespace(update=lambda x:x)
  def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
   out=self.mean+0
   if stochastic_output:self.distribution.update(out)
   return out
 actor=Actor();obs={'policy':torch.zeros(1,25)};obs['policy'][:,7]=.25;obs['policy'][:,8]=.3
 def update():
  actor.zero_grad();out=actor(obs,stochastic_output=True)
  if out.requires_grad:(out.square().sum()).backward()
  return out.detach().clone()
 alg=SimpleNamespace(get_policy=lambda:actor,update=update,symmetry=None)
 runner=SimpleNamespace(alg=alg,logger=SimpleNamespace(log_dir=str(tmp_path)))
 env=SimpleNamespace(cfg=SimpleNamespace(command_active_targets=[[.5,.3]],position_authority_speed_plateau_end=1.));env.unwrapped=env
 configure_consumer_boundary(runner,env,2.)
 before=actor.mean.detach().clone();torch.testing.assert_close(alg.update(),before)
 expected=2*before;expected[0,3]+=2*.1/6
 torch.testing.assert_close(actor.mean.grad,expected)
 actor.requires_grad_(False);torch.testing.assert_close(alg.update(),before)
 actor.requires_grad_(True);torch.testing.assert_close(alg.update(),before)
 torch.testing.assert_close(actor.mean.grad,expected)
