import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rmuc_world_model'))
from transition_reward import overshoot_cost,robust_overshoot_cost

def test_same_direction_and_boundary():
 target=torch.tensor([-.6]);meta=torch.tensor([[-1.,.2]])
 for speed in [-.5,-.6,-.635]:
  assert robust_overshoot_cost(torch.tensor([speed]),target,meta)==0
 v=torch.tensor([-.9],requires_grad=True);loss=robust_overshoot_cost(v,target,meta);loss.backward()
 assert v.grad<0  # Gradient descent increases vx toward -.6.
 assert loss<overshoot_cost(v,target,meta)

def test_quadratic_near_boundary_and_finite_far_tail():
 target=torch.tensor([.5]);m=torch.tensor([[1.,.2]])
 v=torch.tensor([.5+.036+.00002]);old=overshoot_cost(v,target,m);new=robust_overshoot_cost(v,target,m)
 assert torch.allclose(new,old,rtol=1e-4,atol=1e-10)
 v=torch.tensor([100.],requires_grad=True);robust_overshoot_cost(v,target,m).backward()
 assert torch.isfinite(v.grad).all() and v.grad.abs().max()<=101
 assert robust_overshoot_cost(v,target,torch.zeros_like(m))==0
