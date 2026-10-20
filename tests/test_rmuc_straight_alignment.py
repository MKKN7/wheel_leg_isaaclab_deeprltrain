import runpy,math
from pathlib import Path
import torch
import pytest
f=runpy.run_path(str(Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/straight_alignment.py'))['alignment_cost']
def test_mirrored_physical_alignment():
 q=torch.tensor([[.2,.3,0,.2,-.3,0.]])
 c,d,g=f(q,torch.zeros(1),torch.zeros(1));assert c.item()==0 and torch.equal(d,torch.zeros_like(d))
def test_turns_and_spin_never_penalized():
 q=torch.ones(4,6);c,_,_=f(q,torch.tensor([.001,-.1,1.4,-8.]),torch.zeros(4));assert torch.equal(c,torch.zeros(4))
def test_straight_idle_deadband_and_gradient():
 q=torch.zeros(2,6);q[:,0]=torch.tensor([math.radians(1),math.radians(8)]);q.requires_grad_()
 c,_,_=f(q,torch.zeros(2),torch.zeros(2));assert c[0]==0 and 0<c[1]<1;c.sum().backward();assert q.grad[1,0]>0 and q.grad[1,3]<0
 defcost=f(q.detach(),torch.zeros(2),torch.full((2,),.3))[0];assert (defcost==0).all()

def test_reverse_scale_preserves_forward_idle_and_excludes_turns():
 q=torch.zeros(6,6);q[:,0]=math.radians(3);q[:,1]=math.radians(2.5)
 vx=torch.tensor([-.1,-.4,0.,.3,-.4,-.1]);yaw=torch.tensor([0.,0.,0.,0.,.1,-.1])
 old=f(q,yaw,torch.zeros(6),1.,6.)[0]
 new=f(q,yaw,torch.zeros(6),1.,6.,command_vx=vx,reverse_scale_deg=3.)[0]
 assert torch.equal(old[2:],new[2:])
 assert (new[:2]>old[:2]).all() and (new<1).all()
 assert torch.equal(old,f(q,yaw,torch.zeros(6),1.,6.,command_vx=vx)[0])

def test_reverse_scale_improves_near_gate_angle_gradient_without_changing_deadband():
 q=torch.zeros(2,6);q[:,0]=torch.tensor([math.radians(.5),math.radians(2.5)]);q.requires_grad_()
 vx=torch.full((2,),-.15);yaw=torch.zeros(2)
 old=f(q,yaw,yaw,1.,6.)[0];go=torch.autograd.grad(old.sum(),q)[0]
 new=f(q,yaw,yaw,1.,6.,command_vx=vx,reverse_scale_deg=3.)[0];gn=torch.autograd.grad(new.sum(),q)[0]
 assert new[0]==0 and gn[1,0]>go[1,0]>0 and gn[1,3]<go[1,3]<0
 for scale in [-1.,float('nan'),float('inf')]:
  with pytest.raises(ValueError):f(q,yaw,yaw,command_vx=vx,reverse_scale_deg=scale)
 with pytest.raises(ValueError):f(q,yaw,yaw,reverse_scale_deg=3.)
