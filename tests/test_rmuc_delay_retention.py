import torch,pytest
from tensordict import TensorDict
from rmuc_delay_retention import protected_mask,add_kl_gradient,KEY

def test_mask_protects_low_middle_and_other_skills_not_high_target():
 d=torch.tensor([0,1,2,3,4,4]);g=torch.tensor([1.,1.,1.,1.,.5,0.]);m=protected_mask(d,g)
 assert m.flatten().tolist()==[True,True,True,False,False,True]
 z=TensorDict({'policy':torch.arange(6.)[:,None],KEY:m},batch_size=[6]);order=torch.tensor([5,2,4,0,3,1]);assert torch.equal(z[order][KEY],m[order])

def test_centered_distribution_can_learn_targets_at_all_delays_without_releasing_other_commands():
 d=torch.tensor([0,1,2,3,4,0,4]);g=torch.tensor([1.,1.,1.,1.,1.,0.,0.])
 assert protected_mask(d,g,target_all_delays=True).flatten().tolist()==[False]*5+[True,True]
 assert protected_mask(d,g).flatten().tolist()==[True,True,True,False,False,True,True]

def test_gradient_matches_explicit_kl_and_output_unchanged():
 torch.manual_seed(42);a=torch.randn(7,6,requires_grad=True);b=a.detach().clone().requires_grad_(True);ref=torch.randn(7,6);mask=torch.tensor([1,0,1,0,1,0,0],dtype=torch.bool)[:,None];std=torch.ones(6)*.03;before=a.detach().clone();diag=[]
 loss=add_kl_gradient(a,ref,mask,std,1.,diag);a.square().mean().backward();explicit=.5*((b-ref).square()/std.square()*mask).sum()/(mask.sum()*6);(b.square().mean()+explicit).backward();torch.testing.assert_close(a.grad,b.grad);assert torch.equal(a.detach(),before);assert len(diag)==1;assert loss==pytest.approx(float(explicit.detach()))

def test_empty_mask_does_not_change_gradient():
 a=torch.ones(2,6,requires_grad=True);add_kl_gradient(a,torch.zeros_like(a),torch.zeros(2,1,dtype=torch.bool),torch.ones(6),1,[]);a.sum().backward();assert torch.equal(a.grad,torch.ones_like(a))

def test_execution_proxy_releases_same_clipped_mean_only_in_target():
 from rmuc_delay_retention import retention_delta
 obs=torch.zeros(4,25);obs[:,7]=torch.tensor([.25,.25,.25,.5]);obs[:,8]=.3
 mean=torch.zeros(4,6);ref=mean.clone();ref[:,3]=1.1;mean[:,3]=torch.tensor([1.02,.98,-1.1,1.02])
 delta,derivative=retention_delta(mean,ref,obs,[[.5,.3]])
 torch.testing.assert_close(delta[:,3],torch.tensor([0.,-.02,-2.,-.08]))
 torch.testing.assert_close(derivative[:,3],torch.tensor([0.,1.,0.,1.]))
 raw,j=retention_delta(mean,ref)
 torch.testing.assert_close(raw,mean-ref);assert torch.all(j==1)

def test_execution_proxy_keeps_feasible_reference_raw_retention():
 from rmuc_delay_retention import retention_delta
 obs=torch.zeros(1,25);obs[:,7]=.25;obs[:,8]=.3
 mean=torch.zeros(1,6);ref=mean.clone();mean[0,3]=1.1;ref[0,3]=.9
 delta,j=retention_delta(mean,ref,obs,[[.5,.3]])
 torch.testing.assert_close(delta,mean-ref);assert torch.all(j==1)
