import torch,pytest
from rmuc_target_policy_gradient import mask_target_advantages

def test_masks_only_selected_advantages_without_mutation_or_recenter():
 a=torch.tensor([[[2.],[-3.]],[[4.],[-5.]]]);b=a.clone();g=torch.tensor([[1.,0.],[.5,0.]])
 y,r=mask_target_advantages(a,g);assert torch.equal(a,b);assert torch.equal(y,torch.tensor([[[2.],[0.]],[[4.],[0.]]]));assert r['target_samples']==2

def test_no_target_fails_closed():
 with pytest.raises(ValueError):mask_target_advantages(torch.ones(2,3,1),torch.zeros(2,3))
 with pytest.raises(ValueError):mask_target_advantages(torch.ones(2,3,1),torch.ones(3,2))
