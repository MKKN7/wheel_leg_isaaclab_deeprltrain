import pytest
import torch
from rmuc_command_balance import balance_advantages


def test_balanced_counts_preserve_advantages_and_signs():
    advantage=torch.tensor([[[1.],[-2.],[3.],[-4.]]])
    commands=torch.tensor([[[.5,0,0],[.5,0,0],[1.,0,.8],[1.,0,.8]]])
    result,report=balance_advantages(advantage,commands)
    torch.testing.assert_close(result,advantage,atol=0,rtol=0)
    assert report['commands']==2


def test_dominant_command_attenuated_without_inverting_credit():
    command=torch.zeros(2,10,3);command[:,:9,0]=1.;command[:,9,0]=-.25
    advantage=torch.ones(2,10,1);advantage[:,::2]=-1
    result,report=balance_advantages(advantage,command)
    ratio=result/advantage
    assert ratio[:,:9].max()<ratio[:,9].min()
    assert torch.equal(result.sign(),advantage.sign())
    torch.testing.assert_close(ratio.mean(),torch.tensor(1.))
    assert report['maximum_weight']/report['minimum_weight']<=4.00001


def test_misaligned_labels_and_nonfinite_inputs_rejected():
    with pytest.raises(ValueError):balance_advantages(torch.ones(2,3,1),torch.zeros(2,4,3))
    with pytest.raises(ValueError):balance_advantages(torch.full((2,3,1),float('nan')),torch.zeros(2,3,3))


def test_inference_storage_can_receive_weighted_advantages():
    with torch.inference_mode():
        storage=torch.ones(2,10,1)
        labels=torch.zeros(2,10,3)
        labels[:,:9,0]=1
    weighted,_=balance_advantages(storage,labels)
    with torch.inference_mode():
        storage.copy_(weighted)
    assert torch.isfinite(storage).all()
