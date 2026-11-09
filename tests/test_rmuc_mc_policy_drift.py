import pytest
import torch
from rmuc_command_value_training import mc_policy_drift


def test_one_changed_command_cannot_be_hidden_by_pooled_mean():
    ref=torch.zeros(8*64,6)
    mean=ref.clone();mean[3*64:4*64,2]=.06
    expected=torch.zeros(8);expected[3]=2.
    torch.testing.assert_close(mc_policy_drift(mean,ref,8),expected)
    assert float(mc_policy_drift(mean,ref,8).max())>.5
    assert float(mc_policy_drift(mean,ref,8).mean())<.5


def test_nonfinite_probe_is_rejected():
    ref=torch.zeros(64,6);bad=ref.clone();bad[0,0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):
        mc_policy_drift(bad,ref,1)
