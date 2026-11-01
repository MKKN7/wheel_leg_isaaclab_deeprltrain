import torch
import pytest
from rmuc_command_sequence import dynamic_tracking_cost, stop_tracking_scale_adjustment

def test_only_dynamic_zero_command_replaces_scale():
    cmd=torch.zeros(5,3);cmd[2,0]=.5;cmd[3,0]=-.35;cmd[4,2]=.5
    v=torch.full((5,),.1);mask=torch.tensor([True,False,True,True,True])
    delta=stop_tracking_scale_adjustment(cmd,v,mask,.5,.2)
    assert delta[0]>0 and torch.equal(delta[1:],torch.zeros(4))
    base=dynamic_tracking_cost(cmd[:,0],v,mask,.5)
    expected=dynamic_tracking_cost(cmd[:,0],v,mask,.2)
    torch.testing.assert_close((base+delta)[0],expected[0])

def test_disabled_and_symmetry_and_bound():
    v=torch.tensor([0.,.005,.01,.05,.1,.5,10.]);cmd=torch.zeros(7,3);mask=torch.ones(7,dtype=torch.bool)
    assert torch.equal(stop_tracking_scale_adjustment(cmd,v,mask,.5,0),torch.zeros_like(v))
    d=stop_tracking_scale_adjustment(cmd,v,mask,.5,.2)
    torch.testing.assert_close(d,stop_tracking_scale_adjustment(cmd,-v,mask,.5,.2))
    total=dynamic_tracking_cost(cmd[:,0],v,mask,.5)+d
    assert total.max()<=4 and total[0]==0
    with pytest.raises(ValueError):stop_tracking_scale_adjustment(cmd,v,mask,.5,float('nan'))
