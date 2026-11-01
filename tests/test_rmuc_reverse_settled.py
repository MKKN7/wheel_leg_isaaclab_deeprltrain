import torch
from rmuc_command_sequence import reverse_settled_tracking_cost,validate_sequence

def test_sequence_entry_excluded_and_symmetric_precision():
 table=validate_sequence([[0,0,0],[8,-.8,0]],.008,12)
 age=torch.tensor([1000,1250,1375,1375]);cmd=torch.tensor([[-.8,0.,0.]]*4);v=torch.tensor([-.72,-.72,-.72,-.88])
 cost=reverse_settled_tracking_cost(cmd,v,age,table,torch.ones(4,dtype=torch.bool),.008)
 torch.testing.assert_close(cost,torch.tensor([0.,0.,.5,.5]),atol=1e-6,rtol=1e-6)

def test_fixed_commands_old_domain_turn_and_reset():
 cmd=torch.tensor([[-.8,0,0],[-.6,0,0],[.8,0,0],[-.8,0,.3],[-.8,0,0]])
 cost=reverse_settled_tracking_cost(cmd,torch.zeros(5),torch.tensor([500,500,500,500,0]),None,torch.zeros(5,dtype=torch.bool),.008)
 assert 0<cost[0]<1 and not cost[1:].any()
