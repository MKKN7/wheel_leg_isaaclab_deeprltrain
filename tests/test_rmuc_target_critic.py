import sys
from pathlib import Path
import torch,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rmuc_world_model'))
from target_critic import update_target

def test_ema_updates_target_only():
 a=torch.nn.Linear(2,1);b=torch.nn.Linear(2,1)
 with torch.no_grad():
  a.weight.zero_();a.bias.zero_();b.weight.fill_(2);b.bias.fill_(4)
 update_target(a,b,.1)
 assert torch.allclose(a.weight,torch.full_like(a.weight,.2))
 assert torch.allclose(a.bias,torch.full_like(a.bias,.4))
 assert (b.weight==2).all() and (b.bias==4).all()
 assert a.weight.grad is None
 with pytest.raises(ValueError):update_target(a,b,0)
