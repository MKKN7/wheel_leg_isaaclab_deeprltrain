import sys,copy
from pathlib import Path
import pytest,torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_leg_length_actor import LegLengthConditionedActor

def test_zero_migration_and_old_normalizer_frozen_through_training(tmp_path):
 torch.manual_seed(26)
 source=nn.Sequential(nn.BatchNorm1d(25),nn.Linear(25,6)).eval()
 actor=LegLengthConditionedActor(source,[.1,.1,.02,.1,.1,.02]);x=torch.randn(128,26)
 baseline=source(x[:,:25]).detach();assert torch.equal(actor(x),baseline)
 before={k:v.clone() for k,v in actor.source25.state_dict().items()};actor.train()
 opt=torch.optim.Adam(actor.length_network.parameters(),lr=.001)
 for _ in range(3):
  opt.zero_grad();loss=(actor(x)-baseline-.03*x[:,25:26]).square().mean();loss.backward();opt.step()
 assert all(torch.equal(v,actor.source25.state_dict()[k]) for k,v in before.items())
 assert all(v.grad is None for v in actor.source25.parameters())
 assert not actor.source25.training
 delta=actor(x)-baseline;assert (delta.abs()<=actor.action_budget+1e-7).all();assert delta.abs().max()>0
 file=tmp_path/'weights.pt';torch.save(actor.state_dict(),file)
 restored=LegLengthConditionedActor(source,[0]*6);restored.load_state_dict(torch.load(file,weights_only=True));assert torch.equal(restored(x),actor(x))
 assert all(torch.equal(v,source.state_dict()[k]) for k,v in before.items())

@pytest.mark.parametrize('budget',[[1]*5,[-.1]*6,[float('nan')]*6,[1.1]*6])
def test_reject_bad_budget(budget):
 with pytest.raises(ValueError):LegLengthConditionedActor(nn.Linear(25,6),budget)

def test_requires_explicit_new_input_dimension():
 actor=LegLengthConditionedActor(nn.Linear(25,6),[.1]*6)
 with pytest.raises(ValueError):actor(torch.zeros(1,25))
