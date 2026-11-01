import sys
from pathlib import Path
import torch,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rmuc_world_model'))
from replay_window import prefix_sequences

def test_no_crossing_no_mutation():
 q={'state':torch.zeros(100,2),'action':torch.zeros(100,6),'context':torch.zeros(100,3),'observations':torch.zeros(100,25),'starts':torch.arange(4,69),'env':0}
 out=prefix_sequences([q],.4)[0]
 assert len(out['state'])==50 and len(q['state'])==100
 assert int((out['starts']+32).max())<=50
 assert out['env']==0 and len(out['context'])==50
 with pytest.raises(ValueError):prefix_sequences([q],.01)
