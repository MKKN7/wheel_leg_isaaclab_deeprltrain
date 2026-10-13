import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rmuc_eval_dr_streams import WheelDRStreams

def streams():return WheelDRStreams(3,20,67,.1,.5,.05,'cpu')

def test_other_environment_resets_do_not_change_sequence_or_global_rng():
    state=torch.get_rng_state().clone();a=streams();b=streams()
    for _ in range(8):
        expected=a.sample(torch.tensor([1]))
        b.sample(torch.tensor([0,2]));b.sample(torch.tensor([2]))
        assert torch.equal(expected,b.sample(torch.tensor([1])))
    assert torch.equal(state,torch.get_rng_state())

def test_batch_order_does_not_change_draws_and_bounds_hold():
    a=streams();b=streams()
    for _ in range(5):
        expected=a.sample(torch.tensor([0,1,2]))
        actual=b.sample(torch.tensor([2,0,1]))[[1,2,0]]
        assert torch.equal(expected,actual)
        assert bool(((actual[:,0]>=.95)&(actual[:,0]<=1.05)).all())
        assert bool(((actual[:,1]>=.995)&(actual[:,1]<=1.005)).all())
