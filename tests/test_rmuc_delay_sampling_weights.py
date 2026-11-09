from pathlib import Path
import runpy,torch,pytest
Q=runpy.run_path(str(Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/joint_delay_dr.py'))['SampledDelay']
def test_default_rng_unchanged_and_weighted_full_support():
 q=Q(torch.zeros(10000,6),.002,[0,.008],12);g=torch.Generator().manual_seed(12);assert torch.equal(q.steps,torch.randint(0,5,(10000,),generator=g))
 q.set_sampling_weights([1,1,1,3.5,3.5]);q.reset(None);counts=torch.bincount(q.steps,minlength=5);assert (counts>0).all();assert .66<float((q.steps>=3).float().mean())<.74
 assert q.readback()['bounds_ms']==[0.,8.];assert abs(sum(q.readback()['sampling_probabilities'])-1)<1e-6
 q.push(torch.ones(10000,6));assert q.history.shape[1]==5

def test_bad_weights_rejected():
 q=Q(torch.zeros(2,6),.002,[0,.008],3)
 for w in [[1,2],[1,1,1,1,0],[1,1,1,1,float('nan')]]:
  with pytest.raises(ValueError):q.set_sampling_weights(w)
