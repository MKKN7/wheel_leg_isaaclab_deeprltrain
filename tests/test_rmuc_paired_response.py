import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rmuc_world_model'))
from paired_response import PairedResponse


def test_wrong_action_response_penalized_even_with_small_absolute_error():
    obj=object.__new__(PairedResponse)
    obj.metric=lambda x:x
    action=torch.ones(2,32,5)*.002
    obj.base={'state':torch.zeros(2,4,5),'history':torch.zeros(2,4,5),'action':torch.zeros_like(action),'target':torch.zeros_like(action),'context':torch.zeros(2,32,1)}
    obj.probe=dict(obj.base,action=action,target=action)
    class Model:
        def __init__(self,sign):self.sign=torch.tensor(sign,requires_grad=True)
        def rollout(self,s,h,a,c):return a*self.sign
    good=Model(1.);bad=Model(-1.)
    assert obj.loss(good)==0
    loss=obj.loss(bad);assert loss>0;loss.backward();assert bad.sign.grad<0


def test_common_mode_drift_requires_absolute_constraint():
    obj=object.__new__(PairedResponse);obj.metric=lambda x:x
    action=torch.ones(2,32,5)*.002
    obj.base={'state':torch.zeros(2,4,5),'history':torch.zeros(2,4,5),'action':torch.zeros_like(action),'target':torch.zeros_like(action),'context':torch.zeros(2,32,1)}
    obj.probe=dict(obj.base,action=action,target=action)
    class Model:
        def __init__(self):self.bias=torch.tensor(.01,requires_grad=True)
        def rollout(self,s,h,a,c):return a+self.bias
    m=Model();assert obj.loss(m)<1e-10
    loss=obj.loss(m,.1);assert loss>0;loss.backward();assert m.bias.grad>0


def test_policy_pair_requires_identical_prestate_and_holds_out_envs(tmp_path, monkeypatch):
    import json
    import paired_response
    import pytest
    baseline, candidate = tmp_path/'base', tmp_path/'candidate'
    for path, sha in [(baseline, 'a'), (candidate, 'b')]:
        path.mkdir()
        (path/'evaluation.json').write_text(json.dumps(dict(checkpoint_sha256=sha, seed=1,
            initial_state_audit={}, initial_mass_inertia_readback={}, initial_material_readback={})))
    def seq(env):
        return dict(env=env,state=torch.zeros(40,5),action=torch.zeros(40,6),context=torch.zeros(40,2))
    base=[seq(0),seq(3)]; cand=[seq(0),seq(3)]
    meta=dict(state_columns=['x']*5,extra_state_names=[])
    monkeypatch.setattr(paired_response,'load_capture',lambda path,*args:(base if path==baseline else cand,meta))
    result=PairedResponse(baseline,candidate,'cpu',policy_start_seconds=.032)
    assert result.ids == [0]
    cand[0]['state'][3,0]=1
    with pytest.raises(AssertionError):
        PairedResponse(baseline,candidate,'cpu',policy_start_seconds=.032)
    with pytest.raises(ValueError,match='checkpoint_sha256'):
        PairedResponse(baseline,candidate,'cpu')
