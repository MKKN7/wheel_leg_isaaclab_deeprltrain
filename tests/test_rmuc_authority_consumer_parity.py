from pathlib import Path
import sys
import numpy as np
import torch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/rsl_rl'),str(ROOT/'sim2real_RL/sim2sim')]
from rmuc_position_authority import execute_authority_action
from rmuc_policy_interface import consume_action
@pytest.mark.parametrize('end',[1.,1.5])
def test_authority_boundaries_match_training(end):
    speeds=np.array([0.,.8,1.,1.2,1.5,1.7,end+.2],dtype=np.float32)
    speeds=np.unique(np.concatenate([speeds,np.nextafter(speeds,np.float32(-np.inf)),np.nextafter(speeds,np.float32(np.inf))]))
    obs=[]
    for v in speeds:
        for yaw in [-2.,-.6,-.4501,0.,.4501,.6,2.]:
            o=np.zeros(25,np.float32);o[7]=v*.5;o[8]=yaw;obs.append(o)
    obs=np.stack(obs);actions=np.tile(np.array([2.,-2.,2.,-2.,2.,-2.],np.float32),(len(obs),1))
    expected=execute_authority_action(torch.from_numpy(obs),torch.from_numpy(actions),end).numpy()
    actual=np.stack([consume_action(o,a,end) for o,a in zip(obs,actions)])
    np.testing.assert_array_equal(actual,expected)
