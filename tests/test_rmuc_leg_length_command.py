import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rmuc_leg_length_command import append_leg_length_command,leg_length_tracking_cost

def test_preserves_all_legacy_slots_and_normalization():
 x=np.arange(50,dtype=np.float32).reshape(2,25);y=append_leg_length_command(x,[.34,.30])
 assert y.shape==(2,26);np.testing.assert_array_equal(y[:,:25],x);np.testing.assert_allclose(y[:,25],[0,-.4],atol=1e-6)
 np.testing.assert_allclose(append_leg_length_command(x,.36)[:,25],.2,atol=1e-6)

@pytest.mark.parametrize('target',[.16,.37,float('nan')])
def test_unvalidated_target_is_not_silently_clipped(target):
 with pytest.raises(ValueError):append_leg_length_command(np.zeros(25),target)

def test_rejects_26d_input_and_bad_envelope():
 with pytest.raises(ValueError):append_leg_length_command(np.zeros(26),.34)
 with pytest.raises(ValueError):append_leg_length_command(np.zeros(25),.34,low_m=.36,high_m=.30)

def test_cost_does_not_reward_lowering_below_command_or_kill_bank_difference():
 assert leg_length_tracking_cost(.34,.34,.34)==0
 assert leg_length_tracking_cost(.32,.36,.34)==0
 e=leg_length_tracking_cost([.30,.32,.34,.36,.38],[.30,.32,.34,.36,.38],.34)
 assert e[0]>e[1]>e[2] and e[4]>e[3]>e[2]
 np.testing.assert_allclose(e,e[::-1],atol=1e-12)
