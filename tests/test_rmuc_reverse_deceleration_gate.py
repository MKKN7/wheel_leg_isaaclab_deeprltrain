import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rmuc_world_model'))
from reverse_deceleration_gate import review

def test_slow_settling_allowed_but_rebound_and_incomplete_rejected():
    event=dict(old=-.4,target=-.25,overshoot_fraction=.04,arrival_s=2.5)
    assert review(event,True,True)['passed']
    assert not review(dict(event,overshoot_fraction=.5),True,True)['passed']
    assert not review(event,False,True)['passed']
    assert not review(event,True,False)['passed']

def test_reverse_acceleration_unchanged():
    assert not review(dict(old=-.25,target=-.4,overshoot_fraction=0),True,True)['applicable']
