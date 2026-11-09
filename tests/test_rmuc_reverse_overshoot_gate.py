import numpy as np
from scripts.rmuc_world_model.audit_reverse_overshoot_gate import reverse_speed_checks
from scripts.rmuc_world_model.audit_reverse_overshoot_gate import reverse_angle_check

def test_reverse_angle_relaxation_preserves_finite_two_joint_boundary():
    assert reverse_angle_check([3.,4.])
    assert not reverse_angle_check([3.,4.001])
    assert not reverse_angle_check([float('nan'),1.])
    assert not reverse_angle_check([1.])
    assert not reverse_angle_check([3.,3.],2.)

def test_slow_stable_preferred_to_accurate_tail_after_runaway():
    steady=np.array([False,False,True,True])
    slow=reverse_speed_checks([0,-.4,-.62,-.62],-.8,steady,True)
    assert slow['underspeed_pass'] and slow['overshoot_pass']
    runaway=reverse_speed_checks([0,-1.6,-.8,-.8],-.8,steady,True)
    assert runaway['underspeed_pass'] and not runaway['overshoot_pass']

def test_standing_still_and_incomplete_do_not_pass():
    s=reverse_speed_checks([0,0,0],-.8,[False,True,True],True)
    assert not s['underspeed_pass']
    assert not reverse_speed_checks([0,-.6,-.6],-.8,[False,True,True],False)['overshoot_pass']
