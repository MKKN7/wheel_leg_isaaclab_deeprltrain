import torch
from rmuc_world_model.prepare_dynamic_course import ramp_sequence
from rmuc_command_sequence import validate_sequence,sequence_commands

def test_vx_and_yaw_interpolate_together_and_keep_interface():
    events=[[0,0,0],[2,1.,.6],[5,.5,-.6]]
    seq=ramp_sequence(events);table=validate_sequence(seq,.008,10.)
    got=sequence_commands(table,torch.tensor([250,300,350,625,675,725]))
    torch.testing.assert_close(got[:,[0,2]],torch.tensor([[0.,0.],[.5,.3],[1.,.6],[1.,.6],[.75,0.],[.5,-.6]]))
    assert (got[:,1]==0).all()

def test_constant_speed_yaw_transition_is_retained():
    seq=ramp_sequence([[0,.5,.3],[2,.5,-.3]])
    assert all(row[1]==.5 for row in seq)
    assert seq[-1]==[2.8,.5,-.3]
