import torch
from rmuc_command_sequence import reverse_runaway_cost

def cost(vx, previous, accel, target=-.8, yaw=0., valid=True):
 return reverse_runaway_cost(torch.tensor([[target,0.,yaw]]),torch.tensor([vx]),torch.tensor([previous]),torch.tensor([accel]),torch.tensor([valid]),.008).item()

def test_warns_before_crossing_only_when_still_accelerating_backward():
 assert cost(-.78,-.772,1.)>0
 assert cost(-.78,-.788,1.)==0  # filtered lag must not punish real braking
 assert cost(-.78,-.772,-1.)==0
 assert cost(-.78,-.772,1.,valid=False)==0

def test_correct_braking_has_only_existing_speed_excess_cost():
 worsening=cost(-.9,-.892,1.)
 braking=cost(-.9,-.908,1.)
 assert worsening>braking>0
 assert abs(braking-.4)<1e-5

def test_severity_distinguishes_runaway_until_bounded_cap():
 assert cost(-1.,-1.,0.)<cost(-1.3,-1.3,0.)<cost(-1.6,-1.6,0.)
 assert 0<=cost(-100.,-99.,100.)<=6

def test_no_forward_idle_or_yaw_changes():
 assert cost(-2.,-1.,100.,target=.5)==0
 assert cost(-2.,-1.,100.,target=0.)==0
 assert cost(-2.,-1.,100.,yaw=.3)==0

def test_command_step_alone_does_not_create_acceleration_surcharge():
 # Reverse deceleration changes target, but measured velocity is unchanged.
 assert abs(cost(-.4,-.4,0.,target=-.25)-.65)<1e-5
 assert cost(-.4,-.4,0.,target=-1.)==0
