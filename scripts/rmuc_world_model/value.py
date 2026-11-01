"""Training-only stop value model; actor interface remains unchanged."""
import torch
from torch import nn
class StopValue(nn.Module):
    def __init__(self,mean,scale,target_scale):
        super().__init__();self.register_buffer('mean',mean.clone());self.register_buffer('scale',scale.clone());self.register_buffer('target_scale',torch.as_tensor(target_scale).clone())
        self.net=nn.Sequential(nn.Linear(len(mean),128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1),nn.Softplus())
    def forward(self,state):return self.net((state-self.mean)/self.scale).squeeze(-1)*self.target_scale

def discounted_cost_to_go(cost,gamma):
    result=torch.zeros_like(cost);value=torch.zeros_like(cost[0])
    for i in range(len(cost)-1,-1,-1):
        value=cost[i]+gamma*value;result[i]=value
    return result


def stopping_stage_cost(vx,pitch,leg,overshoot_weight=0.,dt=.008,*,profile="legacy",yaw=None,hip=None):
    """Final .5->0 diagnostic: penalize reverse rebound beyond .09 m/s.

    .09 is a shaping margin below the unchanged .10 m/s (20%) acceptance line.
    This is scoped to the stopping event, never applies to commanded reverse.
    """
    if profile=='settling_v1':
        if yaw is None or hip is None:raise ValueError('Settling cost requires actual yaw and hip difference')
        huber=lambda x:(1+x.square()).sqrt()-1
        speed=huber(vx/.02)
        yaw_cost=.25*huber(yaw/.05)
        posture=.1*(torch.relu(pitch.abs()-torch.deg2rad(vx.new_tensor(5.)))/.08726646).square()
        angles=.1*((torch.relu(leg.abs()-.034906585)/.034906585).square()+(torch.relu(hip.abs()-.034906585)/.034906585).square())
        reverse=(torch.relu(-vx-.09)/.01).square()
        return dt*(speed+yaw_cost+posture+angles+reverse)
    if profile!='legacy':raise ValueError('Unknown stopping cost')
    return dt*(vx.square()+.2*pitch.square()+.05*leg.square()+overshoot_weight*torch.relu(-vx-.09).square())
