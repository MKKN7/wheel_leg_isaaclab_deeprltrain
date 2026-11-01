"""Differentiable surrogate prototype; no simulator, teacher or MCU modifications."""
import torch
from torch import nn

class Dynamics(nn.Module):
    def __init__(self,mean,scale,delta_scale,context_mean,context_scale,history=4,width=128,depth=2):
        super().__init__();self.history=history
        for key,val in [('mean',mean),('scale',scale),('delta_scale',delta_scale),('context_mean',context_mean),('context_scale',context_scale)]:self.register_buffer(key,val.clone())
        if depth < 1 or width < 1:raise ValueError('Positive model width/depth required')
        layers=[nn.Linear(history*(len(mean)+6)+6+len(context_mean),width),nn.SiLU()]
        for _ in range(depth-1):layers.extend([nn.Linear(width,width),nn.SiLU()])
        self.net=nn.Sequential(*layers,nn.Linear(width,len(mean)))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,states,action_history,action,context):
        x=torch.cat([((states-self.mean)/self.scale).flatten(1),action_history.flatten(1),action,(context-self.context_mean)/self.context_scale],-1)
        return states[:,-1]+self.delta_scale*self.net(x)
    def rollout(self,states,action_history,actions,context):
        results=[]
        for step,a in enumerate(actions.unbind(1)):
            nxt=self(states,action_history,a,context[:,step] if context.ndim==3 else context);results.append(nxt)
            states=torch.cat([states[:,1:],nxt[:,None]],1)
            action_history=torch.cat([action_history[:,1:],a[:,None]],1)
        return torch.stack(results,1)
