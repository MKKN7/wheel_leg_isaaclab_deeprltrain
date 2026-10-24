"""Independent attitude-estimate error, never a modification of true gravity.

A per-reset clean cohort and private RNG preserve unrelated DR sampling.
All angles are engineering priors, not identified sensor calibration.
"""
import math
import torch

class GravityDirectionDR:
    def __init__(self, count, device, *, strength=0., probability=.5,
                 noise_deg=.1, bias_deg=.2, walk_deg_sqrt_s=.01,
                 drift_bound_deg=.1, seed=1047):
        vals=[strength,probability,noise_deg,bias_deg,walk_deg_sqrt_s,drift_bound_deg]
        if any(not math.isfinite(v) or v<0 for v in vals) or strength>1 or probability>1:
            raise ValueError('Finite nonnegative gravity DR values; strength/probability <=1')
        self.strength=strength;self.probability=probability
        self.noise=math.radians(noise_deg)*strength
        self.bias_max=math.radians(bias_deg)*strength
        self.walk=math.radians(walk_deg_sqrt_s)*strength
        self.bound=math.radians(drift_bound_deg)*strength
        self.generator=torch.Generator(device=device).manual_seed(seed)
        self.bias=torch.zeros(count,3,device=device);self.drift=torch.zeros_like(self.bias)
        self.active=torch.zeros(count,dtype=torch.bool,device=device)
        self.last_error_deg=torch.zeros(count,device=device);self.calls=0
    def reset(self, ids):
        self.drift[ids]=0;self.last_error_deg[ids]=0
        if self.strength==0:return
        n=ids.numel();device=self.bias.device
        self.active[ids]=torch.rand(n,device=device,generator=self.generator)<self.probability
        self.bias[ids]=(2*torch.rand(n,3,device=device,generator=self.generator)-1)*self.bias_max
    def apply(self, observation, dt):
        if self.strength==0:return observation
        if not math.isfinite(dt) or dt<=0:raise ValueError('Positive finite dt required')
        self.calls+=1
        def randn():return torch.randn(self.bias.shape,device=self.bias.device,generator=self.generator)
        self.drift.add_(randn()*self.walk*math.sqrt(dt)).clamp_(-self.bound,self.bound)
        rotation=self.bias+self.drift+randn()*self.noise
        angle=torch.linalg.vector_norm(rotation,dim=-1,keepdim=True)
        axis=rotation/angle.clamp_min(1e-12);g=observation[:,3:6]
        rotated=g*torch.cos(angle)+torch.linalg.cross(axis,g)*torch.sin(angle)+axis*(axis*g).sum(-1,keepdim=True)*(1-torch.cos(angle))
        out=observation.clone();out[:,3:6]=torch.where(self.active[:,None],rotated,g)
        self.last_error_deg.copy_(torch.rad2deg(torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(g,out[:,3:6]),dim=-1),(g*out[:,3:6]).sum(-1))))
        return out
    def readback(self):
        return dict(strength=self.strength,active_count=int(self.active.sum()),total=int(self.active.numel()),calls=self.calls,noise_std_deg=math.degrees(self.noise),bias_component_bound_deg=math.degrees(self.bias_max),drift_component_bound_deg=math.degrees(self.bound),walk_deg_sqrt_s=math.degrees(self.walk),last_direction_error_max_deg=float(self.last_error_deg.max()),scope='Additional gravity rotation only; clean cohort retains preexisting sensor noise')
