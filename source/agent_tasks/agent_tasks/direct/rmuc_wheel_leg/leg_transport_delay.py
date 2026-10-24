"""Post-host-PD, pre-motor-response leg command transport (integer physics steps)."""
import math
import torch

class LegTransportDelay:
    def __init__(self, template, dt, delay_s=0., probability=.5, seed=2047):
        if not all(math.isfinite(v) for v in [dt,delay_s,probability]) or dt<=0 or not 0<=delay_s<=.012 or not 0<=probability<=1:
            raise ValueError('Leg delay <=12ms and valid timing/probability required')
        self.steps=round(delay_s/dt)
        if abs(self.steps*dt-delay_s)>1e-9:raise ValueError('Leg transport must align with physics steps')
        self.dt=dt;self.probability=probability;self.calls=0
        self.history=template.new_zeros((self.steps+1,*template.shape))
        self.active=torch.zeros(template.shape[0],device=template.device,dtype=torch.bool)
        self.generator=torch.Generator(device=template.device).manual_seed(seed)
    def reset(self, ids):
        self.history[:,ids]=0
        if self.steps:
            n=self.active[ids].numel();self.active[ids]=torch.rand(n,device=self.active.device,generator=self.generator)<self.probability
    def step(self, effort):
        if not self.steps:return effort
        self.calls+=1
        self.history[1:]=self.history[:-1].clone();self.history[0]=effort
        return torch.where(self.active[:,None],self.history[-1],effort)
    def readback(self):
        return dict(delay_s=self.steps*self.dt,physics_steps=self.steps,active_count=int(self.active.sum()),total=self.active.numel(),probability=self.probability,calls=self.calls,scope='All leg joints share per-environment transport; post PD/clipping before response; reset history zero')

def install_leg_transport(actuator, dt, delay_s, probability, seed):
    if hasattr(actuator,'_rmuc_leg_transport'):raise ValueError('Leg transport already installed')
    delay=LegTransportDelay(actuator.damping,dt,delay_s,probability,seed)
    compute_original=actuator.compute;reset_original=actuator.reset
    def compute(action, joint_pos, joint_vel):
        result=compute_original(action,joint_pos,joint_vel)
        result.joint_efforts=delay.step(result.joint_efforts)
        actuator.applied_effort=result.joint_efforts
        return result
    def reset(ids):
        reset_original(ids);delay.reset(ids)
    actuator.compute=compute;actuator.reset=reset;actuator._rmuc_leg_transport=delay
    return delay
