"""Independent reset-sampled physical-tick queues. No interpolation or packet reordering."""
import math
import torch

class SampledDelay:
    def __init__(self, template, dt, bounds_s, seed, fill_initial=False):
        if len(bounds_s)!=2 or not all(math.isfinite(v) for v in [dt,*bounds_s]) or dt<=0 or not 0<=bounds_s[0]<=bounds_s[1]<=.012:
            raise ValueError('Ordered finite delay bounds within0..12ms required')
        steps=[round(v/dt) for v in bounds_s]
        if any(abs(v-n*dt)>1e-8 for v,n in zip(bounds_s,steps)):raise ValueError('Delay range must align with physics ticks')
        self.dt=dt;self.bounds=steps;self.seed=seed;self.fill_initial=fill_initial
        self.history=template.new_zeros((template.shape[0],steps[1]+1,template.shape[1]))
        self.steps=torch.zeros(template.shape[0],device=template.device,dtype=torch.long)
        self.valid=torch.zeros_like(self.steps,dtype=torch.bool);self.last_tick=None
        self.rng=torch.Generator(device=template.device).manual_seed(seed);self.reset(torch.arange(template.shape[0],device=template.device))
    def reset(self, ids):
        if ids is None:ids=torch.arange(len(self.steps),device=self.steps.device)
        self.history[ids]=0;self.valid[ids]=False
        self.steps[ids]=torch.randint(self.bounds[0],self.bounds[1]+1,(len(ids),),generator=self.rng,device=self.steps.device)
    def push(self,value,tick=None):
        if tick is not None and self.last_tick==tick:
            # Partial reset at the same timestamp primes only the reset histories.
            invalid=~self.valid
            if self.fill_initial:self.history[invalid]=value[invalid,None,:]
            self.valid[invalid]=True
            return
        self.history[:,1:]=self.history[:,:-1].clone();self.history[:,0]=value
        if self.fill_initial:self.history[~self.valid]=value[~self.valid,None,:]
        self.valid[:]=True;self.last_tick=tick
    def read(self):
        return self.history[torch.arange(len(self.steps),device=self.steps.device),self.steps].clone()
    def step(self,value):self.push(value);return self.read()
    def readback(self):return dict(bounds_ms=[n*self.dt*1000 for n in self.bounds],steps_per_env=self.steps.tolist(),effective_ms_per_env=(self.steps*self.dt*1000).tolist(),sampling='per_environment_per_reset_private_rng',seed=self.seed,dt_s=self.dt)

class JointDelayDR:
    def __init__(self, actions, dt, ranges, seed):
        self.observation=SampledDelay(actions.new_zeros((len(actions),25)),dt,ranges['observation'],seed+1,True)
        self.action=SampledDelay(actions,dt,ranges['action'],seed+2)
        self.wheel=SampledDelay(actions[:,:2],dt,ranges['wheel'],seed+3)
        self.leg=SampledDelay(actions[:,:4],dt,ranges['leg'],seed+4)
        self.pending=torch.zeros_like(actions)
    def reset(self,ids):
        for q in [self.observation,self.action,self.wheel,self.leg]:q.reset(ids)
        self.pending[ids]=0
    def readback(self):return {name:getattr(self,name).readback() for name in ['observation','action','wheel','leg']}
