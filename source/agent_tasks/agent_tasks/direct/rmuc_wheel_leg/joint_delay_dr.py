"""Independent reset-sampled physical-tick queues. No interpolation or packet reordering."""
import math
import torch

class SampledDelay:
    def __init__(self, template, dt, bounds_s, seed, fill_initial=False):
        if len(bounds_s)!=2 or not all(math.isfinite(v) for v in [dt,*bounds_s]) or dt<=0 or not 0<=bounds_s[0]<=bounds_s[1]<=.012:
            raise ValueError('Ordered finite delay bounds within0..12ms required')
        steps=[round(v/dt) for v in bounds_s]
        if any(abs(v-n*dt)>1e-8 for v,n in zip(bounds_s,steps)):raise ValueError('Delay range must align with physics ticks')
        self.sampling_weights=None
        self.dt=dt;self.bounds=steps;self.seed=seed;self.fill_initial=fill_initial
        self.history=template.new_zeros((template.shape[0],steps[1]+1,template.shape[1]))
        self.steps=torch.zeros(template.shape[0],device=template.device,dtype=torch.long)
        self.valid=torch.zeros_like(self.steps,dtype=torch.bool);self.last_tick=None
        self.rng=torch.Generator(device=template.device).manual_seed(seed);self.reset(torch.arange(template.shape[0],device=template.device))
    def reset(self, ids):
        if ids is None:ids=torch.arange(len(self.steps),device=self.steps.device)
        self.history[ids]=0;self.valid[ids]=False
        if self.sampling_weights is None:
            self.steps[ids]=torch.randint(self.bounds[0],self.bounds[1]+1,(len(ids),),generator=self.rng,device=self.steps.device)
        else:
            self.steps[ids]=torch.multinomial(self.sampling_weights,len(ids),replacement=True,generator=self.rng)+self.bounds[0]
    def set_sampling_weights(self, weights):
        w=torch.as_tensor(weights,device=self.steps.device,dtype=torch.float32)
        if w.shape!=(self.bounds[1]-self.bounds[0]+1,) or not torch.isfinite(w).all() or not (w>0).all():
            raise ValueError('Positive finite weights for every delay tick required')
        self.sampling_weights=w/w.sum()

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
    def readback(self):return dict(bounds_ms=[n*self.dt*1000 for n in self.bounds],steps_per_env=self.steps.tolist(),effective_ms_per_env=(self.steps*self.dt*1000).tolist(),sampling='per_environment_per_reset_private_rng',sampling_probabilities=None if self.sampling_weights is None else self.sampling_weights.tolist(),seed=self.seed,dt_s=self.dt)

class JointDelayDR:
    def __init__(self, actions, dt, ranges, seed, combinations=None):
        self.observation=SampledDelay(actions.new_zeros((len(actions),25)),dt,ranges['observation'],seed+1,True)
        self.action=SampledDelay(actions,dt,ranges['action'],seed+2)
        self.wheel=SampledDelay(actions[:,:2],dt,ranges['wheel'],seed+3)
        self.leg=SampledDelay(actions[:,:4],dt,ranges['leg'],seed+4)
        self.pending=torch.zeros_like(actions)
        self.combinations=None
        if combinations:
            table=torch.as_tensor(combinations,device=actions.device,dtype=torch.float64)
            if table.ndim!=2 or table.shape[1]!=5 or not torch.isfinite(table).all() or not (table[:,4]>0).all():
                raise ValueError('Joint combinations require finite [obs_s,action_s,wheel_s,leg_s,weight] rows')
            steps=(table[:,:4]/dt).round().long()
            if ((steps*dt-table[:,:4]).abs()>1e-8).any():
                raise ValueError('Joint combinations must align with physics ticks')
            for i,name in enumerate(['observation','action','wheel','leg']):
                lo,hi=getattr(self,name).bounds
                if ((steps[:,i]<lo)|(steps[:,i]>hi)).any():
                    raise ValueError('Joint combination outside configured channel bounds')
            self.combinations=steps
            self.combination_weights=(table[:,4]/table[:,4].sum()).to(torch.float32)
            self.combination_rng=torch.Generator(device=actions.device).manual_seed(seed+5)
            self.combination_indices=torch.zeros(len(actions),device=actions.device,dtype=torch.long)
            self._sample_combinations(torch.arange(len(actions),device=actions.device))
    def _sample_combinations(self,ids):
        chosen=torch.multinomial(self.combination_weights,len(ids),replacement=True,generator=self.combination_rng)
        self.combination_indices[ids]=chosen
        for i,name in enumerate(['observation','action','wheel','leg']):
            getattr(self,name).steps[ids]=self.combinations[chosen,i]
    def reset(self,ids):
        if ids is None:ids=torch.arange(len(self.pending),device=self.pending.device)
        if self.combinations is not None and self.action.sampling_weights is not None:
            raise ValueError('Joint combination sampling cannot also use independent action weights')
        for q in [self.observation,self.action,self.wheel,self.leg]:q.reset(ids)
        if self.combinations is not None:
            self._sample_combinations(ids)
        self.pending[ids]=0
    def readback(self):
        result={name:getattr(self,name).readback() for name in ['observation','action','wheel','leg']}
        if self.combinations is not None:
            dt=self.action.dt
            result['joint_combinations']=dict(
                columns=['observation','action','wheel','leg'],
                table_ms=(self.combinations*dt*1000).tolist(),
                probabilities=self.combination_weights.tolist(),
                sampled_indices=self.combination_indices.tolist(),
                wheel_serial_queue_sum_ms=((self.observation.steps+self.action.steps+self.wheel.steps)*dt*1000).tolist(),
                leg_serial_queue_sum_ms=((self.observation.steps+self.action.steps+self.leg.steps)*dt*1000).tolist(),
                scope='Queued delays only; response time constants, sensor age and policy phase are separate. Not a measured end-to-end histogram.')
        return result
