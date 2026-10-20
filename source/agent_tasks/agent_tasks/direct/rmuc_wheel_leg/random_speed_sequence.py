"""Reset-sampled vx schedules; pure reads during observation/control/reward."""
import math
import torch


class RandomSpeedSequence:
    def __init__(self, count, device, dt, episode_s, speed_range, hold_s=(2.,4.),
                 ramp_s=(.4,.8), step_fraction=.5, max_delta=1.5, isolated_rng=False,
                 stop_probability=.1):
        values=[dt,episode_s,*speed_range,*hold_s,*ramp_s,step_fraction,max_delta,stop_probability]
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Finite random speed schedule parameters required')
        lo,hi=speed_range
        if not (dt>0 and episode_s>0 and -1.5<=lo<hi<=3.2 and 0<max_delta<=1.5
                and 0<hold_s[0]<=hold_s[1]<episode_s and 0<=ramp_s[0]<=ramp_s[1]<hold_s[0]
                and 0<=step_fraction<=1 and 0<=stop_probability<=.6):
            raise ValueError('Invalid speed/hold/ramp/delta domain')
        # Opt-in stream isolation: schedule length must not consume the RNG
        # used by rehearsal, DR, reset poses, or policy sampling.
        self.generator = None
        if isolated_rng:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed((torch.initial_seed() + 104729) % (2**63 - 1))
        self.lo,self.hi,self.delta=lo,hi,max_delta
        self.hold=(math.ceil(hold_s[0]/dt),math.floor(hold_s[1]/dt))
        self.ramp=(math.ceil(ramp_s[0]/dt),math.floor(ramp_s[1]/dt))
        if self.hold[0]>self.hold[1] or self.ramp[0]>self.ramp[1]:
            raise ValueError('Timing range contains no policy-aligned value')
        self.step_fraction=step_fraction
        self.stop_probability=stop_probability
        self.slots=math.ceil(episode_s/(self.hold[0]*dt))+1
        self.starts=torch.zeros(count,self.slots,device=device,dtype=torch.long)
        self.targets=torch.full((count,self.slots),float(lo),device=device,dtype=torch.float32)
        self.ramps=torch.zeros_like(self.starts)

    def reset(self, ids):
        n=len(ids)
        if not n:return
        dev=self.targets.device
        hold=torch.randint(self.hold[0],self.hold[1]+1,(n,self.slots),device=dev, generator=self.generator)
        starts=torch.cat([torch.zeros(n,1,device=dev,dtype=torch.long),hold[:,:-1].cumsum(1)],1)
        target=torch.empty(n,self.slots,device=dev)
        # Initial target and later targets mix continuous samples with boundary anchors.
        previous=torch.full((n,),float(self.lo),device=dev,dtype=torch.float32)
        for k in range(self.slots):
            low=(previous-self.delta).clamp_min(self.lo);high=(previous+self.delta).clamp_max(self.hi)
            if k==0:low.fill_(self.lo);high.fill_(self.hi)
            value=low+(high-low)*torch.rand(n,device=dev, generator=self.generator)
            choose=torch.rand(n,device=dev, generator=self.generator)
            value=torch.where(choose<.15,low,torch.where(choose>.85,high,value))
            # Signed courses need actual stops, not merely crossing zero.
            # Reuse the existing draw; forward-only courses stay bit-identical.
            if self.lo < 0 < self.hi:
                stop=(choose>=.5-self.stop_probability/2)&(choose<.5+self.stop_probability/2)&(low<=0)&(high>=0)
                value=torch.where(stop,torch.zeros_like(value),value)
            target[:,k]=value;previous=value
        ramps=torch.randint(self.ramp[0],self.ramp[1]+1,(n,self.slots),device=dev, generator=self.generator)
        ramps[torch.rand(n,self.slots,device=dev, generator=self.generator)<self.step_fraction]=0;ramps[:,0]=0
        self.starts[ids]=starts;self.targets[ids]=target;self.ramps[ids]=ramps

    def commands(self, age):
        idx=(age[:,None]>=self.starts).sum(1).sub(1).clamp(0,self.slots-1)
        col=idx[:,None];target=self.targets.gather(1,col).squeeze(1)
        before=self.targets.gather(1,(idx-1).clamp_min(0)[:,None]).squeeze(1)
        duration=self.ramps.gather(1,col).squeeze(1)
        elapsed=age-self.starts.gather(1,col).squeeze(1)
        fraction=torch.where(duration>0,(elapsed/duration.clamp_min(1)).clamp(0,1),torch.ones_like(target))
        command=torch.zeros(len(age),3,device=age.device)
        command[:,0]=before+(target-before)*fraction
        return command

    def snapshot(self):
        """Detached CPU copies: later partial resets must not alter evidence."""
        return {name: getattr(self, name).detach().cpu().clone()
                for name in ('starts', 'targets', 'ramps')}
