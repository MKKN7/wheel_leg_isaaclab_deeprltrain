"""Independent wheel packet ages, shared by actor and electrical compensation.

Transport and refresh are explicit; actor wheel slots replace (not add to) the
whole-frame observation delay. Mechanical losses must use physical velocity.
"""
import torch
from .joint_delay_dr import SampledDelay

class WheelFeedbackAge:
    def __init__(self, template, dt, transport_s, refresh_s, seed):
        self.dt = dt
        self.queues = [SampledDelay(template[:, i:i+1], dt, transport_s, seed+i, True) for i in range(2)]
        self.refresh_bounds = [round(v/dt) for v in refresh_s]
        if len(refresh_s)!=2 or not 1<=self.refresh_bounds[0]<=self.refresh_bounds[1] or any(abs(n*dt-v)>1e-8 for n,v in zip(self.refresh_bounds,refresh_s)):
            raise ValueError('Refresh bounds must be ordered positive physical ticks')
        self.rng = torch.Generator(device=template.device).manual_seed(seed+2)
        self.period = torch.ones_like(template, dtype=torch.long)
        self.phase = torch.zeros_like(self.period)
        self.value = torch.zeros_like(template)
        self.sample_tick = torch.zeros_like(self.period)
        self.valid = torch.zeros_like(template, dtype=torch.bool)
        self.last_tick = None
        self.reset(torch.arange(len(template),device=template.device))

    def reset(self, ids):
        for q in self.queues: q.reset(ids)
        self.period[ids] = torch.randint(self.refresh_bounds[0], self.refresh_bounds[1]+1, self.period[ids].shape, generator=self.rng, device=self.period.device)
        self.phase[ids] = (torch.rand(self.period[ids].shape, generator=self.rng, device=self.period.device)*self.period[ids]).long()
        self.valid[ids] = False
        self.value[ids] = 0

    def update(self, omega, tick):
        for i,q in enumerate(self.queues):
            q.push(omega[:,i:i+1],tick)
            due = ~self.valid[:,i]
            if self.last_tick != tick: due = due | ((tick-self.phase[:,i]) % self.period[:,i] == 0)
            self.value[due,i] = q.read()[due,0]
            self.sample_tick[due,i] = tick-q.steps[due]
            self.valid[due,i] = True
        self.last_tick = tick
        return self.value

    def readback(self):
        return dict(transport=[q.readback() for q in self.queues], refresh_steps_per_env=self.period.tolist(),
                    current_age_ms=((self.last_tick-self.sample_tick)*self.dt*1000).tolist() if self.last_tick is not None else None,
                    maximum_age_ms=(self.queues[0].bounds[1]+self.refresh_bounds[1]-1)*self.dt*1000,
                    observation_semantics='replace_wheel_slots_after_frame_delay', compensation_source='same_feedback_cache',
                    initial_history='primed_from_reset_state_not_real_prehistory')
