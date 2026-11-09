"""Opt-in read-only verification of actual per-physics-tick signal propagation."""
import json
import torch

class ErrorMeter:
    def __init__(self, template):
        self.max=template.new_zeros(());self.sumsq=template.new_zeros(());self.count=template.new_zeros((),dtype=torch.long)
    def add(self,diff,valid=None):
        if valid is None:valid=torch.ones(diff.shape[:-1],device=diff.device,dtype=torch.bool)
        x=torch.where(valid[...,None],diff,torch.zeros_like(diff))
        self.max=torch.maximum(self.max,x.abs().amax());self.sumsq+=x.square().sum();self.count+=valid.sum()*diff.shape[-1]
    def result(self):
        n=int(self.count);return {'scalars':n,'max_abs':float(self.max),'rms':float((self.sumsq/max(n,1)).sqrt()),'valid':n>0}

class TimestampQueueAudit:
    """Use absolute tick ring indices, independent of the queue's shift implementation."""
    def __init__(self,queue,tick):
        self.queue=queue;self.tick=tick;self.depth=queue.history.shape[1]
        self.ring=torch.zeros_like(queue.history);self.stamps=queue.steps.new_full(queue.history.shape[:2],-1000000000)
        self.ids=torch.arange(len(queue.steps),device=queue.steps.device);self.error=ErrorMeter(queue.history);self.calls=0;self.duplicates=0;self.previous=None
        oldpush=queue.push;oldread=queue.read;oldreset=queue.reset
        def push(value,tick=None):
            t=int(self.tick() if tick is None else tick);slot=t%self.depth
            same=self.stamps[:,slot]==t
            # The observer may attach after the environment already primed the
            # current tick. Do not invent a new sample for a push the queue ignores.
            accepted=(~same) & ((queue.last_tick != t) | ((~queue.valid) if queue.fill_initial else torch.zeros_like(queue.valid)))
            self.ring[accepted,slot]=value[accepted];self.stamps[accepted,slot]=t
            self.duplicates+=int(self.previous==t);self.previous=t;self.calls+=1
            result=oldpush(value,tick)
            self.check(oldread(),t)
            return result
        def reset(ids):
            result=oldreset(ids)
            if ids is None:self.stamps.fill_(-1000000000)
            else:self.stamps[ids]=-1000000000
            return result
        queue.push=push;queue.reset=reset
    def check(self,actual,t):
        wanted=t-self.queue.steps;slots=wanted.remainder(self.depth);valid=self.stamps[self.ids,slots]==wanted
        self.error.add(actual-self.ring[self.ids,slots],valid)
    def result(self):return {'calls':self.calls,'same_tick_calls':self.duplicates,**self.error.result()}

class SignalChainAudit:
    def __init__(self,env):
        self.env=env;q=env._joint_delay_dr
        if q is None:raise ValueError('Signal audit requires joint physical-tick DR')
        self.queues={name:TimestampQueueAudit(getattr(q,name),lambda:int(env._sim_step_counter)-1) for name in ['observation','action','wheel','leg']}
        self.pd=ErrorMeter(q.pending);self.responses={};self.policy_history=ErrorMeter(q.pending)
        self.wheel_packet=ErrorMeter(q.pending);self.wheel_cache_compensation=ErrorMeter(q.pending)
        self.phase_counters={'apply':0,'observation':0}
        self._wrap_pd();self._wrap_responses();self._wrap_policy_observation();self._wrap_wheel_cache()
    def _wrap_pd(self):
        a=self.env.robot.actuators['leg_motors'];original=a.compute
        def compute(control_action,joint_pos,joint_vel):
            expected=a.stiffness*(control_action.joint_positions-joint_pos)+a.damping*(control_action.joint_velocities-joint_vel)+control_action.joint_efforts
            result=original(control_action,joint_pos,joint_vel)
            self.pd.add(a.computed_effort-expected)
            return result
        a.compute=compute
    def _wrap_responses(self):
        for name,r in self.env._motor_responses.items():
            meter=ErrorMeter(r.effort);self.responses[name]=meter;original=r.step
            def step(requested,limit,_r=r,_orig=original,_meter=meter):
                expected=torch.where(_r.time_constant>0,_r.effort+(-torch.expm1(-_r.dt/_r.time_constant.clamp_min(1e-12)))*(requested-_r.effort),requested)
                expected=torch.maximum(torch.minimum(expected,limit),-limit)
                result=_orig(requested,limit);_meter.add(result-expected);return result
            r.step=step
    def _wrap_policy_observation(self):
        original=self.env._get_observations
        def get():
            result=original();self.policy_history.add(result['policy'][:,19:25]-self.env._policy_actions);self.phase_counters['observation']+=1;return result
        self.env._get_observations=get
    def _wrap_wheel_cache(self):
        cache=self.env._wheel_feedback_age
        if cache is None:return
        self.packet_queues=[TimestampQueueAudit(q,lambda:int(self.env._sim_step_counter)-1) for q in cache.queues]
        original=cache.update
        def update(omega,tick):
            result=original(omega,tick)
            # Reconstruct sample from each queue's absolute source timestamp.
            for j,audit in enumerate(self.packet_queues):
                wanted=cache.sample_tick[:,j];slots=wanted.remainder(audit.depth);valid=audit.stamps[audit.ids,slots]==wanted
                self.wheel_packet.add((result[:,j:j+1]-audit.ring[audit.ids,slots]),valid)
            return result
        cache.update=update
        apply=self.env._apply_action
        def action():
            result=apply();self.phase_counters['apply']+=1
            motor=self.env._real2sim_wheel
            expected=-motor.damping_compensation*cache.value+motor.coulomb_compensation*torch.tanh(cache.value/motor.epsilon)
            self.wheel_cache_compensation.add(self.env._last_wheel_feedback_torque_nm-expected)
            return result
        self.env._apply_action=action
    def save(self,path):
        report={'queues':{n:a.result() for n,a in self.queues.items()},'leg_pd':self.pd.result(),'motor_response':{n:m.result() for n,m in self.responses.items()},'policy_last_action':self.policy_history.result(),'wheel_packet_timestamp':self.wheel_packet.result(),'compensation_uses_cache':self.wheel_cache_compensation.result(),'phase_calls':self.phase_counters,'scope':'Read-only actual2ms loop audit. Queue startup/reset samples without independenttimestamp excluded. Verifies code propagation, not physical fidelity or realhardware signal alignment.'}
        path.write_text(json.dumps(report,indent=2));return report
