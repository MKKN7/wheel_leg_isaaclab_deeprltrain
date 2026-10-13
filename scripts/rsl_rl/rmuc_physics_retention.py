"""Optional same-rollout line search on the physical head's actual PPO step.

This is an empirical surrogate guard, never a behavior retention certificate.
"""
import copy
import torch
from tensordict import TensorDict


class PhysicsRetentionStep:
    def __init__(self, runner):
        self.actor=runner.alg.get_policy()
        self.params=list(self.actor.rmuc_physics_adaptation.network.parameters())
        self.optimizer=runner.alg.optimizer
        self.before=[p.detach().clone() for p in self.params]
        self.states=[copy.deepcopy(self.optimizer.state.get(p)) for p in self.params]
        s=runner.alg.storage
        self.obs=s.observations['policy'].reshape(-1,25).detach().clone()
        self.actions=s.actions.reshape(-1,6).detach().clone()
        self.advantages=s.advantages.reshape(-1).detach().clone()
        self.old_mean,self.std=[v.reshape(-1,6).detach().clone() for v in s.distribution_params]
        self.clip=runner.alg.clip_param
        self.batch_size=s.actions.shape[1]
        speed=self.obs[:,7]/.5;yaw=self.obs[:,2]/.25+self.obs[:,8]
        self.groups={}
        matched=torch.zeros(len(speed),device=speed.device,dtype=torch.bool)
        for target in (.8,-.8,1.,-1.,1.2,-1.2,1.4,-1.4):
            mask=((speed-1.).abs()<1e-3)&((yaw-target).abs()<1e-3)
            if bool(mask.any()):self.groups[str(target)]=mask
            matched|=mask
        if bool((~matched).any()):self.groups['other']=~matched
        self.baseline=self.scores()

    @torch.no_grad()
    def scores(self):
        means=torch.cat([self.actor(TensorDict({'policy':x},batch_size=[len(x)]),
            stochastic_output=False) for x in self.obs.split(self.batch_size)])
        log_ratio=-.5*(((self.actions-means)/self.std).square()-
                      ((self.actions-self.old_mean)/self.std).square()).sum(-1)
        ratio=log_ratio.exp()
        value=torch.minimum(ratio*self.advantages,
            ratio.clamp(1.-self.clip,1.+self.clip)*self.advantages)
        return {k:float(value[mask].mean()) for k,mask in self.groups.items()}

    @torch.no_grad()
    def finish(self):
        candidate=[p.detach().clone() for p in self.params]
        trials=[];accepted=0.
        # No average-negative allowance: every represented group must retain
        # its measured surrogate, and the failed +0.8 group must improve.
        for scale in (1.,.5,.25,.125,.0625):
            for p,a,b in zip(self.params,self.before,candidate):p.copy_(a+scale*(b-a))
            current=self.scores()
            changes={k:current[k]-self.baseline[k] for k in current}
            ok=all(torch.isfinite(torch.tensor(v)) and v>=0. for v in changes.values())
            ok=bool(ok and changes.get('0.8',0.)>0.)
            trials.append({'scale':scale,'surrogate_changes':changes,'accepted':ok})
            if ok:accepted=scale;break
        if accepted==0.:
            for p,a in zip(self.params,self.before):p.copy_(a)
        if accepted!=1.:
            # Full-step Adam moments are incompatible with a reduced/rejected
            # physical step. Restore just this group's pre-step state.
            for p,state in zip(self.params,self.states):
                if state is None:self.optimizer.state.pop(p,None)
                else:self.optimizer.state[p]=state
        return {'accepted_scale':accepted,'trials':trials,
                'physical_optimizer_restored':accepted!=1.,
                'scope':'Same-rollout surrogate only; other is aggregated; behavior validation still required'}
