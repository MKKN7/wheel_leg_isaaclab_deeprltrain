"""Opt-in fixed-observation action trust region; not a behavior certificate."""
import copy
import torch

class StartupAnchor:
    def __init__(self, actor, optimizer, path, limit=.01):
        if not 0 < limit < 1: raise ValueError('Invalid action anchor limit')
        self.actor,self.optimizer,self.limit=actor,optimizer,limit
        self.params=[p for p in actor.parameters() if p.requires_grad]
        if not self.params: raise ValueError('No trainable actor parameters')
        data=torch.load(path,map_location=self.params[0].device,weights_only=False)
        self.obs=data['obs'];self.target=data['target']
        if self.obs.ndim!=2 or self.obs.shape[1]!=25 or self.target.shape!=(len(self.obs),6):raise ValueError('Invalid anchor shape')
        if not torch.isfinite(self.obs).all() or not torch.isfinite(self.target).all():raise ValueError('Nonfinite anchor')
        if self.error()>limit+1e-6:raise ValueError('Source actor already violates anchor')
    @torch.no_grad()
    def error(self):
        a=self.actor({'policy':self.obs},stochastic_output=False)
        return float((a-self.target).abs().max())
    def begin(self):
        self.before=[p.detach().clone() for p in self.params]
        self.states=[copy.deepcopy(self.optimizer.state.get(p)) for p in self.params]
    @torch.no_grad()
    def finish(self):
        candidate=[p.detach().clone() for p in self.params];trials=[];accepted=0.
        for scale in [1.,.5,.25,.125,.0625,0.]:
            for p,a,b in zip(self.params,self.before,candidate):p.copy_(a if scale==0. else a+scale*(b-a))
            error=self.error();trials.append(dict(scale=scale,max_action_error=error))
            if error<=self.limit+1e-6:accepted=scale;break
        if not error<=self.limit+1e-6:raise RuntimeError('Anchor rollback failed')
        if accepted!=1.:
            for p,state in zip(self.params,self.states):
                if state is None:self.optimizer.state.pop(p,None)
                else:self.optimizer.state[p]=state
        return dict(accepted_scale=accepted,trials=trials,limit=self.limit,scope='Fixed startup observations only; behavior retention unproven')
