"""Isolated, bounded state-conditioned correction for a frozen RMUC policy.

Not installed into production/training automatically. The frozen reference is
an entire composed skill policy; this module bounds NEW adaptation, not the
legacy policy's accumulated residual. Requires explicit integration/validation.
"""
from __future__ import annotations
import torch
from torch import nn


class StateResidualExperts(nn.Module):
    def __init__(self, experts=4, hidden=32):
        super().__init__()
        if experts < 2 or hidden < 1:
            raise ValueError('At least two experts and positive hidden size required')
        self.router=nn.Sequential(nn.Linear(25,hidden),nn.ELU(),nn.Linear(hidden,experts))
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(25,hidden),nn.ELU(),nn.Linear(hidden,6))
                                    for _ in range(experts)])
        # Diverse hidden features; exact zero correction at installation.
        nn.init.normal_(self.router[-1].weight,std=.01)
        nn.init.zeros_(self.router[-1].bias)
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def probabilities(self, observation):
        return torch.softmax(self.router(observation),dim=-1)

    def forward(self, observation):
        weights=self.probabilities(observation)
        values=torch.stack([torch.tanh(expert(observation)) for expert in self.experts],dim=-2)
        return (weights.unsqueeze(-1)*values).sum(dim=-2)

    def balance_loss(self, observation, eligibility):
        """Batch-level soft importance balance; no forced uniform routing per state.

        No token capacity/dropping: all small experts are evaluated densely.
        This is not a sparse top-k load balancing implementation.
        """
        weights=self.probabilities(observation)
        eligible=eligibility.detach().to(weights.dtype)
        count=eligible.sum()
        mean=(weights*eligible.unsqueeze(-1)).sum(dim=0)/count.clamp_min(1.)
        return (len(self.experts)*mean.square().sum()-1.)*(count>0).to(weights.dtype)


class SeededStateResidualExperts(StateResidualExperts):
    """A local learned correction plus an explicit seed within one shared cap."""
    def __init__(self, experts=4, hidden=32, budget=(.08,.08,.06,.08,.08,.06)):
        super().__init__(experts,hidden)
        bound=torch.as_tensor(budget,dtype=torch.float32)
        if bound.shape!=(6,) or not torch.isfinite(bound).all() or not ((bound>0)&(bound<=.3)).all():
            raise ValueError('Six positive bounded action limits required')
        self.register_buffer('local_budget',bound.clone())
        self.register_buffer('seed',torch.zeros(6))

    def forward(self, observation):
        seed=self.seed.clamp(-self.local_budget,self.local_budget)
        remaining=(self.local_budget-seed.abs()).clamp_min(0.)
        return seed+remaining*super().forward(observation)


class IsolatedTurnStateRouter(nn.Module):
    """Independent direction parameters and smooth command-local support.

    Input uses existing25D measured policy state, no privileged contact flags.
    Neither direction updates the other. The surrounding trainer must freeze
    the complete reference policy and attach the explicit auxiliary loss.
    """
    def __init__(self, budget=(.08,.08,.04,.08,.08,.04), experts=4, hidden=32,
                 recovery_positive_2p0=False, recovery_straight_3p0=False, extended_turn_1p0=False, extended_turn_seed=False, extended_turn_0p8=False, extended_positive_turn_1p0=False, e2e_turn_1p0=False):
        super().__init__()
        bound=torch.as_tensor(budget,dtype=torch.float32)
        if bound.shape!=(6,) or not torch.isfinite(bound).all() or not ((bound>0)&(bound<=.3)).all():
            raise ValueError('Six finite normalized action budgets in (0,.3] required')
        self.register_buffer('budget',bound.clone())
        self.positive=StateResidualExperts(experts,hidden)
        self.negative=StateResidualExperts(experts,hidden)
        if e2e_turn_1p0:
            self.e2e_positive_1p0=SeededStateResidualExperts(experts,hidden,budget=(.3,)*6)
            self.e2e_negative_1p0=SeededStateResidualExperts(experts,hidden,budget=(.3,)*6)
        if recovery_positive_2p0:
            self.recovery_positive_2p0=StateResidualExperts(experts,hidden)
            self.register_buffer('recovery_positive_2p0_wheel_seed',
                                 torch.zeros(6))
        if recovery_straight_3p0:
            self.recovery_straight_3p0=StateResidualExperts(experts,hidden)

        if extended_turn_1p0:
            self.extended_positive_1p0=StateResidualExperts(experts,hidden)
            self.extended_negative_1p0=StateResidualExperts(experts,hidden)
            if extended_turn_seed:
                self.register_buffer('extended_positive_1p0_seed',torch.zeros(6))
                self.register_buffer('extended_negative_1p0_seed',torch.zeros(6))
                self.register_buffer('extended_seed_limit',torch.tensor([0.,0.,.06,0.,0.,.06]))


        if extended_turn_0p8:
            self.extended_positive_1p0_0p8=SeededStateResidualExperts(experts,hidden)
            self.extended_negative_1p0_0p8=SeededStateResidualExperts(experts,hidden)

        if extended_positive_turn_1p0:
            self.extended_positive_1p0_1p0=SeededStateResidualExperts(experts,hidden)

    @staticmethod
    def smooth(value,low,high):
        t=((value-low)/(high-low)).clamp(0.,1.)
        return t*t*(3.-2.*t)

    def install_speed_extensions(self):
        """Copy only learned turn branches; old parameters remain independently frozen."""
        import copy
        if hasattr(self, 'speed_extensions'):
            raise ValueError('Speed extensions already installed')
        names = ('extended_positive_1p0', 'extended_negative_1p0',
                 'extended_positive_1p0_0p8', 'extended_negative_1p0_0p8',
                 'extended_positive_1p0_1p0', 'e2e_positive_1p0', 'e2e_negative_1p0')
        copies = {name: copy.deepcopy(getattr(self, name)) for name in names if hasattr(self, name)}
        if not copies:
            raise ValueError('No learned turn branches to extend')
        self.requires_grad_(False)
        self.speed_extensions = nn.ModuleDict(copies)

    def speed_extension_gate(self, observation):
        bounds=getattr(self, 'speed_extension_blend_bounds', (1.,1.05))
        speed=observation[..., 7] / .5
        gate=self.smooth(speed, bounds[0], bounds[1])
        low=getattr(self, 'low_yaw_speed_blend', None)
        if low is not None:
            yaw=(observation[...,2]/.25+observation[...,8]).abs()
            weight=1.-self.smooth(yaw,low[2],low[3])
            early=self.smooth(speed,low[0],low[1])
            gate=gate+weight*(early-gate)
        return gate

    def turn_branch_output(self, name, observation):
        original = getattr(self, name)(observation)
        copies = getattr(self, 'post_authority_anchors', getattr(self, 'speed_extensions', None))
        if copies is None or name not in copies:
            return original
        weight = self.speed_extension_gate(observation)[..., None]
        # Convex replacement: retain each original branch's output budget.
        return original + weight * (copies[name](observation) - original)

    def install_post_authority_extensions(self, budget=(.3,.3,.06,.3,.3,.06)):
        """Freeze the existing pre-clip behavior; learn only a bounded post-clip delta."""
        import copy
        if not hasattr(self, 'speed_extensions') or hasattr(self, 'post_authority_anchors'):
            raise ValueError('One explicit migration of existing speed extensions required')
        bound=torch.as_tensor(budget,device=self.budget.device,dtype=self.budget.dtype).detach().clone()
        maximum=self.budget.new_tensor([.3,.3,.06,.3,.3,.06])
        if bound.shape!=(6,) or not torch.isfinite(bound).all() or not ((bound>0)&(bound<=maximum)).all():
            raise ValueError('Invalid bounded post-authority action correction')
        self.post_authority_anchors=copy.deepcopy(self.speed_extensions).requires_grad_(False)
        self.register_buffer('post_authority_budget',bound)

    def post_authority_correction(self, observation):
        delta=torch.zeros_like(observation[..., :6])
        for direction in ('positive','negative'):
            for name,module,gate in self.speed_extension_branches(observation,direction):
                scale=self.budget if name in ('extended_positive_1p0','extended_negative_1p0') else 1.
                delta=delta+gate[...,None]*scale*(module(observation)-self.post_authority_anchors[name](observation))
        return torch.maximum(torch.minimum(delta,self.post_authority_budget),-self.post_authority_budget)

    def speed_extension_branches(self, observation, direction):
        if direction not in ('positive', 'negative'):
            raise ValueError('Explicit positive or negative extension direction required')
        if not hasattr(self, 'speed_extensions'):
            raise ValueError('No isolated speed extensions installed')
        result=[]
        for name, module in self.speed_extensions.items():
            if direction not in name:
                continue
            if name == 'extended_positive_1p0_1p0':
                gate=self.extended_positive_turn_1p0_eligibility(observation)
            else:
                gates=(self.extended_turn_0p8_eligibility(observation)
                       if name.endswith('_0p8') else self.extended_turn_eligibility(observation))
                gate=gates[0 if direction=='positive' else 1]
            result.append((name,module,gate*self.speed_extension_gate(observation)))
        if not result:
            raise ValueError('Selected direction has no extension modules')
        return result

    def eligibility(self, observation):
        speed=observation[...,7]/.5
        yaw=observation[...,2]/.25+observation[...,8]
        support=self.smooth(speed,2.2,2.4)*(1.-self.smooth(speed,2.4,2.6))
        return support*self.smooth(yaw,.15,.3), support*self.smooth(-yaw,.15,.3)

    def forward(self, observation):
        positive,negative=self.eligibility(observation)
        result=self.budget*(positive.unsqueeze(-1)*self.positive(observation)
                            +negative.unsqueeze(-1)*self.negative(observation))
        recovery=getattr(self,'recovery_positive_2p0',None)
        if recovery is not None:
            gate=self.recovery_eligibility(observation)
            residual_budget=(self.budget-self.recovery_positive_2p0_wheel_seed.abs()).clamp_min(0.)
            result=result+residual_budget*gate.unsqueeze(-1)*recovery(observation)
            result=result+gate.unsqueeze(-1)*self.recovery_positive_2p0_wheel_seed
        straight=getattr(self,'recovery_straight_3p0',None)
        if straight is not None:
            result=result+self.budget*self.straight_recovery_eligibility(observation).unsqueeze(-1)*straight(observation)
        extended=getattr(self,'extended_positive_1p0',None)
        if extended is not None:
            pos,neg=self.extended_turn_eligibility(observation)
            result=result+self.budget*(pos.unsqueeze(-1)*self.turn_branch_output('extended_positive_1p0', observation)
                +neg.unsqueeze(-1)*self.turn_branch_output('extended_negative_1p0', observation))
            seed=getattr(self,'extended_positive_1p0_seed',None)
            if seed is not None:
                # Checkpoint seed allowance is separate from learned residual.
                positive_seed=seed.clamp(-self.extended_seed_limit,self.extended_seed_limit)
                negative_seed=self.extended_negative_1p0_seed.clamp(-self.extended_seed_limit,self.extended_seed_limit)
                result=result+pos.unsqueeze(-1)*positive_seed+neg.unsqueeze(-1)*negative_seed
        next_turn=getattr(self,'extended_positive_1p0_0p8',None)
        if next_turn is not None:
            pos,neg=self.extended_turn_0p8_eligibility(observation)
            result=result+pos.unsqueeze(-1)*self.turn_branch_output('extended_positive_1p0_0p8', observation)
            result=result+neg.unsqueeze(-1)*self.turn_branch_output('extended_negative_1p0_0p8', observation)
        turn1=getattr(self,'extended_positive_1p0_1p0',None)
        if turn1 is not None:
            gate=self.extended_positive_turn_1p0_eligibility(observation)
            result=result+gate.unsqueeze(-1)*self.turn_branch_output('extended_positive_1p0_1p0', observation)
        if hasattr(self,'e2e_positive_1p0'):
            pos,neg=self.extended_turn_eligibility(observation)
            result=result+pos.unsqueeze(-1)*self.turn_branch_output('e2e_positive_1p0', observation)
            result=result+neg.unsqueeze(-1)*self.turn_branch_output('e2e_negative_1p0', observation)
        return result

    def correction_bound(self, observation):
        bound=self.budget
        limit=getattr(self,'extended_seed_limit',None)
        if limit is not None:
            pos,neg=self.extended_turn_eligibility(observation)
            bound=bound+(pos+neg).unsqueeze(-1)*limit
        next_turn=getattr(self,'extended_positive_1p0_0p8',None)
        if next_turn is not None:
            pos,neg=self.extended_turn_0p8_eligibility(observation)
            bound=bound+pos.unsqueeze(-1)*next_turn.local_budget
            bound=bound+neg.unsqueeze(-1)*self.extended_negative_1p0_0p8.local_budget
        turn1=getattr(self,'extended_positive_1p0_1p0',None)
        if turn1 is not None:
            gate=self.extended_positive_turn_1p0_eligibility(observation)
            bound=bound+gate.unsqueeze(-1)*turn1.local_budget
        if hasattr(self,'e2e_positive_1p0'):
            pos,neg=self.extended_turn_eligibility(observation)
            bound=bound+pos.unsqueeze(-1)*self.e2e_positive_1p0.local_budget
            bound=bound+neg.unsqueeze(-1)*self.e2e_negative_1p0.local_budget
        return bound

    def extended_positive_turn_1p0_eligibility(self, observation):
        speed=observation[...,7]/.5
        yaw=observation[...,2]/.25+observation[...,8]
        end=getattr(self, 'turn_speed_plateau_end', 1.0)
        support=self.smooth(speed,.8,1.)*(1.-self.smooth(speed,end,end+.2))
        return support*self.smooth(yaw,.8001,1.)

    def extended_turn_0p8_eligibility(self, observation):
        speed=observation[...,7]/.5
        yaw=observation[...,2]/.25+observation[...,8]
        end=getattr(self, 'turn_speed_plateau_end', 1.0)
        support=self.smooth(speed,.8,1.)*(1.-self.smooth(speed,end,end+.2))
        # Preserve all previous <=0.6 commands, including float reconstruction.
        return support*self.smooth(yaw,.6001,.8), support*self.smooth(-yaw,.6001,.8)

    def extended_turn_eligibility(self, observation):
        speed=observation[...,7]/.5
        yaw=observation[...,2]/.25+observation[...,8]
        # Disjoint from all previous state-router supports. Preserve mastered
        # +/-0.45 anchors with a small reconstruction-roundoff guard.
        end=getattr(self, 'turn_speed_plateau_end', 1.0)
        support=self.smooth(speed,.8,1.)*(1.-self.smooth(speed,end,end+.2))
        return support*self.smooth(yaw,.4501,.6), support*self.smooth(-yaw,.4501,.6)

    def straight_recovery_eligibility(self, observation):
        speed=observation[...,7]/.5
        yaw=observation[...,2]/.25+observation[...,8]
        return self.smooth(speed,2.9,3.0)*(1.-self.smooth(yaw.abs(),.02,.06))

    def recovery_eligibility(self, observation):
        speed=observation[...,7]/.5
        yaw=observation[...,2]/.25+observation[...,8]
        # Disjoint from the original router's speed support (2.2, 2.6).
        return (self.smooth(speed,1.8,2.0)*(1.-self.smooth(speed,2.0,2.2))
                *self.smooth(yaw,.15,.3))

    def branches(self, observation):
        positive,negative=self.eligibility(observation)
        branches=[('positive',self.positive,positive>0),('negative',self.negative,negative>0)]
        recovery=getattr(self,'recovery_positive_2p0',None)
        if recovery is not None:
            branches.append(('recovery_positive_2p0',recovery,self.recovery_eligibility(observation)>0))
        straight=getattr(self,'recovery_straight_3p0',None)
        if straight is not None:
            branches.append(('recovery_straight_3p0',straight,self.straight_recovery_eligibility(observation)>0))
        extended=getattr(self,'extended_positive_1p0',None)
        if extended is not None:
            pos,neg=self.extended_turn_eligibility(observation)
            branches.extend([('extended_positive_1p0',extended,pos>0),
                             ('extended_negative_1p0',self.extended_negative_1p0,neg>0)])
        next_turn=getattr(self,'extended_positive_1p0_0p8',None)
        if next_turn is not None:
            pos,neg=self.extended_turn_0p8_eligibility(observation)
            branches.extend([('extended_positive_1p0_0p8',next_turn,pos>0),
                             ('extended_negative_1p0_0p8',self.extended_negative_1p0_0p8,neg>0)])
        turn1=getattr(self,'extended_positive_1p0_1p0',None)
        if turn1 is not None:
            branches.append(('extended_positive_1p0_1p0',turn1,
                             self.extended_positive_turn_1p0_eligibility(observation)>0))
        if hasattr(self,'e2e_positive_1p0'):
            pos,neg=self.extended_turn_eligibility(observation)
            branches.extend([('e2e_positive_1p0',self.e2e_positive_1p0,pos>0),
                             ('e2e_negative_1p0',self.e2e_negative_1p0,neg>0)])
        return branches

    def balance_loss(self, observation):
        terms=observation.new_zeros(())
        directions=observation.new_zeros(())
        for _,branch,active in self.branches(observation):
            terms=terms+branch.balance_loss(observation,active)
            directions=directions+active.any().to(observation.dtype)
        return terms/directions.clamp_min(1.)

    def isolate_optimizer_step(self, observation):
        """Call after backward, before step: zero gradients do NOT stop Adam momentum."""
        for _,branch,mask in self.branches(observation):
            if not bool(mask.any()):
                for parameter in branch.parameters():
                    parameter.grad=None
