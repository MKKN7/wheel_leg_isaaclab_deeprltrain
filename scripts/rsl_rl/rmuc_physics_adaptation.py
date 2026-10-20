"""One optional bounded residual for low-speed physical-domain adaptation.

Zero initialization preserves the complete source actor exactly. This is a
prototype module until installed, trained and evaluated through the full actor;
it is not an external velocity/pitch/yaw controller.
"""
import copy
import torch
from torch import nn


class PhysicsAdaptation(nn.Module):
    def __init__(self, frozen_source=None, update_domain=None, scoped_network_domain=None,
                 positive_update_domain=None, positive_straight_enabled=None,
                 reference_update_domain=None, frozen_reference_update_domain=None, network_update_domain=None):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(25,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,6))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        # Action-unit residual bounds, NOT actuator effort or position limits.
        # Wheel bound covers the nominal damping shift: .0775*20/5.5=.282.
        # Final consumers still enforce existing joint position and torque limits.
        self.register_buffer('budget',torch.tensor([.15,0.,.35,.15,0.,.35]))
        if frozen_source is not None:
            depth=0
            ancestor=frozen_source
            while hasattr(ancestor,'frozen_source'):
                depth+=1
                ancestor=ancestor.frozen_source
            if depth>=3:
                raise ValueError('At most three explicit frozen physical migration stages are supported')
            self.frozen_source = frozen_source.requires_grad_(False)
            self.budget.copy_(frozen_source.budget)
        if update_domain is not None:
            domain=torch.as_tensor(update_domain,dtype=torch.float32)
            if (frozen_source is None or domain.shape!=(4,) or not torch.isfinite(domain).all()
                    or not 0 <= domain[0] < domain[1] <= 1.7
                    or not 0 <= domain[2] < domain[3] <= 4.2):
                raise ValueError('Scoped updates require frozen source and ordered speed/yaw fade bounds')
            self.register_buffer('update_domain',domain.clone())
        if scoped_network_domain is not None:
            domain = torch.as_tensor(scoped_network_domain, dtype=torch.float32)
            if (domain.shape != (6,) or not torch.isfinite(domain).all()
                    or not 0 <= domain[0] < domain[1] <= domain[2] < domain[3] <= 1.5
                    or not 0 <= domain[4] < domain[5] <= .45):
                raise ValueError('Scoped network requires ordered forward speed and yaw fade bounds')
            self.register_buffer('scoped_network_domain', domain.clone())
            self.scoped_network = copy.deepcopy(self.network).requires_grad_(False)
        if positive_update_domain is not None:
            domain=torch.as_tensor(positive_update_domain,dtype=torch.float32)
            if (domain.shape!=(6,) or not torch.isfinite(domain).all()
                    or not 0 <= domain[0] < domain[1] <= domain[2] < domain[3] <= 1.7
                    or not 0 <= domain[4] < domain[5] <= .6):
                raise ValueError('Scoped updates require speed and positive-yaw fade bounds')
            self.register_buffer('positive_update_domain',domain.clone())
            self.reference_network=copy.deepcopy(self.network).requires_grad_(False)
        if positive_straight_enabled is not None:
            if positive_update_domain is None:
                raise ValueError('Straight selection requires a positive update domain')
            flag=torch.as_tensor(positive_straight_enabled,dtype=torch.float32)
            if flag.numel()!=1 or not bool(((flag==0)|(flag==1)).all()):
                raise ValueError('Straight selection must be zero or one')
            self.register_buffer('positive_straight_enabled',flag.reshape(()).clone())

        if reference_update_domain is not None:
            d=torch.as_tensor(reference_update_domain,dtype=torch.float32)
            if (not hasattr(self,'reference_network') or d.shape!=(8,) or not torch.isfinite(d).all()
                    or not 0 <= d[0] < d[1] <= d[2] < d[3] <= 1.7
                    or not -.6 <= d[4] < d[5] <= d[6] < d[7] <= .6):
                raise ValueError('Reference updates require ordered speed and signed yaw support bounds')
            self.register_buffer('reference_update_domain',d.clone())
            self.frozen_reference_network=copy.deepcopy(self.reference_network).requires_grad_(False)

        if frozen_reference_update_domain is not None:
            d=torch.as_tensor(frozen_reference_update_domain,dtype=torch.float32)
            if (not hasattr(self,'frozen_reference_network') or d.shape!=(8,) or not torch.isfinite(d).all()
                    or not 0 <= d[0] < d[1] <= d[2] < d[3] <= 1.7
                    or not -4.2 <= d[4] < d[5] <= d[6] < d[7] <= 4.2):
                raise ValueError('Frozen reference update requires valid existing network and signed command domain')
            self.register_buffer('frozen_reference_update_domain',d.clone())
            self.reference_anchor_network=copy.deepcopy(self.frozen_reference_network).requires_grad_(False)

        if network_update_domain is not None:
            d=torch.as_tensor(network_update_domain,dtype=torch.float32)
            if (d.shape!=(8,) or not torch.isfinite(d).all()
                    or not 0 <= d[0] < d[1] <= d[2] < d[3] <= 1.7
                    or not -4.2 <= d[4] < d[5] <= d[6] < d[7] <= 4.2):
                raise ValueError('Network update requires ordered speed and signed yaw bounds')
            self.register_buffer('network_update_domain',d.clone())
            self.network_anchor=copy.deepcopy(self.network).requires_grad_(False)

    def network_update_gate(self,observation):
        d=self.network_update_domain
        speed=observation[...,7]/.5
        yaw=observation[...,2]/.25+observation[...,8]
        def smooth(x):
            x=x.clamp(0.,1.)
            return x*x*(3.-2.*x)
        return (smooth((speed-d[0])/(d[1]-d[0]))*(1.-smooth((speed-d[2])/(d[3]-d[2])))
                *smooth((yaw-d[4])/(d[5]-d[4]))*(1.-smooth((yaw-d[6])/(d[7]-d[6]))))

    def install_alignment_network(self, budget, domain):
        """Opt-in zero-output alignment head; does not widen inherited budgets."""
        if hasattr(self, 'alignment_network'):
            raise ValueError('Alignment head already installed')
        b = torch.as_tensor(budget, dtype=self.budget.dtype, device=self.budget.device)
        d = torch.as_tensor(domain, dtype=self.budget.dtype, device=self.budget.device)
        if (b.shape != (6,) or not torch.isfinite(b).all() or (b < 0).any()
                or (b > .5).any() or d.shape != (4,) or not torch.isfinite(d).all()
                or not (-.05 <= d[0] < d[1] and 0 <= d[1] <= d[2] < d[3] <= 1.7)):
            raise ValueError('Finite bounded alignment budget and ordered forward-speed domain required')
        self.register_buffer('alignment_budget', b.clone())
        self.register_buffer('alignment_domain', d.clone())
        self.alignment_network = copy.deepcopy(self.network)
        nn.init.zeros_(self.alignment_network[-1].weight)
        nn.init.zeros_(self.alignment_network[-1].bias)

    def alignment_gate(self, observation):
        d = self.alignment_domain
        speed = observation[..., 7] / .5
        yaw = observation[..., 2] / .25 + observation[..., 8]
        def smooth(t):
            t = t.clamp(0., 1.)
            return t*t*(3.-2.*t)
        # Match the measured straight-only reward. Nonzero turn commands
        # remain outside the new head; actual yaw still enters its network.
        original = (smooth((speed-d[0])/(d[1]-d[0]))
                * (1.-smooth((speed-d[2])/(d[3]-d[2])))
                * (yaw.abs() <= 1e-4).to(observation.dtype))
        return original + PhysicsAdaptation._straight_bridge_extra(self, observation, True)

    def install_straight_bridge(self):
        """Opt-in convex fill of the gap between existing straight branches.

        This changes only the previously incomplete transition interval. Existing
        checkpoints and the ongoing training run keep their original gates.
        """
        if not hasattr(self, 'straight_extension_network') or hasattr(self, 'straight_bridge_enabled'):
            raise ValueError('A single bridge requires both existing straight networks')
        self.register_buffer('straight_bridge_enabled', self.budget.new_tensor(1.))

    def _straight_bridge_extra(self, observation, low_branch):
        if not hasattr(self, 'straight_bridge_enabled'):
            return torch.zeros_like(observation[..., 7])
        a, b = self.alignment_domain, self.straight_extension_domain
        v=observation[...,7]/.5
        def smooth(t):
            t=t.clamp(0.,1.)
            return t*t*(3.-2.*t)
        low=smooth((v-a[0])/(a[1]-a[0]))*(1.-smooth((v-a[2])/(a[3]-a[2])))
        high=smooth((v-b[0])/(b[1]-b[0]))*(1.-smooth((v-b[2])/(b[3]-b[2])))
        missing=(1.-low-high).clamp(0.,1.)
        blend=smooth((v-a[2])/(b[1]-a[2]))
        yaw=observation[...,2]/.25+observation[...,8]
        active=((v>a[2]) & (v<b[1]) & (yaw.abs()<=1e-4) & (observation[...,8].abs()<1.99))
        return missing*(1.-blend if low_branch else blend)*active.to(v.dtype)

    def correction_bound(self):
        alignment = getattr(self, 'alignment_budget', torch.zeros_like(self.budget))
        extension = getattr(self, 'straight_extension_budget', torch.zeros_like(self.budget))
        reverse = getattr(self, 'reverse_budget', torch.zeros_like(self.budget))
        # Base domains are disjoint; optional bridge is a convex combination.
        return self.budget + torch.maximum(alignment, extension) + reverse

    def install_reverse_network(self, budget, domain):
        """Explicit zero-output reverse-only branch; forward routing is untouched."""
        if hasattr(self, 'reverse_network'):
            raise ValueError('Reverse branch already installed')
        b = torch.as_tensor(budget, dtype=self.budget.dtype, device=self.budget.device)
        d = torch.as_tensor(domain, dtype=self.budget.dtype, device=self.budget.device)
        if (b.shape != (6,) or not torch.isfinite(b).all() or (b < 0).any()
                or (b > .5).any() or d.shape != (4,) or not torch.isfinite(d).all()
                or not -1.5 <= d[0] < d[1] <= d[2] < d[3] <= 0):
            raise ValueError('Bounded reverse budget and ordered negative speed domain required')
        self.register_buffer('reverse_budget', b.clone())
        self.register_buffer('reverse_domain', d.clone())
        self.reverse_network = copy.deepcopy(self.network)
        nn.init.zeros_(self.reverse_network[-1].weight)
        nn.init.zeros_(self.reverse_network[-1].bias)

    def reverse_gate(self, observation):
        d = self.reverse_domain
        speed = observation[..., 7] / .5
        yaw = observation[..., 2] / .25 + observation[..., 8]
        def smooth(t):
            t = t.clamp(0., 1.)
            return t*t*(3.-2.*t)
        return (smooth((speed-d[0])/(d[1]-d[0]))
                * (1.-smooth((speed-d[2])/(d[3]-d[2])))
                * (yaw.abs() <= 1e-4).to(observation.dtype))

    def install_straight_extension(self, budget, domain):
        """Checkpoint-defined forward-straight branch, disjoint from crawl.

        Existing domains stay unchanged; an explicitly migrated checkpoint may
        extend through 3.2 m/s with a bounded fade-out ending at most 3.4.
        """
        if hasattr(self, 'straight_extension_network') or not hasattr(self, 'alignment_network'):
            raise ValueError('Straight extension requires existing crawl alignment and one new head only')
        b = torch.as_tensor(budget, device=self.budget.device, dtype=self.budget.dtype)
        d = torch.as_tensor(domain, device=self.budget.device, dtype=self.budget.dtype)
        if (b.shape != (6,) or not torch.isfinite(b).all() or (b < 0).any() or (b > .5).any()
                or d.shape != (4,) or not torch.isfinite(d).all()
                or not float(self.alignment_domain[3]) <= d[0] < d[1] <= d[2] < d[3] <= 3.4):
            raise ValueError('Straight extension must be bounded and not overlap crawl alignment')
        self.register_buffer('straight_extension_budget', b.clone())
        self.register_buffer('straight_extension_domain', d.clone())
        self.straight_extension_network = copy.deepcopy(self.network)
        nn.init.zeros_(self.straight_extension_network[-1].weight)
        nn.init.zeros_(self.straight_extension_network[-1].bias)

    def straight_extension_gate(self, observation):
        d = self.straight_extension_domain
        speed = observation[..., 7] / .5
        error = observation[..., 8]
        yaw = observation[..., 2] / .25 + error
        def smooth(t):
            t = t.clamp(0.,1.)
            return t*t*(3.-2.*t)
        original = (smooth((speed-d[0])/(d[1]-d[0]))
                * (1.-smooth((speed-d[2])/(d[3]-d[2])))
                * ((yaw.abs() <= 1e-4) & (error.abs() < 1.99)).to(observation.dtype))
        return original + PhysicsAdaptation._straight_bridge_extra(self, observation, False)

    def frozen_reference_gate(self,observation):
        d=self.frozen_reference_update_domain
        speed=observation[...,7]/.5
        yaw=observation[...,2]/.25+observation[...,8]
        def smooth(x):
            x=x.clamp(0.,1.)
            return x*x*(3.-2.*x)
        return (smooth((speed-d[0])/(d[1]-d[0]))*(1.-smooth((speed-d[2])/(d[3]-d[2])))
                *smooth((yaw-d[4])/(d[5]-d[4]))*(1.-smooth((yaw-d[6])/(d[7]-d[6]))))

    def reference_gate(self,observation):
        d=self.reference_update_domain
        speed=observation[...,7]/.5
        yaw=observation[...,2]/.25+observation[...,8]
        def smooth(x):
            x=x.clamp(0.,1.)
            return x*x*(3.-2.*x)
        return (smooth((speed-d[0])/(d[1]-d[0]))*(1.-smooth((speed-d[2])/(d[3]-d[2])))
                *smooth((yaw-d[4])/(d[5]-d[4]))*(1.-smooth((yaw-d[6])/(d[7]-d[6]))))

    def positive_gate(self,observation):
        d=self.positive_update_domain
        speed=observation[...,7]/.5
        yaw=observation[...,2]/.25+observation[...,8]
        def smooth(x):
            x=x.clamp(0.,1.)
            return x*x*(3.-2.*x)
        speed_gate=smooth((speed-d[0])/(d[1]-d[0]))*(1.-smooth((speed-d[2])/(d[3]-d[2])))
        straight=(1.-smooth(yaw.abs()/.1))*getattr(self,'positive_straight_enabled',1.)
        positive_turn=smooth((yaw-d[4])/(d[5]-d[4]))
        return speed_gate*torch.maximum(straight,positive_turn)

    def scoped_gate(self, observation):
        d = self.scoped_network_domain
        speed = observation[..., 7] / .5
        yaw = (observation[..., 2] / .25 + observation[..., 8]).abs()
        def smooth(x):
            x = x.clamp(0., 1.)
            return x*x*(3.-2.*x)
        return (smooth((speed-d[0])/(d[1]-d[0]))
                * (1.-smooth((speed-d[2])/(d[3]-d[2])))
                * (1.-smooth((yaw-d[4])/(d[5]-d[4]))))


    @staticmethod
    def gate(observation):
        speed=(observation[...,7]/.5).abs()
        yaw=(observation[...,2]/.25+observation[...,8]).abs()
        def fade(x,start,end):
            t=((x-start)/(end-start)).clamp(0.,1.)
            return 1.-t*t*(3.-2.*t)
        return fade(speed,1.5,1.7)*fade(yaw,4.,4.2)

    def update_gate(self,observation):
        gate=self.gate(observation)
        if hasattr(self,'update_domain'):
            speed=(observation[...,7]/.5).abs()
            yaw=(observation[...,2]/.25+observation[...,8]).abs()
            def fade(x,low,high):
                t=((x-low)/(high-low)).clamp(0.,1.)
                return 1.-t*t*(3.-2.*t)
            gate=gate*fade(speed,self.update_domain[0],self.update_domain[1])*fade(yaw,self.update_domain[2],self.update_domain[3])
        return gate

    def _legacy_correction(self,observation):
        output = torch.tanh(self.network(observation))
        if hasattr(self,'network_anchor'):
            anchor=torch.tanh(self.network_anchor(observation))
            output=anchor+self.network_update_gate(observation)[...,None]*(output-anchor)
        if hasattr(self,'reference_network'):
            reference=torch.tanh(self.reference_network(observation))
            if hasattr(self,'frozen_reference_network'):
                original=torch.tanh(self.frozen_reference_network(observation))
                if hasattr(self,'reference_anchor_network'):
                    anchor=torch.tanh(self.reference_anchor_network(observation))
                    original=anchor+self.frozen_reference_gate(observation)[...,None]*(original-anchor)
                reference=original+self.reference_gate(observation)[...,None]*(reference-original)
            output=reference+self.positive_gate(observation)[...,None]*(output-reference)
        if hasattr(self, 'scoped_network'):
            gate = self.scoped_gate(observation)[..., None]
            output = output + gate * (torch.tanh(self.scoped_network(observation)) - output)
        delta = self.update_gate(observation)[...,None]*self.budget*output
        if not hasattr(self, 'frozen_source'):
            return delta
        inherited = self.frozen_source.correction(observation)
        # Share the existing physical residual envelope; do not double authority.
        delta = torch.maximum(torch.minimum(delta, self.budget-inherited), -self.budget-inherited)
        return inherited + delta

    def correction(self, observation):
        original = self._legacy_correction(observation)
        if hasattr(self, 'reverse_network'):
            original = original + (self.reverse_gate(observation)[..., None]
                * self.reverse_budget * torch.tanh(self.reverse_network(observation)))
        if not hasattr(self, 'alignment_network'):
            return original
        result = original + (self.alignment_gate(observation)[..., None]
                           * self.alignment_budget
                           * torch.tanh(self.alignment_network(observation)))
        if hasattr(self, 'straight_extension_network'):
            result = result + (self.straight_extension_gate(observation)[..., None]
                               * self.straight_extension_budget
                               * torch.tanh(self.straight_extension_network(observation)))
        return result

    def forward(self,observation,source_action):
        return source_action+self.correction(observation)
