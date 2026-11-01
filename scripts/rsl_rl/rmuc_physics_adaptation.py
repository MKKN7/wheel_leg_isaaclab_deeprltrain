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
        return self.budget + torch.maximum(alignment, extension) + reverse + getattr(self,'high_straight_budget',torch.zeros_like(self.budget)) + getattr(self,'timing_budget',torch.zeros_like(self.budget)) + getattr(self,'delay_local_budget',torch.zeros_like(self.budget)) + getattr(self,'stop_budget',torch.zeros_like(self.budget)) + getattr(self,'startup_budget',torch.zeros_like(self.budget)) + getattr(self,'reverse_reserve_budget',torch.zeros_like(self.budget)) + getattr(self,'joint_delay_budget',torch.zeros_like(self.budget))

    def install_reverse_network(self, budget, domain):
        """Explicit zero-output reverse-only branch; forward routing is untouched."""
        if hasattr(self, 'reverse_network'):
            raise ValueError('Reverse branch already installed')
        b = torch.as_tensor(budget, dtype=self.budget.dtype, device=self.budget.device)
        d = torch.as_tensor(domain, dtype=self.budget.dtype, device=self.budget.device)
        if (b.shape != (6,) or not torch.isfinite(b).all() or (b < 0).any()
                or (b > .5).any() or d.shape != (4,) or not torch.isfinite(d).all()
                or not -1.7 <= d[0] < d[1] <= d[2] < d[3] <= 0):
            raise ValueError('Bounded reverse budget and ordered negative speed domain required')
        self.register_buffer('reverse_budget', b.clone())
        self.register_buffer('reverse_domain', d.clone())
        self.reverse_network = copy.deepcopy(self.network)
        nn.init.zeros_(self.reverse_network[-1].weight)
        nn.init.zeros_(self.reverse_network[-1].bias)

    def install_joint_delay_network(self, budget):
        """Zero-output25D residual for all commands, with explicit action bounds."""
        if hasattr(self,'joint_delay_network'):raise ValueError('Joint delay head already installed')
        b=torch.as_tensor(budget,device=self.budget.device,dtype=self.budget.dtype)
        if b.shape!=(6,) or not torch.isfinite(b).all() or (b<=0).any() or (b>.15).any():
            raise ValueError('Six positive joint-delay action budgets<=.15 required')
        self.register_buffer('joint_delay_budget',b.clone())
        self.joint_delay_network=copy.deepcopy(self.network)
        nn.init.zeros_(self.joint_delay_network[-1].weight)
        nn.init.zeros_(self.joint_delay_network[-1].bias)

    def install_joint_delay_scope(self, domain):
        """Optional command-local scope; old checkpoints retain all-command behavior."""
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (not hasattr(self,'joint_delay_network') or hasattr(self,'joint_delay_domain')
                or d.shape!=(6,) or not torch.isfinite(d).all()
                or not -1.5<=d[0]<d[1]<=d[2]<d[3]<=3.5 or not 0<=d[4]<d[5]<=.5):
            raise ValueError('Joint-delay scope requires ordered vx fades and absolute yaw fades')
        self.register_buffer('joint_delay_domain',d.clone())

    def joint_delay_gate(self, observation):
        if not hasattr(self,'joint_delay_domain'):return torch.ones_like(observation[...,0])
        d=self.joint_delay_domain
        vx=observation[...,7]/.5
        yaw=(observation[...,8]+observation[...,2]/.25).abs()
        smooth=lambda x: x.clamp(0.,1.).square()*(3.-2.*x.clamp(0.,1.))
        gate=smooth((vx-d[0])/(d[1]-d[0]))*smooth((d[3]-vx)/(d[3]-d[2]))*smooth((d[5]-yaw)/(d[5]-d[4]))
        # A clipped yaw-error slot cannot establish a straight command reliably.
        return gate*(observation[...,8].abs()<1.999).to(gate.dtype)

    def install_reverse_reserve(self, budget, domain):
        """Optional common-wheel authority; zero init and old-speed isolation."""
        if not hasattr(self, 'reverse_network') or hasattr(self, 'reverse_reserve_network'):
            raise ValueError('Existing reverse branch and no reserve required')
        b=torch.as_tensor(budget,device=self.budget.device,dtype=self.budget.dtype)
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (b.shape!=(6,) or not torch.isfinite(b).all() or b[[0,1,3,4]].abs().max()!=0
                or not 0 < b[2] <= .15 or b[2]!=b[5]
                or d.shape!=(2,) or not torch.isfinite(d).all() or not -1.5<=d[0]<d[1]<=-.6):
            raise ValueError('Common-wheel budget<=.15 and reverse reserve below-.6 required')
        self.register_buffer('reverse_reserve_budget',b.clone())
        self.register_buffer('reverse_reserve_domain',d.clone())
        self.reverse_reserve_network=copy.deepcopy(self.network)
        nn.init.zeros_(self.reverse_reserve_network[-1].weight)
        nn.init.zeros_(self.reverse_reserve_network[-1].bias)

    def reverse_reserve_gate(self, observation):
        d=self.reverse_reserve_domain
        t=((d[1]-observation[...,7]/.5)/(d[1]-d[0])).clamp(0.,1.)
        return self.reverse_gate(observation)*t*t*(3.-2.*t)

    def reverse_reserve_output(self, observation):
        logits=self.reverse_reserve_network(observation)
        common=torch.tanh((logits[...,2]+logits[...,5])*.5)
        return self.reverse_reserve_gate(observation)[...,None]*self.reverse_reserve_budget*common[...,None]

    def install_reverse_update_scope(self, domain):
        if not hasattr(self, 'reverse_network') or hasattr(self, 'reverse_update_anchor'):
            raise ValueError('Existing unscoped reverse network required')
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if d.shape!=(2,) or not torch.isfinite(d).all() or not -1.5<=d[0]<d[1]<0:
            raise ValueError('Ordered negative command-speed bounds required')
        self.register_buffer('reverse_update_domain',d.clone())
        self.reverse_update_anchor=copy.deepcopy(self.reverse_network).requires_grad_(False)

    def install_reverse_extension_scope(self, domain):
        if not hasattr(self, 'reverse_update_anchor') or hasattr(self, 'reverse_extension_anchor'):
            raise ValueError('Existing first reverse scope required; extension already installed or missing source')
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if d.shape!=(2,) or not torch.isfinite(d).all() or not -1.5<=d[0]<d[1]<=self.reverse_update_domain[0]:
            raise ValueError('Extension must preserve the original full-active speed boundary')
        self.register_buffer('reverse_extension_domain',d.clone())
        self.reverse_extension_anchor=copy.deepcopy(self.reverse_network).requires_grad_(False)

    def reverse_output(self, observation):
        value=torch.tanh(self.reverse_network(observation))
        if hasattr(self,'reverse_extension_anchor'):
            reference=torch.tanh(self.reverse_extension_anchor(observation))
            d=self.reverse_extension_domain
            t=((d[1]-observation[...,7]/.5)/(d[1]-d[0])).clamp(0.,1.)
            gate=t*t*(3.-2.*t)
            value=reference+gate[...,None]*(value-reference)
        if hasattr(self,'reverse_update_anchor'):
            anchor=torch.tanh(self.reverse_update_anchor(observation))
            d=self.reverse_update_domain
            t=((d[1]-observation[...,7]/.5)/(d[1]-d[0])).clamp(0.,1.)
            gate=t*t*(3.-2.*t)
            value=anchor+gate[...,None]*(value-anchor)
        return value

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

    def install_straight_update_scope(self, domain):
        """Freeze the previous straight output outside an explicit speed window."""
        if not hasattr(self,'straight_extension_network') or hasattr(self,'straight_update_anchor'):
            raise ValueError('Scope requires one existing straight branch and no previous scope')
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if d.shape!=(4,) or not torch.isfinite(d).all() or not 0<=d[0]<d[1]<=d[2]<d[3]<=3.4:
            raise ValueError('Invalid straight update scope')
        self.register_buffer('straight_update_domain',d.clone())
        self.straight_update_anchor=copy.deepcopy(self.straight_extension_network).requires_grad_(False)

    def straight_update_gate(self, observation):
        if not hasattr(self,'straight_update_domain'):return torch.ones_like(observation[...,7])
        d=self.straight_update_domain;v=observation[...,7]/.5
        def smooth(t):
            t=t.clamp(0.,1.);return t*t*(3.-2.*t)
        return smooth((v-d[0])/(d[1]-d[0]))*(1.-smooth((v-d[2])/(d[3]-d[2])))

    def install_high_straight(self, budget, domain):
        if hasattr(self,'high_straight_network') or not hasattr(self,'straight_update_anchor'):
            raise ValueError('High straight branch requires protected existing straight source')
        b=torch.as_tensor(budget,device=self.budget.device,dtype=self.budget.dtype)
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if b.shape!=(6,) or not torch.isfinite(b).all() or (b<0).any() or (b>.5).any():raise ValueError('Invalid high straight budget')
        if d.shape!=(4,) or not torch.isfinite(d).all() or not 1.5<=d[0]<d[1]<=d[2]<d[3]<=3.4:raise ValueError('Invalid high straight domain')
        self.register_buffer('high_straight_budget',b.clone())
        self.register_buffer('high_straight_domain',d.clone())
        self.high_straight_network=copy.deepcopy(self.network)
        nn.init.zeros_(self.high_straight_network[-1].weight)
        nn.init.zeros_(self.high_straight_network[-1].bias)

    def high_straight_gate(self, observation):
        d=self.high_straight_domain;v=observation[...,7]/.5
        def smooth(t):
            t=t.clamp(0.,1.);return t*t*(3.-2.*t)
        yaw=observation[...,2]/.25+observation[...,8]
        return (smooth((v-d[0])/(d[1]-d[0]))*(1.-smooth((v-d[2])/(d[3]-d[2])))
                *((yaw.abs()<=1e-4)&(observation[...,8].abs()<1.99)).to(v.dtype))

    def install_timing_network(self, budget):
        """Zero-initialized bounded timing adaptation; no command or actuator loop."""
        if hasattr(self,'timing_network') or not hasattr(self,'alignment_network'):
            raise ValueError('Timing branch requires existing physical actor and no timing head')
        b=torch.as_tensor(budget,device=self.budget.device,dtype=self.budget.dtype)
        if b.shape!=(6,) or not torch.isfinite(b).all() or (b<0).any() or (b>.2).any():
            raise ValueError('Invalid timing residual budget')
        self.register_buffer('timing_budget',b.clone())
        self.timing_network=copy.deepcopy(self.network)
        nn.init.zeros_(self.timing_network[-1].weight)
        nn.init.zeros_(self.timing_network[-1].bias)

    def install_delay_local_network(self, budget, domain):
        """Add a zero-output local repair without changing the existing timing actor."""
        if not hasattr(self, 'timing_network') or hasattr(self, 'delay_local_network'):
            raise ValueError('Local delay repair requires one existing timing branch')
        b = torch.as_tensor(budget, device=self.budget.device, dtype=self.budget.dtype)
        d = torch.as_tensor(domain, device=self.budget.device, dtype=self.budget.dtype)
        if (b.shape != (6,) or not torch.isfinite(b).all() or (b < 0).any()
                or (b > .03).any() or d.shape != (4,) or not torch.isfinite(d).all()
                or not bool((d[1:] > d[:-1]).all()) or d[0] < 0 or d[-1] > 3.2):
            raise ValueError('Local delay budget <=.03 and ordered forward speed bounds required')
        self.register_buffer('delay_local_budget', b.clone())
        self.register_buffer('delay_local_domain', d.clone())
        self.delay_local_network = copy.deepcopy(self.network)
        nn.init.zeros_(self.delay_local_network[-1].weight)
        nn.init.zeros_(self.delay_local_network[-1].bias)

    def delay_local_gate(self, observation):
        d = self.delay_local_domain
        v = observation[..., 7] / .5
        def smooth(t):
            t = t.clamp(0., 1.)
            return t*t*(3.-2.*t)
        yaw = observation[..., 2] / .25 + observation[..., 8]
        return (smooth((v-d[0])/(d[1]-d[0]))
                * (1.-smooth((v-d[2])/(d[3]-d[2])))
                * ((yaw.abs() <= 1e-4) & (observation[..., 8].abs() < 1.99)).to(v.dtype))

    def install_startup_network(self, budget, domain):
        """Zero-output forward straight specialist; old controller stays frozen."""
        if hasattr(self, 'startup_network'):
            raise ValueError('Startup head already installed')
        b=torch.as_tensor(budget,device=self.budget.device,dtype=self.budget.dtype)
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (b.shape!=(6,) or not torch.isfinite(b).all() or (b<0).any() or (b>.2).any()
                or d.shape!=(4,) or not torch.isfinite(d).all()
                or not bool((d[1:]>d[:-1]).all()) or d[0]<0 or d[-1]>2.1):
            raise ValueError('Startup budget<=.2, strictly ordered forward domain ending<=2.1 required')
        self.register_buffer('startup_budget',b.clone())
        self.register_buffer('startup_domain',d.clone())
        self.startup_network=copy.deepcopy(self.network)
        nn.init.zeros_(self.startup_network[-1].weight)
        nn.init.zeros_(self.startup_network[-1].bias)

    def startup_gate(self, observation):
        d=self.startup_domain;v=observation[...,7]/.5
        def smooth(t):
            t=t.clamp(0.,1.);return t*t*(3.-2.*t)
        yaw=observation[...,2]/.25+observation[...,8]
        return (smooth((v-d[0])/(d[1]-d[0]))*(1-smooth((v-d[2])/(d[3]-d[2])))
                *((yaw.abs()<=1e-4)&(observation[...,8].abs()<1.99)).to(v.dtype))

    def install_stop_network(self, budget):
        """Zero-initialized bounded stop repair; no change to the observation contract."""
        if hasattr(self, 'stop_network'):
            raise ValueError('Stop repair already installed')
        b = torch.as_tensor(budget, device=self.budget.device, dtype=self.budget.dtype)
        if b.shape != (6,) or not torch.isfinite(b).all() or (b < 0).any() or (b > .2).any():
            raise ValueError('Stop residual budget must be finite, nonnegative and <=.2')
        self.register_buffer('stop_budget', b.clone())
        self.stop_network = copy.deepcopy(self.network)
        nn.init.zeros_(self.stop_network[-1].weight)
        nn.init.zeros_(self.stop_network[-1].bias)

    def install_stop_budget_anchor(self, anchor_budget):
        """Preserve source outputs while allowing bounded changes in a larger box."""
        if not hasattr(self, 'stop_network') or hasattr(self, 'stop_budget_anchor'):
            raise ValueError('One existing unanchored stop head required')
        b = torch.as_tensor(anchor_budget, device=self.budget.device, dtype=self.budget.dtype)
        if b.shape != (6,) or not torch.isfinite(b).all() or (b < 0).any() or (b > self.stop_budget).any():
            raise ValueError('Anchor budget must lie within new stop budget')
        self.register_buffer('stop_anchor_budget', b.clone())
        self.stop_budget_anchor = copy.deepcopy(self.stop_network).requires_grad_(False)

    def install_stop_update_scope(self, domain):
        """Freeze low-speed stopping while learning bounded high-speed corrections."""
        if not hasattr(self, 'stop_network') or hasattr(self, 'stop_update_anchor'):
            raise ValueError('One existing unscoped stop head required')
        d = torch.as_tensor(domain, device=self.budget.device, dtype=self.budget.dtype)
        # obs6 clips at +/-2 and its scale is2.1: do not pretend to distinguish
        # high measured speeds beyond the available observation resolution.
        if d.shape != (2,) or not torch.isfinite(d).all() or not 0 < d[0] < d[1] < 2./2.1:
            raise ValueError('Stop scope needs ordered positive observable speed bounds')
        self.register_buffer('stop_update_domain', d.clone())
        self.stop_update_anchor = copy.deepcopy(self.stop_network).requires_grad_(False)

    def stop_update_gate(self, observation):
        velocity = observation[..., 7] / .5 - observation[..., 6] / 2.1
        d = self.stop_update_domain
        t = ((velocity-d[0])/(d[1]-d[0])).clamp(0., 1.)
        return t*t*(3.-2.*t)

    def install_stop_rebound(self, domain):
        """Zero-initialized correction inside the existing stop residual envelope."""
        if not hasattr(self, 'stop_network') or hasattr(self, 'stop_rebound_network'):
            raise ValueError('One existing stop head without rebound correction required')
        d = torch.as_tensor(domain, device=self.budget.device, dtype=self.budget.dtype)
        if (d.shape != (4,) or not torch.isfinite(d).all()
                or not 0 < d[0] < d[1] < 1. or not 0 < d[2] < d[3] < 2./2.1):
            raise ValueError('Expected pitch-radian and reverse-speed fade bounds')
        self.register_buffer('stop_rebound_domain', d.clone())
        self.stop_rebound_network = copy.deepcopy(self.network)
        nn.init.zeros_(self.stop_rebound_network[-1].weight)
        nn.init.zeros_(self.stop_rebound_network[-1].bias)

    def stop_rebound_gate(self, observation):
        d = self.stop_rebound_domain
        pitch = torch.atan2(observation[..., 3], -observation[..., 5]).abs()
        reverse = observation[..., 6]/2.1 - observation[..., 7]/.5
        def smooth(x):
            t=x.clamp(0., 1.)
            return t*t*(3.-2.*t)
        p=smooth((pitch-d[0])/(d[1]-d[0]))
        v=smooth((reverse-d[2])/(d[3]-d[2]))
        return 1.-(1.-p)*(1.-v)

    def _stop_bounded_value(self, observation, logits):
        value = torch.tanh(logits)
        if hasattr(self, 'stop_budget_anchor'):
            anchor_logits = self.stop_budget_anchor(observation)
            reference = torch.tanh(anchor_logits)
            ratio = self.stop_anchor_budget / self.stop_budget.clamp_min(1.e-12)
            center = torch.atanh((ratio * reference).clamp(-1.+1.e-6, 1.-1.e-6))
            expanded = self.stop_budget * torch.tanh((logits - anchor_logits) + center)
            # Unexpanded slots retain their exact original computation.
            value = torch.where(self.stop_budget > self.stop_anchor_budget,
                                expanded, self.stop_budget * value)
        else:
            value = self.stop_budget * value
        return value

    def stop_residual(self, observation, logits=None):
        if logits is None:
            logits = self.stop_network(observation)
        value = self._stop_bounded_value(observation, logits)
        if hasattr(self, 'stop_update_anchor'):
            anchor = self._stop_bounded_value(observation, self.stop_update_anchor(observation))
            value = anchor + self.stop_update_gate(observation)[..., None] * (value-anchor)
        if hasattr(self, 'stop_rebound_network'):
            delta = self.stop_budget * torch.tanh(self.stop_rebound_network(observation))
            delta = self.stop_rebound_gate(observation)[..., None] * delta
            delta = torch.maximum(torch.minimum(delta, self.stop_budget-value), -self.stop_budget-value)
            value = value + delta
        return self.stop_gate(observation)[..., None] * value

    def stop_gate(self, observation):
        speed = observation[..., 7] / .5
        yaw = observation[..., 2] / .25 + observation[..., 8]
        t = (1. - speed.abs() / .05).clamp(0., 1.)
        return t*t*(3.-2.*t) * ((yaw.abs() <= 1e-4) & (observation[..., 8].abs() < 1.99)).to(t.dtype)

    def install_timing_update_scope(self, domain):
        """Keep a frozen timing anchor outside an explicit command-speed window."""
        if not hasattr(self, 'timing_network') or hasattr(self, 'timing_update_anchor'):
            raise ValueError('Timing scope requires one unscoped timing head')
        d = torch.as_tensor(domain, device=self.budget.device, dtype=self.budget.dtype)
        if d.shape != (4,) or not torch.isfinite(d).all() or not bool((d[1:] > d[:-1]).all()):
            raise ValueError('Timing scope requires four strictly increasing finite bounds')
        self.register_buffer('timing_update_domain', d.clone())
        self.timing_update_anchor = copy.deepcopy(self.timing_network).requires_grad_(False)

    def timing_update_gate(self, observation):
        if not hasattr(self, 'timing_update_domain'):
            return torch.ones_like(observation[..., 7])
        d = self.timing_update_domain
        speed = observation[..., 7] / .5
        def smooth(t):
            t = t.clamp(0., 1.)
            return t*t*(3.-2.*t)
        yaw = observation[..., 2] / .25 + observation[..., 8]
        return (smooth((speed-d[0])/(d[1]-d[0]))
                * (1.-smooth((speed-d[2])/(d[3]-d[2])))
                * ((yaw.abs() <= 1e-4) & (observation[..., 8].abs() < 1.99)).to(speed.dtype))

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
        if hasattr(self,'joint_delay_network'):
            original=original+self.joint_delay_gate(observation)[...,None]*self.joint_delay_budget*torch.tanh(self.joint_delay_network(observation))
        if hasattr(self, 'startup_network'):
            original = original + self.startup_gate(observation)[..., None] * self.startup_budget * torch.tanh(self.startup_network(observation))
        if hasattr(self, 'stop_network'):
            original = original + self.stop_residual(observation)
        if hasattr(self, 'reverse_network'):
            original = original + (self.reverse_gate(observation)[..., None]
                * self.reverse_budget * self.reverse_output(observation))
        if hasattr(self, 'reverse_reserve_network'):
            original = original + self.reverse_reserve_output(observation)
        if not hasattr(self, 'alignment_network'):
            return original
        result = original + (self.alignment_gate(observation)[..., None]
                           * self.alignment_budget
                           * torch.tanh(self.alignment_network(observation)))
        if hasattr(self, 'straight_extension_network'):
            output=torch.tanh(self.straight_extension_network(observation))
            if hasattr(self,'straight_update_anchor'):
                anchor=torch.tanh(self.straight_update_anchor(observation))
                output=anchor+self.straight_update_gate(observation)[...,None]*(output-anchor)
            result = result + (self.straight_extension_gate(observation)[..., None]
                               * self.straight_extension_budget * output)
        if hasattr(self,'high_straight_network'):
            result=result+self.high_straight_gate(observation)[...,None]*self.high_straight_budget*torch.tanh(self.high_straight_network(observation))
        if hasattr(self,'timing_network'):
            output = torch.tanh(self.timing_network(observation))
            if hasattr(self, 'timing_update_anchor'):
                anchor = torch.tanh(self.timing_update_anchor(observation))
                gate = self.timing_update_gate(observation)[..., None]
                blended = anchor + gate * (output-anchor)
                output = torch.where(gate <= 0, anchor, torch.where(gate >= 1, output, blended))
            result=result+self.timing_budget*output
        if hasattr(self, 'delay_local_network'):
            result = result + self.delay_local_gate(observation)[..., None] * self.delay_local_budget * torch.tanh(self.delay_local_network(observation))
        return result

    def forward(self,observation,source_action):
        return source_action+self.correction(observation)
