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
        reserve = getattr(self, 'reverse_reserve_budget', torch.zeros_like(self.budget))
        reserve = torch.maximum(reserve, getattr(self, 'reverse_reserve_frontier_budget', reserve))
        reserve = torch.maximum(reserve, getattr(self, 'reverse_reserve_fast_preserved_budget', reserve))
        reserve = torch.maximum(reserve, getattr(self, 'reverse_reserve_extension_budget', reserve))
        reserve = torch.maximum(reserve, getattr(self, 'reverse_reserve_extension_retained_budget', reserve))
        # Base domains are disjoint; optional bridge is a convex combination.
        return self.budget + torch.maximum(alignment, extension) + reverse + getattr(self,'high_straight_budget',torch.zeros_like(self.budget)) + getattr(self,'timing_budget',torch.zeros_like(self.budget)) + getattr(self,'delay_local_budget',torch.zeros_like(self.budget)) + getattr(self,'stop_budget',torch.zeros_like(self.budget)) + getattr(self,'startup_budget',torch.zeros_like(self.budget)) + reserve + getattr(self,'joint_delay_budget',torch.zeros_like(self.budget))

    def install_reverse_network(self, budget, domain):
        """Explicit zero-output reverse-only branch; forward routing is untouched."""
        if hasattr(self, 'reverse_network'):
            raise ValueError('Reverse branch already installed')
        b = torch.as_tensor(budget, dtype=self.budget.dtype, device=self.budget.device)
        d = torch.as_tensor(domain, dtype=self.budget.dtype, device=self.budget.device)
        if (b.shape != (6,) or not torch.isfinite(b).all() or (b < 0).any()
                or (b > .5).any() or d.shape != (4,) or not torch.isfinite(d).all()
                or not -2.0 <= d[0] < d[1] <= d[2] < d[3] <= 0):
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
                or not -2.0<=d[0]<d[1]<=d[2]<d[3]<=3.5 or not 0<=d[4]<d[5]<=.5):
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

    def install_joint_delay_reference(self):
        """Keep a frozen reference outside a command-scoped head update.

        Both branches share one residual budget. Checkpoints without this
        optional reference retain their original correction exactly.
        """
        if (not hasattr(self, 'joint_delay_domain')
                or hasattr(self, 'joint_delay_reference_network')):
            raise ValueError('Scoped joint-delay head and no existing reference required')
        self.joint_delay_reference_network = copy.deepcopy(self.joint_delay_network)
        self.joint_delay_reference_network.requires_grad_(False)

    def install_joint_turn_reference(self, domain):
        """Select a learned reference-head update inside a signed command box.

        This blends within the existing action budget; it is not an outer loop.
        """
        d = torch.as_tensor(domain, device=self.budget.device, dtype=self.budget.dtype)
        if (not hasattr(self, 'joint_delay_reference_network')
                or hasattr(self, 'joint_turn_source_network') or d.shape != (8,)
                or not torch.isfinite(d).all()
                or not 0 <= d[0] < d[1] <= d[2] < d[3] <= 3.5
                or not -2 <= d[4] < d[5] <= d[6] < d[7] <= 2):
            raise ValueError('Explicit ordered vx/signed-yaw support required')
        self.register_buffer('joint_turn_domain', d.clone())
        self.joint_turn_source_network = copy.deepcopy(self.joint_delay_reference_network)
        self.joint_turn_source_network.requires_grad_(False)

    def joint_turn_gate(self, observation):
        d = self.joint_turn_domain
        vx = observation[..., 7] / .5
        yaw = observation[..., 2] / .25 + observation[..., 8]
        smooth = lambda x: x.clamp(0., 1.).square() * (3. - 2. * x.clamp(0., 1.))
        return (smooth((vx-d[0])/(d[1]-d[0])) * smooth((d[3]-vx)/(d[3]-d[2]))
                * smooth((yaw-d[4])/(d[5]-d[4])) * smooth((d[7]-yaw)/(d[7]-d[6]))
                * (observation[..., 8].abs() < 1.999).to(vx.dtype))

    def install_joint_turn_update_scope(self, domain):
        """Freeze the current turn reference outside a narrower update box."""
        if not hasattr(self, 'joint_turn_source_network') or hasattr(self, 'joint_turn_update_anchor'):
            raise ValueError('Existing turn reference and no update anchor required')
        d = torch.as_tensor(domain, device=self.budget.device, dtype=self.budget.dtype)
        if (d.shape != (8,) or not torch.isfinite(d).all()
                or not 0 <= d[0] < d[1] <= d[2] < d[3] <= 3.5
                or not -2 <= d[4] < d[5] <= d[6] < d[7] <= 2):
            raise ValueError('Explicit ordered update box required')
        self.register_buffer('joint_turn_update_domain', d.clone())
        self.joint_turn_update_anchor = copy.deepcopy(self.joint_delay_reference_network)
        self.joint_turn_update_anchor.requires_grad_(False)

    def joint_turn_update_gate(self, observation):
        from types import SimpleNamespace
        return PhysicsAdaptation.joint_turn_gate(
            SimpleNamespace(joint_turn_domain=self.joint_turn_update_domain), observation)

    def install_joint_turn_source_update_scope(self, domain):
        """Restrict updates of the shared source branch without changing its budget.

        Installation is function preserving. The copied source stays frozen
        outside the explicit command box; legacy checkpoints keep their path.
        """
        if not hasattr(self, 'joint_turn_source_network') or hasattr(self, 'joint_turn_source_update_domain'):
            raise ValueError('Existing turn source and no previous source update scope required')
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (d.shape!=(8,) or not torch.isfinite(d).all()
                or not 0<=d[0]<d[1]<=d[2]<d[3]<=3.5
                or not -2<=d[4]<d[5]<=d[6]<d[7]<=2):
            raise ValueError('Explicit ordered source-update vx/yaw support required')
        self.register_buffer('joint_turn_source_update_domain',d.clone())
        self.joint_turn_source_anchor=copy.deepcopy(self.joint_turn_source_network).requires_grad_(False)

    def joint_turn_source_update_gate(self, observation):
        from types import SimpleNamespace
        return PhysicsAdaptation.joint_turn_gate(
            SimpleNamespace(joint_turn_domain=self.joint_turn_source_update_domain), observation)

    def install_joint_turn_source_difference_anchor(self):
        """Opt-in frozen anchor: retain only a candidate's wheel difference update."""
        if (not hasattr(self,'joint_turn_source_update_domain')
                or hasattr(self,'joint_turn_source_difference_anchor')
                or hasattr(self,'joint_turn_source_leg_difference_anchor')
                or self.joint_delay_budget[2]!=self.joint_delay_budget[5]):
            raise ValueError('Scoped source and equal wheel budgets required for one mode projection')
        self.joint_turn_source_difference_anchor=copy.deepcopy(self.joint_turn_source_network).requires_grad_(False)
        self.register_buffer('joint_turn_source_difference_only',self.budget.new_tensor(1.))

    def install_joint_turn_source_leg_difference_anchor(self):
        """Keep the learned leg and wheel-difference update, restore wheel common."""
        if (not hasattr(self,'joint_turn_source_update_domain')
                or hasattr(self,'joint_turn_source_leg_difference_anchor')
                or hasattr(self,'joint_turn_source_difference_anchor')
                or self.joint_delay_budget[2]!=self.joint_delay_budget[5]):
            raise ValueError('Scoped source and equal wheel budgets required for leg/difference projection')
        self.joint_turn_source_leg_difference_anchor=copy.deepcopy(self.joint_turn_source_network).requires_grad_(False)
        self.register_buffer('joint_turn_source_leg_difference_only',self.budget.new_tensor(1.))

    def joint_turn_source_output(self, observation):
        output=torch.tanh(self.joint_turn_source_network(observation))
        if hasattr(self,'joint_turn_source_leg_difference_anchor'):
            anchor=torch.tanh(self.joint_turn_source_leg_difference_anchor(observation))
            delta=.5*((output[...,5]-anchor[...,5])-(output[...,2]-anchor[...,2]))
            low=torch.maximum(anchor[...,2]-1.,-1.-anchor[...,5])
            high=torch.minimum(anchor[...,2]+1.,1.-anchor[...,5])
            delta=torch.maximum(torch.minimum(delta,high),low)
            return torch.stack((output[...,0],output[...,1],anchor[...,2]-delta,
                                output[...,3],output[...,4],anchor[...,5]+delta),dim=-1)
        if not hasattr(self,'joint_turn_source_difference_anchor'):
            return output
        anchor=torch.tanh(self.joint_turn_source_difference_anchor(observation))
        delta=.5*((output[...,5]-anchor[...,5])-(output[...,2]-anchor[...,2]))
        # Intersect both wheels' available ranges. This preserves the anchor's
        # common wheel output and the existing [-1,1] residual envelope.
        low=torch.maximum(anchor[...,2]-1.,-1.-anchor[...,5])
        high=torch.minimum(anchor[...,2]+1.,1.-anchor[...,5])
        delta=torch.maximum(torch.minimum(delta,high),low)
        return torch.stack((anchor[...,0],anchor[...,1],anchor[...,2]-delta,
                            anchor[...,3],anchor[...,4],anchor[...,5]+delta),dim=-1)

    def install_joint_turn_anchor_scope(self, domain):
        """Make the shared anchor trainable only for forward turns; preserve straight/spin."""
        if not hasattr(self,'joint_turn_source_anchor') or hasattr(self,'joint_turn_anchor_domain'):
            raise ValueError('One existing source anchor and no previous anchor scope required')
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (d.shape!=(6,) or not torch.isfinite(d).all()
                or not 0<=d[0]<d[1]<=d[2]<d[3]<=3.5 or not 0<d[4]<d[5]<=.5):
            raise ValueError('Ordered forward speed support and positive absolute-yaw fade required')
        self.register_buffer('joint_turn_anchor_domain',d.clone())
        self.joint_turn_anchor_frozen=copy.deepcopy(self.joint_turn_source_anchor).requires_grad_(False)

    def joint_turn_anchor_gate(self, observation):
        d=self.joint_turn_anchor_domain;v=observation[...,7]/.5
        yaw=(observation[...,2]/.25+observation[...,8]).abs()
        smooth=lambda t:t.clamp(0,1).square()*(3-2*t.clamp(0,1))
        return (smooth((v-d[0])/(d[1]-d[0]))*smooth((d[3]-v)/(d[3]-d[2]))
                *smooth((yaw-d[4])/(d[5]-d[4]))*(observation[...,8].abs()<1.999).to(v.dtype))

    def install_reverse_reserve(self, budget, domain, preserved_transition=None):
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
        if preserved_transition is not None:
            p=torch.as_tensor(preserved_transition,device=b.device,dtype=b.dtype)
            if (p.shape!=(4,) or not torch.isfinite(p).all()
                    or not -1.5<=p[0]<p[1]<=-.6
                    or not p[0]<=p[2]<p[3]<=max(p[1],d[1])):
                raise ValueError('Expected reference reserve bounds and ordered negative-speed blend bounds')
            self.register_buffer('reverse_reserve_preserved_transition',p.clone())

    def reverse_reserve_gate(self, observation):
        d=self.reverse_reserve_domain
        speed=observation[...,7]/.5
        t=((d[1]-speed)/(d[1]-d[0])).clamp(0.,1.)
        gate=self.reverse_gate(observation)*t*t*(3.-2.*t)
        if hasattr(self,'reverse_reserve_preserved_transition'):
            # Checkpoint-opt-in: retain the verified faster-command branch
            # exactly, blending only the slower acquisition transition.
            p=self.reverse_reserve_preserved_transition
            r=((p[1]-speed)/(p[1]-p[0])).clamp(0.,1.)
            reference=self.reverse_gate(observation)*r*r*(3.-2.*r)
            u=((speed-p[2])/(p[3]-p[2])).clamp(0.,1.)
            mix=u*u*(3.-2.*u)
            blended=reference+mix*(gate-reference)
            gate=torch.where(mix==0,reference,torch.where(mix==1,gate,blended))
        return gate

    def install_reverse_reserve_frontier(self, budget, domain):
        """Opt-in bounded wheel reserve change beyond a preserved command frontier."""
        if not hasattr(self, 'reverse_reserve_network') or hasattr(self, 'reverse_reserve_frontier_budget'):
            raise ValueError('Existing reserve and no frontier configuration required')
        b=torch.as_tensor(budget,device=self.budget.device,dtype=self.budget.dtype)
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (b.shape!=(6,) or not torch.isfinite(b).all() or b[[0,1,3,4]].abs().max()!=0
                or not 0 < b[2] <= .15 or b[2]!=b[5]
                or d.shape!=(2,) or not torch.isfinite(d).all() or not -1.5<=d[0]<d[1]<=-.8):
            raise ValueError('Common-wheel frontier budget<=.15 and ordered bounds below-.8 required')
        self.register_buffer('reverse_reserve_frontier_budget',b.clone())
        self.register_buffer('reverse_reserve_frontier_domain',d.clone())

    def install_reverse_base_leg_response(self, gain, center, domain, limit):
        """Opt-in local affine gain on base-policy leg outputs, with bounded delta.

        This is a teacher parameter probe, not a sensor filter or actuator law.
        Old policies and commands outside the signed reverse scope stay exact.
        """
        if hasattr(self,'reverse_base_leg_gain'):
            raise ValueError('Reverse base leg response already installed')
        g=torch.as_tensor(gain,device=self.budget.device,dtype=self.budget.dtype)
        c=torch.as_tensor(center,device=self.budget.device,dtype=self.budget.dtype)
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        b=torch.as_tensor(limit,device=self.budget.device,dtype=self.budget.dtype)
        # Gain is an explicitly checkpoint-bound neural response parameter.
        # Its independent delta bound still limits the largest action change;
        # raising gain must never silently raise that bound or actuator PD.
        if (g.ndim!=0 or not torch.isfinite(g) or not .5<=g<=2.0
                or c.shape!=(6,) or not torch.isfinite(c).all() or c.abs().max()>2
                or d.shape!=(2,) or not torch.isfinite(d).all() or not -1.5<=d[0]<d[1]<=-.8
                or b.ndim!=0 or not torch.isfinite(b) or not 0<b<=.1):
            raise ValueError('Finite gain.5..2, center6D, reverse domain below-.8, delta limit<=.1 required')
        self.register_buffer('reverse_base_leg_gain',g.clone())
        self.register_buffer('reverse_base_leg_center',c.clone())
        self.register_buffer('reverse_base_leg_domain',d.clone())
        self.register_buffer('reverse_base_leg_limit',b.clone())

    def install_reverse_base_leg_fast_limit(self, limit, domain):
        if not hasattr(self,'reverse_base_leg_gain') or hasattr(self,'reverse_base_leg_fast_limit'):
            raise ValueError('Existing base leg response and no fast limit required')
        b=torch.as_tensor(limit,device=self.budget.device,dtype=self.budget.dtype)
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (b.ndim!=0 or not torch.isfinite(b) or not 0<b<=.1
                or d.shape!=(2,) or not torch.isfinite(d).all()
                or not -1.5<=d[0]<d[1]<=self.reverse_base_leg_domain[0]):
            raise ValueError('Fast leg limit<=.1 with scope below base response transition required')
        self.register_buffer('reverse_base_leg_fast_limit',b.clone())
        self.register_buffer('reverse_base_leg_fast_limit_domain',d.clone())

    def install_reverse_base_leg_fast_gain(self, gain, domain):
        """Command-local gain change; retain the old base response above the boundary."""
        if not hasattr(self,'reverse_base_leg_gain') or hasattr(self,'reverse_base_leg_fast_gain'):
            raise ValueError('Existing base leg response and no fast gain required')
        g=torch.as_tensor(gain,device=self.budget.device,dtype=self.budget.dtype)
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (g.ndim!=0 or not torch.isfinite(g) or not .5<=g<=2.
                or d.shape!=(2,) or not torch.isfinite(d).all()
                or not -1.8<=d[0]<d[1]<=self.reverse_base_leg_domain[0]):
            raise ValueError('Fast gain.5..2 with ordered reverse bounds below base transition required')
        self.register_buffer('reverse_base_leg_fast_gain',g.clone())
        self.register_buffer('reverse_base_leg_fast_gain_domain',d.clone())

    def install_reverse_base_leg_extension_gain(self, gain, domain):
        """Extend a validated fast response without changing its learned domain."""
        if not hasattr(self,'reverse_base_leg_fast_gain') or hasattr(self,'reverse_base_leg_extension_gain'):
            raise ValueError('Existing fast leg response and no extension required')
        g=torch.as_tensor(gain,device=self.budget.device,dtype=self.budget.dtype)
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (g.ndim!=0 or not torch.isfinite(g) or not .5<=g<=2.
                or d.shape!=(2,) or not torch.isfinite(d).all()
                or not -1.8<=d[0]<d[1]<=self.reverse_base_leg_fast_gain_domain[0]):
            raise ValueError('Extension gain.5..2 and domain below verified fast response required')
        self.register_buffer('reverse_base_leg_extension_gain',g.clone())
        self.register_buffer('reverse_base_leg_extension_gain_domain',d.clone())

    def reverse_base_leg_response(self, observation, base_action):
        if not hasattr(self,'reverse_base_leg_gain'):
            return base_action
        d=self.reverse_base_leg_domain
        t=((d[1]-observation[...,7]/.5)/(d[1]-d[0])).clamp(0.,1.)
        yaw=observation[...,2]/.25+observation[...,8]
        gate=t*t*(3.-2.*t)*(yaw.abs()<=1e-4).to(observation.dtype)
        limit=self.reverse_base_leg_limit
        if hasattr(self,'reverse_base_leg_fast_limit'):
            fd=self.reverse_base_leg_fast_limit_domain
            ft=((fd[1]-observation[...,7]/.5)/(fd[1]-fd[0])).clamp(0.,1.)
            fmix=ft*ft*(3.-2.*ft)
            limit=(limit+fmix*(self.reverse_base_leg_fast_limit-limit))[...,None]
        gain=self.reverse_base_leg_gain
        if hasattr(self,'reverse_base_leg_fast_gain'):
            gd=self.reverse_base_leg_fast_gain_domain
            gt=((gd[1]-observation[...,7]/.5)/(gd[1]-gd[0])).clamp(0.,1.)
            gmix=gt*gt*(3.-2.*gt)
            blended=gain+gmix*(self.reverse_base_leg_fast_gain-gain)
            gain=torch.where(gmix==0,gain,torch.where(gmix==1,self.reverse_base_leg_fast_gain,blended))[...,None]
        if hasattr(self,'reverse_base_leg_extension_gain'):
            gd=self.reverse_base_leg_extension_gain_domain
            gt=((gd[1]-observation[...,7]/.5)/(gd[1]-gd[0])).clamp(0.,1.)
            gmix=(gt*gt*(3.-2.*gt))[...,None]
            blended=gain+gmix*(self.reverse_base_leg_extension_gain-gain)
            gain=torch.where(gmix==0,gain,torch.where(gmix==1,self.reverse_base_leg_extension_gain,blended))
        delta=((gain-1)*(base_action-self.reverse_base_leg_center)).clamp(-limit,limit)
        leg_mask=base_action.new_tensor([1.,1.,0.,1.,1.,0.])
        changed=base_action+gate[...,None]*leg_mask*delta
        return torch.where((gate==0)[...,None],base_action,changed)

    def install_forward_base_leg_response(self, gain, center, domain, limit):
        """Checkpoint-opt-in bounded leg response; actuator PD remains unchanged."""
        if hasattr(self, 'forward_base_leg_gain'):
            raise ValueError('Forward leg response already installed')
        g,c,d,b=[torch.as_tensor(v,device=self.budget.device,dtype=self.budget.dtype)
                 for v in (gain,center,domain,limit)]
        if (g.numel()!=1 or b.numel()!=1 or c.shape!=(6,) or d.shape!=(4,)
                or not all(torch.isfinite(v).all() for v in [g,c,d,b])
                or not .5<=g<=1.5 or not 0<b<=.06
                or not 1.5<=d[0]<d[1]<=d[2]<d[3]<=3.4):
            raise ValueError('Bounded forward gain, center and high-speed domain required')
        for name,value in [('gain',g.reshape(())),('center',c),('domain',d),('limit',b.reshape(()))]:
            self.register_buffer('forward_base_leg_'+name,value.clone())

    def forward_base_leg_response(self, observation, base_action):
        if not hasattr(self,'forward_base_leg_gain'):
            return base_action
        d=self.forward_base_leg_domain;v=observation[...,7]/.5
        yaw=observation[...,2]/.25+observation[...,8]
        smooth=lambda t:t.clamp(0,1).square()*(3-2*t.clamp(0,1))
        gate=smooth((v-d[0])/(d[1]-d[0]))*smooth((d[3]-v)/(d[3]-d[2]))
        gate=gate*((yaw.abs()<=1e-4)&(observation[...,8].abs()<1.999)).to(v.dtype)
        delta=((self.forward_base_leg_gain-1)*(base_action-self.forward_base_leg_center)).clamp(
            -self.forward_base_leg_limit,self.forward_base_leg_limit)
        if hasattr(self,'forward_base_leg_upper_gain'):
            upper=((self.forward_base_leg_upper_gain-1)*(base_action-self.forward_base_leg_center)).clamp(
                -self.forward_base_leg_limit,self.forward_base_leg_limit)
            bounds=self.forward_base_leg_upper_domain
            blend=smooth((v-bounds[0])/(bounds[1]-bounds[0]))[...,None]
            delta=torch.where(blend==0,delta,torch.where(blend==1,upper,delta+blend*(upper-delta)))
        changed=base_action+gate[...,None]*base_action.new_tensor([1,1,0,1,1,0])*delta
        return torch.where((gate==0)[...,None],base_action,changed)

    def install_forward_base_leg_upper_gain(self, gain, domain):
        """Keep the validated lower-speed response; reuse an upper-speed gain."""
        if not hasattr(self,'forward_base_leg_gain') or hasattr(self,'forward_base_leg_upper_gain'):
            raise ValueError('Existing forward response and no previous upper gain required')
        g=torch.as_tensor(gain,device=self.budget.device,dtype=self.budget.dtype)
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (g.numel()!=1 or d.shape!=(2,) or not torch.isfinite(g).all() or not torch.isfinite(d).all()
                or not .5<=g<=1.5
                or not self.forward_base_leg_domain[1]<=d[0]<d[1]<=self.forward_base_leg_domain[2]):
            raise ValueError('Upper gain must stay bounded and transition inside the existing plateau')
        self.register_buffer('forward_base_leg_upper_gain',g.reshape(()).clone())
        self.register_buffer('forward_base_leg_upper_domain',d.clone())

    def install_reverse_reserve_fast_preservation(self, budget, domain):
        """Preserve a prior faster-command wheel budget beyond a local repair."""
        if not hasattr(self,'reverse_reserve_frontier_budget') or hasattr(self,'reverse_reserve_fast_preserved_budget'):
            raise ValueError('Existing frontier and no faster preservation required')
        b=torch.as_tensor(budget,device=self.budget.device,dtype=self.budget.dtype)
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (b.shape!=(6,) or not torch.isfinite(b).all() or b[[0,1,3,4]].abs().max()!=0
                or not 0<b[2]<=.15 or b[2]!=b[5]
                or d.shape!=(2,) or not torch.isfinite(d).all()
                or not -1.5<=d[0]<d[1]<=self.reverse_reserve_frontier_domain[0]):
            raise ValueError('Bounded common-wheel preservation fully below frontier transition required')
        self.register_buffer('reverse_reserve_fast_preserved_budget',b.clone())
        self.register_buffer('reverse_reserve_fast_preserved_domain',d.clone())

    def reverse_reserve_output(self, observation):
        logits=self.reverse_reserve_network(observation)
        common=torch.tanh((logits[...,2]+logits[...,5])*.5)
        gate=self.reverse_reserve_gate(observation)[...,None]
        original=gate*self.reverse_reserve_budget*common[...,None]
        if hasattr(self,'reverse_reserve_frontier_budget'):
            d=self.reverse_reserve_frontier_domain
            t=((d[1]-observation[...,7]/.5)/(d[1]-d[0])).clamp(0.,1.)
            mix=t*t*(3.-2.*t)
            budget=self.reverse_reserve_budget+mix[...,None]*(self.reverse_reserve_frontier_budget-self.reverse_reserve_budget)
            changed=gate*budget*common[...,None]
            if hasattr(self,'reverse_reserve_fast_preserved_budget'):
                pd=self.reverse_reserve_fast_preserved_domain
                pt=((observation[...,7]/.5-pd[0])/(pd[1]-pd[0])).clamp(0.,1.)
                pmix=pt*pt*(3.-2.*pt)
                preserved=gate*self.reverse_reserve_fast_preserved_budget*common[...,None]
                blended=preserved+pmix[...,None]*(changed-preserved)
                changed=torch.where((pmix==0)[...,None],preserved,
                                    torch.where((pmix==1)[...,None],changed,blended))
            original=torch.where((mix==0)[...,None],original,changed)
        if hasattr(self,'reverse_reserve_extension_budget'):
            d=self.reverse_reserve_extension_domain
            t=((d[1]-observation[...,7]/.5)/(d[1]-d[0])).clamp(0.,1.)
            blend=t*t*(3.-2.*t)
            extended=gate*self.reverse_reserve_extension_budget*common[...,None]
            if hasattr(self,'reverse_reserve_extension_retained_budget'):
                rd=self.reverse_reserve_extension_retention_domain
                u=((rd[1]-observation[...,7]/.5)/(rd[1]-rd[0])).clamp(0.,1.)
                mix=u*u*(3.-2.*u)
                retained=gate*self.reverse_reserve_extension_retained_budget*common[...,None]
                value=retained+mix[...,None]*(extended-retained)
                extended=torch.where((mix==0)[...,None],retained,
                                     torch.where((mix==1)[...,None],extended,value))
            mixed=original+blend[...,None]*(extended-original)
            original=torch.where((blend==0)[...,None],original,
                                 torch.where((blend==1)[...,None],extended,mixed))
        return original

    def install_reverse_reserve_extension(self, budget, domain):
        """Extend the existing wheel head's bounded output beyond a verified speed.

        No new feedback term: reuse the same network/logits and actuator law.
        Zero blend preserves the prior executed policy path bit for bit.
        """
        if not hasattr(self,'reverse_reserve_fast_preserved_budget') or hasattr(self,'reverse_reserve_extension_budget'):
            raise ValueError('Existing preserved reserve and no extension required')
        b=torch.as_tensor(budget,device=self.budget.device,dtype=self.budget.dtype)
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (b.shape!=(6,) or not torch.isfinite(b).all() or b[[0,1,3,4]].abs().max()!=0
                or not 0<b[2]<=.15 or b[2]!=b[5]
                or d.shape!=(2,) or not torch.isfinite(d).all()
                or not -1.8<=d[0]<d[1]<=self.reverse_reserve_fast_preserved_domain[0]):
            raise ValueError('Wheel-only budget<=.15 with ordered extension below preserved frontier required')
        self.register_buffer('reverse_reserve_extension_budget',b.clone())
        self.register_buffer('reverse_reserve_extension_domain',d.clone())

    def install_reverse_reserve_extension_retention(self, budget, domain):
        """Keep the previous extension amplitude outside a new farther probe."""
        if (not hasattr(self,'reverse_reserve_extension_budget')
                or hasattr(self,'reverse_reserve_extension_retained_budget')):
            raise ValueError('Existing wheel extension and no retained extension required')
        b=torch.as_tensor(budget,device=self.budget.device,dtype=self.budget.dtype)
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (b.shape!=(6,) or not torch.isfinite(b).all() or b[[0,1,3,4]].abs().max()!=0
                or not 0<b[2]<=.15 or b[2]!=b[5]
                or d.shape!=(2,) or not torch.isfinite(d).all()
                or not -1.8<=d[0]<d[1]<=self.reverse_reserve_extension_domain[0]):
            raise ValueError('Retained wheel budget<=.15 and scope below the old extension required')
        self.register_buffer('reverse_reserve_extension_retained_budget',b.clone())
        self.register_buffer('reverse_reserve_extension_retention_domain',d.clone())

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
        if hasattr(self,'reverse_frontier_anchor'):
            reference=torch.tanh(self.reverse_frontier_anchor(observation))
            if hasattr(self,'reverse_frontier_progress_anchor'):
                anchor=torch.tanh(self.reverse_frontier_progress_anchor(observation))
                active=self.reverse_frontier_progress_gate(observation)[...,None]
                mixed=anchor+active*(reference-anchor)
                reference=torch.where(active==0,anchor,torch.where(active==1,reference,mixed))
            if hasattr(self,'reverse_frontier_retention_anchor'):
                retained=torch.tanh(self.reverse_frontier_retention_anchor(observation))
                active=self.reverse_frontier_retention_gate(observation)[...,None]
                mixed=retained+active*(reference-retained)
                reference=torch.where(active==0,retained,torch.where(active==1,reference,mixed))
            if hasattr(self,'reverse_frontier_update_anchor'):
                frozen=torch.tanh(self.reverse_frontier_update_anchor(observation))
                update=self.reverse_frontier_update_gate(observation)[...,None]
                blended=frozen+update*(reference-frozen)
                reference=torch.where(update==0,frozen,torch.where(update==1,reference,blended))
            gate=self.reverse_frontier_gate(observation)[...,None]
            mixed=value+gate*(reference-value)
            value=torch.where(gate==0,value,torch.where(gate==1,reference,mixed))
        if hasattr(self,'reverse_coordination_anchor'):
            reference=torch.tanh(self.reverse_coordination_anchor(observation))
            gate=self.reverse_coordination_gate(observation)[...,None]
            mixed=reference+gate*(value-reference)
            value=torch.where(gate==0,reference,torch.where(gate==1,value,mixed))
        if hasattr(self,'reverse_fast_update_anchor'):
            reference=torch.tanh(self.reverse_fast_update_anchor(observation))
            gate=self.reverse_fast_update_gate(observation)[...,None]
            mixed=reference+gate*(value-reference)
            value=torch.where(gate==0,reference,torch.where(gate==1,value,mixed))
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

    def install_reverse_fast_update_scope(self, domain):
        """Freeze the current reverse function outside a later signed frontier."""
        if not hasattr(self,'reverse_network') or hasattr(self,'reverse_fast_update_anchor'):
            raise ValueError('Existing reverse network and no fast update scope required')
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (d.shape!=(2,) or not torch.isfinite(d).all() or not -1.5<=d[0]<d[1]<=-.8
                or (hasattr(self,'reverse_extension_domain') and d[1]>self.reverse_extension_domain[0])):
            raise ValueError('Fast reverse update must preserve the existing extension frontier')
        self.register_buffer('reverse_fast_update_domain',d.clone())
        self.reverse_fast_update_anchor=copy.deepcopy(self.reverse_network).requires_grad_(False)

    def reverse_fast_update_gate(self, observation):
        d=self.reverse_fast_update_domain
        t=((d[1]-observation[...,7]/.5)/(d[1]-d[0])).clamp(0.,1.)
        return t*t*(3.-2.*t)

    def install_reverse_coordination_scope(self, domain):
        """Freeze the current fast reverse response before learning beyond it."""
        if not hasattr(self,'reverse_fast_update_anchor') or hasattr(self,'reverse_coordination_anchor'):
            raise ValueError('Existing fast reverse scope and no coordination scope required')
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (d.shape!=(2,) or not torch.isfinite(d).all()
                or not -1.8<=d[0]<d[1]<=self.reverse_fast_update_domain[0]):
            raise ValueError('Coordination scope must preserve the full-active fast reverse boundary')
        self.register_buffer('reverse_coordination_domain',d.clone())
        self.reverse_coordination_anchor=copy.deepcopy(self.reverse_network).requires_grad_(False)

    def reverse_coordination_gate(self, observation):
        d=self.reverse_coordination_domain
        t=((d[1]-observation[...,7]/.5)/(d[1]-d[0])).clamp(0.,1.)
        return t*t*(3.-2.*t)

    def install_reverse_frontier_preservation(self, domain, reference_state=None):
        """Keep a proven faster branch while retaining nearer-command gains."""
        if not hasattr(self,'reverse_coordination_anchor') or hasattr(self,'reverse_frontier_anchor'):
            raise ValueError('Existing coordination scope and no frontier preservation required')
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (d.shape!=(2,) or not torch.isfinite(d).all()
                or not -1.8<=d[0]<d[1]<=self.reverse_coordination_domain[0]):
            raise ValueError('Frontier preservation must be below the full-active coordination boundary')
        self.register_buffer('reverse_frontier_domain',d.clone())
        self.reverse_frontier_anchor=copy.deepcopy(self.reverse_network).requires_grad_(False)
        if reference_state is not None:self.reverse_frontier_anchor.load_state_dict(reference_state,strict=True)

    def reverse_frontier_gate(self, observation):
        d=self.reverse_frontier_domain
        t=((d[1]-observation[...,7]/.5)/(d[1]-d[0])).clamp(0.,1.)
        return t*t*(3.-2.*t)

    def install_reverse_frontier_update_scope(self, domain):
        """Preserve the current fast branch at slower commands during expansion."""
        if not hasattr(self,'reverse_frontier_anchor') or hasattr(self,'reverse_frontier_update_anchor'):
            raise ValueError('Existing fast frontier and no update scope required')
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (d.shape!=(2,) or not torch.isfinite(d).all()
                or not -1.8<=d[0]<d[1]<=self.reverse_frontier_domain[0]):
            raise ValueError('Frontier update must protect the previously active boundary')
        self.register_buffer('reverse_frontier_update_domain',d.clone())
        self.reverse_frontier_update_anchor=copy.deepcopy(self.reverse_frontier_anchor).requires_grad_(False)

    def reverse_frontier_update_gate(self, observation):
        d=self.reverse_frontier_update_domain
        t=((d[1]-observation[...,7]/.5)/(d[1]-d[0])).clamp(0.,1.)
        return t*t*(3.-2.*t)

    def install_reverse_frontier_retention(self, domain):
        """Preserve the complete previous scoped response before a farther update.

        The frozen head is mixed BEFORE the older update scope, so that older
        transition bands retain their prior output rather than reverting to an
        earlier anchor when the new boundary moves.
        """
        if (not hasattr(self,'reverse_frontier_update_anchor')
                or hasattr(self,'reverse_frontier_retention_anchor')):
            raise ValueError('Existing frontier update scope and no retention anchor required')
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (d.shape!=(2,) or not torch.isfinite(d).all()
                or not -1.8<=d[0]<d[1]<=self.reverse_frontier_update_domain[0]):
            raise ValueError('Retention must preserve the entire older frontier transition')
        self.register_buffer('reverse_frontier_retention_domain',d.clone())
        self.reverse_frontier_retention_anchor=copy.deepcopy(self.reverse_frontier_anchor).requires_grad_(False)

    def reverse_frontier_retention_gate(self, observation):
        d=self.reverse_frontier_retention_domain
        t=((d[1]-observation[...,7]/.5)/(d[1]-d[0])).clamp(0.,1.)
        return t*t*(3.-2.*t)

    def install_reverse_frontier_progress(self, domain):
        """Freeze the complete existing frontier before a later training step.

        Mix before prior retention scopes, preserving their transition bands.
        No physical parameter or observation changes are introduced.
        """
        if (not hasattr(self,'reverse_frontier_retention_anchor')
                or hasattr(self,'reverse_frontier_progress_anchor')):
            raise ValueError('Existing retained frontier and no progress scope required')
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (d.shape!=(2,) or not torch.isfinite(d).all()
                or not -1.8<=d[0]<d[1]<=self.reverse_frontier_retention_domain[0]):
            raise ValueError('Progress scope must preserve prior full-active boundary')
        self.register_buffer('reverse_frontier_progress_domain',d.clone())
        self.reverse_frontier_progress_anchor=copy.deepcopy(self.reverse_frontier_anchor).requires_grad_(False)

    def reverse_frontier_progress_gate(self, observation):
        d=self.reverse_frontier_progress_domain
        t=((d[1]-observation[...,7]/.5)/(d[1]-d[0])).clamp(0.,1.)
        return t*t*(3.-2.*t)

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

    def install_legacy_reverse_gate_extension(self, domain):
        """Opt-in extension of inherited physical correction for straight reverse.

        Only command magnitudes above the old1.5 plateau change. Preserve
        network inputs, weights, residual bounds and all forward/turn routing.
        Narrower historical update scopes remain narrow.
        """
        if hasattr(self,'legacy_reverse_gate_extension'):
            raise ValueError('Legacy reverse extension already installed')
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if (d.shape!=(2,) or not torch.isfinite(d).all()
                or not 1.5<=d[0]<d[1]<=2.2):
            raise ValueError('Ordered legacy reverse plateau/fade magnitudes required')
        self.register_buffer('legacy_reverse_gate_extension',d.clone())

    def install_legacy_forward_turn_extension(self, domain):
        """Extend inherited correction only above1.5 for moderate forward turns."""
        if hasattr(self,'legacy_forward_turn_extension'):
            raise ValueError('Forward turn extension already installed')
        d=torch.as_tensor(domain,device=self.budget.device,dtype=self.budget.dtype)
        if d.shape!=(2,) or not torch.isfinite(d).all() or not 1.5<=d[0]<d[1]<=2.5:
            raise ValueError('Ordered forward-turn plateau/fade required')
        self.register_buffer('legacy_forward_turn_extension',d.clone())

    def legacy_extended_speed_fade(self, observation, original, plateau):
        if float(plateau)<1.5:
            return original
        speed=observation[...,7]/.5
        yaw=observation[...,2]/.25+observation[...,8]
        result=original
        if hasattr(self,'legacy_reverse_gate_extension'):
            d=self.legacy_reverse_gate_extension
            t=((-speed-d[0])/(d[1]-d[0])).clamp(0.,1.)
            extended=1.-t*t*(3.-2.*t)
            result=torch.where((speed < -1.5)&(yaw.abs()<=1e-4),extended,result)
        if hasattr(self,'legacy_forward_turn_extension'):
            d=self.legacy_forward_turn_extension
            smooth=lambda t:t.clamp(0,1).square()*(3-2*t.clamp(0,1))
            extended=1.-smooth((speed-d[0])/(d[1]-d[0]))
            # Preserve straight, spins, reverse, low-speed and large-yaw paths.
            # This yaw band has original legacy yaw fade exactly1, so the
            # helper serves both the base gate and nested speed-only fade.
            blend=smooth((yaw.abs()-.02)/.03)*smooth((1.9-yaw.abs())/.1)
            blend=blend*((speed>1.5)&(observation[...,8].abs()<1.999)).to(speed.dtype)
            mixed=result+blend*(extended-result)
            result=torch.where(blend==0,result,torch.where(blend==1,extended,mixed))
        return result

    def update_gate(self,observation):
        gate=self.gate(observation)
        gate=self.legacy_extended_speed_fade(observation,gate,1.5)
        if hasattr(self,'update_domain'):
            speed=(observation[...,7]/.5).abs()
            yaw=(observation[...,2]/.25+observation[...,8]).abs()
            def fade(x,low,high):
                t=((x-low)/(high-low)).clamp(0.,1.)
                return 1.-t*t*(3.-2.*t)
            speed_gate=self.legacy_extended_speed_fade(observation,
                fade(speed,self.update_domain[0],self.update_domain[1]),self.update_domain[0])
            gate=gate*speed_gate*fade(yaw,self.update_domain[2],self.update_domain[3])
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
            gate = self.joint_delay_gate(observation)[..., None]
            output = torch.tanh(self.joint_delay_network(observation))
            if hasattr(self, 'joint_delay_reference_network'):
                reference = torch.tanh(self.joint_delay_reference_network(observation))
                if hasattr(self, 'joint_turn_update_anchor'):
                    anchor = torch.tanh(self.joint_turn_update_anchor(observation))
                    update_gate = self.joint_turn_update_gate(observation)[..., None]
                    mix = anchor + update_gate * (reference - anchor)
                    reference = torch.where(update_gate == 0, anchor,
                                            torch.where(update_gate == 1, reference, mix))
                if hasattr(self, 'joint_turn_source_network'):
                    frozen = self.joint_turn_source_output(observation)
                    if hasattr(self, 'joint_turn_source_anchor'):
                        anchor = torch.tanh(self.joint_turn_source_anchor(observation))
                        if hasattr(self,'joint_turn_anchor_frozen'):
                            preserved=torch.tanh(self.joint_turn_anchor_frozen(observation))
                            anchor_gate=self.joint_turn_anchor_gate(observation)[...,None]
                            mixed=preserved+anchor_gate*(anchor-preserved)
                            anchor=torch.where(anchor_gate==0,preserved,
                                               torch.where(anchor_gate==1,anchor,mixed))
                        source_gate = self.joint_turn_source_update_gate(observation)[..., None]
                        blended = anchor + source_gate * (frozen - anchor)
                        frozen = torch.where(source_gate == 0, anchor,
                                             torch.where(source_gate == 1, frozen, blended))
                    turn_gate = self.joint_turn_gate(observation)[..., None]
                    mix = frozen + turn_gate * (reference-frozen)
                    reference = torch.where(turn_gate == 0, frozen,
                                            torch.where(turn_gate == 1, reference, mix))
                blended = reference + gate * (output - reference)
                # Preserve exact endpoint outputs, including rounding behavior.
                output = torch.where(gate == 0, reference,
                                     torch.where(gate == 1, output, blended))
                original = original + self.joint_delay_budget * output
            else:
                original = original + gate * self.joint_delay_budget * output
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
