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

    def correction(self,observation):
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

    def forward(self,observation,source_action):
        return source_action+self.correction(observation)
