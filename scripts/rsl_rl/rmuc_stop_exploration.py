"""Opt-in observation-conditioned diagonal Gaussian for moving-to-stop exploration.

Changes training sampling and its probability model together. No inference mean,
checkpoint tensors, command, actuator authority, or physical feedback changes.
"""
import math
import torch
from torch.distributions import Normal


def moving_stop_gate(obs):
    if obs.shape[-1] != 25:
        raise ValueError('Requires25D observation')
    command = obs[..., 7] / .5
    yaw = obs[..., 2] / .25 + obs[..., 8]
    speed = command - obs[..., 6] / 2.1
    t = ((speed.abs() - .03) / .05).clamp(0., 1.)
    return t.square() * (3. - 2. * t) * (
        (command.abs() <= 1e-6) & (yaw.abs() <= 1e-4)
        & (obs[..., 8].abs() < 1.99)).to(obs.dtype)


class StopExploration:
    def __init__(self, actor, wheel_std):
        from rsl_rl.modules import GaussianDistribution
        dist = actor.distribution
        if type(dist) is not GaussianDistribution:
            raise ValueError('Requires ordinary diagonal GaussianDistribution')
        if any(p.requires_grad for p in dist.parameters()):
            raise ValueError('Base exploration std must be frozen')
        if not math.isfinite(wheel_std) or not .03 <= wheel_std <= .08:
            raise ValueError('Wheel exploration std must be in [.03,.08]')
        if not hasattr(actor, 'rmuc_physics_adaptation') or not hasattr(actor.rmuc_physics_adaptation, 'stop_network'):
            raise ValueError('Requires installed stop residual')
        if hasattr(dist, '_rmuc_stop_exploration'):
            raise ValueError('Already installed')
        self.wheel_std = float(wheel_std)
        self.pending = None
        self.calls = 0
        self.counts = None
        self.std_min = None
        self.std_max = None
        self.original_update = dist.update
        self.hook = actor.rmuc_physics_adaptation.register_forward_pre_hook(self.capture)
        dist._rmuc_stop_exploration = self
        dist.update = self.update
        self.dist = dist

    def capture(self, module, inputs):
        self.pending = inputs[0].detach()

    def update(self, mean):
        obs, self.pending = self.pending, None
        if obs is None or obs.shape[:-1] != mean.shape[:-1]:
            raise RuntimeError('Missing matching current actor observation for Gaussian update')
        self.original_update(mean)
        base = self.dist.std
        gate = moving_stop_gate(obs)
        std = base.clone()
        for i in (2, 5):
            std[..., i] = base[..., i] + gate * (self.wheel_std - base[..., i]).clamp_min(0)
        self.dist._distribution = Normal(mean, std)
        self.calls += 1
        with torch.no_grad():
            counts = torch.stack(((gate > 0).sum(), torch.ones_like(gate, dtype=torch.long).sum()))
            lo = std.reshape(-1, 6).amin(0); hi = std.reshape(-1, 6).amax(0)
            if self.counts is None:
                self.counts = counts; self.std_min = lo; self.std_max = hi
            else:
                # Rollout uses inference_mode; PPO updates do not. Never
                # mutate a persistent inference tensor outside that mode.
                self.counts = self.counts + counts
                self.std_min = torch.minimum(self.std_min, lo)
                self.std_max = torch.maximum(self.std_max, hi)

    def readback(self):
        return dict(enabled=True, wheel_std=self.wheel_std, calls=self.calls,
                    samples=None if self.counts is None else self.counts.cpu().tolist(),
                    observed_std_min=None if self.std_min is None else self.std_min.cpu().tolist(),
                    observed_std_max=None if self.std_max is None else self.std_max.cpu().tolist(),
                    scope='Calls include rollout and PPO minibatch reuse; counts are not independent environment steps',
                    gate='zero vx/yaw command, estimated abs(vx) smooth .03 to .08m/s; wheel slots2,5 only',
                    deterministic_mean_unchanged=True)


def install_stop_exploration(actor, wheel_std=.05):
    return StopExploration(actor, wheel_std)
