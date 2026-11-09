"""Offline, opt-in critic prototype; never changes the deployed 25D actor.

No training entry point enables this module. The reward history must be captured
before action selection, and updated only after that transition's reward exists.
The stationary reward baseline is a hypothesis, not an exact value function.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn


VERSION = "causal_reward_history_v1"
TAUS = (0.25, 1.0, 2.0)


class CausalRewardHistory:
    def __init__(self, num_envs, dt=0.008, device="cpu"):
        if num_envs < 1 or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Positive environment count and policy period required")
        self.dt = float(dt)
        self.alpha = 1 - torch.exp(-dt / torch.tensor(TAUS, device=device))
        self.ema = torch.zeros(num_envs, 3, device=device)
        self.age = torch.zeros(num_envs, device=device)
        self.commands = torch.zeros(num_envs, 3, device=device)
        self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.pending_step = None
        self.completed_step = -1

    def before_step(self, commands, step):
        if commands.shape != self.commands.shape or not torch.isfinite(commands).all():
            raise ValueError("Finite physical commands with matching environment layout required")
        if step != self.completed_step + 1 or self.pending_step not in (None, step):
            raise ValueError("History must precede the next transition, with no skipped update")
        if self.pending_step == step:
            if not torch.equal(commands, self.commands):
                raise ValueError("Command changed between reads of the same transition")
        else:
            changed = ~self.valid | (commands != self.commands).any(-1)
            self.ema[changed] = 0
            self.age[changed] = 0
            self.commands.copy_(commands)
            self.valid.fill_(True)
            self.pending_step = step
        return torch.cat((self.ema.clone(), self.age[:, None].clone()), -1)

    def after_step(self, reward, done, step):
        if self.pending_step != step or step != self.completed_step + 1:
            raise ValueError("Reward history update missing its pre-step read, or duplicated")
        if reward.shape != self.age.shape or done.shape != self.age.shape:
            raise ValueError("Reward/done layout mismatch")
        if done.dtype != torch.bool or not torch.isfinite(reward).all():
            raise ValueError("Finite reward and boolean termination flags required")
        self.ema.add_((reward[:, None] - self.ema) * self.alpha)
        self.age.add_(self.dt)
        self.ema[done] = 0
        self.age[done] = 0
        self.valid[done] = False
        self.completed_step = step
        self.pending_step = None

    def state_dict(self):
        return dict(version=VERSION, dt=self.dt, pending_step=self.pending_step,
                    completed_step=self.completed_step,
                    **{k: getattr(self, k).clone() for k in ("ema", "age", "commands", "valid")})

    def load_state_dict(self, state):
        if state["version"] != VERSION or state["dt"] != self.dt:
            raise ValueError("Reward history contract mismatch")
        for key in ("ema", "age", "commands", "valid"):
            target, value = getattr(self, key), state[key]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError("Reward history tensor mismatch")
            target.copy_(value)
        self.pending_step = state["pending_step"]
        self.completed_step = state["completed_step"]


def make_mlp(dimension):
    return nn.Sequential(nn.Linear(dimension, 256), nn.ELU(), nn.Linear(256, 128),
                         nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))


class RewardHistoryCritic(nn.Module):
    """Frozen startup critic + causal stationary baseline + learned residual.

    Input: original 25D observation, three past-reward EMAs, stable-command age
    in seconds. Exact command changes clear history, conservatively disabling
    this baseline throughout a continuously varying command. That limitation
    must be validated before this prototype can enter online training.
    """
    def __init__(self, source_critic, mean, scale, output_scale, gamma=0.998):
        super().__init__()
        if mean.shape != (29,) or scale.shape != (29,) or not (scale > 0).all():
            raise ValueError("29D train-only affine statistics required")
        if not 0 < gamma < 1 or not math.isfinite(output_scale) or output_scale <= 0:
            raise ValueError("Valid discount and output scale required")
        self.frozen_base = copy.deepcopy(source_critic).requires_grad_(False)
        self.residual = make_mlp(29)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.register_buffer("mean", mean.clone())
        self.register_buffer("scale", scale.clone())
        self.register_buffer("output_scale", torch.tensor(float(output_scale)))
        self.register_buffer("gamma", torch.tensor(float(gamma), dtype=torch.float64))

    @staticmethod
    def gate(age):
        u = ((age - 1.0) / 2.0).clamp(0, 1)
        return u.square() * (3 - 2 * u)

    def forward(self, features):
        if features.shape[-1] != 29:
            raise ValueError("Reward-history critic needs 29D; actor remains 25D")
        g = self.gate(features[..., 28:29])
        with torch.no_grad():
            original = self.frozen_base(features[..., :25])
        baseline = features[..., 26:27] / (1 - self.gamma).to(features.dtype)
        correction = self.output_scale * self.residual((features - self.mean) / self.scale)
        return (1 - g) * original + g * (baseline + correction)

    def checkpoint(self):
        return dict(context_version=VERSION, critic_dim=29, actor_dim=25,
                    taus_s=list(TAUS), gate_age_s=[1., 3.], runtime_enabled=False,
                    gamma=float(self.gamma),
                    critic_state_dict=copy.deepcopy(self.state_dict()))

    def load_checkpoint(self, checkpoint):
        expected = self.checkpoint()
        for key in ("context_version", "critic_dim", "actor_dim", "taus_s", "gate_age_s", "gamma"):
            if checkpoint.get(key) != expected[key]:
                raise ValueError("Reward-history checkpoint contract mismatch")
        if float(checkpoint['critic_state_dict']['gamma']) != checkpoint['gamma']:
            raise ValueError("Reward-history discount buffer differs from its contract")
        self.load_state_dict(checkpoint["critic_state_dict"], strict=True)
