# =============================================================================
# This file contains code derived from the following third-party projects.
#
#   [rsl_rl / legged_gym]
#     License  : BSD-3-Clause
#     Copyright: Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
#
#   [HIMLoco]
#     URL      : https://github.com/InternRobotics/HIMLoco
#     License  : CC BY-NC-SA 4.0
#     Copyright: Copyright (c) 2024 Junfeng Long, Zirui Wang
#
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic

from agent_rl.rsl_rl.modules.policy import *  # noqa: F401
from agent_rl.rsl_rl.modules.nn import Identity, MLP
from agent_rl.rsl_rl.modules.him_estimator import HIMEstimator

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, device):  # shape:the dimension of input data
        self.n = 1e-4
        self.uninitialized = True
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)

    def update(self, x):
        count = self.n
        batch_count = x.size(0)
        tot_count = count + batch_count

        old_mean = self.mean.clone()
        delta = torch.mean(x, dim=0) - old_mean

        self.mean = old_mean + delta * batch_count / tot_count
        m_a = self.var * count
        m_b = x.var(dim=0) * batch_count
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
        self.var = M2 / tot_count
        self.n = tot_count

class Normalization:
    def __init__(self, shape, device='cuda:0'):
        self.running_ms = RunningMeanStd(shape=shape, device=device)

    def __call__(self, x, update=False):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (torch.sqrt(self.running_ms.var) + 1e-4)

        return x

class ActorCriticHIM(nn.Module):
    is_recurrent = False
    def __init__(
            self,  
            num_actor_obs,
            num_critic_obs,
            num_estimate,
            num_actor_obs_hist,
            num_actions,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation='elu',
            init_noise_std=1.0,
            estimator_feature_clip: float = 50.0,
            action_mean_clip: float = 20.0,
            **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticHIM, self).__init__()

        self.history_size = num_actor_obs_hist
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.num_estimate = num_estimate
        self.estimator_feature_clip = estimator_feature_clip
        self.action_mean_clip = action_mean_clip

        mlp_input_dim_a = num_actor_obs + self.num_estimate + 16
        mlp_input_dim_c = num_critic_obs

        # Estimator
        self.estimator = HIMEstimator(
            temporal_steps=self.history_size, 
            num_one_step_obs=num_actor_obs,
            num_estimate=self.num_estimate,
        )

        # Policy
        self.actor = MLP(
            input_dim=mlp_input_dim_a,
            output_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation
        )

        # Value function
        self.critic = MLP(
            input_dim=mlp_input_dim_c,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=activation
        )

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f'Estimator: {self.estimator.encoder}')

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def _sanitize_tensor(self, tensor, clip: float | None = None):
        if clip is None:
            return torch.nan_to_num(tensor, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=clip, neginf=-clip)
        return torch.clamp(tensor, -clip, clip)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs_history):
        obs_history = self._sanitize_tensor(obs_history)
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
        vel = self._sanitize_tensor(vel, self.estimator_feature_clip)
        latent = self._sanitize_tensor(latent, self.estimator_feature_clip)
        actor_input = self._sanitize_tensor(torch.cat((obs_history[:,-self.num_actor_obs:], vel, latent), dim=-1))
        
        mean = self._sanitize_tensor(self.actor(actor_input), self.action_mean_clip)
        std = torch.nan_to_num(self.std.expand_as(mean), nan=1.0, posinf=10.0, neginf=1.0e-6).clamp_min(1.0e-6)
        self.distribution = Normal(mean, std)

    def act(self, obs_dict, **kwargs):
        self.update_distribution(obs_dict['policy_hist'])
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict):
        policy_hist = self._sanitize_tensor(obs_dict['policy_hist'])
        vel, latent = self.estimator(policy_hist)
        vel = self._sanitize_tensor(vel, self.estimator_feature_clip)
        latent = self._sanitize_tensor(latent, self.estimator_feature_clip)
        actor_input = self._sanitize_tensor(torch.cat((policy_hist[:,-self.num_actor_obs:], vel, latent), dim=-1))
        actions_mean = self._sanitize_tensor(self.actor(actor_input), self.action_mean_clip)
        return actions_mean
    
    def test_inference(self, obs_dict):
        policy_hist = self._sanitize_tensor(obs_dict['policy_hist'])
        vel, latent = self.estimator(policy_hist)
        vel = self._sanitize_tensor(vel, self.estimator_feature_clip)
        latent = self._sanitize_tensor(latent, self.estimator_feature_clip)
        actor_input = self._sanitize_tensor(torch.cat((policy_hist[:,-self.num_actor_obs:], vel, latent), dim=-1))
        actions_mean = self._sanitize_tensor(self.actor(actor_input), self.action_mean_clip)
        estimator_output = torch.cat((vel, latent), dim=-1)
        return actions_mean, estimator_output

    def evaluate(self, obs_dict, **kwargs):
        value = self.critic(self._sanitize_tensor(obs_dict['critic']))
        return value
