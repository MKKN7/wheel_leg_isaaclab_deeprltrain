# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic

from agent_rl.rsl_rl.modules.policy import *  # noqa: F401


class ActorCriticExt(ActorCritic):
    def __init__(
            self,
            num_observations,
            num_actions,
            init_noise_std=1.0,
            noise_std_type: str = "scalar",
            **kwargs,
    ):
        # if kwargs:
        #     print(
        #         "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
        #         + str([key for key in kwargs.keys()])
        #     )
        nn.Module.__init__(self)
        self.num_observations = num_observations
        self.num_actions = num_actions

        self.actor: Actor = eval(kwargs.pop("actor_name"))(
            num_observations=num_observations,
            num_actions=num_actions,
            **kwargs
        )

        # Value function
        self.critic: Critic = eval(kwargs.pop("critic_name"))(
            num_observations=num_observations,
            num_values=1,
            **kwargs
        )

        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update_distribution(self, obs_dict):
        # compute mean
        mean = self.actor(obs_dict)["actions"]
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs_dict, **kwargs):
        self.update_distribution(obs_dict)
        return self.distribution.sample()

    def act_inference(self, obs_dict):
        actions_mean = self.actor(obs_dict)["actions"]
        return actions_mean

    def evaluate(self, obs_dict, **kwargs):
        return self.critic(obs_dict)["value"]
