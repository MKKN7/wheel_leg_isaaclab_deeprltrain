# =============================================================================
# This file contains code derived from the following third-party projects.
#
#   [rsl_rl / legged_gym]
#     License  : BSD-3-Clause
#     Copyright: Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
#
#   [DreamWaQ]
#     URL      : https://github.com/Manaro-Alpha/DreamWaQ
#
# =============================================================================

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic

from agent_rl.rsl_rl.modules.nn import MLP


class ActorCriticDreamWaq(ActorCritic):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_estimate,
        cenet_in_dim,
        cenet_out_dim,
        activation="elu",
        cenet_encoder_hidden_dims=(128, 64),
        cenet_decoder_hidden_dims=(64, 128),
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        cenet_logvar_clip: float | None = 10.0,
        cenet_feature_clip: float | None = 50.0,
        action_mean_clip: float | None = 20.0,
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.num_estimate = num_estimate
        self.num_latent = int(cenet_out_dim) - int(num_estimate)
        if self.num_latent <= 0:
            raise ValueError(f"cenet_out_dim ({cenet_out_dim}) must be larger than num_estimate ({num_estimate})")
        self.cenet_logvar_clip = None if cenet_logvar_clip is None else float(cenet_logvar_clip)
        self.cenet_feature_clip = None if cenet_feature_clip is None else float(cenet_feature_clip)
        self.action_mean_clip = None if action_mean_clip is None else float(action_mean_clip)

        actor_hidden_dims = kwargs.pop("actor_hidden_dims", [512, 256, 128])
        critic_hidden_dims = kwargs.pop("critic_hidden_dims", [512, 256, 128])

        self.adaboot_mode = str(kwargs.pop("adaboot_mode", "off")).lower()
        if bool(kwargs.pop("use_adaboot", False)) and self.adaboot_mode == "off":
            self.adaboot_mode = "uncertainty"
        self.adaboot_temperature = float(kwargs.pop("adaboot_temperature", 1.0))
        self.adaboot_bias = float(kwargs.pop("adaboot_bias", 0.0))
        self.adaboot_min = float(kwargs.pop("adaboot_min", 0.0))
        self.adaboot_max = float(kwargs.pop("adaboot_max", 1.0))

        self.actor = MLP(
            input_dim=num_actor_obs + cenet_out_dim,
            output_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
        )
        self.critic = MLP(
            input_dim=num_critic_obs,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )
        self.encoder = MLP(
            input_dim=cenet_in_dim,
            hidden_dims=cenet_encoder_hidden_dims,
            activation=activation,
        )
        self.encode_mean_latent = nn.Linear(cenet_encoder_hidden_dims[-1], self.num_latent)
        self.encode_logvar_latent = nn.Linear(cenet_encoder_hidden_dims[-1], self.num_latent)
        self.encode_mean_vel = nn.Linear(cenet_encoder_hidden_dims[-1], self.num_estimate)
        self.encode_logvar_vel = nn.Linear(cenet_encoder_hidden_dims[-1], self.num_estimate)
        self.decoder = MLP(
            input_dim=cenet_out_dim,
            output_dim=num_actor_obs,
            hidden_dims=cenet_decoder_hidden_dims,
            activation=activation,
        )

        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        self.last_code_vel: torch.Tensor | None = None
        self.last_adaboot_alpha: torch.Tensor | None = None
        self.adaboot_p_boot: torch.Tensor | float | None = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def cenet_parameters(self):
        return itertools.chain(
            self.encoder.parameters(),
            self.encode_mean_latent.parameters(),
            self.encode_logvar_latent.parameters(),
            self.encode_mean_vel.parameters(),
            self.encode_logvar_vel.parameters(),
            self.decoder.parameters(),
        )

    def _sanitize_tensor(self, tensor, clip: float | None = None):
        if clip is None:
            return torch.nan_to_num(tensor, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=clip, neginf=-clip)
        return torch.clamp(tensor, -clip, clip)

    def reparameterise(self, mean, logvar, sample: bool = True):
        mean = self._sanitize_tensor(mean, self.cenet_feature_clip)
        logvar = self._sanitize_tensor(logvar, self.cenet_logvar_clip)
        if not sample:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def set_adaboot_p_boot(self, p_boot: torch.Tensor | float | None):
        self.adaboot_p_boot = p_boot

    def get_adaboot_p_boot(self):
        return self.last_adaboot_alpha

    def _adaboot_velocity(self, code_vel, mean_vel, logvar_vel, gt_vel=None):
        if self.adaboot_mode == "off":
            self.last_adaboot_alpha = None
            return code_vel

        device = code_vel.device
        dtype = code_vel.dtype
        alpha_unc = None
        alpha_boot = None

        if self.adaboot_mode in ("uncertainty", "hybrid"):
            uncertainty = torch.mean(logvar_vel, dim=-1, keepdim=True)
            alpha_unc = torch.sigmoid(-(self.adaboot_temperature * uncertainty + self.adaboot_bias))

        if self.adaboot_mode in ("reward_cv", "hybrid"):
            p_boot = self.adaboot_p_boot
            if p_boot is None:
                alpha_boot = torch.ones((code_vel.shape[0], 1), device=device, dtype=dtype)
            elif isinstance(p_boot, torch.Tensor):
                if p_boot.dim() == 0:
                    alpha_boot = p_boot.to(device=device, dtype=dtype).view(1, 1).expand(code_vel.shape[0], 1)
                elif p_boot.dim() == 1:
                    alpha_boot = p_boot.to(device=device, dtype=dtype).view(-1, 1)
                else:
                    alpha_boot = p_boot.to(device=device, dtype=dtype)
            else:
                alpha_boot = torch.full((code_vel.shape[0], 1), float(p_boot), device=device, dtype=dtype)

        if self.adaboot_mode == "uncertainty":
            alpha = alpha_unc
        elif self.adaboot_mode == "reward_cv":
            alpha = alpha_boot
        elif self.adaboot_mode == "hybrid":
            alpha = alpha_unc * alpha_boot
        else:
            raise ValueError(f"Unknown adaboot_mode: {self.adaboot_mode}")

        alpha = torch.clamp(alpha, min=self.adaboot_min, max=self.adaboot_max)
        teacher_vel = gt_vel if gt_vel is not None else mean_vel
        self.last_adaboot_alpha = alpha.detach()
        return alpha * code_vel + (1.0 - alpha) * teacher_vel

    def _teacher_velocity_from_obs(self, obs_dict):
        critic_obs = obs_dict.get("critic")
        if critic_obs is None:
            critic_obs = obs_dict.get("prev_critic")
        if critic_obs is None:
            return None
        start = self.num_actor_obs
        end = start + self.num_estimate
        if critic_obs.shape[-1] < end:
            return None
        return critic_obs[:, start:end]

    def cenet_forward(self, obs_dict, sample: bool = True):
        policy_hist = self._sanitize_tensor(obs_dict["policy_hist"])
        distribution = self.encoder(policy_hist)
        mean_latent = self._sanitize_tensor(self.encode_mean_latent(distribution), self.cenet_feature_clip)
        logvar_latent = self._sanitize_tensor(self.encode_logvar_latent(distribution), self.cenet_logvar_clip)
        mean_vel = self._sanitize_tensor(self.encode_mean_vel(distribution), self.cenet_feature_clip)
        logvar_vel = self._sanitize_tensor(self.encode_logvar_vel(distribution), self.cenet_logvar_clip)
        code_latent = self.reparameterise(mean_latent, logvar_latent, sample=sample)
        code_vel = self.reparameterise(mean_vel, logvar_vel, sample=sample)
        code = torch.cat((code_vel, code_latent), dim=-1)
        decode = self.decoder(self._sanitize_tensor(code, self.cenet_feature_clip))
        return code, code_vel, code_latent, decode, mean_vel, logvar_vel, mean_latent, logvar_latent

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        observations = self._sanitize_tensor(observations)
        mean = self._sanitize_tensor(self.actor(observations), self.action_mean_clip)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        std = torch.nan_to_num(std, nan=1.0, posinf=10.0, neginf=1.0e-6).clamp_min(1.0e-6)
        self.distribution = Normal(mean, std)

    def act(self, obs_dict, **kwargs):
        _, code_vel, code_latent, _, mean_vel, logvar_vel, _, _ = self.cenet_forward(obs_dict, sample=True)
        self.last_code_vel = code_vel.detach()
        gt_vel = self._teacher_velocity_from_obs(obs_dict)
        code_vel = self._adaboot_velocity(code_vel, mean_vel, logvar_vel, gt_vel=gt_vel)
        actor_code = torch.cat((code_vel, code_latent), dim=-1)
        observations = torch.cat((self._sanitize_tensor(obs_dict["policy"]), actor_code), dim=-1)
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict):
        code, code_vel, _, _, _, _, _, _ = self.cenet_forward(obs_dict, sample=False)
        self.last_code_vel = code_vel.detach()
        observations = torch.cat((self._sanitize_tensor(obs_dict["policy"]), code), dim=-1)
        actions_mean = self._sanitize_tensor(self.actor(observations), self.action_mean_clip)
        return actions_mean

    def evaluate(self, obs_dict, **kwargs):
        return self.critic(self._sanitize_tensor(obs_dict["critic"]))
