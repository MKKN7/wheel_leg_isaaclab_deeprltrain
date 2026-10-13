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

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from agent_rl.rsl_rl.modules import ActorCriticDreamWaq
from agent_rl.rsl_rl.storage import RolloutStorageDreamWaq


class PPODreamWaq:
    policy: ActorCriticDreamWaq

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        vae_learning_rate=None,
        num_adaptation_module_substeps=1,
        kl_weight=1.0,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.policy = policy
        self.policy.to(self.device)
        self.storage: RolloutStorageDreamWaq = None
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        if vae_learning_rate is None:
            vae_learning_rate = learning_rate
        self.vae_optimizer = optim.Adam(self.policy.cenet_parameters(), lr=vae_learning_rate)
        self.transition = RolloutStorageDreamWaq.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.vae_learning_rate = vae_learning_rate
        self.num_adaptation_module_substeps = int(num_adaptation_module_substeps)
        self.kl_weight = kl_weight
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        self.adaboot_reward_window_size = int(kwargs.pop("adaboot_reward_window_size", 1024))
        self.adaboot_reward_cv_scale = float(kwargs.pop("adaboot_reward_cv_scale", 0.5))
        self.adaboot_reward_cv_offset = float(kwargs.pop("adaboot_reward_cv_offset", 0.0))
        self.adaboot_pboot_min = float(kwargs.pop("adaboot_pboot_min", 0.0))
        self.adaboot_pboot_max = float(kwargs.pop("adaboot_pboot_max", 1.0))
        self._ep_partial_returns: torch.Tensor | None = None
        self._ep_return_window: deque[float] = deque(maxlen=self.adaboot_reward_window_size)

    def init_storage(self, num_envs, num_transitions_per_env, observations_shape, action_shape):
        self.storage = RolloutStorageDreamWaq(num_envs, num_transitions_per_env, observations_shape, action_shape, self.device)
        self._ep_partial_returns = torch.zeros(num_envs, device=self.device)
        self._ep_return_window.clear()

    def test_mode(self):
        self.policy.test()

    def train_mode(self):
        self.policy.train()

    def act(self, obs_dict):
        self.transition.actions = self.policy.act(obs_dict).detach()
        self.transition.values = self.policy.evaluate(obs_dict).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs_dict
        return self.transition.actions

    def process_env_step(self, rewards, dones, extras, next_obs_dict=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_observations = next_obs_dict
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs_dict):
        last_values = self.policy.evaluate(obs_dict).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def _update_episode_reward_stats(self):
        if self._ep_partial_returns is None:
            self._ep_partial_returns = torch.zeros(self.storage.num_envs, device=self.device)

        rewards = self.storage.rewards.squeeze(-1)
        dones = self.storage.dones.squeeze(-1).bool()
        running = self._ep_partial_returns
        for t in range(rewards.shape[0]):
            running = running + rewards[t]
            done_ids = torch.nonzero(dones[t], as_tuple=False).flatten()
            if done_ids.numel() > 0:
                for value in running[done_ids]:
                    self._ep_return_window.append(float(value.item()))
                running[done_ids] = 0.0
        self._ep_partial_returns = running

    def _compute_p_boot_from_episode_rewards(self):
        self._update_episode_reward_stats()
        if len(self._ep_return_window) < 2:
            r_episode = self.storage.rewards.squeeze(-1).sum(dim=0)
        else:
            r_episode = torch.tensor(list(self._ep_return_window), device=self.device, dtype=torch.float32)
        r_mean = torch.mean(r_episode)
        r_std = torch.std(r_episode, unbiased=False)
        cv_r = r_std / (torch.abs(r_mean) + 1e-8)
        shaped_cv = self.adaboot_reward_cv_scale * cv_r + self.adaboot_reward_cv_offset
        p_boot = 1.0 - torch.tanh(shaped_cv)
        p_boot = torch.clamp(p_boot, self.adaboot_pboot_min, self.adaboot_pboot_max)
        return p_boot, r_mean, r_std, cv_r

    def _unpack_batch(self, batch):
        return batch
        # if len(batch) == 13:
        #     return batch
        # (
        #     observation_batch,
        #     actions_batch,
        #     target_values_batch,
        #     advantages_batch,
        #     returns_batch,
        #     rewards_batch,
        #     old_actions_log_prob_batch,
        #     old_mu_batch,
        #     old_sigma_batch,
        #     hid_states_batch,
        #     masks_batch,
        # ) = batch
        # dones_batch = torch.zeros_like(rewards_batch, dtype=torch.bool)
        # return (
        #     observation_batch,
        #     actions_batch,
        #     target_values_batch,
        #     advantages_batch,
        #     returns_batch,
        #     rewards_batch,
        #     old_actions_log_prob_batch,
        #     old_mu_batch,
        #     old_sigma_batch,
        #     hid_states_batch,
        #     masks_batch,
        #     observation_batch,
        #     dones_batch,
        # )

    def _velocity_target(self, observation_batch):
        critic_obs = observation_batch.get("critic")
        if critic_obs is None:
            critic_obs = observation_batch.get("prev_critic")
        if critic_obs is None:
            return torch.zeros(
                observation_batch["policy"].shape[0],
                self.policy.num_estimate,
                dtype=torch.float,
                device=self.device,
            )
        start = self.policy.num_actor_obs
        end = start + self.policy.num_estimate
        return critic_obs[:, start:end].detach()

    def update(self, beta=1):
        mean_rl_loss_dict = {
            "value_function": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "recon": 0.0,
            "est": 0.0,
            "kld": 0.0,
            "adaboot_coef": 0.0,
            "adaboot_r_mean": 0.0,
            "adaboot_r_std": 0.0,
            "adaboot_cv_r": 0.0,
        }

        with torch.no_grad():
            p_boot, r_mean, r_std, cv_r = self._compute_p_boot_from_episode_rewards()

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for raw_batch in generator:
            (
                observation_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                _rewards_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                next_observation_batch,
                dones_batch,
            ) = self._unpack_batch(raw_batch)

            self.policy.set_adaboot_p_boot(p_boot)
            self.policy.act(observation_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(observation_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy
            adaboot_alpha = self.policy.get_adaboot_p_boot()
            if adaboot_alpha is not None:
                adaboot_coef = adaboot_alpha.mean()
            else:
                adaboot_coef = p_boot if isinstance(p_boot, torch.Tensor) else torch.tensor(float(p_boot), device=self.device)

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            if self.normalize_advantage_per_mini_batch:
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            advantages = advantages_batch.squeeze(-1)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            valid = (dones_batch == 0).squeeze(-1).bool()
            if not torch.any(valid):
                valid = torch.ones_like(valid, dtype=torch.bool)
            vel_target = self._velocity_target(observation_batch)
            decode_target = next_observation_batch.get("policy", observation_batch["policy"]).detach()

            batch_recon = 0.0
            batch_est = 0.0
            batch_kld = 0.0
            substeps = max(self.num_adaptation_module_substeps, 1)
            for _ in range(substeps):
                self.vae_optimizer.zero_grad()
                _, code_vel, _, decode, _, _, mean_latent, logvar_latent = self.policy.cenet_forward(
                    observation_batch, sample=True
                )
                recons_loss = nn.functional.mse_loss(decode, decode_target, reduction="none").mean(dim=-1)
                vel_loss = nn.functional.mse_loss(code_vel, vel_target, reduction="none").mean(dim=-1)
                kld_loss = -0.5 * torch.sum(
                    1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp(),
                    dim=-1,
                )
                vae_loss = (recons_loss[valid] + vel_loss[valid] + self.kl_weight * kld_loss[valid]).mean()
                vae_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.cenet_parameters(), self.max_grad_norm)
                self.vae_optimizer.step()
                batch_recon += recons_loss[valid].mean().item()
                batch_est += vel_loss[valid].mean().item()
                batch_kld += kld_loss[valid].mean().item()

            mean_rl_loss_dict["value_function"] += value_loss.item()
            mean_rl_loss_dict["surrogate"] += surrogate_loss.item()
            mean_rl_loss_dict["entropy"] += entropy_batch.mean().item()
            mean_rl_loss_dict["recon"] += batch_recon / substeps
            mean_rl_loss_dict["est"] += batch_est / substeps
            mean_rl_loss_dict["kld"] += batch_kld / substeps
            mean_rl_loss_dict["adaboot_coef"] += adaboot_coef.item()
            mean_rl_loss_dict["adaboot_r_mean"] += r_mean.item()
            mean_rl_loss_dict["adaboot_r_std"] += r_std.item()
            mean_rl_loss_dict["adaboot_cv_r"] += cv_r.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for key in mean_rl_loss_dict:
            mean_rl_loss_dict[key] /= num_updates

        self.storage.clear()
        return {**mean_rl_loss_dict}
