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

import os
import statistics
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.runners import OnPolicyRunner
from agent_rl.rsl_rl.compat import store_code_state

try:
    from agent_rl.rsl_rl.env import VecEnv
except ModuleNotFoundError:
    from rsl_rl.env import VecEnv
from agent_rl.rsl_rl.algorithms import PPOHIM
from agent_rl.rsl_rl.modules import ActorCriticHIM

class OnPolicyHIMRunner(OnPolicyRunner):
    """On-policy runner himloco for training and evaluation."""

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        object.__init__(self)

        self.cfg=train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()
        
        # Determine observation dimensions
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_observations['policy']
        
        # Initialize policy network
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # HIMActorCritic
        policy: ActorCriticHIM = actor_critic_class(
            self.env.num_observations['policy'],
            num_critic_obs,  # historical privileged obs
            self.env.cfg.n_state_est,
            self.env.cfg.num_obs_hist,
            self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)
        
        # Initialize algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))  # HIMPPO
        self.alg: PPOHIM = alg_class(policy, device=self.device, **self.alg_cfg)
        
        # Training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        self.experiment_name = self.cfg["experiment_name"]
        self.save_interval = self.cfg["save_interval"]
        self.save_best_after = self.cfg.get("save_best_after", self.save_interval)
        self.save_best_interval = self.cfg.get("save_best_interval", 0)
        self.save_best_epoch = 0

        # Initialize storage and model
        self.alg.init_storage(self.env.num_envs, 
                              self.num_steps_per_env, 
                              self.env.num_observations,
                              num_critic_obs,
                              self.env.num_actions
                              )
        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        # _, _ = self.env.reset() we call this in wrapper
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from agent_rl.rsl_rl.utils.wandb_utils import WandbSummaryWriterExt as WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")
        
        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        obs_dict, extras = self.env.reset()
        obs_dict = {k: v.to(self.device) for k, v in obs_dict.items()}
        self.alg.policy.train() # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        cur_score = best_score = -float('inf')
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs_dict)
                    obs_dict, rewards, dones, extras = self.env.step(actions)

                    # Move to device
                    obs_dict = {k: v.to(self.device) for k, v in obs_dict.items()}
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    termination_ids = extras['termination_ids'].to(self.device)
                    termination_privileged_obs = extras['termination_privileged_obs'].to(self.device)
                    if self.env.num_privileged_obs is not None:
                        critic_obs = obs_dict['critic']
                    else:
                        critic_obs = obs_dict['policy']
                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                    self.alg.process_env_step(rewards, dones, extras, next_critic_obs)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in extras:
                            ep_infos.append(extras['episode'])
                        elif 'log' in extras:
                            ep_infos.append(extras['log'])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_dict)
                
            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Logging info and save checkpoint
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if len(rewbuffer) > max(0.5 * rewbuffer.maxlen, 1):
                    cur_score = statistics.mean(rewbuffer)
                if it > self.save_best_after and it > self.save_best_epoch + self.save_best_interval and cur_score > best_score:
                    best_score = cur_score
                    self.save_best_epoch = it
                    self.save(os.path.join(self.log_dir, f"{self.experiment_name}.pt"),
                              infos={'score': best_score}, is_best=True)
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"), infos={'score': cur_score})

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"),
                      infos={'score': cur_score})

    def save(self, path, infos=None, is_best=False):
        torch.save({
            'model_state_dict': self.alg.policy.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'estimator_optimizer_state_dict': self.alg.policy.estimator.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)
        
        # Upload model to external logging service 
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs and is_best:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = False, load_iteration: bool = False):
        loaded_dict = torch.load(path, weights_only=False, map_location=self.device)
        # -- Load model
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"], strict=False)
        # -- Load optimizer if used
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "estimator_optimizer_state_dict" in loaded_dict:
                self.alg.policy.estimator.optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
        # -- Load current learning iteration
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.policy.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference
