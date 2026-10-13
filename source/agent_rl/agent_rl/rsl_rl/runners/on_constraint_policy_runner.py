# =============================================================================
# This file contains code derived from the following third-party projects.
#
#   [rsl_rl / legged_gym]
#     License  : BSD-3-Clause
#     Copyright: Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
#
#   [ddt_rl_isaacgym]
#     URL      : https://github.com/DDTRobot/ddt_rl_isaacgym
#
# =============================================================================

from __future__ import annotations

import time
import os
from collections import deque
import statistics
import warnings
import torch

import rsl_rl
from rsl_rl.runners import OnPolicyRunner
from agent_rl.rsl_rl.compat import store_code_state

# from modules import ActorCriticRMA,ActorCriticRmaTrans,ActorCriticSF,ActorCriticBarlowTwins,ActorCriticStateTransformer,ActorCriticTransBarlowTwins,ActorCriticMixedBarlowTwins,ActorCriticRnnBarlowTwins,ActorCriticVqvae
from agent_rl.rsl_rl.modules import ActorCriticBarlowTwins 
from agent_rl.rsl_rl.algorithms import NP3O
try:
    from agent_rl.rsl_rl.env import VecEnv
except ModuleNotFoundError:
    from rsl_rl.env import VecEnv
from copy import copy, deepcopy

class OnConstraintPolicyRunner(OnPolicyRunner):

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        object.__init__(self)

        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # if self.env.num_privileged_obs is not None:
        #     num_critic_obs = self.env.num_privileged_obs
        # else:
            # num_critic_obs = self.env.num_observations['policy']
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        # if hasattr(self.env.num_observations, 'scan_dot'):
        #     n_scan = self.env.num_observations['scan_dot']
        # else:
        #     n_scan = 0
        policy: ActorCriticBarlowTwins = actor_critic_class(self.env.num_observations['policy'],
                                                      self.env.cfg.n_scan,
                                                      self.env.cfg.n_state_est,
                                                      self.env.num_observations['priv_latent'],
                                                      self.env.cfg.num_obs_hist,
                                                      self.env.num_actions,
                                                      **self.policy_cfg)
        print("Policy architecture: ",policy)
        # if self.cfg['resume']:
        #     log_root = os.path.join(ROOT_DIR, 'logs', self.cfg['experiment_name'], self.cfg['resume_path'])
        #     resume_path = get_load_path(log_root, load_run=self.cfg['load_run'], checkpoint=self.cfg['checkpoint'])
        #     print("Resume model from: ",resume_path)
        #     model_dict = torch.load(resume_path)
        #     policy.load_state_dict(model_dict['model_state_dict'])
        
        # policy.to(self.device)

        # Create algorithm
        self.alg_cfg['k_value'] = self.env.unwrapped.cost_k_values
        alg_class = eval(self.alg_cfg.pop("class_name")) # PPO
        self.alg = alg_class(policy, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]
        self.experiment_name = self.cfg["experiment_name"]
        self.save_best_after = self.cfg.get("save_best_after", self.save_interval)
        self.save_best_interval = self.cfg.get("save_best_interval", 0)
        self.save_best_epoch = 0

        self.alg.init_storage(
            self.env.num_envs, 
            self.num_steps_per_env, 
            [self.env.num_observations['on_constraint']], 
            [self.env.num_privileged_obs], 
            [self.env.num_actions],
            [self.env.cfg.num_costs],
            self.env.unwrapped.cost_d_values_tensor
        )

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        self.env.reset()

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
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        obs_dict, extras = self.env.reset()
        obs_dict = {k: v.to(self.device) for k, v in obs_dict.items()}
        infos = {}
        self.alg.policy.train() # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        cur_score = best_score = -float('inf')
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        # self.act_shed,self.imi_shed,self.lag_shed = hard_phase_schedualer(max_iters=tot_iter,
        #             phase1_end=self.phase1_end)

        # #imitation_mode
        # if self.alg.policy.imi_flag and self.cfg['resume']: 
        #     self.alg.policy.imitation_mode()
            
        for it in range(self.current_learning_iteration, tot_iter):
            # act_teacher_flag = self.act_shed[it]
            # imi_flag = self.imi_shed[it]
            # lag_flag = self.lag_shed[it]

            # self.alg.set_imi_flag(imi_flag)
            # self.alg.policy.set_teacher_act(act_teacher_flag)
            # # self.env.randomize_lag_timesteps = lag_flag
            # # if self.env.randomize_lag_timesteps:
            # #     print("lag is on")
            # # else:
            # #     print("lag is off")
            # if self.alg.policy.imi_flag and self.cfg['resume']: 
            #     step_size = 1/int(tot_iter/2)
            #     imi_weight = max(0,1 - it * step_size)
            #     print("imi_weight:",imi_weight)
            #     self.alg.set_imi_weight(imi_weight)
            
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs_dict['on_constraint'], obs_dict['on_constraint'])
                    obs_dict, rewards, dones, extras = self.env.step(actions.to(self.env.device))  # obs has changed to next_obs !! if done obs has been reset
                    obs_dict = {k: v.to(self.device) for k, v in obs_dict.items()}
                    rewards, costs, dones = rewards.to(self.device), extras['costs'].to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, costs, dones, extras)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in extras:
                            ep_infos.append(extras['episode'])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_dict['on_constraint'])
                self.alg.compute_cost_returns(obs_dict['on_constraint'])

            #update k value for better expolration
            k_value = self.alg.update_k_value(it)
            
            loss_dict, obs_batch_max, obs_batch_min = self.alg.update()

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
        state_dict = {
            'model_state_dict': self.alg.policy.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }
        torch.save(state_dict, path)
        
        # Upload model to external logging service 
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs and is_best:
            self.writer.save_model(path, self.current_learning_iteration)


    def load(self, path: str, load_optimizer: bool = False, load_iteration: bool = False):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, weights_only=False, map_location=self.device)
        self.alg.policy.load_state_dict(loaded_dict['model_state_dict'])
        # self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        print("*" * 80)
        return loaded_dict['infos']

    def log(self, locs, width=80, pad=35):
        """与 ``OnPolicyRunner.log`` 对齐的 TensorBoard 标量命名与累计步数；保留 NP3O 的 ``loss_dict`` / Data 标量。"""
        collection_size = self.num_steps_per_env * self.env.num_envs * getattr(self, "gpu_world_size", 1)
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info（与 OnPolicyRunner：含 ``/`` 的 key 直接作 tag，否则 ``Episode/<key>``）
        ep_string = ""
        if locs["ep_infos"]:
            all_keys: set[str] = set()
            for ep_info in locs["ep_infos"]:
                all_keys.update(ep_info.keys())
            for key in sorted(all_keys):
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    v = ep_info[key]
                    if not isinstance(v, torch.Tensor):
                        v = torch.tensor([float(v)], device=self.device, dtype=torch.float32)
                    else:
                        v = v.to(self.device)
                    if v.dim() == 0:
                        v = v.unsqueeze(0)
                    infotensor = torch.cat((infotensor, v))
                if infotensor.numel() == 0:
                    continue
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.get_std().mean()
        denom = locs["collection_time"] + locs["learn_time"]
        fps = int(collection_size / denom) if denom > 0 else 0

        # -- Losses（与 OnPolicyRunner 相同：``Loss/<key>``）
        loss_dict = locs.get("loss_dict") or {}
        for key, value in loss_dict.items():
            v = value.item() if isinstance(value, torch.Tensor) else float(value)
            self.writer.add_scalar(f"Loss/{key}", v, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- NP3O 额外：观测范围（不影响与 PPO 对齐的主要曲线）
        if "obs_batch_max" in locs and "obs_batch_min" in locs:
            om = max(locs["obs_batch_max"].item() if isinstance(locs["obs_batch_max"], torch.Tensor) else locs["obs_batch_max"], -float("inf"))
            on = min(locs["obs_batch_min"].item() if isinstance(locs["obs_batch_min"], torch.Tensor) else locs["obs_batch_min"], float("inf"))
            self.writer.add_scalar("Data/obs_max", om, locs["it"])
            self.writer.add_scalar("Data/obs_min", on, locs["it"])

        # -- Training（与 OnPolicyRunner：wandb 不打 time 轴标量）
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if getattr(self, "logger_type", "tensorboard") != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        tot_iter = locs.get("tot_iter")
        if tot_iter is None:
            tot_iter = locs.get("start_iter", 0) + locs.get("num_learning_iterations", 0)
        title = f" \033[1m Learning iteration {locs['it']}/{tot_iter} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in loss_dict.items():
                v = value.item() if isinstance(value, torch.Tensor) else float(value)
                log_string += f"""{f'Mean {key} loss:':>{pad}} {v:.4f}\n"""
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in loss_dict.items():
                v = value.item() if isinstance(value, torch.Tensor) else float(value)
                log_string += f"""{f'{key}:':>{pad}} {v:.4f}\n"""

        log_string += ep_string
        start_iter = locs.get("start_iter", 0)
        num_learn = locs.get("num_learning_iterations", 1)
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                '%H:%M:%S',
                time.gmtime(
                    self.tot_time / (locs['it'] - start_iter + 1) * (start_iter + num_learn - locs['it'])
                ),
            )}\n"""
        )
        print(log_string)

    def get_inference_policy(self, device=None):
        self.alg.policy.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_teacher
    
    def get_actor_critic(self, device=None):
        self.alg.policy.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy
    
