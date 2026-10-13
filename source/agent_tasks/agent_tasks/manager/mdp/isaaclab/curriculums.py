# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

"""自定义课程学习项."""

from __future__ import annotations

from collections import deque
from typing import Sequence

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_inverse

__all__ = [
    "CommandVelocityProgression",
    "ActuatorGainsProgression",
    "BaseVerticalAssistForceProgression",
    "JointFrictionScaleProgression",
    "RewardWeightProgression",
]


class CommandVelocityProgression(ManagerTermBase):
    """基于速度跟踪奖励的命令课程学习.

    该课程学习会在每个环境 episode 结束时统计 ``reward_key`` 对应的回合平均奖励，
    当最近 ``window_size`` 个 episode 的平均值达到设定阈值时，逐步放宽线速度命令范围，
    从而实现由低速到高速的课程学习。
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        params = dict(cfg.params) if cfg.params is not None else {}
        self.reward_key: str = params.get("reward_key", "track_lin_vel_xy")
        self.num_steps_per_env: int = int(params.get("num_steps_per_env", 24))
        self.window_size: int = max(1, int(params.get("window_size", 32)))
        self.min_stage_episodes: int = max(1, int(params.get("min_stage_episodes", self.window_size)))
        self.normalize_by_episode_length: bool = bool(params.get("normalize_by_episode_length", True))
        self.window_size = self.window_size * self.num_steps_per_env
        self.min_stage_episodes = self.min_stage_episodes * self.num_steps_per_env

        stages = params.get("stages")
        if not stages or not isinstance(stages, Sequence):
            raise ValueError("CommandVelocityProgression 需要在 params['stages'] 中提供至少一个阶段配置。")

        self._stage_configs: list[dict[str, tuple[float, float] | None]] = []
        thresholds: list[float] = []
        min_episodes_per_stage: list[int] = []

        for idx, stage_cfg in enumerate(stages):
            if not isinstance(stage_cfg, dict):
                raise ValueError("stages 中的每个元素必须是 dict，用于描述该阶段的命令范围。")

            stage_ranges: dict[str, tuple[float, float] | None] = {}
            for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"):
                if key in stage_cfg and stage_cfg[key] is not None:
                    range_pair = tuple(float(v) for v in stage_cfg[key])
                    if len(range_pair) != 2:
                        raise ValueError(f"阶段 {idx} 的 '{key}' 必须提供长度为 2 的区间。")
                    stage_ranges[key] = range_pair  # type: ignore[assignment]
                else:
                    stage_ranges[key] = None

            if stage_ranges["lin_vel_x"] is None:
                raise ValueError(f"阶段 {idx} 必须提供 'lin_vel_x' 范围。")

            self._stage_configs.append(stage_ranges)

            if idx < len(stages) - 1:
                if "threshold" not in stage_cfg:
                    raise ValueError("除最后一个阶段外，其他阶段必须提供 'threshold' 阈值。")
                thresholds.append(float(stage_cfg["threshold"]))
                min_ep = stage_cfg.get("min_episodes")*self.num_steps_per_env
                min_episodes_per_stage.append(
                    max(0, int(min_ep)) if min_ep is not None else self.min_stage_episodes
                )
            elif "threshold" in stage_cfg:
                thresholds.append(float(stage_cfg["threshold"]))

        if len(thresholds) < len(self._stage_configs) - 1:
            raise ValueError("阈值数量不足：应为阶段数量减一。")

        self._thresholds = thresholds[: len(self._stage_configs) - 1]
        self._min_episodes_per_stage = min_episodes_per_stage
        self._num_stages = len(self._stage_configs)

        initial_stage = int(params.get("initial_stage", 0))
        self._stage = max(0, min(initial_stage, self._num_stages - 1))

        self._recent_rewards: deque[float] = deque(maxlen=self.window_size)
        self._episodes_since_stage_change: int = 0
        self._total_episodes: int = 0
        self._last_window_mean: float = 0.0
        self._last_batch_mean: float = 0.0

        self._update_command_range()

    @property
    def stage(self) -> int:
        """当前课程阶段（从 0 开始计数）。"""
        return self._stage

    def __call__(
        self,
        env,
        env_ids: Sequence[int] | torch.Tensor | None,
        reward_key=None,
        window_size=None,
        min_stage_episodes=None,
        normalize_by_episode_length=None,
        num_steps_per_env=None,
        stages=None,
        initial_stage=None,
    ):
        episode_ids = self._normalize_env_ids(env_ids)
        if not episode_ids:
            return self._build_state_dict()

        self._accumulate_rewards(episode_ids)
        self._try_advance_stage()
        return self._build_state_dict()

    def reset(self, env_ids: Sequence[int] | None = None):
        # 课程学习当前无需额外 reset 操作，保留接口兼容性
        return None

    """
    Internal helpers.
    """

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> list[int]:
        if env_ids is None:
            return list(range(self.num_envs))
        if isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return []
            return env_ids.detach().cpu().tolist()
        return [int(idx) for idx in env_ids]

    def _accumulate_rewards(self, env_ids: list[int]):
        reward_buffer = getattr(self._env, "_episode_sums", {}).get(self.reward_key)
        if reward_buffer is None:
            return

        device_ids = torch.as_tensor(env_ids, device=reward_buffer.device, dtype=torch.long)
        if device_ids.numel() == 0:
            return

        rewards = reward_buffer[device_ids].clone()

        if self.normalize_by_episode_length:
            # 使用设定的最长 episode 长度归一化，与 Episode/Reward 一致
            max_ep_len_s = getattr(self._env, "max_episode_length_s", 20.0)
            rewards = rewards / max_ep_len_s

        rewards_cpu = rewards.detach().cpu().tolist()
        if not rewards_cpu:
            return

        self._last_batch_mean = float(sum(rewards_cpu) / len(rewards_cpu))

        self._recent_rewards.append(self._last_batch_mean)
        self._episodes_since_stage_change += 1
        self._total_episodes += 1

        if len(self._recent_rewards) > 0:
            self._last_window_mean = float(sum(self._recent_rewards) / len(self._recent_rewards))

    def _try_advance_stage(self):
        if self._stage >= self._num_stages - 1:
            return
        if len(self._recent_rewards) < self.window_size:
            return
        min_ep = (
            self._min_episodes_per_stage[self._stage]
            if self._stage < len(self._min_episodes_per_stage)
            else self.min_stage_episodes
        )
        if self._episodes_since_stage_change < min_ep:
            return

        window_values = list(self._recent_rewards)[-self.window_size :]
        window_mean = float(sum(window_values) / self.window_size)
        threshold = self._thresholds[self._stage]

        if window_mean >= threshold:
            self._stage = min(self._stage + 1, self._num_stages - 1)
            self._episodes_since_stage_change = 0
            self._recent_rewards.clear()
            self._last_window_mean = window_mean
            self._update_command_range()
            print(
                "[Curriculum] Command velocity stage -> "
                f"{self._stage + 1}/{self._num_stages}, "
                f"lin_vel_x={self._env.cfg.commands.ranges.lin_vel_x}, "
                f"avg_reward={window_mean:.4f} (threshold {threshold:.4f})"
            )

    def _update_command_range(self):
        stage_cfg = self._stage_configs[self._stage]
        ranges_cfg = self._env.cfg.commands.ranges
        generator_ranges = self._env.command_generator.cfg.ranges

        for key, value in stage_cfg.items():
            if value is None:
                continue
            if hasattr(ranges_cfg, key):
                setattr(ranges_cfg, key, tuple(value))
            if hasattr(generator_ranges, key):
                setattr(generator_ranges, key, tuple(value))

    def _build_state_dict(self) -> dict[str, float]:
        stage_cfg = self._stage_configs[self._stage]
        state = {
            "stage": float(self._stage),
            "stage_count": float(self._num_stages),
            "last_window_mean": float(self._last_window_mean),
            "last_batch_mean": float(self._last_batch_mean),
            "total_episodes": int(self._total_episodes/self.num_steps_per_env),
        }

        for key, value in stage_cfg.items():
            if value is None:
                continue
            state[f"{key}_min"] = float(value[0])
            state[f"{key}_max"] = float(value[1])

        if self._stage < len(self._thresholds):
            state["next_threshold"] = float(self._thresholds[self._stage])
        if self._stage < len(self._min_episodes_per_stage):
            state["min_episodes"] = int(self._min_episodes_per_stage[self._stage]/self.num_steps_per_env)

        return state


class ActuatorGainsProgression(ManagerTermBase):
    """基于奖励的 randomize_actuator_gains 课程学习.

    该课程学习会在每个环境 episode 结束时统计 ``reward_key`` 对应的回合平均奖励，
    当最近 ``window_size`` 个 episode 的平均值达到设定阈值时，逐步放宽 PD 增益随机化范围
    （stiffness_distribution_params、damping_distribution_params），从而实现由小扰动到大扰动的课程学习。

    使用方式：
    1. 在 env_cfg 的 events 中配置 randomize_actuator_gains 对应的 EventTerm（如 robot_joint_stiffness_and_damping）
    2. 在 curriculum 中配置 ActuatorGainsProgression，指定 event_keys 与 stages
    3. stages 中每个阶段需指定 stiffness_distribution_params、damping_distribution_params
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        params = dict(cfg.params) if cfg.params is not None else {}
        self.reward_key: str = params.get("reward_key", "track_lin_vel_xy")
        self.num_steps_per_env: int = int(params.get("num_steps_per_env", 24))
        self.window_size: int = max(1, int(params.get("window_size", 32)))
        self.min_stage_episodes: int = max(1, int(params.get("min_stage_episodes", self.window_size)))
        self.normalize_by_episode_length: bool = bool(params.get("normalize_by_episode_length", True))
        self.window_size = self.window_size * self.num_steps_per_env
        self.min_stage_episodes = self.min_stage_episodes * self.num_steps_per_env

        event_keys = params.get("event_keys")
        if not event_keys:
            raise ValueError("ActuatorGainsProgression 需要在 params['event_keys'] 中提供至少一个 EventTerm 名称。")
        if isinstance(event_keys, str):
            event_keys = [event_keys]
        self._event_keys: list[str] = list(event_keys)

        stages = params.get("stages")
        if not stages or not isinstance(stages, Sequence):
            raise ValueError("ActuatorGainsProgression 需要在 params['stages'] 中提供至少一个阶段配置。")

        self._stage_configs: list[dict] = []
        thresholds: list[float] = []
        min_episodes_per_stage: list[int] = []

        for idx, stage_cfg in enumerate(stages):
            if not isinstance(stage_cfg, dict):
                raise ValueError("stages 中的每个元素必须是 dict。")

            stiffness = stage_cfg.get("stiffness_distribution_params")
            damping = stage_cfg.get("damping_distribution_params")
            if stiffness is None or damping is None:
                raise ValueError(
                    f"阶段 {idx} 必须提供 'stiffness_distribution_params' 和 'damping_distribution_params'。"
                )
            stiffness = tuple(float(v) for v in stiffness)
            damping = tuple(float(v) for v in damping)
            if len(stiffness) != 2 or len(damping) != 2:
                raise ValueError(f"阶段 {idx} 的 stiffness/damping 必须为长度为 2 的区间 (min, max)。")

            self._stage_configs.append({
                "stiffness_distribution_params": stiffness,
                "damping_distribution_params": damping,
            })

            if idx < len(stages) - 1:
                if "threshold" not in stage_cfg:
                    raise ValueError("除最后一个阶段外，其他阶段必须提供 'threshold' 阈值。")
                thresholds.append(float(stage_cfg["threshold"]))
                min_ep = stage_cfg.get("min_episodes")
                min_ep = max(0, int(min_ep * self.num_steps_per_env)) if min_ep is not None else self.min_stage_episodes
                min_episodes_per_stage.append(min_ep)
            elif "threshold" in stage_cfg:
                thresholds.append(float(stage_cfg["threshold"]))

        self._thresholds = thresholds[: len(self._stage_configs) - 1]
        self._min_episodes_per_stage = min_episodes_per_stage
        self._num_stages = len(self._stage_configs)

        initial_stage = int(params.get("initial_stage", 0))
        self._stage = max(0, min(initial_stage, self._num_stages - 1))

        self._recent_rewards: deque[float] = deque(maxlen=self.window_size)
        self._episodes_since_stage_change: int = 0
        self._total_episodes: int = 0
        self._last_window_mean: float = 0.0
        self._last_batch_mean: float = 0.0

        self._update_actuator_gains_params()

    @property
    def stage(self) -> int:
        """当前课程阶段（从 0 开始计数）。"""
        return self._stage

    def __call__(
        self,
        env,
        env_ids: Sequence[int] | torch.Tensor | None,
        reward_key=None,
        window_size=None,
        min_stage_episodes=None,
        normalize_by_episode_length=None,
        num_steps_per_env=None,
        event_keys=None,
        stages=None,
    ):
        episode_ids = self._normalize_env_ids(env_ids)
        if not episode_ids:
            return self._build_state_dict()

        self._accumulate_rewards(episode_ids)
        self._try_advance_stage()
        return self._build_state_dict()

    def reset(self, env_ids: Sequence[int] | None = None):
        return None

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> list[int]:
        if env_ids is None:
            return list(range(self.num_envs))
        if isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return []
            return env_ids.detach().cpu().tolist()
        return [int(idx) for idx in env_ids]

    def _accumulate_rewards(self, env_ids: list[int]):
        reward_buffer = getattr(self._env, "_episode_sums", {}).get(self.reward_key)
        if reward_buffer is None:
            return

        device_ids = torch.as_tensor(env_ids, device=reward_buffer.device, dtype=torch.long)
        if device_ids.numel() == 0:
            return

        rewards = reward_buffer[device_ids].clone()

        if self.normalize_by_episode_length:
            max_ep_len_s = getattr(self._env, "max_episode_length_s", 20.0)
            rewards = rewards / max_ep_len_s

        rewards_cpu = rewards.detach().cpu().tolist()
        if not rewards_cpu:
            return

        self._last_batch_mean = float(sum(rewards_cpu) / len(rewards_cpu))
        self._recent_rewards.append(self._last_batch_mean)
        self._episodes_since_stage_change += 1
        self._total_episodes += 1

        if len(self._recent_rewards) > 0:
            self._last_window_mean = float(sum(self._recent_rewards) / len(self._recent_rewards))

    def _try_advance_stage(self):
        if self._stage >= self._num_stages - 1:
            return
        if len(self._recent_rewards) < self.window_size:
            return
        min_ep = (
            self._min_episodes_per_stage[self._stage]
            if self._stage < len(self._min_episodes_per_stage)
            else self.min_stage_episodes
        )
        if self._episodes_since_stage_change < min_ep:
            return

        window_values = list(self._recent_rewards)[-self.window_size:]
        window_mean = float(sum(window_values) / self.window_size)
        threshold = self._thresholds[self._stage]

        if window_mean >= threshold:
            self._stage = min(self._stage + 1, self._num_stages - 1)
            self._episodes_since_stage_change = 0
            self._recent_rewards.clear()
            self._last_window_mean = window_mean
            self._update_actuator_gains_params()
            stage_cfg = self._stage_configs[self._stage]
            print(
                "[Curriculum] Actuator gains stage -> "
                f"{self._stage + 1}/{self._num_stages}, "
                f"stiffness={stage_cfg['stiffness_distribution_params']}, "
                f"damping={stage_cfg['damping_distribution_params']}, "
                f"avg_reward={window_mean:.4f} (threshold {threshold:.4f})"
            )

    def _update_actuator_gains_params(self):
        """将当前阶段的 PD 增益范围写入对应 EventTerm 的 params。"""
        stage_cfg = self._stage_configs[self._stage]
        stiffness = stage_cfg["stiffness_distribution_params"]
        damping = stage_cfg["damping_distribution_params"]

        events_cfg = getattr(self._env.cfg, "events", None)
        if events_cfg is None:
            return

        for event_key in self._event_keys:
            event_term = getattr(events_cfg, event_key, None)
            if event_term is None:
                continue
            params = getattr(event_term, "params", None)
            if params is None or not isinstance(params, dict):
                continue
            params["stiffness_distribution_params"] = tuple(stiffness)
            params["damping_distribution_params"] = tuple(damping)

    def _build_state_dict(self) -> dict[str, float]:
        stage_cfg = self._stage_configs[self._stage]
        state = {
            "stage": float(self._stage),
            "stage_count": float(self._num_stages),
            "last_window_mean": float(self._last_window_mean),
            "last_batch_mean": float(self._last_batch_mean),
            "total_episodes": int(self._total_episodes / self.num_steps_per_env),
            "stiffness_min": float(stage_cfg["stiffness_distribution_params"][0]),
            "stiffness_max": float(stage_cfg["stiffness_distribution_params"][1]),
            "damping_min": float(stage_cfg["damping_distribution_params"][0]),
            "damping_max": float(stage_cfg["damping_distribution_params"][1]),
        }
        if self._stage < len(self._thresholds):
            state["next_threshold"] = float(self._thresholds[self._stage])
        if self._stage < len(self._min_episodes_per_stage):
            state["min_episodes"] = int(self._min_episodes_per_stage[self._stage] / self.num_steps_per_env)
        return state


class BaseVerticalAssistForceProgression(ManagerTermBase):
    """Curriculum that applies a world-frame upward force to a body.

    The force is intended as a temporary aid for height tracking. As the
    selected reward improves, the curriculum advances through configured
    stages and typically reduces the force until it reaches zero.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        params = dict(cfg.params) if cfg.params is not None else {}
        self.reward_key: str = params.get("reward_key", "track_height_exp")
        self.num_steps_per_env: int = int(params.get("num_steps_per_env", 24))
        self.window_size: int = max(1, int(params.get("window_size", 32)))
        self.min_stage_episodes: int = max(1, int(params.get("min_stage_episodes", self.window_size)))
        self.normalize_by_episode_length: bool = bool(params.get("normalize_by_episode_length", True))
        self.apply_on_compute: bool = bool(params.get("apply_on_compute", True))
        self.window_size = self.window_size * self.num_steps_per_env
        self.min_stage_episodes = self.min_stage_episodes * self.num_steps_per_env

        self.asset_cfg = params.get("asset_cfg", None)
        if self.asset_cfg is None:
            raise ValueError("BaseVerticalAssistForceProgression 需要在 params['asset_cfg'] 中提供目标 body。")

        stages = params.get("stages")
        if not stages or not isinstance(stages, Sequence):
            raise ValueError("BaseVerticalAssistForceProgression 需要在 params['stages'] 中提供至少一个阶段配置。")

        self._stage_forces: list[float] = []
        thresholds: list[float] = []
        min_episodes_per_stage: list[int] = []
        for idx, stage_cfg in enumerate(stages):
            if not isinstance(stage_cfg, dict):
                raise ValueError("stages 中的每个元素必须是 dict。")
            force_z = float(stage_cfg.get("force_z", 0.0))
            if force_z < 0.0:
                raise ValueError(f"阶段 {idx} 的 'force_z' 不能小于 0。")
            self._stage_forces.append(force_z)

            if idx < len(stages) - 1:
                if "threshold" not in stage_cfg:
                    raise ValueError("除最后一个阶段外，其他阶段必须提供 'threshold' 阈值。")
                thresholds.append(float(stage_cfg["threshold"]))
                min_ep = stage_cfg.get("min_episodes")
                min_ep = max(0, int(min_ep * self.num_steps_per_env)) if min_ep is not None else self.min_stage_episodes
                min_episodes_per_stage.append(min_ep)
            elif "threshold" in stage_cfg:
                thresholds.append(float(stage_cfg["threshold"]))

        self._thresholds = thresholds[: len(self._stage_forces) - 1]
        self._min_episodes_per_stage = min_episodes_per_stage
        self._num_stages = len(self._stage_forces)

        initial_stage = int(params.get("initial_stage", 0))
        self._stage = max(0, min(initial_stage, self._num_stages - 1))

        self._recent_rewards: deque[float] = deque(maxlen=self.window_size)
        self._episodes_since_stage_change: int = 0
        self._total_episodes: int = 0
        self._last_window_mean: float = 0.0
        self._last_batch_mean: float = 0.0

        self._apply_assist_force(None)

    @property
    def stage(self) -> int:
        return self._stage

    def __call__(
        self,
        env,
        env_ids: Sequence[int] | torch.Tensor | None,
        reward_key=None,
        window_size=None,
        min_stage_episodes=None,
        normalize_by_episode_length=None,
        num_steps_per_env=None,
        apply_on_compute=None,
        asset_cfg=None,
        stages=None,
        initial_stage=None,
    ):
        episode_ids = self._normalize_env_ids(env_ids)
        if not episode_ids:
            return self._build_state_dict()

        self._accumulate_rewards(episode_ids)
        advanced = self._try_advance_stage()
        if self.apply_on_compute:
            self._apply_assist_force(None if advanced else episode_ids)
        return self._build_state_dict()

    def reset(self, env_ids: Sequence[int] | None = None):
        self._apply_assist_force(env_ids)
        return None

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> list[int]:
        if env_ids is None:
            return list(range(self.num_envs))
        if isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return []
            return env_ids.detach().cpu().tolist()
        return [int(idx) for idx in env_ids]

    def _accumulate_rewards(self, env_ids: list[int]):
        reward_buffer = getattr(self._env, "_episode_sums", {}).get(self.reward_key)
        if reward_buffer is None:
            return

        device_ids = torch.as_tensor(env_ids, device=reward_buffer.device, dtype=torch.long)
        if device_ids.numel() == 0:
            return

        rewards = reward_buffer[device_ids].clone()
        if self.normalize_by_episode_length:
            max_ep_len_s = getattr(self._env, "max_episode_length_s", 20.0)
            rewards = rewards / max_ep_len_s

        rewards_cpu = rewards.detach().cpu().tolist()
        if not rewards_cpu:
            return

        self._last_batch_mean = float(sum(rewards_cpu) / len(rewards_cpu))
        self._recent_rewards.append(self._last_batch_mean)
        self._episodes_since_stage_change += 1
        self._total_episodes += 1

        if len(self._recent_rewards) > 0:
            self._last_window_mean = float(sum(self._recent_rewards) / len(self._recent_rewards))

    def _try_advance_stage(self) -> bool:
        if self._stage >= self._num_stages - 1:
            return False
        if len(self._recent_rewards) < self.window_size:
            return False
        min_ep = (
            self._min_episodes_per_stage[self._stage]
            if self._stage < len(self._min_episodes_per_stage)
            else self.min_stage_episodes
        )
        if self._episodes_since_stage_change < min_ep:
            return False

        window_values = list(self._recent_rewards)[-self.window_size:]
        window_mean = float(sum(window_values) / self.window_size)
        threshold = self._thresholds[self._stage]

        if window_mean < threshold:
            return False

        self._stage = min(self._stage + 1, self._num_stages - 1)
        self._episodes_since_stage_change = 0
        self._recent_rewards.clear()
        self._last_window_mean = window_mean
        print(
            "[Curriculum] Base vertical assist force stage -> "
            f"{self._stage + 1}/{self._num_stages}, "
            f"force_z={self._stage_forces[self._stage]:.3f} N, "
            f"avg_reward={window_mean:.4f} (threshold {threshold:.4f})"
        )
        return True

    def _apply_assist_force(self, env_ids: Sequence[int] | torch.Tensor | None):
        asset = self._env.scene[self.asset_cfg.name]
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=asset.device, dtype=torch.long)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids.to(device=asset.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        body_ids = self._resolve_body_ids(asset)
        num_bodies = int(body_ids.numel())
        force_z = self._stage_forces[self._stage]
        forces = torch.zeros((env_ids_t.numel(), num_bodies, 3), dtype=torch.float, device=asset.device)
        torques = torch.zeros_like(forces)
        if force_z == 0.0:
            self._clear_assist_force(asset, env_ids_t, body_ids)
            return
        force_w = torch.zeros_like(forces)
        force_w[..., 2] = force_z
        body_quat_w = asset.data.body_quat_w[env_ids_t[:, None], body_ids]
        forces[:] = quat_apply_inverse(body_quat_w.reshape(-1, 4), force_w.reshape(-1, 3)).reshape_as(forces)
        asset.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            env_ids=env_ids_t,
            body_ids=body_ids,
        )

    def _resolve_body_ids(self, asset) -> torch.Tensor:
        body_ids = self.asset_cfg.body_ids
        if body_ids is None:
            body_ids = torch.arange(asset.num_bodies, device=asset.device, dtype=torch.long)
        elif isinstance(body_ids, slice):
            body_ids = torch.arange(asset.num_bodies, device=asset.device, dtype=torch.long)[body_ids]
        elif isinstance(body_ids, torch.Tensor):
            body_ids = body_ids.to(device=asset.device, dtype=torch.long)
        else:
            body_ids = torch.as_tensor(body_ids, device=asset.device, dtype=torch.long)
        if body_ids.numel() == 0:
            raise ValueError("BaseVerticalAssistForceProgression 的 asset_cfg 没有解析到任何 body。")
        return body_ids

    def _clear_assist_force(self, asset, env_ids: torch.Tensor, body_ids: torch.Tensor):
        indices = body_ids.repeat(len(env_ids), 1) + env_ids.unsqueeze(1) * asset.num_bodies
        indices = indices.view(-1)
        asset._external_force_b.flatten(0, 1)[indices] = 0.0
        asset._external_torque_b.flatten(0, 1)[indices] = 0.0
        asset.has_external_wrench = bool(asset._external_force_b.any() or asset._external_torque_b.any())

    def _build_state_dict(self) -> dict[str, float]:
        state = {
            "stage": float(self._stage),
            "stage_count": float(self._num_stages),
            "last_window_mean": float(self._last_window_mean),
            "last_batch_mean": float(self._last_batch_mean),
            "total_episodes": int(self._total_episodes / self.num_steps_per_env),
            "force_z": float(self._stage_forces[self._stage]),
        }
        if self._stage < len(self._thresholds):
            state["next_threshold"] = float(self._thresholds[self._stage])
        if self._stage < len(self._min_episodes_per_stage):
            state["min_episodes"] = int(self._min_episodes_per_stage[self._stage] / self.num_steps_per_env)
        return state


class JointFrictionScaleProgression(ManagerTermBase):
    """Scale joint friction randomization ranges according to reward progress.

    This curriculum records the original joint-friction ranges from selected
    EventTerms and writes scaled ranges back into those EventTerm params as
    stages advance. It supports the same reward threshold and min_episodes
    gating style as RewardWeightProgression.
    """

    _FRICTION_PARAM_KEYS = (
        "static_friction_distribution_params",
        "dynamic_friction_distribution_params",
        "viscous_friction_distribution_params",
    )
    _SHORT_KEYS = {
        "static": "static_friction_distribution_params",
        "dynamic": "dynamic_friction_distribution_params",
        "viscous": "viscous_friction_distribution_params",
    }

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        params = dict(cfg.params) if cfg.params is not None else {}
        self.reward_key: str = params.get("reward_key", "track_height_exp")
        self.num_steps_per_env: int = int(params.get("num_steps_per_env", 24))
        self.window_size: int = max(1, int(params.get("window_size", 32)))
        self.min_stage_episodes: int = max(1, int(params.get("min_stage_episodes", self.window_size)))
        self.normalize_by_episode_length: bool = bool(params.get("normalize_by_episode_length", True))
        self.apply_on_compute: bool = bool(params.get("apply_on_compute", True))
        self.window_size = self.window_size * self.num_steps_per_env
        self.min_stage_episodes = self.min_stage_episodes * self.num_steps_per_env

        event_keys = params.get("event_keys")
        if not event_keys:
            raise ValueError("JointFrictionScaleProgression 需要在 params['event_keys'] 中提供至少一个 EventTerm 名称。")
        if isinstance(event_keys, str):
            event_keys = [event_keys]
        self._event_keys: tuple[str, ...] = tuple(str(key) for key in event_keys)

        stages = params.get("stages")
        if not stages or not isinstance(stages, Sequence):
            raise ValueError("JointFrictionScaleProgression 需要在 params['stages'] 中提供至少一个阶段配置。")

        self._stage_configs: list[dict] = []
        thresholds: list[float] = []
        min_episodes_per_stage: list[int] = []
        for idx, stage_cfg in enumerate(stages):
            if not isinstance(stage_cfg, dict):
                raise ValueError("stages 中的每个元素必须是 dict。")
            self._stage_configs.append(self._parse_stage_cfg(stage_cfg, idx))

            if idx < len(stages) - 1:
                if "threshold" not in stage_cfg:
                    raise ValueError("除最后一个阶段外，其他阶段必须提供 'threshold' 阈值。")
                thresholds.append(float(stage_cfg["threshold"]))
                min_ep = stage_cfg.get("min_episodes")
                min_ep = max(0, int(min_ep * self.num_steps_per_env)) if min_ep is not None else self.min_stage_episodes
                min_episodes_per_stage.append(min_ep)
            elif "threshold" in stage_cfg:
                thresholds.append(float(stage_cfg["threshold"]))

        self._thresholds = thresholds[: len(self._stage_configs) - 1]
        self._min_episodes_per_stage = min_episodes_per_stage
        self._num_stages = len(self._stage_configs)

        initial_stage = int(params.get("initial_stage", 0))
        self._stage = max(0, min(initial_stage, self._num_stages - 1))

        self._recent_rewards: deque[float] = deque(maxlen=self.window_size)
        self._episodes_since_stage_change: int = 0
        self._total_episodes: int = 0
        self._last_window_mean: float = 0.0
        self._last_batch_mean: float = 0.0

        self._default_ranges = self._capture_default_ranges()
        self._update_joint_friction_params()

    @property
    def stage(self) -> int:
        return self._stage

    def __call__(
        self,
        env,
        env_ids: Sequence[int] | torch.Tensor | None,
        reward_key=None,
        window_size=None,
        min_stage_episodes=None,
        normalize_by_episode_length=None,
        num_steps_per_env=None,
        apply_on_compute=None,
        event_keys=None,
        stages=None,
        initial_stage=None,
    ):
        episode_ids = self._normalize_env_ids(env_ids)
        if not episode_ids:
            return self._build_state_dict()

        self._accumulate_rewards(episode_ids)
        self._try_advance_stage()
        if self.apply_on_compute:
            self._apply_joint_friction_events(episode_ids)
        return self._build_state_dict()

    def reset(self, env_ids: Sequence[int] | None = None):
        return None

    def _parse_stage_cfg(self, stage_cfg: dict, idx: int) -> dict:
        stage: dict = {
            "scale": float(stage_cfg.get("scale", 1.0)),
            "event_scales": {},
        }
        if stage["scale"] < 0.0:
            raise ValueError(f"阶段 {idx} 的 'scale' 不能小于 0。")

        event_scales = stage_cfg.get("event_scales", {})
        if event_scales is None:
            event_scales = {}
        if not isinstance(event_scales, dict):
            raise ValueError(f"阶段 {idx} 的 'event_scales' 必须是 dict。")

        parsed_event_scales: dict[str, dict[str, float]] = {}
        for event_key, scale_cfg in event_scales.items():
            if isinstance(scale_cfg, (int, float)):
                parsed_event_scales[str(event_key)] = {"scale": max(0.0, float(scale_cfg))}
                continue
            if not isinstance(scale_cfg, dict):
                raise ValueError(f"阶段 {idx} 的 event_scales[{event_key!r}] 必须是数字或 dict。")

            parsed: dict[str, float] = {}
            for key, value in scale_cfg.items():
                canonical_key = self._SHORT_KEYS.get(str(key), str(key))
                if canonical_key == "scale":
                    parsed["scale"] = max(0.0, float(value))
                    continue
                if canonical_key not in self._FRICTION_PARAM_KEYS:
                    raise ValueError(
                        f"阶段 {idx} 的 event_scales[{event_key!r}] 包含未知摩擦参数 {key!r}。"
                    )
                parsed[canonical_key] = max(0.0, float(value))
            parsed_event_scales[str(event_key)] = parsed

        stage["event_scales"] = parsed_event_scales
        return stage

    def _capture_default_ranges(self) -> dict[str, dict[str, tuple[float, float]]]:
        defaults: dict[str, dict[str, tuple[float, float]]] = {}
        events_cfg = getattr(self._env.cfg, "events", None)
        if events_cfg is None:
            return defaults

        for event_key in self._event_keys:
            event_term = getattr(events_cfg, event_key, None)
            params = getattr(event_term, "params", None)
            if event_term is None or params is None or not isinstance(params, dict):
                continue
            defaults[event_key] = {}
            for friction_key in self._FRICTION_PARAM_KEYS:
                value = params.get(friction_key)
                if value is None:
                    continue
                pair = tuple(float(v) for v in value)
                if len(pair) != 2:
                    raise ValueError(f"{event_key}.{friction_key} 必须是长度为 2 的区间。")
                defaults[event_key][friction_key] = pair
        return defaults

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> list[int]:
        if env_ids is None:
            return list(range(self.num_envs))
        if isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return []
            return env_ids.detach().cpu().tolist()
        return [int(idx) for idx in env_ids]

    def _accumulate_rewards(self, env_ids: list[int]):
        reward_buffer = getattr(self._env, "_episode_sums", {}).get(self.reward_key)
        if reward_buffer is None:
            return

        device_ids = torch.as_tensor(env_ids, device=reward_buffer.device, dtype=torch.long)
        if device_ids.numel() == 0:
            return

        rewards = reward_buffer[device_ids].clone()
        if self.normalize_by_episode_length:
            max_ep_len_s = getattr(self._env, "max_episode_length_s", 20.0)
            rewards = rewards / max_ep_len_s

        rewards_cpu = rewards.detach().cpu().tolist()
        if not rewards_cpu:
            return

        self._last_batch_mean = float(sum(rewards_cpu) / len(rewards_cpu))
        self._recent_rewards.append(self._last_batch_mean)
        self._episodes_since_stage_change += 1
        self._total_episodes += 1

        if len(self._recent_rewards) > 0:
            self._last_window_mean = float(sum(self._recent_rewards) / len(self._recent_rewards))

    def _try_advance_stage(self):
        if self._stage >= self._num_stages - 1:
            return
        if len(self._recent_rewards) < self.window_size:
            return
        min_ep = (
            self._min_episodes_per_stage[self._stage]
            if self._stage < len(self._min_episodes_per_stage)
            else self.min_stage_episodes
        )
        if self._episodes_since_stage_change < min_ep:
            return

        window_values = list(self._recent_rewards)[-self.window_size:]
        window_mean = float(sum(window_values) / self.window_size)
        threshold = self._thresholds[self._stage]

        if window_mean >= threshold:
            self._stage = min(self._stage + 1, self._num_stages - 1)
            self._episodes_since_stage_change = 0
            self._recent_rewards.clear()
            self._last_window_mean = window_mean
            self._update_joint_friction_params()
            print(
                "[Curriculum] Joint friction scale stage -> "
                f"{self._stage + 1}/{self._num_stages}, "
                f"scale={self._stage_configs[self._stage]['scale']}, "
                f"avg_reward={window_mean:.4f} (threshold {threshold:.4f})"
            )

    def _scale_for(self, stage_cfg: dict, event_key: str, friction_key: str) -> float:
        event_scale_cfg = stage_cfg["event_scales"].get(event_key, {})
        return float(
            event_scale_cfg.get(
                friction_key,
                event_scale_cfg.get("scale", stage_cfg["scale"]),
            )
        )

    def _update_joint_friction_params(self):
        events_cfg = getattr(self._env.cfg, "events", None)
        if events_cfg is None:
            return
        stage_cfg = self._stage_configs[self._stage]

        for event_key, event_defaults in self._default_ranges.items():
            event_term = getattr(events_cfg, event_key, None)
            params = getattr(event_term, "params", None)
            if event_term is None or params is None or not isinstance(params, dict):
                continue
            for friction_key, default_range in event_defaults.items():
                scale = self._scale_for(stage_cfg, event_key, friction_key)
                params[friction_key] = (
                    float(default_range[0]) * scale,
                    float(default_range[1]) * scale,
                )

    def _apply_joint_friction_events(self, env_ids: Sequence[int] | torch.Tensor | None):
        event_manager = getattr(self._env, "event_manager", None)
        if event_manager is None:
            return
        if env_ids is not None and not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for event_key in self._event_keys:
            try:
                term_cfg = event_manager.get_term_cfg(event_key)
            except Exception:
                continue
            params = getattr(term_cfg, "params", None)
            if params is None or not isinstance(params, dict):
                continue
            term_cfg.func(self._env, env_ids, **params)

    def _build_state_dict(self) -> dict[str, float]:
        stage_cfg = self._stage_configs[self._stage]
        state = {
            "stage": float(self._stage),
            "stage_count": float(self._num_stages),
            "last_window_mean": float(self._last_window_mean),
            "last_batch_mean": float(self._last_batch_mean),
            "total_episodes": int(self._total_episodes / self.num_steps_per_env),
            "scale": float(stage_cfg["scale"]),
        }
        for event_key, event_defaults in self._default_ranges.items():
            for friction_key in event_defaults:
                short_key = friction_key.replace("_friction_distribution_params", "")
                state[f"{event_key}_{short_key}_scale"] = self._scale_for(stage_cfg, event_key, friction_key)
        if self._stage < len(self._thresholds):
            state["next_threshold"] = float(self._thresholds[self._stage])
        if self._stage < len(self._min_episodes_per_stage):
            state["min_episodes"] = int(self._min_episodes_per_stage[self._stage] / self.num_steps_per_env)
        return state


class RewardWeightProgression(ManagerTermBase):
    """基于 track_height_exp 奖励的 reward 权重课程学习.

    早期给 track_height 较大权重、lin_vel_z 较小惩罚，鼓励机器人优先学会站立；
    随着高度跟踪奖励达到阈值，逐步将各项权重恢复到最终目标值。

    使用方式：
    1. 在 stages 中按阶段列出 reward_weights dict（key 为 cfg.rewards 中的奖励名）
    2. 除最后阶段外每个阶段需提供 threshold 和可选 min_episodes
    3. 可选设置 restore_defaults_on_last_stage_threshold=True，并给最后阶段也提供
       threshold/min_episodes；达到后会恢复到 cfg.rewards 的初始默认值
    4. 可选提供 reward_scale dict，对应奖励会被设置为初始默认权重 * scale
    5. 在 CurriculumCfg 中添加对应 CurrTerm，并启用 curriculum
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        params = dict(cfg.params) if cfg.params is not None else {}
        self.reward_key: str = params.get("reward_key", "track_height_exp")
        self.num_steps_per_env: int = int(params.get("num_steps_per_env", 24))
        self.window_size: int = max(1, int(params.get("window_size", 32)))
        self.min_stage_episodes: int = max(1, int(params.get("min_stage_episodes", self.window_size)))
        self.normalize_by_episode_length: bool = bool(params.get("normalize_by_episode_length", True))
        self.restore_defaults_on_last_stage_threshold: bool = bool(
            params.get("restore_defaults_on_last_stage_threshold", False)
        )
        self.window_size = self.window_size * self.num_steps_per_env
        self.min_stage_episodes = self.min_stage_episodes * self.num_steps_per_env

        stages = params.get("stages")
        if not stages or not isinstance(stages, Sequence):
            raise ValueError("RewardWeightProgression 需要在 params['stages'] 中提供至少一个阶段配置。")

        self._stage_configs: list[dict] = []
        thresholds: list[float] = []
        min_episodes_per_stage: list[int] = []
        managed_reward_keys: list[str] = []
        managed_scale_keys: list[str] = []
        final_threshold: float | None = None
        final_min_episodes: int | None = None

        for idx, stage_cfg in enumerate(stages):
            if not isinstance(stage_cfg, dict):
                raise ValueError("stages 中的每个元素必须是 dict。")

            reward_weights = stage_cfg.get("reward_weights", {})
            reward_scale = stage_cfg.get("reward_scale", {})
            if reward_weights is None:
                reward_weights = {}
            if reward_scale is None:
                reward_scale = {}
            if not isinstance(reward_weights, dict):
                raise ValueError(f"阶段 {idx} 的 'reward_weights' 必须是 dict。")
            if not isinstance(reward_scale, dict):
                raise ValueError(f"阶段 {idx} 的 'reward_scale' 必须是 dict。")
            if not reward_weights and not reward_scale:
                raise ValueError(f"阶段 {idx} 必须提供非空的 'reward_weights' 或 'reward_scale' 字典。")

            self._stage_configs.append(
                {
                    "reward_weights": dict(reward_weights),
                    "reward_scale": dict(reward_scale),
                }
            )
            for key in tuple(reward_weights.keys()) + tuple(reward_scale.keys()):
                if key not in managed_reward_keys:
                    managed_reward_keys.append(key)
            for key in reward_scale.keys():
                if key not in managed_scale_keys:
                    managed_scale_keys.append(key)

            if idx < len(stages) - 1:
                if "threshold" not in stage_cfg:
                    raise ValueError("除最后一个阶段外，其他阶段必须提供 'threshold' 阈值。")
                thresholds.append(float(stage_cfg["threshold"]))
                min_ep = stage_cfg.get("min_episodes")
                min_ep = max(0, int(min_ep * self.num_steps_per_env)) if min_ep is not None else self.min_stage_episodes
                min_episodes_per_stage.append(min_ep)
            elif "threshold" in stage_cfg:
                final_threshold = float(stage_cfg["threshold"])
                min_ep = stage_cfg.get("min_episodes")
                final_min_episodes = (
                    max(0, int(min_ep * self.num_steps_per_env))
                    if min_ep is not None
                    else self.min_stage_episodes
                )

        self._thresholds = thresholds
        self._min_episodes_per_stage = min_episodes_per_stage
        self._num_stages = len(self._stage_configs)
        self._managed_reward_keys = tuple(managed_reward_keys)
        self._managed_scale_keys = tuple(managed_scale_keys)
        self._final_threshold = final_threshold
        self._final_min_episodes = final_min_episodes

        initial_stage = int(params.get("initial_stage", 0))
        self._stage = max(0, min(initial_stage, self._num_stages - 1))

        self._recent_rewards: deque[float] = deque(maxlen=self.window_size)
        self._episodes_since_stage_change: int = 0
        self._total_episodes: int = 0
        self._last_window_mean: float = 0.0
        self._last_batch_mean: float = 0.0
        self._defaults_restored: bool = False

        self._default_reward_weights: dict[str, float] = {}
        self._reward_cfg_names = (
            "rewards",
            "mood_mt_normal_rewards",
            "mood_mt_jump_base_rewards",
            "mood_mt_recover_rewards",
        )
        self._default_reward_weights_by_cfg: dict[str, dict[str, float]] = {}
        for cfg_name in self._reward_cfg_names:
            rewards_cfg = getattr(self._env.cfg, cfg_name, None)
            if rewards_cfg is None:
                continue
            defaults: dict[str, float] = {}
            for key in self._managed_reward_keys:
                if key in rewards_cfg:
                    defaults[key] = float(rewards_cfg[key])
            if defaults:
                self._default_reward_weights_by_cfg[cfg_name] = defaults
        self._default_reward_weights = dict(self._default_reward_weights_by_cfg.get("rewards", {}))

        if self.restore_defaults_on_last_stage_threshold and self._final_threshold is None:
            raise ValueError(
                "restore_defaults_on_last_stage_threshold=True 时，最后一个阶段必须提供 'threshold'。"
            )

        self._update_reward_weights()

    @property
    def stage(self) -> int:
        """当前课程阶段（从 0 开始计数）。"""
        return self._stage

    def __call__(
        self,
        env,
        env_ids: Sequence[int] | torch.Tensor | None,
        reward_key=None,
        window_size=None,
        min_stage_episodes=None,
        normalize_by_episode_length=None,
        num_steps_per_env=None,
        stages=None,
        initial_stage=None,
        restore_defaults_on_last_stage_threshold=None,
    ):
        episode_ids = self._normalize_env_ids(env_ids)
        if not episode_ids:
            return self._build_state_dict()

        self._accumulate_rewards(episode_ids)
        self._try_advance_stage()
        return self._build_state_dict()

    def reset(self, env_ids: Sequence[int] | None = None):
        return None

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> list[int]:
        if env_ids is None:
            return list(range(self.num_envs))
        if isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return []
            return env_ids.detach().cpu().tolist()
        return [int(idx) for idx in env_ids]

    def _accumulate_rewards(self, env_ids: list[int]):
        reward_buffer = getattr(self._env, "_episode_sums", {}).get(self.reward_key)
        if reward_buffer is None:
            return

        device_ids = torch.as_tensor(env_ids, device=reward_buffer.device, dtype=torch.long)
        if device_ids.numel() == 0:
            return

        rewards = reward_buffer[device_ids].clone()

        if self.normalize_by_episode_length:
            max_ep_len_s = getattr(self._env, "max_episode_length_s", 20.0)
            rewards = rewards / max_ep_len_s

        rewards_cpu = rewards.detach().cpu().tolist()
        if not rewards_cpu:
            return

        self._last_batch_mean = float(sum(rewards_cpu) / len(rewards_cpu))
        self._recent_rewards.append(self._last_batch_mean)
        self._episodes_since_stage_change += 1
        self._total_episodes += 1

        if len(self._recent_rewards) > 0:
            self._last_window_mean = float(sum(self._recent_rewards) / len(self._recent_rewards))

    def _try_advance_stage(self):
        if self._defaults_restored:
            return
        if len(self._recent_rewards) < self.window_size:
            return
        if self._stage >= self._num_stages - 1:
            self._try_restore_default_rewards()
            return
        min_ep = (
            self._min_episodes_per_stage[self._stage]
            if self._stage < len(self._min_episodes_per_stage)
            else self.min_stage_episodes
        )
        if self._episodes_since_stage_change < min_ep:
            return

        window_values = list(self._recent_rewards)[-self.window_size:]
        window_mean = float(sum(window_values) / self.window_size)
        threshold = self._thresholds[self._stage]

        if window_mean >= threshold:
            self._stage = min(self._stage + 1, self._num_stages - 1)
            self._episodes_since_stage_change = 0
            self._recent_rewards.clear()
            self._last_window_mean = window_mean
            self._update_reward_weights()
            stage_cfg = self._stage_configs[self._stage]
            print(
                "[Curriculum] RewardWeight stage -> "
                f"{self._stage + 1}/{self._num_stages}, "
                f"weights={stage_cfg['reward_weights']}, "
                f"scale={stage_cfg['reward_scale']}, "
                f"avg_reward={window_mean:.4f} (threshold {threshold:.4f})"
            )

    def _try_restore_default_rewards(self):
        if not self.restore_defaults_on_last_stage_threshold or self._final_threshold is None:
            return

        min_ep = (
            self._final_min_episodes
            if self._final_min_episodes is not None
            else self.min_stage_episodes
        )
        if self._episodes_since_stage_change < min_ep:
            return

        window_values = list(self._recent_rewards)[-self.window_size:]
        window_mean = float(sum(window_values) / self.window_size)
        if window_mean < self._final_threshold:
            return

        self._restore_default_reward_weights()
        self._defaults_restored = True
        self._recent_rewards.clear()
        self._last_window_mean = window_mean
        print(
            "[Curriculum] RewardWeight restored defaults, "
            f"avg_reward={window_mean:.4f} (threshold {self._final_threshold:.4f}), "
            f"weights={self._default_reward_weights}"
        )

    def _update_reward_weights(self):
        """将当前阶段的奖励权重写入 cfg.rewards。"""
        stage_cfg = self._stage_configs[self._stage]
        reward_weights = stage_cfg["reward_weights"]
        reward_scale = stage_cfg["reward_scale"]

        for cfg_name, default_weights in self._default_reward_weights_by_cfg.items():
            rewards_cfg = getattr(self._env.cfg, cfg_name, None)
            if rewards_cfg is None:
                continue
            for key, value in reward_weights.items():
                if key in rewards_cfg:
                    rewards_cfg[key] = float(value)
            for key, scale in reward_scale.items():
                if key in rewards_cfg and key in default_weights:
                    rewards_cfg[key] = float(default_weights[key]) * float(scale)

    def _restore_default_reward_weights(self):
        """Restore the managed reward keys to their initial cfg.rewards defaults."""
        for cfg_name, default_weights in self._default_reward_weights_by_cfg.items():
            rewards_cfg = getattr(self._env.cfg, cfg_name, None)
            if rewards_cfg is None:
                continue
            for key, value in default_weights.items():
                if key in rewards_cfg:
                    rewards_cfg[key] = float(value)

    def _build_state_dict(self) -> dict[str, float]:
        state = {
            "stage": float(self._stage),
            "stage_count": float(self._num_stages),
            "last_window_mean": float(self._last_window_mean),
            "last_batch_mean": float(self._last_batch_mean),
            "total_episodes": int(self._total_episodes / self.num_steps_per_env),
        }
        if self._defaults_restored:
            state["defaults_restored"] = 1.0
            rewards_cfg = getattr(self._env.cfg, "rewards", None)
            for key in self._managed_reward_keys:
                if rewards_cfg is not None and key in rewards_cfg:
                    state[f"weight_{key}"] = float(rewards_cfg[key])
            default_rewards = self._default_reward_weights_by_cfg.get("rewards", {})
            for key in self._managed_scale_keys:
                if rewards_cfg is None or key not in rewards_cfg or key not in default_rewards:
                    continue
                default_value = float(default_rewards[key])
                if abs(default_value) > 1.0e-12:
                    state[f"scale_{key}"] = float(rewards_cfg[key]) / default_value
                else:
                    state[f"scale_{key}"] = 1.0 if abs(float(rewards_cfg[key])) <= 1.0e-12 else 0.0
            return state

        stage_cfg = self._stage_configs[self._stage]
        for key, value in stage_cfg["reward_weights"].items():
            state[f"weight_{key}"] = float(value)
        for key, value in stage_cfg["reward_scale"].items():
            state[f"scale_{key}"] = float(value)
        if self._stage < len(self._thresholds):
            state["next_threshold"] = float(self._thresholds[self._stage])
        elif self.restore_defaults_on_last_stage_threshold and self._final_threshold is not None:
            state["next_threshold"] = float(self._final_threshold)
        if self._stage < len(self._min_episodes_per_stage):
            state["min_episodes"] = int(self._min_episodes_per_stage[self._stage] / self.num_steps_per_env)
        elif self.restore_defaults_on_last_stage_threshold and self._final_min_episodes is not None:
            state["min_episodes"] = int(self._final_min_episodes / self.num_steps_per_env)
        return state
