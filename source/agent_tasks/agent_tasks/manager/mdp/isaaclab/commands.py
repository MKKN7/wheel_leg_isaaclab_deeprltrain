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

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import MISSING
from numbers import Real

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

try:
    from omegaconf import DictConfig, ListConfig
except ImportError:  # pragma: no cover - optional dependency
    DictConfig = ()  # type: ignore[assignment,misc]
    ListConfig = ()  # type: ignore[assignment,misc]

RangePair = tuple[float, float]
RangeSpec = RangePair | Sequence[RangePair]


class LinearStandingUniformVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command that only zeros linear velocity for standing envs."""

    cfg: LinearStandingUniformVelocityCommandCfg

    def _update_command(self):
        """Post-process velocity commands while keeping yaw active in standing envs."""
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :2] = 0.0


@configclass
class LinearStandingUniformVelocityCommandCfg(UniformVelocityCommandCfg):
    """Config for uniform velocity commands where standing envs keep sampled yaw commands."""

    class_type: type = LinearStandingUniformVelocityCommand


@configclass
class HeightWaveCfg:
    """Sinusoidal height command used by a special command mode."""

    mean: float = 0.25
    """Mean target height [m]."""

    mean_range: RangePair | None = None
    """Optional per-env random mean range [m]. Overrides ``mean`` when set."""

    amplitude: float = 0.0
    """Sine amplitude [m]."""

    amplitude_range: RangePair | None = None
    """Optional per-env random amplitude range [m]. Overrides ``amplitude`` when set."""

    frequency_hz: float = 0.0
    """Sine frequency [Hz]."""

    frequency_range_hz: RangePair | None = None
    """Optional per-env random frequency range [Hz]. Overrides ``frequency_hz`` when set."""

    phase: float = 0.0
    """Fixed phase offset [rad]."""

    random_phase: bool = False
    """When True, sample an independent phase per env on mode entry."""

    phase_range: RangePair | None = None
    """Optional random phase range [rad]. Defaults to one full cycle."""

    clamp_range: RangePair | None = None
    """Optional output clamp range [m]."""


@configclass
class HeightStepCfg:
    """Square-wave height command used by a height-only special mode."""

    mean: float = 0.25
    """Mean target height [m]."""

    mean_range: RangePair | None = None
    """Optional per-env random mean range [m]. Overrides ``mean`` when set."""

    amplitude: float = 0.0
    """Square-wave amplitude [m]."""

    amplitude_range: RangePair | None = None
    """Optional per-env random amplitude range [m]. Overrides ``amplitude`` when set."""

    frequency_hz: float = 0.0
    """Square-wave frequency [Hz]."""

    frequency_range_hz: RangePair | None = None
    """Optional per-env random frequency range [Hz]. Overrides ``frequency_hz`` when set."""

    phase: float = 0.0
    """Fixed phase offset [rad]."""

    random_phase: bool = False
    """When True, sample an independent phase per env on mode entry."""

    phase_range: RangePair | None = None
    """Optional random phase range [rad]. Defaults to one full cycle."""

    clamp_range: RangePair | None = None
    """Optional output clamp range [m]."""


@configclass
class SpecialModeEntryCfg:
    """Configuration for a single special command mode.

    Each mode is evaluated independently during command resampling.
    The mode is only active within its configured training-iteration
    window and carries its own independent command ranges.
    """

    rel_envs: float = 0.0
    """Fraction of non-standing environments assigned to this mode when active."""

    iteration_start: int = 0
    """Training iteration at which this mode becomes active (inclusive)."""

    iteration_end: int = -1
    """Training iteration at which this mode becomes inactive (exclusive).
    -1 means the mode never expires."""

    @configclass
    class Ranges:
        """Independent command ranges for a special mode.

        Each field accepts a single interval ``(low, high)`` or a sequence of
        intervals ``[(low1, high1), (low2, high2), ...]``. When multiple
        intervals are provided, one is selected at random proportional to its
        width, then a value is uniformly sampled within it.
        """

        lin_vel_x: RangeSpec = MISSING  # type: ignore[assignment]
        """Linear velocity x range(s) [m/s]."""

        lin_vel_y: RangeSpec = MISSING  # type: ignore[assignment]
        """Linear velocity y range(s) [m/s]."""

        ang_vel_z: RangeSpec = MISSING  # type: ignore[assignment]
        """Angular velocity z range(s) [rad/s]."""

    ranges: Ranges = MISSING  # type: ignore[assignment]
    """Command ranges exclusively used when this special mode is active."""

    height_range: RangeSpec | None = None
    """Optional height command range(s) sampled by the environment for this mode."""

    height_wave: HeightWaveCfg | None = None
    """Optional dynamic sinusoidal height command for this mode."""

    height_step: HeightStepCfg | None = None
    """Optional dynamic square-wave height command for this mode."""

    jump_takeoff_enabled: bool = False
    """When True, envs assigned to this mode request the jump-takeoff state machine."""

    disable_jump_takeoff: bool = False
    """When True, envs assigned to this mode cannot enter the jump-takeoff state machine."""

    debug_print: bool = False
    """When True, print the mode name and sampled commands on every resample."""


class SpecialModeUniformVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with multiple train-iteration-gated special modes.

    During each command resample, every active special mode is assigned a disjoint
    slice of the non-standing environments proportional to its ``rel_envs``.
    Modes do not compete via first-match priority; assignment order is shuffled
    each resample so no mode permanently outranks another.

    Assigned envs sample velocity from that mode's independent ranges and are
    NOT overridden by ``heading_command`` in :meth:`_update_command`.

    The set of active modes can be changed at runtime via
    :meth:`set_training_iteration`, which gates each mode on its
    ``iteration_start`` / ``iteration_end`` window.
    """

    cfg: SpecialModeUniformVelocityCommandCfg

    def __init__(self, cfg: SpecialModeUniformVelocityCommandCfg, env):
        super().__init__(cfg, env)

        # Hydra/OmegaConf may deserialize nested configclass entries as plain
        # dicts, so normalize them back to SpecialModeEntryCfg instances here.
        # If special_modes is a mapping, its keys are the mode names used for
        # logging/debugging while values are the actual mode configs.
        raw_modes = cfg.special_modes
        mode_names: tuple[str, ...] | None = None
        if isinstance(raw_modes, (Mapping, DictConfig)):
            raw_items = tuple(raw_modes.items())
            mode_names = tuple(str(name) for name, _ in raw_items)
            raw_modes = tuple(mode for _, mode in raw_items)
        elif isinstance(raw_modes, ListConfig):
            raw_modes = tuple(raw_modes)
        normalized: list[SpecialModeEntryCfg] = []
        for m in raw_modes:
            if isinstance(m, (dict, DictConfig)):
                payload = dict(m)
                # Older tuple-style configs may still contain a legacy name
                # field. Mode names now come from mapping keys.
                payload.pop("name", None)
                if isinstance(payload.get("ranges"), (dict, DictConfig)):
                    payload["ranges"] = SpecialModeEntryCfg.Ranges(**dict(payload["ranges"]))
                if isinstance(payload.get("height_wave"), (dict, DictConfig)):
                    payload["height_wave"] = HeightWaveCfg(**dict(payload["height_wave"]))
                if isinstance(payload.get("height_step"), (dict, DictConfig)):
                    payload["height_step"] = HeightStepCfg(**dict(payload["height_step"]))
                m = SpecialModeEntryCfg(**payload)
            elif isinstance(m, SpecialModeEntryCfg):
                if isinstance(m.ranges, (dict, DictConfig)):
                    m.ranges = SpecialModeEntryCfg.Ranges(**dict(m.ranges))
                if isinstance(m.height_wave, (dict, DictConfig)):
                    m.height_wave = HeightWaveCfg(**dict(m.height_wave))
                if isinstance(m.height_step, (dict, DictConfig)):
                    m.height_step = HeightStepCfg(**dict(m.height_step))
            normalized.append(m)
        # Replace cfg.special_modes with the normalized tuple so all
        # downstream attribute access works uniformly.
        cfg.special_modes = tuple(normalized)  # type: ignore[assignment]

        num_modes = max(len(cfg.special_modes), 1)
        self.special_mode_id = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._training_iteration = 0
        if mode_names is not None:
            if len(mode_names) != len(cfg.special_modes):
                raise ValueError(
                    "special_modes mapping keys and normalized mode configs have different lengths."
                )
            self._mode_names = mode_names
        else:
            self._mode_names = tuple(f"mode_{i}" for i, _ in enumerate(cfg.special_modes))
        self._mode_enabled = torch.zeros(num_modes, dtype=torch.bool, device="cpu")

        # 立即根据当前 _training_iteration 初始化模式启停状态，
        # 防止在外部 set_training_progress 回调之前 _resample_command 被调用时
        # _mode_enabled 仍为全零，导致 iteration_start 门控不生效。
        self.set_training_iteration(self._training_iteration)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_training_iteration(self, iteration: int) -> None:
        """Update the current training iteration and recompute per-mode enabled flags."""
        prev_enabled = self._mode_enabled.clone()
        self._training_iteration = int(iteration)
        for i, mode_cfg in enumerate(self.cfg.special_modes):
            start = int(mode_cfg.iteration_start)
            end = int(mode_cfg.iteration_end)
            self._mode_enabled[i] = (
                self._training_iteration >= start
                and (end < 0 or self._training_iteration < end)
            )
            if (
                self._mode_wants_debug(i)
                and bool(self._mode_enabled[i].item())
                and not bool(prev_enabled[i].item())
            ):
                print(
                    f"[SpecialMode] mode={self._mode_names[i]} enabled at "
                    f"iteration={self._training_iteration}",
                    flush=True,
                )

    def get_special_mode_name(self, env_ids: Sequence[int] | None = None) -> list[str]:
        """Return the human-readable mode name for each environment (or ``"normal"``)."""
        if env_ids is None:
            env_ids = slice(None)  # type: ignore[assignment]
        ids = self.special_mode_id[env_ids].tolist()
        return [self._mode_names[i] if i >= 0 else "normal" for i in ids]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scalar_probability(value, name: str) -> float:
        """Return a scalar probability, accepting accidental one-item sequences."""
        if isinstance(value, (list, tuple, ListConfig)):
            if len(value) != 1:
                raise ValueError(f"{name} must be a scalar probability, got {value!r}.")
            value = value[0]
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}.")
        return value

    @staticmethod
    def _normalize_range_spec(value: RangeSpec) -> tuple[RangePair, ...]:
        """Normalize a single or multi-range spec to a sorted, merged tuple."""
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Real)):
            items = list(value)
            if all(isinstance(item, Real) for item in items) and len(items) == 2:
                raw_ranges: list[RangePair] = [(float(items[0]), float(items[1]))]
            else:
                raw_ranges = []
                for item in items:
                    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                        seq = list(item)
                        if len(seq) == 2 and all(isinstance(v, Real) for v in seq):
                            raw_ranges.append((float(seq[0]), float(seq[1])))
                        else:
                            raise ValueError(f"Invalid range entry: {item!r}")
                    else:
                        raise ValueError(f"Invalid range entry: {item!r}")
        else:
            raise ValueError(f"Range spec must be a pair or sequence of pairs. Got: {value!r}")

        if not raw_ranges:
            raise ValueError("Range spec cannot be empty")

        normalized = [(min(low, high), max(low, high)) for low, high in raw_ranges]
        normalized.sort(key=lambda item: item[0])

        merged: list[list[float]] = []
        for low, high in normalized:
            if not merged:
                merged.append([low, high])
                continue
            if low <= merged[-1][1] + 1.0e-6:
                merged[-1][1] = max(merged[-1][1], high)
            else:
                merged.append([low, high])
        return tuple((float(low), float(high)) for low, high in merged)

    @staticmethod
    def _sample_range_spec(
        range_spec: tuple[RangePair, ...],
        count: int,
        device: str | torch.device,
    ) -> torch.Tensor:
        """Sample *count* values from a normalized range spec."""
        if count <= 0:
            return torch.zeros(0, device=device)

        if len(range_spec) == 1:
            low, high = range_spec[0]
            if low == high:
                return torch.full((count,), low, dtype=torch.float, device=device)
            return torch.empty(count, dtype=torch.float, device=device).uniform_(low, high)

        lengths = torch.tensor(
            [max(high - low, 0.0) for low, high in range_spec],
            dtype=torch.float,
            device=device,
        )
        if torch.sum(lengths) <= 0.0:
            probs = torch.full((len(range_spec),), 1.0 / len(range_spec), dtype=torch.float, device=device)
        else:
            probs = lengths / torch.sum(lengths)

        segment_indices = torch.multinomial(probs, count, replacement=True)
        samples = torch.empty(count, dtype=torch.float, device=device)
        for segment_idx, (low, high) in enumerate(range_spec):
            mask = segment_indices == segment_idx
            if not torch.any(mask):
                continue
            n = int(mask.sum().item())
            if low == high:
                samples[mask] = low
            else:
                samples[mask] = torch.empty(n, dtype=torch.float, device=device).uniform_(low, high)
        return samples

    def _mode_wants_debug(self, mode_idx: int) -> bool:
        return bool(getattr(self.cfg.special_modes[mode_idx], "debug_print", False))

    def _is_mode_active(self, mode_idx: int) -> bool:
        """Return whether *mode_idx* is both configured and within its iteration window."""
        if self.cfg.special_modes[mode_idx].rel_envs <= 0.0:
            return False
        return bool(self._mode_enabled[mode_idx].item())

    def _assign_special_modes(self, non_standing_ids: torch.Tensor) -> None:
        """Assign special modes per env using disjoint probability buckets.

        Each active mode occupies ``rel_envs`` of the resampled non-standing
        environments. A single ``u ~ U(0, 1)`` draw per env selects at most one
        mode, so this works for both bulk reset and single-env resamples
        (unlike ``int(n * rel_envs)`` batch slicing, which assigns zero modes
        when ``n`` is small).
        """
        n_ns = int(non_standing_ids.numel())
        if n_ns <= 0:
            return

        active_entries = [
            (mode_idx, float(mode_cfg.rel_envs))
            for mode_idx, mode_cfg in enumerate(self.cfg.special_modes)
            if self._is_mode_active(mode_idx)
        ]
        if not active_entries:
            return

        order = torch.randperm(len(active_entries), device=self.device).tolist()
        shuffled = [active_entries[i] for i in order]
        r = torch.rand(n_ns, device=self.device)
        cumulative = 0.0
        for mode_idx, rel in shuffled:
            low = cumulative
            high = cumulative + rel
            slot_mask = (r >= low) & (r < high)
            if torch.any(slot_mask):
                self.special_mode_id[non_standing_ids[slot_mask]] = mode_idx
            cumulative = high

    def _get_special_mode_ready_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return envs that may enter special modes at this resample."""
        if env_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        ready = torch.ones(env_ids.numel(), dtype=torch.bool, device=self.device)
        env = getattr(self, "_env", None)
        min_episode_time = float(getattr(self.cfg, "special_mode_min_episode_time", 0.0))
        if min_episode_time > 0.0 and env is not None:
            step_dt = float(getattr(env, "step_dt", 0.0) or getattr(env, "physics_dt", 0.0))
            episode_time = env.episode_length_buf[env_ids].to(dtype=torch.float) * step_dt
            ready &= episode_time >= min_episode_time

        if bool(getattr(self.cfg, "special_mode_require_stable", False)):
            robot_data = self.robot.data
            gravity_xy = robot_data.projected_gravity_b[env_ids, :2]
            upright_max = float(getattr(self.cfg, "special_mode_stable_projected_gravity_xy_norm_max", 0.35))
            if upright_max > 0.0:
                ready &= torch.linalg.norm(gravity_xy, dim=-1) <= upright_max

            lin_vel_max = float(getattr(self.cfg, "special_mode_stable_root_lin_vel_b_abs_max", 3.0))
            if lin_vel_max > 0.0:
                root_lin_vel_abs = torch.amax(torch.abs(robot_data.root_lin_vel_b[env_ids]), dim=-1)
                ready &= root_lin_vel_abs <= lin_vel_max

            ang_vel_max = float(getattr(self.cfg, "special_mode_stable_root_ang_vel_b_abs_max", 5.0))
            if ang_vel_max > 0.0:
                root_ang_vel_abs = torch.amax(torch.abs(robot_data.root_ang_vel_b[env_ids]), dim=-1)
                ready &= root_ang_vel_abs <= ang_vel_max

        return ready

    def _sync_training_iteration_from_env(self) -> None:
        """Refresh iteration gates from env-side extrapolation (scheme B)."""
        env = getattr(self, "_env", None)
        if env is None:
            return
        get_iteration = getattr(env, "_get_training_iteration", None)
        if get_iteration is None:
            try:
                from agent_tasks.direct.wheelbipe.wheelbipe_V14.env import (
                    get_extrapolated_training_iteration,
                )
            except ImportError:
                return
            iteration = int(get_extrapolated_training_iteration(env))
        else:
            iteration = int(get_iteration())
        self.set_training_iteration(iteration)

    # ------------------------------------------------------------------
    # Core overrides
    # ------------------------------------------------------------------

    def _resample_command(self, env_ids: Sequence[int]):
        self._sync_training_iteration_from_env()
        r = torch.empty(len(env_ids), device=self.device)

        # ── standing (highest priority) ──
        rel_standing_envs = self._scalar_probability(self.cfg.rel_standing_envs, "rel_standing_envs")
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= rel_standing_envs

        # ── special modes (middle priority, within non-standing) ──
        self.special_mode_id[env_ids] = -1
        non_standing_ids = env_ids[~self.is_standing_env[env_ids]]
        normal_ids = non_standing_ids

        if non_standing_ids.numel() > 0 and len(self.cfg.special_modes) > 0:
            ready_mask = self._get_special_mode_ready_mask(non_standing_ids)
            special_candidate_ids = non_standing_ids[ready_mask]
            delayed_ids = non_standing_ids[~ready_mask]
            if special_candidate_ids.numel() > 0:
                self._assign_special_modes(special_candidate_ids)

            # ── resample commands per-mode ──
            for mode_idx in range(len(self.cfg.special_modes)):
                mode_ids = special_candidate_ids[self.special_mode_id[special_candidate_ids] == mode_idx]
                if mode_ids.numel() == 0:
                    continue
                ranges = self.cfg.special_modes[mode_idx].ranges
                self.vel_command_b[mode_ids, 0] = self._sample_range_spec(
                    self._normalize_range_spec(ranges.lin_vel_x),
                    int(mode_ids.numel()),
                    self.device,
                )
                self.vel_command_b[mode_ids, 1] = self._sample_range_spec(
                    self._normalize_range_spec(ranges.lin_vel_y),
                    int(mode_ids.numel()),
                    self.device,
                )
                self.vel_command_b[mode_ids, 2] = self._sample_range_spec(
                    self._normalize_range_spec(ranges.ang_vel_z),
                    int(mode_ids.numel()),
                    self.device,
                )

                if self._mode_wants_debug(mode_idx):
                    mode_name = self._mode_names[mode_idx]
                    for eid in mode_ids.tolist():
                        vx = self.vel_command_b[eid, 0].item()
                        vy = self.vel_command_b[eid, 1].item()
                        vz = self.vel_command_b[eid, 2].item()
                        print(
                            f"[SpecialMode] env={eid:3d}  mode={mode_name}  "
                            f"lin_vel_x={vx:+.3f}  lin_vel_y={vy:+.3f}  ang_vel_z={vz:+.3f}",
                            flush=True,
                        )

            normal_candidate_ids = special_candidate_ids[self.special_mode_id[special_candidate_ids] < 0]
            normal_ids = torch.cat((delayed_ids, normal_candidate_ids))

        # ── normal envs: delegate to parent (default ranges + heading) ──
        if normal_ids.numel() > 0:
            super()._resample_command(normal_ids)

    def disable_special_modes(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        """Clear special-mode assignment and resample normal commands for selected envs."""
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return env_ids_t

        special_mask = self.special_mode_id[env_ids_t] >= 0
        affected_ids = env_ids_t[special_mask]
        if affected_ids.numel() == 0:
            return affected_ids

        self.special_mode_id[affected_ids] = -1
        normal_ids = affected_ids[~self.is_standing_env[affected_ids]]
        if normal_ids.numel() > 0:
            super()._resample_command(normal_ids)
        self._update_command()
        return affected_ids

    def _update_command(self):
        """Post-process velocity commands.

        Priority chain (highest to lowest):
        1. standing envs → all velocity commands zeroed
        2. special-mode envs → sampled values kept as-is (skip heading)
        3. normal envs → heading_command applied to heading envs
        """
        special_mask = self.special_mode_id >= 0

        # heading_command only applies to non-special-mode envs
        if self.cfg.heading_command:
            heading_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_ids = heading_ids[~special_mask[heading_ids]]
            if heading_ids.numel() > 0:
                heading_error = math_utils.wrap_to_pi(
                    self.heading_target[heading_ids] - self.robot.data.heading_w[heading_ids]
                )
                self.vel_command_b[heading_ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * heading_error,
                    min=self.cfg.ranges.ang_vel_z[0],
                    max=self.cfg.ranges.ang_vel_z[1],
                )

        # standing envs: zero all velocity (highest priority)
        standing_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if standing_ids.numel() > 0:
            self.vel_command_b[standing_ids, :] = 0.0


@configclass
class SpecialModeUniformVelocityCommandCfg(UniformVelocityCommandCfg):
    """Config for uniform velocity commands with multiple named special modes.

    Special modes are mutually exclusive with standing environments
    and completely bypass ``heading_command`` post-processing. Each mode
    carries its own ``rel_envs`` fraction of the non-standing population,
    optional training-iteration window, and independent command ranges.

    Active modes receive disjoint environment slices proportional to
    ``rel_envs``; assignment order is shuffled each resample.
    """

    class_type: type = SpecialModeUniformVelocityCommand

    special_modes: Mapping[str, SpecialModeEntryCfg] | tuple[SpecialModeEntryCfg, ...] = ()
    """Special modes; mapping keys are mode names, values hold sampling settings."""

    special_mode_min_episode_time: float = 0.0
    """Minimum seconds after reset before an env may enter any special mode."""

    special_mode_require_stable: bool = False
    """When True, special modes also require upright and bounded root velocities."""

    special_mode_stable_projected_gravity_xy_norm_max: float = 0.35
    """Maximum horizontal projected-gravity norm for special-mode readiness."""

    special_mode_stable_root_lin_vel_b_abs_max: float = 3.0
    """Maximum absolute root body linear velocity component for readiness; <=0 disables this check."""

    special_mode_stable_root_ang_vel_b_abs_max: float = 5.0
    """Maximum absolute root body angular velocity component for readiness; <=0 disables this check."""
