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

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import torch

from isaaclab.utils import configclass

RangePair = tuple[float, float]
RangeSpec = RangePair | Sequence[RangePair]


@configclass
class TerrainCommandOverrideCfg:
    """Per-terrain command overrides layered on top of the base environment config."""

    height_range: RangeSpec | None = None
    lin_vel_x: RangeSpec | None = None
    lin_vel_y: RangeSpec | None = None
    ang_vel_z: RangeSpec | None = None
    ang_vel_z_heading: RangeSpec | None = None
    ang_vel_z_non_heading: RangeSpec | None = None
    reset_heading_axis_aligned_only: bool | None = None
    disable_predefined_reset_air: bool | None = None
    disable_predefined_reset_ground: bool | None = None
    disable_jump_takeoff: bool | None = None
    disable_special_mode: bool | None = None


@dataclass(frozen=True)
class ResolvedTerrainCommandProfile:
    """Fully resolved command profile used at runtime for one terrain type."""

    height_range: tuple[RangePair, ...]
    lin_vel_x: tuple[RangePair, ...]
    lin_vel_y: tuple[RangePair, ...]
    ang_vel_z_heading: tuple[RangePair, ...]
    ang_vel_z_non_heading: tuple[RangePair, ...]
    reset_heading_axis_aligned_only: bool


class TerrainCommandManager:
    """Resolve and sample terrain-specific command overrides for V13 envs."""

    _RANGE_FIELDS = (
        "height_range",
        "lin_vel_x",
        "lin_vel_y",
        "ang_vel_z_heading",
        "ang_vel_z_non_heading",
    )
    _COMMAND_FIELDS = ("lin_vel_x", "lin_vel_y", "ang_vel_z_heading", "ang_vel_z_non_heading")
    _COMMAND_FIELD_TO_INDEX = {
        "lin_vel_x": 0,
        "lin_vel_y": 1,
        "ang_vel_z_heading": 2,
        "ang_vel_z_non_heading": 2,
    }
    _MERGE_EPS = 1.0e-6

    def __init__(self, terrain, cfg, device: str | torch.device, num_envs: int) -> None:
        self.terrain = terrain
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.enabled = False
        self._switch_hold_steps = max(int(getattr(cfg, "terrain_command_switch_hold_steps", 0)), 0)

        self.current_terrain_key_indices = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.current_profile_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._stable_key_indices = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._candidate_key_indices = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._candidate_counts = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.command_override_values = {
            field: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for field in self._COMMAND_FIELDS
        }
        self.command_override_masks = {
            field: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for field in self._COMMAND_FIELDS
        }

        self._base_profile = self._build_base_profile()
        self._sub_terrain_keys: tuple[str, ...] = ()
        self._tile_key_indices: torch.Tensor | None = None
        self._terrain_key_to_profile_id = torch.zeros(0, dtype=torch.long, device=self.device)
        self._terrain_key_reset_heading_mask = torch.zeros(0, dtype=torch.bool, device=self.device)
        self._terrain_key_disable_predefined_reset_air_mask = torch.zeros(
            0, dtype=torch.bool, device=self.device
        )
        self._terrain_key_disable_predefined_reset_ground_mask = torch.zeros(
            0, dtype=torch.bool, device=self.device
        )
        self._terrain_key_disable_jump_takeoff_mask = torch.zeros(
            0, dtype=torch.bool, device=self.device
        )
        self._terrain_key_disable_special_mode_mask = torch.zeros(
            0, dtype=torch.bool, device=self.device
        )
        self._command_override_enabled_by_field = {
            field: torch.zeros(0, dtype=torch.bool, device=self.device) for field in self._COMMAND_FIELDS
        }
        self._range_specs_by_field: dict[str, list[tuple[RangePair, ...]]] = {
            field: [] for field in self._RANGE_FIELDS
        }
        self._profiles_by_id: list[ResolvedTerrainCommandProfile] = []

        overrides_cfg = dict(getattr(cfg, "terrain_command_overrides", {}) or {})
        if not any(self._override_has_effect(override) for override in overrides_cfg.values()):
            return

        if (
            terrain is None
            or not hasattr(terrain, "cfg")
            or getattr(terrain.cfg, "terrain_type", None) != "generator"
            or getattr(terrain.cfg, "terrain_generator", None) is None
            or not hasattr(terrain, "terrain_origins")
            or terrain.terrain_origins is None
        ):
            print("[TerrainCommandManager] terrain generator metadata unavailable, manager disabled")
            return

        gen_cfg = terrain.cfg.terrain_generator
        self._num_rows = int(gen_cfg.num_rows)
        self._num_cols = int(gen_cfg.num_cols)
        self._cell_size_x = float(gen_cfg.size[0])
        self._cell_size_y = float(gen_cfg.size[1])
        terrain_origins = terrain.terrain_origins
        self._x_start = float(terrain_origins[0, 0, 0].item()) - 0.5 * self._cell_size_x
        self._y_start = float(terrain_origins[0, 0, 1].item()) - 0.5 * self._cell_size_y
        self._sub_terrain_keys = tuple(gen_cfg.sub_terrains.keys())
        if not self._sub_terrain_keys:
            print("[TerrainCommandManager] no sub-terrains found, manager disabled")
            return

        unexpected_keys = sorted(set(overrides_cfg.keys()) - set(self._sub_terrain_keys))
        if unexpected_keys:
            print(
                "[TerrainCommandManager] ignoring terrain_command_overrides for unavailable "
                "terrain keys: "
                + ", ".join(unexpected_keys)
            )
            overrides_cfg = {
                key: value for key, value in overrides_cfg.items() if key in self._sub_terrain_keys
            }
            if not any(self._override_has_effect(override) for override in overrides_cfg.values()):
                print(
                    "[TerrainCommandManager] no matching terrain_command_overrides for "
                    "current terrain generator, manager disabled"
                )
                return

        tile_key_indices = self._build_tile_key_indices()
        if tile_key_indices is None:
            print("[TerrainCommandManager] failed to build terrain key map, manager disabled")
            return
        self._tile_key_indices = tile_key_indices.to(device=self.device, dtype=torch.long)

        self._terrain_key_to_profile_id = torch.zeros(
            len(self._sub_terrain_keys), dtype=torch.long, device=self.device
        )
        self._terrain_key_reset_heading_mask = torch.zeros(
            len(self._sub_terrain_keys), dtype=torch.bool, device=self.device
        )
        self._terrain_key_disable_predefined_reset_air_mask = torch.zeros(
            len(self._sub_terrain_keys), dtype=torch.bool, device=self.device
        )
        self._terrain_key_disable_predefined_reset_ground_mask = torch.zeros(
            len(self._sub_terrain_keys), dtype=torch.bool, device=self.device
        )
        self._terrain_key_disable_jump_takeoff_mask = torch.zeros(
            len(self._sub_terrain_keys), dtype=torch.bool, device=self.device
        )
        self._terrain_key_disable_special_mode_mask = torch.zeros(
            len(self._sub_terrain_keys), dtype=torch.bool, device=self.device
        )
        for field in self._COMMAND_FIELDS:
            self._command_override_enabled_by_field[field] = torch.zeros(
                len(self._sub_terrain_keys), dtype=torch.bool, device=self.device
            )

        unique_profiles: dict[ResolvedTerrainCommandProfile, int] = {}
        for key_idx, terrain_key in enumerate(self._sub_terrain_keys):
            override_cfg = overrides_cfg.get(terrain_key)
            profile, override_flags = self._resolve_profile(override_cfg)
            profile_id = unique_profiles.get(profile)
            if profile_id is None:
                profile_id = len(self._profiles_by_id)
                unique_profiles[profile] = profile_id
                self._profiles_by_id.append(profile)
            self._terrain_key_to_profile_id[key_idx] = profile_id
            self._terrain_key_reset_heading_mask[key_idx] = profile.reset_heading_axis_aligned_only
            self._terrain_key_disable_predefined_reset_air_mask[key_idx] = bool(
                getattr(override_cfg, "disable_predefined_reset_air", False)
            )
            self._terrain_key_disable_predefined_reset_ground_mask[key_idx] = bool(
                getattr(override_cfg, "disable_predefined_reset_ground", False)
            )
            self._terrain_key_disable_jump_takeoff_mask[key_idx] = bool(
                getattr(override_cfg, "disable_jump_takeoff", False)
            )
            self._terrain_key_disable_special_mode_mask[key_idx] = bool(
                getattr(override_cfg, "disable_special_mode", False)
            )
            for field in self._RANGE_FIELDS:
                self._range_specs_by_field[field].append(getattr(profile, field))
            for field in self._COMMAND_FIELDS:
                self._command_override_enabled_by_field[field][key_idx] = override_flags[field]

        self.enabled = True
        print(
            "[TerrainCommandManager] enabled for "
            f"{len(self._sub_terrain_keys)} terrain types, "
            f"{len(self._profiles_by_id)} unique command profiles"
        )

    @property
    def terrain_keys(self) -> tuple[str, ...]:
        """Ordered terrain-key tuple matching the runtime marker/profile indices."""
        return self._sub_terrain_keys

    def get_current_terrain_key_indices(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return current stable terrain-key indices for the provided envs."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        return self.current_terrain_key_indices[env_ids_t]

    def get_command_override_mask(
        self, field: str, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the current active override mask for one command field."""
        if field not in self.command_override_masks:
            raise KeyError(f"Unknown command override field: {field}")
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        return self.command_override_masks[field][env_ids_t]

    def sync_envs(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
        *,
        root_pos_w: torch.Tensor | None = None,
        use_env_origins: bool = False,
    ) -> torch.Tensor:
        """Refresh active terrain/profile ids and return envs whose resolved profile changed."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if not self.enabled or env_ids_t.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)

        source_pos = self._get_source_positions(
            env_ids_t, root_pos_w=root_pos_w, use_env_origins=use_env_origins
        )
        row_indices, col_indices = self._lookup_tile_indices(source_pos)
        raw_key_indices = self._tile_key_indices[row_indices, col_indices]
        key_indices = self._apply_switch_hold(
            env_ids_t, raw_key_indices, force_immediate=use_env_origins
        )
        profile_ids = self._terrain_key_to_profile_id[key_indices]
        changed_mask = profile_ids != self.current_profile_ids[env_ids_t]

        self.current_terrain_key_indices[env_ids_t] = key_indices
        self.current_profile_ids[env_ids_t] = profile_ids
        return env_ids_t[changed_mask]

    def sample_height(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        """Sample height commands for the currently active terrain profile of each env."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return torch.zeros(0, dtype=torch.float, device=self.device)

        key_indices = self.current_terrain_key_indices[env_ids_t]
        samples = torch.empty(env_ids_t.numel(), dtype=torch.float, device=self.device)
        fallback_mask = key_indices < 0
        if torch.any(fallback_mask):
            samples[fallback_mask] = self._sample_range_spec(
                self._base_profile.height_range, int(fallback_mask.sum().item())
            )

        valid_key_indices = key_indices[~fallback_mask]
        if valid_key_indices.numel() > 0:
            valid_samples = torch.empty(valid_key_indices.numel(), dtype=torch.float, device=self.device)
            for key_idx in torch.unique(valid_key_indices).tolist():
                target_mask = valid_key_indices == key_idx
                valid_samples[target_mask] = self._sample_range_spec(
                    self._range_specs_by_field["height_range"][key_idx],
                    int(target_mask.sum().item()),
                )
            samples[~fallback_mask] = valid_samples

        return samples

    def resample_command_overrides(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        """Resample the overridden command fields for the specified envs."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if not self.enabled or env_ids_t.numel() == 0:
            return

        key_indices = self.current_terrain_key_indices[env_ids_t]
        for field in self._COMMAND_FIELDS:
            self.command_override_masks[field][env_ids_t] = False

        valid_mask = key_indices >= 0
        if not torch.any(valid_mask):
            return

        valid_env_ids = env_ids_t[valid_mask]
        valid_key_indices = key_indices[valid_mask]
        for field in self._COMMAND_FIELDS:
            enabled_by_key = self._command_override_enabled_by_field[field]
            for key_idx in torch.unique(valid_key_indices).tolist():
                if not bool(enabled_by_key[key_idx].item()):
                    continue
                target_mask = valid_key_indices == key_idx
                targets = valid_env_ids[target_mask]
                self.command_override_values[field][targets] = self._sample_range_spec(
                    self._range_specs_by_field[field][key_idx],
                    int(target_mask.sum().item()),
                )
                self.command_override_masks[field][targets] = True

    def apply_command_overrides(
        self, command: torch.Tensor, is_heading_env: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply cached terrain-specific command overrides in-place."""
        if not self.enabled or command.numel() == 0:
            return command

        for field in ("lin_vel_x", "lin_vel_y"):
            mask = self.command_override_masks[field]
            if torch.any(mask):
                command[mask, self._COMMAND_FIELD_TO_INDEX[field]] = self.command_override_values[field][mask]

        heading_mask = self.command_override_masks["ang_vel_z_heading"]
        non_heading_mask = self.command_override_masks["ang_vel_z_non_heading"]
        if is_heading_env is None:
            effective_heading_mask = heading_mask
            effective_non_heading_mask = non_heading_mask & ~effective_heading_mask
        else:
            is_heading_env = is_heading_env.to(device=self.device, dtype=torch.bool)
            effective_heading_mask = heading_mask & is_heading_env
            effective_non_heading_mask = non_heading_mask & (~is_heading_env)

        if torch.any(effective_heading_mask):
            command[effective_heading_mask, 2] = self.command_override_values["ang_vel_z_heading"][
                effective_heading_mask
            ]
        if torch.any(effective_non_heading_mask):
            command[effective_non_heading_mask, 2] = self.command_override_values["ang_vel_z_non_heading"][
                effective_non_heading_mask
            ]
        return command

    def get_reset_heading_mask(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the terrain-resolved reset-heading policy for the active episode."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        default_value = self._base_profile.reset_heading_axis_aligned_only
        values = torch.full(
            (env_ids_t.numel(),),
            default_value,
            dtype=torch.bool,
            device=self.device,
        )
        if not self.enabled:
            return values

        key_indices = self.current_terrain_key_indices[env_ids_t]
        valid_mask = key_indices >= 0
        if torch.any(valid_mask):
            values[valid_mask] = self._terrain_key_reset_heading_mask[key_indices[valid_mask]]
        return values

    def get_disable_predefined_reset_air_mask(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return whether each env's current terrain disables predefined air reset modes."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        values = torch.zeros(env_ids_t.numel(), dtype=torch.bool, device=self.device)
        if not self.enabled:
            return values

        key_indices = self.current_terrain_key_indices[env_ids_t]
        valid_mask = key_indices >= 0
        if torch.any(valid_mask):
            values[valid_mask] = self._terrain_key_disable_predefined_reset_air_mask[
                key_indices[valid_mask]
            ]
        return values

    def get_disable_predefined_reset_ground_mask(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return whether each env's current terrain disables predefined ground reset modes."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        values = torch.zeros(env_ids_t.numel(), dtype=torch.bool, device=self.device)
        if not self.enabled:
            return values

        key_indices = self.current_terrain_key_indices[env_ids_t]
        valid_mask = key_indices >= 0
        if torch.any(valid_mask):
            values[valid_mask] = self._terrain_key_disable_predefined_reset_ground_mask[
                key_indices[valid_mask]
            ]
        return values

    def get_disable_jump_takeoff_mask(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return whether each env's current terrain disables jump-takeoff triggers."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        values = torch.zeros(env_ids_t.numel(), dtype=torch.bool, device=self.device)
        if not self.enabled:
            return values

        key_indices = self.current_terrain_key_indices[env_ids_t]
        valid_mask = key_indices >= 0
        if torch.any(valid_mask):
            values[valid_mask] = self._terrain_key_disable_jump_takeoff_mask[
                key_indices[valid_mask]
            ]
        return values

    def get_disable_special_mode_mask(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return whether each env's current terrain disables special-mode command sampling."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        values = torch.zeros(env_ids_t.numel(), dtype=torch.bool, device=self.device)
        if not self.enabled:
            return values

        key_indices = self.current_terrain_key_indices[env_ids_t]
        valid_mask = key_indices >= 0
        if torch.any(valid_mask):
            values[valid_mask] = self._terrain_key_disable_special_mode_mask[
                key_indices[valid_mask]
            ]
        return values

    def _apply_switch_hold(
        self,
        env_ids_t: torch.Tensor,
        raw_key_indices: torch.Tensor,
        *,
        force_immediate: bool,
    ) -> torch.Tensor:
        if force_immediate or self._switch_hold_steps <= 0:
            self._stable_key_indices[env_ids_t] = raw_key_indices
            self._candidate_key_indices[env_ids_t] = raw_key_indices
            self._candidate_counts[env_ids_t] = 0
            return raw_key_indices

        stable = self._stable_key_indices[env_ids_t].clone()
        candidate = self._candidate_key_indices[env_ids_t].clone()
        counts = self._candidate_counts[env_ids_t].clone()

        uninitialized_mask = stable < 0
        if torch.any(uninitialized_mask):
            stable[uninitialized_mask] = raw_key_indices[uninitialized_mask]
            candidate[uninitialized_mask] = raw_key_indices[uninitialized_mask]
            counts[uninitialized_mask] = 0

        changed_mask = raw_key_indices != stable
        candidate = torch.where(changed_mask, candidate, stable)
        counts = torch.where(changed_mask, counts, torch.zeros_like(counts))

        same_candidate_mask = raw_key_indices == candidate
        candidate = torch.where(changed_mask, raw_key_indices, candidate)
        counts = torch.where(
            changed_mask,
            torch.where(same_candidate_mask, counts + 1, torch.ones_like(counts)),
            counts,
        )

        ready_switch_mask = changed_mask & (counts >= self._switch_hold_steps)
        stable = torch.where(ready_switch_mask, candidate, stable)
        counts = torch.where(ready_switch_mask, torch.zeros_like(counts), counts)
        candidate = torch.where(ready_switch_mask, stable, candidate)

        self._stable_key_indices[env_ids_t] = stable
        self._candidate_key_indices[env_ids_t] = candidate
        self._candidate_counts[env_ids_t] = counts
        return stable

    def _build_base_profile(self) -> ResolvedTerrainCommandProfile:
        base_ang_vel_z = self._normalize_range_spec(
            getattr(self.cfg.commands.ranges, "ang_vel_z", None),
            "cfg.commands.ranges.ang_vel_z",
        )
        return ResolvedTerrainCommandProfile(
            height_range=self._normalize_range_spec(
                getattr(self.cfg, "height_range", None), "cfg.height_range"
            ),
            lin_vel_x=self._normalize_range_spec(
                getattr(self.cfg.commands.ranges, "lin_vel_x", None),
                "cfg.commands.ranges.lin_vel_x",
            ),
            lin_vel_y=self._normalize_range_spec(
                getattr(self.cfg.commands.ranges, "lin_vel_y", None),
                "cfg.commands.ranges.lin_vel_y",
            ),
            ang_vel_z_heading=base_ang_vel_z,
            ang_vel_z_non_heading=base_ang_vel_z,
            reset_heading_axis_aligned_only=bool(
                getattr(self.cfg, "reset_heading_axis_aligned_only", False)
            ),
        )

    def _resolve_profile(
        self, override_cfg: TerrainCommandOverrideCfg | None
    ) -> tuple[ResolvedTerrainCommandProfile, dict[str, bool]]:
        override_flags = {field: False for field in self._COMMAND_FIELDS}
        resolved_ranges: dict[str, tuple[RangePair, ...]] = {
            "height_range": self._base_profile.height_range,
            "lin_vel_x": self._base_profile.lin_vel_x,
            "lin_vel_y": self._base_profile.lin_vel_y,
            "ang_vel_z_heading": self._base_profile.ang_vel_z_heading,
            "ang_vel_z_non_heading": self._base_profile.ang_vel_z_non_heading,
        }
        if override_cfg is not None:
            for field in ("height_range", "lin_vel_x", "lin_vel_y"):
                override_value = getattr(override_cfg, field, None)
                if override_value is None:
                    continue
                resolved_ranges[field] = self._normalize_range_spec(
                    override_value, f"terrain_command_overrides.{field}"
                )
                if field in override_flags:
                    override_flags[field] = True

            shared_ang_vel_override = getattr(override_cfg, "ang_vel_z", None)
            if shared_ang_vel_override is not None:
                normalized_shared_ang_vel = self._normalize_range_spec(
                    shared_ang_vel_override, "terrain_command_overrides.ang_vel_z"
                )
                resolved_ranges["ang_vel_z_heading"] = normalized_shared_ang_vel
                resolved_ranges["ang_vel_z_non_heading"] = normalized_shared_ang_vel
                override_flags["ang_vel_z_heading"] = True
                override_flags["ang_vel_z_non_heading"] = True

            heading_ang_vel_override = getattr(override_cfg, "ang_vel_z_heading", None)
            if heading_ang_vel_override is not None:
                resolved_ranges["ang_vel_z_heading"] = self._normalize_range_spec(
                    heading_ang_vel_override, "terrain_command_overrides.ang_vel_z_heading"
                )
                override_flags["ang_vel_z_heading"] = True

            non_heading_ang_vel_override = getattr(override_cfg, "ang_vel_z_non_heading", None)
            if non_heading_ang_vel_override is not None:
                resolved_ranges["ang_vel_z_non_heading"] = self._normalize_range_spec(
                    non_heading_ang_vel_override, "terrain_command_overrides.ang_vel_z_non_heading"
                )
                override_flags["ang_vel_z_non_heading"] = True

        reset_heading_override = (
            override_cfg.reset_heading_axis_aligned_only if override_cfg is not None else None
        )
        reset_heading = (
            bool(reset_heading_override)
            if reset_heading_override is not None
            else self._base_profile.reset_heading_axis_aligned_only
        )
        return (
            ResolvedTerrainCommandProfile(
                height_range=resolved_ranges["height_range"],
                lin_vel_x=resolved_ranges["lin_vel_x"],
                lin_vel_y=resolved_ranges["lin_vel_y"],
                ang_vel_z_heading=resolved_ranges["ang_vel_z_heading"],
                ang_vel_z_non_heading=resolved_ranges["ang_vel_z_non_heading"],
                reset_heading_axis_aligned_only=reset_heading,
            ),
            override_flags,
        )

    def _build_tile_key_indices(self) -> torch.Tensor | None:
        terrain_types_map = self._build_tile_key_indices_from_terrain_types()
        if terrain_types_map is not None:
            return terrain_types_map
        return self._build_tile_key_indices_from_proportions()

    def _build_tile_key_indices_from_terrain_types(self) -> torch.Tensor | None:
        terrain_type_indices_map = getattr(self.terrain, "terrain_type_indices_map", None)
        if terrain_type_indices_map is not None:
            terrain_type_indices_map_t = torch.as_tensor(
                terrain_type_indices_map, dtype=torch.long, device=self.device
            )
            if tuple(terrain_type_indices_map_t.shape) == (self._num_rows, self._num_cols):
                return self._validate_type_grid(terrain_type_indices_map_t)

        terrain_types = getattr(self.terrain, "terrain_types", None)
        if terrain_types is None:
            return None

        terrain_types_t = torch.as_tensor(terrain_types, dtype=torch.long, device=self.device)
        if terrain_types_t.ndim == 2 and tuple(terrain_types_t.shape) == (self._num_rows, self._num_cols):
            return self._validate_type_grid(terrain_types_t)

        if terrain_types_t.ndim == 1 and terrain_types_t.numel() == self._num_rows * self._num_cols:
            return self._validate_type_grid(terrain_types_t.view(self._num_rows, self._num_cols))

        env_origins = getattr(self.terrain, "env_origins", None)
        if (
            terrain_types_t.ndim == 1
            and env_origins is not None
            and terrain_types_t.numel() == int(env_origins.shape[0])
        ):
            env_origins_t = torch.as_tensor(env_origins, dtype=torch.float, device=self.device)
            row_indices, col_indices = self._lookup_tile_indices(env_origins_t)
            type_grid = torch.full(
                (self._num_rows, self._num_cols), -1, dtype=torch.long, device=self.device
            )
            type_grid[row_indices, col_indices] = terrain_types_t
            if torch.any(type_grid < 0):
                return None
            return self._validate_type_grid(type_grid)

        return None

    def _build_tile_key_indices_from_proportions(self) -> torch.Tensor | None:
        gen_cfg = self.terrain.cfg.terrain_generator
        if not bool(getattr(gen_cfg, "curriculum", False)):
            return None

        proportions = torch.tensor(
            [float(sub_cfg.proportion) for sub_cfg in gen_cfg.sub_terrains.values()],
            dtype=torch.float,
            device=self.device,
        )
        if proportions.numel() == 0:
            return None
        if torch.sum(proportions) <= 0.0:
            return None

        proportions = proportions / torch.sum(proportions)
        cumulative = torch.cumsum(proportions, dim=0)
        key_indices = torch.zeros((self._num_rows, self._num_cols), dtype=torch.long, device=self.device)
        for col in range(self._num_cols):
            fraction = (float(col) / float(self._num_cols)) + 1.0e-3
            match_indices = torch.nonzero(fraction < cumulative, as_tuple=False)
            terrain_idx = int(match_indices[0].item()) if match_indices.numel() > 0 else len(self._sub_terrain_keys) - 1
            key_indices[:, col] = terrain_idx
        return key_indices

    def _validate_type_grid(self, type_grid: torch.Tensor) -> torch.Tensor | None:
        if torch.any(type_grid < 0):
            return None
        if torch.any(type_grid >= len(self._sub_terrain_keys)):
            return None
        return type_grid.to(device=self.device, dtype=torch.long)

    def _get_source_positions(
        self,
        env_ids_t: torch.Tensor,
        *,
        root_pos_w: torch.Tensor | None,
        use_env_origins: bool,
    ) -> torch.Tensor:
        if use_env_origins:
            env_origins = getattr(self.terrain, "env_origins", None)
            if env_origins is None:
                raise RuntimeError("terrain.env_origins is required for reset-time terrain lookup")
            return torch.as_tensor(env_origins, device=self.device, dtype=torch.float)[env_ids_t]
        if root_pos_w is None:
            raise RuntimeError("root_pos_w is required for runtime terrain lookup")
        return root_pos_w[env_ids_t]

    def _lookup_tile_indices(self, positions_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        row_indices = torch.floor((positions_w[:, 0] - self._x_start) / self._cell_size_x).long()
        col_indices = torch.floor((positions_w[:, 1] - self._y_start) / self._cell_size_y).long()
        row_indices = row_indices.clamp_(0, self._num_rows - 1)
        col_indices = col_indices.clamp_(0, self._num_cols - 1)
        return row_indices, col_indices

    def _sample_range_spec(self, range_spec: tuple[RangePair, ...], count: int) -> torch.Tensor:
        if count <= 0:
            return torch.zeros(0, dtype=torch.float, device=self.device)

        if len(range_spec) == 1:
            low, high = range_spec[0]
            if low == high:
                return torch.full((count,), low, dtype=torch.float, device=self.device)
            return torch.empty(count, dtype=torch.float, device=self.device).uniform_(low, high)

        lengths = torch.tensor(
            [max(high - low, 0.0) for low, high in range_spec],
            dtype=torch.float,
            device=self.device,
        )
        if torch.sum(lengths) <= 0.0:
            probs = torch.full((len(range_spec),), 1.0 / len(range_spec), dtype=torch.float, device=self.device)
        else:
            probs = lengths / torch.sum(lengths)

        segment_indices = torch.multinomial(probs, count, replacement=True)
        samples = torch.empty(count, dtype=torch.float, device=self.device)
        for segment_idx, (low, high) in enumerate(range_spec):
            mask = segment_indices == segment_idx
            if not torch.any(mask):
                continue
            num_segment_samples = int(mask.sum().item())
            if low == high:
                samples[mask] = low
            else:
                samples[mask] = torch.empty(
                    num_segment_samples, dtype=torch.float, device=self.device
                ).uniform_(low, high)
        return samples

    def _normalize_range_spec(
        self, range_spec: RangeSpec | None, field_name: str
    ) -> tuple[RangePair, ...]:
        if range_spec is None:
            raise ValueError(f"{field_name} cannot be None")

        if self._is_range_pair(range_spec):
            raw_ranges = [(float(range_spec[0]), float(range_spec[1]))]
        elif isinstance(range_spec, Sequence) and not isinstance(range_spec, (str, bytes)):
            raw_ranges = []
            for item in range_spec:
                if not self._is_range_pair(item):
                    raise ValueError(f"{field_name} contains invalid range entry: {item!r}")
                raw_ranges.append((float(item[0]), float(item[1])))
        else:
            raise ValueError(f"{field_name} must be a range pair or a sequence of range pairs")

        if not raw_ranges:
            raise ValueError(f"{field_name} cannot be empty")

        normalized = [(min(low, high), max(low, high)) for low, high in raw_ranges]
        normalized.sort(key=lambda item: item[0])

        merged: list[list[float]] = []
        for low, high in normalized:
            if not merged:
                merged.append([low, high])
                continue
            if low <= merged[-1][1] + self._MERGE_EPS:
                merged[-1][1] = max(merged[-1][1], high)
            else:
                merged.append([low, high])
        if not merged:
            raise ValueError(f"{field_name} produced no valid ranges after normalization")
        return tuple((float(low), float(high)) for low, high in merged)

    def _override_has_effect(self, override_cfg: TerrainCommandOverrideCfg | None) -> bool:
        if override_cfg is None:
            return False
        return any(getattr(override_cfg, field, None) is not None for field in self._RANGE_FIELDS) or (
            override_cfg.ang_vel_z is not None
        ) or (
            override_cfg.reset_heading_axis_aligned_only is not None
        ) or (
            override_cfg.disable_predefined_reset_air is True
        ) or (
            override_cfg.disable_predefined_reset_ground is True
        ) or (
            override_cfg.disable_jump_takeoff is True
        ) or (
            override_cfg.disable_special_mode is True
        )

    def _as_env_ids_tensor(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    @staticmethod
    def _is_range_pair(value) -> bool:
        return (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == 2
            and all(isinstance(v, Real) for v in value)
        )
