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

"""地形任务管理器：仅做 rough curriculum 地形映射，不施加任何控制影响。

当前版本只负责根据机器人所在子地形生成地形标识：
    - 仅在 rough 任务且 terrain_generator.curriculum=True 时启用
    - 仅做 terrain key / terrain id / bool mask 映射
    - 不修改 command、height_cmd 或任何奖励/控制逻辑

说明：
    - 启用时优先使用机器人世界坐标 + terrain_origins + sub_terrains.proportion
      反查当前所在列，并按 curriculum=True 的列分布规则映射到 terrain key。
    - 若位置反查条件不满足，则退回到 terrain_types 张量。
    - 其它情况（flat/play/usd/non-curriculum）统一视为停用。
"""
from __future__ import annotations
import numpy as np
import torch


class TerrainTaskManager:
    """根据机器人所在地形类型，为每个 env 生成地形映射。

    Parameters
    ----------
    terrain_importer : TerrainImporter
        Isaac Lab TerrainImporter 实例（即 env.terrain）。
    sub_terrain_keys : list[str]
        TerrainGeneratorCfg.sub_terrains 的键名列表，顺序决定地形类型索引。
        平面/USD 地形时传入空列表。
    device : str | torch.device
        计算设备（通常与 env.device 一致）。
    """

    def __init__(
        self,
        terrain_importer,
        sub_terrain_keys: list[str],
        device: str | torch.device,
        *,
        enabled: bool = True,
    ) -> None:
        self.terrain = terrain_importer
        self.device = device
        self.enabled = False
        # generator 类型地形才有 terrain_types 属性
        self._is_generator = hasattr(terrain_importer, "terrain_types")
        self._terrain_type = getattr(getattr(terrain_importer, "cfg", None), "terrain_type", None)
        self._use_position_lookup = False
        self._generator_curriculum = False
        # 名称 → 列索引映射（OrderedDict 顺序 = 地形类型 idx）
        self.name_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(sub_terrain_keys)
        }
        self._terrain_names: tuple[str, ...] = tuple(sub_terrain_keys)
        self._terrain_name_to_enum: dict[str, int] = {
            name: i for i, name in enumerate(self._terrain_names)
        }
        self._terrain_name_to_type_idx: dict[str, int] = {
            name: self.name_to_idx[name] for name in self._terrain_names
        }
        self._col_to_key: list[str] = []
        self._col_to_terrain_enum: torch.Tensor | None = None
        self._num_cols = 0
        self._cell_size_y = 0.0
        self._y_start = 0.0

        gen_cfg = None
        if (
            hasattr(terrain_importer, "terrain_origins")
            and terrain_importer.terrain_origins is not None
            and hasattr(terrain_importer, "cfg")
            and getattr(terrain_importer.cfg, "terrain_generator", None) is not None
        ):
            gen_cfg = terrain_importer.cfg.terrain_generator
            self._generator_curriculum = bool(getattr(gen_cfg, "curriculum", False))
            self._num_cols = int(gen_cfg.num_cols)
            self._cell_size_y = float(gen_cfg.size[1])
            self._y_start = float(terrain_importer.terrain_origins[0, 0, 1].item()) - self._cell_size_y * 0.5

            proportions = np.array([cfg.proportion for cfg in gen_cfg.sub_terrains.values()], dtype=float)
            if proportions.size > 0 and proportions.sum() > 0.0:
                proportions /= proportions.sum()
                cumsum = np.cumsum(proportions)
                for col in range(self._num_cols):
                    idx = int(np.min(np.where(col / self._num_cols + 1e-3 < cumsum)[0]))
                    self._col_to_key.append(sub_terrain_keys[idx])
                self._use_position_lookup = True
                self._col_to_terrain_enum = torch.full(
                    (self._num_cols,),
                    fill_value=-1,
                    dtype=torch.long,
                    device=self.device,
                )
                for col, terrain_name in enumerate(self._col_to_key):
                    terrain_enum = self._terrain_name_to_enum.get(terrain_name)
                    if terrain_enum is not None:
                        self._col_to_terrain_enum[col] = terrain_enum

        self.enabled = bool(
            enabled
            and self._terrain_type == "generator"
            and self._generator_curriculum
            and self._terrain_names
        )

        present = list(self._terrain_names)
        if self.enabled:
            idx_info = ", ".join(f"{n}={self.name_to_idx[n]}" for n in present)
            print(
                f"[TerrainTaskManager] 已启用 rough curriculum 地形映射，"
                f"generator 地形共 {len(sub_terrain_keys)} 种，"
                f"可识别地形: {present}，索引: {idx_info}"
            )
        else:
            print(
                "[TerrainTaskManager] 地形任务映射停用："
                f"enabled={enabled}, terrain_type={self._terrain_type}, "
                f"generator_curriculum={self._generator_curriculum}, "
                f"task_names={present}"
            )

    # ------------------------------------------------------------------
    def get_task_ids(self, root_pos_w: torch.Tensor | None = None) -> torch.Tensor:
        """返回每个 env 的地形枚举 id，停用时为 -1。"""
        if root_pos_w is not None:
            num_envs = int(root_pos_w.shape[0])
        elif hasattr(self.terrain, "terrain_types"):
            num_envs = int(self.terrain.terrain_types.shape[0])
        else:
            num_envs = 0

        task_ids = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        if not self.enabled:
            return task_ids

        if self._use_position_lookup and root_pos_w is not None and self._col_to_terrain_enum is not None:
            col = torch.floor((root_pos_w[:, 1] - self._y_start) / self._cell_size_y).long()
            col = col.clamp_(0, self._num_cols - 1)
            return self._col_to_terrain_enum[col]

        types: torch.Tensor = self.terrain.terrain_types
        for name, type_idx in self._terrain_name_to_type_idx.items():
            task_ids[types == type_idx] = self._terrain_name_to_enum[name]
        return task_ids

    # ------------------------------------------------------------------
    def get_task_masks(self, root_pos_w: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """返回各子地形对应的 bool Tensor，形状 [num_envs]。

        返回字典键名与当前 TerrainGeneratorCfg.sub_terrains 的 key 一一对应。
        当前仅在 rough curriculum 地形映射启用时返回非空结果。
        """
        if not self.enabled:
            return {}
        masks: dict[str, torch.Tensor] = {}

        task_ids = self.get_task_ids(root_pos_w)
        for name, task_enum in self._terrain_name_to_enum.items():
            masks[name] = task_ids == task_enum
        return masks
