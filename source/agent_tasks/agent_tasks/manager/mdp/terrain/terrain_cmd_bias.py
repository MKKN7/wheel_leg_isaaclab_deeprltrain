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

"""
Terrain-specific command bias manager for rough terrain environments.

使用方法：
  1. 在 env_cfg.py 中为各地形类型创建 TerrainCmdBias 实例，组成 dict
  2. 在 WheelbipeV13Env.__init__ 中用 TerrainCommandBiasManager 初始化
  3. 在 _on_command_updated() 钩子中调用 apply_bias()

TerrainCmdBias 字段说明：
  ang_vel_scale  - 角速度乘以该系数        (1.0 = 不变)
  lin_vel_delta  - 线速度加上 sign(vx)*delta (0.0 = 不变)，结果 clamp 到 cmd_ranges
  height_delta   - 高度加上该值             (0.0 = 不变)，结果 clamp 到 height_range
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 数据类：单种地形的偏置描述
# ---------------------------------------------------------------------------

@dataclass
class TerrainCmdBias:
    """Command bias applied when robot is on a specific sub-terrain type.

    所有偏置均作用于 command_generator 输出之后；结果被 clamp 到 cfg 配置的范围内。
    Keys 需与 TerrainGeneratorCfg.sub_terrains 中的 key 名称一一对应。
    """
    # 角速度：cmd_omega = cmd_omega * ang_vel_scale
    ang_vel_scale: float = 1.0
    # 线速度：cmd_vx += sign(cmd_vx) * lin_vel_delta，再 clamp 到 [vx_min, vx_max]
    lin_vel_delta: float = 0.0
    # 高度命令：height_cmd += height_delta，再 clamp 到 height_range
    height_delta: float = 0.0

    lin_vel_delta_positive: float = 0.0

    def is_identity(self) -> bool:
        """True 表示该偏置不做任何修改，可提前跳过。"""
        return self.ang_vel_scale == 1.0 and self.lin_vel_delta == 0.0 and self.height_delta == 0.0


# ---------------------------------------------------------------------------
# 管理器：地形检测 + 批量偏置应用
# ---------------------------------------------------------------------------

class TerrainCommandBiasManager:
    """实时地形检测与按地形类型分发命令偏置。

    地形几何参数（格子尺寸、网格起点）全部从 TerrainImporter 的
    terrain_origins / TerrainGeneratorCfg 中读取，不硬编码任何数值。

    col → sub_terrain_key 的映射基于 curriculum=True 的确定性公式：
      same-column all-rows share the same terrain type。
    """

    def __init__(
        self,
        terrain,
        sub_terrain_biases: dict[str, TerrainCmdBias],
        device: str = "cuda",
        switch_hold_steps: int = 0,
    ) -> None:
        """
        Args:
            terrain: TerrainImporter 实例 (env.terrain)。
            sub_terrain_biases: {sub_terrain_key: TerrainCmdBias} 偏置配置。
                                未出现在 dict 中的地形类型不做任何修改。
            device: torch 设备字符串。
        """
        self.device = device
        self.sub_terrain_biases = sub_terrain_biases
        self._valid = False
        # 列切换防抖：只有连续 N 步检测到新列才真正切换
        self._switch_hold_steps = max(int(switch_hold_steps), 0)
        self._stable_col_indices: torch.Tensor | None = None
        self._candidate_col_indices: torch.Tensor | None = None
        self._candidate_counts: torch.Tensor | None = None

        # ---- 检查依赖 ----
        if terrain is None or not hasattr(terrain, "terrain_origins") or terrain.terrain_origins is None:
            print("[TerrainCmdBias] ⚠ terrain_origins 不可用，地形偏置功能已禁用")
            return
        if not hasattr(terrain, "cfg") or terrain.cfg.terrain_generator is None:
            print("[TerrainCmdBias] ⚠ terrain_generator 不可用，地形偏置功能已禁用")
            return

        gen_cfg = terrain.cfg.terrain_generator
        self._num_rows: int = gen_cfg.num_rows
        self._num_cols: int = gen_cfg.num_cols
        # 从 cfg 读取格子尺寸（支持非正方形地形）
        self._cell_size_x: float = float(gen_cfg.size[0])
        self._cell_size_y: float = float(gen_cfg.size[1])

        # terrain_origins shape: (num_rows, num_cols, 3)，每格的世界中心坐标
        t_origins = terrain.terrain_origins
        self._x_start: float = float(t_origins[0, 0, 0].item()) - self._cell_size_x * 0.5
        self._y_start: float = float(t_origins[0, 0, 1].item()) - self._cell_size_y * 0.5

        # ---- 构建 col → sub_terrain_key 映射（与 curriculum 生成器公式一致）----
        sub_keys: list[str] = list(gen_cfg.sub_terrains.keys())
        proportions = np.array(
            [v.proportion for v in gen_cfg.sub_terrains.values()], dtype=float
        )
        proportions /= proportions.sum()
        cumsum = np.cumsum(proportions)
        self._col_to_key: list[str] = []
        for col in range(self._num_cols):
            idx = int(np.min(np.where(col / self._num_cols + 1e-3 < cumsum)[0]))
            self._col_to_key.append(sub_keys[idx])

        # ---- 预计算每种偏置 key 对应的列集合（加速 mask 构建）----
        self._key_to_cols: dict[str, list[int]] = {}
        for col, key in enumerate(self._col_to_key):
            self._key_to_cols.setdefault(key, []).append(col)

        # 过滤掉无效（identity）的偏置，避免无谓计算
        self._active_biases: dict[str, TerrainCmdBias] = {
            k: v for k, v in sub_terrain_biases.items() if not v.is_identity()
        }

        self._valid = True
        print(
            f"[TerrainCmdBias] 初始化完成\n"
            f"  格子尺寸: {self._cell_size_x}m × {self._cell_size_y}m\n"
            f"  网格起点: x={self._x_start:.2f}, y={self._y_start:.2f}\n"
            f"  列切换保持步数: {self._switch_hold_steps}\n"
            f"  列→类型: {dict(enumerate(self._col_to_key))}\n"
            f"  激活偏置: { {k: v for k, v in self._active_biases.items()} }"
        )

    # -----------------------------------------------------------------------
    # 地形位置查询
    # -----------------------------------------------------------------------

    def get_col_indices(self, root_pos_w: torch.Tensor) -> torch.Tensor:
        """从世界坐标计算各 env 当前所在的列索引。

        Args:
            root_pos_w: (num_envs, 3) 机器人世界坐标。
        Returns:
            (num_envs,) int64 列索引，已 clamp 到 [0, num_cols-1]。
        """
        col = torch.floor((root_pos_w[:, 1] - self._y_start) / self._cell_size_y).long()
        return col.clamp_(0, self._num_cols - 1)

    def get_row_indices(self, root_pos_w: torch.Tensor) -> torch.Tensor:
        """从世界坐标计算各 env 当前所在的行索引。"""
        row = torch.floor((root_pos_w[:, 0] - self._x_start) / self._cell_size_x).long()
        return row.clamp_(0, self._num_rows - 1)

    def get_terrain_type_names(self, root_pos_w: torch.Tensor) -> list[str]:
        """返回每个 env 当前所在的 sub_terrain key 名称列表。"""
        col = self.get_col_indices(root_pos_w)
        return [self._col_to_key[int(c.item())] for c in col]

    def _apply_col_switch_hold(self, raw_col_indices: torch.Tensor) -> torch.Tensor:
        """对列索引应用最短保持步数防抖。"""
        if self._switch_hold_steps <= 0:
            return raw_col_indices

        # 首次或 env 数变化时重建状态
        if (
            self._stable_col_indices is None
            or self._candidate_col_indices is None
            or self._candidate_counts is None
            or self._stable_col_indices.shape[0] != raw_col_indices.shape[0]
        ):
            self._stable_col_indices = raw_col_indices.clone()
            self._candidate_col_indices = raw_col_indices.clone()
            self._candidate_counts = torch.zeros_like(raw_col_indices, dtype=torch.long, device=self.device)
            return self._stable_col_indices

        stable = self._stable_col_indices
        candidate = self._candidate_col_indices
        counts = self._candidate_counts

        changed = raw_col_indices != stable
        # 未变化时，重置候选与计数
        candidate = torch.where(changed, candidate, stable)
        counts = torch.where(changed, counts, torch.zeros_like(counts))

        # 变化时：若候选没变则计数+1，否则重置为新候选并从1开始计数
        same_candidate = raw_col_indices == candidate
        candidate = torch.where(changed, raw_col_indices, candidate)
        counts = torch.where(
            changed,
            torch.where(same_candidate, counts + 1, torch.ones_like(counts)),
            counts,
        )

        ready_switch = changed & (counts >= self._switch_hold_steps)
        stable = torch.where(ready_switch, candidate, stable)
        # 已切换后清空计数，候选回到稳定列
        counts = torch.where(ready_switch, torch.zeros_like(counts), counts)
        candidate = torch.where(ready_switch, stable, candidate)

        self._stable_col_indices = stable
        self._candidate_col_indices = candidate
        self._candidate_counts = counts
        return stable

    # -----------------------------------------------------------------------
    # 偏置应用
    # -----------------------------------------------------------------------

    def apply_bias(
        self,
        command: torch.Tensor,
        height_cmd: torch.Tensor,
        root_pos_w: torch.Tensor,
        cmd_ranges,
        height_range: list,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按地形类型对命令和高度指令应用偏置。

        Args:
            command:     (num_envs, 3) [vx, vy, omega_z]，不修改原张量。
            height_cmd:  (num_envs,)，不修改原张量。
            root_pos_w:  (num_envs, 3) 机器人世界坐标（实时读取）。
            cmd_ranges:  UniformVelocityCommandCfg.Ranges，提供 lin_vel_x/ang_vel_z 范围。
            height_range: [h_min, h_max]。
        Returns:
            (biased_command, biased_height_cmd) 新张量，原张量不变。
        """
        if not self._valid or not self._active_biases:
            return command, height_cmd

        cmd = command.clone()
        h_cmd = height_cmd.clone()
        raw_col_indices = self.get_col_indices(root_pos_w)
        col_indices = self._apply_col_switch_hold(raw_col_indices)

        vx_min = float(cmd_ranges.lin_vel_x[0])
        vx_max = float(cmd_ranges.lin_vel_x[1])
        h_min = float(height_range[0])
        h_max = float(height_range[1])

        for terrain_key, bias in self._active_biases.items():
            matching_cols = self._key_to_cols.get(terrain_key, [])
            if not matching_cols:
                continue

            # 向量化构建 mask（不对每个 env 做 Python 循环）
            mask = torch.zeros(cmd.shape[0], dtype=torch.bool, device=self.device)
            for mc in matching_cols:
                mask |= col_indices == mc
            if not mask.any():
                continue

            # 角速度缩放
            if bias.ang_vel_scale != 1.0:
                cmd[mask, 2] = cmd[mask, 2] * bias.ang_vel_scale

            # 线速度偏置：方向不变，幅值增大，clamp 到 range
            if bias.lin_vel_delta != 0.0:
                vx = cmd[mask, 0]
                cmd[mask, 0] = torch.clamp(
                    vx + torch.sign(vx) * bias.lin_vel_delta,
                    vx_min, vx_max,
                )

            # 线速度取正加正偏置
            if bias.lin_vel_delta_positive != 0.0:
                vx = cmd[mask, 0]
                # vx 是批量张量，不能用 Python if；用 where 做逐元素偏置
                # 语义：先取正(|vx|)，再加正偏置 delta（统一推向正向速度幅值）
                delta = float(bias.lin_vel_delta_positive)
                cmd[mask, 0] = torch.clamp(
                    vx.abs() + delta,
                    vx_min,
                    vx_max,
                )
            # 高度偏置
            if bias.height_delta != 0.0:
                h_cmd[mask] = torch.clamp(h_cmd[mask] + bias.height_delta, h_min, h_max)

        return cmd, h_cmd
