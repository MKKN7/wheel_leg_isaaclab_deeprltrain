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

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from agent_world import AssetPath
from agent_world.terrains import (
    FourQuadrantTerrainCfg,
    HfCliffInvertedPyramidStairsTerrainCfg,
    HfCustomDirectionalWaveTerrainCfg,
    HfCustomNpyTerrainCfg,
    HfCustomRaisedInvertedPyramidSlopedTerrainCfg,
    HfCustomTruncatedSlopedTerrainCfg,
    MeshCustomGridBarsTerrainCfg,
    HfCustomGridBarsTerrainCfg,
    MeshCustomSplitGridBarsTerrainCfg,
)


PLANE_FOR_RM = terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.1,
        )

PLANE_FOR_RM_ROT = terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,
        )

SLOPE_FOR_RM = terrain_gen.HfPyramidSlopedTerrainCfg( # 金字塔斜坡，中间平台四周斜坡
            proportion=0.1, 
            slope_range=(0.05, 0.3), # 斜坡斜率
            platform_width=2.0, # 中间平台宽度
            inverted=False, # True中间凹陷，False中间凸起
            border_width=1.
        )

INV_SLOPE_FOR_RM = terrain_gen.HfInvertedPyramidSlopedTerrainCfg( # 上一个的反
            proportion=0.1, 
            slope_range=(0.05, 0.3), 
            platform_width=2.0, 
            inverted=True,
            border_width=1.
        )

SLOPE_FOR_RM_LOW = terrain_gen.HfPyramidSlopedTerrainCfg( # 金字塔斜坡，中间平台四周斜坡
            proportion=0.1, 
            slope_range=(0.05, 0.15), # 斜坡斜率
            platform_width=0.0, # 中间平台宽度
            inverted=False, # True中间凹陷，False中间凸起
            border_width=1.
        )

SLOPE_FOR_RM_LOW_ROT = terrain_gen.HfPyramidSlopedTerrainCfg( # 金字塔斜坡，中间平台四周斜坡
            proportion=0.1, 
            slope_range=(0.05, 0.15), # 斜坡斜率
            platform_width=0.0, # 中间平台宽度
            inverted=False, # True中间凹陷，False中间凸起
            border_width=0.25
        )

SLOPE_FOR_RM_HIGH = terrain_gen.HfPyramidSlopedTerrainCfg( # 金字塔斜坡，中间平台四周斜坡
            proportion=0.1, 
            slope_range=(0.15, 0.35), # 斜坡斜率
            platform_width=2.0, # 中间平台宽度
            inverted=False, # True中间凹陷，False中间凸起
            border_width=2.
        )

INV_SLOPE_FOR_RM_LOW = terrain_gen.HfInvertedPyramidSlopedTerrainCfg( # 上一个的反
            proportion=0.1, 
            slope_range=(0.05, 0.15), 
            platform_width=0.0, 
            inverted=True,
            border_width=1.
        )

INV_SLOPE_FOR_RM_LOW_ROT = terrain_gen.HfInvertedPyramidSlopedTerrainCfg( # 上一个的反
            proportion=0.1, 
            slope_range=(0.05, 0.15), 
            platform_width=0.0, 
            inverted=True,
            border_width=0.25
        )

INV_SLOPE_FOR_RM_HIGH = terrain_gen.HfInvertedPyramidSlopedTerrainCfg( # 上一个的反
            proportion=0.1, 
            slope_range=(0.15, 0.3), 
            platform_width=2.0, 
            inverted=True,
            border_width=1.
        )

WAVE_FOR_RM = terrain_gen.HfWaveTerrainCfg(
            proportion=0.1,
            amplitude_range=(0.04,0.08), # 波形的振幅范围
            num_waves=int(5./0.24), # 波浪的个数，波长=地形长度/波浪个数
            border_width=1.
        )

RANDOM_UNIFORM_FOR_RM = terrain_gen.HfRandomUniformTerrainCfg( # 随机均匀地形
            proportion=0.1, # 出现概率
            noise_range=(0.0, 0.03), # 最低和最高
            noise_step=0.005, # 两点之间最低高度变化，不能小于 terrain vertical_scale
            downsampled_scale=0.2, # 两点之间最近距离， None则采用horizontal_scale
            border_width=1., # 环绕地形的平地距离
        )

RANDOM_UNIFORM_FOR_RM_ROT = terrain_gen.HfRandomUniformTerrainCfg( # 随机均匀地形
            proportion=0.1, # 出现概率
            noise_range=(0.0, 0.03), # 最低和最高
            noise_step=0.005, # 两点之间最低高度变化，不能小于 terrain vertical_scale
            downsampled_scale=0.2, # 两点之间最近距离， None则采用horizontal_scale
            border_width=0.25, # 环绕地形的平地距离
        )

STAIR_SLOPE_FOR_RM = terrain_gen.HfPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.01,0.035), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=2.0,
            border_width=1.,
        )

STAIR_SLOPE_FOR_RM_LOW = terrain_gen.HfPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.005,0.015), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=1.0,
            border_width=1.,
        )

STAIR_SLOPE_FOR_RM_LOW_ROT = terrain_gen.HfPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.005,0.015), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=1.0,
            border_width=0.25,
        )

STAIR_SLOPE_FOR_RM_MID = terrain_gen.HfPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.2,
            # step_height_range=(0.015,0.04), # 楼梯高度范围
            step_height_range=(0.025,0.04), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=1.0,
            border_width=1.0,
        )

STAIR_SLOPE_FOR_RM_MID_ROT = terrain_gen.HfPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.015,0.03), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=0.5,
            border_width=0.25,
        )

STAIR_SLOPE_FOR_RM_HIGH = terrain_gen.HfPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.015,0.032), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=1.0,
            border_width=1.,
        )

STAIR_SLOPE_FOR_RM_FORT = terrain_gen.HfPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.02,0.043), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=1.0,
            border_width=0.25,
        )

INV_STAIR_SLOPE_FOR_RM = terrain_gen.HfInvertedPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.05,
            step_height_range=(0.01,0.035), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=2.0,
            inverted=True,
            border_width=1.,
        )

INV_STAIR_SLOPE_FOR_RM_LOW = terrain_gen.HfInvertedPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.005,0.015), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=2.0,
            inverted=True,
            border_width=1.,
        )

INV_STAIR_SLOPE_FOR_RM_LOW_ROT = terrain_gen.HfInvertedPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.005,0.015), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=0.5,
            inverted=True,
            border_width=0.25,
        )

INV_STAIR_SLOPE_FOR_RM_MID = terrain_gen.HfInvertedPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.2,
            # step_height_range=(0.015,0.04), # 楼梯高度范围
            step_height_range=(0.025,0.04), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=1.0,
            inverted=True,
            border_width=1.0,
        )

INV_STAIR_SLOPE_FOR_RM_MID_ROT = terrain_gen.HfInvertedPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.015,0.03), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=0.5,
            inverted=True,
            border_width=0.25,
        )

INV_STAIR_SLOPE_FOR_RM_HIGH = terrain_gen.HfInvertedPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.2,
            step_height_range=(0.015,0.032), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=2.0,
            inverted=True,
            border_width=1.,
        )

CLIFF_INV_STAIR_SLOPE_FOR_RM = HfCliffInvertedPyramidStairsTerrainCfg( # 断崖反金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.01,0.025), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=3.0,
            height_offset_range=(0.12, 0.22),
            border_width=2.,
        )

CLIFF_INV_STAIR_SLOPE_TALL_FOR_RM = HfCliffInvertedPyramidStairsTerrainCfg( # 断崖反金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.015,0.032), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=3.0,
            height_offset_range=(0.3, 0.4),
            border_width=2.,
        )

CLIFF_INV_STAIR_SLOPE_SHORT_FOR_RM = HfCliffInvertedPyramidStairsTerrainCfg( # 断崖反金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.025,0.032), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=3.0,
            height_offset_range=(0.3, 0.4),
            border_width=2.,
        )

CLIFF_INV_STAIR_SLOPE_LONG_FOR_RM = HfCliffInvertedPyramidStairsTerrainCfg( # 断崖反金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.010,0.032), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=2.0,
            height_offset_range=(0.3, 0.4),
            border_width=2.0,
        )

CLIFF_INV_STAIR_SLOPE_FLAT_FOR_RM = HfCliffInvertedPyramidStairsTerrainCfg( # 断崖反金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.01,0.032), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=3.0,
            height_offset_range=(0.0, 0.0),
            border_width=2.,
        )

CLIFF_INV_STAIR_SLOPE_SHORT_FOR_RM_PLAY = HfCliffInvertedPyramidStairsTerrainCfg( # 断崖反金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.03,0.03), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=4.0,
            height_offset_range=(0.3, 0.4),
            border_width=1.,
        )

# Backward-compatible aliases for historical names.
RAMDOM_FOR_RM = RANDOM_UNIFORM_FOR_RM
INV_STAIR_SLOPE_FOR_RM = CLIFF_INV_STAIR_SLOPE_FOR_RM
INV_STAIR_SLOPE_FOR_RM_2 = CLIFF_INV_STAIR_SLOPE_TALL_FOR_RM
INV_STAIR_SLOPE_FOR_RM_3 = CLIFF_INV_STAIR_SLOPE_FLAT_FOR_RM
INV_STAIR_SLOPE_FOR_RM_4 = INV_STAIR_SLOPE_FOR_RM


STAIR_FOR_RM = terrain_gen.HfPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.20,0.22), # 楼梯高度范围
            step_width=1.5, # 楼梯宽度范围
            platform_width=2.0,
            border_width=0.50,
            # border_width=1.50,
        )

STAIR_FOR_RM1 = terrain_gen.HfPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.3,
            step_height_range=(0.18,0.22), # 楼梯高度范围
            step_width=1.5, # 楼梯宽度范围
            platform_width=2.0,
            border_width=0.50,
            # border_width=1.50,
        )

INV_STAIR_FOR_RM = terrain_gen.HfInvertedPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.20,0.22), # 楼梯高度范围
            step_width=1., # 楼梯宽度范围
            # platform_width=3.0,
            platform_width=4.0,
            inverted=True,
            border_width=0.5,
            # border_width=1.5,
        )

INV_STAIR_FOR_RM1 = terrain_gen.HfInvertedPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.3,
            step_height_range=(0.18,0.22), # 楼梯高度范围
            step_width=1., # 楼梯宽度范围
            # platform_width=3.0,
            platform_width=4.0,
            inverted=True,
            border_width=0.5,
            # border_width=1.5,
        )

# 定义自定义飞坡瓦片（MeshImportTerrainCfg 在当前 IsaacLab 版本中不存在，暂时禁用）
# FLYING_SLOPE_CUSTOM = terrain_gen.MeshImportTerrainCfg(
#     usd_path="{AssetPath}/usd_files/RMUC2026/RL_FLY_COLLISION.usd",
#     proportion=0.3,
#     collision_props=terrain_gen.MeshCollisionPropertiesCfg(
#         mesh_approximation="triangle_mesh",
#     ),
# )

# -------------------------------------------------------------------------
# USD 地形配置：直接导入 USD 文件作为整体地形
# terrain_type="usd" 时所有机器人共享同一张地形，通过 env_spacing 网格排列
# physics_material 参数在 usd 模式下不生效（由 USD 文件本身携带物理属性）
# -------------------------------------------------------------------------
RMUC2026_USD_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="usd",
    collision_group=-1,
    usd_path=f"{AssetPath}/usd_files/RMUC2026/RMUC2026_PHYSICS.usd",
    env_spacing=3.0,   # 机器人之间的网格间距（米），可根据实际地形大小调整
    debug_vis=False,
)

RMUC2026_FLY_SLOPE_USD_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="usd",
    collision_group=-1,
    usd_path=f"{AssetPath}/usd_files/RMUC2026/RL_FLY.usd",
    env_spacing=0.0,
    debug_vis=False,
)

HIGH_STAIR_FOR_RM = terrain_gen.HfPyramidStairsTerrainCfg(
    proportion=0.1,
    step_height_range=(0.15, 0.35), 
    step_width=1.25,        # 台阶宽度变得不重要
    platform_width=2.5,    # 几乎占满整个 7m 空间（扣除 border）
    border_width=0.,
)

LOW_SPEED_STAIR_FOR_RM = terrain_gen.HfPyramidStairsTerrainCfg(
    proportion=0.1,
    step_height_range=(0.15, 0.35), 
    step_width=1.5,        # 台阶宽度变得不重要
    platform_width=3.0,    # 几乎占满整个 7m 空间（扣除 border）
    border_width=0.,
)

STAIR_FOR_RM_ROT = terrain_gen.HfPyramidStairsTerrainCfg(
    proportion=0.1,
    step_height_range=(0.15, 0.25), 
    step_width=1.,        # 台阶宽度变得不重要
    platform_width=1.5,    # 几乎占满整个 7m 空间（扣除 border）
    border_width=0.,
)

HIGH_SPEED_STAIR_FOR_RM = terrain_gen.HfPyramidStairsTerrainCfg(
    proportion=0.1,
    step_height_range=(0.15, 0.40), 
    step_width=1.5,        # 台阶宽度变得不重要
    platform_width=3.0,    # 几乎占满整个 7m 空间（扣除 border）
    border_width=0.,
)

# 自定义飞坡地形（从 NPY 文件加载）
FLYING_SLOPE_FOR_RM = HfCustomNpyTerrainCfg(
    proportion=0.2,
    npy_path=f"{AssetPath}/usd_files/RMUC2026/RL-FLY-7m.npy",
    horizontal_scale=0.01,
    vertical_scale=0.005,
    border_width=0.0,
)
# 自定义大波浪
BIGWAVE_SLOPE_FOR_RM = HfCustomNpyTerrainCfg(
    proportion=0.2,
    npy_path=f"{AssetPath}/usd_files/RMUC2026/RL-BIGWAVE-7m.npy",
    horizontal_scale=0.01,
    vertical_scale=0.01,
    border_width=0.0,
        )

wave_cfg = HfCustomDirectionalWaveTerrainCfg(
    proportion=0.5,
    amplitude=0.035,
    frequency=4.16,
    axis="x",
    phase=0.0,
    border_width=0.0,
    horizontal_scale=0.01,
    vertical_scale=0.01,
)

slope_cfg = HfCustomTruncatedSlopedTerrainCfg(
    proportion=0.5,
    # slope_angle_deg=17.0,
    angle_range=(15.0, 19.0), # 斜坡角度范围
    # ramp_length=1.2,
    ramp_length_range=(1.1, 1.3), # 斜坡长度范围
    randomize_each_ramp=True,
    axis="x",
    centered=False,
    bias=1.0,
    num_ramps=2,
    ramp_spacing=-3.5,
    border_width=0.25,
    horizontal_scale=0.1, # 每一小块地形采样的长宽
    vertical_scale=0.005, # 每一小块地形的高度变化
)


# FLYING_SLOPE_FOR_RM = HfCustomRaisedInvertedPyramidSlopedTerrainCfg(
#     # size=(8.0, 8.0),
#     proportion=0.1,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     border_width=0.0,
#     raised_height_range=(0.2, 0.8),
#     platform_width=1.5,
#     angle_range=(8.0, 14.0),
#     ramp_length_range=(1.0, 1.2),
# )
# FLYING_SLOPE_FOR_RM_2 = HfCustomRaisedInvertedPyramidSlopedTerrainCfg(
#     # size=(8.0, 8.0),
#     proportion=0.1,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     border_width=0.0,
#     raised_height_range=(0.2, 0.8),
#     platform_width=1.5,
#     angle_range=(8.0, 14.0),
#     ramp_length_range=(1.0, 1.2),
# )

# 自定义井字小台阶风格
TINY_STEP_FOR_RM = MeshCustomGridBarsTerrainCfg(
    proportion=0.2,
    num_horizontal_range=(2, 4),
    num_vertical_range=(2, 4),
    randomize_bar_count_difficulty=True,
    force_unequal_counts=False,
    force_even_counts=True,
    bar_width_range=(0.05, 0.2),
    randomize_bar_width_difficulty=True,
    bar_height_range=(0.02, 0.06),
)

TINY_STEP_FOR_RM_ROT = MeshCustomGridBarsTerrainCfg(
    proportion=0.2,
    num_horizontal_range=(2, 4),
    num_vertical_range=(2, 4),
    randomize_bar_count_difficulty=True,
    force_unequal_counts=False,
    force_even_counts=True,
    bar_width_range=(0.05, 0.2),
    randomize_bar_width_difficulty=True,
    bar_height_range=(0.01, 0.03),
)

TINY_SPLIT_STEP_FOR_RM = MeshCustomSplitGridBarsTerrainCfg(
    proportion=0.2,
    num_horizontal_range=(2, 4),
    num_vertical_range=(2, 4),
    randomize_bar_count_difficulty=True,
    force_unequal_counts=False,
    force_even_counts=True,
    bar_width_range=(0.05, 0.2),
    randomize_bar_width_difficulty=True,
    bar_height_range=(0.02, 0.06),
    split_ratio=0.35,
    horizontal_side="positive",
    randomize_side=True,
)

FOUR_QUAD_SLOPE_FOR_RM = FourQuadrantTerrainCfg(
    proportion=0.1,
    quadrants=STAIR_SLOPE_FOR_RM_FORT,
)

FOUR_QUAD_MIX_FOR_RM = FourQuadrantTerrainCfg(
    proportion=0.1,
    quadrants={
        "front_left": SLOPE_FOR_RM_LOW,
        "front_right": INV_SLOPE_FOR_RM_LOW,
        "rear_left": STAIR_SLOPE_FOR_RM_LOW,
        "rear_right": RANDOM_UNIFORM_FOR_RM,
    },
)


# RM_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
#     seed=None, # 随机种子
#     curriculum=False, # 如果True，则会根据难度参数生成地形
#     size=(9., 9.), # 每一块地形的尺寸
#     border_width=200.0, # 每一块地形环绕的分隔距离
#     border_height=1.0, # 每一块地形边缘的维兰高度，正数，围栏在ground下面，负数，围栏在ground上面
#     num_rows=20, # 子地形的行数
#     num_cols=20, # 子地形的列数
#     color_scheme='none', # 地形颜色，'height'颜色根据地形的高度变化，'random'颜色随机，'none'没有颜色
#     horizontal_scale=0.1, # 每一小块地形采样的长宽
#     vertical_scale=0.005, # 每一小块地形的高度变化
#     slope_threshold=0.5, # 斜面变垂直面的阈值
#     difficulty_range=(0.0,1.0), # 如果启用curriculum，则会按难度从低到高依次生成，否则会在这个范围内随机采样
#     use_cache=False, # 是否从缓存中导入子地形
#     cache_dir="/tmp/isaaclab/terrains", # 缓存地址
#     sub_terrains={
#         'cliff_inv_stair_slope_for_rm1': CLIFF_INV_STAIR_SLOPE_FOR_RM,
#         'plane_for_rm': PLANE_FOR_RM,
#         'cliff_inv_stair_slope_flat_for_rm': CLIFF_INV_STAIR_SLOPE_FLAT_FOR_RM,
#         'cliff_inv_stair_slope_for_rm2': CLIFF_INV_STAIR_SLOPE_FOR_RM,
#         'slope_for_rm': SLOPE_FOR_RM,
#         'inv_slope_for_rm': INV_SLOPE_FOR_RM,
#         # 'random_uniform_for_rm': RANDOM_UNIFORM_FOR_RM,
#         # 'cliff_inv_stair_slope_tall_for_rm': CLIFF_INV_STAIR_SLOPE_TALL_FOR_RM,
#         'cliff_inv_stair_slope_for_rm3': CLIFF_INV_STAIR_SLOPE_FOR_RM,
#         'stair_slope_for_rm': STAIR_SLOPE_FOR_RM,
#         'stair_for_rm': STAIR_FOR_RM,
#         # 'high_stair_for_rm': HIGH_STAIR_FOR_RM,
#         # 'inv_stair_slope_for_rm': INV_STAIR_SLOPE_FOR_RM,
#         'inv_stair_for_rm': INV_STAIR_FOR_RM,
#         # 'big_wave': BIGWAVE_SLOPE_FOR_RM,
#         # 'fly_slope': FLYING_SLOPE_FOR_RM,
#         'tiny_step': TINY_STEP_FOR_RM,
#         'cliff_inv_stair_slope_for_rm4': CLIFF_INV_STAIR_SLOPE_FOR_RM,
#     },
# )

# RM_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
#     seed=None, # 随机种子
#     curriculum=False, # 如果True，则会根据难度参数生成地形
#     size=(9., 9.), # 每一块地形的尺寸
#     border_width=200.0, # 每一块地形环绕的分隔距离
#     border_height=1.0, # 每一块地形边缘的维兰高度，正数，围栏在ground下面，负数，围栏在ground上面
#     num_rows=20, # 子地形的行数
#     num_cols=20, # 子地形的列数
#     color_scheme='none', # 地形颜色，'height'颜色根据地形的高度变化，'random'颜色随机，'none'没有颜色
#     horizontal_scale=0.1, # 每一小块地形采样的长宽
#     vertical_scale=0.005, # 每一小块地形的高度变化
#     slope_threshold=0.5, # 斜面变垂直面的阈值
#     difficulty_range=(0.0,1.0), # 如果启用curriculum，则会按难度从低到高依次生成，否则会在这个范围内随机采样
#     use_cache=False, # 是否从缓存中导入子地形
#     cache_dir="/tmp/isaaclab/terrains", # 缓存地址
#     sub_terrains={
#         # 'cliff_inv_stair_slope_for_rm1': CLIFF_INV_STAIR_SLOPE_FOR_RM,
#         'plane_for_rm': PLANE_FOR_RM,
#         # 'cliff_inv_stair_slope_flat_for_rm': CLIFF_INV_STAIR_SLOPE_FLAT_FOR_RM,
#         # 'cliff_inv_stair_slope_for_rm2': CLIFF_INV_STAIR_SLOPE_FOR_RM,
#         # 'slope_for_rm': SLOPE_FOR_RM,
#         # 'inv_slope_for_rm': INV_SLOPE_FOR_RM,
#         # 'random_uniform_for_rm': RANDOM_UNIFORM_FOR_RM,
#         # 'cliff_inv_stair_slope_tall_for_rm': CLIFF_INV_STAIR_SLOPE_TALL_FOR_RM,
#         # 'cliff_inv_stair_slope_for_rm3': CLIFF_INV_STAIR_SLOPE_FOR_RM,
#         # 'stair_slope_for_rm': STAIR_SLOPE_FOR_RM,
#         'stair_for_rm': STAIR_FOR_RM,
#         # 'high_stair_for_rm': HIGH_STAIR_FOR_RM,
#         # 'inv_stair_slope_for_rm': INV_STAIR_SLOPE_FOR_RM,
#         'inv_stair_for_rm': INV_STAIR_FOR_RM,
#         # 'big_wave': BIGWAVE_SLOPE_FOR_RM,
#         # 'fly_slope': FLYING_SLOPE_FOR_RM,
#         # 'tiny_step': TINY_STEP_FOR_RM,
#         # 'cliff_inv_stair_slope_for_rm4': CLIFF_INV_STAIR_SLOPE_FOR_RM,
#     },
# )

RM_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=None, # 随机种子
    curriculum=False, # 如果True，则会根据难度参数生成地形
    size=(9., 9.), # 每一块地形的尺寸
    border_width=10.0, # 每一块地形环绕的分隔距离
    border_height=1.0, # 每一块地形边缘的维兰高度，正数，围栏在ground下面，负数，围栏在ground上面
    num_rows=10, # 子地形的行数
    num_cols=13, # 子地形的列数
    color_scheme='none', # 地形颜色，'height'颜色根据地形的高度变化，'random'颜色随机，'none'没有颜色
    horizontal_scale=0.1, # 每一小块地形采样的长宽
    vertical_scale=0.005, # 每一小块地形的高度变化
    slope_threshold=0.5, # 斜面变垂直面的阈值
    difficulty_range=(0.0,1.0), # 如果启用curriculum，则会按难度从低到高依次生成，否则会在这个范围内随机采样
    use_cache=False, # 是否从缓存中导入子地形
    cache_dir="/tmp/isaaclab/terrains", # 缓存地址
    sub_terrains={
        'low_speed_stair_for_rm': LOW_SPEED_STAIR_FOR_RM,
        'tiny_step': TINY_STEP_FOR_RM,
        # 'inv_stair_for_rm': INV_STAIR_FOR_RM,
        # 'slope_for_rm_low': SLOPE_FOR_RM_LOW,
        'slope_for_rm_high': SLOPE_FOR_RM_HIGH,
        'inv_slope_for_rm_low': INV_SLOPE_FOR_RM_LOW,
        # 'inv_slope_for_rm_high': INV_SLOPE_FOR_RM_HIGH,
        # 'stair_for_rm': STAIR_FOR_RM,
        # 'high_stair_for_rm': HIGH_STAIR_FOR_RM,
        'high_speed_stair_for_rm' : HIGH_SPEED_STAIR_FOR_RM,
        # 'stair_slope_for_rm_low': STAIR_SLOPE_FOR_RM_LOW,
        'stair_slope_for_rm_high': STAIR_SLOPE_FOR_RM_HIGH,
        'inv_stair_slope_for_rm_low': INV_STAIR_SLOPE_FOR_RM_LOW,
        'inv_stair_slope_for_rm_high': INV_STAIR_SLOPE_FOR_RM_HIGH,
        'plane_for_rm': PLANE_FOR_RM,
        # 'cliff_inv_stair_slope_short_for_rm': CLIFF_INV_STAIR_SLOPE_SHORT_FOR_RM,
        # 'cliff_inv_stair_slope_for_rm': CLIFF_INV_STAIR_SLOPE_FOR_RM,
        'random_uniform_for_rm': RANDOM_UNIFORM_FOR_RM,
        # 'cliff_inv_stair_slope_tall_for_rm': CLIFF_INV_STAIR_SLOPE_TALL_FOR_RM,
        # 'cliff_inv_stair_slope_long_for_rm': CLIFF_INV_STAIR_SLOPE_LONG_FOR_RM,
        'cliff_inv_stair_slope_short_for_rm': CLIFF_INV_STAIR_SLOPE_SHORT_FOR_RM,
    },
)

RM_ROTATION_TERRAINS_CFG_66 = TerrainGeneratorCfg(
    seed=None, # 随机种子
    curriculum=True, # 如果True，则会根据难度参数生成地形
    size=(6., 6.), # 每一块地形的尺寸
    border_width=5.0, # 每一块地形环绕的分隔距离
    border_height=1.0, # 每一块地形边缘的维兰高度，正数，围栏在ground下面，负数，围栏在ground上面
    num_rows=10, # 子地形的行数
    num_cols=14, # 子地形的列数
    color_scheme='none', # 地形颜色，'height'颜色根据地形的高度变化，'random'颜色随机，'none'没有颜色
    horizontal_scale=0.1, # 每一小块地形采样的长宽
    vertical_scale=0.005, # 每一小块地形的高度变化
    slope_threshold=0.5, # 斜面变垂直面的阈值
    difficulty_range=(0.0,1.0), # 如果启用curriculum，则会按难度从低到高依次生成，否则会在这个范围内随机采样
    use_cache=False, # 是否从缓存中导入子地形
    cache_dir="/tmp/isaaclab/terrains", # 缓存地址
    sub_terrains={
        'tiny_step_rot': TINY_STEP_FOR_RM_ROT,
        'slope_for_rm_low': SLOPE_FOR_RM_LOW_ROT,
        'inv_slope_for_rm_low': INV_SLOPE_FOR_RM_LOW_ROT,
        'stair_slope_for_rm_low': STAIR_SLOPE_FOR_RM_LOW_ROT,
        'stair_slope_for_rm_mid': STAIR_SLOPE_FOR_RM_MID_ROT,
        'inv_stair_slope_for_rm_low': INV_STAIR_SLOPE_FOR_RM_LOW_ROT,
        'inv_stair_slope_for_rm_mid': INV_STAIR_SLOPE_FOR_RM_MID_ROT,
        'plane_for_rm_rot': PLANE_FOR_RM_ROT,
        'stair_for_rm_rot': STAIR_FOR_RM_ROT,
        'random_uniform_for_rm': RANDOM_UNIFORM_FOR_RM_ROT,
        'fort_for_rm': FOUR_QUAD_SLOPE_FOR_RM,
    },
)

RM_ROTATION_TERRAINS_CFG_99 = TerrainGeneratorCfg(
    seed=None, # 随机种子
    curriculum=True, # 如果True，则会根据难度参数生成地形
    size=(9., 9.), # 每一块地形的尺寸
    border_width=5.0, # 每一块地形环绕的分隔距离
    border_height=1.0, # 每一块地形边缘的维兰高度，正数，围栏在ground下面，负数，围栏在ground上面
    num_rows=10, # 子地形的行数
    num_cols=10, # 子地形的列数
    color_scheme='none', # 地形颜色，'height'颜色根据地形的高度变化，'random'颜色随机，'none'没有颜色
    horizontal_scale=0.1, # 每一小块地形采样的长宽
    vertical_scale=0.005, # 每一小块地形的高度变化
    slope_threshold=0.5, # 斜面变垂直面的阈值
    difficulty_range=(0.0,1.0), # 如果启用curriculum，则会按难度从低到高依次生成，否则会在这个范围内随机采样
    use_cache=False, # 是否从缓存中导入子地形
    cache_dir="/tmp/isaaclab/terrains", # 缓存地址
    sub_terrains={
        'tiny_step_rot': TINY_STEP_FOR_RM_ROT,
        'slope_for_rm_low': SLOPE_FOR_RM_LOW,
        'inv_slope_for_rm_low': INV_SLOPE_FOR_RM_LOW,
        'stair_slope_for_rm_low': STAIR_SLOPE_FOR_RM_LOW,
        # 'stair_slope_for_rm_mid': STAIR_SLOPE_FOR_RM_MID,
        'inv_stair_slope_for_rm_low': INV_STAIR_SLOPE_FOR_RM_LOW,
        # 'inv_stair_slope_for_rm_mid': INV_STAIR_SLOPE_FOR_RM_MID,
        'plane_for_rm_rot': PLANE_FOR_RM_ROT,
        'random_uniform_for_rm': RANDOM_UNIFORM_FOR_RM,
        # 'stair_for_rm_rot': LOW_SPEED_STAIR_FOR_RM,
    },
)

RM_ROUGH_STAIRS_CFG = TerrainGeneratorCfg(
    seed=None, # 随机种子
    curriculum=False, # 如果True，则会根据难度参数生成地形
    size=(9., 9.), # 每一块地形的尺寸
    border_width=10.0, # 每一块地形环绕的分隔距离
    border_height=1.0, # 每一块地形边缘的维兰高度，正数，围栏在ground下面，负数，围栏在ground上面
    num_rows=10, # 子地形的行数
    num_cols=13, # 子地形的列数
    color_scheme='none', # 地形颜色，'height'颜色根据地形的高度变化，'random'颜色随机，'none'没有颜色
    horizontal_scale=0.1, # 每一小块地形采样的长宽
    vertical_scale=0.005, # 每一小块地形的高度变化
    slope_threshold=0.5, # 斜面变垂直面的阈值
    difficulty_range=(0.0,1.0), # 如果启用curriculum，则会按难度从低到高依次生成，否则会在这个范围内随机采样
    use_cache=False, # 是否从缓存中导入子地形
    cache_dir="/tmp/isaaclab/terrains", # 缓存地址
    sub_terrains={
        'inv_stair_for_rm': INV_STAIR_FOR_RM1,
        'slope_for_rm_low': SLOPE_FOR_RM_LOW,
        # 'slope_for_rm_high': SLOPE_FOR_RM_HIGH,
        'inv_slope_for_rm_low': INV_SLOPE_FOR_RM_LOW,
        # 'inv_slope_for_rm_high': INV_SLOPE_FOR_RM_HIGH,
        'stair_for_rm': STAIR_FOR_RM1,
        'stair_slope_for_rm_low': STAIR_SLOPE_FOR_RM_LOW,
        # 'stair_slope_for_rm_high': STAIR_SLOPE_FOR_RM_HIGH,
        'inv_stair_slope_for_rm_low': INV_STAIR_SLOPE_FOR_RM_LOW,
        # 'inv_stair_slope_for_rm_high': INV_STAIR_SLOPE_FOR_RM_HIGH,
        'plane_for_rm': PLANE_FOR_RM,
        'cliff_inv_stair_slope_for_rm': CLIFF_INV_STAIR_SLOPE_FOR_RM,
        'random_uniform_for_rm': RANDOM_UNIFORM_FOR_RM,
        # 'cliff_inv_stair_slope_tall_for_rm': CLIFF_INV_STAIR_SLOPE_TALL_FOR_RM,
    },
)

RM_ROUGH_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=None, # 随机种子
    curriculum=False, # 如果True，则会根据难度参数生成地形
    size=(9., 9.), # 每一块地形的尺寸
    border_width=10.0, # 每一块地形环绕的分隔距离
    border_height=1.0, # 每一块地形边缘的维兰高度，正数，围栏在ground下面，负数，围栏在ground上面
    num_rows=1, # 子地形的行数
    num_cols=1, # 子地形的列数
    color_scheme='none', # 地形颜色，'height'颜色根据地形的高度变化，'random'颜色随机，'none'没有颜色
    horizontal_scale=0.1, # 每一小块地形采样的长宽
    vertical_scale=0.005, # 每一小块地形的高度变化
    slope_threshold=0.5, # 斜面变垂直面的阈值
    difficulty_range=(0.0,1.0), # 如果启用curriculum，则会按难度从低到高依次生成，否则会在这个范围内随机采样
    use_cache=False, # 是否从缓存中导入子地形
    cache_dir="/tmp/isaaclab/terrains", # 缓存地址
    sub_terrains={
        # 'plane_for_rm': PLANE_FOR_RM,
        # 'random_uniform_for_rm': RANDOM_UNIFORM_FOR_RM,
        # # 'slope_for_rm_low': SLOPE_FOR_RM_LOW,
        # 'slope_for_rm_high': SLOPE_FOR_RM_HIGH,
        # # 'inv_slope_for_rm_low': INV_SLOPE_FOR_RM_LOW,
        # 'inv_slope_for_rm_high': INV_SLOPE_FOR_RM_HIGH,
        # # 'stair_slope_for_rm_low': STAIR_SLOPE_FOR_RM_LOW,
        # 'stair_slope_for_rm_high': STAIR_SLOPE_FOR_RM_HIGH,
        # # 'inv_stair_slope_for_rm_low': INV_STAIR_SLOPE_FOR_RM_LOW,
        # 'inv_stair_slope_for_rm_high': INV_STAIR_SLOPE_FOR_RM_HIGH,
        # 'cliff_inv_stair_slope_for_rm': CLIFF_INV_STAIR_SLOPE_FOR_RM,
        # 'cliff_inv_stair_slope_tall_for_rm': CLIFF_INV_STAIR_SLOPE_TALL_FOR_RM,
        # 'stair_for_rm': STAIR_FOR_RM,
        # 'inv_stair_for_rm': INV_STAIR_FOR_RM,
        'cliff_inv_stair_slope_short_for_rm_play': CLIFF_INV_STAIR_SLOPE_SHORT_FOR_RM_PLAY,
    },
)


# Backward compatibility: existing rough envs historically referenced RM_TERRAINS_CFG.
RM_TERRAINS_CFG = RM_ROUGH_TERRAINS_CFG
