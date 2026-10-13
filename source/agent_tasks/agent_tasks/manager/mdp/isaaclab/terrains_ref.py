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
    HfCliffInvertedPyramidStairsTerrainCfg,
    HfCustomDirectionalWaveTerrainCfg,
    HfCustomNpyTerrainCfg,
    HfCustomRadialStairSlopeTerrainCfg,
    HfCustomRaisedInvertedPyramidSlopedTerrainCfg,
    HfCustomTruncatedSlopedTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfTwoStepDepressedPlatformTerrainCfg,
    HfWaveTerrainCfg,
    MeshCustomGridBarsTerrainCfg,
    HfCustomGridBarsTerrainCfg,
    MeshCustomSplitGridBarsTerrainCfg,
)


PLANE_FOR_RM = terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,
        )

HIGH_SPIN_FOR_RM = terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.1,
        )

SLOPE_FOR_RM = HfPyramidSlopedTerrainCfg( # 金字塔斜坡，中间平台四周斜坡
            proportion=0.15,
            slope_range=(0.05, 0.25), # 斜坡斜率
            platform_width=2.0, # 中间平台宽度
            inverted=False, # True中间凹陷，False中间凸起
            border_width=0.25
        )

INV_SLOPE_FOR_RM = HfInvertedPyramidSlopedTerrainCfg( # 上一个的反
            proportion=0.1,
            slope_range=(0.05, 0.25),
            platform_width=3.0, 
            inverted=True,
            border_width=0.25
        )

WAVE_FOR_RM = HfWaveTerrainCfg(
            proportion=0.1,
            amplitude_range=(0.04,0.08), # 波形的振幅范围
            num_waves=int(5./0.24), # 波浪的个数，波长=地形长度/波浪个数
            border_width=0.25
        )

RANDOM_UNIFORM_FOR_RM = HfRandomUniformTerrainCfg( # 随机均匀地形
            proportion=0.05, # 出现概率
            noise_range=(0.0, 0.05), # 最低和最高
            noise_step=0.005, # 两点之间最低高度变化
            downsampled_scale=0.2, # 两点之间最近距离， None则采用horizontal_scale
            border_width=0.25, # 环绕地形的平地距离
        )

STAIR_SLOPE_FOR_RM = HfPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.05,
            step_height_range=(0.025,0.035), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=3.0,
            border_width=0.5,
        )

CLIFF_INV_STAIR_SLOPE_FOR_RM = HfCliffInvertedPyramidStairsTerrainCfg( # 断崖反金字塔阶梯，中间平台四周台阶
            proportion=0.15,
            step_height_range=(0.025,0.032), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=4.0,
            height_offset_range=(0.3, 0.4),
            mask_size=1.5,
            mask_height=1.0,
            border_width=1.0,
        )

CLIFF_INV_STAIR_SLOPE_TALL_FOR_RM = HfCliffInvertedPyramidStairsTerrainCfg( # 断崖反金字塔阶梯，中间平台四周台阶
            proportion=0.15,
            step_height_range=(0.010,0.032), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=2.0,
            height_offset_range=(0.3, 0.4),
            mask_size=1.5,
            mask_height=1.0,
            border_width=1.5,
        )

CLIFF_INV_STAIR_SLOPE_FLAT_FOR_RM = HfCliffInvertedPyramidStairsTerrainCfg( # 断崖反金字塔阶梯，中间平台四周台阶
            proportion=0.25,
            step_height_range=(0.015,0.032), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=1.0,
            height_offset_range=(0.0, 0.0),
            # mask_size=2.0,
            # mask_height=1.0,
            border_width=2.0,
        )

INV_PYRAMID_STAIR_SLOPE_FOR_RM = HfInvertedPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.05,
            step_height_range=(0.028,0.032), # 楼梯高度范围
            step_width=0.1, # 楼梯宽度范围
            platform_width=2.6,
            inverted=True,
            border_width=0.5,
        )

# Backward-compatible aliases for historical names.
RAMDOM_FOR_RM = RANDOM_UNIFORM_FOR_RM
INV_STAIR_SLOPE_FOR_RM = CLIFF_INV_STAIR_SLOPE_FOR_RM
INV_STAIR_SLOPE_FOR_RM_2 = CLIFF_INV_STAIR_SLOPE_TALL_FOR_RM
INV_STAIR_SLOPE_FOR_RM_3 = CLIFF_INV_STAIR_SLOPE_FLAT_FOR_RM
INV_STAIR_SLOPE_FOR_RM_4 = INV_PYRAMID_STAIR_SLOPE_FOR_RM
RADIAL_STAIR_SLOPE_FOR_RM = HfCustomRadialStairSlopeTerrainCfg(
            proportion=0.1,
            bottom_radius=1.0, # 坡底到中心距离
            ramp_length=1.2, # 坡的径向长度
            slope_angle_deg=17.0, # 坡面角度
            direction="outward",
            border_width=0.25,
            horizontal_scale=0.05,
            vertical_scale=0.005,
        )


STAIR_FOR_RM = HfPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.05,
            step_height_range=(0.15,0.21), # 楼梯高度范围
            step_width=2.5, # 楼梯宽度范围
            platform_width=2.0,
            border_width=0.0,
        )

STAIR_FOR_RM_2 = HfPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            proportion=0.1,
            step_height_range=(0.15,0.21), # 楼梯高度范围
            step_width=2.0, # 楼梯宽度范围
            platform_width=2.0,
            border_width=0.0,
        )

# INV_STAIR_FOR_RM = HfInvertedPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
#             proportion=0.1,
#             step_height_range=(0.1,0.35), # 楼梯高度范围
#             step_width=1., # 楼梯宽度范围
#             platform_width=3.0,
#             inverted=True,
#             border_width=0.25,
#         )
INV_STAIR_FOR_RM = HfInvertedPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            # proportion=0.1,
            # step_height_range=(0.15,0.20), # 楼梯高度范围
            # step_width=1., # 楼梯宽度范围
            # platform_width=3.0,
            # inverted=True,
            # border_width=0.25,
            proportion=0.1,
            step_height_range=(0.18,0.21), # 楼梯高度范围
            step_width=2.0, # 楼梯宽度范围
            platform_width=2.0,
            border_width=0.0,
            inverted=True,
        )

INV_STAIR_FOR_RM_2 = HfInvertedPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
            # proportion=0.1,
            # step_height_range=(0.15,0.20), # 楼梯高度范围
            # step_width=1., # 楼梯宽度范围
            # platform_width=3.0,
            # inverted=True,
            # border_width=0.25,
            proportion=0.1,
            step_height_range=(0.15,0.21), # 楼梯高度范围
            step_width=3.0, # 楼梯宽度范围
            platform_width=2.0,
            border_width=0.0,
            inverted=True,
        )

# INV_STAIR_FOR_RM = HfInvertedPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
#             # proportion=0.1,
#             # step_height_range=(0.15,0.20), # 楼梯高度范围
#             # step_width=1., # 楼梯宽度范围
#             # platform_width=3.0,
#             # inverted=True,
#             # border_width=0.25,
#             proportion=0.5,
#             step_height_range=(0.12,0.21), # 楼梯高度范围
#             step_width=2.4, # 楼梯宽度范围
#             platform_width=3.0,
#             border_width=0.0,
#             inverted=True,
#         )

# INV_STAIR_FOR_RM_2 = HfInvertedPyramidStairsTerrainCfg( # 金字塔阶梯，中间平台四周台阶
#             # proportion=0.1,
#             # step_height_range=(0.15,0.20), # 楼梯高度范围
#             # step_width=1., # 楼梯宽度范围
#             # platform_width=3.0,
#             # inverted=True,
#             # border_width=0.25,
#             proportion=0.5,
#             step_height_range=(0.12,0.21), # 楼梯高度范围
#             step_width=1.4, # 楼梯宽度范围
#             platform_width=3.0,
#             border_width=0.0,
#             inverted=True,
#         )

TWO_STEP_DEPRESSED_UPSTAIR_FOR_RM = HfTwoStepDepressedPlatformTerrainCfg(
            proportion=0.2,
            first_step_height_range=(0.15, 0.20),
            first_step_width_range=(0.90, 1.20),
            second_step_height_range=(0.15, 0.20),
            second_step_rim_width_range=(0.10, 0.20),
            platform_width=2.4,
            depression_depth=0.05,
            edge_margin=0.0,
            border_width=1.0,
        )

TWO_STEP_DEPRESSED_UPSTAIR_VIS_FOR_RM = HfTwoStepDepressedPlatformTerrainCfg(
            proportion=0.2,
            first_step_height=0.20,
            first_step_height_range=None,
            first_step_width=0.80,
            first_step_width_range=None,
            second_step_height=0.20,
            second_step_height_range=None,
            second_step_rim_width=0.20,
            second_step_rim_width_range=None,
            platform_width=2.6,
            depression_depth=0.05,
            edge_margin=0.0,
            border_width=1.0,
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

HIGH_STAIR_FOR_RM = HfPyramidStairsTerrainCfg(
    proportion=0.2,
    step_height_range=(0.25, 0.40),
    step_width=1.75,        # 台阶宽度变得不重要
    platform_width=2.5,    # 几乎占满整个 7m 空间（扣除 border）
    border_width=0.0,
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
    proportion=0.1,
    num_horizontal_range=(2, 4),
    num_vertical_range=(2, 4),
    randomize_bar_count_difficulty=True,
    force_unequal_counts=False,
    force_even_counts=True,
    bar_width_range=(0.05, 0.2),
    randomize_bar_width_difficulty=True,
    bar_height_range=(0.02, 0.06),
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


RM_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=None, # 随机种子
    curriculum=False, # 如果True，则会根据难度参数生成地形
    size=(10., 10.), # 每一块地形的尺寸
    border_width=200.0, # 每一块地形环绕的分隔距离
    border_height=1.0, # 每一块地形边缘的维兰高度，正数，围栏在ground下面，负数，围栏在ground上面
    num_rows=20, # 子地形的行数
    num_cols=20, # 子地形的列数
    color_scheme='none', # 地形颜色，'height'颜色根据地形的高度变化，'random'颜色随机，'none'没有颜色
    horizontal_scale=0.1, # 每一小块地形采样的长宽
    vertical_scale=0.005, # 每一小块地形的高度变化
    slope_threshold=0.45, # 斜面变垂直面的阈值；two-step 5cm/10cm 下陷坎需要小于 0.5
    difficulty_range=(0.0,1.0), # 如果启用curriculum，则会按难度从低到高依次生成，否则会在这个范围内随机采样
    use_cache=False, # 是否从缓存中导入子地形
    cache_dir="/tmp/isaaclab/terrains", # 缓存地址
    sub_terrains={
        'plane_for_rm': PLANE_FOR_RM,
        'high_spin_for_rm': PLANE_FOR_RM,
        'cliff_inv_stair_slope_flat_for_rm': CLIFF_INV_STAIR_SLOPE_FLAT_FOR_RM,
        # 'slope_for_rm': SLOPE_FOR_RM,
        # 'inv_slope_for_rm': INV_SLOPE_FOR_RM,
        'random_uniform_for_rm': RANDOM_UNIFORM_FOR_RM,
        'cliff_inv_stair_slope_tall_for_rm': CLIFF_INV_STAIR_SLOPE_TALL_FOR_RM,
        # 'stair_slope_for_rm': STAIR_SLOPE_FOR_RM,
        # 'radial_stair_slope_for_rm': RADIAL_STAIR_SLOPE_FOR_RM,
        'cliff_inv_stair_slope_for_rm': CLIFF_INV_STAIR_SLOPE_FOR_RM,
        'stair_for_rm': STAIR_FOR_RM,
        # 'two_step_depressed_upstair_for_rm': TWO_STEP_DEPRESSED_UPSTAIR_FOR_RM,
        # 'high_stair_for_rm': HIGH_STAIR_FOR_RM,
        # "stair_for_rm_2": STAIR_FOR_RM_2,
        # 'inv_pyramid_stair_slope_for_rm': INV_PYRAMID_STAIR_SLOPE_FOR_RM,
        'inv_stair_for_rm': INV_STAIR_FOR_RM,
        'inv_stair_for_rm_2': INV_STAIR_FOR_RM_2,
        # 'big_wave': BIGWAVE_SLOPE_FOR_RM,
        # 'fly_slope': FLYING_SLOPE_FOR_RM,
        # 'tiny_step': TINY_STEP_FOR_RM,
    },
)

RM_SLOPE_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=None, # 随机种子
    curriculum=False, # 随机采样地形类型和 difficulty_range 内的难度
    size=(9, 9), # 每一块地形的尺寸
    border_width=200.0, # 每一块地形环绕的分隔距离
    border_height=1.0, # 每一块地形边缘的维兰高度，正数，围栏在ground下面，负数，围栏在ground上面
    num_rows=20, # 子地形的行数
    num_cols=20, # 子地形的列数
    color_scheme='none', # 地形颜色，'height'颜色根据地形的高度变化，'random'颜色随机，'none'没有颜色
    horizontal_scale=0.1, # 每一小块地形采样的长宽
    vertical_scale=0.005, # 每一小块地形的高度变化
    slope_threshold=0.45, # 斜面变垂直面的阈值；two-step 5cm/10cm 下陷坎需要小于 0.5
    difficulty_range=(0.0,1.0), # 如果启用curriculum，则会按难度从低到高依次生成，否则会在这个范围内随机采样
    use_cache=False, # 是否从缓存中导入子地形
    cache_dir="/tmp/isaaclab/terrains", # 缓存地址
    sub_terrains={
        'plane_for_rm': PLANE_FOR_RM,
        'high_spin_for_rm': HIGH_SPIN_FOR_RM,
        'random_uniform_for_rm': RANDOM_UNIFORM_FOR_RM,
        'high_stair_for_rm': HIGH_STAIR_FOR_RM,
        # 'stair_for_rm': STAIR_FOR_RM,
        'inv_stair_for_rm_2': INV_STAIR_FOR_RM_2,
        # 'stair_for_rm_2': STAIR_FOR_RM_2,
        'tiny_step_for_rm': TINY_STEP_FOR_RM,
        # 'slope_for_rm': SLOPE_FOR_RM,
        # 'inv_slope_for_rm': INV_SLOPE_FOR_RM,
        'stair_slope_for_rm': STAIR_SLOPE_FOR_RM,
        'cliff_inv_stair_slope_flat_for_rm': CLIFF_INV_STAIR_SLOPE_FLAT_FOR_RM,
        'cliff_inv_stair_slope_tall_for_rm': CLIFF_INV_STAIR_SLOPE_TALL_FOR_RM,
        'cliff_inv_stair_slope_for_rm': CLIFF_INV_STAIR_SLOPE_FOR_RM,

    },
)

RM_STAIR_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=None, # 随机种子
    curriculum=False, # 如果True，则会根据难度参数生成地形
    size=(7, 7), # 每一块地形的尺寸
    border_width=200.0, # 每一块地形环绕的分隔距离
    border_height=1.0, # 每一块地形边缘的维兰高度，正数，围栏在ground下面，负数，围栏在ground上面
    num_rows=20, # 子地形的行数
    num_cols=20, # 子地形的列数
    color_scheme='none', # 地形颜色，'height'颜色根据地形的高度变化，'random'颜色随机，'none'没有颜色
    horizontal_scale=0.1, # 每一小块地形采样的长宽
    vertical_scale=0.005, # 每一小块地形的高度变化
    slope_threshold=0.45, # 斜面变垂直面的阈值；two-step 5cm/10cm 下陷坎需要小于 0.5
    difficulty_range=(0.0,1.0), # 如果启用curriculum，则会按难度从低到高依次生成，否则会在这个范围内随机采样
    use_cache=False, # 是否从缓存中导入子地形
    cache_dir="/tmp/isaaclab/terrains", # 缓存地址
    sub_terrains={
        'plane_for_rm': PLANE_FOR_RM,
        'high_spin_for_rm': HIGH_SPIN_FOR_RM,
        'stair_for_rm': STAIR_FOR_RM,
        'stair_for_rm_2': STAIR_FOR_RM_2,
        # 'high_stair_for_rm': HIGH_STAIR_FOR_RM,
        'inv_stair_for_rm': INV_STAIR_FOR_RM,
        'inv_stair_for_rm_2': INV_STAIR_FOR_RM_2,
        # 'two_step_depressed_upstair_for_rm': TWO_STEP_DEPRESSED_UPSTAIR_FOR_RM,
        'cliff_inv_stair_slope_flat_for_rm': CLIFF_INV_STAIR_SLOPE_FLAT_FOR_RM,
        'random_uniform_for_rm': RANDOM_UNIFORM_FOR_RM,
        'stair_slope_for_rm': STAIR_SLOPE_FOR_RM
    },
)

RM_SIMPLE_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=None, # 随机种子
    curriculum=False, # 如果True，则会根据难度参数生成地形
    size=(7., 7.), # 每一块地形的尺寸
    border_width=200.0, # 每一块地形环绕的分隔距离
    border_height=1.0, # 每一块地形边缘的维兰高度，正数，围栏在ground下面，负数，围栏在ground上面
    num_rows=20, # 子地形的行数
    num_cols=20, # 子地形的列数
    color_scheme='none', # 地形颜色，'height'颜色根据地形的高度变化，'random'颜色随机，'none'没有颜色
    horizontal_scale=0.1, # 每一小块地形采样的长宽
    vertical_scale=0.005, # 每一小块地形的高度变化
    slope_threshold=0.45, # 斜面变垂直面的阈值；two-step 5cm/10cm 下陷坎需要小于 0.5
    difficulty_range=(0.0,1.0), # 如果启用curriculum，则会按难度从低到高依次生成，否则会在这个范围内随机采样
    use_cache=False, # 是否从缓存中导入子地形
    cache_dir="/tmp/isaaclab/terrains", # 缓存地址
    sub_terrains={
        'plane_for_rm': PLANE_FOR_RM,
        # 'high_spin_for_rm': PLANE_FOR_RM,
        # 'cliff_inv_stair_slope_flat_for_rm': CLIFF_INV_STAIR_SLOPE_FLAT_FOR_RM,
        'slope_for_rm': SLOPE_FOR_RM,
        # "wave_for_rm": WAVE_FOR_RM,
        'inv_slope_for_rm': INV_SLOPE_FOR_RM,
        'random_uniform_for_rm': RANDOM_UNIFORM_FOR_RM,
        # 'cliff_inv_stair_slope_tall_for_rm': CLIFF_INV_STAIR_SLOPE_TALL_FOR_RM,
        'stair_slope_for_rm': STAIR_SLOPE_FOR_RM,
        # 'cliff_inv_stair_slope_for_rm': CLIFF_INV_STAIR_SLOPE_FOR_RM,
        # 'stair_for_rm': STAIR_FOR_RM,
        # 'two_step_depressed_upstair_for_rm': TWO_STEP_DEPRESSED_UPSTAIR_FOR_RM,
        # 'high_stair_for_rm': HIGH_STAIR_FOR_RM,
        # "stair_for_rm_2": STAIR_FOR_RM_2,
        # 'inv_pyramid_stair_slope_for_rm': INV_PYRAMID_STAIR_SLOPE_FOR_RM,
        # 'inv_stair_for_rm': INV_STAIR_FOR_RM,
        # 'inv_stair_for_rm_2': INV_STAIR_FOR_RM_2,
        # 'big_wave': BIGWAVE_SLOPE_FOR_RM,
        # 'fly_slope': FLYING_SLOPE_FOR_RM,
        # 'tiny_step': TINY_STEP_FOR_RM,
    },
)

RM_TWO_STEP_DEPRESSED_UPSTAIR_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=None,
    curriculum=False,
    size=(7.0, 7.0),
    border_width=2.0,
    border_height=1.0,
    num_rows=1,
    num_cols=1,
    color_scheme="height",
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.45,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    cache_dir="/tmp/isaaclab/terrains",
    sub_terrains={
        "two_step_depressed_upstair_for_rm": TWO_STEP_DEPRESSED_UPSTAIR_VIS_FOR_RM,
    },
)


# Backward compatibility: existing rough envs historically referenced RM_TERRAINS_CFG.
RM_TERRAINS_CFG = RM_ROUGH_TERRAINS_CFG
