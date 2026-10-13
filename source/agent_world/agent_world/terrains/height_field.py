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

import os
from dataclasses import MISSING
from collections.abc import Mapping

import numpy as np
import scipy.interpolate as interpolate
import trimesh

from isaaclab.terrains import HfTerrainBaseCfg, SubTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


def discretize_size_to_pixels(size: float, scale: float) -> int:
    """Convert a metric size to a discrete height-field resolution robustly."""

    return int(np.round(size / scale))


def _resolve_fixed_or_difficulty_value(
    fixed_value: float | None,
    value_range: tuple[float, float] | None,
    difficulty: float,
    name: str,
) -> float:
    """Resolve a terrain parameter from a fixed value or a difficulty-interpolated range."""

    if fixed_value is not None:
        return float(fixed_value)
    if value_range is None:
        raise ValueError(f"Either '{name}' or '{name}_range' must be specified.")
    lower, upper = value_range
    if lower > upper:
        raise ValueError(f"The range for '{name}' must satisfy lower <= upper. Got: {value_range}.")
    return float(lower + difficulty * (upper - lower))


def _resolve_fixed_or_difficulty_int(
    fixed_value: int | None,
    value_range: tuple[int, int] | None,
    difficulty: float,
    name: str,
) -> int:
    """Resolve an integer terrain parameter from a fixed value or a difficulty-interpolated range."""

    if fixed_value is not None:
        return int(fixed_value)
    if value_range is None:
        raise ValueError(f"Either '{name}' or '{name}_range' must be specified.")
    lower, upper = value_range
    if lower > upper:
        raise ValueError(f"The range for '{name}' must satisfy lower <= upper. Got: {value_range}.")
    return int(np.rint(lower + difficulty * (upper - lower)))


_FOUR_QUADRANT_KEYS = ("front_left", "front_right", "rear_left", "rear_right")


def _resolve_four_quadrant_cfgs(cfg: "FourQuadrantTerrainCfg") -> dict[str, SubTerrainBaseCfg]:
    """Resolve a single terrain cfg or a partial quadrant mapping into four quadrant cfgs."""

    quadrants = cfg.quadrants
    if isinstance(quadrants, SubTerrainBaseCfg):
        return {key: quadrants for key in _FOUR_QUADRANT_KEYS}
    if not isinstance(quadrants, Mapping) or len(quadrants) == 0:
        raise ValueError("'quadrants' must be a SubTerrainBaseCfg or a non-empty mapping.")

    default_cfg = cfg.default_quadrant
    if default_cfg is None:
        default_cfg = next(iter(quadrants.values()))
    return {key: quadrants.get(key, default_cfg) for key in _FOUR_QUADRANT_KEYS}


def four_quadrant_terrain(
    difficulty: float, cfg: "FourQuadrantTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain tile split into four independently configured quadrants."""

    size_x, size_y = float(cfg.size[0]), float(cfg.size[1])
    sub_size = (0.5 * size_x, 0.5 * size_y)
    offsets = {
        "rear_left": (0.0, 0.0),
        "rear_right": (sub_size[0], 0.0),
        "front_left": (0.0, sub_size[1]),
        "front_right": (sub_size[0], sub_size[1]),
    }
    quadrant_cfgs = _resolve_four_quadrant_cfgs(cfg)

    meshes_list: list[trimesh.Trimesh] = []
    origin_z_values: list[float] = []
    for key in _FOUR_QUADRANT_KEYS:
        sub_cfg = quadrant_cfgs[key].copy()
        sub_cfg.size = sub_size
        sub_meshes, sub_origin = sub_cfg.function(difficulty, sub_cfg)
        transform = np.eye(4)
        transform[0, -1], transform[1, -1] = offsets[key]
        for mesh in sub_meshes:
            mesh = mesh.copy()
            mesh.apply_transform(transform)
            meshes_list.append(mesh)
        if sub_origin is not None and len(sub_origin) >= 3:
            origin_z_values.append(float(sub_origin[2]))

    origin = np.asarray(
        [0.5 * size_x, 0.5 * size_y, max(origin_z_values) if origin_z_values else 0.0],
        dtype=float,
    )
    return meshes_list, origin


@configclass
class FourQuadrantTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a tile split into four equal terrain quadrants."""

    function = four_quadrant_terrain

    quadrants: SubTerrainBaseCfg | dict[str, SubTerrainBaseCfg] = MISSING
    """Single terrain cfg for all quadrants, or a mapping keyed by front/rear + left/right."""

    default_quadrant: SubTerrainBaseCfg | None = None
    """Fallback cfg for missing quadrant keys. Defaults to the first mapping value."""


def _resolve_int_bounds(
    fixed_value: int | None,
    value_range: tuple[int, int] | None,
    name: str,
) -> tuple[int, int]:
    """Return the valid integer bounds for a fixed-or-ranged count parameter."""

    if fixed_value is not None:
        value = int(fixed_value)
        return value, value
    if value_range is None:
        raise ValueError(f"Either '{name}' or '{name}_range' must be specified.")
    lower, upper = value_range
    if lower > upper:
        raise ValueError(f"The range for '{name}' must satisfy lower <= upper. Got: {value_range}.")
    return int(lower), int(upper)


def _force_even_count(
    value: int,
    fixed_value: int | None,
    value_range: tuple[int, int] | None,
    name: str,
) -> int:
    """Adjust an odd resolved count to the nearest valid even count."""

    if value % 2 == 0:
        return value

    lower, upper = _resolve_int_bounds(fixed_value, value_range, name)
    candidates = [candidate for candidate in (value + 1, value - 1) if lower <= candidate <= upper]
    if candidates:
        return int(candidates[0])

    raise ValueError(
        f"'force_even_counts' requires '{name}' to resolve to an even value or have an adjacent even value within"
        f" its range. Got resolved value {value} with bounds ({lower}, {upper})."
    )


def _resolve_height_offset_curriculum(
    difficulty: float, cfg: "HfCliffInvertedPyramidStairsTerrainCfg"
) -> tuple[float, float]:
    """Resolve row-bucket scale and local interpolation difficulty for height offsets."""

    if not cfg.height_offset_curriculum_scale_by_difficulty:
        return 1.0, difficulty

    num_levels = max(int(cfg.height_offset_curriculum_num_levels), 1)
    if num_levels <= 1:
        return 1.0, difficulty

    clipped_difficulty = float(np.clip(difficulty, 0.0, 1.0))
    scaled_difficulty = clipped_difficulty * num_levels
    level = int(np.floor(scaled_difficulty))
    level = int(np.clip(level, 0, num_levels - 1))
    local_difficulty = float(np.clip(scaled_difficulty - level, 0.0, 1.0))
    return float(level) / float(num_levels - 1), local_difficulty


@height_field_to_mesh
def import_custom_npy_terrain(difficulty: float, cfg: "HfCustomNpyTerrainCfg") -> np.ndarray:
    """Load a custom height field from an NPY file."""

    del difficulty

    if not os.path.exists(cfg.npy_path):
        raise FileNotFoundError(f"Height field file not found: {cfg.npy_path}")

    height_field_raw = np.load(cfg.npy_path).astype(np.float32)

    width_pixels = discretize_size_to_pixels(cfg.size[0], cfg.horizontal_scale)
    length_pixels = discretize_size_to_pixels(cfg.size[1], cfg.horizontal_scale)

    x_src = np.linspace(0.0, 1.0, height_field_raw.shape[0])
    y_src = np.linspace(0.0, 1.0, height_field_raw.shape[1])
    interp_func = interpolate.RectBivariateSpline(x_src, y_src, height_field_raw)
    x_dst = np.linspace(0.0, 1.0, width_pixels)
    y_dst = np.linspace(0.0, 1.0, length_pixels)
    height_field_resized = interp_func(x_dst, y_dst)

    return np.rint(height_field_resized / cfg.vertical_scale).astype(np.int16)


@configclass
class HfCustomNpyTerrainCfg(HfTerrainBaseCfg):
    """Configuration for loading a custom NPY height field."""

    function = import_custom_npy_terrain

    npy_path: str = MISSING
    """Absolute or relative path to the NPY height field file."""


def _symmetric_bar_centers(count: int, lower: float, upper: float, bar_width: float, name: str) -> np.ndarray:
    """Return bar centers with equal clear gaps and symmetric side margins."""

    if count <= 0:
        return np.zeros((0,), dtype=np.float32)
    usable_width = upper - lower
    if usable_width <= 0.0:
        raise ValueError(f"No usable terrain span for {name}: lower={lower}, upper={upper}.")
    if count * bar_width > usable_width:
        raise ValueError(
            f"{count} '{name}' bars with bar_width ({bar_width}) exceed the usable span ({usable_width})."
        )

    if count == 1:
        return np.array([(lower + upper) * 0.5], dtype=np.float32)
    gap = (usable_width - count * bar_width) / (count + 1)
    first_center = lower + gap + 0.5 * bar_width
    center_step = bar_width + gap
    return (first_center + np.arange(count, dtype=np.float32) * center_step).astype(np.float32)


def _centered_fraction_span(lower: float, upper: float, fraction: float, name: str) -> tuple[float, float]:
    """Return a centered sub-span covering a fraction of the input span."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"'{name}' must be in (0, 1]. Got: {fraction}.")
    span = upper - lower
    if span <= 0.0:
        raise ValueError(f"No usable span for {name}: lower={lower}, upper={upper}.")
    margin = 0.5 * span * (1.0 - fraction)
    return lower + margin, upper - margin


def _resolve_grid_bars_params(
    difficulty: float,
    cfg: "HfCustomGridBarsTerrainCfg | MeshCustomGridBarsTerrainCfg | MeshCustomSplitGridBarsTerrainCfg",
) -> tuple[int, int, float, float]:
    """Resolve and validate common grid-bar terrain parameters."""

    horizontal_count_difficulty = difficulty
    vertical_count_difficulty = difficulty
    if getattr(cfg, "randomize_bar_count_difficulty", False):
        horizontal_count_difficulty = float(np.random.uniform(0.0, 1.0))
        vertical_count_difficulty = float(np.random.uniform(0.0, 1.0))

    width_difficulty = difficulty
    if getattr(cfg, "randomize_bar_width_difficulty", False):
        width_difficulty = float(np.random.uniform(0.0, 1.0))

    num_horizontal = _resolve_fixed_or_difficulty_int(
        cfg.num_horizontal, cfg.num_horizontal_range, horizontal_count_difficulty, "num_horizontal"
    )
    num_vertical = _resolve_fixed_or_difficulty_int(
        cfg.num_vertical, cfg.num_vertical_range, vertical_count_difficulty, "num_vertical"
    )
    force_even_counts = bool(getattr(cfg, "force_even_counts", False))
    if force_even_counts:
        num_horizontal = _force_even_count(
            num_horizontal, cfg.num_horizontal, cfg.num_horizontal_range, "num_horizontal"
        )
        num_vertical = _force_even_count(num_vertical, cfg.num_vertical, cfg.num_vertical_range, "num_vertical")

    if getattr(cfg, "force_unequal_counts", False) and num_horizontal == num_vertical:
        vertical_min, vertical_max = _resolve_int_bounds(cfg.num_vertical, cfg.num_vertical_range, "num_vertical")
        horizontal_min, horizontal_max = _resolve_int_bounds(
            cfg.num_horizontal, cfg.num_horizontal_range, "num_horizontal"
        )
        count_step = 2 if force_even_counts else 1
        if num_vertical + count_step <= vertical_max:
            num_vertical += count_step
        elif num_vertical - count_step >= vertical_min:
            num_vertical -= count_step
        elif num_horizontal + count_step <= horizontal_max:
            num_horizontal += count_step
        elif num_horizontal - count_step >= horizontal_min:
            num_horizontal -= count_step
        else:
            raise ValueError(
                "'force_unequal_counts' requires at least one count range that can change the resolved value."
            )
    bar_width = _resolve_fixed_or_difficulty_value(cfg.bar_width, cfg.bar_width_range, width_difficulty, "bar_width")
    bar_height = _resolve_fixed_or_difficulty_value(cfg.bar_height, cfg.bar_height_range, difficulty, "bar_height")

    if num_horizontal < 0:
        raise ValueError(f"'num_horizontal' must be non-negative. Got: {num_horizontal}.")
    if num_vertical < 0:
        raise ValueError(f"'num_vertical' must be non-negative. Got: {num_vertical}.")
    if num_horizontal == 0 and num_vertical == 0:
        raise ValueError("At least one of 'num_horizontal' or 'num_vertical' must be positive.")
    if bar_width <= 0.0:
        raise ValueError(f"'bar_width' must be positive. Got: {bar_width}.")
    if bar_height < 0.0:
        raise ValueError(f"'bar_height' must be non-negative. Got: {bar_height}.")

    return num_horizontal, num_vertical, bar_width, bar_height


@height_field_to_mesh
def grid_bars_terrain(difficulty: float, cfg: "HfCustomGridBarsTerrainCfg") -> np.ndarray:
    """Generate a centered hash/grid terrain made from horizontal and vertical rectangular bars."""

    num_horizontal, num_vertical, bar_width, bar_height = _resolve_grid_bars_params(difficulty, cfg)
    width_pixels = discretize_size_to_pixels(cfg.size[0], cfg.horizontal_scale)
    length_pixels = discretize_size_to_pixels(cfg.size[1], cfg.horizontal_scale)
    bar_height_pixels = int(np.rint(bar_height / cfg.vertical_scale))

    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)

    # ``height_field_to_mesh`` handles cfg.border_width by shrinking cfg.size before calling this function.
    x_lower, x_upper = 0.0, float(cfg.size[0])
    y_lower, y_upper = 0.0, float(cfg.size[1])
    x_bar_lower, x_bar_upper = _centered_fraction_span(
        x_lower, x_upper, float(getattr(cfg, "bar_length_ratio", 0.95)), "bar_length_ratio"
    )
    y_bar_lower, y_bar_upper = _centered_fraction_span(
        y_lower, y_upper, float(getattr(cfg, "bar_length_ratio", 0.95)), "bar_length_ratio"
    )

    x_positions = (np.arange(width_pixels, dtype=np.float32) + 0.5) * cfg.horizontal_scale
    y_positions = (np.arange(length_pixels, dtype=np.float32) + 0.5) * cfg.horizontal_scale
    x_center_inside = (x_positions >= x_lower) & (x_positions < x_upper)
    y_center_inside = (y_positions >= y_lower) & (y_positions < y_upper)
    x_bar_inside = (x_positions >= x_bar_lower) & (x_positions < x_bar_upper)
    y_bar_inside = (y_positions >= y_bar_lower) & (y_positions < y_bar_upper)

    horizontal_centers = _symmetric_bar_centers(num_horizontal, y_lower, y_upper, bar_width, "horizontal")
    vertical_centers = _symmetric_bar_centers(num_vertical, x_lower, x_upper, bar_width, "vertical")
    half_width = 0.5 * bar_width

    def _bar_mask(positions: np.ndarray, inside: np.ndarray, center: float) -> np.ndarray:
        mask = (np.abs(positions - center) <= half_width) & inside
        if not mask.any():
            inside_indices = np.flatnonzero(inside)
            if inside_indices.size > 0:
                nearest_idx = inside_indices[np.argmin(np.abs(positions[inside_indices] - center))]
                mask[nearest_idx] = True
        return mask

    for center_y in horizontal_centers:
        y_mask = _bar_mask(y_positions, y_center_inside, center_y)
        hf_raw[np.ix_(x_bar_inside, y_mask)] = np.maximum(
            hf_raw[np.ix_(x_bar_inside, y_mask)], bar_height_pixels
        )

    for center_x in vertical_centers:
        x_mask = _bar_mask(x_positions, x_center_inside, center_x)
        hf_raw[np.ix_(x_mask, y_bar_inside)] = np.maximum(
            hf_raw[np.ix_(x_mask, y_bar_inside)], bar_height_pixels
        )

    return np.rint(hf_raw).astype(np.int16)


@configclass
class HfCustomGridBarsTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a centered grid of rectangular bars."""

    function = grid_bars_terrain

    num_horizontal: int | None = None
    """Fixed number of horizontal bars running along the terrain x direction."""

    num_horizontal_range: tuple[int, int] | None = None
    """Difficulty-interpolated range for the number of horizontal bars."""

    num_vertical: int | None = None
    """Fixed number of vertical bars running along the terrain y direction."""

    num_vertical_range: tuple[int, int] | None = None
    """Difficulty-interpolated range for the number of vertical bars."""

    randomize_bar_count_difficulty: bool = False
    """Whether to use independent random difficulties for horizontal and vertical bar counts."""

    force_unequal_counts: bool = False
    """Whether to force horizontal and vertical bar counts to be different after interpolation."""

    force_even_counts: bool = False
    """Whether to force horizontal and vertical bar counts to even values after interpolation."""

    bar_width: float | None = None
    """Fixed bar width in meters."""

    bar_width_range: tuple[float, float] | None = None
    """Difficulty-interpolated range for bar width in meters."""

    randomize_bar_width_difficulty: bool = False
    """Whether to use a random difficulty for bar width instead of the incoming terrain difficulty."""

    bar_height: float | None = None
    """Fixed bar height in meters."""

    bar_height_range: tuple[float, float] | None = None
    """Difficulty-interpolated range for bar height in meters."""

    bar_length_ratio: float = 0.95
    """Fraction of the terrain span covered by each bar along its long direction."""


def mesh_grid_bars_terrain(
    difficulty: float, cfg: "MeshCustomGridBarsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a centered grid terrain from rectangular bar meshes.

    Vertical bars are kept as full-length boxes. Horizontal bars are split at
    vertical-bar spans so the intersection is occupied by one box only instead
    of two overlapping boxes. This keeps the top-down shape equivalent to
    max-compositing while avoiding duplicate coplanar contact surfaces.
    """

    num_horizontal, num_vertical, bar_width, bar_height = _resolve_grid_bars_params(difficulty, cfg)
    if cfg.ground_thickness < 0.0:
        raise ValueError(f"'ground_thickness' must be non-negative. Got: {cfg.ground_thickness}.")

    x_lower, x_upper = 0.0, float(cfg.size[0])
    y_lower, y_upper = 0.0, float(cfg.size[1])
    x_bar_lower, x_bar_upper = _centered_fraction_span(x_lower, x_upper, cfg.bar_length_ratio, "bar_length_ratio")
    y_bar_lower, y_bar_upper = _centered_fraction_span(y_lower, y_upper, cfg.bar_length_ratio, "bar_length_ratio")
    horizontal_centers = _symmetric_bar_centers(num_horizontal, y_lower, y_upper, bar_width, "horizontal")
    vertical_centers = _symmetric_bar_centers(num_vertical, x_lower, x_upper, bar_width, "vertical")

    meshes_list = []

    def _append_box(dims: tuple[float, float, float], pos: tuple[float, float, float]) -> None:
        transform = trimesh.transformations.translation_matrix(pos)
        meshes_list.append(trimesh.creation.box(dims, transform))

    if cfg.ground_thickness > 0.0:
        dims = (cfg.size[0], cfg.size[1], cfg.ground_thickness)
        pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * cfg.ground_thickness)
        _append_box(dims, pos)

    vertical_spans = [
        (
            max(x_bar_lower, float(center_x) - 0.5 * bar_width),
            min(x_bar_upper, float(center_x) + 0.5 * bar_width),
        )
        for center_x in vertical_centers
    ]
    vertical_spans = [(span_start, span_end) for span_start, span_end in vertical_spans if span_end > span_start]
    vertical_spans.sort()

    for center_y in horizontal_centers:
        segment_start = x_bar_lower
        for span_start, span_end in vertical_spans:
            segment_end = min(max(span_start, segment_start), x_bar_upper)
            segment_length = segment_end - segment_start
            if segment_length > 1.0e-6:
                dims = (segment_length, bar_width, bar_height)
                pos = (segment_start + 0.5 * segment_length, float(center_y), 0.5 * bar_height)
                _append_box(dims, pos)
            segment_start = max(segment_start, min(max(span_end, x_bar_lower), x_bar_upper))
        segment_length = x_bar_upper - segment_start
        if segment_length > 1.0e-6:
            dims = (segment_length, bar_width, bar_height)
            pos = (segment_start + 0.5 * segment_length, float(center_y), 0.5 * bar_height)
            _append_box(dims, pos)

    for center_x in vertical_centers:
        v_length = y_bar_upper - y_bar_lower
        dims = (bar_width, v_length, bar_height)
        pos = (float(center_x), y_bar_lower + 0.5 * v_length, 0.5 * bar_height)
        _append_box(dims, pos)

    origin = np.asarray([0.5 * cfg.size[0], 0.5 * cfg.size[1], bar_height])
    return meshes_list, origin


def mesh_split_grid_bars_terrain(
    difficulty: float, cfg: "MeshCustomSplitGridBarsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a split terrain with horizontal bars on one half and vertical bars on the other.

    This keeps the same bar count/width/height randomization as
    ``MeshCustomGridBarsTerrainCfg`` but avoids cross intersections entirely:
    one side of the tile contains only bars running along x, and the other side
    contains only bars running along y.
    """

    num_horizontal, num_vertical, bar_width, bar_height = _resolve_grid_bars_params(difficulty, cfg)
    if cfg.ground_thickness < 0.0:
        raise ValueError(f"'ground_thickness' must be non-negative. Got: {cfg.ground_thickness}.")

    split_ratio = float(cfg.split_ratio)
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"'split_ratio' must be in (0, 1). Got: {split_ratio}.")

    split_side = str(np.random.choice(("x", "-x", "y", "-y"))) if bool(cfg.randomize_side) else None
    if split_side is None:
        split_axis = str(cfg.split_axis).lower().replace("+", "")
        if split_axis not in ("x", "-x", "y", "-y"):
            raise ValueError(f"'split_axis' must be one of 'x', '-x', 'y', '-y'. Got: {cfg.split_axis!r}.")
        if split_axis.startswith("-"):
            split_side = split_axis
        else:
            horizontal_side = str(cfg.horizontal_side).lower()
            if horizontal_side not in ("negative", "positive"):
                raise ValueError(f"'horizontal_side' must be 'negative' or 'positive'. Got: {cfg.horizontal_side!r}.")
            split_side = f"-{split_axis}" if horizontal_side == "negative" else split_axis

    x_lower, x_upper = 0.0, float(cfg.size[0])
    y_lower, y_upper = 0.0, float(cfg.size[1])
    x_span = x_upper - x_lower
    y_span = y_upper - y_lower

    if split_side == "-x":
        x_split = x_lower + split_ratio * x_span
        horizontal_region = (x_lower, x_split, y_lower, y_upper)
        vertical_region = (x_split, x_upper, y_lower, y_upper)
    elif split_side == "x":
        x_split = x_upper - split_ratio * x_span
        horizontal_region = (x_split, x_upper, y_lower, y_upper)
        vertical_region = (x_lower, x_split, y_lower, y_upper)
    elif split_side == "-y":
        y_split = y_lower + split_ratio * y_span
        vertical_region = (x_lower, x_upper, y_lower, y_split)
        horizontal_region = (x_lower, x_upper, y_split, y_upper)
    elif split_side == "y":
        y_split = y_upper - split_ratio * y_span
        vertical_region = (x_lower, x_upper, y_split, y_upper)
        horizontal_region = (x_lower, x_upper, y_lower, y_split)
    else:
        raise ValueError(f"'split_side' must be one of 'x', '-x', 'y', '-y'. Got: {split_side!r}.")

    # For y-splits the selected split_ratio side is vertical bars, so rotate the count concepts too.
    if split_side in ("y", "-y"):
        num_horizontal, num_vertical = num_vertical, num_horizontal

    meshes_list = []

    def _append_box(dims: tuple[float, float, float], pos: tuple[float, float, float]) -> None:
        transform = trimesh.transformations.translation_matrix(pos)
        meshes_list.append(trimesh.creation.box(dims, transform))

    if cfg.ground_thickness > 0.0:
        dims = (cfg.size[0], cfg.size[1], cfg.ground_thickness)
        pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * cfg.ground_thickness)
        _append_box(dims, pos)

    h_x_lower, h_x_upper, h_y_lower, h_y_upper = horizontal_region
    h_bar_x_lower, h_bar_x_upper = _centered_fraction_span(
        h_x_lower, h_x_upper, cfg.bar_length_ratio, "bar_length_ratio"
    )
    h_length = h_bar_x_upper - h_bar_x_lower
    horizontal_centers = _symmetric_bar_centers(num_horizontal, h_y_lower, h_y_upper, bar_width, "horizontal")
    for center_y in horizontal_centers:
        dims = (h_length, bar_width, bar_height)
        pos = (h_bar_x_lower + 0.5 * h_length, float(center_y), 0.5 * bar_height)
        _append_box(dims, pos)

    v_x_lower, v_x_upper, v_y_lower, v_y_upper = vertical_region
    v_bar_y_lower, v_bar_y_upper = _centered_fraction_span(
        v_y_lower, v_y_upper, cfg.bar_length_ratio, "bar_length_ratio"
    )
    v_length = v_bar_y_upper - v_bar_y_lower
    vertical_centers = _symmetric_bar_centers(num_vertical, v_x_lower, v_x_upper, bar_width, "vertical")
    for center_x in vertical_centers:
        dims = (bar_width, v_length, bar_height)
        pos = (float(center_x), v_bar_y_lower + 0.5 * v_length, 0.5 * bar_height)
        _append_box(dims, pos)

    origin = np.asarray([0.5 * cfg.size[0], 0.5 * cfg.size[1], bar_height])
    return meshes_list, origin


@configclass
class MeshCustomGridBarsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a centered grid of rectangular bars generated as box meshes."""

    function = mesh_grid_bars_terrain

    num_horizontal: int | None = None
    """Fixed number of horizontal bars running along the terrain x direction."""

    num_horizontal_range: tuple[int, int] | None = None
    """Difficulty-interpolated range for the number of horizontal bars."""

    num_vertical: int | None = None
    """Fixed number of vertical bars running along the terrain y direction."""

    num_vertical_range: tuple[int, int] | None = None
    """Difficulty-interpolated range for the number of vertical bars."""

    randomize_bar_count_difficulty: bool = False
    """Whether to use independent random difficulties for horizontal and vertical bar counts."""

    force_unequal_counts: bool = False
    """Whether to force horizontal and vertical bar counts to be different after interpolation."""

    force_even_counts: bool = False
    """Whether to force horizontal and vertical bar counts to even values after interpolation."""

    bar_width: float | None = None
    """Fixed bar width in meters."""

    bar_width_range: tuple[float, float] | None = None
    """Difficulty-interpolated range for bar width in meters."""

    randomize_bar_width_difficulty: bool = False
    """Whether to use a random difficulty for bar width instead of the incoming terrain difficulty."""

    bar_height: float | None = None
    """Fixed bar height in meters."""

    bar_height_range: tuple[float, float] | None = None
    """Difficulty-interpolated range for bar height in meters."""

    bar_length_ratio: float = 0.95
    """Fraction of the terrain span covered by each bar along its long direction."""

    ground_thickness: float = 0.05
    """Thickness of the flat ground box below the bars in meters. Set to 0.0 to disable it."""


@configclass
class MeshCustomSplitGridBarsTerrainCfg(MeshCustomGridBarsTerrainCfg):
    """Configuration for split rectangular bars generated as box meshes.

    The tile is split into two regions. One region contains only horizontal
    bars, and the other contains only vertical bars, which removes bar-bar
    intersections while preserving the same randomization controls as the
    regular grid-bar terrain.
    """

    function = mesh_split_grid_bars_terrain

    split_axis: str = "x"
    """Selected split side. Use 'x', '-x', 'y', or '-y'."""

    split_ratio: float = 0.5
    """Fraction of the tile area assigned to the selected side."""

    horizontal_side: str = "negative"
    """Side sign used when split_axis is unsigned. Use 'negative' or 'positive'."""

    randomize_side: bool = False
    """Whether to randomly choose the selected side from 'x', '-x', 'y', and '-y'."""


@height_field_to_mesh
def cliff_inverted_pyramid_stairs_terrain(
    difficulty: float, cfg: "HfCliffInvertedPyramidStairsTerrainCfg"
) -> np.ndarray:
    """Generate an inverted pyramid stairs terrain elevated above the surrounding border."""

    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    step_height *= -1
    height_offset_scale, height_offset_difficulty = _resolve_height_offset_curriculum(difficulty, cfg)
    height_offset = _resolve_fixed_or_difficulty_value(
        cfg.height_offset,
        (
            tuple(value * height_offset_scale for value in cfg.height_offset_range)
            if cfg.height_offset_range is not None
            else None
        ),
        height_offset_difficulty,
        "height_offset",
    )

    width_pixels = discretize_size_to_pixels(cfg.size[0], cfg.horizontal_scale)
    length_pixels = discretize_size_to_pixels(cfg.size[1], cfg.horizontal_scale)
    step_width = max(1, discretize_size_to_pixels(cfg.step_width, cfg.horizontal_scale))
    step_height_pixels = int(np.rint(step_height / cfg.vertical_scale))
    platform_width = discretize_size_to_pixels(cfg.platform_width, cfg.horizontal_scale)
    height_offset_pixels = int(np.rint(height_offset / cfg.vertical_scale))

    hf_raw = np.full((width_pixels, length_pixels), height_offset_pixels, dtype=np.float32)

    current_step_height = 0
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        current_step_height += step_height_pixels
        hf_raw[start_x:stop_x, start_y:stop_y] = height_offset_pixels + current_step_height

    return np.rint(hf_raw).astype(np.int16)


@configclass
class HfCliffInvertedPyramidStairsTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a raised inverted pyramid stairs terrain with a cliff at the border."""

    function = cliff_inverted_pyramid_stairs_terrain

    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""

    step_width: float = MISSING
    """The width of the steps (in m)."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain (in m)."""

    height_offset: float | None = None
    """The fixed z-offset applied to the whole internal terrain (in m)."""

    height_offset_range: tuple[float, float] | None = None
    """The range of z-offset values, interpolated from terrain difficulty (in m)."""

    height_offset_curriculum_scale_by_difficulty: bool = False
    """Whether to scale ``height_offset_range`` by a difficulty bucket before interpolation."""

    height_offset_curriculum_num_levels: int = 11
    """Number of difficulty buckets used by the height-offset curriculum scale."""


@height_field_to_mesh
def raised_inverted_pyramid_sloped_terrain(
    difficulty: float, cfg: "HfCustomRaisedInvertedPyramidSlopedTerrainCfg"
) -> np.ndarray:
    """Generate a raised inverted pyramid terrain."""

    raised_height = _resolve_fixed_or_difficulty_value(
        cfg.raised_height, cfg.raised_height_range, difficulty, "raised_height"
    )
    slope_angle_deg = _resolve_fixed_or_difficulty_value(
        cfg.slope_angle_deg, cfg.angle_range, difficulty, "slope_angle_deg"
    )
    ramp_length = _resolve_fixed_or_difficulty_value(cfg.ramp_length, cfg.ramp_length_range, difficulty, "ramp_length")

    width_pixels = discretize_size_to_pixels(cfg.size[0], cfg.horizontal_scale)
    length_pixels = discretize_size_to_pixels(cfg.size[1], cfg.horizontal_scale)
    ramp_length_pixels = discretize_size_to_pixels(ramp_length, cfg.horizontal_scale)
    platform_width_pixels = discretize_size_to_pixels(cfg.platform_width, cfg.horizontal_scale)

    if ramp_length_pixels <= 0:
        raise ValueError(
            f"Ramp length must discretize to at least one pixel. Got: {ramp_length} m with"
            f" horizontal scale {cfg.horizontal_scale}."
        )
    if platform_width_pixels <= 0:
        raise ValueError(f"Platform width must discretize to at least one pixel. Got: {cfg.platform_width}.")

    depressed_region_pixels = platform_width_pixels + 2 * ramp_length_pixels
    if depressed_region_pixels > width_pixels or depressed_region_pixels > length_pixels:
        raise ValueError(
            "Ramp length and platform width leave no room for the raised inverted pyramid region:"
            f" ramp_length={ramp_length}, platform_width={cfg.platform_width}, size={cfg.size}."
        )

    raised_height_pixels = int(np.rint(raised_height / cfg.vertical_scale))
    ramp_run = ramp_length_pixels * cfg.horizontal_scale
    depth = np.tan(np.deg2rad(slope_angle_deg)) * ramp_run
    depth_pixels = int(np.rint(depth / cfg.vertical_scale))

    hf_raw = np.full((width_pixels, length_pixels), raised_height_pixels, dtype=np.float32)

    center_x = width_pixels // 2
    center_y = length_pixels // 2
    start_x = center_x - depressed_region_pixels // 2
    start_y = center_y - depressed_region_pixels // 2
    stop_x = start_x + depressed_region_pixels
    stop_y = start_y + depressed_region_pixels

    local_center = depressed_region_pixels // 2
    x_profile = (local_center - np.abs(local_center - np.arange(depressed_region_pixels))) / max(local_center, 1)
    y_profile = (local_center - np.abs(local_center - np.arange(depressed_region_pixels))) / max(local_center, 1)
    inner_hf = -depth_pixels * x_profile[:, None] * y_profile[None, :]

    platform_half_width = platform_width_pixels // 2
    x_pf = np.clip(depressed_region_pixels // 2 - platform_half_width, 0, depressed_region_pixels - 1)
    y_pf = np.clip(depressed_region_pixels // 2 - platform_half_width, 0, depressed_region_pixels - 1)
    z_pf = inner_hf[x_pf, y_pf]
    inner_hf = np.clip(inner_hf, min(0, z_pf), max(0, z_pf))

    hf_raw[start_x:stop_x, start_y:stop_y] = raised_height_pixels + inner_hf
    return np.rint(hf_raw).astype(np.int16)


@configclass
class HfCustomRaisedInvertedPyramidSlopedTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a raised inverted pyramid sloped terrain."""

    function = raised_inverted_pyramid_sloped_terrain

    raised_height: float | None = None
    """The fixed amount by which the whole terrain is raised above the ground (in m)."""

    raised_height_range: tuple[float, float] | None = None
    """The range of raised heights interpolated from terrain difficulty (in m)."""

    slope_angle_deg: float | None = None
    """The fixed angle of the inverted pyramid faces in degrees."""

    angle_range: tuple[float, float] | None = None
    """The range of inverted pyramid face angles in degrees."""

    ramp_length: float | None = None
    """The fixed horizontal projection length of each inverted pyramid face (in m)."""

    ramp_length_range: tuple[float, float] | None = None
    """The range of horizontal projection lengths interpolated from terrain difficulty (in m)."""

    platform_width: float = 1.0
    """The width of the square platform at the bottom of the inverted pyramid (in m)."""


@height_field_to_mesh
def directional_wave_terrain(difficulty: float, cfg: "HfCustomDirectionalWaveTerrainCfg") -> np.ndarray:
    """Generate a sinusoidal terrain profile along a single axis."""

    del difficulty

    if cfg.frequency < 0.0:
        raise ValueError(f"Wave frequency must be non-negative. Got: {cfg.frequency}.")

    width_pixels = discretize_size_to_pixels(cfg.size[0], cfg.horizontal_scale)
    length_pixels = discretize_size_to_pixels(cfg.size[1], cfg.horizontal_scale)
    amplitude_pixels = cfg.amplitude / cfg.vertical_scale

    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)

    if cfg.axis == "x":
        positions = np.arange(width_pixels) * cfg.horizontal_scale
        profile = amplitude_pixels * np.sin(2.0 * np.pi * cfg.frequency * positions + cfg.phase)
        hf_raw[:] = profile[:, None]
    elif cfg.axis == "y":
        positions = np.arange(length_pixels) * cfg.horizontal_scale
        profile = amplitude_pixels * np.sin(2.0 * np.pi * cfg.frequency * positions + cfg.phase)
        hf_raw[:] = profile[None, :]
    else:
        raise ValueError(f"Unknown axis '{cfg.axis}'. Must be 'x' or 'y'.")

    return np.rint(hf_raw).astype(np.int16)


@configclass
class HfCustomDirectionalWaveTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a single-axis sinusoidal wave height field terrain."""

    function = directional_wave_terrain

    amplitude: float = MISSING
    """The amplitude of the wave (in m)."""

    frequency: float = MISSING
    """The wave frequency in cycles per meter."""

    axis: str = "x"
    """The axis along which the wave varies. Supported values are ``x`` and ``y``."""

    phase: float = 0.0
    """The phase offset of the wave (in radians)."""


@height_field_to_mesh
def truncated_sloped_terrain(difficulty: float, cfg: "HfCustomTruncatedSlopedTerrainCfg") -> np.ndarray:
    """Generate one or more truncated single-axis ramps."""

    if cfg.num_ramps <= 0:
        raise ValueError(f"Number of ramps must be positive. Got: {cfg.num_ramps}.")

    width_pixels = discretize_size_to_pixels(cfg.size[0], cfg.horizontal_scale)
    length_pixels = discretize_size_to_pixels(cfg.size[1], cfg.horizontal_scale)
    bias_pixels = int(np.rint(cfg.bias / cfg.horizontal_scale))
    spacing_pixels = int(np.rint(cfg.ramp_spacing / cfg.horizontal_scale))

    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)

    def _resolve_ramp_specs() -> tuple[list[float], list[float]]:
        if cfg.slope_angle_deg is not None:
            ramp_angles = [cfg.slope_angle_deg] * cfg.num_ramps
        elif cfg.randomize_each_ramp:
            ramp_angles = np.random.uniform(cfg.angle_range[0], cfg.angle_range[1], size=cfg.num_ramps).tolist()
        else:
            angle = cfg.angle_range[0] + difficulty * (cfg.angle_range[1] - cfg.angle_range[0])
            ramp_angles = [angle] * cfg.num_ramps

        if cfg.ramp_length is not None:
            ramp_lengths = [cfg.ramp_length] * cfg.num_ramps
        elif cfg.randomize_each_ramp:
            ramp_lengths = np.random.uniform(
                cfg.ramp_length_range[0], cfg.ramp_length_range[1], size=cfg.num_ramps
            ).tolist()
        else:
            length = cfg.ramp_length_range[0] + difficulty * (
                cfg.ramp_length_range[1] - cfg.ramp_length_range[0]
            )
            ramp_lengths = [length] * cfg.num_ramps

        return ramp_angles, ramp_lengths

    def _place_ramps_in_domain(domain_pixels: int):
        ramp_angles, ramp_lengths = _resolve_ramp_specs()
        ramps = []
        for ramp_id, (ramp_angle_deg, ramp_length) in enumerate(zip(ramp_angles, ramp_lengths, strict=True)):
            if ramp_length <= 0.0:
                raise ValueError(f"Ramp length must be positive. Got: {ramp_lengths}.")
            start = base_start + ramp_id * spacing_pixels
            ramp_length_pixels = min(max(1, discretize_size_to_pixels(ramp_length, cfg.horizontal_scale)), domain_pixels)
            if 0 <= start and start + ramp_length_pixels <= domain_pixels:
                slope = np.tan(np.deg2rad(ramp_angle_deg))
                positions = np.arange(ramp_length_pixels) * cfg.horizontal_scale
                ramp_profile = slope * positions / cfg.vertical_scale
                ramps.append((start, ramp_length_pixels, ramp_profile))
        return ramps

    base_start = (width_pixels if cfg.axis == "x" else length_pixels) // 2
    if not cfg.centered:
        base_start += bias_pixels
    domain_pixels = width_pixels if cfg.axis == "x" else length_pixels
    base_start = int(np.clip(base_start, 0, domain_pixels - 1))

    if cfg.axis == "x":
        for start, ramp_length_pixels, ramp_profile in _place_ramps_in_domain(width_pixels):
            hf_raw[start : start + ramp_length_pixels, :] = ramp_profile[:, None]
    elif cfg.axis == "y":
        for start, ramp_length_pixels, ramp_profile in _place_ramps_in_domain(length_pixels):
            hf_raw[:, start : start + ramp_length_pixels] = ramp_profile[None, :]
    else:
        raise ValueError(f"Unknown axis '{cfg.axis}'. Must be 'x' or 'y'.")

    return np.rint(hf_raw).astype(np.int16)


@configclass
class HfCustomTruncatedSlopedTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a single-axis ramp that truncates back to the ground."""

    function = truncated_sloped_terrain

    slope_angle_deg: float | None = None
    """The fixed ramp angle in degrees."""

    angle_range: tuple[float, float] | None = None
    """The ramp angle range in degrees."""

    ramp_length: float | None = None
    """The fixed ramp footprint length along the varying axis (in m)."""

    ramp_length_range: tuple[float, float] | None = None
    """The ramp length range along the varying axis (in m)."""

    axis: str = "x"
    """The axis along which the ramp varies. Supported values are ``x`` and ``y``."""

    centered: bool = True
    """Whether to place the first ramp foot at the center along the varying axis."""

    bias: float = 0.0
    """Signed offset of the first ramp foot along the varying axis (in m)."""

    num_ramps: int = 1
    """Number of ramps to place along the varying axis."""

    ramp_spacing: float = 0.0
    """Signed spacing between consecutive ramp feet along the varying axis (in m)."""

    randomize_each_ramp: bool = False
    """Whether to independently sample each ramp from the provided ranges."""
