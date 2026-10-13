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

"""M3508 + C620 wheel actuator model."""

from __future__ import annotations

import csv
import math
from dataclasses import MISSING
from pathlib import Path

import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class M3508Actuator(IdealPDActuator):
    """Ideal PD actuator clipped by an M3508+C620 torque-speed envelope."""

    cfg: "M3508ActuatorCfg"

    def __init__(self, cfg: "M3508ActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        speed, torque = self._load_torque_speed_curve(cfg)
        self._curve_speed = speed.to(device=self._device, dtype=torch.float32)
        self._curve_torque = torque.to(device=self._device, dtype=torch.float32)
        self._zeros_effort = torch.zeros_like(self.computed_effort)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def torque_limit(self, joint_vel: torch.Tensor) -> torch.Tensor:
        """Return the symmetric torque envelope for the provided joint velocities."""
        return self._lookup_torque_limit(torch.abs(joint_vel))

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        max_effort = torch.minimum(self.torque_limit(self._joint_vel), self.effort_limit)
        min_effort = -max_effort
        return torch.clip(effort, min=min_effort, max=max_effort)

    def _lookup_torque_limit(self, abs_speed: torch.Tensor) -> torch.Tensor:
        flat_speed = abs_speed.reshape(-1)
        idx_hi = torch.searchsorted(self._curve_speed, flat_speed, right=False)
        idx_hi = torch.clamp(idx_hi, min=1, max=self._curve_speed.numel() - 1)
        idx_lo = idx_hi - 1

        speed_lo = self._curve_speed[idx_lo]
        speed_hi = self._curve_speed[idx_hi]
        torque_lo = self._curve_torque[idx_lo]
        torque_hi = self._curve_torque[idx_hi]
        alpha = (flat_speed - speed_lo) / torch.clamp(speed_hi - speed_lo, min=1.0e-6)
        limit = torque_lo + alpha * (torque_hi - torque_lo)

        limit = torch.where(flat_speed <= self._curve_speed[0], self._curve_torque[0], limit)
        limit = torch.where(flat_speed > self._curve_speed[-1], self._zeros_effort.reshape(-1)[:1], limit)
        limit = torch.clamp(limit, min=0.0, max=float(self.cfg.output_stall_torque))
        return limit.reshape_as(abs_speed)

    @classmethod
    def _load_torque_speed_curve(cls, cfg: "M3508ActuatorCfg") -> tuple[torch.Tensor, torch.Tensor]:
        curve_path = cls._resolve_curve_path(cfg.curve_path)
        points: list[tuple[float, float]] = []
        with curve_path.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required = {"Motor_shaft_torque_Nm_ideal", "Motor_shaft_speed_rpm"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"M3508 curve CSV is missing columns: {sorted(missing)}")
            for row in reader:
                motor_torque = float(row["Motor_shaft_torque_Nm_ideal"])
                motor_speed_rpm = float(row["Motor_shaft_speed_rpm"])
                output_torque = motor_torque * float(cfg.gear_ratio) * float(cfg.gearbox_efficiency)
                output_speed = motor_speed_rpm / float(cfg.gear_ratio) * 2.0 * math.pi / 60.0
                if output_speed >= 0.0 and output_torque >= 0.0 and math.isfinite(output_speed + output_torque):
                    points.append((output_speed, output_torque))

        if len(points) < 2:
            raise ValueError(f"M3508 curve CSV has too few valid points: {curve_path}")

        points.sort(key=lambda item: item[0])
        speed = torch.tensor([item[0] for item in points], dtype=torch.float32)
        torque = torch.tensor([item[1] for item in points], dtype=torch.float32)
        speed, torque = cls._prepend_low_speed_extension(speed, torque, cfg)
        return speed.contiguous(), torque.contiguous()

    @staticmethod
    def _resolve_curve_path(curve_path: str) -> Path:
        path = Path(curve_path).expanduser()
        if path.is_absolute() and path.exists():
            return path
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path
        repo_path = Path(__file__).resolve().parents[4] / path
        if repo_path.exists():
            return repo_path
        raise FileNotFoundError(f"M3508 torque-speed curve not found: {curve_path}")

    @staticmethod
    def _prepend_low_speed_extension(
        speed: torch.Tensor,
        torque: torch.Tensor,
        cfg: "M3508ActuatorCfg",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stall_torque = float(cfg.output_stall_torque)
        min_known_speed = float(speed[0])
        min_known_torque = float(torque[0])
        if min_known_torque >= stall_torque:
            return speed, torch.clamp(torque, max=stall_torque)

        fit_count = min(max(int(cfg.tail_fit_points), 2), speed.numel())
        fit_speed = speed[:fit_count]
        fit_torque = torque[:fit_count]
        speed_mean = fit_speed.mean()
        torque_mean = fit_torque.mean()
        denom = torch.sum((fit_speed - speed_mean) ** 2)
        slope = torch.sum((fit_speed - speed_mean) * (fit_torque - torque_mean)) / torch.clamp(denom, min=1.0e-9)
        intercept = torque_mean - slope * speed_mean
        fitted_stall_speed = (stall_torque - float(intercept)) / float(slope) if abs(float(slope)) > 1.0e-9 else 0.0

        if not math.isfinite(fitted_stall_speed) or fitted_stall_speed < 0.0 or fitted_stall_speed >= min_known_speed:
            fitted_stall_speed = 0.0

        ext_points = max(int(cfg.tail_extension_points), 2)
        ext_speed = torch.linspace(float(fitted_stall_speed), min_known_speed, ext_points + 1, dtype=torch.float32)[:-1]
        span = max(min_known_speed - float(fitted_stall_speed), 1.0e-6)
        phase = torch.clamp((ext_speed - float(fitted_stall_speed)) / span, min=0.0, max=1.0)
        smooth = phase * phase * (3.0 - 2.0 * phase)
        ext_torque = stall_torque + (min_known_torque - stall_torque) * smooth
        merged_speed = torch.cat((ext_speed, speed))
        merged_torque = torch.cat((ext_torque, torque))
        return merged_speed, torch.clamp(merged_torque, min=0.0, max=stall_torque)


@configclass
class M3508ActuatorCfg(IdealPDActuatorCfg):
    """Configuration for an M3508 + C620 torque-speed envelope actuator."""

    class_type: type = M3508Actuator

    curve_path: str = MISSING
    """CSV file with M3508 motor-shaft torque and speed columns."""

    gear_ratio: float = 268.0 / 17.0
    """Gear ratio from motor shaft to wheel output shaft."""

    gearbox_efficiency: float = 1.0
    """Output torque efficiency multiplier after gearing."""

    output_stall_torque: float = 5.5
    """Maximum wheel-output torque used as the low-speed envelope cap."""

    tail_fit_points: int = 64
    """Number of lowest-speed measured points used to estimate the tail trend."""

    tail_extension_points: int = 128
    """Number of synthetic low-speed samples prepended before the measured curve."""
