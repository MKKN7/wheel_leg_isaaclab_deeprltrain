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

"""Learned velocity-controlled actuator for IsaacLab.

Replaces the standard PD velocity controller with a neural network that
predicts joint torque from velocity command and velocity history.

Usage in an ``ArticulationCfg``::

    from agent_world.actuators.learned_velocity_actuator_cfg import LearnedVelocityActuatorCfg

    actuators = {
        "wheel": LearnedVelocityActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            model_path="pretrained/motor_model/model.pt",
            norm_stats_path="pretrained/motor_model/norm_stats.json",
            history_len=10,
            kd=0.175,
        ),
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from isaaclab.actuators import IdealPDActuator
from isaaclab.utils.types import ArticulationActions


class LearnedVelocityActuator(IdealPDActuator):
    """Neural-network-based velocity-controlled actuator.

    Instead of computing effort through a PD law, this actuator feeds the
    joint velocity command, current velocity, and a fixed damping coefficient
    through a trained model to predict the output torque.

    A rolling history buffer provides temporal context for the model.
    """

    cfg: "LearnedVelocityActuatorCfg"

    def __init__(self, cfg: "LearnedVelocityActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # Load TorchScript model
        model_path = self._resolve_path(cfg.model_path)
        self._model = torch.jit.load(str(model_path), map_location=self._device)
        self._model.eval()

        # Load normalization stats
        norm_path = self._resolve_path(cfg.norm_stats_path)
        with open(norm_path) as f:
            stats = json.load(f)
        self._norm_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=self._device).view(3)
        self._norm_std = torch.tensor(stats["std"], dtype=torch.float32, device=self._device).view(3)

        # History buffer: circular rolling window
        self._history_len: int = cfg.history_len
        self._kd: float = cfg.kd
        self._buffer_full: bool = False
        self._buffer_ptr: int = 0
        # Created lazily in first compute() call because num_envs / num_joints
        # are not known until after articulation setup.
        self._buffer: torch.Tensor | None = None

        # Non-instantaneous effort (for optional smoothing)
        self._prev_effort: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    #  Path resolution – mirrors M3508Actuator behaviour                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str).expanduser()
        if path.is_absolute() and path.exists():
            return path
        cwd = Path.cwd() / path
        if cwd.exists():
            return cwd
        repo = Path(__file__).resolve().parents[4] / path
        if repo.exists():
            return repo
        raise FileNotFoundError(f"Model/norm file not found: {path_str}")

    # ------------------------------------------------------------------ #
    #  Core logic                                                         #
    # ------------------------------------------------------------------ #

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """Predict joint torque via neural network.

        Args:
            control_action: Contains ``joint_velocities`` (velocity commands)
                and accepts ``joint_efforts`` (output torques).
            joint_pos: Current joint positions [rad].
            joint_vel: Current joint velocities [rad/s].

        Returns:
            ``control_action`` with ``joint_efforts`` filled.
        """
        vel_cmd = control_action.joint_velocities          # [B, J]
        B, J = vel_cmd.shape

        # Lazy buffer allocation
        H = self._history_len
        if self._buffer is None:
            self._buffer = torch.zeros(B, J, H, 3, dtype=torch.float32, device=self._device)
            self._prev_effort = torch.zeros(B, J, dtype=torch.float32, device=self._device)

        # Build current feature vector: [vel_cmd, vel, kd] per joint
        kd_tensor = torch.full_like(vel_cmd, self._kd)     # [B, J]
        feat = torch.stack([vel_cmd, joint_vel, kd_tensor], dim=-1)  # [B, J, 3]

        # Push to circular buffer
        self._buffer[:, :, self._buffer_ptr, :] = feat
        self._buffer_ptr = (self._buffer_ptr + 1) % H
        if self._buffer_ptr == 0:
            self._buffer_full = True

        # Normalize and reshape for model
        if self._buffer_full:
            # Reorder so that the most recent step is last:
            #   buffer[:,:, ptr:] are older, buffer[:,:, :ptr] are newer
            idx = [(self._buffer_ptr + i) % H for i in range(H)]
            buf_seq = self._buffer[:, :, idx, :]  # [B, J, H, 3]
        else:
            # Not yet full – replicate the last valid step backwards
            buf_seq = self._buffer.clone()
            last = buf_seq[:, :, self._buffer_ptr - 1 : self._buffer_ptr, :]  # [B, J, 1, 3]
            for i in range(self._buffer_ptr, H):
                buf_seq[:, :, i, :] = last[:, :, 0, :]

        # Normalize: (x - mean) / std
        n = (buf_seq - self._norm_mean) / self._norm_std  # [B, J, H, 3]

        # Run model
        if hasattr(self._model, "forward"):
            # TorchScript or standard nn.Module
            # Flatten batch-joint dimension: [B*J, H, 3]
            n_flat = n.reshape(B * J, H, 3)
            with torch.no_grad():
                effort_flat = self._model(n_flat)          # [B*J, 1]
            effort = effort_flat.reshape(B, J)              # [B, J]
        else:
            raise RuntimeError("Loaded model does not expose a 'forward' method.")

        # Clamp to effort limits
        effort_limit = getattr(self.cfg, "effort_limit", None)
        if effort_limit is not None:
            effort = torch.clamp(effort, -effort_limit, effort_limit)

        # Optional: exponential moving average smoothing
        smoothing = getattr(self.cfg, "effort_smoothing", 0.0)
        if smoothing > 0.0 and self._prev_effort is not None:
            effort = smoothing * self._prev_effort + (1.0 - smoothing) * effort
        self._prev_effort = effort

        control_action.joint_efforts = effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action
