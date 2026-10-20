"""Opt-in root-body COM translation DR using the installed PhysX index API.

Offsets are an engineering prior, not a measured COM bound. RMUC physical
Forward/Left/Up is [-body Z, -body X, +body Y], so the local translation is
[-Left, +Up, -Forward]. The asset's initial +90deg X pose is NOT applied a
second time: the API takes COM poses relative to each body's link frame.
Geometry, body mass, principal axes and principal inertia are not modified.
"""
import math

import torch


DEFAULT_MAX_OFFSET_FLU_M = (.01, .01, .005)


def flu_to_body(offset_flu):
    """Map physical body-attached FLU translations to RMUC root-link axes."""
    if offset_flu.shape[-1] != 3:
        raise ValueError("COM offsets must end in three Forward/Left/Up components")
    return torch.stack((-offset_flu[..., 1], offset_flu[..., 2], -offset_flu[..., 0]), dim=-1)


class RootComRandomizer:
    """Resample COM offsets from one immutable nominal root COM snapshot.

    Construct after the articulation initializes and before any COM mutations.
    ``reset`` is intended for episode reset, never inside a physics-step loop:
    PhysX model-property writes internally synchronize through CPU.
    """

    def __init__(self, robot, root_body_id, *, enabled=False,
                 max_offset_flu_m=DEFAULT_MAX_OFFSET_FLU_M, backend="physx"):
        if type(enabled) is not bool:
            raise ValueError("COM DR enabled must be boolean")
        if backend != "physx":
            raise ValueError("This COM runtime helper is verified for PhysX only")
        if (not isinstance(max_offset_flu_m, (tuple, list)) or len(max_offset_flu_m) != 3
                or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0
                       for v in max_offset_flu_m)):
            raise ValueError("COM maximum offsets must be three finite nonnegative SI lengths")
        if (type(root_body_id) is not int or not 0 <= root_body_id < len(robot.body_names)
                or robot.body_names[root_body_id] != "robot"):
            raise ValueError("COM DR must target the RMUC robot root rigid body")
        self.robot = robot
        self.root_body_id = root_body_id
        self.enabled = enabled
        self.max_offset_flu_m = tuple(float(v) for v in max_offset_flu_m)
        nominal = robot.data.body_com_pose_b.torch[:, root_body_id:root_body_id + 1, :]
        if nominal.ndim != 3 or nominal.shape[1:] != (1, 7) or not nominal.is_floating_point():
            raise ValueError("Expected PhysX local COM pose tensor [num_envs, 1, 7]")
        if not bool(torch.isfinite(nominal).all()):
            raise ValueError("Nominal COM pose must be finite")
        self.nominal_com_pose_b = nominal.detach().clone()
        self.offset_flu_m = nominal.new_zeros((nominal.shape[0], 3))
        self.offset_body_m = torch.zeros_like(self.offset_flu_m)
        self.strength_per_env = nominal.new_zeros(nominal.shape[0])
        self.reset_write_calls = 0
        self.updated_env_samples = 0

    def reset(self, env_ids, strength):
        """Write fresh nominal+offset poses; disabled mode is an exact no-op."""
        if not self.enabled:
            return
        if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("COM DR strength must be finite in [0, 1]")
        if env_ids is None:
            ids = torch.arange(self.nominal_com_pose_b.shape[0], device=self.nominal_com_pose_b.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.nominal_com_pose_b.device)
            if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("COM reset env_ids must be one-dimensional integer indices")
            ids = ids.to(torch.long)
        if ids.numel() == 0:
            return
        if bool(((ids < 0) | (ids >= self.nominal_com_pose_b.shape[0])).any()):
            raise ValueError("COM reset environment index is out of range")
        if torch.unique(ids).numel() != ids.numel():
            raise ValueError("COM reset environment indices must be unique")
        offsets = self.offset_flu_m.new_zeros((ids.numel(), 3))
        bounds = offsets.new_tensor(self.max_offset_flu_m) * float(strength)
        if strength > 0 and any(self.max_offset_flu_m):
            offsets.uniform_(-1., 1.).mul_(bounds)
        offsets_body = flu_to_body(offsets)
        poses = self.nominal_com_pose_b[ids].clone()
        poses[:, 0, :3].add_(offsets_body)
        # Preserve nominal xyzw principal-axis rotation byte-for-byte. This
        # partial PhysX write does not request mass/inertia/geometry changes.
        self.robot.set_coms_index(coms=poses, body_ids=[self.root_body_id], env_ids=ids)
        self.offset_flu_m[ids] = offsets
        self.offset_body_m[ids] = offsets_body
        self.strength_per_env[ids] = float(strength)
        self.reset_write_calls += 1
        self.updated_env_samples += ids.numel()

    def readback(self):
        actual = self.robot.data.body_com_pose_b.torch[:, self.root_body_id:self.root_body_id + 1, :].detach()
        expected = self.nominal_com_pose_b.clone()
        expected[:, 0, :3].add_(self.offset_body_m)
        return {
            "enabled": self.enabled,
            "scope": "engineering_prior_root_body_com_translation_not_measured_calibration",
            "root_body_name": self.robot.body_names[self.root_body_id],
            "root_body_id": self.root_body_id,
            "input_offset_frame": "physical_body_forward_left_up",
            "body_offset_mapping": "body_xyz=(-left,+up,-forward)",
            "max_offset_flu_m_at_strength1": list(self.max_offset_flu_m),
            "strength_per_env": self.strength_per_env.cpu().tolist(),
            "offset_flu_m_per_env": self.offset_flu_m.cpu().tolist(),
            "offset_body_m_per_env": self.offset_body_m.cpu().tolist(),
            "nominal_com_pose_body_xyzw_per_env": self.nominal_com_pose_b[:, 0].cpu().tolist(),
            "actual_com_pose_body_xyzw_per_env": actual[:, 0].cpu().tolist(),
            "actual_matches_requested": bool(torch.allclose(actual, expected, rtol=0., atol=1e-7)),
            "com_quaternion_exactly_preserved": bool(torch.equal(actual[..., 3:], self.nominal_com_pose_b[..., 3:])),
            "mass_inertia_geometry_written_by_helper": False,
            "reset_write_calls": self.reset_write_calls,
            "updated_env_samples": self.updated_env_samples,
            "behavior_certified": False,
        }
