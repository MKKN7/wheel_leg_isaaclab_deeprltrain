"""Physical bank geometry for the fixed legacy RMUC [-Z,+X,+Y] observation axes."""
import torch

BANK_CONTRACT = 'world_heading_inward_v2'

def bank_reference(gravity, command):
    """Observed lean toward heading-left and steady commanded lean in radians.

    gravity is normalized projected gravity in legacy forward/right/up axes.
    Positive world yaw requires positive lean toward heading-left. The
    horizontal-forward projection factor makes this equal the body-up/world
    heading geometry even with nonzero pitch. No observation slots change.
    """
    unit = gravity / torch.linalg.vector_norm(gravity, dim=-1, keepdim=True).clamp_min(1e-8)
    forward_horizontal = torch.linalg.vector_norm(unit[..., 1:3], dim=-1).clamp_min(1e-6)
    observed = torch.atan2(-unit[..., 1], -unit[..., 2] * forward_horizontal)
    target = torch.atan2(command[..., 0] * command[..., 2],
                         torch.full_like(command[..., 0], 9.81))
    return observed, target
