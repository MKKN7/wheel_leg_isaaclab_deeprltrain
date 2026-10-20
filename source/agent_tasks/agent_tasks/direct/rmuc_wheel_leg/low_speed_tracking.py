"""Keep zero/creep targets distinct from numerical denominator floors."""
import torch


def rebase_aligned_history(history, previous_direction, current_direction):
    """Express a stored scalar in the new command-aligned frame.

    Directions are +/-1 (zero command uses +1). Rebase both measured-speed
    and matched-target histories; changing command sign is not acceleration.
    """
    return history * previous_direction * current_direction


def tracking_targets(command_vx, margin_mps, exact_low_speed=False):
    magnitude=command_vx.abs()
    normalizer=magnitude.clamp_min(.1)
    if not exact_low_speed:
        return normalizer, normalizer, command_vx.sign(), torch.full_like(command_vx,margin_mps)
    # Zero command must still observe measured drift through aligned velocity.
    direction=torch.where(command_vx<0,-torch.ones_like(command_vx),torch.ones_like(command_vx))
    margin=torch.where(magnitude<.2,
        torch.minimum(torch.full_like(command_vx,margin_mps),(.04*magnitude).clamp_min(.002)),
        torch.full_like(command_vx,margin_mps))
    return magnitude,normalizer,direction,margin
