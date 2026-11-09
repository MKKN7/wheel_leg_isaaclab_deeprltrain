"""Opt-in bounded command weighting of PPO advantages, without changing signs."""
import torch


def balance_advantages(advantages, physical_commands):
    if physical_commands.shape != (*advantages.shape[:-1], 3):
        raise ValueError('Physical command labels must match rollout advantage layout')
    if not torch.isfinite(advantages).all() or not torch.isfinite(physical_commands).all():
        raise ValueError('Nonfinite rollout for command balancing')
    labels = torch.round(physical_commands[..., [0, 2]].reshape(-1, 2) * 10000).long()
    _, inverse, counts = torch.unique(labels, dim=0, return_inverse=True, return_counts=True)
    # Attenuate overrepresented commands; bound minority amplification.
    raw = (counts.float().median() / counts.float()).sqrt().clamp(.5, 2.)
    weights = raw[inverse]
    weights = weights / weights.mean()
    result = advantages * weights.reshape_as(advantages)
    report = dict(commands=len(counts), samples=len(inverse),
                  minimum_count=int(counts.min()), maximum_count=int(counts.max()),
                  minimum_weight=float(weights.min()), maximum_weight=float(weights.max()),
                  mean_weight=float(weights.mean()),
                  scope='Positive bounded inverse-sqrt command weighting; no recentering, no critic target changes; behavior unproven')
    return result, report
