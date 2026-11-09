"""Monte Carlo return diagnostics with explicit bounds on unobserved tails."""
import torch


def boundary_censored_returns(rewards, terminated, truncated, boundary_only, gamma, reward_abs_bound):
    """Sensitivity labels: unknown continuation at pure spatial-boundary resets.

    This is a censoring diagnostic, not a change to the environment's MDP.
    Retain the ordinary episodic labels alongside it.
    """
    if boundary_only.shape != terminated.shape or (boundary_only & ~terminated).any():
        raise ValueError('Boundary censor mask must be a subset of terminations')
    return returns_and_tail_bounds(rewards, terminated & ~boundary_only,
                                   truncated | boundary_only, gamma, reward_abs_bound)


def returns_and_tail_bounds(rewards, terminated, truncated, gamma, reward_abs_bound):
    if not 0 < gamma < 1 or reward_abs_bound <= 0:
        raise ValueError('Invalid discount or reward bound')
    if rewards.shape != terminated.shape or rewards.shape != truncated.shape:
        raise ValueError('Reward/done shapes differ')
    returns = torch.empty_like(rewards)
    bounds = torch.empty_like(rewards)
    maximum_value = reward_abs_bound / (1 - gamma)
    future = torch.zeros_like(rewards[0])
    tail = torch.full_like(future, maximum_value)
    for step in range(len(rewards) - 1, -1, -1):
        terminal, timeout = terminated[step], truncated[step]
        future = rewards[step] + gamma * future * ~(terminal | timeout)
        # True terminations have zero future value; time limits do not.
        tail = torch.where(terminal, 0., torch.where(timeout, maximum_value, tail) * gamma)
        returns[step], bounds[step] = future, tail
    return returns, bounds


def value_metrics(prediction, target):
    if prediction.numel() == 0:
        return {'samples': 0}
    error = prediction - target
    variance = target.var(unbiased=False)
    return {'samples': prediction.numel(), 'prediction_mean': float(prediction.mean()),
            'return_mean': float(target.mean()), 'mean_bias': float(error.mean()),
            'rmse': float(error.square().mean().sqrt()), 'mae': float(error.abs().mean()),
            'explained_variance': float(1 - error.var(unbiased=False) / variance) if variance > 1e-9 else None}
