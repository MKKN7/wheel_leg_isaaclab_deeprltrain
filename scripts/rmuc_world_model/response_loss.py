"""Randomized-action moment constraint complements ordinary trajectory MSE."""
import torch

def randomized_residual_loss(prediction,target,noise,scale):
    """Penalize local response remaining in model residuals, not state-independent bias.

    Noise must be actual independently randomized action perturbations. Fitting
    only observational policy actions would confound policy and dynamics.
    """
    if prediction.shape!=target.shape or noise.shape[0]!=prediction.shape[0]:raise ValueError('Batch mismatch')
    x=noise-noise.mean(0)
    residual=(prediction-target)/scale
    residual=residual-residual.mean(0)
    gram=x.T@x/len(x)
    ridge=torch.eye(noise.shape[1],device=x.device,dtype=x.dtype)*1e-7
    coefficient=torch.linalg.solve(gram+ridge,x.T@residual/len(x))
    return coefficient.square().mean()
