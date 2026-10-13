"""Optional local correction for the complete actor; training is isolated.

Refine the frozen authority correction near failed commands without increasing
its existing total budget. Gates use only the delivered policy observation.
"""
import torch
from torch import nn
from rmuc_state_router import StateResidualExperts, IsolatedTurnStateRouter


class LocalAuthorityCorrection(nn.Module):
    def __init__(self, centers=None):
        super().__init__()
        values = torch.as_tensor([-.8, -1., 1.2] if centers is None else centers, dtype=torch.float32).clone()
        self._validate_centers(values)
        self.register_buffer('centers', values)
        self.register_buffer('half_width', torch.tensor(.15))
        self.experts = nn.ModuleList([StateResidualExperts() for _ in range(values.numel())])

    @staticmethod
    def _validate_centers(values):
        if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all() or values.unique().numel() != values.numel():
            raise ValueError('Local centers must be a nonempty finite unique vector')

    def append_centers(self, centers):
        """Explicit checkpoint migration; keep old experts and center order."""
        extra = torch.as_tensor(centers, dtype=self.centers.dtype, device=self.centers.device)
        self._validate_centers(extra)
        combined = torch.cat((self.centers, extra))
        self._validate_centers(combined)
        start = len(self.experts)
        additions = [StateResidualExperts().to(self.centers.device) for _ in range(extra.numel())]
        self.experts.extend(additions)
        self.centers = combined
        return list(range(start, len(self.experts)))

    def gates(self, observation, speed_plateau_end=1.0):
        speed = observation[..., 7] / .5
        yaw = observation[..., 2] / .25 + observation[..., 8]
        smooth = IsolatedTurnStateRouter.smooth
        speed_gate = smooth(speed, .8, 1.) * (1. - smooth(speed, speed_plateau_end, speed_plateau_end+.2))
        distance = (yaw[..., None] - self.centers).abs()
        # Smooth compact support; isolation is checked for each declared set
        # of protected commands when appending new centers.
        local = 1. - smooth(distance, 0., self.half_width)
        return local * speed_gate[..., None]

    def forward(self, observation, source_correction, budget, speed_plateau_end=1.0):
        weights = self.gates(observation, speed_plateau_end)
        delta = torch.zeros_like(source_correction)
        for index, expert in enumerate(self.experts):
            delta = delta + weights[..., index, None] * expert(observation)
        corrected = source_correction + budget * delta
        corrected = torch.maximum(torch.minimum(corrected, budget), -budget)
        return torch.where(weights.sum(-1, keepdim=True) > 0, corrected, source_correction)
