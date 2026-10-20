"""CPU reward-landscape audit; derivatives are NOT PPO policy gradients."""
import argparse
import json
import math
from pathlib import Path
import torch
from rmuc_command_sequence import dynamic_tracking_cost


def audit():
    errors = torch.tensor([0., .01, .05, .1, .2, .5, .7, 1., 1.5], dtype=torch.float64)
    rows = {}
    for scale in [.5, .25]:
        v = errors.clone().requires_grad_(True)
        costs = 4 * dynamic_tracking_cost(torch.zeros_like(v), v, torch.ones_like(v, dtype=torch.bool), scale)
        slope, = torch.autograd.grad(costs.sum(), v)
        rows[str(scale)] = dict(weighted_cost=costs.tolist(), cost_slope_wrt_error=slope.tolist(), saturation_error_mps=scale * math.sqrt(24))
    return dict(errors_mps=errors.tolist(), curves=rows,
                interpretation='Scale .25 increases error discrimination; it does not prove a better policy. Cost saturation starts at 1.2247 m/s, so do not extend this scale blindly to larger steps.',
                constraints=['Same old-skill rewards and 75% rehearsal', 'No added action or posture penalty', 'No relaxed smooth tracking gate', 'No actuator or command interface change'],
                promoted=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.write_text(json.dumps(audit(), indent=2) + '\n')
