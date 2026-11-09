"""Opt-in wheel Gaussian learning; unchanged action means and sampling semantics."""
import json
import math
from pathlib import Path

import torch


class BoundedWheelStd:
    def __init__(self, actor, optimizer, bounds, learning_rate):
        low, high = map(float, bounds)
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high <= .08):
            raise ValueError('Wheel std requires 0 < lower < upper <= .08')
        if not math.isfinite(learning_rate) or not 0 < learning_rate <= .001:
            raise ValueError('Bounded exploration learning rate outside (0,.001]')
        parameters = [(n, p) for n, p in actor.distribution.named_parameters()
                      if n in {'std_param', 'log_std_param'}]
        if len(parameters) != 1 or parameters[0][1].shape != (6,):
            raise ValueError('Six-dimensional diagonal Gaussian required')
        name, parameter = parameters[0]
        self.parameter, self.log_parameter = parameter, name == 'log_std_param'
        self.low, self.high = low, high
        self.initial = parameter.detach().clone()
        initial_std = self.std()
        if not torch.isfinite(initial_std).all() or not ((initial_std[[2, 5]] >= low) & (initial_std[[2, 5]] <= high)).all():
            raise ValueError('Initial wheel std must already be inside declared bounds')
        self.lower = math.log(low) if self.log_parameter else low
        self.upper = math.log(high) if self.log_parameter else high
        # One optimizer owner, including when the old frozen parameter was listed.
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if p is not parameter]
        optimizer.state.pop(parameter, None)
        parameter.requires_grad_(True)
        mask = parameter.new_tensor([0, 0, 1, 0, 0, 1])
        self.gradient_hook = parameter.register_hook(lambda gradient: gradient * mask)
        optimizer.add_param_group({'params': [parameter], 'lr': learning_rate,
                                   'rmuc_group': 'bounded_wheel_exploration'})
        self.step_hook = optimizer.register_step_post_hook(self.after_step)

    def std(self):
        p = self.parameter.detach()
        return p.exp() if self.log_parameter else p

    def after_step(self, optimizer, args, kwargs):
        with torch.no_grad():
            p = self.parameter
            if not torch.isfinite(p).all():
                raise RuntimeError('Nonfinite Gaussian exploration parameter')
            # Assignment is required: advanced indexing alone returns a copy.
            p[[2, 5]] = p[[2, 5]].clamp(self.lower, self.upper)
            p[[0, 1, 3, 4]] = self.initial[[0, 1, 3, 4]]

    def readback(self):
        return {'wheel_bounds': [self.low, self.high], 'std': self.std().cpu().tolist(),
                'leg_std_fixed': bool(torch.equal(self.parameter.detach()[[0, 1, 3, 4]],
                                                  self.initial[[0, 1, 3, 4]])),
                'sampling_and_log_probability': 'original GaussianDistribution',
                'deterministic_mean_modified': False}


def configure_bounded_wheel_exploration(runner, bounds, learning_rate):
    if not runner.alg.entropy_coef > 0:
        raise ValueError('Bounded entropy experiment requires explicit positive entropy_coef')
    control = BoundedWheelStd(runner.alg.get_policy(), runner.alg.optimizer, bounds, learning_rate)
    path = Path(runner.logger.log_dir)
    (path / 'bounded_exploration_contract.json').write_text(json.dumps(
        {**control.readback(), 'entropy_coef': runner.alg.entropy_coef, 'learning_rate': learning_rate}, indent=2))
    previous = runner.alg.update
    count = 0

    def update(*args, **kwargs):
        nonlocal count
        result = previous(*args, **kwargs)
        with (path / 'bounded_exploration.jsonl').open('a') as output:
            output.write(json.dumps({'update': count, **control.readback()}) + '\n')
        count += 1
        return result

    runner.alg.update = update
    runner._rmuc_bounded_exploration = control
    return control
