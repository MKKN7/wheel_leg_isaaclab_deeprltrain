"""Opt-in isolated command value functions. Training critic only, never actor.

The source value network is a frozen fallback. Each supported command gets a
separate supervised model and fixed return scale. Compact, nonoverlapping gates
avoid extrapolating these models across the entire command domain.
"""
import copy

import torch
from torch import nn


VERSION = 'isolated_command_value_v1'


def mlp():
    return nn.Sequential(nn.Linear(25, 256), nn.ELU(), nn.Linear(256, 128),
                         nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))


class CommandValueNetwork(nn.Module):
    def __init__(self, source_state, supervised, half_width=(.02, .04)):
        super().__init__()
        if supervised['arm'] != 'isolated_command_scale':
            raise ValueError('Expected isolated command-scale MC checkpoint')
        grid = torch.as_tensor(supervised['grid'], dtype=torch.float32)
        widths = torch.as_tensor(half_width, dtype=torch.float32)
        if (grid.ndim != 2 or grid.shape[1] != 2 or len(grid) < 1 or
                not torch.isfinite(grid).all() or widths.shape != (2,) or
                not torch.isfinite(widths).all() or not (widths > 0).all()):
            raise ValueError('Finite command centers and positive widths required')
        separation = (grid[:, None] - grid[None]).abs() / widths
        overlap = separation.amax(-1) < 2
        overlap.fill_diagonal_(False)
        if overlap.any():
            raise ValueError('Command value supports must not overlap')
        self.base = mlp()
        self.base.load_state_dict({k.removeprefix('mlp.'): v for k, v in source_state.items()}, strict=True)
        self.base.requires_grad_(False)
        self.experts = nn.ModuleList([mlp() for _ in grid])
        self.experts.load_state_dict(supervised['model_state_dict'], strict=True)
        for key, value in dict(centers=grid, half_width=widths,
                               input_mean=supervised['mean'], input_scale=supervised['scale'],
                               value_mean=supervised['command_mean'],
                               value_scale=supervised['command_scale']).items():
            tensor = torch.as_tensor(value).detach().clone().float()
            if not torch.isfinite(tensor).all():
                raise ValueError(f'Nonfinite {key}')
            self.register_buffer(key, tensor)
        if self.input_mean.shape != (25,) or self.input_scale.shape != (25,):
            raise ValueError('Input contract is unchanged25D')
        if self.value_mean.shape != (len(grid), 1) or self.value_scale.shape != (len(grid), 1):
            raise ValueError('One return scale and mean per command required')
        if not (self.input_scale > 0).all() or not (self.value_scale > 0).all():
            raise ValueError('Positive normalization scales required')
        self.normalize_value_gradients = False

    def gates(self, observation):
        if observation.shape[-1] != 25:
            raise ValueError('Expected25D policy observation')
        # Both yaw terms use the same delivered angular-velocity observation.
        command = torch.stack((observation[..., 7] / .5,
                               observation[..., 2] / .25 + observation[..., 8]), -1)
        distance = ((command[..., None, :] - self.centers).abs() / self.half_width).amax(-1)
        # Flat central half of the support tolerates float32 yaw reconstruction.
        u = ((distance - .5) / .5).clamp(0, 1)
        return 1 - u.square() * (3 - 2*u)

    def forward(self, observation):
        shape = observation.shape[:-1]
        x = observation.reshape(-1, 25)
        gates = self.gates(x)
        with torch.no_grad():
            base = self.base(x)
        result = base
        for i, expert in enumerate(self.experts):
            active = gates[:, i] > 0
            if not active.any():
                continue
            prediction = expert((x[active] - self.input_mean) / self.input_scale)
            prediction = prediction * self.value_scale[i] + self.value_mean[i]
            if self.normalize_value_gradients and prediction.requires_grad:
                # Equivalent to MSE in normalized-return coordinates for exact
                # command centers; forward value and GAE units remain unchanged.
                scale = self.value_scale[i].detach().square()
                prediction.register_hook(lambda grad, scale=scale: grad / scale)
            delta = gates[active, i, None] * (prediction - base[active])
            indices = torch.nonzero(active).flatten()
            result = result.index_add(0, indices, delta)
        return result.reshape(*shape, 1)

    def metadata(self):
        return dict(version=VERSION, actor_dim=25, critic_input_dim=25,
                    centers=self.centers.cpu().tolist(), half_width=self.half_width.cpu().tolist(),
                    fallback='Frozen source critic', value_units='Unnormalized environment return',
                    expert_gradient_scaling='1 / fixed_command_return_std**2 when explicitly enabled',
                    actor_or_reward_modified=False)

    def checkpoint(self):
        return dict(contract=self.metadata(), critic_state_dict=copy.deepcopy(self.state_dict()))

    def load_checkpoint(self, checkpoint):
        if checkpoint['contract'] != self.metadata():
            raise ValueError('Isolated critic routing/observation contract mismatch')
        self.load_state_dict(checkpoint['critic_state_dict'], strict=True)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        state = checkpoint['critic_state_dict']
        source = {k.removeprefix('base.'): v for k, v in state.items() if k.startswith('base.')}
        spec = dict(arm='isolated_command_scale', grid=state['centers'], mean=state['input_mean'],
                    scale=state['input_scale'], command_mean=state['value_mean'],
                    command_scale=state['value_scale'], model_state_dict={
                        k.removeprefix('experts.'): v for k, v in state.items() if k.startswith('experts.')})
        model = cls(source, spec, half_width=state['half_width'])
        model.load_checkpoint(checkpoint)
        return model


def protect_inactive_expert_momentum(network, optimizer):
    """Set inactive experts' gradients toNone even if Adam has old momentum.

    Current selective forward already leaves unused experts out of autograd.
    This hook additionally guards callers that clear gradients with set_to_none=False.
    """
    state = {'active': None}

    def before_forward(module, inputs):
        if torch.is_grad_enabled():
            active = module.gates(inputs[0]).reshape(-1, len(module.experts)).any(0)
            state['active'] = active if state['active'] is None else state['active'] | active

    def before_step(optimizer, args, kwargs):
        if state['active'] is not None:
            for active, expert in zip(state['active'], network.experts):
                if not active:
                    for parameter in expert.parameters():
                        parameter.grad = None
        state['active'] = None

    return (network.register_forward_pre_hook(before_forward), optimizer.register_step_pre_hook(before_step))
