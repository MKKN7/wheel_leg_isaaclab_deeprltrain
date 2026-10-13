"""Per-update frozen-weight audit and fixed-reference action regularization.

The mean-gradient hook adds exactly d(lambda * mean((mu-anchor)^2))/dmu
to PPO's backward pass, before its existing gradient clipping. This avoids
copying/vendor-patching RSL-RL's PPO.update. No action, log probability, reward,
or inference output is changed. Supported path: feed-forward, non-AMP PPO.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import torch


def rollout_probe(observations, limit=512):
    """Stratify over time AND environment, avoiding flattened-stride aliasing."""
    steps, envs, features = observations.shape
    env_count = min(envs, limit)
    time_count = min(steps, max(limit // env_count, 1))
    times = torch.linspace(0, steps - 1, time_count, device=observations.device).long()
    ids = torch.linspace(0, envs - 1, env_count, device=observations.device).long()
    return observations[times[:, None], ids[None, :]].reshape(-1, features).detach().clone()


def regularize_mean(mean, anchor, coefficient, diagnostics=None):
    delta = (mean - anchor).detach()
    if coefficient and mean.requires_grad:
        derivative = 2.0 * coefficient * delta / delta.numel()
        def add_anchor_gradient(gradient):
            if diagnostics is not None:
                g = gradient.detach().float()
                r = derivative.float()
                gn, rn = g.norm(), r.norm()
                diagnostics.append({
                    'ppo_mean_gradient_norm': float(gn),
                    'anchor_mean_gradient_norm': float(rn),
                    'anchor_to_ppo_norm_ratio': float(rn / gn) if gn > 0 else None,
                    'cosine': float((g * r).sum() / (gn * rn)) if gn > 0 and rn > 0 else None,
                    'scope': 'action mean before parameter Jacobian and gradient clipping',
                })
            return gradient + derivative
        mean.register_hook(add_anchor_gradient)
    return float(coefficient * delta.square().mean())


def configure_residual_safety(runner, coefficient: float, anchor_checkpoint: str | None = None) -> None:
    if coefficient < 0 or not torch.isfinite(torch.tensor(coefficient)):
        raise ValueError('Residual anchor coefficient must be finite and nonnegative')
    actor = runner.alg.get_policy()
    if actor.is_recurrent or getattr(runner.alg, 'symmetry', None):
        raise ValueError('Residual safety requires feed-forward PPO without symmetry augmentation')
    distribution = actor.distribution
    if distribution is None or not hasattr(distribution, '_distribution'):
        raise ValueError('Residual safety requires the Gaussian mean distribution contract')
    cached = distribution._distribution
    distribution._distribution = None
    try:
        reference = copy.deepcopy(actor).eval()
    finally:
        distribution._distribution = cached
    if anchor_checkpoint:
        payload = torch.load(anchor_checkpoint, map_location=next(actor.parameters()).device, weights_only=False)
        reference.load_state_dict(payload['actor_state_dict'], strict=True)
    reference.requires_grad_(False)
    # Persist immutable base/expert weights separately from heads whose trainable
    # status legitimately changes at a curriculum boundary.
    immutable_prefixes = ('mlp.', 'rmuc_turn_expert.', 'rmuc_speed_expert.', 'rmuc_turn_adapter.')
    immutable = {n: p.detach().clone() for n, p in actor.named_parameters()
                 if n.startswith(immutable_prefixes)}
    if not immutable or any(p.requires_grad for n, p in actor.named_parameters() if n in immutable):
        raise ValueError('Base/expert weights must already be frozen before residual safety is installed')
    immutable_buffers = {n: b.detach().clone() for n, b in actor.named_buffers()
                         if n.startswith('rmuc_')}
    path = Path(runner.logger.log_dir) / 'residual_safety.jsonl'
    state = {'updating': False, 'anchor': None, 'losses': [], 'gradient_diagnostics': [], 'updates': 0,
             'start_iteration': int(runner.current_learning_iteration)}
    original_forward = actor.forward
    original_distribution_update = distribution.update
    original_update = runner.alg.update

    def forward(obs, masks=None, hidden_state=None, stochastic_output=False):
        if state['updating'] and stochastic_output and torch.is_grad_enabled():
            with torch.no_grad():
                state['anchor'] = reference(obs, masks=masks, hidden_state=hidden_state)
        else:
            state['anchor'] = None
        return original_forward(obs, masks=masks, hidden_state=hidden_state,
                                stochastic_output=stochastic_output)

    def update_distribution(mean):
        if state['updating'] and state['anchor'] is not None:
            state['losses'].append(regularize_mean(mean, state['anchor'], coefficient,
                                                  state['gradient_diagnostics']))
        return original_distribution_update(mean)

    def update(*args, **kwargs):
        probe = rollout_probe(runner.alg.storage.observations['policy'])
        frozen_before = {n: p.detach().clone() for n, p in actor.named_parameters()
                         if not p.requires_grad and not n.startswith('distribution.')}
        state['updating'] = True
        state['losses'] = []
        state['gradient_diagnostics'] = []
        try:
            result = original_update(*args, **kwargs)
        finally:
            state['updating'] = False
            state['anchor'] = None
        changed_frozen = []
        frozen_max = 0.0
        immutable_max = 0.0
        for name, parameter in actor.named_parameters():
            current = parameter.detach()
            if name in immutable:
                immutable_max = max(immutable_max, float((current - immutable[name]).abs().max()))
                if not torch.equal(current, immutable[name]):
                    changed_frozen.append(name)
            if name in frozen_before and not parameter.requires_grad:
                frozen_max = max(frozen_max, float((current - frozen_before[name]).abs().max()))
                if not torch.equal(current, frozen_before[name]):
                    changed_frozen.append(name)
        buffer_max = 0.0
        current_buffers = dict(actor.named_buffers())
        for name, expected in immutable_buffers.items():
            current = current_buffers.get(name)
            if current is None or current.shape != expected.shape or not torch.equal(current, expected):
                changed_frozen.append(name)
                delta = float('inf') if current is None or current.shape != expected.shape else float((current.float() - expected.float()).abs().max())
                buffer_max = max(buffer_max, delta)
        with torch.no_grad():
            output = actor({'policy': probe})
            anchor = reference({'policy': probe})
            base = actor.mlp(actor.get_latent({'policy': probe}))
            anchor_delta = output - anchor
            base_delta = output - base
            command = torch.stack((probe[:, 7] / .5, probe[:, 2] / .25 + probe[:, 8]), -1)
            # Rounded commands are for reporting only, never routing/control.
            commands, inverse = torch.unique(torch.round(command * 1000) / 1000, dim=0, return_inverse=True)
            buckets = []
            for i, target in enumerate(commands):
                mask = inverse == i
                delta = anchor_delta[mask].abs()
                buckets.append({'command_vx_yaw': target.tolist(), 'samples': int(mask.sum()),
                    'anchor_delta_abs_max': float(delta.max()),
                    'anchor_delta_rms': float(delta.square().mean().sqrt()),
                    'composition_minus_base_rms': float(base_delta[mask].square().mean().sqrt()),
                    'action_saturation_fraction': float((output[mask].abs() >= 1).float().mean())})
        state['updates'] += 1
        record = {'time_unix': time.time(), 'update': state['updates'],
            'iteration': state['start_iteration'] + state['updates'] - 1, 'anchor_l2_coefficient': coefficient,
            'anchor_checkpoint': anchor_checkpoint,
            'anchor_l2_loss': sum(state['losses']) / max(len(state['losses']), 1),
            'regularized_minibatches': len(state['losses']),
            'mean_gradient_diagnostics': state['gradient_diagnostics'],
            'immutable_weight_abs_max_delta': immutable_max,
            'immutable_routing_buffer_abs_max_delta': buffer_max,
            'frozen_step_weight_abs_max_delta': frozen_max,
            'changed_frozen_names': sorted(set(changed_frozen)),
            'anchor_action_abs_max_delta': float(anchor_delta.abs().max()),
            'composition_minus_base_rms': float(base_delta.square().mean().sqrt()),
            'finite_actions': bool(torch.isfinite(output).all()), 'buckets': buckets}
        def finite_json(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {k: finite_json(v) for k, v in value.items()}
            if isinstance(value, list):
                return [finite_json(v) for v in value]
            return value
        with path.open('a') as stream:
            stream.write(json.dumps(finite_json(record), allow_nan=False) + '\n')
            stream.flush()
        result.update(residual_anchor_l2=record['anchor_l2_loss'],
                      residual_anchor_action_abs_max_delta=record['anchor_action_abs_max_delta'],
                      residual_immutable_weight_abs_max_delta=immutable_max,
                      residual_frozen_step_weight_abs_max_delta=frozen_max)
        if changed_frozen or not record['finite_actions']:
            runner.save(str(path.parent / 'model_residual_safety_violation.pt'))
            raise RuntimeError('Residual safety violation; checkpoint and JSONL audit saved')
        return result

    actor.forward = forward
    distribution.update = update_distribution
    runner.alg.update = update
    print(f'[RMUC_RESIDUAL_SAFETY] anchor_l2={coefficient} audit={path}', flush=True)
