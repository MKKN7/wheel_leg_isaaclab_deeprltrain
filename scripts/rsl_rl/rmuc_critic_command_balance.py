"""Critic-only bounded command balancing. Actor batches and GAE stay unchanged."""
import json
from pathlib import Path

import torch

KEY = '_rmuc_critic_command_weight'


def command_weights(commands, fraction=.5, maximum=3.):
    if commands.shape[-1] != 3 or not torch.isfinite(commands).all():
        raise ValueError('Finite physical vx/vy/yaw commands required')
    if not 0 <= fraction <= 1 or not maximum >= 1:
        raise ValueError('Invalid balancing fraction or maximum weight')
    # Continuous commands share finite bins, rather than becoming singleton groups.
    c = commands[..., [0, 2]].reshape(-1, 2)
    labels = torch.round(c / c.new_tensor([.025, .05])).long()
    keys, inverse, counts = torch.unique(labels, dim=0, return_inverse=True, return_counts=True)
    raw = (1 - fraction) + fraction * len(inverse) / (len(counts) * counts.float())
    per_sample = raw[inverse]
    # Preserve mean one AND the declared bounds after clipping.
    lo, hi = 0., max(2., float(maximum))
    for _ in range(40):
        mid = (lo + hi) / 2
        if float((per_sample * mid).clamp(.25, maximum).mean()) < 1:
            lo = mid
        else:
            hi = mid
    per_group = (raw * ((lo + hi) / 2)).clamp(.25, maximum)
    weights = per_group[inverse].reshape(commands.shape[:-1] + (1,))
    rows = [{'vx_bin': float(k[0]) * .025, 'yaw_bin': float(k[1]) * .05,
             'samples': int(n), 'weight': float(w), 'effective_sample_mass': float(n * w)}
            for k, n, w in zip(keys.cpu(), counts.cpu(), per_group.cpu())]
    return weights, {'fraction': fraction, 'maximum': maximum,
                     'weight_mean': float(weights.mean()), 'weight_min': float(weights.min()),
                     'weight_max': float(weights.max()), 'groups': rows,
                     'scope': 'Mixture of original sample mass and equal command-bin mass; bounded after normalization.'}


def weighted_value_output(values, weights):
    if values.shape != weights.shape or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Positive weights must match scalar value outputs')
    if values.requires_grad:
        values.register_hook(lambda gradient: gradient * weights.detach())
    return values


def configure_critic_command_balance(runner, env, fraction=.5, maximum=3.):
    alg = runner.alg
    if alg.critic.is_recurrent or alg.actor.is_recurrent or alg.symmetry or alg.use_clipped_value_loss:
        raise ValueError('Critic balancing requires feedforward, unclipped value MSE and no symmetry augmentation')
    env = env.unwrapped
    labels = []
    state = {'updating': False, 'calls': 0, 'iteration': 0}
    original_act, original_update = alg.act, alg.update
    path = Path(runner.logger.log_dir)
    (path / 'critic_command_balance_contract.json').write_text(json.dumps({
        'fraction': fraction, 'maximum': maximum, 'minimum': .25,
        'command_bin_width': [.025, .05], 'actor_sampling_unchanged': True,
        'actor_advantages_unchanged': True, 'critic_targets_unchanged': True,
        'scope': 'Weighted critic MSE gradient, equivalent in expectation to command oversampling; no duplicated physics.'}, indent=2))

    def act(obs):
        labels.append(env._current_command().detach().clone())
        return original_act(obs)

    def forward_hook(module, inputs, output):
        if state['updating'] and torch.is_grad_enabled():
            obs = inputs[0]
            if KEY not in obs.keys():
                raise RuntimeError('Critic command weights lost during minibatch shuffle')
            state['calls'] += 1
            return weighted_value_output(output, obs[KEY])
        return output

    alg.critic.register_forward_hook(forward_hook)

    def update(*args, **kwargs):
        storage = alg.storage
        commands = torch.stack(labels); labels.clear()
        if tuple(commands.shape[:2]) != tuple(storage.observations.batch_size) or KEY in storage.observations.keys():
            raise RuntimeError('Critic command labels do not match rollout layout')
        weights, report = command_weights(commands, fraction, maximum)
        storage.observations[KEY] = weights
        state.update(updating=True, calls=0)
        try:
            result = original_update(*args, **kwargs)
        finally:
            state['updating'] = False
            del storage.observations[KEY]
        if state['calls'] != alg.num_learning_epochs * alg.num_mini_batches:
            raise RuntimeError('Unexpected critic gradient weighting call count')
        with (path / 'critic_command_balance.jsonl').open('a') as f:
            f.write(json.dumps({'iteration': state['iteration'], 'gradient_calls': state['calls'], **report}) + '\n')
        state['iteration'] += 1
        return result

    alg.act, alg.update = act, update
