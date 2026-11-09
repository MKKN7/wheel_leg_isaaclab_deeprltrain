"""Opt-in, bounded critic freeze for a paired credit-assignment diagnostic."""
import json
from pathlib import Path

import torch


def configure_critic_freeze(runner, updates):
    if not isinstance(updates, int) or updates <= 0:
        raise ValueError('Critic freeze needs a positive update budget')
    alg = runner.alg
    if getattr(alg.critic, 'is_recurrent', False):
        raise ValueError('Critic freeze diagnostic supports feedforward critics only')
    params = list(alg.critic.parameters())
    expected = {k: v.detach().clone() for k, v in alg.critic.state_dict().items()}
    state = {'completed_updates': 0, 'suppressed_optimizer_steps': 0}
    path = Path(runner.logger.log_dir)
    (path/'critic_freeze_contract.json').write_text(json.dumps(dict(
        updates=updates, mechanism='Clear critic gradients before optimizer.step; preserve optimizer moments',
        actor_gradients_unchanged=True, gae_unchanged=True, diagnostic_only=True), indent=2))

    def before_step(optimizer, args, kwargs):
        if state['completed_updates'] < updates:
            for parameter in params:
                parameter.grad = None
            state['suppressed_optimizer_steps'] += 1

    alg.optimizer.register_step_pre_hook(before_step)
    original = alg.update

    def update(*args, **kwargs):
        frozen = state['completed_updates'] < updates
        state['suppressed_optimizer_steps'] = 0
        result = original(*args, **kwargs)
        changed = [key for key, value in alg.critic.state_dict().items()
                   if not torch.equal(value, expected[key])] if frozen else []
        if changed:
            raise RuntimeError(f'Frozen critic changed: {changed}')
        with (path/'critic_freeze.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(update=state['completed_updates']+1, frozen=frozen,
                suppressed_optimizer_steps=state['suppressed_optimizer_steps'], changed=changed))+'\n')
        state['completed_updates'] += 1
        return result

    alg.update = update
    return state
