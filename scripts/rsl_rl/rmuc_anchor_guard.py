"""Protect a restored local straight head during nominal-only continuation."""
import torch


def protect_optimizer_slices(optimizer, parameters, index, active):
    """Keep selected slices exact despite Adam momentum and decoupled decay.

    active is evaluated for each gradient/step, so curriculum changes do not
    require recreating the optimizer. State tensors at other indices are intact.
    """
    parameters = tuple(parameters)
    owned = {id(p) for group in optimizer.param_groups for p in group['params']}
    if not parameters or any(id(p) not in owned for p in parameters):
        raise ValueError('Protected parameters must belong to the optimizer')
    handles, snapshots = [], {}

    def gradient_hook(gradient):
        if not active():
            return gradient
        result = gradient.clone()
        result[index] = 0
        return result

    for p in parameters:
        handles.append(p.register_hook(gradient_hook))

    def pre_step(opt, args, kwargs):
        snapshots.clear()
        if not active():
            return
        with torch.no_grad():
            for p in parameters:
                snapshots[p] = p[index].clone()
                if p.grad is not None:
                    p.grad[index] = 0
                for value in opt.state.get(p, {}).values():
                    if torch.is_tensor(value) and value.shape == p.shape:
                        value[index] = 0

    def post_step(opt, args, kwargs):
        with torch.no_grad():
            for p, saved in snapshots.items():
                p[index] = saved
                for value in opt.state.get(p, {}).values():
                    if torch.is_tensor(value) and value.shape == p.shape:
                        value[index] = 0
        snapshots.clear()

    handles.append(optimizer.register_step_pre_hook(pre_step))
    handles.append(optimizer.register_step_post_hook(post_step))
    return handles


def configure_nominal_straight_guard(runner):
    actor = runner.alg.get_policy()
    bank = actor.rmuc_domain_residual_bank
    heads = bank.speed_heads
    if actor.rmuc_speed_expert_mix_gain.item() != 0 or bank.legacy_straight_anchor_suppression.item() != 1:
        raise ValueError('Nominal straight guard requires the explicitly restored base route')
    index = int((heads.speed_knots - 2.9).abs().argmin())
    if not torch.isclose(heads.speed_knots[index], torch.tensor(2.9, device=heads.speed_knots.device)):
        raise ValueError('Missing 2.9 m/s anchor')
    env = runner.env.unwrapped
    active = lambda: float(env._progressive_domain_randomization_strength()) == 0.0
    runner._rmuc_nominal_straight_guard = protect_optimizer_slices(
        runner.alg.optimizer, (heads.weight1, heads.bias1, heads.weight2, heads.bias2),
        (index, 0), active)
    print('[RMUC_ANCHOR_GUARD] 2.9 straight local head protected while scheduled DR strength is zero; Adam state cleared for protected slices.', flush=True)
