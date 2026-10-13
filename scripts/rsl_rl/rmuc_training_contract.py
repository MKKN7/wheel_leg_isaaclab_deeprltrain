"""Checkpoint observation metadata and fail-closed delivery sampling checks."""
import copy
import json
import math


def observation_contract(cfg=None):
    value = dict(schema_version=1, observation_dim=25,
                 command_observation_mode='tracking_error_with_target',
                 command_observation_scale=[2.1, .5, 1.],
                 command_observation_error_clip=2., command_yaw_observation_error_clip=2.)
    if cfg is not None:
        for name in tuple(value)[2:]:
            value[name] = getattr(cfg, name)
        value['command_observation_scale'] = list(value['command_observation_scale'])
    return validate_observation_contract(value)


def validate_observation_contract(value):
    value = copy.deepcopy(value)
    expected = dict(schema_version=1, observation_dim=25,
                    command_observation_mode='tracking_error_with_target',
                    command_observation_scale=[2.1, .5, 1.], command_observation_error_clip=2.)
    if not isinstance(value, dict) or set(value) != {*expected, 'command_yaw_observation_error_clip'}:
        raise ValueError('Incomplete RMUC observation contract')
    if any(value[k] != v for k, v in expected.items()):
        raise ValueError('Unsupported RMUC observation layout or scaling')
    clip = value['command_yaw_observation_error_clip']
    if isinstance(clip, bool) or not isinstance(clip, (int, float)) or not math.isfinite(clip) or clip < 2:
        raise ValueError('Explicit finite yaw error clip >=2 required')
    return value


def checkpoint_observation_contract(payload):
    recorded = (payload.get('infos') or {}).get('rmuc_observation_contract')
    return observation_contract() if recorded is None else validate_observation_contract(recorded)


def configure_checkpoint_observation_contract(runner, cfg):
    contract = observation_contract(cfg)
    runner.alg.get_policy().rmuc_observation_contract = copy.deepcopy(contract)
    original = runner.save

    def save(path, infos=None):
        info = dict(infos or {})
        info['rmuc_observation_contract'] = copy.deepcopy(contract)
        return original(path, infos=info)

    runner.save = save


def required_training_commands(cfg):
    """Gate IDs do not feed the sampler: validate the actual reachable pool."""
    active = cfg.command_start_level
    ids = tuple(cfg.command_rehearsal_priority_stage_ids)
    weights = tuple(cfg.command_rehearsal_priority_stage_weights)
    if len(ids) != len(weights) or len(set(ids)) != len(ids):
        raise ValueError('Rehearsal priority IDs and weights must match and be unique')
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('Rehearsal weights must be finite and positive')
    reachable = {active}
    if (cfg.command_rehearsal_enabled and cfg.command_rehearsal_fraction > 0
            and active >= getattr(cfg, 'command_rehearsal_start_level', 0)):
        pool = {i for i in (*cfg.command_rehearsal_anchor_stage_ids, *ids) if 0 <= i < active}
        pool.add(max(active - 1, 0))
        fraction = cfg.command_rehearsal_priority_fraction
        if not 0 <= fraction <= 1:
            raise ValueError('Invalid priority fraction')
        priority = pool.intersection(ids)
        reachable.update(pool if fraction < 1 or not priority else priority)
    required = set(cfg.command_rehearsal_gate_stage_ids) | {active}
    missing = required - reachable
    if missing:
        raise ValueError(f'Required command stages absent from rehearsal sampler: {sorted(missing)}')
    pairs = set()
    for i in required:
        if not 0 <= i < len(cfg.command_stage_linear_x_mps):
            raise ValueError(f'Invalid command stage {i}')
        vx, yaw = cfg.command_stage_linear_x_mps[i], abs(cfg.command_stage_yaw_rate_rad_s[i])
        pairs.add((float(vx), float(yaw)))
        if yaw:
            pairs.add((float(vx), -float(yaw)))
    if any(abs(yaw) > cfg.command_yaw_observation_error_clip for _, yaw in pairs):
        raise ValueError('Yaw clip aliases a required command at rest')
    return sorted(pairs)


def highspeed_guard_applies(cfg):
    """Extended low-speed turn levels must not inherit high-speed restrictions."""
    level = int(getattr(cfg, 'command_start_level', 0))
    start = int(getattr(cfg, 'command_high_speed_gate_start_level', 9))
    speeds = cfg.command_stage_linear_x_mps
    return level >= start and abs(speeds[level]) >= abs(speeds[start])


def validate_trainable_target_gate(state, vx, yaw, *, physics_only=False, local_indices=None, physics_modules=None, router_speed_direction=None):
    """Reject a selected optimizer whose command gate is zero at the target.

    Pure CPU preflight before gym.make; neither constructs networks nor consumes RNG.
    Rehearsal can intentionally freeze other commands, so check only active signs.
    """
    import torch
    from rmuc_physics_adaptation import PhysicsAdaptation
    from rmuc_position_authority import PositionAuthorityHead
    from rmuc_state_router import IsolatedTurnStateRouter
    commands = [(float(vx), abs(float(yaw)))]
    if yaw:
        commands.append((float(vx), -abs(float(yaw))))
    obs = torch.zeros(len(commands), 25)
    obs[:, 7] = float(vx) * .5
    obs[:, 8] = torch.tensor([w for _, w in commands])
    if router_speed_direction is not None:
        if router_speed_direction not in ('positive','negative') or physics_only or local_indices is not None or physics_modules is not None:
            raise ValueError('Invalid or mixed router speed selection')
        prefix='rmuc_state_router.speed_extensions.'
        if not any(k.startswith(prefix) and router_speed_direction in k for k in state):
            raise ValueError('Selected speed extension direction missing')
        end=float(state.get('rmuc_state_router.turn_speed_plateau_end',1.0))
        if not 1.0 < end <= 1.5:
            raise ValueError('Explicit extended speed support required')
        smooth=IsolatedTurnStateRouter.smooth
        speed=obs[:,7]/.5; signed=obs[:,8]*(1 if router_speed_direction=='positive' else -1)
        bounds=state.get('rmuc_state_router.speed_extension_blend_bounds',(1.,1.05))
        if len(bounds)!=2 or not 1.0 <= float(bounds[0]) < float(bounds[1]) <= 1.5:
            raise ValueError('Invalid speed extension blend bounds')
        blend=smooth(speed,float(bounds[0]),float(bounds[1]))
        low=state.get('rmuc_state_router.low_yaw_speed_blend')
        if low is not None:
            if len(low)!=4 or not torch.isfinite(torch.as_tensor(low)).all() or not 1.0 <= float(low[0]) < float(low[1]) <= 1.5 or not .4501 <= float(low[2]) < float(low[3]) <= 1.0:
                raise ValueError('Invalid low-yaw speed blend contract')
            weight=1.-smooth(signed.abs(),float(low[2]),float(low[3]))
            blend=blend+weight*(smooth(speed,float(low[0]),float(low[1]))-blend)
        gate=blend*(1.-smooth(speed,end,end+.2))*smooth(signed,.4501,.6)
    elif physics_modules is not None:
        if not physics_only:
            raise ValueError('Explicit physics modules require physics_only')
        from rmuc_physics_selection import validate_paths, state_network_gate
        paths=validate_paths(physics_modules,state)
        gate=torch.stack([state_network_gate(state,path,obs) for path in paths]).amax(0)
    elif physics_only:
        if not any(k.startswith('rmuc_physics_adaptation.network.') for k in state):
            raise ValueError('Selected physics adapter is absent from checkpoint')
        from types import SimpleNamespace
        gate = PhysicsAdaptation.gate(obs)
        positive_domain = state.get('rmuc_physics_adaptation.positive_update_domain')
        if positive_domain is not None:
            gate = gate * PhysicsAdaptation.positive_gate(
                SimpleNamespace(positive_update_domain=positive_domain,
                    positive_straight_enabled=state.get('rmuc_physics_adaptation.positive_straight_enabled',1.)), obs)
        scoped_domain = state.get('rmuc_physics_adaptation.scoped_network_domain')
        if scoped_domain is not None:
            gate = gate * (1. - PhysicsAdaptation.scoped_gate(
                SimpleNamespace(scoped_network_domain=scoped_domain), obs))
        domain=state.get('rmuc_physics_adaptation.update_domain')
        if domain is not None:
            def fade(value,low,high):
                t=((value-low)/(high-low)).clamp(0.,1.)
                return 1.-t*t*(3.-2.*t)
            gate=gate*fade((obs[:,7]/.5).abs(),domain[0],domain[1])*fade(obs[:,8].abs(),domain[2],domain[3])
    else:
        pos, neg = PositionAuthorityHead.gates(obs)
        gate = pos + neg
        prefix = 'rmuc_position_authority.local_correction.'
        if prefix + 'centers' in state:
            centers = state[prefix + 'centers'].detach().cpu()
            indices = list(range(len(centers))) if local_indices is None else list(local_indices)
            if not indices or any(i < 0 or i >= len(centers) for i in indices):
                raise ValueError('Invalid selected local indices')
            width = float(state[prefix + 'half_width'])
            distance = (obs[:, 8, None] - centers[indices]).abs()
            support = 1. - IsolatedTurnStateRouter.smooth(distance, 0., width)
            gate = gate * support.amax(-1)
    # A signed local repair may deliberately keep the opposite sign frozen.
    if not bool((gate > 0).any()):
        raise ValueError(f'Selected trainable gate is zero at active target: {commands}, gates={gate.tolist()}')
    return dict(commands=commands, gates=gate.tolist(), passed=True,
                scope='Target command gate only; rollout coverage and behavior still required')


class RolloutCoverage:
    """Check delivered observations during critic warmup, before wasting actor updates."""
    def __init__(self, commands, path, grace_updates=3):
        self.commands = commands
        self.path = path
        self.grace = grace_updates
        self.counts = [0] * len(commands)
        self.updates = 0

    def observe(self, observations):
        obs = observations.reshape(-1, 25)
        speed, yaw = obs[:, 7] / .5, obs[:, 2] / .25 + obs[:, 8]
        for i, (v, w) in enumerate(self.commands):
            self.counts[i] += int((((speed-v).abs() < 1e-3) & ((yaw-w).abs() < 1e-3)).sum())
        self.updates += 1
        missing = [list(c) for c, n in zip(self.commands, self.counts) if n == 0]
        report = dict(updates=self.updates, grace_updates=self.grace,
                      commands=self.commands, samples=self.counts, missing=missing,
                      passed=not missing, scope='Rollout coverage only; not retention or behavior')
        self.path.write_text(json.dumps(report, indent=2)+'\n')
        if self.updates >= self.grace and missing:
            raise RuntimeError(f'Required commands missing from actual rollouts: {missing}')
        return report
