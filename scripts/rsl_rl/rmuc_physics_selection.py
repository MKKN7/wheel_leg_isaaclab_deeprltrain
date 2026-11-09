"""Explicit selection of existing physics networks; never changes routing or capacity."""
import re
from types import SimpleNamespace
import torch
from rmuc_physics_adaptation import PhysicsAdaptation

_PATTERN = re.compile(r'rmuc_physics_adaptation\.(?:frozen_source\.){0,3}(?:network|reference_network|frozen_reference_network|scoped_network|alignment_network|straight_extension_network|straight_update_anchor|high_straight_network|timing_network|delay_local_network|stop_network|startup_network|reverse_network|reverse_update_anchor|reverse_extension_anchor|reverse_fast_update_anchor|reverse_coordination_anchor|reverse_frontier_anchor|reverse_frontier_update_anchor|reverse_reserve_network|joint_delay_network|joint_delay_reference_network|joint_turn_source_network|joint_turn_source_anchor|joint_turn_update_anchor)\Z')


def validate_paths(paths, state):
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Nonempty unique physics network paths required')
    for path in paths:
        if not _PATTERN.fullmatch(path) or not any(k.startswith(path+'.') for k in state):
            raise ValueError(f'Unsupported or missing physics network: {path}')
    return list(paths)


def network_gate(owner, leaf, obs):
    """Structural mixing gate, before inherited residual clamping.

    An inherited correction remains active even when its parent's update gate is
    zero. Real rollout gradients must additionally check clamp saturation.
    """
    if leaf == 'joint_delay_network':return owner.joint_delay_gate(obs)
    if leaf == 'joint_delay_reference_network':
        gate = 1. - owner.joint_delay_gate(obs)
        if hasattr(owner, 'joint_turn_update_domain'):gate = gate * owner.joint_turn_update_gate(obs)
        return gate * owner.joint_turn_gate(obs) if hasattr(owner, 'joint_turn_domain') else gate
    if leaf == 'joint_turn_source_network':
        gate=(1. - owner.joint_delay_gate(obs)) * (1. - owner.joint_turn_gate(obs))
        if hasattr(owner, 'joint_turn_source_update_domain'):
            gate=gate*owner.joint_turn_source_update_gate(obs)
        return gate
    if leaf == 'joint_turn_source_anchor':
        return ((1-owner.joint_delay_gate(obs))*(1-owner.joint_turn_gate(obs))
                *(1-owner.joint_turn_source_update_gate(obs))*owner.joint_turn_anchor_gate(obs))
    if leaf == 'joint_turn_update_anchor':
        return ((1. - owner.joint_delay_gate(obs)) * owner.joint_turn_gate(obs)
                * (1. - owner.joint_turn_update_gate(obs)))
    if leaf == 'alignment_network':
        return owner.alignment_gate(obs)
    if leaf == 'straight_extension_network':
        return owner.straight_extension_gate(obs)*owner.straight_update_gate(obs)
    if leaf == 'straight_update_anchor':
        return owner.straight_extension_gate(obs)*(1. - owner.straight_update_gate(obs))
    if leaf == 'high_straight_network':return owner.high_straight_gate(obs)
    if leaf == 'timing_network':return owner.timing_update_gate(obs)
    if leaf == 'startup_network':return owner.startup_gate(obs)
    if leaf == 'stop_network':return owner.stop_gate(obs)
    if leaf == 'delay_local_network':return owner.delay_local_gate(obs)
    if leaf == 'reverse_reserve_network':return owner.reverse_reserve_gate(obs)
    if leaf in ('reverse_network','reverse_update_anchor','reverse_extension_anchor','reverse_fast_update_anchor','reverse_coordination_anchor','reverse_frontier_anchor','reverse_frontier_update_anchor'):
        gate=owner.reverse_gate(obs)
        if hasattr(owner,'reverse_update_domain'):
            d=owner.reverse_update_domain;t=((d[1]-obs[...,7]/.5)/(d[1]-d[0])).clamp(0,1);scope=t*t*(3-2*t)
            gate=gate*((1-scope) if leaf=='reverse_update_anchor' else scope)
        elif leaf=='reverse_update_anchor':raise ValueError('Missing reverse anchor scope')
        if leaf in ('reverse_network','reverse_extension_anchor','reverse_fast_update_anchor','reverse_coordination_anchor','reverse_frontier_anchor','reverse_frontier_update_anchor') and hasattr(owner,'reverse_extension_domain'):
            d=owner.reverse_extension_domain;t=((d[1]-obs[...,7]/.5)/(d[1]-d[0])).clamp(0,1);scope=t*t*(3-2*t)
            gate=gate*((1-scope) if leaf=='reverse_extension_anchor' else scope)
        elif leaf=='reverse_extension_anchor':raise ValueError('Missing reverse extension scope')
        if leaf in ('reverse_network','reverse_fast_update_anchor','reverse_coordination_anchor','reverse_frontier_anchor','reverse_frontier_update_anchor') and hasattr(owner,'reverse_fast_update_domain'):
            scope=owner.reverse_fast_update_gate(obs)
            gate=gate*((1-scope) if leaf=='reverse_fast_update_anchor' else scope)
        elif leaf=='reverse_fast_update_anchor':raise ValueError('Missing fast reverse update scope')
        if leaf in ('reverse_network','reverse_coordination_anchor','reverse_frontier_anchor','reverse_frontier_update_anchor') and hasattr(owner,'reverse_coordination_domain'):
            scope=owner.reverse_coordination_gate(obs)
            gate=gate*((1-scope) if leaf=='reverse_coordination_anchor' else scope)
        elif leaf=='reverse_coordination_anchor':raise ValueError('Missing reverse coordination scope')
        if leaf in ('reverse_network','reverse_frontier_anchor','reverse_frontier_update_anchor') and hasattr(owner,'reverse_frontier_domain'):
            scope=owner.reverse_frontier_gate(obs)
            gate=gate*(scope if leaf in ('reverse_frontier_anchor','reverse_frontier_update_anchor') else (1-scope))
        elif leaf in ('reverse_frontier_anchor','reverse_frontier_update_anchor'):raise ValueError('Missing reverse frontier preservation')
        if leaf in ('reverse_frontier_anchor','reverse_frontier_update_anchor') and hasattr(owner,'reverse_frontier_update_domain'):
            scope=owner.reverse_frontier_update_gate(obs)
            gate=gate*(scope if leaf=='reverse_frontier_anchor' else (1-scope))
        elif leaf=='reverse_frontier_update_anchor':raise ValueError('Missing frontier update scope')
        if leaf=='reverse_frontier_anchor' and hasattr(owner,'reverse_frontier_retention_domain'):
            gate=gate*owner.reverse_frontier_retention_gate(obs)
        if leaf=='reverse_frontier_anchor' and hasattr(owner,'reverse_frontier_progress_domain'):
            gate=gate*owner.reverse_frontier_progress_gate(obs)
        return gate
    gate = owner.update_gate(obs)
    scoped = owner.scoped_gate(obs) if hasattr(owner, 'scoped_network_domain') else torch.zeros_like(gate)
    positive = owner.positive_gate(obs) if hasattr(owner, 'positive_update_domain') else torch.ones_like(gate)
    reference = owner.reference_gate(obs) if hasattr(owner, 'reference_update_domain') else torch.ones_like(gate)
    if leaf == 'network':
        local=owner.network_update_gate(obs) if hasattr(owner,'network_update_domain') else torch.ones_like(gate)
        return gate*(1-scoped)*positive*local
    if leaf == 'reference_network':return gate*(1-scoped)*(1-positive)*reference
    if leaf == 'frozen_reference_network':
        local=owner.frozen_reference_gate(obs) if hasattr(owner,'frozen_reference_update_domain') else torch.ones_like(gate)
        return gate*(1-scoped)*(1-positive)*(1-reference)*local
    if leaf == 'scoped_network':return gate*scoped
    raise ValueError(f'Unsupported physics network leaf: {leaf}')


def state_network_gate(state, path, obs):
    validate_paths([path],state)
    owner_path,leaf=path.rsplit('.',1)
    prefix=owner_path+'.'
    values={k[len(prefix):]:v.to(device=obs.device,dtype=obs.dtype) for k,v in state.items()
            if k.startswith(prefix) and '.' not in k[len(prefix):]}
    owner=SimpleNamespace(**values)
    owner.gate=PhysicsAdaptation.gate
    owner.legacy_extended_speed_fade=lambda x,original,plateau:PhysicsAdaptation.legacy_extended_speed_fade(owner,x,original,plateau)
    for name in ['update_gate','positive_gate','reference_gate','scoped_gate','frozen_reference_gate','network_update_gate','alignment_gate','straight_extension_gate','straight_update_gate','high_straight_gate','timing_update_gate','delay_local_gate','stop_gate','startup_gate','reverse_gate','reverse_fast_update_gate','reverse_coordination_gate','reverse_frontier_gate','reverse_frontier_update_gate','reverse_frontier_retention_gate','reverse_frontier_progress_gate','reverse_reserve_gate','joint_delay_gate','joint_turn_gate','joint_turn_update_gate','joint_turn_source_update_gate','joint_turn_anchor_gate']:
        setattr(owner,name,lambda x,method=getattr(PhysicsAdaptation,name):method(owner,x))
    return network_gate(owner,leaf,obs)


def selected_modules(actor, paths):
    validate_paths(paths,actor.state_dict())
    return [actor.get_submodule(path) for path in paths]
