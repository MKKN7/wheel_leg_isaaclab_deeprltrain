"""Explicit selection of existing physics networks; never changes routing or capacity."""
import re
from types import SimpleNamespace
import torch
from rmuc_physics_adaptation import PhysicsAdaptation

_PATTERN = re.compile(r'rmuc_physics_adaptation\.(?:frozen_source\.){0,3}(?:network|reference_network|frozen_reference_network|scoped_network|alignment_network|straight_extension_network|reverse_network)\Z')


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
    if leaf == 'alignment_network':
        return owner.alignment_gate(obs)
    if leaf == 'straight_extension_network':
        return owner.straight_extension_gate(obs)
    if leaf == 'reverse_network':
        return owner.reverse_gate(obs)
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
    for name in ['update_gate','positive_gate','reference_gate','scoped_gate','frozen_reference_gate','network_update_gate','alignment_gate','straight_extension_gate','reverse_gate']:
        setattr(owner,name,lambda x,method=getattr(PhysicsAdaptation,name):method(owner,x))
    return network_gate(owner,leaf,obs)


def selected_modules(actor, paths):
    validate_paths(paths,actor.state_dict())
    return [actor.get_submodule(path) for path in paths]
