"""Export the full RMUC deterministic actor, including every gated residual.

The result emits raw normalized actions. Observation construction and the
environment's bounded feedback/actuator mapping remain deployment obligations.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn


class RmucPolicyExport(nn.Module):
    """Trace the installed actor forward instead of RSL-RL's base-MLP wrapper."""

    def __init__(self, actor: nn.Module) -> None:
        super().__init__()
        if actor.obs_groups != ["policy"] or actor.obs_dim != 25:
            raise ValueError("RMUC export requires the single 25-D policy observation group")
        self.actor = copy.deepcopy(actor).cpu().eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor({"policy": observation}, stochastic_output=False)


def export_rmuc_policy(actor: nn.Module, path: str | Path) -> tuple[Path, Path]:
    """Write JIT/ONNX from the exact installed mixture; support dynamic batches."""
    authority = getattr(actor, 'rmuc_position_authority', None)
    if hasattr(getattr(actor, 'rmuc_state_router', None), 'turn_speed_plateau_end'):
        raise ValueError('Router support migration is diagnostic-only; not qualified for deployment')
    if float(getattr(authority, 'speed_plateau_end', 1.0)) > 1.0:
        raise ValueError('Expanded authority is diagnostic-only; deployment consumer contract is not yet qualified')
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    wrapper = RmucPolicyExport(actor)
    sample = torch.zeros(1, 25)
    sample[:, 5] = -1.0
    sample[:, 7] = 1.25
    sample[:, 8] = 0.15
    jit_path = directory / "policy.pt"
    onnx_path = directory / "policy.onnx"
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, sample)
        traced.save(str(jit_path))
        torch.onnx.export(
            wrapper, sample, str(onnx_path), export_params=True,
            opset_version=18, dynamo=False, input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        )
    head = getattr(actor, 'rmuc_position_authority', None)
    contract = {
        'observation_dim': 25, 'action_dim': 6,
        'position_authority_enabled': head is not None,
        'local_authority_correction_enabled': head is not None and head.local_correction is not None,
        'output_semantics': 'legacy action units; active-leg local range may exceed one' if head is not None else 'legacy raw normalized actions',
        'uniform_clip_one_compatible': head is None,
        'task_outer_feedback_required': False if head is not None else None,
        'runtime_validation_complete': False,
    }
    if hasattr(actor, 'rmuc_observation_contract'):
        contract['observation_contract'] = copy.deepcopy(actor.rmuc_observation_contract)
    physics = getattr(actor, 'rmuc_physics_adaptation', None)
    contract['physics_adaptation_enabled'] = physics is not None
    startup = getattr(actor, 'rmuc_startup_adaptation', None)
    contract['startup_adaptation_enabled'] = startup is not None
    if startup is not None:
        contract['startup_adaptation_budget'] = startup.budget.detach().cpu().tolist()
        contract['startup_adaptation_domain'] = 'command vx1+/-.1, yaw1.4+/-.1; observed vx estimate fades .35 to .75mps; no timer'
    if physics is not None:
        contract['physics_adaptation_budget'] = physics.budget.detach().cpu().tolist()
        if hasattr(physics, 'positive_update_domain'):
            contract['physics_positive_update_entry_bounds'] = physics.positive_update_domain.detach().cpu().tolist()
            contract['physics_positive_reference_frozen'] = True
            contract['physics_positive_straight_enabled'] = bool(getattr(physics,'positive_straight_enabled',True))
        if hasattr(physics, 'scoped_network_domain'):
            contract['physics_frozen_scoped_network_domain'] = physics.scoped_network_domain.detach().cpu().tolist()
            contract['physics_scoped_network_semantics'] = 'convex blend before shared residual clip; speed ramp-up/plateau/ramp-down then abs-yaw fade'
        if hasattr(physics, 'update_domain'):
            contract['physics_trainable_update_domain_speed_yaw_fades'] = physics.update_domain.detach().cpu().tolist()
        contract['physics_frozen_source_enabled'] = hasattr(physics, 'frozen_source')
        if hasattr(physics, 'frozen_source'):
            contract['physics_inherited_budget'] = physics.frozen_source.budget.detach().cpu().tolist()
            contract['physics_migration_budget_semantics'] = 'new correction shares declared current total envelope; read actual budget values above'
        contract['physics_adaptation_domain'] = 'abs speed <=1.5 fades to zero at1.7; abs yaw <=4 fades to zero at4.2'
    if head is not None:
        contract.update(local_action_limits=head.limit.detach().cpu().tolist(),
            local_residual_budget=head.budget.detach().cpu().tolist(),
            active_leg_scale_rad_at_1mps=.22,
            domain='forward speed (0.8,1.2), reconstructed abs yaw >0.4501',
            consumer='rmuc_position_authority.execute_authority_action; physical joint margins also required')
        if head.local_correction is not None:
            contract.update(local_authority_yaw_centers=head.local_correction.centers.detach().cpu().tolist(),
                            local_authority_yaw_half_width=float(head.local_correction.half_width))
    (directory / 'action_contract.json').write_text(json.dumps(contract, indent=2))
    return jit_path, onnx_path
