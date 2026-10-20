"""Command-gated residual adapter for the fixed 25-D RMUC actor contract."""

from __future__ import annotations

import copy
import types
from pathlib import Path

import torch
from torch import nn
from rsl_rl.utils import unpad_trajectories


# Each new turn stage gets its own zero-initialized residual.  A head is zero
# at the preceding certified speed, so later stages cannot change an earlier
# speed anchor even though their outputs are accumulated at higher speeds.
RMUC_TURN_DELTA_SPECS = (
    # command level, speed start/end, yaw start/end, label
    (28, 2.2, 2.4, 0.15, 0.30, "2p4_large"),
    (29, 2.4, 2.5, 0.02, 0.06, "2p5_small"),
    (30, 2.4, 2.5, 0.15, 0.30, "2p5_large"),
    (31, 2.5, 2.6, 0.02, 0.06, "2p6_small"),
    (32, 2.5, 2.6, 0.15, 0.30, "2p6_large"),
    (33, 2.6, 2.7, 0.02, 0.06, "2p7_small"),
    (34, 2.6, 2.7, 0.15, 0.30, "2p7_large"),
    (35, 2.7, 2.8, 0.02, 0.06, "2p8_small"),
    (36, 2.7, 2.8, 0.15, 0.30, "2p8_large"),
    (37, 2.8, 2.9, 0.02, 0.06, "2p9_small"),
    (38, 2.8, 2.9, 0.15, 0.30, "2p9_large"),
    (42, 2.9, 3.0, 0.02, 0.06, "3p0_small"),
    (43, 2.9, 3.0, 0.15, 0.30, "3p0_large"),
    (44, 3.0, 3.2, 0.02, 0.06, "3p2_small"),
    (45, 3.0, 3.2, 0.15, 0.30, "3p2_large"),
    # Repair the last inherited anchor before extending the bank.  This head
    # is zero for the independently certified 2.0 m/s large turn and for the
    # 2.2 m/s small turn, but can improve full-episode survival at Level 26.
    (26, 2.0, 2.2, 0.15, 0.30, "2p2_large_repair"),
    # Append only: preserve old checkpoint indices. Propulsion/posture
    # extensions transfer to future turns and are zero at earlier speeds.
    (39, 2.9, 3.0, 0.02, 0.06, "3p0_speed"),
    (40, 3.0, 3.1, 0.02, 0.06, "3p1_speed"),
    (41, 3.1, 3.2, 0.02, 0.06, "3p2_speed"),
)


def install_rmuc_position_authority(actor, state=None):
    """Explicit install or reconstruction only when checkpoint contains this head."""
    from rmuc_position_authority import PositionAuthorityHead
    key = 'rmuc_position_authority.positive.router.0.weight'
    if state is not None and key not in state:
        return None
    if not hasattr(actor, 'rmuc_turn_adapter'):
        raise ValueError('Position authority requires the complete composed actor')
    head = getattr(actor, 'rmuc_position_authority', None)
    if head is None:
        head = PositionAuthorityHead().to(actor.mlp[0].weight.device)
        actor.add_module('rmuc_position_authority', head)
    if state is not None and 'rmuc_position_authority.speed_plateau_end' in state:
        from rmuc_position_authority import checkpoint_authority_speed_plateau_end
        end = checkpoint_authority_speed_plateau_end(state)
        if not hasattr(head, 'speed_plateau_end'):
            head.register_buffer('speed_plateau_end', head.budget.new_tensor(end))
    if state is not None and 'rmuc_position_authority.local_correction.centers' in state:
        install_rmuc_local_authority(actor, state['rmuc_position_authority.local_correction.centers'])
    if state is not None and 'rmuc_physics_adaptation.network.0.weight' in state:
        install_rmuc_physics_adaptation(actor, state)
    if state is not None and 'rmuc_startup_adaptation.network.0.weight' in state:
        from rmuc_startup_adaptation import StartupAdaptation
        if not hasattr(actor, 'rmuc_startup_adaptation'):
            actor.add_module('rmuc_startup_adaptation', StartupAdaptation().to(actor.mlp[0].weight.device))
    return head


def install_rmuc_physics_adaptation(actor, state=None):
    """Explicit addition; old checkpoints never acquire physical correction."""
    from rmuc_physics_adaptation import PhysicsAdaptation
    if not hasattr(actor, 'rmuc_position_authority'):
        raise ValueError('Physics adaptation requires complete authority actor')
    head = getattr(actor, 'rmuc_physics_adaptation', None)
    migrated = state is not None and 'rmuc_physics_adaptation.frozen_source.network.0.weight' in state
    def reconstruct(prefix, depth=0):
        if depth > 3:
            raise ValueError('At most three frozen physical migration stages are supported')
        inner = reconstruct(prefix+'frozen_source.',depth+1) if prefix+'frozen_source.network.0.weight' in state else None
        module = PhysicsAdaptation(inner, update_domain=state.get(prefix+'update_domain'),
                                 scoped_network_domain=state.get(prefix+'scoped_network_domain'),
                                 positive_update_domain=state.get(prefix+'positive_update_domain'),
                                 positive_straight_enabled=state.get(prefix+'positive_straight_enabled'),
                                 reference_update_domain=state.get(prefix+'reference_update_domain'),
                                 frozen_reference_update_domain=state.get(prefix+'frozen_reference_update_domain'),
                                 network_update_domain=state.get(prefix+'network_update_domain'))
        if prefix+'alignment_network.0.weight' in state:
            if depth:
                raise ValueError('Alignment head is supported only at the outer physics stage')
            module.install_alignment_network(state[prefix+'alignment_budget'], state[prefix+'alignment_domain'])
        if prefix+'straight_extension_network.0.weight' in state:
            if depth:
                raise ValueError('Straight extension is supported only at outer physics stage')
            module.install_straight_extension(state[prefix+'straight_extension_budget'], state[prefix+'straight_extension_domain'])
        if prefix+'straight_bridge_enabled' in state:
            if float(state[prefix+'straight_bridge_enabled']) != 1.:
                raise ValueError('Explicit straight bridge flag must be one')
            module.install_straight_bridge()
        if prefix+'reverse_network.0.weight' in state:
            if depth:
                raise ValueError('Reverse branch is supported only at outer physics stage')
            module.install_reverse_network(state[prefix+'reverse_budget'], state[prefix+'reverse_domain'])
        return module
    if head is None:
        head = (reconstruct('rmuc_physics_adaptation.') if state is not None else PhysicsAdaptation()).to(actor.mlp[0].weight.device)
        actor.add_module('rmuc_physics_adaptation', head)
    elif migrated != hasattr(head, 'frozen_source') and state is not None:
        raise ValueError('Physical migration checkpoint structure differs from installed actor')
    return head


def install_rmuc_local_authority(actor, centers=None):
    """Opt-in local refinement; old checkpoints never gain this implicitly."""
    from rmuc_local_authority import LocalAuthorityCorrection
    head = getattr(actor, 'rmuc_position_authority', None)
    if head is None:
        raise ValueError('Local correction requires the existing authority head')
    if head.local_correction is None:
        head.local_correction = LocalAuthorityCorrection(centers).to(head.budget.device)
    elif centers is not None and not torch.equal(
            head.local_correction.centers,
            torch.as_tensor(centers, device=head.budget.device, dtype=head.local_correction.centers.dtype)):
        raise ValueError('Existing local centers differ; use explicit migration before strict restoration')
    return head.local_correction


def install_rmuc_state_router(actor, state=None):
    """Install explicitly, or reconstruct only when checkpoint contains a router."""
    from rmuc_state_router import IsolatedTurnStateRouter
    if not hasattr(actor, 'rmuc_turn_adapter'):
        raise ValueError('State router requires the full RMUC composed forward')
    key = 'rmuc_state_router.positive.router.0.weight'
    if state is not None and key not in state:
        return None
    existing = getattr(actor, 'rmuc_state_router', None)
    if existing is not None:
        return existing
    hidden = state[key].shape[0] if state is not None else 32
    count = state['rmuc_state_router.positive.router.2.weight'].shape[0] if state is not None else 4
    budget = state['rmuc_state_router.budget'] if state is not None else (.08,.08,.04,.08,.08,.04)
    recovery = state is not None and 'rmuc_state_router.recovery_positive_2p0.router.0.weight' in state
    straight_recovery = state is not None and 'rmuc_state_router.recovery_straight_3p0.router.0.weight' in state
    extended = state is not None and any(k.startswith('rmuc_state_router.extended_') for k in state)
    router = IsolatedTurnStateRouter(budget=budget, experts=count, hidden=hidden,
                                    recovery_positive_2p0=recovery,
                                    recovery_straight_3p0=straight_recovery,
                                    extended_turn_1p0=extended,
                                    extended_turn_seed=state is not None and 'rmuc_state_router.extended_positive_1p0_seed' in state,
                                    extended_turn_0p8=state is not None and 'rmuc_state_router.extended_positive_1p0_0p8.router.0.weight' in state,
                                    extended_positive_turn_1p0=state is not None and 'rmuc_state_router.extended_positive_1p0_1p0.router.0.weight' in state,
                                    e2e_turn_1p0=state is not None and 'rmuc_state_router.e2e_positive_1p0.router.0.weight' in state).to(actor.mlp[0].weight.device)
    # Backward-compatible reconstruction for v183 checkpoints created before
    # the explicit wheel seed buffer existed. Missing seeds MUST be zero:
    # nonzero defaults alter both the output and the learned residual budget.
    # Explicit seeds already stored in diagnostic checkpoints are preserved.
    if state is not None and recovery:
        state.setdefault('rmuc_state_router.recovery_positive_2p0_wheel_seed',
                         torch.zeros_like(router.recovery_positive_2p0_wheel_seed))
    if state is not None and 'rmuc_state_router.turn_speed_plateau_end' in state:
        end = float(state['rmuc_state_router.turn_speed_plateau_end'])
        if not 1.0 <= end <= 1.5:
            raise ValueError('Router turn speed plateau must be finite and in [1.0, 1.5]')
        router.register_buffer('turn_speed_plateau_end', router.budget.new_tensor(end))
    if state is not None and any(k.startswith('rmuc_state_router.speed_extensions.') for k in state):
        if not hasattr(router, 'turn_speed_plateau_end'):
            raise ValueError('Speed extensions require explicit router speed support')
        router.install_speed_extensions()
    if state is not None and 'rmuc_state_router.speed_extension_blend_bounds' in state:
        bounds=torch.as_tensor(state['rmuc_state_router.speed_extension_blend_bounds'])
        if (not hasattr(router, 'speed_extensions') or bounds.shape!=(2,)
                or not torch.isfinite(bounds).all() or not 1.0 <= float(bounds[0]) < float(bounds[1]) <= 1.5):
            raise ValueError('Explicit speed extension blend bounds must satisfy1<=start<end<=1.5')
        router.register_buffer('speed_extension_blend_bounds', bounds.to(router.budget).clone())
    if state is not None and 'rmuc_state_router.low_yaw_speed_blend' in state:
        domain=torch.as_tensor(state['rmuc_state_router.low_yaw_speed_blend'])
        if (not hasattr(router, 'speed_extensions') or domain.shape!=(4,)
                or not torch.isfinite(domain).all() or not 1.0 <= float(domain[0]) < float(domain[1]) <= 1.5
                or not 0.4501 <= float(domain[2]) < float(domain[3]) <= 1.0):
            raise ValueError('Invalid low-yaw speed blend contract')
        router.register_buffer('low_yaw_speed_blend',domain.to(router.budget).clone())
    if state is not None:
        anchor_present=any(k.startswith('rmuc_state_router.post_authority_anchors.') for k in state)
        bound=state.get('rmuc_state_router.post_authority_budget')
        if anchor_present != (bound is not None):
            raise ValueError('Post-authority migration requires both frozen anchors and budget')
        if anchor_present:
            router.install_post_authority_extensions(bound)
    actor.add_module('rmuc_state_router', router)
    return router


class RmucActuatorBridgeAdapter(nn.Module):
    """Zero-initialized residual used while migrating actuator dynamics."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        nn.init.orthogonal_(self.net[0].weight, gain=1.0)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class RmucSpeedDomainHeads(nn.Module):
    """Vectorized local heads: two adjacent speeds, five yaw regimes.

    At every command anchor exactly one speed head receives gradient. Batched
    gather/einsum avoids evaluating hundreds of separate Python MLP modules.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        positive = (1.0, 1.2, 1.35, 1.5, 1.6, 1.75, 2.0, 2.2, 2.4,
                    2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2)
        self.register_buffer("speed_knots", torch.tensor(sorted((*positive, *(-x for x in positive), 0.0))))
        count = len(self.speed_knots)
        self.weight1 = nn.Parameter(torch.empty(count, 5, hidden_dim, observation_dim))
        self.bias1 = nn.Parameter(torch.zeros(count, 5, hidden_dim))
        self.weight2 = nn.Parameter(torch.zeros(count, 5, action_dim, hidden_dim))
        self.bias2 = nn.Parameter(torch.zeros(count, 5, action_dim))
        for index in range(count):
            for regime in range(5):
                nn.init.orthogonal_(self.weight1[index, regime], gain=1.0)

    def forward(self, observations: torch.Tensor, regime_weights: torch.Tensor) -> torch.Tensor:
        shape = observations.shape[:-1]
        obs = observations.reshape(-1, observations.shape[-1])
        speed = (obs[:, 7] / 0.5).clamp(self.speed_knots[0], self.speed_knots[-1])
        # Comparison/reduction exports to ONNX, unlike searchsorted.
        upper = (speed[:, None] >= self.speed_knots[None, :]).long().sum(-1)
        upper = upper.clamp(1, self.speed_knots.numel() - 1)
        lower = upper - 1
        alpha = (speed - self.speed_knots[lower]) / (self.speed_knots[upper] - self.speed_knots[lower])
        alpha = alpha.square() * (3.0 - 2.0 * alpha)
        speed_weights = torch.stack((1.0 - alpha, alpha), dim=-1)
        indices = torch.stack((lower, upper), dim=-1)
        hidden = torch.nn.functional.elu(
            torch.einsum("bkrhi,bi->bkrh", self.weight1[indices], obs) + self.bias1[indices])
        actions = torch.einsum("bkrjh,bkrh->bkrj", self.weight2[indices], hidden) + self.bias2[indices]
        weights = speed_weights[:, :, None] * regime_weights.reshape(-1, 5)[:, None, :]
        output = (actions * weights[..., None]).sum(dim=(1, 2))
        return output.reshape(*shape, self.bias2.shape[-1])


class RmucDomainResidualBank(nn.Module):
    """Bounded, command-routed adaptation for flat-domain randomization."""

    HEAD_NAMES = (
        "straight",
        "positive_small_yaw",
        "negative_small_yaw",
        "positive_large_yaw",
        "negative_large_yaw",
    )

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                name: self._make_head(observation_dim, action_dim, hidden_dim)
                for name in self.HEAD_NAMES
            }
        )
        self.enabled = True

        self.register_buffer("legacy_straight_anchor_suppression", torch.zeros(()))
        self.register_buffer("positive_2p0_bank_seed", torch.zeros(action_dim))
        self.register_buffer("negative_2p0_yaw_seed", torch.zeros(action_dim))
        self.register_buffer("positive_2p4_bank_seed", torch.zeros(action_dim))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for seed_name in ("positive_2p0_bank_seed", "negative_2p0_yaw_seed", "positive_2p4_bank_seed"):
            seed_key = prefix + seed_name
            reference = getattr(self, seed_name)
            if seed_key not in state_dict:
                state_dict[seed_key] = torch.zeros_like(reference)
            seed = state_dict[seed_key]
            if seed.shape != reference.shape or not bool(torch.isfinite(seed).all()) or bool((seed.abs() > 0.24).any()):
                error_msgs.append("Local bank seed must be finite and bounded by0.24")
        key = prefix + "legacy_straight_anchor_suppression"
        if key not in state_dict:
            state_dict[key] = torch.zeros_like(self.legacy_straight_anchor_suppression)
        value = state_dict[key]
        if value.numel() != 1 or not bool(torch.isfinite(value).all()) or not bool(((value >= 0) & (value <= 1)).all()):
            error_msgs.append("Legacy straight suppression must be a finite scalar in [0,1]")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def enable_speed_isolation(self) -> None:
        if not hasattr(self, "speed_heads"):
            reference = self.heads["straight"]
            self.add_module("speed_heads", RmucSpeedDomainHeads(
                reference[0].in_features, reference[-1].out_features,
                reference[0].out_features).to(reference[0].weight.device))
        # Preserve the legacy correction exactly; only local additions adapt.
        for parameter in self.heads.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _make_head(
        observation_dim: int, action_dim: int, hidden_dim: int
    ) -> nn.Sequential:
        head = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        nn.init.orthogonal_(head[0].weight, gain=1.0)
        nn.init.zeros_(head[0].bias)
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    @staticmethod
    def _smooth_gate(value: torch.Tensor, start: float, end: float) -> torch.Tensor:
        fraction = ((value - start) / max(end - start, 1.0e-6)).clamp(0.0, 1.0)
        return fraction.square() * (3.0 - 2.0 * fraction)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(
                (*observations.shape[:-1], self.heads["straight"][-1].out_features),
                device=observations.device,
                dtype=observations.dtype,
            )
        target_yaw = observations[..., 2] / 0.25 + observations[..., 8]
        yaw_abs = target_yaw.abs()
        turn_gate = self._smooth_gate(yaw_abs, 0.02, 0.06)
        large_gate = self._smooth_gate(yaw_abs, 0.20, 0.25)
        straight_gate = 1.0 - turn_gate
        small_gate = turn_gate * (1.0 - large_gate)
        large_gate = turn_gate * large_gate
        positive = (target_yaw >= 0.0).to(observations.dtype)
        negative = (target_yaw < 0.0).to(observations.dtype)
        speed = observations[..., 7] / .5
        anchor_gate = self._smooth_gate(speed, 2.8, 2.9) * (1.0 - self._smooth_gate(speed, 2.9, 3.0))
        legacy_straight_gate = straight_gate * (1.0 - self.legacy_straight_anchor_suppression * anchor_gate)
        output = legacy_straight_gate.unsqueeze(-1) * self.heads["straight"](observations)
        output = output + (small_gate * positive).unsqueeze(-1) * self.heads[
            "positive_small_yaw"
        ](observations)
        output = output + (small_gate * negative).unsqueeze(-1) * self.heads[
            "negative_small_yaw"
        ](observations)
        output = output + (large_gate * positive).unsqueeze(-1) * self.heads[
            "positive_large_yaw"
        ](observations)
        output = output + (large_gate * negative).unsqueeze(-1) * self.heads[
            "negative_large_yaw"
        ](observations)
        if hasattr(self, "speed_heads"):
            weights = torch.stack((straight_gate, small_gate * positive, small_gate * negative,
                                   large_gate * positive, large_gate * negative), dim=-1)
            output = output + self.speed_heads(observations, weights)
        # The frozen skill graph remains dominant even if a domain head drifts.
        # Explicit frozen initialization at the failed positive 2.0 large-turn
        # anchor. Old checkpoints default to zero. Neighboring speed anchors and
        # small/negative yaw remain unchanged; domain heads still fine-tune.
        seed_gate = (self._smooth_gate(speed, 1.75, 2.0)
                     * (1.0 - self._smooth_gate(speed, 2.0, 2.2))
                     * self._smooth_gate(target_yaw, 0.25, 0.30))
        negative_seed_gate = (self._smooth_gate(speed, 1.75, 2.0)
                              * (1.0 - self._smooth_gate(speed, 2.0, 2.2))
                              * self._smooth_gate(-target_yaw, 0.25, 0.30))
        return (0.04 * torch.tanh(output) + seed_gate.unsqueeze(-1) * self.positive_2p0_bank_seed
                + negative_seed_gate.unsqueeze(-1) * self.negative_2p0_yaw_seed
                + self.positive_2p4_gate(observations).unsqueeze(-1) * self.positive_2p4_bank_seed)

    @classmethod
    def positive_2p4_gate(cls, observations: torch.Tensor) -> torch.Tensor:
        speed = observations[..., 7] / .5
        yaw = observations[..., 2] / .25 + observations[..., 8]
        return (cls._smooth_gate(speed, 2.2, 2.4)
                * (1.0 - cls._smooth_gate(speed, 2.4, 2.6))
                * cls._smooth_gate(yaw, .15, .30))


class RmucTurnResidualAdapter(nn.Module):
    """Add trainable capacity only when the reconstructed target yaw is nonzero."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        nn.init.orthogonal_(self.net[0].weight, gain=1.0)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.enabled = True

    @staticmethod
    def turn_gate(observations: torch.Tensor) -> torch.Tensor:
        # obs[2] = body yaw rate * 0.25; obs[8] = (target_yaw-yaw_rate) * 1.0.
        target_yaw = observations[..., 2] / 0.25 + observations[..., 8]
        gate = ((target_yaw.abs() - 0.02) / 0.06).clamp(0.0, 1.0)
        return gate.square() * (3.0 - 2.0 * gate)

    @staticmethod
    def residual_gate(observations: torch.Tensor) -> torch.Tensor:
        # obs[7] = target_vx * 0.50. Keep <=2.0 m/s turns exactly on the
        # certified turn expert and release trainable capacity above 2.0 m/s.
        target_vx = observations[..., 7] / 0.50
        speed_gate = ((target_vx - 2.0) / 0.20).clamp(0.0, 1.0)
        speed_gate = speed_gate.square() * (3.0 - 2.0 * speed_gate)
        return RmucTurnResidualAdapter.turn_gate(observations) * speed_gate

    @staticmethod
    def expert_gate(observations: torch.Tensor) -> torch.Tensor:
        # The low/medium turn expert is certified through 1.5 m/s. The base
        # actor is independently certified at 2.0 m/s large yaw. Blend only in
        # the untested interval instead of extrapolating either expert.
        target_vx = observations[..., 7] / 0.50
        base_fraction = ((target_vx - 1.5) / 0.40).clamp(0.0, 1.0)
        base_fraction = base_fraction.square() * (3.0 - 2.0 * base_fraction)
        return RmucTurnResidualAdapter.turn_gate(observations) * (
            1.0 - base_fraction
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(
                (*observations.shape[:-1], self.net[-1].out_features),
                device=observations.device,
                dtype=observations.dtype,
            )
        return self.residual_gate(observations).unsqueeze(-1) * self.net(observations)


class RmucTurnDeltaAdapter(nn.Module):
    """A residual head whose support starts above an already-certified anchor."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        *,
        speed_start: float,
        speed_end: float,
        yaw_start: float,
        yaw_end: float,
        directional: bool = False,
        speed_only: bool = False,
    ) -> None:
        super().__init__()
        self.speed_only = speed_only
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        nn.init.orthogonal_(self.net[0].weight, gain=1.0)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        if directional:
            self._install_directional_nets(observation_dim, action_dim, hidden_dim)
        self.register_buffer("speed_start", torch.tensor(float(speed_start)))
        self.register_buffer("speed_end", torch.tensor(float(speed_end)))
        self.register_buffer("yaw_start", torch.tensor(float(yaw_start)))
        self.register_buffer("yaw_end", torch.tensor(float(yaw_end)))
        self.register_buffer("positive_2p4_suppression", torch.zeros(()))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "positive_2p4_suppression"
        if key not in state_dict:
            state_dict[key] = torch.zeros_like(self.positive_2p4_suppression)
        value = state_dict[key]
        if value.shape != self.positive_2p4_suppression.shape or not bool(torch.isfinite(value).all()) or not bool(((value >= 0) & (value <= 1)).all()):
            error_msgs.append("Local stage suppression must be a finite scalar in [0,1]")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    @staticmethod
    def _make_zero_residual_net(
        observation_dim: int, action_dim: int, hidden_dim: int
    ) -> nn.Sequential:
        net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        nn.init.orthogonal_(net[0].weight, gain=1.0)
        nn.init.zeros_(net[0].bias)
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        return net

    def _install_directional_nets(
        self, observation_dim: int, action_dim: int, hidden_dim: int
    ) -> None:
        if hasattr(self, "positive_net"):
            return
        self.add_module(
            "positive_net",
            self._make_zero_residual_net(observation_dim, action_dim, hidden_dim),
        )
        self.add_module(
            "negative_net",
            self._make_zero_residual_net(observation_dim, action_dim, hidden_dim),
        )

    @staticmethod
    def _smooth_gate(value: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        fraction = ((value - start) / (end - start).clamp_min(1.0e-6)).clamp(0.0, 1.0)
        return fraction.square() * (3.0 - 2.0 * fraction)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        target_vx = observations[..., 7] / 0.50
        target_yaw = observations[..., 2] / 0.25 + observations[..., 8]
        speed_gate = self._smooth_gate(target_vx, self.speed_start, self.speed_end)
        yaw_gate = self._smooth_gate(target_yaw.abs(), self.yaw_start, self.yaw_end)
        if self.speed_only:
            yaw_gate = torch.ones_like(yaw_gate)
        gate = speed_gate * yaw_gate
        output = self.net(observations)
        if hasattr(self, "positive_net"):
            positive = (target_yaw > 0.0).to(observations.dtype).unsqueeze(-1)
            negative = (target_yaw < 0.0).to(observations.dtype).unsqueeze(-1)
            output = (
                output
                + positive * self.positive_net(observations)
                + negative * self.negative_net(observations)
            )
        local_scale = 1.0 - self.positive_2p4_suppression * RmucDomainResidualBank.positive_2p4_gate(observations)
        return (gate * local_scale).unsqueeze(-1) * output


def checkpoint_turn_adapter_hidden_dim(path: str | Path) -> int | None:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    state = payload.get("actor_state_dict") or payload.get("model_state_dict")
    if not isinstance(state, dict):
        return None
    weight = state.get("rmuc_turn_adapter.net.0.weight")
    return int(weight.shape[0]) if isinstance(weight, torch.Tensor) else None


def checkpoint_has_turn_delta_bank(path: str | Path) -> bool:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    state = payload.get("actor_state_dict") or payload.get("model_state_dict")
    return isinstance(state, dict) and any(
        key.startswith("rmuc_turn_delta_adapters.") for key in state
    )


def checkpoint_turn_delta_bank_size(path: str | Path) -> int | None:
    """Restore exactly the bank stored by old or extended checkpoints."""
    payload = torch.load(path, weights_only=False, map_location="cpu")
    state = payload.get("actor_state_dict") or payload.get("model_state_dict", {})
    indices = {int(key.split(".")[1]) for key in state
               if key.startswith("rmuc_turn_delta_adapters.")}
    if not indices:
        return None
    if indices != set(range(len(indices))) or len(indices) > len(RMUC_TURN_DELTA_SPECS):
        raise RuntimeError("Unsupported or non-contiguous RMUC residual bank checkpoint")
    return len(indices)


def checkpoint_has_directional_turn_delta_bank(path: str | Path) -> bool:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    state = payload.get("actor_state_dict") or payload.get("model_state_dict")
    return isinstance(state, dict) and any(
        key.startswith("rmuc_turn_delta_adapters.")
        and (".positive_net." in key or ".negative_net." in key)
        for key in state
    )


def checkpoint_has_actuator_bridge_adapter(path: str | Path) -> bool:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    state = payload.get("actor_state_dict") or payload.get("model_state_dict")
    return isinstance(state, dict) and any(
        key.startswith("rmuc_actuator_bridge_adapter.") for key in state
    )


def checkpoint_has_speed_expert(path: str | Path) -> bool:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    state = payload.get("actor_state_dict") or payload.get("model_state_dict")
    return isinstance(state, dict) and any(
        key.startswith("rmuc_speed_expert.") for key in state
    )


def checkpoint_has_domain_residual_bank(path: str | Path) -> bool:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    state = payload.get("actor_state_dict") or payload.get("model_state_dict")
    return isinstance(state, dict) and any(
        key.startswith("rmuc_domain_residual_bank.") for key in state
    )


def checkpoint_has_speed_isolated_domain_bank(path: str | Path) -> bool:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    state = payload.get("actor_state_dict") or payload.get("model_state_dict", {})
    return any(key.startswith("rmuc_domain_residual_bank.speed_heads.") for key in state)


def install_rmuc_domain_residual_bank(
    actor, *, hidden_dim: int = 32, optimizer=None, speed_isolated: bool = False
) -> RmucDomainResidualBank:
    """Install one bounded domain head per non-conflicting command regime."""

    existing = getattr(actor, "rmuc_domain_residual_bank", None)
    if existing is None:
        existing = RmucDomainResidualBank(
            int(actor.mlp[0].weight.shape[1]),
            int(actor.mlp[-1].weight.shape[0]),
            hidden_dim=hidden_dim,
        ).to(actor.mlp[0].weight.device)
        actor.add_module("rmuc_domain_residual_bank", existing)
    if speed_isolated:
        existing.enable_speed_isolation()
    if optimizer is not None:
        known = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        new_parameters = [
            parameter for parameter in existing.parameters() if id(parameter) not in known
        ]
        if new_parameters:
            optimizer.add_param_group({"params": new_parameters})
    existing.enabled = True
    return existing


def _load_speed_expert_mix_gain(module, state_dict, prefix, local_metadata, strict,
                                missing_keys, unexpected_keys, error_msgs):
    key = prefix + "rmuc_speed_expert_mix_gain"
    if key not in state_dict:
        # Old checkpoints always applied the full gate; preserve that behavior.
        state_dict[key] = torch.ones_like(module.rmuc_speed_expert_mix_gain)
    value = state_dict[key]
    if value.numel() != 1 or not bool(torch.isfinite(value).all()) or not bool(((value >= 0) & (value <= 1)).all()):
        error_msgs.append("RMUC speed expert mix gain must be a finite scalar in [0,1]")


def install_rmuc_speed_expert(
    actor, *, checkpoint_path: str | Path | None = None, optimizer=None
) -> nn.Module:
    """Embed a frozen straight-speed MLP without changing the 25-D contract."""

    existing = getattr(actor, "rmuc_speed_expert", None)
    if existing is None:
        expert = copy.deepcopy(actor.mlp)
        if checkpoint_path is not None:
            payload = torch.load(
                checkpoint_path, weights_only=False, map_location="cpu"
            )
            source_state = payload.get("actor_state_dict") or payload.get(
                "model_state_dict"
            )
            if not isinstance(source_state, dict):
                raise KeyError("speed expert checkpoint has no actor state")
            expert_state = {
                key.removeprefix("mlp."): value
                for key, value in source_state.items()
                if key.startswith("mlp.")
            }
            expert.load_state_dict(expert_state, strict=True)
        expert.to(actor.mlp[0].weight.device)
        expert.eval()
        for parameter in expert.parameters():
            parameter.requires_grad_(False)
        actor.add_module("rmuc_speed_expert", expert)
    else:
        expert = existing
    if not hasattr(actor, "rmuc_speed_expert_mix_gain"):
        actor.register_buffer("rmuc_speed_expert_mix_gain", torch.ones((), device=actor.mlp[0].weight.device))
        actor.register_load_state_dict_pre_hook(_load_speed_expert_mix_gain)
    actor._rmuc_speed_expert_enabled = True
    return expert


def install_rmuc_actuator_bridge_adapter(
    actor, *, hidden_dim: int = 64, optimizer=None
) -> RmucActuatorBridgeAdapter:
    existing = getattr(actor, "rmuc_actuator_bridge_adapter", None)
    if existing is not None:
        return existing
    observation_dim = int(actor.mlp[0].weight.shape[1])
    action_dim = int(actor.mlp[-1].weight.shape[0])
    adapter = RmucActuatorBridgeAdapter(
        observation_dim, action_dim, hidden_dim
    ).to(actor.mlp[0].weight.device)
    actor.add_module("rmuc_actuator_bridge_adapter", adapter)
    if optimizer is not None:
        known = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        new_parameters = [
            parameter for parameter in adapter.parameters() if id(parameter) not in known
        ]
        if new_parameters:
            optimizer.add_param_group({"params": new_parameters})
    return adapter


def install_rmuc_turn_adapter(
    actor,
    *,
    hidden_dim: int = 64,
    freeze_base: bool = True,
    optimizer=None,
    turn_expert_checkpoint: str | Path | None = None,
    enable_delta_bank: bool = False,
    enable_directional_bank: bool = False,
    bank_size: int | None = None,
) -> RmucTurnResidualAdapter:
    """Install an adapter while preserving the stock Gaussian actor API."""

    existing = getattr(actor, "rmuc_turn_adapter", None)
    if existing is not None:
        if enable_delta_bank and not hasattr(actor, "rmuc_turn_delta_adapters"):
            _install_turn_delta_bank(
                actor, hidden_dim, optimizer, directional=enable_directional_bank, bank_size=bank_size
            )
            _freeze_module(existing)
        elif enable_directional_bank:
            _install_directional_bank_nets(actor, optimizer)
        return existing
    observation_dim = int(actor.mlp[0].weight.shape[1])
    action_dim = int(actor.mlp[-1].weight.shape[0])
    if observation_dim != 25 or action_dim != 6:
        raise RuntimeError(
            f"RMUC turn adapter expects 25-D/6-D actor, got {observation_dim}/{action_dim}"
        )
    adapter = RmucTurnResidualAdapter(observation_dim, action_dim, hidden_dim).to(
        actor.mlp[0].weight.device
    )
    actor.add_module("rmuc_turn_adapter", adapter)
    turn_expert = copy.deepcopy(actor.mlp)
    actor.add_module("rmuc_turn_expert", turn_expert)
    if turn_expert_checkpoint is not None:
        payload = torch.load(
            turn_expert_checkpoint, weights_only=False, map_location="cpu"
        )
        source_state = payload.get("actor_state_dict") or payload.get(
            "model_state_dict"
        )
        if not isinstance(source_state, dict):
            raise KeyError("turn expert checkpoint has no actor state")
        expert_state = {
            key.removeprefix("mlp."): value
            for key, value in source_state.items()
            if key.startswith("mlp.")
        }
        turn_expert.load_state_dict(expert_state, strict=True)
    turn_expert.to(actor.mlp[0].weight.device)
    turn_expert.eval()
    for parameter in turn_expert.parameters():
        parameter.requires_grad_(False)
    actor._rmuc_turn_expert_enabled = True
    if freeze_base:
        for parameter in actor.mlp.parameters():
            parameter.requires_grad_(False)
    if enable_delta_bank:
        _install_turn_delta_bank(
            actor, hidden_dim, optimizer, directional=enable_directional_bank, bank_size=bank_size
        )
        _freeze_module(adapter)

    def _forward_with_turn_adapter(
        self,
        obs,
        masks=None,
        hidden_state=None,
        stochastic_output: bool = False,
    ):
        obs = (
            unpad_trajectories(obs, masks)
            if masks is not None and not self.is_recurrent
            else obs
        )
        latent = self.get_latent(obs, masks, hidden_state)
        base_output = self.mlp(latent)
        if self._rmuc_turn_expert_enabled:
            turn_gate = self.rmuc_turn_adapter.expert_gate(latent).unsqueeze(-1)
            expert_output = self.rmuc_turn_expert(latent)
            mlp_output = (
                base_output
                + turn_gate * (expert_output - base_output)
                + self.rmuc_turn_adapter(latent)
            )
            delta_adapters = getattr(self, "rmuc_turn_delta_adapters", None)
            if delta_adapters is not None:
                for delta_adapter in delta_adapters:
                    mlp_output = mlp_output + delta_adapter(latent)
        else:
            mlp_output = base_output
        speed_expert = getattr(self, "rmuc_speed_expert", None)
        if speed_expert is not None and getattr(
            self, "_rmuc_speed_expert_enabled", True
        ):
            target_vx = latent[..., 7] / 0.50
            target_yaw = latent[..., 2] / 0.25 + latent[..., 8]
            speed_gate_up = RmucTurnDeltaAdapter._smooth_gate(
                target_vx,
                torch.as_tensor(2.80, device=latent.device),
                torch.as_tensor(2.88, device=latent.device),
            )
            speed_gate_down = 1.0 - RmucTurnDeltaAdapter._smooth_gate(
                target_vx,
                torch.as_tensor(2.92, device=latent.device),
                torch.as_tensor(3.00, device=latent.device),
            )
            straight_gate = 1.0 - RmucTurnDeltaAdapter._smooth_gate(
                target_yaw.abs(),
                torch.as_tensor(0.02, device=latent.device),
                torch.as_tensor(0.06, device=latent.device),
            )
            speed_gate = (speed_gate_up * speed_gate_down * straight_gate
                          * self.rmuc_speed_expert_mix_gain).unsqueeze(-1)
            speed_output = speed_expert(latent)
            mlp_output = mlp_output + speed_gate * (speed_output - mlp_output)
        domain_residual_bank = getattr(self, "rmuc_domain_residual_bank", None)
        if domain_residual_bank is not None:
            mlp_output = mlp_output + domain_residual_bank(latent)
        bridge_adapter = getattr(self, "rmuc_actuator_bridge_adapter", None)
        if bridge_adapter is not None:
            mlp_output = mlp_output + bridge_adapter(latent)
        state_router = getattr(self, 'rmuc_state_router', None)
        if state_router is not None:
            mlp_output = mlp_output + state_router(latent)
        position_authority = getattr(self, 'rmuc_position_authority', None)
        if position_authority is not None:
            mlp_output = position_authority(latent, mlp_output)
        if state_router is not None and hasattr(state_router, 'post_authority_anchors'):
            if position_authority is None:
                raise ValueError('Post-authority correction requires an authority consumer')
            from rmuc_position_authority import execute_authority_action
            post_output = execute_authority_action(
                latent, mlp_output + state_router.post_authority_correction(latent),
                getattr(position_authority, 'speed_plateau_end', 1.0))
            pos,neg=position_authority.gates(latent,getattr(position_authority,'speed_plateau_end',1.0))
            mlp_output = torch.where((pos+neg)[...,None]>0,post_output,mlp_output)
        physics_adaptation = getattr(self, 'rmuc_physics_adaptation', None)
        if physics_adaptation is not None:
            mlp_output = physics_adaptation(latent, mlp_output)
        startup_adaptation = getattr(self, 'rmuc_startup_adaptation', None)
        if startup_adaptation is not None:
            mlp_output = startup_adaptation(latent, mlp_output)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    actor.forward = types.MethodType(_forward_with_turn_adapter, actor)
    if optimizer is not None:
        known = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        new_parameters = [
            parameter for parameter in adapter.parameters() if id(parameter) not in known
        ]
        if new_parameters:
            optimizer.add_param_group({"params": new_parameters})
    return adapter


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _install_turn_delta_bank(
    actor, hidden_dim: int, optimizer=None, *, directional: bool = False, bank_size: int | None = None
) -> nn.ModuleList:
    existing = getattr(actor, "rmuc_turn_delta_adapters", None)
    if existing is not None:
        return existing
    observation_dim = int(actor.mlp[0].weight.shape[1])
    action_dim = int(actor.mlp[-1].weight.shape[0])
    bank = nn.ModuleList(
        [
            RmucTurnDeltaAdapter(
                observation_dim,
                action_dim,
                hidden_dim,
                speed_start=speed_start,
                speed_end=speed_end,
                yaw_start=yaw_start,
                yaw_end=yaw_end,
                directional=directional and level not in (39, 40, 41),
                speed_only=level in (39, 40, 41),
            )
            for level, speed_start, speed_end, yaw_start, yaw_end, _ in RMUC_TURN_DELTA_SPECS[:bank_size]
        ]
    ).to(actor.mlp[0].weight.device)
    actor.add_module("rmuc_turn_delta_adapters", bank)
    _freeze_module(bank)
    if optimizer is not None:
        known = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        new_parameters = [
            parameter for parameter in bank.parameters() if id(parameter) not in known
        ]
        if new_parameters:
            optimizer.add_param_group({"params": new_parameters})
    return bank


def _install_directional_bank_nets(actor, optimizer=None) -> nn.ModuleList:
    bank = getattr(actor, "rmuc_turn_delta_adapters", None)
    if bank is None:
        raise RuntimeError("Directional turn bank requires the residual bank")
    observation_dim = int(actor.mlp[0].weight.shape[1])
    action_dim = int(actor.mlp[-1].weight.shape[0])
    for head in bank:
        if head.speed_only:
            continue
        hidden_dim = int(head.net[0].out_features)
        head._install_directional_nets(observation_dim, action_dim, hidden_dim)
        head.to(actor.mlp[0].weight.device)
    _freeze_module(bank)
    if optimizer is not None:
        known = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        new_parameters = [
            parameter for parameter in bank.parameters() if id(parameter) not in known
        ]
        if new_parameters:
            optimizer.add_param_group({"params": new_parameters})
    return bank


def set_rmuc_turn_delta_trainable(actor, command_level: int) -> str | None:
    """Freeze all certified heads and open only the head for the active level."""

    bank = getattr(actor, "rmuc_turn_delta_adapters", None)
    if bank is None:
        return None
    active_index = None
    active_label = None
    for index, (level, _, _, _, _, label) in enumerate(RMUC_TURN_DELTA_SPECS):
        if index >= len(bank):
            break
        trainable = int(command_level) == level
        head = bank[index]
        _freeze_module(head)
        if trainable:
            directional_modules = [
                module
                for module in (
                    getattr(head, "positive_net", None),
                    getattr(head, "negative_net", None),
                )
                if module is not None
            ]
            train_modules = directional_modules or [head.net]
            for module in train_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        if trainable:
            active_index = index
            active_label = label
    actor._rmuc_turn_delta_active_index = active_index
    return active_label


def disable_rmuc_turn_adapter(actor) -> None:
    adapter = getattr(actor, "rmuc_turn_adapter", None)
    if adapter is not None:
        adapter.enabled = False
    if hasattr(actor, "_rmuc_turn_expert_enabled"):
        actor._rmuc_turn_expert_enabled = False
    if hasattr(actor, "_rmuc_speed_expert_enabled"):
        actor._rmuc_speed_expert_enabled = False
    domain_bank = getattr(actor, "rmuc_domain_residual_bank", None)
    if domain_bank is not None:
        domain_bank.enabled = False
        _freeze_module(domain_bank)
    bank = getattr(actor, "rmuc_turn_delta_adapters", None)
    if bank is not None:
        _freeze_module(bank)
