"""CPU restore/export parity check for the complete 25-D/6-D RMUC actor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import onnx
from onnx.reference import ReferenceEvaluator
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

sys.path.insert(0, str(Path(__file__).resolve().parent / "rsl_rl"))
from rmuc_command_adapter import (  # noqa: E402
    install_rmuc_actuator_bridge_adapter, install_rmuc_domain_residual_bank, install_rmuc_state_router,
    install_rmuc_speed_expert, install_rmuc_turn_adapter, install_rmuc_position_authority,
    checkpoint_turn_delta_bank_size,
    checkpoint_has_speed_isolated_domain_bank,
)
from rmuc_policy_export import export_rmuc_policy  # noqa: E402


def restore_actor(checkpoint: Path) -> MLPModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("actor_state_dict") or payload["model_state_dict"]
    obs = TensorDict({"policy": torch.zeros(1, 25)}, batch_size=[1])
    actor = MLPModel(
        obs, {"actor": ["policy"]}, "actor", 6, hidden_dims=[128, 64, 32],
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": .004, "std_type": "scalar"},
    )
    if "rmuc_turn_adapter.net.0.weight" in state:
        hidden = state["rmuc_turn_adapter.net.0.weight"].shape[0]
        install_rmuc_turn_adapter(
            actor, hidden_dim=hidden,
            enable_delta_bank=any(k.startswith("rmuc_turn_delta_adapters.") for k in state),
            enable_directional_bank=any(".positive_net." in k for k in state),
            bank_size=checkpoint_turn_delta_bank_size(checkpoint),
        )
    if any(k.startswith("rmuc_speed_expert.") for k in state):
        install_rmuc_speed_expert(actor)
    if any(k.startswith("rmuc_domain_residual_bank.") for k in state):
        install_rmuc_domain_residual_bank(actor,
            speed_isolated=checkpoint_has_speed_isolated_domain_bank(checkpoint))
    if "rmuc_actuator_bridge_adapter.net.0.weight" in state:
        install_rmuc_actuator_bridge_adapter(
            actor, hidden_dim=state["rmuc_actuator_bridge_adapter.net.0.weight"].shape[0])
    if any(k.startswith('rmuc_state_router.') for k in state):
        install_rmuc_state_router(actor, state)
    install_rmuc_position_authority(actor, state)
    actor.load_state_dict(state, strict=True)
    from rmuc_training_contract import checkpoint_observation_contract
    actor.rmuc_observation_contract = checkpoint_observation_contract(payload)
    return actor.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    actor = restore_actor(args.checkpoint)
    jit_path, onnx_path = export_rmuc_policy(actor, args.output)
    onnx.checker.check_model(str(onnx_path))
    jit = torch.jit.load(str(jit_path))
    reference = ReferenceEvaluator(str(onnx_path))
    # Both signs, route boundaries, high-speed straight windows, and several
    # angular rates exercise the reconstructed command-yaw gates.
    speeds = torch.tensor([-1., 0., .8, .9, 1., 1.1, 1.2, 1.5, 1.7, 1.9, 2., 2.2, 2.4, 2.5,
                           2.6, 2.7, 2.8, 2.88, 2.9, 2.92, 3., 3.1, 3.2])
    yaws = torch.tensor([-2.5, -1.2, -1., -.8001, -.8, -.6, -.4501, -.45, -.3, -.25, -.2, -.15, -.06, -.02, 0., .02, .06, .15, .2, .25, .3, .45, .4501, .6, .8, .8001, 1., 1.2, 2.5])
    authority = getattr(actor, 'rmuc_position_authority', None)
    local = getattr(authority, 'local_correction', None)
    if local is not None:
        width = float(local.half_width)
        offsets = torch.tensor([-width-1e-4, -width, -width+1e-4, 0., width-1e-4, width, width+1e-4])
        yaws = torch.unique(torch.cat((yaws, (local.centers[:, None]+offsets).flatten())))
    if getattr(actor, 'rmuc_physics_adaptation', None) is not None:
        offsets=torch.tensor([-1e-4,0.,1e-4])
        # Reference scopes can occur in any inherited physical stage.
        stage=actor.rmuc_physics_adaptation
        while stage is not None:
            reference_domain=getattr(stage,'reference_update_domain',None)
            if reference_domain is not None:
                speeds=torch.unique(torch.cat((speeds,(reference_domain[:4,None]+offsets).flatten())))
                yaws=torch.unique(torch.cat((yaws,(reference_domain[4:,None]+offsets).flatten())))
            stage=getattr(stage,'frozen_source',None)
        scoped = getattr(actor.rmuc_physics_adaptation, 'scoped_network_domain', None)
        if scoped is not None:
            speeds=torch.unique(torch.cat((speeds,(scoped[:4,None]+offsets).flatten())))
            yaws=torch.unique(torch.cat((yaws,(torch.cat((-scoped[4:],scoped[4:]))[:,None]+offsets).flatten())))
        positive = getattr(actor.rmuc_physics_adaptation, 'positive_update_domain', None)
        if positive is not None:
            speed_knots=torch.cat((positive[:4],(positive[1:2]+positive[2:3])*.5,torch.tensor([.25])))
            yaw_knots=torch.cat((positive[4:],torch.tensor([0.,.1,.3])))
            speeds=torch.unique(torch.cat((speeds,(speed_knots[:,None]+offsets).flatten())))
            yaws=torch.unique(torch.cat((yaws,(torch.cat((-yaw_knots,yaw_knots))[:,None]+offsets).flatten())))
        speeds=torch.unique(torch.cat((speeds,(torch.tensor([-1.7,-1.5,1.5,1.7])[:,None]+offsets).flatten())))
        yaws=torch.unique(torch.cat((yaws,(torch.tensor([-4.2,-4.,4.,4.2])[:,None]+offsets).flatten())))
    commands = torch.cartesian_prod(speeds, yaws)
    x = torch.randn(len(commands), 25) * .1
    x[:, 5] = -1.
    x[:, 7] = commands[:, 0] * .5
    x[:, 8] = commands[:, 1] - x[:, 2] / .25
    results = []
    with torch.inference_mode():
        for batch in [x[:1], x[:7], x]:
            expected = actor({"policy": batch})
            actual_jit = jit(batch)
            actual_onnx = torch.from_numpy(reference.run(None, {"obs": batch.numpy()})[0])
            torch.testing.assert_close(actual_jit, expected, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(actual_onnx, expected, atol=2e-6, rtol=2e-5)
            results.append({"batch": len(batch),
                            "jit_max_abs_error": float((actual_jit - expected).abs().max()),
                            "onnx_max_abs_error": float((actual_onnx - expected).abs().max())})
        legacy_difference = float((actor({"policy": x}) - actor.as_onnx(False)(x)).abs().max())
    report = {"checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "passed": True, "backend": "onnx.reference.ReferenceEvaluator",
              "raw_normalized_actions_only": True, "parity": results,
              "legacy_base_only_max_abs_difference": legacy_difference}
    (args.output / "export_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
