# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch


def export_policy_as_jit(
        policy: object, obs_dict: dict[str, torch.Tensor], path: str, filename="policy.pt", device="cpu"
):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        obs_dict: input observation.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, device)
    policy_exporter.export(obs_dict, path, filename)


def export_policy_as_onnx(
        policy: object, obs_dict: dict[str, torch.Tensor], path: str, filename="policy.onnx", verbose=False,
        device="cpu"
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        obs_dict: input observation.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    os.makedirs(path, exist_ok=True)
    # copy policy parameters
    if hasattr(policy, "actor"):
        actor = copy.deepcopy(policy.actor).to(device)
    else:
        raise ValueError("Policy does not have an actor/student module.")
    obs_dict = {k: v[:1].to(device) for k, v in obs_dict.items()}
    with torch.no_grad():
        net_out = actor(obs_dict)
    torch.onnx.export(
        actor,
        (obs_dict, {}),
        os.path.join(path, filename),
        export_params=True,
        opset_version=11,
        # do_constant_folding=False,
        verbose=verbose,
        input_names=list(obs_dict.keys()),
        output_names=list(net_out.keys()),
        dynamic_axes=None,
    )


class _TorchPolicyExporter(torch.nn.Module):
    def __init__(self, policy, device):
        super().__init__()
        self.device = device

        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor).to(device)
        else:
            raise ValueError("Policy does not have an actor/student module.")

    def forward(self, obs_dict):
        return list(self.actor(obs_dict).values())

    def export(self, obs_dict, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        obs_dict = {k: v[:1].to(self.device) for k, v in obs_dict.items()}
        traced_script_module = torch.jit.trace(self, obs_dict)
        traced_script_module.save(path)