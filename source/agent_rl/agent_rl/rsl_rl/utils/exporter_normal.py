# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
import torch
import torch.nn as nn


def resolve_policy_export_dir(checkpoint_path: str) -> str:
    """与训练 checkpoint 同目录，用于放置 ``model.pt`` / ``model.onnx`` 等部署产物。"""
    return os.path.dirname(os.path.abspath(checkpoint_path))


class _BarlowTwinsActorExportWrapper(nn.Module):
    """ONNX / TorchScript 用：双输入 ``(obs_prop, obs_hist)``，与 ``MlpBarlowTwinsActor.forward`` 一致。"""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, obs_prop: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs_prop, obs_hist)


def export_mlp_barlow_twins_actor_torchscript(
    backbone: nn.Module,
    num_prop: int,
    num_hist: int,
    export_dir: str,
    *,
    device: str | torch.device = "cpu",
    filename: str = "model.pt",
    use_fp16: bool = False,
) -> str:
    """导出 ``MlpBarlowTwinsActor``（或同接口子模块）为 TorchScript，对齐 ``actor_critic_balowtwins`` 中注释写法。

    Args:
        backbone: ``MlpBarlowTwinsActor`` 实例（建议已加载权重）。
        num_prop: 本体观测维度，与 ``act_teacher`` 中 ``obs_prop`` 一致（全维 ``num_prop``，非 ``num_prop-3``）。
        num_hist: 历史帧数，与 ``obs_hist`` 的时间维长度一致。
        export_dir: 输出目录。
        device: trace 所用设备。
        filename: 默认 ``model.pt``。
        use_fp16: 为 true 时将子模块与示例输入转为 half 再 trace。
    """
    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, filename)
    m = copy.deepcopy(backbone).to(device).eval()
    dtype = torch.float16 if use_fp16 else torch.float32
    if use_fp16:
        m = m.half()
    obs_prop = torch.randn(1, num_prop, device=device, dtype=dtype)
    obs_hist = torch.randn(1, num_hist, num_prop, device=device, dtype=dtype)
    wrapper = _BarlowTwinsActorExportWrapper(m).to(device)
    if use_fp16:
        wrapper = wrapper.half()
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, (obs_prop, obs_hist))
    traced.save(out_path)
    return out_path


def export_mlp_barlow_twins_actor_onnx(
    backbone: nn.Module,
    num_prop: int,
    num_hist: int,
    export_dir: str,
    *,
    device: str | torch.device = "cpu",
    filename: str = "model.onnx",
    opset_version: int = 13,
    verbose: bool = False,
) -> str:
    """导出同构策略为 ONNX（float32），输入名 ``nn_input0`` / ``nn_input1``，输出 ``nn_output``。"""
    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, filename)
    m = copy.deepcopy(backbone).to(device).eval()
    obs_prop = torch.randn(1, num_prop, device=device, dtype=torch.float32)
    obs_hist = torch.randn(1, num_hist, num_prop, device=device, dtype=torch.float32)
    wrapper = _BarlowTwinsActorExportWrapper(m).to(device)
    with torch.inference_mode():
        _ = wrapper(obs_prop, obs_hist)
    torch.onnx.export(
        wrapper,
        (obs_prop, obs_hist),
        out_path,
        input_names=["obs", "obs_hist"],
        output_names=["actions"],
        export_params=True,
        opset_version=opset_version,
        verbose=verbose,
    )
    return out_path


def export_barlow_twins_actor_from_policy(
    policy: nn.Module,
    export_dir: str,
    *,
    device: str | torch.device = "cpu",
    use_fp16_jit: bool = False,
    jit_filename: str = "policy.pt",
    onnx_filename: str = "policy.onnx",
) -> tuple[str, str]:
    """从 ``ActorCriticBarlowTwins``（NP3O 的 ``alg.policy``）导出 ``actor_teacher_backbone`` 为 TorchScript / ONNX。

    输出目录应与 play 中其他任务的 ``.../exported`` 一致。
    """
    os.makedirs(export_dir, exist_ok=True)
    backbone = policy.actor_teacher_backbone
    num_prop = int(getattr(policy, "num_prop"))
    num_hist = int(getattr(policy, "num_hist"))
    pt_path = export_mlp_barlow_twins_actor_torchscript(
        backbone,
        num_prop,
        num_hist,
        export_dir,
        device=device,
        filename=jit_filename,
        use_fp16=use_fp16_jit,
    )
    onnx_path = export_mlp_barlow_twins_actor_onnx(
        backbone,
        num_prop,
        num_hist,
        export_dir,
        device=device,
        filename=onnx_filename,
    )
    return pt_path, onnx_path


def export_actor_critic_barlow_twins_actor(
    policy: nn.Module,
    checkpoint_path: str,
    *,
    device: str | torch.device = "cpu",
    use_fp16_jit: bool = False,
    jit_filename: str = "barlow_twins_actor.pt",
    onnx_filename: str = "barlow_twins_actor.onnx",
) -> tuple[str, str]:
    """兼容入口：导出到 checkpoint 文件所在目录。"""
    export_dir = resolve_policy_export_dir(checkpoint_path)
    return export_barlow_twins_actor_from_policy(
        policy,
        export_dir,
        device=device,
        use_fp16_jit=use_fp16_jit,
        jit_filename=jit_filename,
        onnx_filename=onnx_filename,
    )

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
    obs_dict = {k: v[:1].to(device) for k, v in obs_dict.items()}
    obs_keys = list(obs_dict.keys())
    obs_values = [obs_dict[k] for k in obs_keys]
    
    exporter = _OnnxPolicyExporter(policy, device, obs_keys)
    with torch.no_grad():
        net_out = exporter(*obs_values)
    output_names = ["actions"]
    torch.onnx.export(
        exporter,
        tuple(obs_values),
        os.path.join(path, filename),
        export_params=True,
        opset_version=11,
        verbose=verbose,
        input_names=obs_keys,
        output_names=output_names,
        dynamic_axes=None,
    )


class _TorchPolicyExporter(torch.nn.Module):
    def __init__(self, policy, device):
        super().__init__()
        self.device = device
        self.policy = copy.deepcopy(policy).to(device)

    def forward(self, obs_dict):
        out = self.policy.act_inference(obs_dict)
        return [out] if isinstance(out, torch.Tensor) else list(out.values())

    def export(self, obs_dict, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        obs_dict = {k: v[:1].to(self.device) for k, v in obs_dict.items()}
        traced_script_module = torch.jit.trace(self, obs_dict)
        traced_script_module.save(path)

class _OnnxPolicyExporter(torch.nn.Module):
    """ONNX exporter wrapper that accepts dict inputs."""
    def __init__(self, policy, device, obs_keys):
        super().__init__()
        self.device = device
        self.policy = copy.deepcopy(policy).to(device)
        self._use_act_inference = hasattr(self.policy, "act_inference")
        self.obs_keys = obs_keys

    def forward(self, *args):
        # Reconstruct dict from positional arguments
        obs_dict = {key: args[i] for i, key in enumerate(self.obs_keys)}
        if self._use_act_inference:
            out = self.policy.act_inference(obs_dict)
            return out if isinstance(out, torch.Tensor) else out.get("actions", list(out.values())[0])
        actor_out = self.policy.actor(obs_dict)
        return actor_out.get("actions", actor_out) if isinstance(actor_out, dict) else actor_out