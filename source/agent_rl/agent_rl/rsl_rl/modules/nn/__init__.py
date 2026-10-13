# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for NN modules."""

from .mlp import Identity, MLP, MLPBatchNorm
from .vqvae import VQVAE
from .cnn import CustomCNN
from .normalizer import EmpiricalNormalization

__all__ = [
    "Identity",
    "MLP",
    "VQVAE",
    "CustomCNN",
    "EmpiricalNormalization",
]
