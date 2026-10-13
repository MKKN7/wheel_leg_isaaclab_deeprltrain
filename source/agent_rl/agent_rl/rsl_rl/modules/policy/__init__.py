# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for Actor modules."""

from .base import Actor, Critic, BaseModules
from .legged_policy import LeggedActor, LeggedCritic, LeggedActorScanDotCNN, LeggedCriticScanDotCNN #, LeggedActorB, LeggedActorScanDotCNNB, LeggedCriticScanDotCNNB
from .deploy_policy import DeployLeggedActor, VQVAEDeployLeggedActor, StateHistoryEncoder, MlpBarlowTwinsActor
