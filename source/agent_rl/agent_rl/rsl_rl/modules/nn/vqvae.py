# =============================================================================
# This file contains code derived from the following third-party projects.
#
#   [rsl_rl / legged_gym]
#     License  : BSD-3-Clause
#     Copyright: Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
#
#   [DreamWaQ]
#     URL      : https://github.com/Manaro-Alpha/DreamWaQ
#
# =============================================================================

import torch
from torch import nn
import torch.nn.functional as F

from .mlp import MLP


class VQVAE(nn.Module):
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim,
            num_embeddings,
            embedding_dim,
            vq_hidden_dims,
            activation,
    ):
        super(VQVAE, self).__init__()

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = MLP(state_dim + goal_dim, embedding_dim, vq_hidden_dims, activation)
        self.decoder = MLP(state_dim + embedding_dim, action_dim, vq_hidden_dims, activation)
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1, 1)

        self.quantized = None
        self.encoding = None
        self.encoding_indices = None

    def forward(self, state, goal):
        encoding = self.encoder(torch.cat([state, goal], dim=-1))
        encoding_indices = self.compute_encoding_indices(encoding)
        quantized = self.quantize(encoding_indices)
        # Straight-Through Estimator
        z = encoding + (quantized - encoding).detach()
        action = self.decoder(torch.cat([state, z], dim=-1))
        self.quantized = quantized
        self.encoding = encoding
        self.encoding_indices = encoding_indices
        return action

    def quantize(self, encoding_indices):
        return self.embeddings(encoding_indices)

    def compute_encoding_indices(self, encoding):
        distances = (
                torch.sum(encoding ** 2, dim=1, keepdim=True) +
                torch.sum(self.embeddings.weight ** 2, dim=1) -
                2. * torch.matmul(encoding, self.embeddings.weight.t())
        )
        return torch.argmin(distances, dim=1)

    def compute_vq_loss(self):
        q_latent_loss = F.mse_loss(self.quantized, self.encoding.detach())
        e_latent_loss = F.mse_loss(self.quantized.detach(), self.encoding)
        with torch.inference_mode():
            encoding_onehot = F.one_hot(self.encoding_indices, self.num_embeddings).float()
            avg_probs = torch.mean(encoding_onehot, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))) / len(avg_probs)
            # perplexity_loss = 1 / (perplexity + 1e-10)
        return {
            "q_latent": q_latent_loss,
            "e_latent": e_latent_loss,
            "perplexity": perplexity,
        }
