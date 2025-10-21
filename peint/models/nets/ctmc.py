import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.modules import ESM1bLayerNorm, gelu
from torch import Tensor

from peint.models.nets.peint import ESMEncoder


class RateMatrixOutputHead(nn.Module):
    """
    Output head that parameterizes CTMC rate matrix from ESM embeddings.
    Uses the Pande transformation to ensure detailed balance and fixed stationary distribution.
    """

    def __init__(self, embed_dim: int, num_states: int):
        super().__init__()
        self.num_states = num_states
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.theta_fc = nn.Linear(embed_dim, num_states)
        self.Theta_fc = nn.Linear(embed_dim, num_states * (num_states - 1) // 2)

    def forward(self, hx: Tensor) -> Tensor:
        # hx shape: (B, L, embed_dim)
        B, L, _ = hx.size()

        hx = self.layer_norm(gelu(self.dense(hx)))

        # Stationary distribution and symmetric S parameters
        pi = F.softmax(self.theta_fc(hx), dim=-1)  # (B, L, V)
        Theta = F.softplus(self.Theta_fc(hx))  # (B, L, V*(V-1)/2)

        # Build symmetric S matrix
        S = torch.zeros(B, L, self.num_states, self.num_states, device=hx.device)
        triu_indices = torch.triu_indices(self.num_states, self.num_states, offset=1)
        S[:, :, triu_indices[0], triu_indices[1]] = Theta
        S = S + S.transpose(-2, -1)

        # Pande transformation: Q = π^(-1/2) @ S @ π^(1/2)
        pi_sqrt = pi.sqrt()
        Q = torch.diag_embed(1.0 / pi_sqrt) @ S @ torch.diag_embed(pi_sqrt)
        Q = Q - torch.diag_embed(Q.sum(dim=-1))

        return Q


class NeuralCTMC(ESMEncoder):
    """Learn CTMC rate matrix with neural networks."""

    def __init__(self, *args, **kwargs):
        super().__init__(finetune=True, *args, **kwargs)
        self.in_embedding = self.get_in_embedding()
        self.output_head = self.get_out_lm_head()
        self.criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.vocab.pad_idx)

    def get_out_lm_head(self) -> nn.Module:
        # output layer parameterizes position-wise rate matrix
        return RateMatrixOutputHead(embed_dim=self.embed_dim, num_states=len(self.vocab))

    def forward(self, x: Tensor, x_sizes: Tensor) -> Tensor:
        if self.embed_x_per_chain:
            # Multi-sequence processing: compute ESM per chain
            hx = self.forward_per_chain(x, x_sizes)
        else:
            # Single-sequence processing: process concatenated sequence
            res = self.esm(x, repr_layers=[self.esm.num_layers], need_head_weights=False)
            hx = res["representations"][self.esm.num_layers]

        # hx shape: (B, L, esm_embed_dim) -> output rates
        Q = self.output_head(hx)  # (B, L, V, V)
        return Q
