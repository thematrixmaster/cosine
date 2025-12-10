from typing import Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from esm.modules import ESM1bLayerNorm, gelu
from peint.models.nets.peint import ESMEncoder
from peint.models.frameworks.ctmc import MarkovBridgeSampler, BridgeTrajectory


class RateMatrixOutputHead(nn.Module):
    """
    Output head that parameterizes CTMC rate matrices from embeddings.
    Uses the Pande transformation to ensure detailed balance.
    
    The rate matrix Q satisfies:
    - Rows sum to zero: Σ_j Q_ij = 0
    - Off-diagonal entries non-negative: Q_ij ≥ 0 for i ≠ j
    - Detailed balance: π_i Q_ij = π_j Q_ji
    """

    def __init__(self, embed_dim: int, num_states: int):
        super().__init__()
        self.num_states = num_states
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.theta_fc = nn.Linear(embed_dim, num_states)
        self.Theta_fc = nn.Linear(embed_dim, num_states * (num_states - 1) // 2)

    def forward(self, hx: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            hx: Hidden states (B, L, embed_dim)
            
        Returns:
            Q: Rate matrices (B, L, V, V)
            pi: Stationary distributions (B, L, V)
        """
        B, L, _ = hx.size()
        hx = self.layer_norm(gelu(self.dense(hx)))

        # Stationary distribution
        pi = F.softmax(self.theta_fc(hx), dim=-1)  # (B, L, V)
        
        # Symmetric exchange parameters
        Theta = F.softplus(self.Theta_fc(hx))  # (B, L, V*(V-1)/2)

        # Build symmetric S matrix from upper triangular parameters
        S = torch.zeros(B, L, self.num_states, self.num_states, device=hx.device, dtype=hx.dtype)
        triu_idx = torch.triu_indices(self.num_states, self.num_states, offset=1, device=hx.device)
        S[:, :, triu_idx[0], triu_idx[1]] = Theta
        S = S + S.transpose(-2, -1)

        # Pande transformation: Q = diag(1/√π) S diag(√π) - diag(row_sums)
        pi_sqrt = pi.sqrt()
        Q = torch.diag_embed(1.0 / pi_sqrt) @ S @ torch.diag_embed(pi_sqrt)
        Q = Q - torch.diag_embed(Q.sum(dim=-1))

        return Q, pi


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

    def forward(self, x: Tensor, x_sizes: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tokens (B, L)
            x_sizes: Size of each chain
            
        Returns:
            Q: Rate matrices (B, L, V, V)
            pi: Stationary distributions (B, L, V)
        """
        if self.embed_x_per_chain:
            hx = self.forward_per_chain(x, x_sizes)
        else:
            res = self.esm(x, repr_layers=[self.esm.num_layers], need_head_weights=False)
            hx = res["representations"][self.esm.num_layers]

        return self.output_head(hx)

    def transition_probs(self, Q: Tensor, t: Tensor) -> Tensor:
        """Compute P(t) = exp(Qt) via matrix exponential."""
        B = Q.size(0)
        t = t.view(B, 1, 1, 1)
        return torch.matrix_exp(Q * t)

    def exp_Qt(self, Q: Tensor, t: Tensor) -> Tensor:
        """
        Performs matrix exponential of Q*t -> P_t
        Args:
            t: time tensor of shape (B,) or (B,1)
            Q: per position rate matrix tensor of shape (B, L, V, V)
        Returns:
            P: probability transition matrix of shape (B, L, V, V)
        """
        B = Q.size(0)
        t = t.reshape(B, 1, 1, 1)
        P = torch.matrix_exp(Q * t)  # (B, L, V, V)
        return P

    def log_Px(self, P: Tensor, x: Tensor) -> Tensor:
        # Indexes into P using current state x to get the log transition probabilities to other states
        batch_indices = torch.arange(x.size(0), device=x.device).unsqueeze(-1)  # (B, 1)
        seq_indices = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, L)
        P_selected = P[batch_indices, seq_indices, x]  # (B, L, V)
        log_probs = torch.log(P_selected + 1e-8)  # (B, L, V)
        return log_probs  # (B, L, V)

    @torch.no_grad()
    def perplexity(self, t: Tensor, x: Tensor, y: Tensor, x_sizes: Tensor) -> Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            Q, _ = self.forward(x, x_sizes=x_sizes)  # (B, L, V, V)
            P: Tensor = self.exp_Qt(Q, t)
            log_probs: Tensor = self.log_Px(P, x)

        # Keep unreduced to get per-site time likelihood (B, L)
        y_logits = log_probs.transpose(-1, -2)
        nll = F.cross_entropy(y_logits, y, ignore_index=self.vocab.pad_idx, reduction="none")

        y_tgt_mask = y != self.vocab.pad_idx
        nll_mean = (nll * y_tgt_mask).sum(dim=1) / y_tgt_mask.sum(dim=1)
        ppl = torch.exp(nll_mean)
        return ppl


class NeuralCTMCGenerator:
    """CTMC model with sequence generation methods."""

    def __init__(self, neural_ctmc: NeuralCTMC, *args, **kwargs):
        self.device = next(neural_ctmc.parameters()).device
        self.vocab = neural_ctmc.vocab
        special_tok_idxs = [
            self.vocab.bos_idx,
            self.vocab.pad_idx,
            self.vocab.eos_idx,
            self.vocab.unk_idx,
            self.vocab.mask_idx,
            self.vocab.tokens_to_idx["."],  # cannot change to separator token
        ]
        self.sp_tok_idxs = torch.tensor(special_tok_idxs, device=self.device)
        self.__neural_ctmc = neural_ctmc

    def __getattr__(self, name):
        return getattr(self.__neural_ctmc, name)

    @torch.no_grad()
    def generate_with_independent_sites(
        self,
        x: Tensor,
        t: Tensor,
        x_sizes: Tensor,
        temperature: float = 1.0,
        no_special_toks: bool = True,
    ) -> Tensor:
        """
        Generates sequences by simulating the CTMC over time t independently at each site.
        Args
            x: starting state vector (B, L)
            t: time tensor of shape (B,) or (B, 1)
            x_sizes: size of each chain in x (B, L)
        Returns
            y ~ exp(Q*t)[x,:]
        """
        Q, _ = self.forward(x, x_sizes=x_sizes)
        log_probs = self.log_Px(self.exp_Qt(Q, t), x)  # (B, L, V)

        if no_special_toks:
            log_probs[:, :, self.sp_tok_idxs] = -float("inf")
            log_probs = log_probs - log_probs.logsumexp(dim=-1, keepdim=True)

        probs = torch.exp(log_probs / temperature)  # (B, L, V)
        y = torch.distributions.Categorical(probs).sample()  # (B, L)

        dnc_mask = torch.isin(x, self.sp_tok_idxs)
        y[dnc_mask] = x[dnc_mask]
        return y

    @torch.no_grad()
    def generate_with_fake_gillespie(
        self,
        x: Tensor,
        t: Tensor,
        x_sizes: Tensor,
        temperature: float = 1.0,
        no_special_toks: bool = True,
        max_decode_steps: int = 1024,
    ) -> Tensor:
        """
        Gillespie simulation: iteratively sample mutations until time t is reached.

        At each step:
        1. Sample holding times: τ ~ Exp(-Q[i,i]) = -log(U) / |Q[i,i]|
        2. Choose position with minimum holding time
        3. Sample new state: p(j) = Q[i,j] / Σ_k Q[i,k] (k≠i)
        4. Update sequence and advance time
        """
        B, L = x.shape
        y = x.clone()
        target_t = t.view(-1)
        elapsed_t = torch.zeros_like(t).view(-1)
        active = torch.ones(B, dtype=torch.bool, device=x.device)
        dnc_mask = torch.isin(x, self.sp_tok_idxs)  # (B, L)

        for _ in tqdm(range(max_decode_steps)):
            if not active.any():
                break

            Q, _ = self.forward(y, x_sizes=x_sizes)
            exit_rates = -torch.diagonal(Q, dim1=-2, dim2=-1).gather(2, y.unsqueeze(-1)).squeeze(-1)

            holding_times = -torch.log(torch.rand_like(exit_rates) + 1e-10) / (
                exit_rates + 1e-10
            )  # (B, L)
            holding_times[dnc_mask] = float("inf")  # do not change special tokens

            min_times, min_pos = holding_times.min(dim=1)
            active &= (elapsed_t + min_times) < target_t
            if not active.any():
                break

            batch_idx = torch.arange(B, device=y.device)
            current_state = y[batch_idx, min_pos]
            rates = Q[batch_idx, min_pos, current_state].clamp(min=0.0)  # (B,)
            rates[batch_idx, current_state] = 0.0  # do not allow self-transition

            if no_special_toks:
                rates[:, self.sp_tok_idxs] = 0.0  # do not allow transition to special tokens

            probs = rates / (rates.sum(dim=-1, keepdim=True) + 1e-10)
            if temperature != 1.0:
                probs = (probs + 1e-10).pow(1.0 / temperature)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            new_state = torch.distributions.Categorical(probs).sample()
            y[active, min_pos[active]] = new_state[active]
            y[dnc_mask] = x[dnc_mask]
            elapsed_t[active] += min_times[active]

        return y
    
    @torch.no_grad()
    def sample_markov_bridge(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        x_sizes: Tensor,
        max_substitutions: int = 50,
    ) -> BridgeTrajectory:
        """
        Convenience function to sample bridge trajectories.
        
        Args:
            model: Neural CTMC model
            x: Starting sequences (B, L)
            y: Ending sequences (B, L)
            t: Branch lengths (B,) or (B, 1)
            x_sizes: Chain sizes for encoding
            max_substitutions: Maximum substitutions per position
            
        Returns:
            BridgeTrajectory with sampled evolutionary paths
        """
        sampler = MarkovBridgeSampler(self.__neural_ctmc)
        return sampler.sample(x, y, t, x_sizes, max_substitutions)
