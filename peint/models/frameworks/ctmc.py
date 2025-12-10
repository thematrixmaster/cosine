"""
Neural CTMC with Markov Bridge Sampling via Uniformization

This module implements:
1. Neural parameterization of position-wise CTMC rate matrices using ESM embeddings
2. Markov bridge sampling using the uniformization method (Hobolth 2008)

The bridge sampling algorithm reconstructs intermediate evolutionary trajectories
between aligned protein sequences by:
1. Uniformizing the CTMC to a discrete-time chain subordinated to a Poisson process
2. Sampling the number of state changes from the conditional distribution
3. Ancestral sampling of intermediate states
4. Merging per-position trajectories into full sequence trajectories
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass


# =============================================================================
# Eigendecomposition Utilities for Reversible Rate Matrices
# =============================================================================

def diagonalize_reversible_Q(
    Q: Tensor, 
    pi: Tensor, 
    eps: float = 1e-10
) -> Tuple[Tensor, Tensor]:
    """
    Diagonalize a reversible rate matrix Q using its stationary distribution.
    
    For a reversible CTMC with detailed balance (π_i Q_ij = π_j Q_ji),
    the symmetrized matrix Q_sym = diag(√π) Q diag(1/√π) is symmetric.
    We compute Q_sym = V Λ V^T via eigen-decomposition.
    
    Args:
        Q: Rate matrices (..., V, V)
        pi: Stationary distributions (..., V)
        eps: Small constant for numerical stability
        
    Returns:
        eigenvalues: (..., V) eigenvalues of Q (real, non-positive)
        V: (..., V, V) orthonormal eigenvectors of the symmetrized Q
    """
    pi_sqrt = (pi + eps).sqrt()
    pi_sqrt_inv = 1.0 / pi_sqrt
    
    # Symmetrize: Q_sym = diag(√π) @ Q @ diag(1/√π)
    Q_sym = torch.diag_embed(pi_sqrt) @ Q @ torch.diag_embed(pi_sqrt_inv)
    Q_sym = 0.5 * (Q_sym + Q_sym.transpose(-2, -1))  # Enforce symmetry
    
    # Real eigendecomposition (eigenvalues are real for symmetric matrices)
    eigenvalues, V = torch.linalg.eigh(Q_sym)
    
    return eigenvalues, V


def compute_transition_prob_element(
    eigenvalues: Tensor,
    V: Tensor,
    pi: Tensor,
    t: Tensor,
    i: Tensor,
    j: Tensor,
    eps: float = 1e-10,
) -> Tensor:
    """
    Compute P_ij(t) = [exp(Qt)]_ij for specific (i,j) pairs using eigendecomposition.
    
    Using P(t) = diag(1/√π) V exp(Λt) V^T diag(√π), we have:
    P_ij(t) = (1/√π_i) Σ_k V_ik V_jk exp(λ_k t) √π_j
    
    Args:
        eigenvalues: (..., V) eigenvalues
        V: (..., V, V) eigenvector matrix
        pi: (..., V) stationary distribution
        t: (...,) or broadcastable time values
        i: (...,) starting state indices
        j: (...,) ending state indices
        
    Returns:
        P_ij: (...,) transition probabilities
    """
    *batch_shape, vocab_size = eigenvalues.shape
    
    pi_sqrt = (pi + eps).sqrt()
    pi_sqrt_inv = 1.0 / pi_sqrt
    
    # Gather π values at i and j
    pi_sqrt_inv_i = torch.gather(pi_sqrt_inv, dim=-1, index=i.unsqueeze(-1)).squeeze(-1)
    pi_sqrt_j = torch.gather(pi_sqrt, dim=-1, index=j.unsqueeze(-1)).squeeze(-1)
    
    # Gather rows of V at i and j: V[..., i, :] and V[..., j, :]
    i_exp = i.unsqueeze(-1).unsqueeze(-1).expand(*batch_shape, 1, vocab_size)
    j_exp = j.unsqueeze(-1).unsqueeze(-1).expand(*batch_shape, 1, vocab_size)
    
    V_i = torch.gather(V, dim=-2, index=i_exp).squeeze(-2)  # (..., V)
    V_j = torch.gather(V, dim=-2, index=j_exp).squeeze(-2)  # (..., V)
    
    # exp(λ_k * t)
    t_expanded = t.unsqueeze(-1) if t.dim() < eigenvalues.dim() else t
    exp_lambda_t = torch.exp(eigenvalues * t_expanded)
    
    # P_ij(t) = (1/√π_i) * Σ_k V_ik V_jk exp(λ_k t) * √π_j
    inner_sum = (V_i * V_j * exp_lambda_t).sum(dim=-1)
    P_ij = pi_sqrt_inv_i * inner_sum * pi_sqrt_j
    
    return P_ij.clamp(min=eps, max=1.0)


# =============================================================================
# Uniformization Utilities
# =============================================================================

def compute_uniformization_rate(Q: Tensor, eps: float = 1e-10) -> Tensor:
    """
    Compute the uniformization rate μ = max_i(-Q_ii).
    
    Args:
        Q: Rate matrices (..., V, V)
        
    Returns:
        mu: (...,) uniformization rates
    """
    Q_diag = torch.diagonal(Q, dim1=-2, dim2=-1)
    return (-Q_diag).max(dim=-1).values.clamp(min=eps)


def compute_uniformized_transition_matrix(Q: Tensor, mu: Tensor) -> Tensor:
    """
    Compute the uniformized (discrete-time) transition matrix R = I + Q/μ.
    
    The matrix R is stochastic: rows sum to 1, entries in [0,1].
    Self-loops (R_ii > 0) represent "virtual" jumps in the uniformized chain.
    
    Args:
        Q: Rate matrices (..., V, V)
        mu: Uniformization rates (...,)
        
    Returns:
        R: Stochastic transition matrices (..., V, V)
    """
    V = Q.size(-1)
    I = torch.eye(V, device=Q.device, dtype=Q.dtype)
    # Expand mu to broadcast: (..., 1, 1)
    mu_expanded = mu.unsqueeze(-1).unsqueeze(-1)
    return I + Q / mu_expanded


def compute_matrix_powers(M: Tensor, max_power: int) -> Tensor:
    """
    Compute matrix powers M^0, M^1, ..., M^max_power.
    
    Args:
        M: Matrices (..., V, V)
        max_power: Maximum power to compute
        
    Returns:
        powers: (max_power+1, ..., V, V) where powers[n] = M^n
    """
    *batch_shape, V, _ = M.shape
    device, dtype = M.device, M.dtype
    
    powers = torch.zeros(max_power + 1, *batch_shape, V, V, device=device, dtype=dtype)
    powers[0] = torch.eye(V, device=device, dtype=dtype)
    
    if max_power >= 1:
        powers[1] = M
        for n in range(2, max_power + 1):
            powers[n] = powers[n - 1] @ M
            
    return powers


def sample_num_jumps(
    mu: Tensor,
    t: Tensor,
    P_ij_t: Tensor,
    R_powers: Tensor,
    i: Tensor,
    j: Tensor,
    max_n: int,
    eps: float = 1e-10,
) -> Tensor:
    """
    Sample the number of jumps N from the conditional distribution (eq 8):
    
    P(N=n | X_0=i, X_t=j) = exp(-μt) * (μt)^n / n! * (R^n)_ij / P_ij(t)
    
    Uses inverse CDF sampling.
    
    Args:
        mu: Uniformization rates (...,)
        t: Time values (...,)
        P_ij_t: Transition probabilities P_ij(t) (...,)
        R_powers: Precomputed R^n (max_n+1, ..., V, V)
        i: Starting states (...,)
        j: Ending states (...,)
        max_n: Maximum number of jumps
        
    Returns:
        N: Sampled number of jumps (...,)
    """
    batch_shape = i.shape
    device = i.device
    
    mu_t = mu * t
    u = torch.rand(batch_shape, device=device)
    
    # Initialize: N = max_n (fallback if CDF never exceeds u)
    N = torch.full(batch_shape, max_n, dtype=torch.long, device=device)
    cumprob = torch.zeros(batch_shape, device=device)
    sampled = torch.zeros(batch_shape, dtype=torch.bool, device=device)
    
    log_factorial = torch.tensor(0.0, device=device)
    
    for n in range(max_n + 1):
        if n > 0:
            log_factorial = log_factorial + torch.log(torch.tensor(float(n), device=device))
        
        # (R^n)_ij: gather R_powers[n][..., i, j]
        R_n = R_powers[n]  # (..., V, V)
        i_exp = i.unsqueeze(-1).unsqueeze(-1)
        R_n_i = torch.gather(R_n, dim=-2, index=i_exp.expand(*batch_shape, 1, R_n.size(-1))).squeeze(-2)
        R_n_ij = torch.gather(R_n_i, dim=-1, index=j.unsqueeze(-1)).squeeze(-1)
        
        # Poisson(μt) probability mass
        log_poisson = -mu_t + n * torch.log(mu_t + eps) - log_factorial
        
        # P(N=n | endpoints)
        prob_n = torch.exp(log_poisson) * R_n_ij / (P_ij_t + eps)
        cumprob = cumprob + prob_n
        
        # Sample where CDF first exceeds u
        newly_sampled = (cumprob >= u) & (~sampled)
        N = torch.where(newly_sampled, torch.tensor(n, device=device), N)
        sampled = sampled | newly_sampled
        
        if sampled.all():
            break
            
    return N


def sample_bridge_states(
    R: Tensor,
    R_powers: Tensor,
    i: Tensor,
    j: Tensor,
    N: Tensor,
    max_n: int,
    eps: float = 1e-10,
) -> Tensor:
    """
    Sample intermediate states using ancestral sampling (eq 9):
    
    P(X_{t_k}=b | X_{t_{k-1}}=a, X_{t_N}=j) = R_ab * (R^{N-k})_bj / (R^{N-k+1})_aj
    
    Args:
        R: Transition matrices (..., V, V)
        R_powers: Precomputed R^n (max_n+1, ..., V, V)
        i: Starting states (...,)
        j: Ending states (...,)
        N: Number of jumps per element (...,)
        max_n: Maximum jumps
        
    Returns:
        states: (..., max_n+1) sampled states where:
                states[..., 0] = i
                states[..., N] = j (and all positions after N)
    """
    *batch_shape, V, _ = R.shape
    device = R.device
    
    # Initialize output
    states = j.unsqueeze(-1).expand(*batch_shape, max_n + 1).clone()
    states[..., 0] = i
    
    current = i.clone()
    
    for k in range(1, max_n + 1):
        # Active positions: those where k < N (still have intermediate states to sample)
        active = k < N
        
        if not active.any():
            break
        
        # R[current, :] for all positions
        current_exp = current.unsqueeze(-1).unsqueeze(-1).expand(*batch_shape, 1, V)
        R_row = torch.gather(R, dim=-2, index=current_exp).squeeze(-2)  # (..., V)
        
        # For each possible remaining count, compute sampling probabilities
        probs = torch.zeros(*batch_shape, V, device=device)
        
        for remaining in range(max_n):
            # Mask for positions where N - k == remaining
            mask = ((N - k) == remaining) & active
            
            if not mask.any():
                continue
            
            # (R^remaining)[:, j]: probability from any state b to reach j in 'remaining' steps
            j_exp = j.unsqueeze(-1).unsqueeze(-2).expand(*batch_shape, V, 1)
            R_rem_to_j = torch.gather(R_powers[remaining], dim=-1, index=j_exp).squeeze(-1)  # (..., V)
            
            # (R^{remaining+1})[current, j]: normalizing constant
            current_exp2 = current.unsqueeze(-1).unsqueeze(-1).expand(*batch_shape, 1, V)
            R_rem1_row = torch.gather(R_powers[remaining + 1], dim=-2, index=current_exp2).squeeze(-2)
            j_exp2 = j.unsqueeze(-1)
            R_rem1_cj = torch.gather(R_rem1_row, dim=-1, index=j_exp2).squeeze(-1)  # (...,)
            
            # Probability: R[current, b] * R^rem[b, j] / R^{rem+1}[current, j]
            prob = R_row * R_rem_to_j / (R_rem1_cj.unsqueeze(-1) + eps)
            probs = torch.where(mask.unsqueeze(-1), prob, probs)
        
        # Normalize and sample
        probs = probs.clamp(min=0)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(probs_sum > eps, probs / probs_sum, torch.ones_like(probs) / V)
        
        new_state = torch.distributions.Categorical(probs).sample()
        
        # Update only active positions
        states[..., k] = torch.where(active, new_state, j)
        current = torch.where(active, new_state, current)
    
    return states


# =============================================================================
# Bridge Trajectory Output
# =============================================================================

@dataclass
class BridgeTrajectory:
    """Container for sampled bridge trajectories."""
    sequences: Tensor      # (B, T, L) tokenized sequences at each time point
    times: Tensor          # (B, T) corresponding times
    num_events: Tensor     # (B,) number of actual mutation events per batch
    
    def __repr__(self):
        B, T, L = self.sequences.shape
        return f"BridgeTrajectory(batch_size={B}, max_timepoints={T}, seq_len={L})"


# =============================================================================
# Markov Bridge Sampler
# =============================================================================

class MarkovBridgeSampler:
    """
    Samples evolutionary trajectories between endpoint sequences using uniformization.
    
    Given aligned sequences x (ancestor) and y (descendant) separated by time t,
    this sampler reconstructs plausible intermediate evolutionary paths by:
    
    1. Computing position-wise rate matrices Q from the neural CTMC
    2. Uniformizing each Q to get discrete transition matrix R = I + Q/μ
    3. Sampling the number of substitutions at each position from eq [8]
    4. Sampling intermediate states via ancestral sampling using eq [9]
    5. Merging per-position events into full sequence trajectories
    
    Reference: Hobolth & Stone (2009), "Simulation from endpoint-conditioned, 
               continuous-time Markov chains on a finite state space"
    """
    from peint.models.nets.ctmc import NeuralCTMC

    def __init__(self, model: NeuralCTMC):
        self.model = model
        self.device = next(model.parameters()).device
        self.vocab = model.vocab
        
        # Special tokens that should not be mutated
        self._special_tokens = torch.tensor([
            self.vocab.bos_idx,
            self.vocab.pad_idx,
            self.vocab.eos_idx,
            self.vocab.unk_idx,
            self.vocab.mask_idx,
            self.vocab.tokens_to_idx.get(".", -1),
        ], device=self.device)
        self._special_tokens = self._special_tokens[self._special_tokens >= 0]

    @torch.no_grad()
    def sample(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        x_sizes: Tensor,
        max_substitutions: int = 50,
    ) -> BridgeTrajectory:
        """
        Sample bridge trajectories between sequences x and y.
        
        Args:
            x: Starting sequences (B, L)
            y: Ending sequences (B, L)  
            t: Branch lengths (B,) or (B, 1)
            x_sizes: Chain sizes for ESM encoding
            max_substitutions: Maximum substitutions per position
            
        Returns:
            BridgeTrajectory containing merged sequence trajectories and times
        """
        B, L = x.shape
        V = len(self.vocab)
        t = t.view(B)
        
        # Mask for special tokens (don't mutate these)
        special_mask = torch.isin(x, self._special_tokens)  # (B, L)
        
        # ===== Step 1: Get rate matrices from neural network =====
        Q, pi = self.model(x, x_sizes=x_sizes)  # (B, L, V, V), (B, L, V)
        
        # ===== Step 2: Eigendecomposition for P_ij(t) =====
        eigenvalues, V_mat = diagonalize_reversible_Q(Q, pi)
        
        # ===== Step 3: Uniformization =====
        mu = compute_uniformization_rate(Q)  # (B, L)
        R = compute_uniformized_transition_matrix(Q, mu)  # (B, L, V, V)
        R_powers = compute_matrix_powers(R, max_substitutions)  # (max_n+1, B, L, V, V)
        
        # ===== Step 4: Compute endpoint transition probabilities =====
        # Expand t for broadcasting: (B,) -> (B, L)
        t_expanded = t.unsqueeze(1).expand(B, L)
        P_ij_t = compute_transition_prob_element(
            eigenvalues, V_mat, pi, t_expanded, x, y
        )  # (B, L)
        
        # ===== Step 5: Sample number of jumps per position =====
        N = sample_num_jumps(
            mu, t_expanded, P_ij_t, R_powers, x, y, max_substitutions
        )  # (B, L)
        N[special_mask] = 0  # No jumps for special tokens
        
        # ===== Step 6: Sample intermediate states =====
        states = sample_bridge_states(
            R, R_powers, x, y, N, max_substitutions
        )  # (B, L, max_n+1)
        
        # ===== Step 7: Sample jump times =====
        # Uniform times in [0, t], then sort
        raw_times = torch.rand(B, L, max_substitutions, device=self.device)
        raw_times = raw_times * t.view(B, 1, 1)
        jump_times, _ = torch.sort(raw_times, dim=-1)  # (B, L, max_n)
        
        # Mask invalid times (beyond N for each position)
        time_idx = torch.arange(max_substitutions, device=self.device)
        valid_time = time_idx.view(1, 1, -1) < N.unsqueeze(-1)  # (B, L, max_n)
        jump_times = torch.where(valid_time, jump_times, torch.full_like(jump_times, float('inf')))
        
        # ===== Step 8: Identify real (non-virtual) jumps =====
        # Virtual jump = self-transition (state doesn't change)
        state_changes = states[:, :, 1:] != states[:, :, :-1]  # (B, L, max_n)
        real_jump = state_changes & valid_time
        
        # ===== Step 9: Merge events across positions =====
        return self._merge_trajectories(x, y, t, states, jump_times, real_jump)

    def _merge_trajectories(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        states: Tensor,
        jump_times: Tensor,
        real_jump: Tensor,
    ) -> BridgeTrajectory:
        """
        Merge per-position events into full sequence trajectories.
        
        For each batch, collects all (time, position, new_state) events,
        sorts by time, and reconstructs the sequence at each timepoint.
        """
        B, L, max_n = jump_times.shape
        device = x.device
        
        # Maximum possible events per batch + 2 (start and end)
        max_events = (real_jump.sum(dim=(1, 2)).max().item()) + 2
        max_events = max(max_events, 2)  # At least start and end
        
        # Output tensors
        sequences = torch.zeros(B, max_events, L, dtype=torch.long, device=device)
        times = torch.zeros(B, max_events, device=device)
        num_events = torch.zeros(B, dtype=torch.long, device=device)
        
        for b in range(B):
            # Collect all real events for this batch
            events = []
            for l in range(L):
                for k in range(max_n):
                    if real_jump[b, l, k]:
                        event_time = jump_times[b, l, k].item()
                        new_state = states[b, l, k + 1].item()
                        events.append((event_time, l, new_state))
            
            # Sort by time
            events.sort(key=lambda e: e[0])
            
            # Build trajectory
            current_seq = x[b].clone()
            sequences[b, 0] = current_seq
            times[b, 0] = 0.0
            
            for idx, (event_time, pos, new_state) in enumerate(events):
                current_seq[pos] = new_state
                sequences[b, idx + 1] = current_seq.clone()
                times[b, idx + 1] = event_time
            
            # Final state = y
            n_events = len(events) + 2
            sequences[b, n_events - 1] = y[b]
            times[b, n_events - 1] = t[b].item()
            
            # Fill remaining slots
            if n_events < max_events:
                sequences[b, n_events:] = y[b]
                times[b, n_events:] = t[b].item()
            
            num_events[b] = n_events
        
        return BridgeTrajectory(sequences, times, num_events)
