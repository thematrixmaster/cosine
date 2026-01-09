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

    def __init__(self, embed_dim: int, num_states: int, valid_token_mask: Tensor):
        super().__init__()
        self.num_states = num_states
        self.register_buffer("valid_token_mask", valid_token_mask)  # (V,) boolean
        mask_2d = valid_token_mask.unsqueeze(0) & valid_token_mask.unsqueeze(1)
        self.register_buffer("valid_token_2d_mask", mask_2d) # (V, V)

        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.theta_fc = nn.Linear(embed_dim, num_states)
        self.Theta_fc = nn.Linear(embed_dim, num_states * (num_states - 1) // 2)

    def forward(self, hx: Tensor, return_log_pi: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Args:
            hx: Hidden states (B, L, embed_dim)
            return_log_pi: Whether to return the log of the stationary distribution
        Returns:
            Q: Rate matrices (B, L, V, V)
            pi: Stationary distributions (B, L, V) or log of stationary distributions (B, L, V) if return_log_pi is True
        """
        B, L, _ = hx.size()
        hx = self.layer_norm(gelu(self.dense(hx)))

        # Stationary distribution
        pi_logits = self.theta_fc(hx)
        pi_logits = pi_logits.masked_fill(~self.valid_token_mask, float('-inf'))
        log_pi = torch.log_softmax(pi_logits, dim=-1)  # (B, L, V)

        # Symmetric exchange parameters
        Theta = F.softplus(self.Theta_fc(hx))  # (B, L, V*(V-1)/2)

        # Build symmetric S matrix from upper triangular parameters
        S = torch.zeros(B, L, self.num_states, self.num_states, device=hx.device, dtype=hx.dtype)
        triu_idx = torch.triu_indices(self.num_states, self.num_states, offset=1, device=hx.device)
        S[:, :, triu_idx[0], triu_idx[1]] = Theta
        S = S + S.transpose(-2, -1)

        # Mask S before Pande transformation (no invalid state interactions)
        S = S * self.valid_token_2d_mask

        # Get a safe version of sqrt(pi) for Pande transform
        log_pi_safe = torch.where(self.valid_token_mask, log_pi, torch.zeros_like(log_pi))
        pi_sqrt = log_pi_safe.exp().sqrt()

        # Pande transformation: Q = diag(1/√π) S diag(√π) - diag(row_sums)
        Q = torch.diag_embed(1.0 / pi_sqrt) @ S @ torch.diag_embed(pi_sqrt)
        Q = Q - torch.diag_embed(Q.sum(dim=-1))

        return Q, log_pi if return_log_pi else log_pi.exp()
    
    
class FreeRateMatrixOutputHead(nn.Module):
    """
    Output head that parameterizes a general (non-reversible) CTMC rate matrix.
    
    The rate matrix Q satisfies:
    - Rows sum to zero: Σ_j Q_ij = 0
    - Off-diagonal entries non-negative: Q_ij ≥ 0 for i ≠ j
    - No detailed balance constraint (free rate matrix).
    """

    def __init__(self, embed_dim: int, num_states: int, valid_token_mask: Tensor):
        super().__init__()
        self.num_states = num_states
        self.register_buffer("valid_token_mask", valid_token_mask)  # (V,) boolean
        mask_2d = valid_token_mask.unsqueeze(0) & valid_token_mask.unsqueeze(1)
        self.register_buffer("valid_token_2d_mask", mask_2d) # (V, V)
        
        # Standard projection components
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        
        # Predicts V*V entries directly. We will mask the diagonal later.
        self.q_fc = nn.Linear(embed_dim, num_states * num_states)

    def forward(self, hx: Tensor, return_log_pi: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Args:
            hx: Hidden states (B, L, embed_dim)
            return_log_pi: Whether to return the log of the stationary distribution
        Returns:
            Q: Rate matrices (B, L, V, V)
            pi: Placeholder uniform distributions (B, L, V)
        """
        B, L, _ = hx.size()
        hx = self.layer_norm(gelu(self.dense(hx)))

        # Predict all potential rates, shape: (B, L, V, V)
        q_logits = self.q_fc(hx).view(B, L, self.num_states, self.num_states)
        Q_full = F.softplus(q_logits)   # non negative rates

        # Zero out the diagonal (self-transitions are calculated later)
        eye = torch.eye(self.num_states, device=hx.device, dtype=torch.bool)
        Q_off_diag = Q_full.masked_fill(eye, 0.0)

        # Mask invalid tokens (e.g., padding or special tokens)
        Q_off_diag = Q_off_diag * self.valid_token_2d_mask

        # Construct the final Rate Matrix Q
        row_sums = Q_off_diag.sum(dim=-1)
        Q = Q_off_diag - torch.diag_embed(row_sums) # Q_ii = - sum_{j!=i} Q_ij

        # Generate uniform distribution over valid tokens only
        num_valid = self.valid_token_mask.sum()
        uni_dist = torch.zeros_like(self.valid_token_mask, dtype=hx.dtype)
        uni_dist = uni_dist.masked_fill(self.valid_token_mask, 1.0 / num_valid)
        pi = uni_dist.unsqueeze(0).unsqueeze(0).expand(B, L, -1)

        if return_log_pi:
            log_pi = torch.log(pi + 1e-9) 
            log_pi = log_pi.masked_fill(~self.valid_token_mask, float('-inf'))
            return Q, log_pi
        else:
            return Q, pi


class NeuralCTMC(ESMEncoder):
    """Learn CTMC rate matrix with neural networks."""

    def __init__(self, reversible: bool = False, *args, **kwargs):
        super().__init__(finetune=True, *args, **kwargs)
        self.reversible = reversible
        self.in_embedding = self.get_in_embedding()
        self.output_head = self.get_out_lm_head()
        self.criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.vocab.pad_idx)

    def get_out_lm_head(self) -> nn.Module:
        _special_tok_idxs = torch.tensor([
            self.vocab.bos_idx,
            self.vocab.pad_idx,
            self.vocab.eos_idx,
            self.vocab.unk_idx,
            self.vocab.mask_idx,
            self.vocab.tokens_to_idx["<null_1>"],
            self.vocab.tokens_to_idx["."],
        ])
        valid_token_mask = torch.ones(len(self.vocab), dtype=torch.bool)
        valid_token_mask[_special_tok_idxs] = False
        if self.reversible:
            return RateMatrixOutputHead(
                embed_dim=self.embed_dim,
                num_states=len(self.vocab),
                valid_token_mask=valid_token_mask,
            )
        else:
            return FreeRateMatrixOutputHead(
                embed_dim=self.embed_dim,
                num_states=len(self.vocab),
                valid_token_mask=valid_token_mask,
            )

    def forward(self, x: Tensor, x_sizes: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
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

        return self.output_head(hx, *args, **kwargs)

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
            self.vocab.tokens_to_idx.get("<null_1>", -1),
            self.vocab.tokens_to_idx.get(".", -1),
            self.vocab.tokens_to_idx.get("X", -1),
            self.vocab.tokens_to_idx.get("B", -1),
            self.vocab.tokens_to_idx.get("Z", -1),
            self.vocab.tokens_to_idx.get("O", -1),
            self.vocab.tokens_to_idx.get("U", -1),
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
    def generate_with_adapted_gillespie(
        self,
        x: Tensor,
        t: Tensor,
        x_sizes: Tensor,
        temperature: float = 1.0,
        no_special_toks: bool = True,
        max_decode_steps: int = 1024,
        use_scalar_steps: bool = False,
        verbose: bool = False,
    ) -> Tensor:
        B, L = x.shape
        V = len(self.vocab)
        y = x.clone()
        
        target_t = t.view(-1)
        elapsed_t = torch.zeros_like(target_t)
        active = torch.ones(B, dtype=torch.bool, device=x.device)
        dnc_mask = torch.isin(x, self.sp_tok_idxs)

        # Setup scalar stepping if requested
        if use_scalar_steps:
            target_t = target_t.int()
            max_decode_steps = int(target_t.max().item())

        iterator = range(max_decode_steps)
        if verbose:
            import tqdm
            iterator = tqdm.tqdm(iterator)

        for _ in iterator:
            if not active.any():
                break

            # 1. Forward pass to get Rate Matrix Q: (B, L, V, V)
            # We assume Q[b, l, i, j] is the rate of transitioning from i -> j at pos l
            Q, _ = self.forward(y, x_sizes=x_sizes)
            
            # 2. Extract rates for the CURRENT state at every position
            # Create indices for advanced indexing
            batch_idx = torch.arange(B, device=x.device)[:, None]
            seq_idx = torch.arange(L, device=x.device)[None, :]
            
            # transition_rates shape: (B, L, V)
            transition_rates = Q[batch_idx, seq_idx, y]

            # 3. Mask invalid transitions
            # Zero out self-transitions (diagonal)
            transition_rates.scatter_(2, y.unsqueeze(-1), 0.0)
            
            if no_special_toks:
                # Zero out transitions TO special tokens
                sp_mask = torch.zeros((1, 1, V), device=x.device, dtype=torch.bool)
                sp_mask[..., self.sp_tok_idxs] = True
                transition_rates.masked_fill_(sp_mask, 0.0)

            # Apply temperature if needed (usually applied to logits before exp, 
            # but if Q are raw rates, strictly speaking temperature acts differently. 
            # Assuming Q are rates here.)
            
            # 4. Calculate Total Exit Rate (Lambda) per sequence
            # Flatten rates to (B, L*V) to treat the whole sequence as one system
            flat_rates = transition_rates.view(B, -1)
            total_exit_rate = flat_rates.sum(dim=-1)  # (B,)

            # 5. Sample Holding Time (tau)
            # Add epsilon to prevent div by zero
            safe_rate = total_exit_rate + 1e-10
            tau = -torch.log(torch.rand_like(safe_rate)) / safe_rate

            # Check if time budget exceeded
            if use_scalar_steps:
                # In scalar mode, we step by 1 unit, but we still use rates to pick mutation
                tau = torch.ones_like(tau)
            
            active &= (elapsed_t + tau) <= target_t
            if not active.any():
                break

            # 6. Sample Mutation (Gillespie Step)
            # We need to pick ONE index from the flattened (L*V) rates
            # torch.multinomial expects probabilities, so we normalize
            probs = flat_rates / (safe_rate.unsqueeze(-1))
            
            # Sample 1 event per batch
            flat_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)

            # 7. Update Sequence and Time
            # Decode flattened index back to (position, token)
            mutation_pos = flat_indices // V
            mutation_token = flat_indices % V

            # Only update active sequences
            batch_active_idx = torch.where(active)[0]
            
            y[batch_active_idx, mutation_pos[active]] = mutation_token[active]
            elapsed_t[active] += tau[active]

            # Reset fixed positions (if any)
            y[dnc_mask] = x[dnc_mask]

            if verbose:
                print(f"Time: {elapsed_t[active].mean().item():.4f}")

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
        use_scalar_steps: bool = False,
        verbose: bool = False,
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

        if use_scalar_steps:
            t = t.int()
            max_decode_steps = t.int().max().item()
            if max_decode_steps < 10:
                print(f"Warning: t is an integer number of discrete steps, but max_decode_steps is less than 10. This may be an error with the scalar steps t.")

        for _ in tqdm(range(max_decode_steps), disable=not verbose):
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

            if use_scalar_steps:
                elapsed_t[active] += 1
            else:
                elapsed_t[active] += min_times[active]

            if verbose:
                print(f"Time: {elapsed_t[active].mean().item():.4f}, Sequence: {self.vocab.decode(y[active])}")

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


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = NeuralCTMC.from_pretrained(embed_x_per_chain=True).to(device)
    sampler = MarkovBridgeSampler(model=model)
    vocab = model.vocab

    # Dummy data (start and end sequences)
    _x = ["ACDEFGHIK", "GVLMIP"]
    _y = ["MWVLSTQRN", "STYWKF"]
    sizes = [len(seq) for seq in _x]
    x = torch.from_numpy(vocab.encode_batched_sequences(_x)).to(device)
    y = torch.from_numpy(vocab.encode_batched_sequences(_y)).to(device)

    x_sizes = torch.zeros_like(x)
    x_sizes[:, 0] = torch.tensor(sizes, device=device)

    B, L = x.shape
    t = torch.rand(B, device=device) * 5.0  # Random branch lengths between 0 and 5

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        bridge_trajectories = sampler.sample(
            x=x,
            y=y,
            t=t,
            x_sizes=x_sizes,
            max_substitutions=20,
        )

    for b in range(B):
        print(f"Trajectory for sequence {b}:")
        num_events = bridge_trajectories.num_events[b]
        sequences = vocab.decode(bridge_trajectories.sequences[b, :num_events])
        times = bridge_trajectories.times[b][:num_events].tolist()

        for idx, (seq, time) in enumerate(zip(sequences, times)):
            if idx == 0:
                print(f"Time: {time:.4f}, Sequence: {seq} (start) == {_x[b]}")
            elif idx == len(sequences) - 1:
                print(f"Time: {time:.4f} == {t[b].item()}, Sequence: {seq} (end) == {_y[b]}")
            else:
                print(f"Time: {time:.4f}, Sequence: {seq}")
        print()
        
    breakpoint()
    