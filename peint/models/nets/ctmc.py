from typing import Tuple, Optional, Callable
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from scipy.special import ndtr  # Normal CDF

from esm.modules import ESM1bLayerNorm, gelu
from evo.oracles import DifferentiableOracle, GaussianOracle
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
            y ~ exp(Q*t)[x,:] of shape (B, L)
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
    def generate_with_gillespie(
        self,
        x: Tensor,
        t: Tensor,
        x_sizes: Tensor,
        temperature: float = 1.0,
        no_special_toks: bool = True,
        max_decode_steps: int = 1024,
        use_scalar_steps: bool = False,
        verbose: bool = False,
        mask: Optional[Tensor] = None,
        use_guidance: bool = False,
        use_uniform_rates: bool = False,
        use_independent_sites: bool = False,
        oracle: Optional[GaussianOracle] = None,
        guidance_strength: float = 1.0,
        use_taylor_approx: bool = True,
        oracle_chunk_size: int = 5000,
        **kwargs,
    ) -> Tensor:
        """
        Generates sequences using Gillespie sampling with optional region masking and oracle guidance.

        Args:
            x: Starting sequences (B, L)
            t: Target time (B,) or (B, 1)
            x_sizes: Chain sizes for encoding (B,)
            temperature: Sampling temperature (default: 1.0)
            no_special_toks: Mask out special tokens from mutations (default: True)
            max_decode_steps: Maximum number of Gillespie steps (default: 1024)
            use_scalar_steps: Use fixed step size instead of exponential holding times (default: False)
            use_taylor_approx: Use Taylor approximation for oracle (faster but approximate) (default: True)
            oracle_chunk_size: Max sequences per oracle call to avoid OOM (default: 5000)
            verbose: Print progress information (default: False)
            mask: Optional boolean tensor (B, L) restricting mutations to specific regions
                  (e.g., CDR regions of antibodies). When provided, mutations can only occur
                  at positions where mask=True. Holding time calculation uses ALL sites.
                  Default: None (no restriction)
            use_guidance: Whether to use oracle guidance (default: False)
            use_uniform_rates: Dummy testing to use uniform rates before guidance is applied
            use_independent_sites: Samples a holding time for each site independently, then 
                  updates the residue at that site only using a matrix exponential. This is
                  a mixture between the independent sites and Gillespie sampling methods.
            oracle: GaussianOracle instance (required if use_guidance=True)
            guidance_strength: Guidance parameter γ. Higher values → stronger guidance

        Returns:
            y: Generated sequences (B*num_samples, L)

        Raises:
            ValueError: If mask shape doesn't match x shape or if all positions are masked
        """
        B, L = x.shape
        V = len(self.vocab)

        if use_uniform_rates and verbose:
            print("WARNING: use_uniform_rates is True. This is a dummy test that will use uniform rates before guidance is applied.")

        # Validate oracle if using guidance
        if use_guidance and oracle is None:
            raise ValueError("Oracle is required if use_guidance=True")
        if use_guidance and not isinstance(oracle, GaussianOracle):
            raise ValueError("Oracle must be a GaussianOracle instance if use_guidance=True")
        if use_guidance and use_independent_sites:
            raise ValueError("use_guidance and use_independent_sites cannot be True at the same time")

        # Validate mask if provided
        if mask is not None:
            if mask.shape != x.shape:
                raise ValueError(
                    f"mask shape {mask.shape} doesn't match x shape {x.shape}. "
                    f"Expected shape: ({B}, {L})"
                )
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

        for step_idx in iterator:
            if not active.any():
                break

            # Forward pass to get Rate Matrix Q: (B, L, V, V)
            if use_uniform_rates:
                Q = torch.ones((B, L, V, V), device=x.device) * 1e-3  # (B, L, V, V)
                if no_special_toks:
                    Q[:, :, self.sp_tok_idxs, :] = 0.0
                    Q[:, :, :, self.sp_tok_idxs] = 0.0
                Q.diagonal(dim1=-2, dim2=-1).fill_(0.0)
                row_sums = Q.sum(dim=-1)
                Q.diagonal(dim1=-2, dim2=-1).copy_(-row_sums)
            else:
                Q, _ = self.forward(y, x_sizes=x_sizes)

            # Apply oracle guidance to Q if needed
            if use_guidance:
                apply_guidance_fn = self._apply_exact_guidance if not use_taylor_approx else self._apply_taylor_guidance
                Q_guided = apply_guidance_fn(
                    Q=Q, y=y, oracle=oracle, guidance_strength=guidance_strength,
                    verbose=verbose,
                    oracle_chunk_size=oracle_chunk_size,
                )   # (B, L, V, V)
            else:
                # No guidance: use unguided Q
                Q_guided = Q

            # Extract transition rates for the CURRENT state at every position: (B, L, V)
            batch_idx = torch.arange(B, device=x.device)[:, None]
            seq_idx = torch.arange(L, device=x.device)[None, :]
            transition_rates = Q_guided[batch_idx, seq_idx, y]

            # Zero out self-transitions (diagonal): (B, L, V)
            transition_rates.scatter_(2, y.unsqueeze(-1), 0.0)

            if no_special_toks:
                # Zero out transitions TO special tokens: (B, L, V)
                sp_mask = torch.zeros((1, 1, V), device=x.device, dtype=torch.bool)
                sp_mask[..., self.sp_tok_idxs] = True
                transition_rates.masked_fill_(sp_mask, 0.0)

            if use_independent_sites:
                # Calculate Exit Rate (Lambda_i) per site: (B, L)
                exit_rates = transition_rates.sum(dim=-1)
                # Sample Holding Time (tau_i) for each site: (B, L)
                taus = -torch.log(torch.rand_like(exit_rates) + 1e-10) / (exit_rates + 1e-10)
                # Select the min and argmin of the holding times 
                tau, min_tau_idx = taus.min(dim=-1)
                assert tau.shape == (B,)
                assert min_tau_idx.shape == (B,)
            else:
                # Calculate Total Exit Rate (Lambda) per sequence: (B,)
                flat_rates = transition_rates.view(B, -1)
                total_exit_rate = flat_rates.sum(dim=-1)  # (B,)
                # Sample Holding Time (tau): (B,)
                safe_rate = total_exit_rate + 1e-10
                tau = -torch.log(torch.rand_like(safe_rate)) / safe_rate

            # Use fixed step size if requested: (B,)
            if use_scalar_steps:
                tau = torch.ones_like(tau).int()

            # Update active sequences: (B,)
            active &= (elapsed_t + tau) <= target_t
            if not active.any():
                break

            # Sample Mutation Position and Token
            if use_independent_sites:
                # Use matrix exponential
                # Q_site = Q_guided[batch_idx.squeeze(), min_tau_idx]  # (B, V, V)
                # y_site = y[batch_idx.squeeze(), min_tau_idx]  # (B,)
                # P_site = torch.matrix_exp(Q_site * tau.reshape(B, 1, 1))    # (B, V, V)
                # log_probs = torch.log(P_site[batch_idx.squeeze(), y_site] + 1e-8)     # (B, V)
                # # log_probs[batch_idx.squeeze(), y_site] = -float("inf") # zero out self-transitions
                # if no_special_toks:
                #     log_probs[:, self.sp_tok_idxs] = -float("inf")
                #     log_probs = log_probs - log_probs.logsumexp(dim=-1, keepdim=True)
                # probs = torch.exp(log_probs / temperature)

                # Use normalized rates in current row
                min_transition_rates = transition_rates[batch_idx.squeeze(), min_tau_idx]
                probs = min_transition_rates / min_transition_rates.sum(dim=-1, keepdim=True)

                # Save the mutation position and token
                mutation_pos = min_tau_idx
                mutation_token = torch.distributions.Categorical(probs).sample()

                if mask is not None:
                    can_mutate_mask = mask[batch_idx, mutation_pos] # true for positions that can be mutated
                    mutation_token[~can_mutate_mask] = x[batch_idx, mutation_pos][~can_mutate_mask]
            else:
                # Apply region mask if provided (affects mutation sampling only, NOT holding time)
                if mask is not None:
                    # Expand mask from (B, L) to (B, L*V) by repeating for each vocab token
                    # mask[b, l] applies to all possible token transitions at position l
                    flat_mask = mask.unsqueeze(-1).expand(-1, -1, V).reshape(B, -1)  # (B, L*V)

                    # Zero out rates for masked-out positions
                    masked_flat_rates = flat_rates.masked_fill(~flat_mask, 0.0)

                    # Check for sequences with no valid mutation positions
                    masked_sum = masked_flat_rates.sum(dim=-1)  # (B,)
                    has_no_valid_positions = masked_sum < 1e-10

                    if has_no_valid_positions.any():
                        invalid_batch_indices = torch.where(has_no_valid_positions)[0].tolist()
                        raise ValueError(
                            f"Sequences at batch indices {invalid_batch_indices} have no valid "
                            f"mutation positions (all positions are masked). Cannot sample mutations."
                        )

                    # Normalize to get probabilities
                    probs = masked_flat_rates / masked_sum.unsqueeze(-1)
                else:
                    # No mask: use all rates (original behavior)
                    probs = flat_rates / (safe_rate.unsqueeze(-1))

                # Sample 1 event per sequence
                flat_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)

                # Update sequence and time
                mutation_pos = flat_indices // V
                mutation_token = flat_indices % V

            # Update active sequences and time: (B, L)
            batch_active_idx = torch.where(active)[0]
            y[batch_active_idx, mutation_pos[active]] = mutation_token[active]
            elapsed_t[active] += tau[active]

            # Reset fixed positions (if any): (B, L)
            y[dnc_mask] = x[dnc_mask]

            if verbose:
                print(f"Step {step_idx}: Time={elapsed_t[active].min().item():.4f}")

        return y
    
    def _apply_exact_guidance(
        self,
        Q: Tensor,
        y: Tensor,
        oracle: GaussianOracle,
        guidance_strength: float,
        verbose: bool = False,
        oracle_chunk_size: int = 5000,
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Apply oracle guidance by evaluating on all single mutants (exact method).
        
        Vectorized version: collects ALL mutants across ALL sequences and makes
        chunked oracle calls for efficiency while avoiding OOM errors.

        Args:
            Q: Rate matrix (B, L, V, V)
            y: Current sequences (B, L)
            oracle: BaseOracle or DifferentiableOracle instance
            guidance_strength: Guidance parameter γ
            verbose: Print info
            oracle_chunk_size: Maximum number of sequences per oracle call (default: 5000)

        Returns:
            Q_guided: Guided rate matrix (B, L, V, V)
        """
        B, L, V, _ = Q.shape
        Q_guided = Q.clone()

        # Convert current sequences to strings
        parent_seqs = self.vocab.decode(y)

        # Evaluate oracle on parent sequences
        parent_means, _ = oracle.predict_batch(parent_seqs)
        parent_means = torch.tensor(parent_means, device=Q.device, dtype=torch.float32)

        if verbose:
            print(f"[Exact Guidance] Parent mean scores: {parent_means.cpu().numpy()}")

        # VECTORIZED: Generate ALL mutants for ALL sequences at once
        all_mutant_seqs = []
        mutant_metadata = []  # (batch_idx, position, token)
        seq_to_indices = {}  # Map unique sequences to their indices for deduplication

        # Define valid amino acids that the oracle can handle (standard 20 AAs)
        valid_aas = set("ARNDCQEGHILKMFPSTWYV")

        for b in range(B):
            for l in range(L):
                for v in range(V):
                    if v == y[b, l].item():
                        # Skip self-transitions
                        continue

                    # Skip tokens that are not standard amino acids
                    # (e.g., special tokens like <eos>, <pad>, gap tokens, extended AAs)
                    token_str = self.vocab.token(v)
                    if token_str not in valid_aas:
                        continue

                    # Create mutant sequence
                    mutant = y[b].clone()
                    mutant[l] = v
                    mutant_seq_str = self.vocab.decode_single_sequence(mutant.cpu().numpy())

                    # Track which index this sequence maps to
                    if mutant_seq_str not in seq_to_indices:
                        seq_to_indices[mutant_seq_str] = len(all_mutant_seqs)
                        all_mutant_seqs.append(mutant_seq_str)

                    mutant_metadata.append((b, l, v, seq_to_indices[mutant_seq_str]))

        if len(all_mutant_seqs) == 0:
            return Q_guided
        
        # CHUNKED oracle calls to avoid OOM
        num_mutants = len(all_mutant_seqs)
        num_chunks = (num_mutants + oracle_chunk_size - 1) // oracle_chunk_size

        if verbose:
            print(f"[Exact Guidance] Evaluating {num_mutants} mutants in {num_chunks} chunk(s) of size {oracle_chunk_size}")
    
        mutant_means_list = []
        mutant_variances_list = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * oracle_chunk_size
            end_idx = min(start_idx + oracle_chunk_size, num_mutants)
            chunk_seqs = all_mutant_seqs[start_idx:end_idx]

            chunk_means, chunk_variances = oracle.predict_batch(chunk_seqs)
            mutant_means_list.append(chunk_means)
            mutant_variances_list.append(chunk_variances)

        # Concatenate results from all chunks
        mutant_means = np.concatenate(mutant_means_list)
        mutant_variances = np.concatenate(mutant_variances_list)
        
        mutant_means = torch.tensor(mutant_means, device=Q.device, dtype=torch.float32)
        mutant_variances = torch.tensor(mutant_variances, device=Q.device, dtype=torch.float32)
        
        # Compute guidance weights for all mutants at once
        # Expand parent_means to match each mutant
        parent_means_expanded = torch.tensor(
            [parent_means[b].item() for b, l, v, seq_idx in mutant_metadata],
            device=Q.device, dtype=torch.float32
        )

        # Map unique sequence indices to their predictions
        mutant_means_expanded = torch.tensor(
            [mutant_means[seq_idx].item() for b, l, v, seq_idx in mutant_metadata],
            device=Q.device, dtype=torch.float32
        )
        mutant_variances_expanded = torch.tensor(
            [mutant_variances[seq_idx].item() for b, l, v, seq_idx in mutant_metadata],
            device=Q.device, dtype=torch.float32
        )

        z_scores = (parent_means_expanded - mutant_means_expanded) / torch.sqrt(mutant_variances_expanded + 1e-8)
        z_scores_np = z_scores.cpu().numpy()

        # Use scipy's ndtr for normal CDF (more stable than torch)
        probs = 1.0 - ndtr(z_scores_np)
        guidance_weights = (2.0 * probs) ** guidance_strength
        guidance_weights = torch.tensor(guidance_weights, device=Q.device, dtype=torch.float32)

        # Apply guidance weights to Q (vectorized)
        # Extract batch indices, positions, and tokens from metadata
        batch_indices = torch.tensor([b for b, l, v, seq_idx in mutant_metadata], device=Q.device, dtype=torch.long)
        position_indices = torch.tensor([l for b, l, v, seq_idx in mutant_metadata], device=Q.device, dtype=torch.long)
        to_tokens = torch.tensor([v for b, l, v, seq_idx in mutant_metadata], device=Q.device, dtype=torch.long)

        # Get current tokens for each mutant
        from_tokens = y[batch_indices, position_indices]

        # Vectorized multiplication: Q_guided[b, l, current_token, v] *= weight
        Q_guided[batch_indices, position_indices, from_tokens, to_tokens] *= guidance_weights

        return Q_guided

    def _apply_taylor_guidance(
        self,
        Q: Tensor,
        y: Tensor,
        oracle: GaussianOracle,
        guidance_strength: float,
        verbose: bool = False,
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Apply oracle guidance using Taylor approximation (gradient-based).

        Implements Eq 9:
        weight = [2 * Phi( (Δμ) / σ )]^γ
        where Δμ ≈ ∇μ(x) · (x_mutant - x_parent)

        Complexity: O(B) backward passes instead of O(B*L*V) forward passes.

        Args:
            Q: Rate matrix (B, L, V, V)
            y: Current sequences (B, L)
            oracle: Must implement compute_fitness_gradient()
            guidance_strength: Gamma factor
        """
        # Ensure oracle supports gradient computation
        if not hasattr(oracle, "compute_fitness_gradient"):
             raise ValueError("Oracle provided to _apply_taylor_guidance must implement 'compute_fitness_gradient'.")

        B, L, V, _ = Q.shape
        Q_guided = Q.clone()
        
        # Convert indices to strings for the oracle
        seq_strs = self.vocab.decode(y)
        
        if verbose:
            print(f"[Taylor Guidance] Computing gradients for {B} sequences...")

        # Process batch
        # Note: We loop over B because gradients are typically computed per-sample 
        # or require specific graph retention which is tricky to fully vectorize 
        # without a custom batch-grad oracle implementation.
        for b in range(B):
            # 1. Compute Gradients and Variance (1 backward pass)
            # grads: (L, V_oracle) matrix where grads[l, v] is d(score)/d(one_hot[l, v])
            # variance: scalar estimate of σ²(x)
            grads_oracle_np, variance_val = oracle.compute_fitness_gradient(seq_strs[b])

            # Map gradients from oracle vocab (25 AAs) to CTMC vocab (33 tokens with special tokens)
            # Create mapping from CTMC vocab indices to oracle vocab indices
            from evo.oracles.covid_model import AA_VOCAB as ORACLE_VOCAB

            L_seq = grads_oracle_np.shape[0]  # Length without special tokens (e.g., 119)
            grads = torch.zeros(L, V, device=Q.device, dtype=Q.dtype)  # (L_full, V_ctmc) where L_full includes special tokens (e.g., 121)

            # Build mapping from full sequence positions to amino acid positions
            # aa_positions[i] = position in y[b] for the i-th amino acid in the decoded sequence
            valid_aas = set("ARNDCQEGHILKMFPSTWYV")
            aa_positions = []
            for pos_idx, token_idx in enumerate(y[b]):
                token = self.vocab.token(token_idx.item())
                if token in valid_aas:
                    aa_positions.append(pos_idx)

            # Verify mapping is correct
            if len(aa_positions) != L_seq:
                raise ValueError(
                    f"Mismatch: decoded sequence has {L_seq} amino acids but found "
                    f"{len(aa_positions)} amino acid positions in y[b]. "
                    f"Sequence length: {L}, vocab tokens: {[self.vocab.token(t.item()) for t in y[b]]}"
                )

            # Map oracle gradients to the full sequence positions
            # grads_oracle_np[aa_idx, v_oracle] is the gradient for the aa_idx-th amino acid
            # This should go to position aa_positions[aa_idx] in the full sequence
            # Convert oracle gradients to torch tensor once
            grads_oracle_torch = torch.from_numpy(grads_oracle_np).to(device=Q.device, dtype=Q.dtype)

            for v_ctmc in range(V):
                aa = self.vocab.token(v_ctmc)
                if aa in ORACLE_VOCAB:
                    v_oracle = ORACLE_VOCAB[aa]
                    for aa_idx in range(L_seq):
                        full_pos = aa_positions[aa_idx]
                        grads[full_pos, v_ctmc] = grads_oracle_torch[aa_idx, v_oracle]
                # else: gradient remains 0 for special tokens

            sigma = torch.tensor(np.sqrt(variance_val + 1e-10), device=Q.device, dtype=Q.dtype)

            # 2. Compute Δμ for all potential mutations
            # We only care about transitions FROM the current state y[b]
            # Δμ(u->v) = Gradient(v) - Gradient(u)
            
            current_indices = y[b] # (L,)
            
            # Get the gradient value of the CURRENT amino acid at each position
            # gather requires matching dims, so we unsqueeze to (L, 1)
            current_grads = grads.gather(1, current_indices.unsqueeze(-1)) # (L, 1)
            
            # Broadcast subtract: (L, V) - (L, 1) -> (L, V)
            # This gives us the predicted change in fitness for mutating to any token v
            delta_mu = grads - current_grads
            
            # 3. Calculate Guidance Weights
            # Z-score = Δμ / σ
            z_scores = delta_mu / sigma
            
            # Compute CDF Phi(z)
            # torch.special.ndtr is the standard normal CDF
            probs = torch.special.ndtr(z_scores)
            
            # Weight = (2 * Prob)^γ
            weights = (2.0 * probs).pow(guidance_strength) # (L, V)

            # 4. Update Rate Matrix
            # Q is (B, L, V, V) where last dims are (from, to).
            # We update the row corresponding to transition FROM current_indices.
            
            # Create row indices [0, 1, ... L-1]
            l_indices = torch.arange(L, device=Q.device)
            
            # Update: Q[b, l, current_token, :] *= weights[l, :]
            Q_guided[b, l_indices, current_indices, :] *= weights

        return Q_guided

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
    