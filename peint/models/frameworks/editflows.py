import random
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from evo.dataset import CollatableVocabDataset
from evo.tokenization import Vocab
from torch import Tensor
from tqdm import tqdm

from peint.models.frameworks.dfm import Coupling, DiscreteSampler

#####################################################################################################
# Coupling Classes
#####################################################################################################


class EmptyCoupling(Coupling):
    """A coupling that samples empty prior sequences"""

    def sample(self, x1: Tensor):
        x0 = torch.empty((x1.shape[0], 0), dtype=x1.dtype, device=x1.device).long()
        return x0, x1


class GeneratorCoupling(Coupling):
    """A coupling that samples prior sequences from a generator function"""

    def __init__(self, generator_fn: Callable[[Optional[Tensor]], Tensor]):
        self.generator_fn = generator_fn

    def sample(self, x1: Tensor):
        x0 = self.generator_fn(x1)
        return x0, x1


class PureBirthCoupling(Coupling):
    """A coupling that inserts tokens into the input sequence following a position-wise Poisson process"""

    def __init__(
        self,
        vocab: Vocab,
        gamma: float = 1 / 1.1,
        t_min: float = 1e-5,
        n_stationary_samples: int = 2e6,
        stationary_toks=list(range(4, 24)),
    ):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.t_min = t_min
        self.alpha_0 = t_min ** (gamma / (1 - t_min))
        self.n_stationary_samples = n_stationary_samples
        self.stationary_toks = torch.LongTensor(stationary_toks)

    def setup(self, dataset: CollatableVocabDataset, *args, **kwargs):
        p0 = torch.ones(self.vocab_size)
        pbar = tqdm(total=self.n_stationary_samples)
        for x1 in dataset:
            if p0.sum() > self.n_stationary_samples:
                break
            new = (
                F.one_hot(x1, num_classes=self.vocab_size)
                .to(torch.float32)
                .view((-1, self.vocab_size))
                .sum(0)
            )
            p0 += new
            pbar.update(new.sum().item())

        pbar.close()
        non_stationary_mask = torch.ones(self.vocab_size, dtype=torch.bool)
        non_stationary_mask[self.stationary_toks] = False
        p0[non_stationary_mask] = 0
        self.p0 = p0 / p0.sum()

    def get_stationary(self):
        return self.p0

    def sample_point_given_M(self, x: torch.Tensor, M: torch.Tensor):
        Ls = (x != self.vocab.pad_idx).sum(dim=1)
        B = x.shape[0]

        max_len = int((M + Ls).max())

        # Set up x_0 with random elements
        x_0 = (
            torch.multinomial(self.get_stationary(), B * max_len, replacement=True)
            .view(B, max_len)
            .to(x.device)
            .to(x.dtype)
        )
        x_0[:, 0] = self.vocab.bos_idx

        # Insert original sequences from x into randomly-selected indices of x_0
        for i in range(B):
            insertion_indices = 1 + torch.randperm(int(M[i] + Ls[i] - 2))[: Ls[i] - 2].sort().values
            x_0[i, insertion_indices] = x[i, 1 : int(Ls[i]) - 1]  # don't copy over bos/eos
            x_0[i, int(M[i] + Ls[i]) - 1] = self.vocab.eos_idx
            x_0[i, int(M[i] + Ls[i]) :] = self.vocab.pad_idx

        return x_0

    def sample(self, x1: torch.Tensor):
        if x1.ndim == 1:
            x1 = x1.unsqueeze(0)
        Ls = (x1 != self.vocab.pad_idx).sum(dim=1) - 2

        p = torch.full((Ls.shape[0],), 1 - self.alpha_0, device=x1.device)
        dist = torch.distributions.NegativeBinomial(Ls + 1, p)
        M = dist.sample()

        x0 = self.sample_point_given_M(x1, M)
        return x0, x1, M


class UniformCoupling(Coupling):
    """A coupling that samples uniform prior sequences within a given length range"""

    def __init__(
        self,
        min_len: int = 0,
        max_len: int = 100,
        vocab_size: int = 128,
        mirror_len: bool = False,
        pad_token: int = 0,
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.mirror_len = mirror_len
        self.pad_token = pad_token

    def sample(self, x1: Tensor):
        batch_size, _ = x1.shape
        x1_pad_mask = x1 == self.pad_token
        if self.mirror_len:
            x0_pad_mask = x1_pad_mask
            x0_max_len = x1.shape[1]
        else:
            x0_seq_len = torch.randint(self.min_len, self.max_len + 1, size=(batch_size,)).long()
            x0_max_len = int(x0_seq_len.max().item())
            x0_pad_mask = torch.arange(x0_max_len, device=x1.device).expand(
                batch_size, -1
            ) >= x0_seq_len.unsqueeze(1)

        x0 = torch.randint(
            0, self.vocab_size, size=(batch_size, x0_max_len), dtype=x1.dtype, device=x1.device
        )
        x0[x0_pad_mask] = self.pad_token
        return x0, x1


######################################################################################################
# Alignment Classes
######################################################################################################


def remove_and_pad(tensor: Tensor, mask: Tensor, pad_token_id: int):
    # tensor: (batch, seq)
    out = []
    for row, m in zip(tensor, mask):
        filtered = row[m]
        out.append(filtered)
    max_len = max(len(row) for row in out)
    out = [F.pad(row, (0, max_len - len(row)), value=pad_token_id) for row in out]
    return torch.stack(out, dim=0)


class Alignment(ABC):
    """Base class for pairwise sequence alignment methods."""

    def __init__(
        self,
        vocab: Vocab,
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        gap_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.bos_token_id = vocab.bos_idx
        self.eos_token_id = vocab.eos_idx
        self.pad_token_id = vocab.pad_idx
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.gap_token_id = gap_token_id if gap_token_id is not None else len(vocab)

    def pre(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Removes bos and eos tokens if present, and repads the tensors"""
        assert x.shape[0] == y.shape[0], "Input tensors must have the same batch size"
        x_mask = (x != self.bos_token_id) & (x != self.eos_token_id)
        y_mask = (y != self.bos_token_id) & (y != self.eos_token_id)
        x_out = remove_and_pad(x, x_mask, self.pad_token_id)
        y_out = remove_and_pad(y, y_mask, self.pad_token_id)
        return x_out, y_out

    @abstractmethod
    def align(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Batch aligns the pairwise sequences in `x` and `y`.

        Args:
            x (Tensor): First sequence tensor.
            y (Tensor): Second sequence tensor.

        Returns:
            Tuple[Tensor, Tensor]: Aligned sequences as tensors.
        """

    def post(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Adds bos and eos tokens to the aligned sequences, if necessary"""
        assert x.shape[0] == y.shape[0], "Input tensors must have the same batch size"
        if self.add_bos_token:
            x = F.pad(x, (1, 0), value=self.bos_token_id)
            y = F.pad(y, (1, 0), value=self.bos_token_id)
        if self.add_eos_token:
            # scatter the eos token to the end of each sequence
            new_x = torch.full(
                (x.shape[0], x.shape[1] + 1), self.pad_token_id, dtype=x.dtype, device=x.device
            )
            new_y = torch.full(
                (y.shape[0], y.shape[1] + 1), self.pad_token_id, dtype=y.dtype, device=y.device
            )
            new_x[:, :-1] = x
            new_y[:, :-1] = y
            x_seq_lens = (x != self.pad_token_id).sum(dim=1)
            y_seq_lens = (y != self.pad_token_id).sum(dim=1)
            batch_indices = torch.arange(x.shape[0], device=x.device)
            new_x[batch_indices, x_seq_lens] = self.eos_token_id
            new_y[batch_indices, y_seq_lens] = self.eos_token_id
            x, y = new_x, new_y
        return x, y

    def __call__(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x, y = self.pre(x, y)
        xp, yp = self.post(x, y)
        z0, z1 = self.align(x, y)
        assert z0.shape == z1.shape, "Aligned sequences must have the same shape"
        z0, z1 = self.post(z0, z1)
        assert (
            z0.shape == z1.shape
        ), "Aligned sequences must have the same shape after post-processing"
        return xp, yp, z0, z1


class NaiveAlignment(Alignment):
    """Naively aligns sequences by padding them with the GAP token"""

    def align(self, x, y):
        x_seq_lens = (x != self.pad_token_id).sum(dim=1)
        y_seq_lens = (y != self.pad_token_id).sum(dim=1)
        z_seq_lens = torch.maximum(x_seq_lens, y_seq_lens)
        z_max_len = z_seq_lens.max().item()

        # Create masks for efficient assignment
        pos_mask = torch.arange(z_max_len, device=x.device)[None, :]
        xp = torch.where(
            pos_mask < x_seq_lens[:, None],
            F.pad(x, (0, z_max_len - x.shape[1]))[:, :z_max_len],
            torch.where(pos_mask < z_seq_lens[:, None], self.gap_token_id, self.pad_token_id),
        )
        yp = torch.where(
            pos_mask < y_seq_lens[:, None],
            F.pad(y, (0, z_max_len - y.shape[1]))[:, :z_max_len],
            torch.where(pos_mask < z_seq_lens[:, None], self.gap_token_id, self.pad_token_id),
        )
        return xp, yp


class ShiftAlignment(Alignment):
    """Aligns pair of sequences by shifting one to the right and padding with the GAP token"""

    def align(self, x, y):
        batch_size, _ = x.shape
        x_seq_lens = (x != self.pad_token_id).sum(dim=1)
        y_seq_lens = (y != self.pad_token_id).sum(dim=1)
        z_seq_lens = torch.maximum(x_seq_lens, y_seq_lens)
        z_max_len = z_seq_lens.max().item()

        xp = torch.full((batch_size, z_max_len), self.gap_token_id, dtype=x.dtype, device=x.device)
        yp = torch.full((batch_size, z_max_len), self.gap_token_id, dtype=y.dtype, device=y.device)
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)
        xp[batch_indices, :x_seq_lens] = x
        yp[batch_indices, x_seq_lens:] = y
        xp[batch_indices, z_seq_lens:] = self.pad_token_id
        yp[batch_indices, z_seq_lens:] = self.pad_token_id
        return xp, yp


class OptimalAlignment(Alignment):
    """Aligns sequences using dynamic programming to minimize the edit distance"""

    def align(self, x, y):
        x_nopad_mask = x != self.pad_token_id
        y_nopad_mask = y != self.pad_token_id
        xp, yp = [], []
        for b, mx, my in zip(range(x.shape[0]), x_nopad_mask, y_nopad_mask):
            dp = self._fill_dp_table(x[b][mx], y[b][my])
            _xp, _yp = self._backtrack_align(x[b][mx], y[b][my], dp)
            xp.append(torch.tensor(_xp, dtype=x.dtype, device=x.device))
            yp.append(torch.tensor(_yp, dtype=y.dtype, device=y.device))

        xp_max_len = max(len(seq) for seq in xp)
        yp_max_len = max(len(seq) for seq in yp)
        xp = torch.stack(
            [F.pad(seq, (0, xp_max_len - len(seq)), value=self.pad_token_id) for seq in xp], dim=0
        )
        yp = torch.stack(
            [F.pad(seq, (0, yp_max_len - len(seq)), value=self.pad_token_id) for seq in yp], dim=0
        )
        return xp, yp

    def _fill_dp_table(self, seq_0: List[int], seq_1: List[int]) -> List[List[int]]:
        m, n = len(seq_0), len(seq_1)

        # DP table
        dp = [[i + j if i == 0 or j == 0 else 0 for j in range(n + 1)] for i in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = (
                    dp[i - 1][j - 1]
                    if seq_0[i - 1] == seq_1[j - 1]
                    else 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                )
        return dp

    def _backtrack_align(
        self, seq_0: torch.Tensor, seq_1: torch.Tensor, dp: List[List[int]]
    ) -> Tuple[List[int], List[int]]:
        seq_0, seq_1 = seq_0.cpu().numpy(), seq_1.cpu().numpy()
        m, n = len(seq_0), len(seq_1)
        aligned_0, aligned_1 = [], []
        i, j = m, n
        while i or j:
            if i and j and seq_0[i - 1] == seq_1[j - 1]:
                aligned_0.append(seq_0[i - 1])
                aligned_1.append(seq_1[j - 1])
                i, j = i - 1, j - 1
            elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
                aligned_0.append(seq_0[i - 1])
                aligned_1.append(seq_1[j - 1])
                i, j = i - 1, j - 1
            elif i and dp[i][j] == dp[i - 1][j] + 1:
                aligned_0.append(seq_0[i - 1])
                aligned_1.append(self.gap_token_id)
                i -= 1
            else:
                aligned_0.append(self.gap_token_id)
                aligned_1.append(seq_1[j - 1])
                j -= 1
        return aligned_0[::-1], aligned_1[::-1]


class AllOptimalAlignments(OptimalAlignment):
    """Aligns sequences using dynamic programming to minimize the edit distance"""

    def __init__(self, strategy="uniform", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy

    def align(self, x, y):
        x_nopad_mask = x != self.pad_token_id
        y_nopad_mask = y != self.pad_token_id

        xp, yp = [], []
        for b, mx, my in zip(range(x.shape[0]), x_nopad_mask, y_nopad_mask):
            _dp = self._fill_dp_table(x[b][mx], y[b][my])
            _alignments = self._backtrack_align(x[b][mx], y[b][my], _dp)
            _xp, _yp = self._select_opt_alignment(_alignments)
            xp.append(torch.tensor(_xp, dtype=x.dtype, device=x.device))
            yp.append(torch.tensor(_yp, dtype=y.dtype, device=y.device))

        xp_max_len = max(len(seq) for seq in xp)
        yp_max_len = max(len(seq) for seq in yp)

        xp = torch.stack(
            [F.pad(seq, (0, xp_max_len - len(seq)), value=self.pad_token_id) for seq in xp], dim=0
        )
        yp = torch.stack(
            [F.pad(seq, (0, yp_max_len - len(seq)), value=self.pad_token_id) for seq in yp], dim=0
        )

        return xp, yp

    def _backtrack_align(
        self, seq_0: torch.Tensor, seq_1: torch.Tensor, dp: List[List[int]]
    ) -> List[Tuple[List[int], List[int]]]:
        """Backtrack all optimal alignments from the DP table."""
        seq_0, seq_1 = seq_0.cpu().numpy(), seq_1.cpu().numpy()
        m, n = len(seq_0), len(seq_1)
        results = []

        def backtrack(i, j, aligned_0, aligned_1):
            if i == 0 and j == 0:
                results.append((aligned_0[::-1], aligned_1[::-1]))
                return

            # Try all valid directions with equal minimal cost
            if i > 0 and j > 0:
                if seq_0[i - 1] == seq_1[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
                    backtrack(i - 1, j - 1, aligned_0 + [seq_0[i - 1]], aligned_1 + [seq_1[j - 1]])
                elif dp[i][j] == dp[i - 1][j - 1] + 1:
                    backtrack(i - 1, j - 1, aligned_0 + [seq_0[i - 1]], aligned_1 + [seq_1[j - 1]])

            if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                backtrack(i - 1, j, aligned_0 + [seq_0[i - 1]], aligned_1 + [self.gap_token_id])

            if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
                backtrack(i, j - 1, aligned_0 + [self.gap_token_id], aligned_1 + [seq_1[j - 1]])

        backtrack(m, n, [], [])
        return results

    def _select_opt_alignment(
        self, alignments: List[Tuple[List[int], List[int]]]
    ) -> Tuple[List[int], List[int]]:
        if self.strategy == "first":
            return alignments[0]
        elif self.strategy == "uniform":
            return random.choice(alignments)
        elif self.strategy == "longest":
            return max(alignments, key=lambda x: len(x[0]))
        elif self.strategy == "shortest":
            return min(alignments, key=lambda x: len(x[0]))
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


######################################################################################################
# Sampler Classes
######################################################################################################


class EditFlowsSampler(DiscreteSampler):
    """Sampler for Edit Flows model from [Havasi et al.](https://arxiv.org/abs/2506.09018)"""

    def __init__(
        self,
        forward_fn: Callable[..., Tuple[Tensor, Tensor, Tensor]],
        max_seq_len: int,
        bos_token_id: int = 0,
        pad_token_id: int = 1,
        eos_token_id: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.max_seq_len = max_seq_len
        self.forward_fn = forward_fn

    def u(self, xt, t=None, S=None, **kwargs):
        device = xt.device
        x_pad_mask = (xt == self.pad_token_id).to(device)
        x_pad_mask = x_pad_mask
        u_t, ins_probs, sub_probs = self.forward_fn(xt, t, x_pad_mask, S=S, **kwargs)
        return u_t.cpu(), ins_probs.cpu(), sub_probs.cpu()

    def apply_u(self, xt, ut, h, S=None, event_types=None, jump_schedule=None, t=None, **kwargs):
        ut, ins_probs, sub_probs = ut

        # Get rates for each operation separately
        l_ins = ut[:, :, 0].cpu()
        l_sub = ut[:, :, 1].cpu()
        l_del = ut[:, :, 2].cpu()

        # Get probs for each operation separately
        l_ins_p = l_ins.softmax(dim=-1)
        l_sub_p = l_sub.softmax(dim=-1)
        l_del_p = l_del.softmax(dim=-1)

        if event_types is not None:
            batch_size, seq_len = l_sub.shape
            sub_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
            ins_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
            del_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

            for b in range(batch_size):
                if event_types[b] == 0:  # substitution
                    sub_mask[b, torch.multinomial(l_sub_p[b], 1).item()] = True
                elif event_types[b] == 1:  # insertion
                    ins_mask[b, torch.multinomial(l_ins_p[b], 1).item()] = True
                elif event_types[b] == 2:  # deletion
                    del_mask[b, torch.multinomial(l_del_p[b], 1).item()] = True
                else:
                    raise ValueError(f"Unknown event type: {event_types[b]}")

        elif jump_schedule is not None:
            assert t is not None, "t must be provided when using jump_schedule"
            # Figure out how many events of each type occur in the [t, t+h) interval
            th = t + h
            jump_mask = (jump_schedule >= t) & (jump_schedule < th)
            S_h = jump_mask.sum(dim=-1)  # (batch_size, 3)
            subs_h, ins_h, dels_h = S_h[:, 0], S_h[:, 1], S_h[:, 2]

            # sample (subs_h, ins_h, dels_h) positions for each operations based on rates (batched multinomial sampling)
            sub_pos = torch.multinomial(
                l_sub.softmax(dim=-1), max(subs_h.max().item(), 1), replacement=True
            )
            ins_pos = torch.multinomial(
                l_ins.softmax(dim=-1), max(ins_h.max().item(), 1), replacement=True
            )
            del_pos = torch.multinomial(
                l_del.softmax(dim=-1), max(dels_h.max().item(), 1), replacement=True
            )

            # convert positions to masks, truncate to the actual number of events
            batch_size, seq_len = l_sub.shape
            sub_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
            ins_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
            del_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

            for b in range(batch_size):
                if subs_h[b] > 0:
                    sub_mask[b, sub_pos[b, : subs_h[b]]] = True
                if ins_h[b] > 0:
                    ins_mask[b, ins_pos[b, : ins_h[b]]] = True
                if dels_h[b] > 0:
                    del_mask[b, del_pos[b, : dels_h[b]]] = True

            raise NotImplementedError("Generic jump schedule sampling not implemented yet")
        else:
            # Sample insertions and deletion/substitutions based on rates
            ins_mask = torch.rand(size=l_ins.shape, device=l_ins.device) < 1 - torch.exp(-h * l_ins)
            del_sub_mask = torch.rand(size=l_sub.shape, device=l_sub.device) < 1 - torch.exp(
                -h * (l_sub + l_del)
            )

            # For deletion/substitution, sample based on the relative rates
            prob_del = torch.where(del_sub_mask, l_del / (l_sub + l_del), torch.zeros_like(l_del))
            del_mask = torch.bernoulli(prob_del).bool()
            sub_mask = del_sub_mask & ~del_mask
            assert sub_mask.sum() + del_mask.sum() == del_sub_mask.sum()

        # don't allow substitution or deletion of <bos> token, never touch <eos> token
        bos_mask = xt == self.bos_token_id
        eos_mask = xt == self.eos_token_id
        sub_mask[bos_mask], del_mask[bos_mask] = False, False
        ins_mask[eos_mask], del_mask[eos_mask], sub_mask[eos_mask] = False, False, False

        # Only sample tokens for non-pad positions, fill pad positions with PAD_TOKEN
        ins_tokens = torch.full(ins_probs.shape[:2], self.pad_token_id, dtype=torch.long)
        sub_tokens = torch.full(sub_probs.shape[:2], self.pad_token_id, dtype=torch.long)

        x_pad_mask = (xt == self.pad_token_id).to(xt.device)
        non_pad_mask = ~x_pad_mask

        if non_pad_mask.any():
            ins_sampled = torch.multinomial(
                ins_probs[non_pad_mask].cpu(), num_samples=1, replacement=True
            ).squeeze(-1)
            sub_sampled = torch.multinomial(
                sub_probs[non_pad_mask].cpu(), num_samples=1, replacement=True
            ).squeeze(-1)
            ins_tokens[non_pad_mask] = ins_sampled
            sub_tokens[non_pad_mask] = sub_sampled

        # If edit budget is already exceeded (< 0), zero out corresponding masks
        if S is not None:
            S = S.cpu()  # (batch_size, 3)
            sub_mask = torch.where(S[:, 0:1] > 0, sub_mask, torch.zeros_like(sub_mask).bool())
            ins_mask = torch.where(S[:, 1:2] > 0, ins_mask, torch.zeros_like(ins_mask).bool())
            del_mask = torch.where(S[:, 2:3] > 0, del_mask, torch.zeros_like(del_mask).bool())

        # Apply operations based on masks
        xt[sub_mask] = sub_tokens[sub_mask]
        xt = apply_ins_del_operations(
            xt,
            ins_mask,
            del_mask,
            ins_tokens,
            max_seq_len=self.max_seq_len,
            pad_token=self.pad_token_id,
        )

        # count number of subs, ins, dels performed
        n_subs = sub_mask.sum(-1)
        n_ins = ins_mask.sum(-1)
        n_dels = del_mask.sum(-1)

        return xt, n_subs, n_ins, n_dels


class InfTimeEditFlowsSampler(EditFlowsSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def h(xt: Tensor, S: Tensor) -> Tensor:
            """Adaptive step size function based on the final desired sequence length"""
            l0s = (xt != self.pad_token_id).sum(dim=1).float()
            n_ins, n_del = S[:, 1], S[:, 2]
            l1s = (n_ins - n_del) + l0s
            l1s = torch.clamp(l1s, min=100)
            adapt_h = (1.0 / l1s).reshape(-1, 1)
            return adapt_h

        self.h = h

    @torch.no_grad()
    def __call__(self, xt: Tensor, S: Tensor, *args, **kwargs) -> List[Tensor]:
        device = xt.device
        x_ts = [xt.clone()]
        S_traj = [S.clone()]
        adapt_h = self.h(xt, S).cpu()
        print(adapt_h.tolist())

        with tqdm(desc="Infinite Time Sampling") as pbar:
            while S.max() > 0:  # continue until all edit budgets are exhausted
                ut = self.u(xt, t=None, S=S, *args, **kwargs)
                xt, n_sub, n_ins, n_del = self.apply_u(xt.cpu(), ut, adapt_h, S=S)
                delta_S = torch.stack([n_sub, n_ins, n_del], dim=1).float().to(device)
                S = torch.clamp(S - delta_S, min=0)
                print(S.tolist())
                x_ts.append(xt.clone())
                S_traj.append(S.clone())
                xt = xt.to(device)
                pbar.update(1)

        return x_ts, S_traj


class SCUMEditFlowsSampler(EditFlowsSampler):
    @torch.no_grad()
    def __call__(
        self, xt: Tensor, S: Tensor, jump_schedule: Tensor, *args, **kwargs
    ) -> List[Tensor]:
        device = xt.device
        x_ts = [xt.clone()]
        S_traj = [S.clone()]
        bs, _, _ = jump_schedule.shape

        with tqdm(desc="Exact Schedule Conditioned Sampling") as pbar:
            while S.max() > 0:
                ut = self.u(xt, t=None, S=S, *args, **kwargs)

                # For each batch element, find the event type with the minimum scheduled time and its index, take care of nan values
                min_S_values, min_S_indices = jump_schedule.min(dim=-1)
                min_S_event_type = min_S_values.argmin(dim=-1)  # (batch_size,)
                if min_S_values.min() == float("inf"):
                    break

                # Sample one event per batch element based on the min_S_event_type, and apply it
                xt, n_sub, n_ins, n_del = self.apply_u(
                    xt.cpu(), ut, h=None, S=S, event_types=min_S_event_type
                )
                delta_S = torch.stack([n_sub, n_ins, n_del], dim=1).float().to(device)
                assert delta_S.max() <= 1, "More than one event applied in a single step!"
                S = S - delta_S
                assert (S >= 0).all(), "Edit budget exceeded!"

                # Set the used events in the jump_schedule to infinity to prevent re-use
                sampled_min_S_indices = min_S_indices.gather(
                    1, min_S_event_type.unsqueeze(1)
                ).squeeze(1)
                jump_schedule[torch.arange(bs), min_S_event_type, sampled_min_S_indices] = float(
                    "inf"
                )

                print(S.max().item(), jump_schedule.min().item())
                x_ts.append(xt.clone())
                S_traj.append(S.clone())
                xt = xt.to(device)
                pbar.update(1)

        return x_ts, S_traj


######################################################################################################
# Utility Functions
######################################################################################################


def make_ut_mask_from_z(
    z_t: Tensor,
    z_1: Tensor,
    vocab_size: int,
    pad_token: int = 0,
    gap_token: int = 130,
) -> Tensor:
    """
    Create a mask for u_cat for indexing the output rate tensor based on differences between z_t and z_1.
    For each position i where z_t and z_1 differ, we index as follows:

    - z_t[i] = GAP_TOKEN & z_1[i] = c => u_mask[i, insert, c] = 1
    - z_t[i] = c & z_1[i] = GAP_TOKEN => u_mask[i, delete] = 1
    - z_t[i] = c1 & z_1[i] = c2 => u_mask[i, substitute, c1, c2] = 1
    """
    batch_size, z_seq_len = z_t.shape
    n_ops = 2 * vocab_size + 1  # insert + substitute + delete

    z_neq = (z_t != z_1) & (z_t != pad_token) & (z_1 != pad_token)
    z_ins = (z_t == gap_token) & (z_1 != gap_token) & z_neq  # (batch_size, z_seq_len)
    z_del = (z_t != gap_token) & (z_1 == gap_token) & z_neq  # (batch_size, z_seq_len)
    z_sub = z_neq & ~z_ins & ~z_del  # (batch_size, z_seq_len)

    # mask (batch_size, z_seq_len, u_ops) where 1 indicates operation that bring z_t closer to z_1
    u_mask = torch.zeros((batch_size, z_seq_len, n_ops), dtype=torch.bool, device=z_t.device)
    u_mask[z_ins, z_1[z_ins]] = True
    u_mask[z_sub, z_1[z_sub] + vocab_size] = True
    u_mask[:, :, -1][z_del] = True

    assert z_neq.sum() == (z_ins | z_del | z_sub).sum(), "Mismatch in number of edits"
    assert z_neq.sum() == u_mask.sum(), "Mismatch in number of edits in mask"

    return u_mask


def fill_gap_tokens_with_repeats(
    x_ut: torch.Tensor,
    z_gap_mask: torch.Tensor,
    z_pad_mask: torch.Tensor,
):
    batch_size, _ = z_gap_mask.shape
    _, x_seq_len, _ = x_ut.shape

    # Use cumsum on non-gap positions to point to the last valid non-gap position
    non_gap_mask = ~z_gap_mask  # Invert mask to get non-gap positions
    indices = non_gap_mask.cumsum(dim=1) - 1  # (batch_size, z_seq_len)
    indices = indices.clamp(min=0, max=x_seq_len - 1)  # Ensure indices are within bounds

    # Use indices to gather from x_ut
    batch_indices = torch.arange(batch_size, device=x_ut.device).unsqueeze(1)
    result = x_ut[batch_indices, indices]  # (batch_size, z_seq_len, vocab_size)
    result[z_pad_mask] = 0  # Set pad positions to 0
    return result


def apply_ins_del_operations(
    x_t: Tensor,
    ins_mask: Tensor,
    del_mask: Tensor,
    ins_tokens: Tensor,
    pad_token: int = 0,
    max_seq_len: int = 512,
) -> Tensor:
    """
    Apply insertion and deletion operations to a sequence x_t based on the provided masks.
    """
    batch_size, seq_len = x_t.shape
    device = x_t.device

    # Handle simultaneous ins+del as substitutions
    replace_mask = ins_mask & del_mask
    x_t_modified = x_t.clone()
    x_t_modified[replace_mask] = ins_tokens[replace_mask]

    # Update ins/del masks after handling replacements
    eff_ins_mask = ins_mask & ~replace_mask
    eff_del_mask = del_mask & ~replace_mask

    # Compute new lengths after applying ins/del operations
    xt_pad_mask = x_t == pad_token  # (batch_size, seq_len)
    xt_seq_lens = (~xt_pad_mask).sum(dim=1)  # (batch_size,)
    new_lengths = xt_seq_lens + eff_ins_mask.sum(dim=1) - eff_del_mask.sum(dim=1)
    max_new_len = int(new_lengths.max().item())

    if max_new_len <= 0:
        print(f"Unexpected max_new_len <= 0: {max_new_len}, did we delete everything?")
        return torch.full((batch_size, 1), pad_token, dtype=x_t.dtype, device=device)

    # Pre-allocate result
    x_new = torch.full((batch_size, max_new_len), pad_token, dtype=x_t.dtype, device=device)

    # Compute positions
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)  # (batch_size, 1)
    pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
    cum_del = torch.cumsum(eff_del_mask.float(), dim=1)  # num del up to & incl. current pos
    cum_ins = torch.cumsum(eff_ins_mask.float(), dim=1)  # num ins up to & incl. current pos
    cum_ins_before = F.pad(cum_ins[:, :-1], (1, 0), value=0)  # num ins before current pos

    # Place non-deleted tokens
    new_pos = pos_idx + cum_ins_before - cum_del  # new pos of tokens shifted by ins/del
    keep_mask = (
        ~eff_del_mask & (new_pos >= 0) & (new_pos < max_new_len)
    )  # tokens to keep (non-deleted)
    if keep_mask.any():
        x_new[batch_idx.expand(-1, seq_len)[keep_mask], new_pos[keep_mask].long()] = x_t_modified[
            keep_mask
        ]

    # Place insertions
    if eff_ins_mask.any():
        ins_pos = new_pos + 1  # insertions go 1 after new shifted pos
        ins_valid = eff_ins_mask & (ins_pos >= 0) & (ins_pos < max_new_len)  # tokens to insert
        if ins_valid.any():
            x_new[batch_idx.expand(-1, seq_len)[ins_valid], ins_pos[ins_valid].long()] = ins_tokens[
                ins_valid
            ]

    if max_new_len > max_seq_len:
        print(f"Warning: max_new_len {max_new_len} exceeds max_seq_len {max_seq_len}, truncating.")
        max_new_len = max_seq_len

    return x_new[:, :max_new_len]


def rm_gap_tokens(
    z: torch.Tensor, GAP_TOKEN: int = 130, PAD_TOKEN: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remove gap tokens from a batched tensor and right-pad with PAD_TOKEN.
    """
    batch_size, _ = z.shape
    z_no_gap = []
    for b in range(batch_size):
        z_no_pad = z[b][z[b] != PAD_TOKEN]
        z_no_gap.append(z_no_pad[z_no_pad != GAP_TOKEN])
    max_len = max(len(z) for z in z_no_gap)
    x = torch.stack(
        [F.pad(z, (0, max_len - len(z)), value=PAD_TOKEN) for z in z_no_gap], dim=0
    ).long()
    x_pad_mask = x == PAD_TOKEN
    z_gap_mask = z == GAP_TOKEN
    z_pad_mask = z == PAD_TOKEN
    assert ((~x_pad_mask).sum(1) + z_gap_mask.sum(1)).equal((~z_pad_mask).sum(1))
    return x, x_pad_mask, z_gap_mask, z_pad_mask


def rv_gap_tokens(
    x: torch.Tensor,
    z_gap_mask: torch.Tensor,
    z_pad_mask: torch.Tensor,
    GAP_TOKEN: int = 130,
    PAD_TOKEN: int = 0,
) -> torch.Tensor:
    """
    Reinsert gap tokens into a tensor at specified positions.
    """
    assert x.shape[0] == z_gap_mask.shape[0]
    assert x.shape[1] <= z_gap_mask.shape[1]
    assert z_gap_mask.shape == z_pad_mask.shape
    batch_size, _ = x.shape
    _, z_seq_len = z_gap_mask.shape
    z = torch.full((batch_size, z_seq_len), PAD_TOKEN, dtype=x.dtype, device=x.device)
    z[~z_gap_mask & ~z_pad_mask] = x[x != PAD_TOKEN]
    z[z_gap_mask] = GAP_TOKEN
    return z


def count_num_mutations(
    za: Tensor,
    zb: Tensor,
    GAP_TOKEN: int = 130,
    PAD_TOKEN: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    n_ins = (za == GAP_TOKEN) & (zb != GAP_TOKEN)  # n ins left to go from za to zb
    n_del = (za != GAP_TOKEN) & (zb == GAP_TOKEN)  # n dels left to go from za to zb
    n_sub = (za != zb) & (za != GAP_TOKEN) & (zb != GAP_TOKEN)  # n subs left to go from za to zb
    return n_ins.sum(-1), n_del.sum(-1), n_sub.sum(-1)


# Usage example
if __name__ == "__main__":
    from esm.data import Alphabet
    from evo.tokenization import Vocab

    vocab = Vocab.from_esm_alphabet(Alphabet.from_architecture("ESM-1b"))

    align = OptimalAlignment(strategy="random", vocab=vocab, add_bos_token=True, add_eos_token=True)

    # Example sequences (batch_size=2)
    x = torch.tensor(
        [
            [vocab.bos_idx, 5, 6, 7, 20, 20, 5, 6, 7, vocab.eos_idx],
            [vocab.bos_idx, 20, 20, 20, 4, 20, 20, vocab.eos_idx, vocab.pad_idx, vocab.pad_idx],
        ]
    )
    y = torch.tensor(
        [
            [vocab.bos_idx, 5, 6, 7, vocab.eos_idx],
            [vocab.bos_idx, 20, 20, vocab.eos_idx, vocab.pad_idx],
        ]
    )

    xp, yp, z0, z1 = align(x, y)

    for i in range(x.shape[0]):
        print("x: ", x[i])
        print("y: ", y[i])
        print()
        print("z0:", z0[i])
        print("z1:", z1[i])
        print()

    # breakpoint()