import multiprocessing as mp
import os
from functools import partial
from itertools import chain
from typing import Iterator, Union

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.data import Alphabet
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, random_split
from tqdm import tqdm

CHAIN_BREAK = "."
TransitionsType = list[tuple[str, str, float]] | list[tuple[str, str, str, str, float, float]]


def read_transitions(transitions_path: str) -> TransitionsType:
    """
    If single_protein is True, the transition is of the form (x1, x2, tx)
    Else, it is of the form (x1, y1, x2, y2, tx, ty)
    """
    transitions = []
    # Use context manager (with) to ensure file is closed
    with open(transitions_path, "r") as f:
        lines = f.read().strip().split("\n")

    if len(lines) == 0:
        raise Exception(f"The transitions file at {transitions_path} is empty")
    for i, line in enumerate(lines):
        tokens = line.split(" ")
        if i == 0:
            if len(tokens) != 2:
                raise ValueError(
                    f"Transitions file at '{transitions_path}' should start "
                    f"with '[NUM_TRANSITIONS] transitions'."
                )
            if tokens[1] != "transitions":
                raise ValueError(
                    f"Transitions file at '{transitions_path}' should start "
                    f"with '[NUM_TRANSITIONS] transitions'."
                )
            if len(lines) - 1 != int(tokens[0]):
                raise ValueError(
                    f"Expected {int(tokens[0])} transitions at "
                    f"'{transitions_path}', but found only {len(lines) - 1}."
                )
        else:
            # Determine the line structure by the number of tokens separatedly by " "
            if len(tokens) == 3:
                # Single protein transition
                x1, x2, tx_str = tokens
                tx = float(tx_str)
                transitions.append((x1, x2, tx))
            elif len(tokens) == 6:
                # Paired protein transition
                x1, y1, x2, y2, tx_str, ty_str = tokens
                tx, ty = float(tx_str), float(ty_str)
                transitions.append((x1, y1, x2, y2, tx, ty))
            elif len(tokens) == 4:
                # Transition names for paired protein transition
                l1, l2, tx_str, ty_str = tokens
                tx, ty = float(tx_str), float(ty_str)
                transitions.append((l1, l2, tx, ty))
            else:
                raise ValueError(f"The line has {len(tokens)} components, expected 3, 4, or 6")
    return transitions


def _process_pair_transition(
    transition_full_length: tuple,
    vocab: Alphabet,
    return_aligned_sequences: bool = False,
    transition_aligned: tuple | None = None,
    align_mask: tuple | None = None,
    max_length: int = 1022,
):
    """Process a single transition tuple and return its processed data"""
    x1_seq, y1_seq, x2_seq, y2_seq, tx, ty = transition_full_length

    # Get sequence lengths without gaps
    x1_seq_len = len(x1_seq)
    x2_seq_len = len(x2_seq)
    y1_seq_len = len(y1_seq)
    y2_seq_len = len(y2_seq)

    # Filter out transitions that are too long
    # Technically, this should be dealt with in transition_to_keep
    # But I want to avoid passing in too many unnecessary arguments at training time
    if (
        x1_seq_len > max_length
        or x2_seq_len > max_length
        or y1_seq_len > max_length
        or y2_seq_len > max_length
    ):
        return None

    # Store encoded sequences as lists instead of numpy arrays
    result = {
        "x1_len": x1_seq_len,
        "x2_len": x2_seq_len,
        "y1_len": y1_seq_len,
        "y2_len": y2_seq_len,
        "x1_toks": vocab.encode(x1_seq),
        "x2_toks": vocab.encode(x2_seq),
        "y1_toks": vocab.encode(y1_seq),
        "y2_toks": vocab.encode(y2_seq),
        "tx": tx,
        "ty": ty,
    }

    if return_aligned_sequences:
        assert transition_aligned is not None and align_mask is not None
        x1_aln, y1_aln, x2_aln, y2_aln, _, _ = transition_aligned
        amx1, amy1, amx2, amy2, _, _ = align_mask
        result.update(
            {
                "x1_aln_toks": vocab.encode(x1_aln),
                "x2_aln_toks": vocab.encode(x2_aln),
                "y1_aln_toks": vocab.encode(y1_aln),
                "y2_aln_toks": vocab.encode(y2_aln),
                "x1_aln_masks": [int(i) for i in amx1],
                "x2_aln_masks": [int(i) for i in amx2],
                "y1_aln_masks": [int(i) for i in amy1],
                "y2_aln_masks": [int(i) for i in amy2],
            }
        )

    return result


def _process_pair(
    pair_name: str,
    full_length_transitions_dir: str,
    vocab: Alphabet,
    return_aligned_sequences: bool,
    aligned_transitions_dir: str | None = None,
    alignment_mask_dir: str | None = None,
    transitions_to_keep_dir: str | None = None,
    max_length: int = 1022,
):
    """Process all transitions for a single pair"""
    transitions_full_length = read_transitions(
        os.path.join(full_length_transitions_dir, pair_name + ".txt")
    )

    # Read the aligned transitions
    if aligned_transitions_dir is not None:
        transitions_aligned = read_transitions(
            os.path.join(aligned_transitions_dir, pair_name + ".txt")
        )
        assert len(transitions_aligned) == len(
            transitions_full_length
        ), "transitions_aligned and transitions_full_length don't have the same length"
    else:
        transitions_aligned = [None for _ in range(len(transitions_full_length))]

    # Read the alignment mask
    if alignment_mask_dir is not None:
        alignment_masks = read_transitions(alignment_mask_dir / f"{pair_name}.txt")
        assert len(alignment_masks) == len(
            transitions_full_length
        ), "transitions_aligned and transitions_full_length don't have the same length"
    else:
        alignment_masks = [None for _ in range(len(transitions_full_length))]

    # Read the transition to keep mask
    if transitions_to_keep_dir is not None:
        transitions_to_keep = np.loadtxt(transitions_to_keep_dir / f"{pair_name}.txt")
        assert len(transitions_full_length) == len(
            transitions_to_keep
        ), "transitions_to_keep and transitions_full_length don't have the same length"
    else:
        # By default keep all transitions
        transitions_to_keep = [True for _ in range(len(transitions_full_length))]

    process_func = partial(
        _process_pair_transition,
        vocab=vocab,
        return_aligned_sequences=return_aligned_sequences,
        max_length=max_length,
    )
    results = []
    for (
        transition_full_length,
        transition_aligned,
        align_mask,
        transition_to_keep,
    ) in zip(
        transitions_full_length,
        transitions_aligned,
        alignment_masks,
        transitions_to_keep,
    ):
        if not transition_to_keep:
            continue
        result = process_func(
            transition_full_length=transition_full_length,
            transition_aligned=transition_aligned,
            align_mask=align_mask,
        )
        if result is not None:
            results.append(result)

    return results


class PairMSADataset(Dataset):
    def __init__(
        self,
        pair_names: list[str],
        full_length_transitions_dir: str,
        vocab: Alphabet,
        aligned_transitions_dir: str | None = None,
        alignment_mask_dir: str | None = None,
        transitions_to_keep_dir: str | None = None,
        return_aligned_sequences: bool = False,
        num_workers: int = None,
        max_length: int = 1022,
    ):
        """
        Dataset class for processing transitions extracted from a set of pair MSAs
        """
        super(PairMSADataset, self).__init__()
        self.vocab = vocab
        self.return_aligned_sequences = return_aligned_sequences

        if return_aligned_sequences:
            assert (
                aligned_transitions_dir is not None
            ), "aligned_transitions_dir needs to be provided when return_aligned_sequences=True"
            assert (
                alignment_mask_dir is not None
            ), "alignment_mask_dir needs to be provided when return_aligned_sequences=True"
            assert (
                transitions_to_keep_dir is not None
            ), "transitions_to_keep_dir needs to be provided when return_aligned_sequences=True"

        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)  # Limit max workers

        # Use 'spawn' context for better compatibility
        ctx = mp.get_context("spawn")

        # Create a pool of workers
        with ctx.Pool(num_workers) as pool:
            process_func = partial(
                _process_pair,
                full_length_transitions_dir=full_length_transitions_dir,
                aligned_transitions_dir=aligned_transitions_dir,
                alignment_mask_dir=alignment_mask_dir,
                transitions_to_keep_dir=transitions_to_keep_dir,
                vocab=vocab,
                return_aligned_sequences=return_aligned_sequences,
                max_length=max_length,
            )

            # Process all pairs and flatten results
            results = list(
                tqdm(
                    pool.imap(process_func, pair_names),
                    total=len(pair_names),
                    desc="Processing pairs",
                )
            )
            all_results = list(chain.from_iterable(results))

        # Convert results to tensors after all parallel processing is done
        self.x1_toks = []
        self.x2_toks = []
        self.y1_toks = []
        self.y2_toks = []
        self.x1_len = []
        self.x2_len = []
        self.y1_len = []
        self.y2_len = []
        self.tx = []
        self.ty = []

        if return_aligned_sequences:
            self.x1_aln_toks = []
            self.x2_aln_toks = []
            self.y1_aln_toks = []
            self.y2_aln_toks = []
            self.x1_aln_masks = []
            self.x2_aln_masks = []
            self.y1_aln_masks = []
            self.y2_aln_masks = []

        # Convert to tensors in the main process
        device = "cpu"  # Keep tensors on CPU initially
        for result in all_results:
            self.x1_toks.append(torch.tensor(result["x1_toks"], device=device))
            self.x2_toks.append(torch.tensor(result["x2_toks"], device=device))
            self.y1_toks.append(torch.tensor(result["y1_toks"], device=device))
            self.y2_toks.append(torch.tensor(result["y2_toks"], device=device))
            self.x1_len.append(result["x1_len"])
            self.x2_len.append(result["x2_len"])
            self.y1_len.append(result["y1_len"])
            self.y2_len.append(result["y2_len"])
            self.tx.append(result["tx"])
            self.ty.append(result["ty"])

            if return_aligned_sequences:
                self.x1_aln_toks.append(torch.tensor(result["x1_aln_toks"], device=device))
                self.x2_aln_toks.append(torch.tensor(result["x2_aln_toks"], device=device))
                self.y1_aln_toks.append(torch.tensor(result["y1_aln_toks"], device=device))
                self.y2_aln_toks.append(torch.tensor(result["y2_aln_toks"], device=device))
                self.x1_aln_masks.append(torch.tensor(result["x1_aln_masks"], device=device))
                self.x2_aln_masks.append(torch.tensor(result["x2_aln_masks"], device=device))
                self.y1_aln_masks.append(torch.tensor(result["y1_aln_masks"], device=device))
                self.y2_aln_masks.append(torch.tensor(result["y2_aln_masks"], device=device))

    def __len__(self):
        return len(self.x1_toks)

    def __getitem__(self, index):
        item = {
            "x1_toks": self.x1_toks[index],
            "x2_toks": self.x2_toks[index],
            "y1_toks": self.y1_toks[index],
            "y2_toks": self.y2_toks[index],
            "x1_len": self.x1_len[index],
            "x2_len": self.x2_len[index],
            "y1_len": self.y1_len[index],
            "y2_len": self.y2_len[index],
            "tx": self.tx[index],
            "ty": self.ty[index],
        }
        if self.return_aligned_sequences:
            item.update(
                {
                    "x1_aln_toks": self.x1_aln_toks[index],
                    "x2_aln_toks": self.x2_aln_toks[index],
                    "y1_aln_toks": self.y1_aln_toks[index],
                    "y2_aln_toks": self.y2_aln_toks[index],
                    "x1_aln_masks": self.x1_aln_masks[index],
                    "x2_aln_masks": self.x2_aln_masks[index],
                    "y1_aln_masks": self.y1_aln_masks[index],
                    "y2_aln_masks": self.y2_aln_masks[index],
                }
            )

        return item


def _process_family_transition(
    transition_full_length: tuple,
    vocab: Alphabet,
    transition_aligned: tuple | None,
    align_mask: tuple | None,
    return_aligned_sequences=True,
):
    """
    Process a single transition tuple of a single protein family
    Useful for evaluating on peint test set
    Return its processed data
    """
    # The full length sequences, including insertions relative to query
    x1_seq, x2_seq, tx = transition_full_length
    x1_seq_len = len(x1_seq)
    x2_seq_len = len(x2_seq)

    # Store encoded sequences as lists instead of numpy arrays
    result = {
        "x1_len": x1_seq_len,
        "x2_len": x2_seq_len,
        "x1_toks": vocab.encode(x1_seq),
        "x2_toks": vocab.encode(x2_seq),
        "tx": tx,
    }

    if return_aligned_sequences:
        assert transition_aligned is not None and align_mask is not None
        # The aligned sequences with gaps
        x1_aln, x2_aln, _ = transition_aligned
        # Get the positions in the alignment using the align_mask
        x1_aln_mask, x2_aln_mask, _ = align_mask
        result.update(
            {
                "x1_aln_toks": vocab.encode(x1_aln),
                "x2_aln_toks": vocab.encode(x2_aln),
                "x1_aln_masks": [int(i) for i in x1_aln_mask],
                "x2_aln_masks": [int(i) for i in x2_aln_mask],
            }
        )

    return result


def _process_family(
    family: str,
    full_length_transitions_dir: str,
    vocab: Alphabet,
    return_aligned_sequences: bool,
    aligned_transitions_dir: str | None = None,
    transitions_to_keep_dir: str | None = None,
    alignment_mask_dir: str | None = None,
):
    """
    Process all transitions for a single family
    """
    transitions_full_length = read_transitions(
        os.path.join(full_length_transitions_dir, family + ".txt")
    )
    if aligned_transitions_dir is not None:
        transitions_aligned = read_transitions(
            os.path.join(aligned_transitions_dir, family + ".txt")
        )
        assert len(transitions_full_length) == len(
            transitions_aligned
        ), "transitions_aligned and transitions_full_length don't have the same length"
    else:
        transitions_aligned = [None for _ in range(len(transitions_full_length))]

    if transitions_to_keep_dir is not None:
        # Shape (#transitions), 1 if valid
        transitions_to_keep = np.loadtxt(os.path.join(transitions_to_keep_dir, f"{family}.txt"))
        assert len(transitions_full_length) == len(
            transitions_to_keep
        ), "transitions_full_length and transitions_to_keep don't have the same length"
    else:
        # By default keep all transitions
        transitions_to_keep = [True for _ in range(len(transitions_full_length))]

    if alignment_mask_dir is not None:
        # Contains the binary alignment mask for each transition.
        # For example, if in the original MSA the
        # transition is (AG-L, A--L, 0.1) and the full transition is
        # (AGLPD, AXLP, 0.1) then the alignment mask will be
        # (11100, 1010, 0.1). This way, the mask provides a mapping between
        # the sites in the full sequence and the aligned sites in the
        # aligned transitions.
        with open(os.path.join(alignment_mask_dir, family + ".txt"), "r") as fin:
            _ = next(fin)
            align_masks = [l.rstrip("\n").split() for l in fin]
    else:
        align_masks = [None for _ in range(len(transitions_full_length))]

    process_func = partial(
        _process_family_transition,
        vocab=vocab,
        return_aligned_sequences=return_aligned_sequences,
    )

    results = []
    for (
        transition_full_length,
        transition_aligned,
        align_mask,
        transition_to_keep,
    ) in zip(transitions_full_length, transitions_aligned, align_masks, transitions_to_keep):
        if not transition_to_keep:
            continue
        result = process_func(
            transition_full_length=transition_full_length,
            transition_aligned=transition_aligned,
            align_mask=align_mask,
        )
        if result is not None:
            results.append(result)

    return results


class MSADataset(Dataset):
    def __init__(
        self,
        families: list[str],
        full_length_transitions_dir: str,
        vocab: Alphabet,
        aligned_transitions_dir: str | None = None,
        transitions_to_keep_dir: str | None = None,
        alignment_mask_dir: str | None = None,
        return_aligned_sequences: bool = True,
        num_workers: int = None,
    ):
        super(MSADataset, self).__init__()
        self.vocab = vocab
        self.return_aligned_sequences = return_aligned_sequences

        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)  # Limit max workers

        # Use 'spawn' context for better compatibility
        ctx = mp.get_context("spawn")

        # Create a pool of workers
        with ctx.Pool(num_workers) as pool:
            process_func = partial(
                _process_family,
                full_length_transitions_dir=full_length_transitions_dir,
                aligned_transitions_dir=aligned_transitions_dir,
                transitions_to_keep_dir=transitions_to_keep_dir,
                alignment_mask_dir=alignment_mask_dir,
                vocab=vocab,
                return_aligned_sequences=return_aligned_sequences,
            )
            # Process all pairs and flatten results
            results = list(
                tqdm(
                    pool.imap(process_func, families),
                    total=len(families),
                    desc="Processing families",
                )
            )
            all_results = list(chain.from_iterable(results))

        # Convert results to tensors after all parallel processing is done
        self.x1_toks = []
        self.x2_toks = []
        self.x1_len = []
        self.x2_len = []
        self.tx = []

        if return_aligned_sequences:
            self.x1_aln_toks = []
            self.x2_aln_toks = []
            self.x1_aln_masks = []
            self.x2_aln_masks = []

        # Convert to tensors in the main process
        device = "cpu"  # Keep tensors on CPU initially
        for result in all_results:
            self.x1_toks.append(torch.tensor(result["x1_toks"], device=device))
            self.x2_toks.append(torch.tensor(result["x2_toks"], device=device))
            self.x1_len.append(result["x1_len"])
            self.x2_len.append(result["x2_len"])
            self.tx.append(result["tx"])

            if return_aligned_sequences:
                self.x1_aln_toks.append(torch.tensor(result["x1_aln_toks"], device=device))
                self.x2_aln_toks.append(torch.tensor(result["x2_aln_toks"], device=device))
                self.x1_aln_masks.append(torch.tensor(result["x1_aln_masks"], device=device))
                self.x2_aln_masks.append(torch.tensor(result["x2_aln_masks"], device=device))

    def __len__(self):
        return len(self.x1_toks)

    def __getitem__(self, index):
        item = {
            "x1_toks": self.x1_toks[index],
            "x2_toks": self.x2_toks[index],
            "x1_len": self.x1_len[index],
            "x2_len": self.x2_len[index],
            "tx": self.tx[index],
        }
        if self.return_aligned_sequences:
            item.update(
                {
                    "x1_aln_toks": self.x1_aln_toks[index],
                    "x2_aln_toks": self.x2_aln_toks[index],
                    "x1_aln_masks": self.x1_aln_masks[index],
                    "x2_aln_masks": self.x2_aln_masks[index],
                }
            )
        return item


def mask_for_mlm(seqs: list[torch.Tensor], mask_prob: float, mask_idx: int, padding_idx: int):
    """
    Apply uniform masking on sequences
    Should be done prior to padding since in the targets we fill all the
    non-masked positions with paddings
    """
    x_mlm_masks = [torch.rand_like(seq, dtype=torch.float) < mask_prob for seq in seqs]

    x_mlm_inputs = [seq.masked_fill(mask, mask_idx) for seq, mask in zip(seqs, x_mlm_masks)]

    x_mlm_targets = [seq.masked_fill(~mask, padding_idx) for seq, mask in zip(seqs, x_mlm_masks)]

    return x_mlm_inputs, x_mlm_targets


def mask_set_for_mlm(
    seqset: list[list[torch.Tensor]], mask_prob: float, mask_idx: int, padding_idx: int
):
    inputs = []
    targets = []
    for seqs in seqset:
        x_mlm_inputs, x_mlm_targets = mask_for_mlm(
            seqs=seqs, mask_prob=mask_prob, mask_idx=mask_idx, padding_idx=padding_idx
        )
        inputs.append(x_mlm_inputs)
        targets.append(x_mlm_targets)
    return inputs, targets


class PipetCollator:
    def __init__(
        self,
        vocab: Alphabet,
        mask_prob: float = 0.15,
        device: torch.device = torch.device("cuda"),
        return_aln_seqs: bool = False,
        shuffle_protein_order: bool = False,
    ):
        """
        Batch collator for PairMSA Dataset used for the Pipet

        Args:
            shuffle_protein_order: If true, then we swap the order of protein x and y
                so y is on the left hand side
        """
        self.vocab = vocab
        self.mask_prob = mask_prob
        # We use one of the superfluous token in the ESM-1b vocab
        # as the token to separate the chains in the decoder
        self.sep_idx = self.vocab.get_idx(CHAIN_BREAK)
        self.device = device
        self.return_aln_seqs = return_aln_seqs
        self.shuffle_protein_order = shuffle_protein_order

    def __call__(self, batch):
        """
        Process batch
        """
        # Sequences pairs for the encoder and decoder
        enc_seqs, dec_seqs = [], []
        # Length of sequence pairs for the encoder and decoder
        enc_lengths, dec_lengths = [], []
        # Distance between x1 and x2; between y1 and y2
        distances = []
        # Aligned sequences and masks for the decoder
        dec_aln_seqs, dec_aln_masks = [], []
        for b in batch:
            if not self.shuffle_protein_order:
                enc_seqs.append([b["x1_toks"], b["y1_toks"]])
                dec_seqs.append([b["x2_toks"], b["y2_toks"]])
                # Each sequence in the encoder inputs have <cls> and <eos>
                enc_lengths.append([b["x1_len"] + 2, b["y1_len"] + 2])
                # Each sequence in the decoder input has <.> or <eos>
                dec_lengths.append([b["x2_len"] + 1, b["y2_len"] + 1])
                distances.append([b["tx"], b["ty"]])
                if self.return_aln_seqs:
                    dec_aln_seqs.append([b["x2_aln_toks"], b["y2_aln_toks"]])
                    # x2_aln_masks = b["x2_aln_masks"]
                    # print(f"x2_aln_mask: {x2_aln_masks}")
                    # y2_aln_masks = b["y2_aln_masks"]
                    # print(f"y2_aln_mask: {y2_aln_masks}")
                    dec_aln_masks.append([b["x2_aln_masks"], b["y2_aln_masks"]])
            else:
                # Swap the order of x and y
                enc_seqs.append([b["y1_toks"], b["x1_toks"]])
                dec_seqs.append([b["y2_toks"], b["x2_toks"]])
                # Each sequence in the encoder inputs have <cls> and <eos>
                enc_lengths.append([b["y1_len"] + 2, b["x1_len"] + 2])
                # Each sequence in the decoder input has <.> or <eos>
                dec_lengths.append([b["y2_len"] + 1, b["x2_len"] + 1])
                distances.append([b["ty"], b["tx"]])
                if self.return_aln_seqs:
                    dec_aln_seqs.append([b["y2_aln_toks"], b["x2_aln_toks"]])
                    dec_aln_masks.append([b["y2_aln_masks"], b["x2_aln_masks"]])

        # Create masked x1 and y1 for MLM loss
        enc_inputs, enc_targets = mask_set_for_mlm(
            seqset=enc_seqs,
            mask_prob=self.mask_prob,
            mask_idx=self.vocab.mask_idx,
            padding_idx=self.vocab.padding_idx,
        )

        # Add <cls> and <eos> to both encoder input sequences
        # Add <pad> to start and end of the encoder target sequences
        # since we don't need <cls> and <eos> to predict specific tokens
        # Shape: list[list[torch.Tensor(x1_len), torch.Tensor(y1_len)]]
        # seqset = (x1, y1)
        enc_inputs = [
            [F.pad(seq, (0, 1), value=self.vocab.eos_idx) for seq in seqset]
            for seqset in enc_inputs
        ]
        enc_inputs = [
            [F.pad(seq, (1, 0), value=self.vocab.cls_idx) for seq in seqset]
            for seqset in enc_inputs
        ]
        enc_targets = [
            [F.pad(seq, (0, 1), value=self.vocab.padding_idx) for seq in seqset]
            for seqset in enc_targets
        ]
        enc_targets = [
            [F.pad(seq, (1, 0), value=self.vocab.padding_idx) for seq in seqset]
            for seqset in enc_targets
        ]

        # Concatenate x1 and y1
        # Shape: (B, x1_len + y1_len)
        enc_inputs = [torch.cat(seqset, dim=0) for seqset in enc_inputs]
        enc_targets = [torch.cat(seqset, dim=0) for seqset in enc_targets]
        # Add padding to make all concat sequences in the batch have the same lengths
        enc_inputs = nn.utils.rnn.pad_sequence(
            enc_inputs, batch_first=True, padding_value=self.vocab.padding_idx
        )
        enc_targets = nn.utils.rnn.pad_sequence(
            enc_targets, batch_first=True, padding_value=self.vocab.padding_idx
        )

        # Add '.' to the end of x2 and <eos> to end of y2
        dec_seqs = [
            [
                F.pad(seqset[0], (0, 1), value=self.sep_idx),
                F.pad(seqset[1], (0, 1), value=self.vocab.eos_idx),
            ]
            for seqset in dec_seqs
        ]
        # Concatenate x2 and y2
        # Shape: (B, x2_len + y2_len)
        dec_seqs = [torch.cat(seqset, dim=0) for seqset in dec_seqs]
        # Add paddings to make all concat sequences the same length
        dec_seqs = nn.utils.rnn.pad_sequence(
            dec_seqs, batch_first=True, padding_value=self.vocab.padding_idx
        )
        # Input has the <cls> to start of concat sequences to form the
        # decoder input
        dec_inputs = F.pad(dec_seqs[:, :-1], (1, 0), value=self.vocab.cls_idx)

        # Create masks to record the lengths of sequences in each pair
        # Non-zero values indicate the start and
        # length of each sequence, while zeros represent padding.
        attn_mask_enc_lengths = torch.zeros_like(enc_targets)
        attn_mask_dec_lengths = torch.zeros_like(dec_seqs)
        for i, l in enumerate(enc_lengths):
            attn_mask_enc_lengths[i, : len(l)] = torch.tensor(l)
        for i, l in enumerate(dec_lengths):
            attn_mask_dec_lengths[i, : len(l)] = torch.tensor(l)

        # Store distance tx and ty
        distances_tensor = torch.zeros_like(dec_seqs, dtype=torch.float32)
        for i, d in enumerate(distances):
            distances_tensor[i, : len(d)] = torch.tensor(d)

        batch = (
            enc_inputs,
            enc_targets,
            dec_inputs,
            dec_seqs,
            attn_mask_enc_lengths,
            attn_mask_dec_lengths,
            distances_tensor,
        )
        batch = [b.to(self.device) for b in batch]
        if self.return_aln_seqs:
            batch.append(dec_aln_seqs)
            batch.append(dec_aln_masks)
        return batch


class PairMSADataModule(pl.LightningDataModule):
    """
    Pytorch lightning data module to wrap the PairMSADataset and its loader
    Same arguments, but also batch size, and train-val split
    """

    def __init__(
        self,
        pair_names: list[str],
        transitions_dir: str,
        vocab: Alphabet,
        mask_prob: float = 0.15,
        batch_size: int = 32,
        train_frac: float = 0.85,
        num_workers: int = 0,
        use_batch_sampler: bool = True,
        max_num_tokens: int = 200000,
        shuffle_protein_order: bool = False,
        which_protein: str | None = None,
        max_length: int = 1022,
    ):
        super().__init__()
        self.pair_names = pair_names
        self.transitions_dir = transitions_dir
        self.vocab = vocab
        self.batch_size = batch_size
        self.mask_prob = mask_prob
        self.train_frac = train_frac
        self.val_frac = 1 - train_frac
        self.num_workers = num_workers
        self.use_batch_sampler = use_batch_sampler
        self.max_num_tokens = max_num_tokens
        self.shuffle_protein_order = shuffle_protein_order
        self.which_protein = which_protein
        self.max_length = max_length

        # Multi-processing setup
        self.multiprocessing_context = (
            "spawn" if num_workers > 0 else None
        )  # added for multi-device
        self.persistent_workers = True if num_workers > 1 else False
        self.pin_memory = True

    def setup(self, stage=None):
        if stage == "fit" and not hasattr(self, "dataset"):
            # Use parallelized version of dataset
            # Note: we use multi-processing in dataset processing, but not loading
            self.dataset = PairMSADataset(
                pair_names=self.pair_names,
                full_length_transitions_dir=self.transitions_dir,
                vocab=self.vocab,
                num_workers=1,
                max_length=self.max_length,
            )
            # Create train/val split
            self.train_dataset, self.val_dataset = random_split(
                self.dataset,
                [self.train_frac, self.val_frac],
                generator=torch.Generator().manual_seed(42),  # for reproducibility
            )
            # Keep collation in cpu
            if self.which_protein is None:
                # By default, load in both proteins in the pair for pipet
                self.collate_fn = PipetCollator(
                    vocab=self.vocab,
                    mask_prob=self.mask_prob,
                    device="cpu",
                    shuffle_protein_order=self.shuffle_protein_order,
                )
            else:
                # If which protein is specified, we only load in only the proteins
                # we assume this is for training peint
                self.collate_fn = PeintCollator(
                    vocab=self.vocab,
                    which_protein=self.which_protein,
                    mask_prob=self.mask_prob,
                    device="cpu",
                    return_aln_seq=False,
                    which_model="peint",
                )
            if self.use_batch_sampler:
                # Use dynamic batch size with batch sampler
                self.train_batch_sampler = BatchSampler(
                    dataset=self.train_dataset,
                    max_tokens=self.max_num_tokens,
                    shuffle=True,
                )
                self.val_batch_sampler = BatchSampler(
                    dataset=self.val_dataset,
                    max_tokens=self.max_num_tokens,
                    shuffle=False,
                )

    def train_dataloader(self):
        if self.use_batch_sampler:
            return DataLoader(
                self.train_dataset,
                batch_sampler=self.train_batch_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                multiprocessing_context=self.multiprocessing_context,
                pin_memory=self.pin_memory,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                multiprocessing_context=self.multiprocessing_context,
                pin_memory=self.pin_memory,
            )

    def val_dataloader(self):
        if self.use_batch_sampler:
            return DataLoader(
                self.val_dataset,
                batch_sampler=self.val_batch_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                multiprocessing_context=self.multiprocessing_context,
                pin_memory=self.pin_memory,
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                multiprocessing_context=self.multiprocessing_context,
                pin_memory=self.pin_memory,
            )


class BatchSampler(Sampler):
    """
    BatchSampler that creates length-efficient batches while respecting max_tokens constraint.
    Uses total sequence length (x1 + y1 + x2 + y2) as a proxy for computational cost.

    Note: Haven't extensively tested this class since it doesn't currently work with multi-GPU training
    """

    def __init__(
        self,
        dataset: Union[PairMSADataset, Subset],
        max_tokens: int,
        shuffle: bool = True,
    ):
        """
        Args:
            dataset: PairMSADataset object or Subset of PairMSADataset
            max_tokens: Maximum total tokens per batch (encoder + decoder)
            shuffle: Whether to shuffle batches during iteration
        """
        # Get the base dataset and the mapping to original indices if needed
        if isinstance(dataset, Subset):
            self.original_indices = dataset.indices  # Mapping to original dataset
            base_dataset = dataset.dataset
        else:
            self.original_indices = None
            base_dataset = dataset

        # Validate that we have a PairMSADataset
        if not isinstance(base_dataset, PairMSADataset):
            raise TypeError("Base dataset must be a PairMSADataset")

        # Calculate total lengths
        if self.original_indices is not None:
            # For Subset, calculate lengths only for subset indices
            self.total_lengths = [
                base_dataset.x1_len[i]
                + base_dataset.y1_len[i]
                + base_dataset.x2_len[i]
                + base_dataset.y2_len[i]
                for i in self.original_indices
            ]
        else:
            # For full dataset, calculate all lengths
            self.total_lengths = [
                base_dataset.x1_len[i]
                + base_dataset.y1_len[i]
                + base_dataset.x2_len[i]
                + base_dataset.y2_len[i]
                for i in range(len(base_dataset))
            ]

        self.max_tokens = max_tokens
        self.shuffle = shuffle

        # Sort indices relative to the provided dataset (not the original)
        self.local_indices = list(range(len(self.total_lengths)))
        self.sorted_indices = [
            idx for _, idx in sorted(zip(self.total_lengths, self.local_indices))
        ]

        # Pre-compute batches
        self.batches = self._create_batches()

    def _create_batches(self) -> list[list[int]]:
        """Create batches based on length and max_tokens constraint."""
        batches = []
        current_batch = []
        current_length = 0

        for idx in self.sorted_indices:
            sample_length = self.total_lengths[idx]

            if current_length + sample_length > self.max_tokens and current_batch:
                batches.append(current_batch)
                current_batch = [idx]
                current_length = sample_length
            else:
                current_batch.append(idx)
                current_length += sample_length

        if current_batch:
            batches.append(current_batch)

        return batches

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            rng = torch.Generator()
            batch_order = torch.randperm(len(self.batches), generator=rng).tolist()
            yield from (self.batches[i] for i in batch_order)
        else:
            yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


class PeintCollator:
    def __init__(
        self,
        vocab: Alphabet,
        which_protein: str = "y",
        mask_prob: float = 0.0,
        device: torch.device = torch.device("cuda"),
        return_aln_seq: bool = True,
        which_model: str = "peint",
    ):
        """
        Batch collator for Peint (which doesn't model complexes)
        Follows https://github.com/songlab-cal/protein-evolution/blob/91e64da3a87fe8497694acaac22aefdb233d2210/protevo/datasets/_torch_datasets.py#L442
        Also can be used to load single protein to evaluate Pipet

        Args:
            which_protein: Indicate which protein in the complex we want Peint to model
            either 'x' or 'y'
            return_aln_seq: If true, also return the tokenized aligned target sequence
            (no special token added) as a list of tensors
            which_model: Indicate which model we will use on the batched output
            the batched output will changed accordingly, either 'pipet' or 'peint'
        """
        self.vocab = vocab
        self.which_protein = which_protein
        if self.which_protein not in ["x", "y"]:
            raise ValueError("`which_protein` should be 'x' or 'y'")
        self.mask_prob = mask_prob
        self.device = device
        self.return_aln_seq = return_aln_seq
        self.which_model = which_model
        if self.which_model not in ["peint", "pipet"]:
            raise NotImplementedError

    def __call__(self, batch):
        if self.which_protein == "x":
            enc_seqs = [b["x1_toks"] for b in batch]
            dec_seqs = [b["x2_toks"] for b in batch]
            distances = [b["tx"] for b in batch]
            # Each sequence in the encoder has <cls> and <eos>
            enc_lengths = [b["x1_len"] + 2 for b in batch]
            # Each sequence in the decoder input has <eos>
            dec_lengths = [b["x2_len"] + 1 for b in batch]
            if self.return_aln_seq:
                dec_aln_seqs = [b["x2_aln_toks"] for b in batch]
                # This is only relevant for the peint dataset
                # which has insertion relative to query
                dec_aln_masks = (
                    [b["x2_aln_masks"] for b in batch] if "x2_aln_masks" in batch[0] else None
                )
        else:
            enc_seqs = [b["y1_toks"] for b in batch]
            dec_seqs = [b["y2_toks"] for b in batch]
            distances = [b["ty"] for b in batch]
            # Each sequence in the encoder has <cls> and <eos>
            enc_lengths = [b["y1_len"] + 2 for b in batch]
            # Each sequence in the decoder input has <eos>
            dec_lengths = [b["y2_len"] + 1 for b in batch]
            if self.return_aln_seq:
                dec_aln_seqs = [b["y2_aln_toks"] for b in batch]
                # This is only relevant for the peint dataset
                # which has insertion relative to query
                dec_aln_masks = (
                    [b["y2_aln_masks"] for b in batch] if "y2_aln_masks" in batch[0] else None
                )

        # Process encoder inputs
        enc_inputs, enc_targets = mask_for_mlm(
            seqs=enc_seqs,
            mask_prob=self.mask_prob,
            mask_idx=self.vocab.mask_idx,
            padding_idx=self.vocab.padding_idx,
        )
        # Add <cls>, <eos>, then pad --> shape (B, L)
        enc_inputs = [F.pad(seq, (0, 1), value=self.vocab.eos_idx) for seq in enc_inputs]
        enc_inputs = [F.pad(seq, (1, 0), value=self.vocab.cls_idx) for seq in enc_inputs]
        enc_inputs = nn.utils.rnn.pad_sequence(
            enc_inputs, batch_first=True, padding_value=self.vocab.padding_idx
        )
        enc_targets = [F.pad(seq, (0, 1), value=self.vocab.eos_idx) for seq in enc_targets]
        enc_targets = [F.pad(seq, (1, 0), value=self.vocab.cls_idx) for seq in enc_targets]
        enc_targets = nn.utils.rnn.pad_sequence(
            enc_targets, batch_first=True, padding_value=self.vocab.padding_idx
        )

        # Process decoder inputs
        # Add <eos> then pad the decoder sequence
        dec_seqs = [F.pad(seq, (0, 1), value=self.vocab.eos_idx) for seq in dec_seqs]
        dec_seqs = nn.utils.rnn.pad_sequence(
            dec_seqs, batch_first=True, padding_value=self.vocab.padding_idx
        )
        # Remove <eos> then add <cls> to start to form decoder input
        dec_inputs = F.pad(dec_seqs[:, :-1], (1, 0), value=self.vocab.cls_idx)
        dec_targets = dec_seqs

        if self.which_model == "peint":
            # Process distance, shape (B, 1)
            distance_tensor = torch.tensor(distances).unsqueeze(-1)
            # Peint needs padding mask to tell the model which positions are valid
            # for attention
            enc_padding_mask = enc_inputs == self.vocab.padding_idx
            dec_padding_mask = dec_inputs == self.vocab.padding_idx
            batch = (
                enc_inputs,
                enc_targets,
                dec_inputs,
                dec_targets,
                distance_tensor,
                enc_padding_mask,
                dec_padding_mask,
            )

        elif self.which_model == "pipet":
            # Pipet needs attn_mask_lengths to record the length of the sequence
            # Create masks to record the lengths of sequences in each pair
            # Non-zero values indicate the start and
            # length of each sequence, while zeros represent padding.
            attn_mask_enc_lengths = torch.zeros_like(enc_targets)
            attn_mask_dec_lengths = torch.zeros_like(dec_targets)
            # There is only one sequence
            attn_mask_enc_lengths[:, 0] = torch.tensor(enc_lengths)
            attn_mask_dec_lengths[:, 0] = torch.tensor(dec_lengths)
            # Store distance
            distances_tensor = torch.zeros_like(dec_targets, dtype=torch.float32)
            distances_tensor[:, 0] = torch.tensor(distances)
            batch = (
                enc_inputs,
                enc_targets,
                dec_inputs,
                dec_seqs,
                attn_mask_enc_lengths,
                attn_mask_dec_lengths,
                distances_tensor,
            )

        batch = [b.to(self.device) for b in batch]
        if self.return_aln_seq:
            batch.append(dec_aln_seqs)
            batch.append(dec_aln_masks)
        return batch


if __name__ == "__main__":
    vocab = Alphabet.from_architecture("ESM-1b")

    # Define your pair names (should match your .txt filenames)
    pair_names = ["pair_001"]  # This corresponds to pair_001.txt

    # Path to directory containing transition files
    transitions_dir = "/accounts/projects/yss/stephen.lu/protevo/plmr/data/misc"  # Directory where pair_001.txt is located

    # Create the data module
    data_module = PairMSADataModule(
        pair_names=pair_names,
        transitions_dir=transitions_dir,
        vocab=vocab,
        # Training parameters
        mask_prob=0.15,  # Probability of masking tokens for MLM
        batch_size=4,  # Batch size (start small for testing)
        train_frac=0.8,  # 80% for training, 20% for validation
        # Performance parameters
        num_workers=0,  # Number of workers for data loading
        use_batch_sampler=False,  # Set to True for dynamic batching
        max_num_tokens=200000,  # Max tokens per batch (only if use_batch_sampler=True)
        # Model-specific parameters
        shuffle_protein_order=False,  # Whether to randomly swap x and y proteins
        which_protein=None,  # None for Pipet (both proteins), "x" or "y" for Peint
        max_length=1022,  # Maximum sequence length
    )

    # Setup the data module (splits the dataset)
    data_module.setup(stage="fit")

    # Get the training dataloader
    train_loader = data_module.train_dataloader()

    # Get the first batch of data
    batch = next(iter(train_loader))

    [
        enc_inputs,
        enc_targets,
        dec_inputs,
        dec_seqs,
        attn_mask_enc_lengths,
        attn_mask_dec_lengths,
        distances_tensor,
    ] = batch

    # breakpoint()
