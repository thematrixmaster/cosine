import os
import random
import sys
from collections import deque
from itertools import product
from typing import Callable, List

import numpy as np
import torch
from ete3 import Tree

from evo.sequence import _FASTA_VOCAB
from evo.tokenization import Vocab
from peint.models.modules.peint_module import PEINTModule
from peint.models.nets.peint import PEINT, ESMEncoder


def load_from_old_checkpoint(checkpoint_path: str, device: str = "cpu") -> PEINTModule:
    """
    Load a PEINTModule from an old checkpoint format.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    hyperparams = checkpoint.get("hyper_parameters", {})

    enc_model = ESMEncoder.from_pretrained(
        dropout_p=hyperparams.get("dropout_p", 0.0),
        finetune=False,
        embed_x_per_chain=False,
    )
    peint = PEINT(
        enc_model=enc_model,
        evo_vocab=enc_model.vocab,
        num_chains=1,
        causal_decoder=True,
        use_chain_embedding=False,
        num_heads=hyperparams["num_heads"],
        embed_dim=hyperparams["embed_dim"],
        max_len=hyperparams["max_seq_len"],
        num_encoder_layers=hyperparams["num_encoder_layers"],
        num_decoder_layers=hyperparams["num_decoder_layers"],
        dropout_p=hyperparams.get("dropout_p", 0.0),
        use_attention_bias=hyperparams.get("use_attention_bias", True),
    )

    module = PEINTModule(
        net=peint,
        weight_decay=hyperparams.get("weight_decay", 0.0),
        compile=hyperparams.get("compile", False),
    )

    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model.esm"):
            continue
        assert k.startswith("model.")
        state_dict[k.replace("model.", "net.")] = v

    missing_keys, unexpected_keys = module.load_state_dict(state_dict=state_dict)
    assert all(k.startswith("net.esm") for k in missing_keys)
    assert len(unexpected_keys) == 0

    return module.to(device).eval()


def load_from_new_checkpoint(checkpoint_path: str, device: str = "cpu") -> PEINTModule:
    """
    Load a PEINTModule from a checkpoint that contains the PEINT model.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")

    import peint

    sys.modules["plmr"] = peint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hyperparams = checkpoint.get("hyper_parameters", {})
    net = hyperparams.get("net")

    _net_type = type(net).__name__
    if _net_type != "PEINT":
        raise ValueError("Checkpoint does not contain a valid PEINT model.")

    if net.finetune_esm:
        module = PEINTModule.load_from_checkpoint(checkpoint_path=checkpoint_path, strict=True)
    else:
        peint = PEINT.from_pretrained_esm2(
            embed_dim=net.embed_dim,
            num_heads=net.num_heads,
            num_encoder_layers=net.num_encoder_layers,
            num_decoder_layers=net.num_decoder_layers,
            max_len=net.max_len,
            dropout_p=net.dropout_p,
            use_attention_bias=net.use_bias,
            finetune_esm=net.finetune_esm,
        )
        module = PEINTModule.load_from_checkpoint(checkpoint_path, net=peint, strict=False)

    del sys.modules["plmr"]
    return module.to(device).eval()


def sampling_function(logits, p=0.9, argmax_sample=False):
    """
    Perform top p sampling on the given logits.

    Args:
    logits (torch.Tensor): Logits of shape [batch_size, vocab_size]
    p (float): Nucleus sampling parameter, default is 0.9

    Returns:
    torch.Tensor: Sampled token indices of shape [batch_size, 1]
    """
    probs = torch.nn.functional.softmax(logits, dim=-1)

    if argmax_sample or p == 0.0:
        return probs.argmax(-1, keepdim=True)

    # If p == 1, perform standard sampling
    if p >= 1.0:
        return torch.multinomial(probs, 1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cumulative_probs < p

    # Create a mask for the nucleus
    nucleus_mask = nucleus.clone()
    nucleus_mask[:, 1:] = nucleus[:, :-1]
    nucleus_mask[:, 0] = True
    sorted_probs = sorted_probs.masked_fill(~nucleus_mask, 0)

    # Redistribute the probabilities
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    sampled_indices = torch.multinomial(sorted_probs, 1)
    next_tok = torch.gather(sorted_indices, 1, sampled_indices)

    return next_tok


def filter_sequences(
    decoded_sequences: List[str],
    length_criterion: Callable[[str], bool],
    likelihood_fn: Callable[[str], bool] | None = None,
) -> str | None:
    """
    Filters the decoded sequences based on a length criterion and optionally a likelihood function.
    Returns a single sequence that meets the criteria, or None if no sequences are valid.
    If a likelihood function is provided, the sequence with the highest likelihood is returned.
    """
    filtered_sequences = [seq for seq in decoded_sequences if length_criterion(len(seq))]
    if not filtered_sequences:
        return None

    if likelihood_fn:
        likelihoods = [likelihood_fn(seq) for seq in filtered_sequences]
        chosen_sequence = filtered_sequences[np.argmax(likelihoods)]
    else:
        chosen_sequence = random.choice(filtered_sequences)

    return chosen_sequence


def simulate_evolution_with_rejection_sampling(
    model: PEINTModule,
    root_sequence: str,
    tree: Tree,
    vocab: Vocab,
    device: str = "cpu",
    max_decode_steps: int = 200,
    max_batch_size: int = 1000,
    n_sequences: int = 5,
    p_threshold: float = 1.0,
    length_criterion: Callable[[str], bool] = lambda x: 50 <= x <= 200,
    likelihood_fn=None,
    max_retries: int = 3,
    seed: int = 42,
):
    """
    Simulates evolution on a fixed tree with given branch lengths using a trained PEINT model.
    Uses rejection sampling based on sequence length and optionally a likelihood function.
    Returns a dictionary mapping node names to their simulated sequences.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # (node, sequence, remaining_children, retry_count)
    queue = deque([(tree, root_sequence, list(tree.children), 0)])

    # results: { real_node_name: simulated_sequence }
    all_sequences = {}

    while queue:
        batch_nodes = []
        batch_sequences = []
        batch_branch_lengths = []
        batch_retry_counts = []

        effective_branch_count = max_batch_size // n_sequences

        while queue and len(batch_nodes) < max_batch_size:
            node, parent_sequence, remaining_children, retry_count = queue.popleft()

            # Extract all the children of the current node,
            # which we will simulate using the parent sequence, up to the maximum batch size
            if remaining_children:
                child = remaining_children.pop(0)
                batch_nodes.extend([child] * n_sequences)
                batch_sequences.extend([parent_sequence] * n_sequences)
                batch_branch_lengths.extend([child.dist] * n_sequences)
                batch_retry_counts.extend([retry_count] * n_sequences)

                if remaining_children:
                    queue.appendleft((node, parent_sequence, remaining_children, retry_count))

                if len(batch_nodes) // n_sequences >= effective_branch_count:
                    break

        if not batch_nodes:
            continue

        # Tokenize the sequences and prepare the branch lengths
        x_toks = torch.from_numpy(vocab.encode_batched_sequences(batch_sequences))
        time = torch.tensor(batch_branch_lengths, dtype=torch.float32).unsqueeze(-1)
        x_toks, time = x_toks.to(device), time.to(device)

        # Simulate child sequences using p(child | parent, time)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sequences = model.net.generate(
                x=x_toks,
                t=time,
                device=device,
                max_decode_steps=max_decode_steps,
                p=p_threshold,
            )
            decoded_sequences = vocab.decode(sequences)

        for i in range(0, len(batch_nodes), n_sequences):
            node = batch_nodes[i]  # real node in the tree that we simulated
            node_sequences = decoded_sequences[i : i + n_sequences]  # simulated sequences for node
            retry_count = batch_retry_counts[
                i
            ]  # number of failed attempts to find simulated sequences

            # Rejection sampling based on length criterion and likelihood function
            chosen_sequence = filter_sequences(node_sequences, length_criterion, likelihood_fn)

            if chosen_sequence is None:
                if retry_count < max_retries:
                    # Add the parent node back to the left of the queue for another try
                    parent_node = node.up
                    queue.appendleft(
                        (
                            parent_node,
                            all_sequences.get(parent_node.name, root_sequence),
                            [node],
                            retry_count + 1,
                        )
                    )
                else:
                    # If max retries reached, use a random sequence from the generated ones
                    chosen_sequence = random.choice(node_sequences)

            if chosen_sequence is not None:
                all_sequences[node.name] = chosen_sequence
                if not node.is_leaf():
                    # Notice new parent sequence is the simulated sequence, not the real one
                    queue.append(
                        (node, chosen_sequence, list(node.children), 0)
                    )  # Reset retry count for child nodes

    return all_sequences


def get_mutations(input_seqs: np.ndarray, only_subs=False):
    """For each seq, return [len_seq, alphabet_size] subs,
    [len_seq+1, alphabet_size] insertions, [len_seq] deletions and concatenate them.
    Turn invalid muts into nans.

    Parameters:
    input_seqs: numpy array
        Sequences in OHE representation. Can have any number of dims. Must not have
        a stop. Assumes last entry of OHE representation is empty (raises error if not).
    only_subs: bool or int, default = False
        Only return substitutions. If int, only return substitutions within only_subs
        in the cyclic nbh.

    Returns:
    all_seqs: numpy array
        Same OHE length as original, so if not only_subs, make sure input seqs has extra
        OHE column (raises warning if not).
    signs: numpy array
        Signs from lexicographic ordering: 1 for deletions, -1 for insertions,
        1 if substitution is > wild-type and -1 otherwise.
    """
    assert np.sum(input_seqs[..., -1, :]) == 0, "No extra entry in OHE!"
    shape = np.shape(input_seqs)[:-2]
    seqs = input_seqs.reshape((-1,) + np.shape(input_seqs)[-2:]).astype(float)
    num_seqs, seq_len, alphabet_size = np.shape(input_seqs)
    assert alphabet_size > 1 or not only_subs, "no subs for single letter alphabet!"
    seq_len = seq_len - 1
    empty = np.sum(seqs, axis=-1) == 0  # empty positions
    empty_for_ins = np.concatenate([np.zeros([num_seqs, 1], dtype=bool), empty[:, :-1]], axis=-1)

    # First substitutions.
    if alphabet_size > 1:
        # take only nbh if only_subs is an int
        if not isinstance(only_subs, bool):
            nbh = np.r_[np.arange(only_subs), alphabet_size - 2 - np.arange(only_subs)]
            nbh = np.isin(range(alphabet_size - 1), nbh)
        else:
            nbh = np.s_[:]
        perm_mat = np.eye(alphabet_size)
        perm_mat = np.r_[perm_mat[1:], perm_mat[[0]]]
        substitutions = np.array(
            [
                [
                    np.concatenate(
                        [
                            seqs[:, :i, :],
                            np.einsum(
                                "nb,bk->nk", seqs[:, i], np.linalg.matrix_power(perm_mat, b + 1)
                            )[:, None, :],
                            seqs[:, i + 1 :, :],
                        ],
                        axis=-2,
                    )
                    for b in range(alphabet_size - 1)
                ]
                for i in range(seq_len)
            ]
        )
        substitutions = np.transpose(substitutions, [2, 0, 1, 3, 4])[:, :, nbh, :, :]
        substitutions[empty[:, :-1]] = np.nan  # substitutions in empty positions are nan'ed
        substitutions = substitutions.reshape([num_seqs, -1, seq_len + 1, alphabet_size])
        sub_signs = np.cumsum(seqs[..., ::-1], axis=-1)[..., ::-1]
        sub_signs = sub_signs[..., 1:][..., ::-1]
        sub_signs[
            np.tile(empty[..., None], len(np.shape(empty)) * (1,) + (np.shape(sub_signs)[-1],))
        ] = np.nan
        sub_signs = sub_signs[..., :-1, :][..., nbh]
        sub_signs = 2 * sub_signs.reshape([num_seqs, -1]) - 1
    else:
        substitutions = []
        sub_signs = []

    if not only_subs:
        # Then deletions.
        deletions = np.array(
            [
                np.concatenate(
                    [
                        seqs[:, :i, :],
                        seqs[:, i + 1 :, :],
                        np.tile(np.zeros(alphabet_size)[None, None, :], (num_seqs, 1, 1)),
                    ],
                    axis=-2,
                )
                for i in range(seq_len)
            ]
        )
        deletions = np.transpose(deletions, [1, 0, 2, 3])
        deletions[empty[:, :-1]] = np.nan
        del_signs = np.ones(np.shape(deletions)[:-2])
        del_signs[empty[:, :-1]] = np.nan

        # Finally, insertions
        insertions = np.array(
            [
                [
                    np.concatenate(
                        [
                            seqs[:, :i, :],
                            np.tile(base[None, None, :], (num_seqs, 1, 1)),
                            seqs[:, i:-1, :],
                        ],
                        axis=-2,
                    )
                    for base in np.eye(alphabet_size)
                ]
                for i in range(seq_len + 1)
            ]
        )
        insertions = np.transpose(insertions, [2, 0, 1, 3, 4])
        insertions[empty_for_ins] = np.nan
        ins_signs = -np.ones(np.shape(insertions)[:-2])
        ins_signs[empty_for_ins] = np.nan
        insertions = insertions.reshape([num_seqs, -1, seq_len + 1, alphabet_size])
        ins_signs = ins_signs.reshape([num_seqs, -1])

    # Concatenate.
    if only_subs:
        all_seqs = substitutions
        all_signs = sub_signs
    else:
        all_seqs = np.concatenate(
            [substitutions] * (alphabet_size > 1) + [insertions, deletions], axis=1
        ).reshape(shape + (-1,) + np.shape(input_seqs)[-2:])
        all_signs = np.concatenate(
            [sub_signs] * (alphabet_size > 1) + [ins_signs, del_signs], axis=1
        )
    return all_seqs, all_signs


def single_substitution_names(sequence: str, vocab=_FASTA_VOCAB) -> List[str]:
    """Returns the names of all single mutants of a sequence."""
    mutants = []
    for (i, wt), mut in product(enumerate(sequence), vocab):
        if wt == mut or wt != "-":
            continue
        mutant = f"{wt}{i + 1}{mut}"
        mutants.append(mutant)
    return mutants
