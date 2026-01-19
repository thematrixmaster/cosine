import random
from collections import deque
from itertools import product
from typing import Callable, List

import numpy as np
import torch
from ete3 import Tree

from evo.sequence import _FASTA_VOCAB
from evo.tokenization import Vocab


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
    generate_fn: Callable,
    root_sequence: str,
    tree: Tree,
    vocab: Vocab,
    device: str = "cpu",
    max_decode_steps: int = 200,
    max_batch_size: int = 1000,
    n_sequences: int = 5,
    x_sizes: torch.Tensor = None,
    p_threshold: float = 1.0,
    length_criterion: Callable[[str], bool] = lambda x: 50 <= x <= 200,
    likelihood_fn=None,
    max_retries: int = 3,
    seed: int = 42,
    verbose: bool = True,
    *args,
    **kwargs,
):
    """
    Simulates evolution on a fixed tree with given branch lengths using a trained PEINT model.
    Uses rejection sampling based on sequence length and optionally a likelihood function.
    Returns a dictionary mapping node names to their simulated sequences.
    """
    seed_ticker = 0

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

        # Print update with number of nodes sampled compared to total tree size
        n_leaves = len(tree)
        n_total = n_leaves * 2 - 1
        if verbose:
            print(f"Simulated {len(all_sequences)} / {n_total} nodes")

        # set a different seed each time we go through the model to increase diversity
        random.seed(seed + seed_ticker)
        np.random.seed(seed + seed_ticker)
        torch.manual_seed(seed + seed_ticker)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + seed_ticker)
        seed_ticker += 1
        # print(seed+seed_ticker)

        # Simulate child sequences using p(child | parent, time)
        with (torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad()):
            sequences = generate_fn(
                x=x_toks,
                t=time,
                x_sizes=x_sizes,
                device=device,
                p=p_threshold,
                max_decode_steps=max_decode_steps,
                *args,
                **kwargs,
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
                print(f"Retry {retry_count} for node {node.name} failed to pass filters.")
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


def simulate_evolution_with_rejection_sampling_batched(
    generate_fn: Callable,
    root_sequence: str,
    tree: Tree,
    vocab: Vocab,
    num_realizations: int = 10,
    device: str = "cpu",
    max_decode_steps: int = 200,
    max_batch_size: int = 1000,
    n_sequences: int = 5,
    x_sizes: torch.Tensor = None,
    p_threshold: float = 1.0,
    length_criterion: Callable[[str], bool] = lambda x: 50 <= x <= 200,
    likelihood_fn=None,
    max_retries: int = 3,
    base_seed: int = 42,
    verbose: bool = True,
    *args,
    **kwargs,
):
    """
    Simulates evolution on a fixed tree with given branch lengths using a trained PEINT model.
    This batched version simulates multiple independent realizations of the tree in parallel.

    Each realization independently samples mutations at every node, resulting in different
    sequences at leaves even though all start from the same root.

    Args:
        generate_fn: Function to generate sequences given parent sequences and branch lengths
        root_sequence: Starting sequence for all realizations
        tree: ete3.Tree structure defining the phylogeny
        vocab: Vocabulary for tokenization
        num_realizations: Number of independent tree realizations to simulate
        device: Device for computation ('cpu' or 'cuda')
        max_decode_steps: Maximum mutations per branch
        max_batch_size: Maximum sequences to process in single GPU call
        n_sequences: Number of candidates per rejection sampling attempt
        x_sizes: Tensor of sequence sizes (for models that need it)
        p_threshold: Nucleus sampling threshold
        length_criterion: Function to filter sequences by length
        likelihood_fn: Optional function to select best sequence
        max_retries: Maximum retry attempts for rejection sampling
        base_seed: Base random seed
        verbose: Print progress updates

    Returns:
        List[Dict[str, str]]: List of dictionaries (one per realization) mapping node names to sequences
    """

    # Set base seed for reproducibility
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    random.seed(base_seed)

    # Build node list in BFS order (same for all realizations)
    # Store as list of (node_name, parent_name, branch_length, is_leaf)
    root = tree.get_tree_root() if hasattr(tree, 'get_tree_root') else tree

    node_order = []
    node_info = {}  # node_name -> (parent_name, branch_length, children_names, is_leaf)

    bfs_queue = deque([root])
    node_info[root.name] = (None, 0.0, [c.name for c in root.children], root.is_leaf())

    while bfs_queue:
        node = bfs_queue.popleft()

        if node != root:
            parent_name = node.up.name
            branch_length = node.dist if node.dist is not None else 0.0
            node_order.append((node.name, parent_name, branch_length))

        for child in node.children:
            node_info[child.name] = (node.name, child.dist, [c.name for c in child.children], child.is_leaf())
            bfs_queue.append(child)

    if verbose:
        print(f"Tree structure: {len(node_info)} total nodes, {len(node_order)} edges to sample")
        print(f"Sampling {num_realizations} independent realizations")

    # Initialize storage for each realization
    # trajectories[realization_idx][node_name] = sequence
    trajectories = [{root.name: root_sequence} for _ in range(num_realizations)]

    # Track retry counts per (realization_idx, node_name)
    retry_counts = {}

    # Process nodes in BFS order
    nodes_processed = 0
    total_nodes = len(node_order) * num_realizations

    for node_name, parent_name, branch_length in node_order:
        # For each node, sample for all realizations
        # Build batch of parent sequences
        batch_sequences = []
        batch_branch_lengths = []
        batch_realization_indices = []

        for real_idx in range(num_realizations):
            parent_seq = trajectories[real_idx][parent_name]

            # Add n_sequences copies for rejection sampling
            batch_sequences.extend([parent_seq] * n_sequences)
            batch_branch_lengths.extend([branch_length] * n_sequences)
            batch_realization_indices.extend([real_idx] * n_sequences)

        # Tokenize and prepare tensors
        x_toks = torch.from_numpy(vocab.encode_batched_sequences(batch_sequences))
        time = torch.tensor(batch_branch_lengths, dtype=torch.float32).unsqueeze(-1)
        x_toks, time = x_toks.to(device), time.to(device)

        # Generate sequences
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sequences = generate_fn(
                x=x_toks,
                t=time,
                device=device,
                p=p_threshold,
                max_decode_steps=max_decode_steps,
                x_sizes=x_sizes,
                *args,
                **kwargs,
            )
            decoded_sequences = vocab.decode(sequences)

        # Process results for each realization
        for real_idx in range(num_realizations):
            start_idx = real_idx * n_sequences
            end_idx = start_idx + n_sequences
            node_sequences = decoded_sequences[start_idx:end_idx]

            # Get retry count
            retry_key = (real_idx, node_name)
            retry_count = retry_counts.get(retry_key, 0)

            # Apply rejection sampling
            chosen_sequence = filter_sequences(node_sequences, length_criterion, likelihood_fn)

            if chosen_sequence is None:
                if retry_count < max_retries:
                    if verbose and retry_count == 0:
                        print(f"Realization {real_idx}, node {node_name}: Failed length filter, retrying...")

                    # Retry with different random seed
                    retry_counts[retry_key] = retry_count + 1

                    # Generate new candidates for just this realization
                    parent_seq = trajectories[real_idx][parent_name]
                    retry_batch = [parent_seq] * n_sequences

                    x_retry = torch.from_numpy(vocab.encode_batched_sequences(retry_batch)).to(device)
                    t_retry = torch.tensor([branch_length] * n_sequences, dtype=torch.float32).unsqueeze(-1).to(device)

                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        retry_seqs = generate_fn(
                            x=x_retry,
                            t=t_retry,
                            device=device,
                            p=p_threshold,
                            max_decode_steps=max_decode_steps,
                            x_sizes=x_sizes,
                            *args,
                            **kwargs,
                        )
                        retry_decoded = vocab.decode(retry_seqs)

                    chosen_sequence = filter_sequences(retry_decoded, length_criterion, likelihood_fn)

                # If still None, use random sequence from original batch
                if chosen_sequence is None:
                    if verbose:
                        print(f"Realization {real_idx}, node {node_name}: Max retries reached, using fallback")
                    chosen_sequence = random.choice(node_sequences)

            # Store chosen sequence
            trajectories[real_idx][node_name] = chosen_sequence
            nodes_processed += 1

        # Progress update
        if verbose and (nodes_processed % (num_realizations * 5) == 0 or nodes_processed == total_nodes):
            progress = nodes_processed / total_nodes * 100
            print(f"Progress: {nodes_processed}/{total_nodes} ({progress:.1f}%) nodes sampled")

    if verbose:
        print(f"Completed all {num_realizations} realizations")

        # Verify independence at leaves
        leaves = [node_name for node_name, info in node_info.items() if info[3]]
        if len(leaves) > 0:
            example_leaf = leaves[0]
            leaf_seqs = [trajectories[i][example_leaf] for i in range(num_realizations)]
            unique_seqs = len(set(leaf_seqs))
            print(f"Independence check: Leaf '{example_leaf}' has {unique_seqs}/{num_realizations} unique sequences")

    return trajectories


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
