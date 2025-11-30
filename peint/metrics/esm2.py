"""
Compute MMD between two sets of protein sequences using ESM embeddings.
"""

from typing import Dict, List

import esm
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon


def get_esm_embeddings(sequences: List[str], model, alphabet, device: str = "cuda") -> torch.Tensor:
    """Extract mean-pooled ESM embeddings for a list of sequences."""
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()

    # Prepare batch
    data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    num_layers = model.num_layers

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[num_layers], return_contacts=False)
        token_embeddings = results["representations"][num_layers]

    # Mean pool over sequence length (excluding BOS/EOS)
    embeddings = []
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        emb = token_embeddings[i, 1 : seq_len + 1].mean(dim=0)
        embeddings.append(emb)

    return torch.stack(embeddings)


def mmd_rbf(X: torch.Tensor, Y: torch.Tensor, gamma: float = None) -> torch.Tensor:
    """
    Compute MMD with RBF kernel between two sets of embeddings.

    Args:
        X: (n, d) tensor of embeddings from distribution P
        Y: (m, d) tensor of embeddings from distribution Q
        gamma: RBF kernel bandwidth. If None, uses median heuristic.

    Returns:
        MMD^2 estimate (unbiased)
    """
    if gamma is None:
        # Median heuristic for bandwidth
        with torch.no_grad():
            dists = torch.cdist(X, Y)
            gamma = 1.0 / (2 * dists.median() ** 2 + 1e-8)

    XX = torch.cdist(X, X) ** 2
    YY = torch.cdist(Y, Y) ** 2
    XY = torch.cdist(X, Y) ** 2

    K_XX = torch.exp(-gamma * XX)
    K_YY = torch.exp(-gamma * YY)
    K_XY = torch.exp(-gamma * XY)

    n, m = X.shape[0], Y.shape[0]

    # Unbiased estimator
    sum_XX = (K_XX.sum() - K_XX.trace()) / (n * (n - 1))
    sum_YY = (K_YY.sum() - K_YY.trace()) / (m * (m - 1))
    sum_XY = K_XY.mean()

    return sum_XX + sum_YY - 2 * sum_XY


def get_threemer_counts(sequences: List[str]) -> np.ndarray:
    """
    Compute three-mer frequency distribution for a list of protein sequences.

    Args:
        sequences: List of protein sequences (single-letter amino acid codes)

    Returns:
        Normalized frequency array of shape (8000,) representing the distribution
        of all possible three-mers. Each value is the proportion of that three-mer
        among all three-mers in the sequences.
    """
    # Standard 20 amino acids
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    # Create mapping from three-mer to index
    threemer_to_idx = {}
    idx = 0
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            for aa3 in amino_acids:
                threemer_to_idx[aa1 + aa2 + aa3] = idx
                idx += 1

    # Count three-mers
    counts = np.zeros(8000, dtype=np.float64)
    total_threemers = 0

    for seq in sequences:
        # Convert to uppercase and filter to standard amino acids
        seq = seq.upper()
        seq_clean = "".join([aa for aa in seq if aa in amino_acids])

        # Extract all three-mers
        for i in range(len(seq_clean) - 2):
            threemer = seq_clean[i : i + 3]
            if threemer in threemer_to_idx:
                counts[threemer_to_idx[threemer]] += 1
                total_threemers += 1

    # Normalize to get frequency distribution
    if total_threemers > 0:
        counts = counts / total_threemers

    return counts


def compare_threemer_distributions(
    sequences_a: List[str], sequences_b: List[str]
) -> Dict[str, float]:
    """
    Compare three-mer distributions between two sets of protein sequences.

    This function computes normalized three-mer frequency distributions for both
    sequence sets and measures their similarity using divergence metrics that are
    independent of sequence set sizes and lengths.

    Args:
        sequences_a: First set of protein sequences
        sequences_b: Second set of protein sequences

    Returns:
        Dictionary containing similarity metrics:
        - 'jsd': Jensen-Shannon Divergence (0 = identical, 1 = completely different)
        - 'hellinger': Hellinger distance (0 = identical, 1 = completely different)
        - 'cosine_similarity': Cosine similarity (1 = identical, 0 = orthogonal)
    """
    # Get normalized frequency distributions
    dist_a = get_threemer_counts(sequences_a)
    dist_b = get_threemer_counts(sequences_b)

    # Jensen-Shannon Divergence (symmetric KL divergence)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    dist_a_safe = dist_a + eps
    dist_b_safe = dist_b + eps

    # Renormalize after adding epsilon
    dist_a_safe = dist_a_safe / dist_a_safe.sum()
    dist_b_safe = dist_b_safe / dist_b_safe.sum()

    jsd = jensenshannon(dist_a_safe, dist_b_safe)

    # Hellinger distance
    hellinger = np.sqrt(np.sum((np.sqrt(dist_a_safe) - np.sqrt(dist_b_safe)) ** 2)) / np.sqrt(2)

    # Cosine similarity (for reference)
    cosine_sim = np.dot(dist_a, dist_b) / (np.linalg.norm(dist_a) * np.linalg.norm(dist_b) + eps)

    return {
        "jsd": float(jsd),
        "hellinger": float(hellinger),
        "cosine_similarity": float(cosine_sim),
    }


def compute_mmd(
    sequences_p: List[str],
    sequences_q: List[str],
    model_name: str = "esm2_t30_150M_UR50D",
    device: str = "cuda",
    batch_size: int = 32,
) -> float:
    """
    Compute MMD between two sets of protein sequences.

    Args:
        sequences_p: First set of sequences
        sequences_q: Second set of sequences
        model_name: ESM model to use
        device: Device to run on
        batch_size: Batch size for embedding extraction

    Returns:
        MMD^2 value (0 = identical distributions, larger = more different)
    """
    # Load model
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)

    # Get embeddings in batches
    def embed_batched(seqs):
        all_embs = []
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i : i + batch_size]
            embs = get_esm_embeddings(batch, model, alphabet, device)
            all_embs.append(embs.cpu())
        return torch.cat(all_embs, dim=0)

    emb_p = embed_batched(sequences_p)
    emb_q = embed_batched(sequences_q)

    mmd_value = mmd_rbf(emb_p, emb_q)
    return mmd_value.item()


if __name__ == "__main__":
    # Example usage
    seqs_a = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGVQLTLFREVSEVGSGQFKHL",
    ]
    seqs_b = [
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVK",
        "QVQLQESGPGLVKPSETLSLTCTVSGGSVSSGDYYWTWIRQSPGKGLEWIGHIYYSGNT",
        "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIG",
    ]

    print("=== Three-mer Distribution Analysis ===")
    # Get three-mer counts for each set
    counts_a = get_threemer_counts(seqs_a)
    counts_b = get_threemer_counts(seqs_b)
    print(f"Three-mer distribution A: {counts_a.shape}, sum={counts_a.sum():.4f}")
    print(f"Three-mer distribution B: {counts_b.shape}, sum={counts_b.sum():.4f}")
    print(f"Non-zero three-mers in A: {np.count_nonzero(counts_a)}")
    print(f"Non-zero three-mers in B: {np.count_nonzero(counts_b)}")

    # Compare distributions
    similarity = compare_threemer_distributions(seqs_a, seqs_b)
    print(f"\nSimilarity metrics:")
    print(f"  Jensen-Shannon Divergence: {similarity['jsd']:.6f} (0=identical, 1=different)")
    print(f"  Hellinger distance: {similarity['hellinger']:.6f} (0=identical, 1=different)")
    print(f"  Cosine similarity: {similarity['cosine_similarity']:.6f} (1=identical, 0=orthogonal)")

    print("\n=== ESM-based MMD Analysis ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mmd = compute_mmd(seqs_a, seqs_b, device=device)
    print(f"MMD^2: {mmd:.6f}")
