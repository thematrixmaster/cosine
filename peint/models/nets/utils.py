import torch


def _expand_distances_to_seqlen(distances_in_length, attn_mask_in_length) -> torch.Tensor:
    """
    Expand the per sequence distance to a positional encoding tensor.

    Args:
        distances_in_length: (batch_size, num_chains) - time values per chain
        attn_mask_in_length: (batch_size, num_chains) - lengths per chain

    Returns:
        pos: (batch_size, max_seq_len) - time values expanded to token positions
    """
    B, num_chains = distances_in_length.shape
    max_seq_len = attn_mask_in_length.sum(dim=1).max().item()
    pos = torch.zeros(B, max_seq_len, device=distances_in_length.device, dtype=torch.float32)

    for i in range(B):
        current_pos = 0
        for j in range(num_chains):
            chain_length = attn_mask_in_length[i, j].item()
            if chain_length > 0:
                chain_distance = distances_in_length[i, j].item()
                # Fill positions for this chain with its distance value
                pos[i, current_pos : current_pos + chain_length] = chain_distance
                current_pos += chain_length

    return pos


def _create_sequence_mask(attn_mask_in_length: torch.Tensor, sequence_idx: int) -> torch.Tensor:
    """
    Generate a boolean mask for a specific sequence index across all batch elements.

    Args:
        attn_mask_in_length (torch.Tensor): Tensor of shape (B, num_chains) containing
            the lengths of each chain in each batch element
        sequence_idx (int): The index of the sequence for which to generate the mask

    Returns:
        torch.Tensor: Boolean tensor of shape (B, max_seq_len) where True indicates positions
            belonging to the sequence_idx'th sequence in each batch element

    Example:
        >>> attn_mask = torch.tensor([
        ...     [2, 4],  # First batch: chain 0 has length 2, chain 1 has length 4
        ...     [3, 2]   # Second batch: chain 0 has length 3, chain 1 has length 2
        ... ])
        >>> _create_sequence_mask(attn_mask, 0)
        tensor([[ True,  True, False, False, False, False],
                [ True,  True,  True, False, False, False]])
        >>> _create_sequence_mask(attn_mask, 1)
        tensor([[False, False,  True,  True,  True,  True],
                [False, False, False,  True,  True, False]])
    """
    batch_size, num_chains = attn_mask_in_length.shape
    max_seq_len = attn_mask_in_length.sum(dim=1).max().item()
    device = attn_mask_in_length.device

    # Create mask tensor
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)

    for i in range(batch_size):
        current_pos = 0
        for j in range(num_chains):
            chain_length = attn_mask_in_length[i, j].item()
            if chain_length > 0:
                if j == sequence_idx:
                    # Mark positions for the requested sequence
                    mask[i, current_pos : current_pos + chain_length] = True
                current_pos += chain_length

    return mask


def _create_padding_mask(
    attention_mask_in_length: torch.Tensor, max_seq_len: int = None
) -> torch.Tensor:
    """
    Create an attention padding mask from the lengths of chains in batch.

    Args:
        attention_mask_in_length: (batch_size, num_chains) - lengths of each chain
        max_seq_len: Optional maximum sequence length to use instead of calculating from data

    Returns:
        mask: (batch_size, max_seq_len) - True for positions that are NOT paddings
    """
    batch_size, num_chains = attention_mask_in_length.shape
    if max_seq_len is None:
        max_seq_len = attention_mask_in_length.sum(dim=1).max().item()
    device = attention_mask_in_length.device

    # Create position indices for the full sequence length
    positions = torch.arange(max_seq_len, device=device).expand(batch_size, -1)
    # Total sequence lengths per batch element
    total_lengths = attention_mask_in_length.sum(dim=1, keepdim=True)

    return positions < total_lengths


def _create_chain_mask(attention_mask_in_length: torch.Tensor, which_attn: str) -> torch.Tensor:
    """
    Helper function to create mask of shape (B, L, L) for residue pairs
    that are intra or inter chain.

    Args:
        attention_mask_in_length: (batch_size, num_chains) - lengths of each chain
        which_attn: "intra" or "inter"

    Returns:
        chain_mask: (batch_size, max_seq_len, max_seq_len) - attention mask
    """
    bsz, num_chains = attention_mask_in_length.shape
    max_seq_len = attention_mask_in_length.sum(dim=1).max().item()
    chain_mask = torch.full(
        (bsz, max_seq_len, max_seq_len), False, device=attention_mask_in_length.device
    )

    for b in range(bsz):
        current_pos = 0
        chain_positions = []

        # Calculate start and end positions for each chain
        for j in range(num_chains):
            chain_length = attention_mask_in_length[b, j].item()
            if chain_length > 0:
                chain_positions.append((current_pos, current_pos + chain_length))
                current_pos += chain_length
            else:
                chain_positions.append((0, 0))  # Empty chain

        # Create intra-chain or inter-chain masks
        if which_attn == "intra":
            # Allow attention within each chain
            for start, end in chain_positions:
                if end > start:
                    chain_mask[b, start:end, start:end] = True
        elif which_attn == "inter":
            # Allow attention between different chains
            for i, (start_i, end_i) in enumerate(chain_positions):
                for j, (start_j, end_j) in enumerate(chain_positions):
                    if i != j and end_i > start_i and end_j > start_j:
                        chain_mask[b, start_i:end_i, start_j:end_j] = True
        else:
            raise ValueError("which_attn should be 'intra' or 'inter'")

    return chain_mask
