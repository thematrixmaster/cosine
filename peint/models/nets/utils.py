import torch


def _expand_distances_to_seqlen(
    distances_in_length: torch.Tensor, attn_mask_in_length: torch.Tensor
) -> torch.Tensor:
    """
    Expand the per sequence distance to a positional encoding tensor
    using the length of each sequence in the batch
    If distances_in_length and attn_mask_in_length have different length
    the output will be the length of attn_mask_in_length
    """
    pos = torch.zeros_like(attn_mask_in_length, dtype=distances_in_length.dtype)
    B, _ = distances_in_length.size()
    for i in range(B):
        lengths = attn_mask_in_length[
            i, torch.nonzero(attn_mask_in_length[i, :], as_tuple=False).flatten()
        ].long()
        dists = distances_in_length[
            i, torch.nonzero(attn_mask_in_length[i, :], as_tuple=False).flatten()
        ].float()
        # Repeat dists for each length
        dists2 = dists.repeat_interleave(lengths)
        pos[i, : dists2.size(0)] = dists2
    return pos


def _create_sequence_mask(attn_mask_in_length: torch.Tensor, sequence_idx: int) -> torch.Tensor:
    """
    Generate a boolean mask for a specific sequence index across all batch elements.

    Args:
        attn_mask_in_length (torch.Tensor): Tensor of shape (B, L) containing nonzero elements
            that indicate the lengths of sequences in each batch element
        sequence_idx (int): The index of the sequence for which to generate the mask

    Returns:
        torch.Tensor: Boolean tensor of shape (B, L) where True indicates positions
            belonging to the sequence_idx'th sequence in each batch element

    Example:
        >>> attn_mask = torch.tensor([
        ...     [2, 4, 0, 0, 0, 0],
        ...     [3, 2, 0, 0, 0, 0]
        ... ])
        >>> get_sequence_mask(attn_mask, 0)
        tensor([[ True,  True, False, False, False, False],
                [ True,  True,  True, False, False, False]])
        >>> get_sequence_mask(attn_mask, 1)
        tensor([[False, False,  True,  True,  True,  True],
                [False, False, False,  True,  True, False]])
    """
    # Get batch size and sequence length
    batch_size, seq_len = attn_mask_in_length.shape
    device = attn_mask_in_length.device

    # Create cumulative sum of lengths, padding with zeros
    cumsum = torch.zeros((batch_size, seq_len + 1), device=device)
    cumsum[:, 1:] = torch.cumsum(attn_mask_in_length, dim=1)

    # Create position indices tensor
    positions = torch.arange(seq_len, device=device).expand(batch_size, -1)

    # Get start and end positions for the requested sequence
    start_pos = cumsum[:, sequence_idx]
    end_pos = cumsum[:, sequence_idx + 1]

    # Create mask where True indicates positions within the sequence_idx'th sequence
    mask = (positions >= start_pos.unsqueeze(1)) & (positions < end_pos.unsqueeze(1))

    return mask


def _create_padding_mask(attention_mask_in_length: torch.Tensor) -> torch.Tensor:
    """
    Create an attention padding mask from the lengths of sequences in batch
    The padding starts at the sum of the lengths of sequences in the batch
    Shape (B, L), True for positions that are NOT paddings
    """
    batch_size, seq_len = attention_mask_in_length.shape
    mask = torch.arange(seq_len, device=attention_mask_in_length.device).expand(batch_size, seq_len)
    return mask < attention_mask_in_length.sum(dim=-1, keepdim=True)


def _create_chain_mask(attention_mask_in_length: torch.Tensor, which_attn: str) -> torch.Tensor:
    """
    Helper function to create mask of shape (B, L, L) for residue pairs
    that are intra or inter chain
    """
    bsz, seq_len = attention_mask_in_length.shape
    chain_mask = torch.full((bsz, seq_len, seq_len), False, device=attention_mask_in_length.device)
    for b in range(bsz):
        len_x, len_y = attention_mask_in_length[b][:2]
        if which_attn == "intra":
            chain_mask[b, :len_x, :len_x] = True
            chain_mask[b, len_x : len_x + len_y, len_x : len_x + len_y] = True
        elif which_attn == "inter":
            chain_mask[b, :len_x, len_x : len_x + len_y] = True
            chain_mask[b, len_x : len_x + len_y, :len_x] = True
        else:
            raise ValueError("which_attn should be 'intra' or 'inter'")
    return chain_mask
