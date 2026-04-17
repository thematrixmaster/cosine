from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from esm.modules import gelu
from torch import Tensor
from torchtyping import TensorType as TT

from cosine.models.nets.utils import _create_chain_mask, _create_padding_mask

# Optional flash attention imports
try:
    from flash_attn import (
        flash_attn_kvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
    )
    from flash_attn.bert_padding import (
        pad_input,
        unpad_input,
        unpad_input_for_concatenated_sequences,
    )
    from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding
    from flash_attn.layers.rotary import apply_rotary_emb_torch, rotate_half
    from flash_attn.ops.triton.rotary import apply_rotary
except ImportError:
    apply_rotary = None
    flash_attn_varlen_kvpacked_func = None
    flash_attn_varlen_qkvpacked_func = None
    flash_attn_kvpacked_func = None
    FlashRotaryEmbedding = None
    apply_rotary_emb_torch = None
    rotate_half = None
    pad_input = None
    unpad_input = None
    unpad_input_for_concatenated_sequences = None


########################################################
# Embedding Modules (Time and Other Conditionings)
########################################################


class SinusoidalTimeEmbedding(nn.Module):
    """
    Simple sinusoidal time embedding for Transformer model
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, t: TT["batch", 1, "float"]) -> TT["batch", "hidden_dim"]:
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # Make it (batch_size, 1)

        half_dim = self.hidden_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t * emb.unsqueeze(0)  # Broadcasting: (batch_size, 1) * (1, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Handle odd hidden_dim
        if self.hidden_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb  # (batch_size, hidden_dim)


class MultiSequenceRotaryPositionalEncoding(nn.Module):
    """
    Modified Rotary Positional Encoding (RoPE) module that applies positional encoding
    to input tensors based on sequence lengths within each example in a batch.

    Args:
        dim (int): The dimension of the input features (usually the head dimension).
        base (int, optional): The base for the positional encoding calculation.
                              Defaults to 10000.
        use_fp32_for_idx (bool, optional): Whether to use float32 for positional indices.
                                            This is important when working with larger
                                            sequence lengths in lower precision.
                                            Defaults to True.

    Input:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_length, num_heads, head_dim).
        index_tensor (torch.Tensor): Tensor of shape (batch_size, seq_length) containing
                                     the lengths of sequences in each example within the batch.

    Output:
        torch.Tensor: Tensor of the same shape as input `x` with rotary positional
                      encoding applied.

    Example:
        >>> B, L, n, h = 2, 10, 20, 32
        >>> rope_module = ModifiedRotaryPositionalEncoding(h)
        >>> x = torch.randn(B, L, n, h)
        >>> index_tensor = torch.tensor([
        ...     [4, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        ...     [3, 5, 0, 0, 0, 0, 0, 0, 0, 0]
        ... ])
        >>> output = rope_module(x, index_tensor)
    """

    def __init__(self, dim: int, base: int = 10000, use_fp32_for_idx: bool = True):
        super().__init__()

        self.dim = dim
        self.base = base
        self.use_fp32_for_idx = use_fp32_for_idx

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _compute_rope(self, positions: torch.Tensor):
        freqs = torch.einsum("bl,d->bld", positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def _create_positional_indices(self, index_tensor: torch.Tensor):
        B, _ = index_tensor.shape
        device = index_tensor.device
        # Create position indices
        dtype = torch.float32 if self.use_fp32_for_idx else index_tensor.dtype
        positions = torch.zeros_like(index_tensor, dtype=dtype, device=device)
        for i in range(B):
            lengths = index_tensor[i, torch.nonzero(index_tensor[i, :], as_tuple=False).flatten()]
            pos = torch.cat([torch.arange(l, device=device) for l in lengths], dim=0)
            positions[i, : pos.shape[0]] = pos
        return positions

    def forward(self, x: torch.Tensor, index_tensor: torch.Tensor):
        """
        Apply rotary positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, num_heads, head_dim).
            index_tensor (torch.Tensor): Tensor of shape (batch_size, seq_length) containing
                                         the lengths of sequences in each batch.

        Returns:
            torch.Tensor: Tensor of the same shape as input `x` with rotary positional
                          encoding applied.
        """
        B, L, n, h = x.shape

        # Create positional indexes based on the index_tensor
        positions = self._create_positional_indices(index_tensor)

        # Compute RoPE components
        cos, sin = self._compute_rope(positions)

        # Reshape for broadcasting
        cos = cos.view(B, L, 1, h)
        sin = sin.view(B, L, 1, h)

        # Apply RoPE
        return (x * cos) + (rotate_half(x) * sin)


class GeometricTimeEmbedder(nn.Module):
    def __init__(self, frequency_embedding_size=256, start=1e-5, stop=0.25):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.start = start
        self.stop = stop

    def timestep_embedding(self, timesteps, dim):
        freqs = torch.tensor(
            np.geomspace(start=self.start, stop=self.stop, num=dim // 2), dtype=timesteps.dtype
        ).to(timesteps.device)
        args = timesteps[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_emb = self.timestep_embedding(t, dim=self.frequency_embedding_size)
        return t_emb


class TimestepEmbedderNew(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=1280):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = (
            2
            * 3.14159
            * torch.exp(
                -math.log(max_period)
                * (torch.arange(start=0, end=half, dtype=torch.float32) - half / 3)
                / half
            ).to(device=t.device)
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class MLPEmbedder(nn.Module):
    """
    Embeds a vector of scalars into a higher dimensional embeddings using an MLP
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x):
        x = x.to(self.mlp[0].weight.dtype)
        return self.mlp(x)


class TorchRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        device=None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            # We want fp32 here as well since inv_freq will be multiplied with t, and the output
            # will be large. Having it in bf16 will lose a lot of precision and cause the
            # cos & sin output to change significantly.
            # We want to recompute self.inv_freq if it was not loaded in fp32
            if self.inv_freq.dtype != torch.float32:
                inv_freq = self._compute_inv_freq(device=device)
            else:
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to bf16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    @staticmethod
    def rotate_half(x: Tensor, interleaved=False):
        if not interleaved:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        else:
            x1, x2 = x[..., ::2], x[..., 1::2]
            return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)

    @staticmethod
    def apply_rotary_emb_torch(x: Tensor, cos: Tensor, sin: Tensor, interleaved=False):
        """
        x: (batch_size, seqlen, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
        """
        ro_dim = cos.shape[-1] * 2
        assert ro_dim <= x.shape[-1]
        cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
        sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
        return torch.cat(
            [
                x[..., :ro_dim] * cos
                + TorchRotaryEmbedding.rotate_half(x[..., :ro_dim], interleaved) * sin,
                x[..., ro_dim:],
            ],
            dim=-1,
        )

    def forward(
        self,
        q,
        k,
        max_seqlen: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) if kv is none,
             else it's just q of shape (batch, seqlen, nheads, headdim)
        kv: (batch, seqlen, 2, nheads, headdim)
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding *inplace* to qkv and / or kv.
        """
        seqlen_q = q.shape[1]
        seqlen_k = k.shape[1]
        self._update_cos_sin_cache(max_seqlen, device=q.device, dtype=q.dtype)
        q = TorchRotaryEmbedding.apply_rotary_emb_torch(
            q,
            self._cos_cached[:seqlen_q, :],
            self._sin_cached[:seqlen_q, :],
        )
        k = TorchRotaryEmbedding.apply_rotary_emb_torch(
            k, self._cos_cached[:seqlen_k, :], self._sin_cached[:seqlen_k, :], interleaved=False
        )
        return q, k


######################################
# Flash Multi-Head Attention Modules #
######################################


class RopeFlashMHA(nn.Module):
    """Flash Multi-Head Attention module for transformer, with Rotary Embedding"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        add_bias_kv=False,
        dropout=0.0,
        self_attn=True,
        causal: bool = False,
        layer_idx=None,
    ):
        super().__init__()

        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.causal = causal
        self.self_attn = self_attn
        self.layer_idx = layer_idx
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv

        hidden_size = embed_dim
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        # NOTE: the FlashRotaryEmbedding module is used here, make sure that pos_idx_in_fp32 is set to True,
        # otherwise positional indexing will fail for large indices due to range limitations of bf16
        # i.e. 265 = 270 due to rounding
        self.rot_emb = FlashRotaryEmbedding(
            dim=self.head_dim,
            base=10000.0,
            interleaved=False,
            scale_base=None,
            # pos_idx_in_fp32=True,  # very important for bf16
        )

    def forward(
        self,
        x,
        y=None,
        x_attn_mask=None,
        y_attn_mask=None,
        *args,
        **kwargs,
    ):
        Bx, Lx, Dx = x.size()

        if self.self_attn:
            # project x to q, k, v
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            # y comes from encoder, provides keys and values
            assert y is not None, "Cross attention requires y input"
            q = self.q_proj(x)
            k = self.k_proj(y)
            v = self.v_proj(y)

        # rescale q
        q *= self.head_dim**-0.5

        q = rearrange(q, "b l (n h) -> b l n h", n=self.num_heads)
        k = rearrange(k, "b l (n h) -> b l n h", n=self.num_heads)
        v = rearrange(v, "b l (n h) -> b l n h", n=self.num_heads)

        # NOTE: flash atten's rot emb performs this in-place
        q, k = self.rot_emb(
            q,
            torch.stack([k, v], dim=2),
            seqlen_offset=0,
            max_seqlen=max(
                q.shape[1], k.shape[1]
            ),  # this is important if q, k are not the same shape
        )

        # at this point, k contains k and v, and is of shape [B, L, 2, N, H]
        if x_attn_mask is None:
            x_attn_mask = torch.ones(Bx, Lx, device=x.device, dtype=torch.bool)

        q, idx_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, x_attn_mask)

        if self.self_attn:
            k, idx_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, x_attn_mask)  # k = kv

            qkv = torch.cat([q.unsqueeze(1), k], dim=1)  # (total_nonpad, 3, N, H)
            out = flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens_q,
                max_seqlen_q,
                dropout_p=self.dropout,
                softmax_scale=1.0,  # q has been scaled already
                causal=self.causal,
            )
        else:
            # cross attention
            if y_attn_mask is None:
                By, Ly, Dy = y.size()
                y_attn_mask = torch.ones(By, Ly, device=y.device, dtype=torch.bool)
            k, idx_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, y_attn_mask)

            out = flash_attn_varlen_kvpacked_func(
                q,
                k,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=self.dropout,
                softmax_scale=1.0,  # q has been scaled already
                causal=self.causal,
            )

        out = pad_input(out, idx_q, Bx, Lx)  # repad
        out = rearrange(out, "... h d -> ... (h d)")  # concatenate heads

        return self.out_proj(out)  # linear projection


class FlashMHAEncoderBlock(nn.Module):
    """Flash Multi-Head Attention Encoder Block for transformer, with Rotary Embedding.
    This implementation yields identical results to that of ESM2's MHA.
    """

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        use_bias=True,
        add_bias_kv=False,
        dropout_p=0.0,
        layer_idx=None,
        **kwargs,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.ffn_embed_dim = ffn_embed_dim
        self.add_bias_kv = add_bias_kv
        self.dropout_p = dropout_p
        self.layer_idx = layer_idx
        self.attention_heads = attention_heads

        self.self_attn = RopeFlashMHA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            bias=use_bias,
            add_bias_kv=add_bias_kv,
            dropout=dropout_p,
            self_attn=True,
            causal=False,
            layer_idx=layer_idx,
        )

        # layer norms
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        # ffn layers
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x, x_attn_mask=None, **kwargs):
        """
        Forward pass using the x sequence. kwargs should contain the x_attn_mask
        """
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x_attn_mask=x_attn_mask)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x


class FlashMHADecoderBlock(nn.Module):
    """Flash Multi-Head Attention Decoder Block for transformer, with Rotary Embedding.
    Will perform both causal self-attention and cross-attention."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        use_bias=True,
        add_bias_kv=False,
        dropout_p=0.0,
        layer_idx=None,
        causal=True,
        **kwargs,
    ):
        super().__init__()

        self.causal = causal
        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.ffn_embed_dim = ffn_embed_dim
        self.add_bias_kv = add_bias_kv
        self.dropout_p = dropout_p
        self.layer_idx = layer_idx
        self.attention_heads = attention_heads

        self.self_attn = RopeFlashMHA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            # bias=use_bias,
            # add_bias_kv=add_bias_kv,
            dropout=dropout_p,
            self_attn=True,
            causal=causal,
            layer_idx=layer_idx,
        )
        self.cross_attn = RopeFlashMHA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            # bias=use_bias,
            # add_bias_kv=add_bias_kv,
            dropout=dropout_p,
            self_attn=False,
            causal=False,
            layer_idx=layer_idx,
        )

        # layer norms
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        # ffn layers
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x, y, x_attn_mask=None, y_attn_mask=None, **kwargs):
        """
        Forward pass using the x sequence and the y sequence for cross-attention.

        The unpadding and repadding is done in the FlashMHA module so that rotary embeddings can be used.

        My notation is not ideal - the x, y refer to the x,y sequences from an x,y,t trio, but since this is
        the decoder, x is y and y is x (i.e. y provides the q, while k,v come from x).
        This is handled in the overall transformer forward pass (reassigning x to y and the padding masks accordingly).
        """
        # self-attention (causal or bidirectional)
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x=x, x_attn_mask=x_attn_mask)
        x = residual + x

        # cross attention
        residual = x
        x = self.cross_attn_layer_norm(x)
        x = self.cross_attn(x=x, y=y, x_attn_mask=x_attn_mask, y_attn_mask=y_attn_mask)
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x


##########################
## KV Caching Functions ##
##########################


class KVCached_MHCA(RopeFlashMHA):
    """For single token decoding - q will be shape B x 1 x D"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        bias: bool = True,
        dropout: float = 0.0,
        layer_idx: Optional[int] = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            self_attn=False,
            causal=False,
            dropout=dropout,
            layer_idx=layer_idx,
        )

        self.max_seq_len = max_seq_len
        self.kv_cache = None
        self.cache_size = 0

    def init_kv_cache(self, batch_size):
        # Initialize the KV cache with empty tensors
        self.kv_cache = torch.empty(
            batch_size,
            self.max_seq_len,
            2,
            self.num_heads,
            self.head_dim,
            device=self.q_proj.weight.device,
        )
        self.cache_size = 0

    def forward(
        self,
        x,
        y=None,
        x_attn_mask=None,
        y_attn_mask=None,
        decoder_cache_size=0,
        *args,
        **kwargs,
    ):
        Bx, Lx, Dx = x.size()

        if self.kv_cache is None or self.kv_cache.size(0) != Bx:
            self.init_kv_cache(Bx)

        q = self.q_proj(x)
        q *= self.head_dim**-0.5
        q = rearrange(q, "b l (n h) -> b l n h", n=self.num_heads)

        if y is not None:
            # Update KV cache
            By, Ly, Dy = y.size()
            k = self.k_proj(y)
            v = self.v_proj(y)
            k = rearrange(k, "b l (n h) -> b l n h", n=self.num_heads)
            v = rearrange(v, "b l (n h) -> b l n h", n=self.num_heads)

            # rotate q and k,v here
            q, kv = self.rot_emb(
                q, torch.stack([k, v], dim=2), seqlen_offset=0, max_seqlen=self.max_seq_len
            )

            self.kv_cache[:, self.cache_size : self.cache_size + Ly] = kv
            self.cache_size += Ly

        else:
            # Use cached KV
            kv = self.kv_cache[:, : self.cache_size].to(q.dtype)

            # rotate just q, use the cached_cos_sin
            # check if need to update
            if Lx + decoder_cache_size > self.max_seq_len:
                self.rot_emb._update_cos_sin_cache(
                    Lx + decoder_cache_size, device=q.device, dtype=q.dtype
                )

            cos, sin = self.rot_emb._cos_cached, self.rot_emb._sin_cached
            q = apply_rotary_emb_torch(
                q, cos[decoder_cache_size], sin[decoder_cache_size]
            )  # this gets the seqlen offset for you

        if x_attn_mask is None:
            x_attn_mask = torch.ones(Bx, Lx, device=x.device, dtype=torch.bool)

        q, idx_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, x_attn_mask)

        if y_attn_mask is None:
            y_attn_mask = torch.ones(Bx, self.cache_size, device=x.device, dtype=torch.bool)

        kv, idx_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(kv, y_attn_mask)

        out = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=self.dropout,
            softmax_scale=1.0,
            causal=self.causal,
        )

        out = pad_input(out, idx_q, Bx, Lx)
        out = rearrange(out, "... h d -> ... (h d)")

        return self.out_proj(out)


class EncoderCachedFlashMHCA(RopeFlashMHA):
    """Cache the encoder, pass in a full y sequence for likelihood evaluation."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        bias: bool = True,
        dropout: float = 0.0,
        layer_idx: Optional[int] = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            self_attn=False,
            causal=False,
            dropout=dropout,
            layer_idx=layer_idx,
        )

        self.max_seq_len = max_seq_len
        self.kv_cache = None
        self.cache_size = 0

    def init_kv_cache(self, batch_size):
        # Initialize the KV cache with empty tensors
        self.kv_cache = torch.empty(
            batch_size,
            self.max_seq_len,
            2,
            self.num_heads,
            self.head_dim,
            device=self.q_proj.weight.device,
        )
        self.cache_size = 0

    def forward(self, x, y=None, x_padding_mask=None, y_padding_mask=None):
        Bx, Lx, Dx = x.size()

        if self.kv_cache is None or self.kv_cache.size(0) < Bx:
            self.init_kv_cache(Bx)

        q = self.q_proj(x)
        q *= self.head_dim**-0.5
        q = rearrange(q, "b l (n h) -> b l n h", n=self.num_heads)

        if y is not None:
            # Update KV cache
            By, Ly, Dy = y.size()
            k = self.k_proj(y)
            v = self.v_proj(y)
            k = rearrange(k, "b l (n h) -> b l n h", n=self.num_heads)
            v = rearrange(v, "b l (n h) -> b l n h", n=self.num_heads)

            kv = torch.stack([k, v], dim=2)

            self.kv_cache[:, self.cache_size : self.cache_size + Ly] = kv
            self.cache_size += Ly
        else:
            # Use cached KV
            # note that batch size may be smaller than max during final batch
            kv = self.kv_cache[:Bx, : self.cache_size].to(q.dtype)

        q, kv = self.rot_emb(q, kv, seqlen_offset=0, max_seqlen=max(q.shape[1], kv.shape[1]))

        if x_padding_mask is None:
            x_padding_mask = torch.ones(Bx, Lx, device=x.device, dtype=torch.bool)

        q, idx_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, x_padding_mask)

        if y_padding_mask is None:
            y_padding_mask = torch.ones(Bx, self.cache_size, device=x.device, dtype=torch.bool)

        kv, idx_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(kv, y_padding_mask)

        out = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=self.dropout,
            softmax_scale=1.0,
            causal=self.causal,
        )

        out = pad_input(out, idx_q, Bx, Lx)
        out = rearrange(out, "... h d -> ... (h d)")

        return self.out_proj(out)


class KVCached_MHSA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        bias: bool = True,
        dropout: float = 0.0,
        causal: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len
        self.causal = causal
        self.dropout = dropout
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.rot_emb = FlashRotaryEmbedding(
            dim=self.head_dim,
            base=10000.0,
            interleaved=False,
            scale_base=None,
            # pos_idx_in_fp32=True,
        )

        self.kv_cache = None
        self.cache_size = 0

    def init_kv_cache(self, batch_size, seq_len=None):
        if seq_len is None:
            seq_len = self.max_seq_len
        self.kv_cache = torch.empty(
            batch_size, seq_len, 2, self.num_heads, self.head_dim, device=self.q_proj.weight.device
        )
        self.cache_size = 0

    def forward(self, x, x_attn_mask=None, *args, **kwargs):
        batch_size, seq_len, _ = x.shape

        if self.kv_cache is None or self.kv_cache.size(0) != batch_size:
            self.init_kv_cache(batch_size)

        assert self.cache_size + seq_len <= self.kv_cache.size(1), "KV cache is full"

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q *= self.head_dim**-0.5

        q = rearrange(q, "b l (n h) -> b l n h", n=self.num_heads)
        k = rearrange(k, "b l (n h) -> b l n h", n=self.num_heads)
        v = rearrange(v, "b l (n h) -> b l n h", n=self.num_heads)

        # Apply rotary embeddings - returns k (kv)
        q, k = self.rot_emb(q, torch.stack([k, v], dim=2), seqlen_offset=self.cache_size)

        # Update KV cache
        self.kv_cache[:, self.cache_size : self.cache_size + seq_len] = k

        # Combine current and cached KV
        k = self.kv_cache[:, : self.cache_size + seq_len].to(q.dtype)

        # Update cache size
        self.cache_size += seq_len

        output = flash_attn_kvpacked_func(
            q=q,
            kv=k,
            dropout_p=self.dropout,
            softmax_scale=1.0,  # q already rescaled
            causal=self.causal,
        )
        output = rearrange(output, "b l h d -> b l (h d)")

        return self.out_proj(output)

    def reset_kv_cache(self):
        if self.kv_cache is not None:
            self.kv_cache.zero_()
        self.cache_size = 0


class KV_CachedFlashMHADecoderBlock(nn.Module):
    """Flash Multi-Head Attention Decoder Block for transformer, with Rotary Embedding.
    Will perform both causal self-attention and cross-attention."""

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        use_bias: bool = True,
        add_bias_kv: bool = False,
        dropout_p: float = 0.0,
        max_encoder_seq_len: int = 1024,
        max_decoder_seq_len: int = 1024,
        layer_idx: Optional[int] = None,
        causal: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.ffn_embed_dim = ffn_embed_dim
        self.add_bias_kv = add_bias_kv
        self.dropout_p = dropout_p
        self.layer_idx = layer_idx
        self.attention_heads = attention_heads
        self.max_encoder_seq_len = max_encoder_seq_len
        self.max_decoder_seq_len = max_decoder_seq_len

        self.self_attn = KVCached_MHSA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            bias=use_bias,
            causal=causal,
            layer_idx=layer_idx,
            dropout=dropout_p,
            max_seq_len=max_decoder_seq_len,  # can decode longer
        )

        self.cross_attn = KVCached_MHCA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            max_seq_len=max_encoder_seq_len,
            layer_idx=layer_idx,
            dropout=dropout_p,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x, y, x_attn_mask=None, y_attn_mask=None, **kwargs):

        # causal self-attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x=x, x_attn_mask=x_attn_mask)
        x = x + residual

        # cross attention
        residual = x
        x = self.cross_attn_layer_norm(x)
        x = self.cross_attn(
            x=x,
            y=y,
            x_attn_mask=x_attn_mask,
            y_attn_mask=y_attn_mask,
            decoder_cache_size=self.self_attn.cache_size
            - 1,  # -1 because it gets updated in the prior step
        )
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        return x


class EncoderCachedFlashMHADecoderBlock(nn.Module):
    """Caches the encoder output for likelihood evaluation. Assumes that y is passed in as the full y sequence.
    Can still be used for generation, but will be slower than the fully KV cached version, as you perform a full
    attention pass on the entire y rather than the last token."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        use_bias=True,
        add_bias_kv=False,
        dropout_p=0.0,
        layer_idx=None,
        max_seq_len: int = 1024,
        **kwargs,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.ffn_embed_dim = ffn_embed_dim
        self.add_bias_kv = add_bias_kv
        self.dropout_p = dropout_p
        self.layer_idx = layer_idx
        self.attention_heads = attention_heads
        self.max_seq_len = max_seq_len

        self.self_attn = RopeFlashMHA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            self_attn=True,
            causal=True,
            layer_idx=layer_idx,
            dropout=dropout_p,
        )

        self.cross_attn = EncoderCachedFlashMHCA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            max_seq_len=max_seq_len,
            layer_idx=layer_idx,
            dropout=dropout_p,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x, y, **kwargs):
        self_attn_kwargs = {"x_padding_mask": kwargs.get("x_padding_mask", None)}

        # causal self-attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x=x, **self_attn_kwargs)
        x = x + residual

        # cross attention
        residual = x
        x = self.cross_attn_layer_norm(x)
        x = self.cross_attn(x=x, y=y, **kwargs)
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        return x


##############################################################
## Non-Flash Versions of the MHA and Encoder/Decoder Blocks ##
##############################################################


class VanillaMHA(nn.Module):
    """Vanilla Attention module for transformer using ROPE

    Args:
    head_dim: int - the dimension of the attention head
    num_heads: int - the number of attention heads
    causal: bool - whether to use causal attention (in decoder self-attention)
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        add_bias_kv=False,
        dropout=0.0,
        self_attn=True,
        causal: bool = False,
        layer_idx=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.layer_idx = layer_idx
        self.dropout = dropout
        self.causal = causal
        self.self_attn = self_attn

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.rot_emb = TorchRotaryEmbedding(
            dim=self.head_dim,
            base=10000.0,
            interleaved=False,
            scale_base=None,
        )

    def forward(
        self,
        x,
        y=None,
        attention_mask=None,
        *args,
        **kwargs,
    ):
        if self.self_attn:
            # project x to q, k, v
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            # y comes from encoder, provides keys and values
            assert y is not None, "Cross attention requires y input"
            q = self.q_proj(x)
            k = self.k_proj(y)
            v = self.v_proj(y)

        # rescale q
        q *= self.head_dim**-0.5

        q = rearrange(q, "b l (n h) -> b l n h", n=self.num_heads, h=self.head_dim)
        k = rearrange(k, "b l (n h) -> b l n h", n=self.num_heads, h=self.head_dim)
        v = rearrange(v, "b l (n h) -> b l n h", n=self.num_heads, h=self.head_dim)

        # rotary here
        q, k = self.rot_emb(q, k, max_seqlen=max(q.shape[1], k.shape[1]))

        q = rearrange(q, "b l n h -> b n l h")
        k = rearrange(k, "b l n h -> b n l h")
        v = rearrange(v, "b l n h -> b n l h")

        dots = torch.einsum("bhid,bhjd->bhij", q, k)  # q already scaled

        if attention_mask is not None:
            # attention mask should be positions to mask
            dots = dots.masked_fill(attention_mask[:, None, None, :], float("-inf"))

        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j, device=dots.device, dtype=bool).triu(j - i + 1)
            dots = dots.masked_fill(mask, float("-inf"))

        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b n l h -> b l (n h)", n=self.num_heads, h=self.head_dim)

        return self.out_proj(out), attn


class ESMEncoderBlock(nn.Module):
    """Vanilla Multi-Head Attention Encoder Block for transformer, compatible with ESM2.
    Notably, this module is identical to the FlashMHAEncoderBlock, but does not use FLASH
    Positional encoding is handled using Rotary Embeddings.
    """

    def __init__(
        self,
        embed_dim,
        attention_heads,
        ffn_embed_dim,
        use_bias=True,
        dropout_p=0.0,
        layer_idx=None,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads

        self.self_attn = VanillaMHA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            bias=use_bias,
            dropout=dropout_p,
            self_attn=True,
            layer_idx=layer_idx,
        )

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x, x_padding_mask, **kwargs):
        # atten block
        residual = x
        x = self.self_attn_layer_norm(x)
        x, att = self.self_attn(x, attention_mask=x_padding_mask)
        x = x + residual
        # MLP block
        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        return x


class ESMDecoderBlock(nn.Module):
    """Vanilla Multi-Head Attention Decoder Block for transformer, compatible with ESM2."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        use_bias=True,
        add_bias_kv=False,
        dropout_p=0.0,
        layer_idx=None,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = attention_heads
        self.head_dim = embed_dim // attention_heads
        self.layer_idx = layer_idx
        self.dropout = dropout_p

        self.self_attn = VanillaMHA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            bias=use_bias,
            dropout=dropout_p,
            self_attn=True,
            causal=True,
            layer_idx=layer_idx,
        )

        self.cross_attn = VanillaMHA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            bias=use_bias,
            dropout=dropout_p,
            self_attn=False,
            layer_idx=layer_idx,
            causal=False,
        )

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x, y, x_padding_mask=None, y_padding_mask=None, **kwargs):
        # self attend block
        residual = x
        x = self.self_attn_layer_norm(x)
        x, self_att = self.self_attn(
            x, x, x_padding_mask
        )  # here the x is the decoding sequence (y)
        x = x + residual
        # cross attend block
        residual = x
        x = self.cross_attn_layer_norm(x)
        x, cross_att = self.cross_attn(x, y, y_padding_mask)
        x = x + residual
        # MLP block
        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        # return x, (self_att, cross_att)
        return x


#############################################################################
## Small vanilla Transformer model with timestep and positional embeddings ##
#############################################################################


class VanillaTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        output_size: int = None,
        num_heads=8,
        max_seq_len=512,
    ):
        super(VanillaTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.output_size = output_size if output_size is not None else vocab_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)  # Token embeddings
        self.time_embedding = nn.Sequential(  # Time embeddings
            SinusoidalTimeEmbedding(hidden_dim=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)  # Positional embeddings
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.output_size),  # Output logits
        )
        self._init_weights()  # Initialize weights

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        tokens: TT["batch", "x_seq_len", "long"],
        time_step: TT["batch", 1, "float"],
        padding_mask: TT["batch", "x_seq_len", "bool"],
    ) -> Dict[str, torch.Tensor]:
        # Tuple[
        #     TT["batch", "x_seq_len", "float"],  # Rates (3 values)
        #     TT["batch", "x_seq_len", "vocab_size"],  # Insert probabilities (vocab_size values)
        #     TT["batch", "x_seq_len", "vocab_size"],  # Substitute probabilities (vocab_size values)
        # ]:
        """Forward pass takes in x_t, t, and padding mask, returns rates and probabilities"""
        batch_size, x_seq_len = tokens.shape
        token_emb = self.token_embedding(tokens)  # (batch_size, x_seq_len, hidden_dim)

        time_emb = self.time_embedding(time_step)  # (batch_size, hidden_dim)
        time_emb = time_emb.unsqueeze(1).expand(
            -1, x_seq_len, -1
        )  # (batch_size, x_seq_len, hidden_dim)

        positions = (
            torch.arange(x_seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        )
        pos_emb = self.pos_embedding(positions)  # (batch_size, x_seq_len, hidden_dim)

        x = token_emb + time_emb + pos_emb  # (batch_size, x_seq_len, hidden_dim)
        x = x.transpose(0, 1)  # expects (x_seq_len, batch_size, hidden_dim)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)

        x = x.transpose(0, 1)  # (batch_size, x_seq_len, hidden_dim)
        x = self.final_layer_norm(x)  # (batch_size, x_seq_len, hidden_dim)
        logits = self.output_head(x)  # (batch_size, x_seq_len, output_size)

        # ins_logits = self.ins_logits_out(x)  # (batch_size, x_seq_len, vocab_size)
        # sub_logits = self.sub_logits_out(x)  # (batch_size, x_seq_len, vocab_size)
        # rates = F.softplus(self.rates_out(x))  # (batch_size, x_seq_len, 3) - ensure positive rates

        # ins_probs = F.softmax(ins_logits, dim=-1)  # (batch_size, x_seq_len, vocab_size)
        # sub_probs = F.softmax(sub_logits, dim=-1)  # (batch_size, x_seq_len, vocab_size)

        # Zero out outputs for padded positions
        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # (batch_size, x_seq_len, 1)
        logits = logits * mask_expanded

        # rates = rates * mask_expanded
        # ins_probs = ins_probs * mask_expanded
        # sub_probs = sub_probs * mask_expanded

        if torch.isnan(logits).any():
            raise ValueError("NaN detected in logits output")

        return dict(logits=logits)


########################################################################################
## Adapted from flash_attn trition based rotary to support variable sequence length   ##
## https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py ##
########################################################################################


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, cos, sin, cu_seqlens, max_seqlen):
        q, k = qkv[:, 0], qkv[:, 1]

        apply_rotary(q, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, inplace=True)
        apply_rotary(k, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, inplace=True)

        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen

        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        max_seqlen = ctx.max_seqlen
        cos, sin, cu_seqlens = ctx.saved_tensors

        dq, dk = dqkv[:, 0], dqkv[:, 1]

        apply_rotary(
            dq,
            cos,
            sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            inplace=True,
            conjugate=True,
        )
        apply_rotary(
            dk,
            cos,
            sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            inplace=True,
            conjugate=True,
        )

        return dqkv, None, None, None, None


def apply_rotary_emb_qkv_(qkv, cos, sin, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
    """Apply rotary embedding *inplace* to the first rotary_dim of Q and K.

    Arguments:
        qkv: (batch_size * seqlen, 3, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        cu_seqlen: (batch_size + 1) the cumulative sum of the sequence lengths
        max_seqlen: int the maximum sequence length in the batch
    Return:
        qkv: (batch_size * seqlen, 3, nheads, headdim)
    """
    return ApplyRotaryEmbQKV_.apply(qkv, cos, sin, cu_seqlens, max_seqlen)


class FAEsmRotaryEmbedding(torch.nn.Module):
    """The rotary position embeddings from RoFormer_ (Su et.

    al).
    """

    def __init__(self, dim: int, base=10000.0, pos_idx_in_fp32=True, device=None, persistent=True):
        """
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=persistent)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq

            freqs = torch.outer(t, inv_freq)
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(
        self, qkv: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int, *args, **kwargs
    ) -> torch.Tensor:
        """Apply rotary embedding *inplace*.

        Arguments:
            qkv: (batch * seqlen, 3, nheads, headdim) query, key, value.
            cu_seqlens: (batch + 1) the cumulative sum of the sequence lengths.
            max_seqlen: int the maximum sequence length in the batch.
        Return:
            qkv: (batch_size * seqlen, 3, nheads, headdim)
        """
        self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)

        return apply_rotary_emb_qkv_(
            qkv,
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )


##############################################################
## Multi Sequence (within same sample) Attention Modules    ##
##############################################################


class IntraOnlyMultiSequenceSelfAttention(nn.Module):
    """
    Multi-head self-attention module for sets of sequences with restricted attention within each sequence.
    (i.e. only attention "intra" each protein)

    This module applies attention separately to each sequence within a set, using rotary positional
    encoding that resets for each sequence. It utilizes Flash Attention for efficient computation.
    This works for single sequences in the set as well, and should replicate a normal FlashAttentionMHA module.

    Args:
        hidden_size (int): Size of the input and output features.
        num_attention_heads (int): Number of attention heads.
        max_position_embeddings (int): Maximum number of position embeddings.
        attention_dropout (float, optional): Dropout probability for attention weights. Defaults to 0.0.

    Example:
        >>> hidden_size = 768
        >>> num_attention_heads = 12
        >>> max_position_embeddings = 2048
        >>> attention = MultiSequenceMultiHeadAttention(hidden_size, num_attention_heads, max_position_embeddings)
        >>> hidden_states = torch.randn(2, 10, hidden_size)
        >>> attention_mask_in_length = torch.tensor([[4, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        ...                                          [3, 5, 0, 0, 0, 0, 0, 0, 0, 0]])
        >>> output = attention(hidden_states, attention_mask_in_length)
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        dropout_p=0.0,
        causal=False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        assert (
            self.head_dim * num_attention_heads == hidden_size
        ), "hidden_size must be divisible by num_attention_heads"

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.rotary_emb = MultiSequenceRotaryPositionalEncoding(
            dim=self.head_dim,
            use_fp32_for_idx=True,
        )

        self.dropout_p = dropout_p
        self.causal = causal
        self.layer_idx = layer_idx

    def forward(self, hidden_states, attention_mask_in_length):
        """

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
                Contains the input features for the attention mechanism.
            attention_mask_in_length (torch.Tensor): Tensor of shape (batch_size, seq_len) containing
                the lengths of sequences in each batch. Non-zero values indicate the start and
                length of each sequence, while zeros represent padding.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size) containing
                the attended features.

        Note:
            This method assumes that sequences within each batch are concatenated and padded to
            the maximum sequence length. The `attention_mask_in_length` tensor is used to identify
            the boundaries of individual sequences within each batch. It also keeps track of padding
        """
        bsz, q_len, _ = hidden_states.shape

        # Project input to query, key, and value
        qkv = self.qkv_proj(hidden_states)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_attention_heads
        )

        # Split qkv into separate tensors
        q, k, v = qkv.unbind(dim=2)
        q = q * self.head_dim**-0.5  # rescale

        # Apply rotary positional encoding to q and k
        q_rotary = self.rotary_emb(q, attention_mask_in_length)
        k_rotary = self.rotary_emb(k, attention_mask_in_length)

        # Recombine into qkv
        qkv_rotary = torch.stack([q_rotary, k_rotary, v], dim=2)

        # Prepare for Flash Attention
        qkv_flash = rearrange(qkv_rotary, "b s three h d -> b s (three h d)")
        qkv_unpad, indices, cu_q_lens, max_s = unpad_input_for_concatenated_sequences(
            qkv_flash, attention_mask_in_length
        )
        qkv_unpad = rearrange(
            qkv_unpad,
            "nnz (three h d) -> nnz three h d",
            three=3,
            h=self.num_attention_heads,
        )

        # Apply Flash Attention
        output_unpad = flash_attn_varlen_qkvpacked_func(
            qkv_unpad,
            cu_q_lens,
            max_s,
            self.dropout_p,
            softmax_scale=1.0,  # query has already been scaled
            causal=self.causal,
        )

        # Pad the output back to the original shape
        output = rearrange(
            pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len),
            "b s (h d) -> b s h d",
            h=self.num_attention_heads,
        )

        # Final projection
        attn_output = rearrange(output, "b s h d -> b s (h d)")
        attn_output = self.out_proj(attn_output)

        return attn_output


class FullMultiSequenceSelfAttention(nn.Module):
    """
    Multi-head self-attention module for pairs of sequences with full attention between sequences.

    This module applies attention to pairs of sequences (e.g., interacting proteins) within each item in a batch.
    It uses rotary positional encoding that resets for each sequence in the pair, while allowing full attention
    between the sequences. It utilizes Flash Attention for efficient computation.

    Args:
        hidden_size (int): Size of the input and output features.
        num_attention_heads (int): Number of attention heads.
        dropout_p (float, optional): Dropout probability for attention weights. Defaults to 0.0.
        causal (bool, optional): Whether to apply causal masking. Defaults to False.
        layer_idx (Optional[int], optional): Layer index, used for logging or debugging. Defaults to None.

    Attributes:
        hidden_size (int): Size of the input and output features.
        num_attention_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        qkv_proj (nn.Linear): Linear projection for query, key, and value.
        out_proj (nn.Linear): Output projection.
        rotary_emb (MultiSequenceRotaryPositionalEncoding): Rotary positional encoding module.
        dropout_p (float): Dropout probability.
        causal (bool): Whether to apply causal masking.
        layer_idx (Optional[int]): Layer index.

    Example:
        >>> hidden_size = 768
        >>> num_attention_heads = 12
        >>> attention = IntraOnlyMultiSequenceSelfAttention(hidden_size, num_attention_heads)
        >>> hidden_states = torch.randn(2, 1000, hidden_size)
        >>> attention_mask_in_length = torch.tensor([[446, 355, 0, ..., 0],
        ...                                          [335, 355, 0, ..., 0]])
        >>> output = attention(hidden_states, attention_mask_in_length)
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        dropout_p=0.0,
        causal=False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        assert (
            self.head_dim * num_attention_heads == hidden_size
        ), "hidden_size must be divisible by num_attention_heads"

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.rotary_emb = MultiSequenceRotaryPositionalEncoding(
            dim=self.head_dim,
            use_fp32_for_idx=True,
        )

        self.dropout_p = dropout_p
        self.causal = causal
        self.layer_idx = layer_idx

    def forward(self, hidden_states, attention_mask_in_length):
        """
        Forward pass of the multi-sequence multi-head self-attention module.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
                Contains the input features for the attention mechanism.
            attention_mask_in_length (torch.Tensor): Tensor of shape (batch_size, seq_len) containing
                the lengths of sequences in each batch. Non-zero values indicate the length of each sequence,
                while zeros represent padding. For example, [446, 355, 0, ..., 0] represents two sequences
                of lengths 446 and 355 in one item of the batch.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size) containing
                the attended features.
        """
        bsz, seq_len, _ = hidden_states.shape

        # Project input to query, key, and value
        qkv = self.qkv_proj(hidden_states)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_attention_heads
        )

        # Split qkv into separate tensors
        q, k, v = qkv.unbind(dim=2)
        q = q * self.head_dim**-0.5  # rescale

        # Apply rotary positional encoding to q and k
        q_rotary = self.rotary_emb(q, attention_mask_in_length)
        k_rotary = self.rotary_emb(k, attention_mask_in_length)

        # Recombine into qkv
        qkv_rotary = torch.stack([q_rotary, k_rotary, v], dim=2)
        qkv_rotary = rearrange(qkv_rotary, "b s three h d -> b s (three h d)")

        # Create padding mask
        # Note: the main difference to Treeformer is the padding_mask here
        # is on the full sequence and we use `unpad_input`
        # rather than `unpad_input_for_concatenated_sequences` with attention_mask_in_length
        padding_mask = _create_padding_mask(attention_mask_in_length)
        # Unpad inputs
        qkv_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(qkv_rotary, padding_mask)
        qkv_unpad = rearrange(
            qkv_unpad,
            "nnz (three h d) -> nnz three h d",
            three=3,
            h=self.num_attention_heads,
        )

        # Apply Flash Attention
        output_unpad = flash_attn_varlen_qkvpacked_func(
            qkv_unpad,
            cu_seqlens,
            max_seqlen,
            self.dropout_p,
            softmax_scale=1.0,  # query has already been scaled
            causal=self.causal,
        )

        # Pad the output back to the original shape
        output = rearrange(
            pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, seq_len),
            "b s (h d) -> b s h d",
            h=self.num_attention_heads,
        )

        # Final projection
        attn_output = rearrange(output, "b s h d -> b s (h d)")
        attn_output = self.out_proj(attn_output)

        return attn_output


class DecoupledIntraInterMultiSequenceSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        dropout_p=0.0,
        causal=False,
        layer_idx: Optional[int] = None,
    ):
        """
        Implements decoupled self-attention for processing multiple protein sequences.
        This module computes attention weights separately for:
        1) Positions within the same protein chain (intra-attention)
        2) Positions between different chains (inter-attention)

        The final output combines both types of attention.

        Args:
            hidden_size (int): Total hidden dimension size
            num_attention_heads (int): Number of attention heads
            dropout_p (float): Dropout probability (currently unused)
            causal (bool): Whether to apply causal masking
            layer_idx (Optional[int]): Layer index for logging/debugging

        Attributes:
            head_dim (int): Hidden dimension size per attention head
            qkv_proj_intra (nn.Linear): QKV projection for intra-chain attention
            qkv_proj_inter (nn.Linear): QKV projection for inter-chain attention
            out_proj (nn.Linear): Output projection
            rotary_emb (MultiSequenceRotaryPositionalEncoding): Rotary position encoding
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        assert (
            self.head_dim * num_attention_heads == hidden_size
        ), "hidden_size must be divisible by num_attention_heads"

        self.qkv_proj_intra = nn.Linear(hidden_size, 3 * hidden_size)
        self.qkv_proj_inter = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.rotary_emb = MultiSequenceRotaryPositionalEncoding(
            dim=self.head_dim,
            use_fp32_for_idx=True,
        )

        self.dropout_p = dropout_p
        self.causal = causal
        self.layer_idx = layer_idx

        # Add dropout layers
        self.attention_dropout = nn.Dropout(dropout_p)

    def compute_attention_weights(
        self,
        hidden_states,
        attention_mask_in_length,
        which_attn,
    ):
        """
        Compute attention weights for either intra or inter-chain attention.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [B, L, D]
            attention_mask_in_length (torch.Tensor): Tensor specifying valid sequence lengths
            which_attn (str): Either "intra" or "inter" to specify attention type

        Returns:
            tuple:
                - attn_output (torch.Tensor): Attention output of shape [B, L, D]
                - attn_weights (torch.Tensor): Attention weights of shape [B, H, L, L]
        """
        bsz, seq_len, _ = hidden_states.shape

        # Project input to query, key, and value
        # d is the dimension of each head = D / H
        # Shape [B, L, D] -> [B, L, 3 * D] -> [B, L, 3, H, d]
        if which_attn == "intra":
            qkv = self.qkv_proj_intra(hidden_states)
        elif which_attn == "inter":
            qkv = self.qkv_proj_inter(hidden_states)
        else:
            raise ValueError("which_attn should be 'intra' or 'inter'")
        qkv = rearrange(
            qkv,
            "b s (three h d) -> b s three h d",
            three=3,
            h=self.num_attention_heads,
        )
        # Split qkv into separate tensors
        # Each of shape (B, L, H, d)
        q, k, v = qkv.unbind(dim=2)

        q = q * self.head_dim**-0.5  # rescale
        # Apply rotary positional encoding
        q_rotary = self.rotary_emb(q, attention_mask_in_length)
        k_rotary = self.rotary_emb(k, attention_mask_in_length)
        # Rearrange tensors for attention computation
        # Move heads to batch dimension
        # [B, L, H, d] -> [B*H, L, d]
        q_rotary = rearrange(q_rotary, "b s h d -> (b h) s d")
        k_rotary = rearrange(k_rotary, "b s h d -> (b h) s d")
        v = rearrange(v, "b s h d -> (b h) s d")
        # Compute attention scores explicitly
        # [B*H, L, d] @ [B*H, d, L] -> [B*H, L, L]
        attn_weights = torch.bmm(q_rotary, k_rotary.transpose(-1, -2))
        # Reshape attention weights back to [B, H, L, L]
        attn_weights = rearrange(
            attn_weights,
            "(b h) t s -> b h t s",
            h=self.num_attention_heads,
        )

        if self.causal:
            # Apply causal mask explicitly
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # Create and apply padding mask explicitly
        padding_mask = _create_padding_mask(attention_mask_in_length)
        # Add head dims and dimension for the other sequence
        # Shape (B, 1, 1, L)
        attn_weights = attn_weights.masked_fill(
            ~padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        # Create mask for intra or inter chain, shape (B, L, L)
        chain_mask = _create_chain_mask(attention_mask_in_length, which_attn=which_attn)
        attn_weights = attn_weights.masked_fill(~chain_mask.unsqueeze(1), float("-inf"))

        # Apply softmax, shape (B, H, L, L)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # For inter attention weights, there will be rows that are all zeros
        # softmax returns np.nan, we fill them with zeros
        attn_weights = attn_weights.masked_fill(attn_weights.isnan(), 0.0)

        # Apply dropout to attention weights
        attn_weights = self.attention_dropout(attn_weights)

        # Reshape attention weights for value multiplication
        # (B, H, L, L) -> (B*H, L, L)
        attn_probs = rearrange(attn_weights, "b h t s -> (b h) t s")

        # Compute attention output
        # (B*H, L, L) @ (B*H, L, D) -> (B*H, L, D)
        attn_output = torch.bmm(attn_probs, v)

        # Reshape output back to original dimensions
        # (B*H, L, d) -> (B, L, H*d)
        attn_output = rearrange(
            attn_output,
            "(b h) s d -> b s (h d)",
            h=self.num_attention_heads,
        )

        return attn_output, attn_weights

    def forward(self, hidden_states, attention_mask_in_length, **kwargs):
        """
        Forward pass computing both intra and inter-chain attention.
        Intra and inter-chain attention are computed respectively by first
        computing the full attention weights (before softmax) using the intra / inter parameters
        then mask out the irrelevant portions (just like with paddings)
        before softmax

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [B, L, D]
            attention_mask_in_length (torch.Tensor): Tensor specifying valid sequence lengths
            **kwargs: Additional arguments including return_decoder_self_attn_weights

        Returns:
            tuple:
                - attn_output (torch.Tensor): Combined attention output of shape [B, L, D]
                - attn_weights (Optional[tuple]): If requested, tuple of intra and inter attention weights
        """
        attn_output_intra, attn_weights_intra = self.compute_attention_weights(
            hidden_states,
            attention_mask_in_length,
            which_attn="intra",
        )
        attn_output_inter, attn_weights_inter = self.compute_attention_weights(
            hidden_states,
            attention_mask_in_length,
            which_attn="inter",
        )
        # The attention output is the sum of the intra output
        # (computed using only the intra parameters on the intra positions)
        # and the inter output
        attn_output = attn_output_intra + attn_output_inter
        # Final projection
        attn_output = self.out_proj(attn_output)

        # Attention weights for intra and inter should have the appropriate positions
        # zeroed out, we could sum them to get the overall attention weights
        # But for debugging purpose, we return them separately so we can verify
        # the positions are appropriately zeroed out
        if kwargs.get("return_decoder_self_attn_weights", False):
            attn_weights = (attn_weights_intra, attn_weights_inter)
        else:
            attn_weights = None

        return attn_output, attn_weights


class MultiSequenceCrossAttention(nn.Module):
    """
    This module performs cross-attention between a query sequence and a set of key-value sequences,
    applying both rotary positional encoding within sequences and distance-based sinusoidal encoding.
    It uses Flash Attention for efficient computation.

    Args:
        hidden_size (int): Size of the input and output features.
        num_attention_heads (int): Number of attention heads.
        max_position_embeddings (int): Maximum number of position embeddings for rotary encoding.
        max_distance (int): Maximum distance for sinusoidal encoding.
        dropout_p (float, optional): Dropout probability for attention weights. Defaults to 0.0.

    """

    def __init__(self, hidden_size, num_attention_heads, dropout_p=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        assert (
            self.head_dim * num_attention_heads == hidden_size
        ), "hidden_size must be divisible by num_attention_heads"

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.rotary_emb = MultiSequenceRotaryPositionalEncoding(self.head_dim)

        self.dropout_p = dropout_p

    def forward(
        self,
        q_states,
        kv_states,
        attention_mask_in_length_q,
        attention_mask_in_length_kv,
    ):
        # confusingly, y is the query sequence and x is the key-value sequence
        bsz, q_len, _ = q_states.shape
        _, kv_len, _ = kv_states.shape

        # Project inputs
        q = self.q_proj(q_states)
        kv = self.kv_proj(kv_states)

        # rescale q
        q *= self.head_dim**-0.5

        # reshape
        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_attention_heads)
        kv = rearrange(kv, "b s (two h d) -> b s two h d", two=2, h=self.num_attention_heads)

        # Apply rotary positional encoding to q and k
        q_rotary = self.rotary_emb(q, attention_mask_in_length_q)
        k_rotary = self.rotary_emb(kv[:, :, 0], attention_mask_in_length_kv)

        # # Apply sinusoidal distance encoding to k
        # distance_positions = _expand_distances_to_seqlen(
        #     distances_in_length, attention_mask_in_length_kv
        # )
        # k_distance_embeddings = self.distance_emb(distance_positions)
        # k_distance_embeddings = k_distance_embeddings.view(
        #     k_distance_embeddings.shape[0], k_distance_embeddings.shape[1], 1, -1
        # )

        # k_rotary += k_distance_embeddings

        # Recombine kv
        kv_encoded = torch.stack([k_rotary, kv[:, :, 1]], dim=2)

        # Create attention masks
        q_mask = _create_padding_mask(attention_mask_in_length_q)
        kv_mask = _create_padding_mask(attention_mask_in_length_kv)

        # Unpad inputs
        q, idx_q, cu_seqlens_q, max_s_q, _ = unpad_input(q_rotary, q_mask)
        kv, _, cu_seqlens_k, max_s_k, _ = unpad_input(kv_encoded, kv_mask)

        out = flash_attn_varlen_kvpacked_func(
            q=q,
            kv=kv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_s_q,
            max_seqlen_k=max_s_k,
            dropout_p=self.dropout_p,
            softmax_scale=1.0,  # q has already been rescaled
            causal=False,
        )
        out = pad_input(out, idx_q, bsz, q_len)
        out = rearrange(out, "... h d -> ... (h d)")  # concat heads

        return self.out_proj(out)


#########################################################
## Encoder and Decoder Modules for Multiple Sequences. ##
#########################################################


class MultiSequenceEncoderBlock(nn.Module):
    def __init__(
        self,
        attention_heads: int,
        embed_dim: int,
        ffn_embed_dim: int,
        dropout_p: float,
        layer_idx: Optional[int] = None,
        self_attn_type: str = "full",  # Default to full
        **kwargs,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.layer_idx = layer_idx
        self.dropout_p = dropout_p
        self.ffn_embed_dim = ffn_embed_dim

        if self_attn_type == "full":
            self.self_attn = FullMultiSequenceSelfAttention(
                hidden_size=embed_dim,
                num_attention_heads=attention_heads,
                dropout_p=dropout_p,
                causal=False,
                layer_idx=layer_idx,
            )
        elif self_attn_type == "intra_only":
            self.self_attn = IntraOnlyMultiSequenceSelfAttention(
                hidden_size=embed_dim,
                num_attention_heads=attention_heads,
                dropout_p=dropout_p,
                causal=False,
                layer_idx=layer_idx,
            )
        elif self_attn_type == "intra_inter":
            self.self_attn = DecoupledIntraInterMultiSequenceSelfAttention(
                hidden_size=embed_dim,
                num_attention_heads=attention_heads,
                dropout_p=dropout_p,
                causal=False,
                layer_idx=layer_idx,
                **kwargs,
            )
        else:
            raise ValueError(
                f"self_attn_type should be 'full', 'intra_only' or 'intra_inter', not {self_attn_type}"
            )

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x, attn_mask):
        """
        Self attention forward pass.
        Applies self attention to the input, the attention is over the full input sequence
        so x1 can attend to y1, and vice versa
        attn_mask is used to define the positions for the rotary positional encoding,
        so that it resets at each sequence boundary.

        e.g. if attn_mask =
        [[4, 6, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 5, 0, 0, 0, 0, 0, 0, 0, 0]]

        This means that for the first element in the batch, we have two sequences of length 4 and 6. For the second
        element, we have two sequences of length 3 and 5. So the corresponding rotary positional encoding for the input
        is:

        [[0, 1, 2, 3, 0, 1, 2, 3, 4, 5],
            [0, 1, 2, 0, 1, 2, 3, 4, 0, 0]]
        NB: the final 0s don't matter because they go beyond the sequence length and will be ignored by the unpad.
        """
        # atten block
        residual = x
        x = self.self_attn_layer_norm(x)
        if isinstance(self.self_attn, DecoupledIntraInterMultiSequenceSelfAttention):
            # DecoupledIntraInterMultiSequenceSelfAttention returns a tuple (output, attention_weights)
            x, _ = self.self_attn(x, attn_mask)
        else:
            x = self.self_attn(x, attn_mask)
        x = x + residual
        # MLP block
        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        return x


class MultiSequenceDecoderBlock(nn.Module):
    def __init__(
        self,
        attention_heads: int,
        embed_dim: int,
        ffn_embed_dim: int,
        dropout_p: float,
        layer_idx: Optional[int] = None,
        self_attn_type: str = "full",
        **kwargs,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.layer_idx = layer_idx
        self.dropout_p = dropout_p
        self.ffn_embed_dim = ffn_embed_dim

        if self_attn_type == "full":
            self.self_attn = FullMultiSequenceSelfAttention(
                hidden_size=embed_dim,
                num_attention_heads=attention_heads,
                dropout_p=dropout_p,
                causal=True,
                layer_idx=layer_idx,
            )
        elif self_attn_type == "intra_only":
            self.self_attn = IntraOnlyMultiSequenceSelfAttention(
                hidden_size=embed_dim,
                num_attention_heads=attention_heads,
                dropout_p=dropout_p,
                causal=True,
                layer_idx=layer_idx,
            )
        elif self_attn_type == "intra_inter":
            self.self_attn = DecoupledIntraInterMultiSequenceSelfAttention(
                hidden_size=embed_dim,
                num_attention_heads=attention_heads,
                dropout_p=dropout_p,
                causal=True,
                layer_idx=layer_idx,
                **kwargs,
            )
        else:
            raise ValueError(
                f"self_attn_type should be 'full', 'intra_only' or 'intra_inter', not {self_attn_type}"
            )

        self.cross_attn = MultiSequenceCrossAttention(
            hidden_size=embed_dim,
            num_attention_heads=attention_heads,
            dropout_p=dropout_p,
        )

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x, enc_out, dec_attn_mask, enc_attn_mask, **kwargs):
        """
        Forward pass using the x (the decoder hiddens, abbreviated as "dec_out")
        and enc_out (the encoder output hiddens)
        as well as their attention length masks, and the distance mask of each sequence
        dec_out provides the queries, enc_out provides the keys and values.

        Unpadding and repadding occur in the various modules, and the rotary positional encoding is applied.
        Note that for both dec_out and enc_out, rotary restarts at each sequence boundary.
        The attn masks define the lengths of sequence in each batch for dec_out and enc_out.
        This helps define padding boundaries, as well as the rotary positional encoding positions.

        The distance mask is similar to the attn_masks, but defines for each sequence
        (x1 and x2 are separated by tx, y1 and y2 are separated by ty)
        Hence, when (x2, y2) goes into the cross attention module as q,
        the distance mask defines a positional encoding offset to k (which comes from x1, y1)
        """
        # Self attention of decoder input
        residual = x
        x = self.self_attn_layer_norm(x)
        if isinstance(self.self_attn, DecoupledIntraInterMultiSequenceSelfAttention):
            # If uses vanilla self attention, it is because we need the
            # attention weights
            x, attn_weights = self.self_attn(x, dec_attn_mask, **kwargs)
        else:
            x = self.self_attn(x, dec_attn_mask)
            attn_weights = None
        x = x + residual

        # Cross attention between decoder input and encoder output
        # query: decoder input
        # key, value: encoder output
        residual = x
        x = self.cross_attn_layer_norm(x)
        x = self.cross_attn(x, enc_out, dec_attn_mask, enc_attn_mask)
        x = x + residual

        # Layer norm + feed forward
        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        return x, attn_weights
