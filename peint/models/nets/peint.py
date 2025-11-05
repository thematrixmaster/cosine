from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.model.esm2 import ESM2
from esm.modules import RobertaLMHead
from torch import Tensor
from tqdm import tqdm

from evo.tokenization import CodonVocab, Vocab
from peint.models.frameworks.peint import sampling_function
from peint.models.nets.esm2 import ESM2Flash
from peint.models.nets.transformer import (
    FlashMHADecoderBlock,
    FlashMHAEncoderBlock,
    GeometricTimeEmbedder,
    KV_CachedFlashMHADecoderBlock,
)
from peint.models.nets.utils import (
    _create_sequence_mask,
    _expand_chain_attr_to_seqlen,
)


class PretrainedEncoder(ABC, nn.Module):
    """Abstract base class for pretrained encoders."""

    def __init__(self, vocab: Vocab, embed_x_per_chain: bool = False):
        super().__init__()
        self.vocab = vocab
        self.embed_x_per_chain = embed_x_per_chain

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        pass

    @abstractmethod
    def get_in_embedding(self) -> nn.Module:
        pass

    @abstractmethod
    def get_out_lm_head(self) -> nn.Module:
        pass

    @abstractmethod
    def forward(self, x: Tensor, x_sizes: Tensor) -> Tensor:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args, **kwargs) -> "PretrainedEncoder":
        pass


class ESMEncoder(PretrainedEncoder):
    """ESM encoder wrapper."""

    def __init__(
        self,
        vocab: Vocab,
        esm_model: ESM2,
        finetune: bool = False,
        embed_x_per_chain: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(vocab=vocab, embed_x_per_chain=embed_x_per_chain, *args, **kwargs)
        self.esm = esm_model
        self.finetune = finetune
        self.esm.eval() if not self.finetune else self.esm.train()
        self.esm.requires_grad_(self.finetune)  # freeze the ESM model

    @property
    def embed_dim(self) -> int:
        return self.esm.embed_dim

    def get_in_embedding(self) -> nn.Module:
        # embedding layer (optionally from the encoder)
        in_embedding = nn.Embedding(len(self.vocab), self.embed_dim)
        in_embedding.load_state_dict(self.esm.embed_tokens.state_dict())
        in_embedding.requires_grad_(self.finetune)
        return in_embedding

    def get_out_lm_head(self) -> nn.Module:
        # output layer
        lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=len(self.vocab),
            weight=self.esm.embed_tokens.weight,
        )
        lm_head.load_state_dict(self.esm.lm_head.state_dict())
        lm_head.requires_grad_(self.finetune)
        return lm_head

    def forward(self, x: Tensor, x_sizes: Tensor) -> Tensor:
        if self.embed_x_per_chain:
            # Multi-sequence processing: compute ESM per chain
            return self.forward_per_chain(x, x_sizes)
        else:
            # Single-sequence processing: process concatenated sequence
            res = self.esm(x, repr_layers=[self.esm.num_layers], need_head_weights=False)
            esm_s = res["representations"][self.esm.num_layers]
            return esm_s

    def forward_per_chain(self, x: Tensor, x_sizes: Tensor):
        """
        Compute ESM embedding separately for each sequence in the batch (multi-sequence style)
        """
        bsz, seq_len = x.shape
        _num_sequences = (x_sizes > 0).sum(dim=1)
        max_num_sequences = _num_sequences.max().int().item()

        combined_embedding = torch.zeros(
            (bsz, seq_len, (self.esm.embed_dim)), dtype=torch.float, device=x.device
        )

        for seq_idx in range(max_num_sequences):
            # Shape (B, L), True for the residues of `seq_idx`th sequence of each batch element
            seq_mask = _create_sequence_mask(x_sizes, sequence_idx=seq_idx)
            # Create a copy of the input to avoid modifying the original
            x_seq = x.clone()
            # Set padding for all other positions
            x_seq.masked_fill_(~seq_mask, self.vocab.pad_idx)

            if (x_seq == self.vocab.pad_idx).all():
                # For a normal use case we don't need this check
                # But for ablation, we might mask out one of the sequence
                # so at least for some seq_idx, x_seq would have only paddings
                # in that case ESM will give nans
                continue

            # Compute embeddings for the current sequence
            with torch.no_grad():
                output = self.esm(x_seq, repr_layers=[self.esm.num_layers])
                embedding = output["representations"][self.esm.num_layers]

            # Replace nans with zeros (can happen for sequences with less chains)
            embedding = torch.nan_to_num(embedding, nan=0.0)

            # Add the current sequence embedding to the combined embedding
            combined_embedding += embedding * seq_mask.unsqueeze(-1)

        return combined_embedding

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "ESMEncoder":
        """Create an ESMEncoder from a pretrained ESM model."""
        import esm

        _esm_model, esm_vocab = esm.pretrained.esm2_t30_150M_UR50D()
        esm_model = ESM2Flash(
            alphabet="ESM-1b",
            num_layers=_esm_model.num_layers,
            embed_dim=_esm_model.embed_dim,
            attention_heads=_esm_model.attention_heads,
            token_dropout=_esm_model.token_dropout,
            dropout_p=kwargs.get("dropout_p", 0.0),
        )

        ckpt_path = kwargs.pop("ckpt_path", None)
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            esm_model.load_state_dict(state_dict=state_dict, strict=False)
        else:
            esm_model.load_state_dict(_esm_model.state_dict(), strict=False)
            del _esm_model

        evo_vocab = Vocab.from_esm_alphabet(esm_vocab)
        return cls(esm_model=esm_model, vocab=evo_vocab, *args, **kwargs)


class HybridEmbedding(nn.Module):
    def __init__(
        self,
        aa_vocab_size: int,
        esm_embed_dim: int,
        esm_embed_tokens_state_dict: Dict[str, Tensor],
        codon_embedding: nn.Embedding,
        codon_to_aa_mapping: Tensor,
        finetune=False,
    ):
        super().__init__()
        self.aa_embedding = nn.Embedding(aa_vocab_size, esm_embed_dim)
        self.aa_embedding.load_state_dict(esm_embed_tokens_state_dict)
        self.aa_embedding.requires_grad_(finetune)
        self.codon_embedding = codon_embedding
        self.codon_to_aa_mapping = codon_to_aa_mapping

    def forward(self, x: Tensor) -> Tensor:
        aa_tok_idx = self.codon_to_aa_mapping[x]  # (B, L)
        aa_emb = self.aa_embedding(aa_tok_idx)  # (B, L, esm_embed_dim)
        codon_emb = self.codon_embedding(x)  # (B, L, codon_embed_dim)
        return torch.cat([codon_emb, aa_emb], dim=-1)


class ESMCodonEncoder(ESMEncoder):
    """Encoder that concatenates learned codon embeddings with frozen ESM amino acid embeddings."""

    vocab: CodonVocab
    codon_to_aa_mapping: Tensor

    def __init__(
        self,
        vocab: Vocab,
        codon_vocab: CodonVocab,
        codon_embed_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(vocab=codon_vocab, *args, **kwargs)
        self.aa_vocab = vocab
        self.codon_embed_dim = codon_embed_dim
        codon_to_aa_mapping = codon_vocab.translation_tensor_map(aa_vocab=vocab).cuda()
        self.register_buffer("codon_to_aa_mapping", codon_to_aa_mapping)
        self.codon_embedding = nn.Embedding(len(self.vocab), self.codon_embed_dim)
        nn.init.normal_(self.codon_embedding.weight, mean=0.0, std=0.02)
        self.codon_embedding.requires_grad_(True)

    @property
    def embed_dim(self) -> int:
        return self.esm.embed_dim + self.codon_embed_dim

    def get_in_embedding(self) -> nn.Module:
        """Hybrid embedding layer that concatenates codon and amino acid embeddings."""
        return HybridEmbedding(
            aa_vocab_size=len(self.aa_vocab),
            esm_embed_dim=self.esm.embed_dim,
            esm_embed_tokens_state_dict=self.esm.embed_tokens.state_dict(),
            codon_embedding=self.codon_embedding,
            codon_to_aa_mapping=self.codon_to_aa_mapping,
            finetune=self.finetune,
        )

    def get_out_lm_head(self) -> nn.Module:
        # output layer trained from scratch
        # create learnable weight for the lm head
        weight = torch.randn(len(self.vocab), self.embed_dim) * 0.02
        lm_head_weight = nn.Parameter(weight, requires_grad=True)
        lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=len(self.vocab),
            weight=lm_head_weight,
        )
        lm_head.requires_grad_(True)  # always finetune the output layer
        return lm_head

    def forward(self, x: Tensor, x_sizes: Tensor) -> Tensor:
        aa_tok_idx = self.codon_to_aa_mapping[x]  # (B, L)
        if self.embed_x_per_chain:
            # Multi-sequence processing: compute ESM per chain
            aa_hx = self.forward_per_chain(aa_tok_idx, x_sizes)
        else:
            # Single-sequence processing: process concatenated sequence
            res = self.esm(aa_tok_idx, repr_layers=[self.esm.num_layers], need_head_weights=False)
            aa_hx = res["representations"][self.esm.num_layers]

        codon_emb = self.codon_embedding(x)  # (B, L, codon_embed_dim)
        joint_hx = torch.cat([codon_emb, aa_hx], dim=-1)  # (B, L, codon+esm_embed_dim)
        return joint_hx


class PEINT(nn.Module):
    """Unified encoder-decoder transformer for protein evolution modeling.

    This model supports both single-sequence (original PEINT) and multi-sequence (PIPET)
    protein evolution modeling through configurable attention mechanisms and data processing.

    The ESM model encodes sequences, and the final hidden representation goes into an
    encoder/decoder stack. The encoder/decoder stack works as a standard transformer
    but with configurable multi-sequence attention support.

    Key Features:
    - Support for both single and multi-sequence inputs
    - Flexible ESM processing (per-chain or concatenated)
    - Unified size-based mask interface

    Args:
        causal_decoder: If True, use causal attention in the decoder (for autoregressive).
                        If False, use bidirectional attention (for masked prediction / discrete diffusion).
    """

    def __init__(
        self,
        enc_model: PretrainedEncoder,
        evo_vocab: Vocab,
        embed_dim: int,
        num_heads: int,
        num_chains: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        max_len=1022,
        dropout_p: float = 0.0,
        use_chain_embedding: bool = True,
        use_attention_bias: bool = True,
        causal_decoder: bool = True,
    ):
        super().__init__()

        self.vocab = evo_vocab
        self.max_len = max_len
        self.enc_model = enc_model
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.num_chains = num_chains
        self.use_bias = use_attention_bias
        self.causal_decoder = causal_decoder
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.use_chain_embedding = use_chain_embedding

        assert self.embed_dim == enc_model.embed_dim
        assert num_encoder_layers >= num_decoder_layers

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.vocab.pad_idx)

        # Always use multi-sequence blocks (they handle single sequences as special case)
        self.enc_layers = nn.ModuleList(
            [
                FlashMHAEncoderBlock(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=4 * self.embed_dim,
                    attention_heads=self.num_heads,
                    add_bias_kv=False,
                    dropout_p=self.dropout_p,
                    use_bias=self.use_bias,
                    layer_idx=l,
                )
                for l in range(num_encoder_layers)
            ]
        )
        self.dec_layers = nn.ModuleList(
            [
                FlashMHADecoderBlock(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=4 * self.embed_dim,
                    attention_heads=self.num_heads,
                    add_bias_kv=False,
                    dropout_p=self.dropout_p,
                    use_bias=self.use_bias,
                    layer_idx=l,
                    causal=causal_decoder,
                )
                for l in range(num_decoder_layers)
            ]
        )

        # time embedding
        self.time_embedding = GeometricTimeEmbedder(
            frequency_embedding_size=embed_dim, start=1e-5, stop=0.25
        )

        # chain embedding (0 is padding idx)
        self.chain_embedding = (
            nn.Embedding(num_chains + 1, embed_dim, padding_idx=0) if use_chain_embedding else None
        )

        # get input and output layers from the encoder
        self.in_embedding = self.enc_model.get_in_embedding()
        self.out_lm_head = self.enc_model.get_out_lm_head()

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        x_sizes: Tensor,
        y_sizes: Tensor,
        chain_ids: Tensor = None,
        use_cache: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Unified forward pass using standardized size-based interface.

        Args:
            x: Encoder input sequences (batch_size, seq_len)
            y: Decoder input sequences (batch_size, seq_len)
            t: Time values (batch_size, num_chains)
            x_sizes: Sequence sizes for encoder (batch_size, seq_len)
            y_sizes: Sequence sizes for decoder (batch_size, seq_len)
            chain_ids: Chain IDs for each sequence in the batch (batch_size, num_chains)
            use_cache: Whether to use cached key/value states in the decoder (for generation)

        Returns:
            Dict with 'enc_logits' and 'dec_logits' keys
        """
        # Embed decoder input with time
        h_y = self.in_embedding(y)

        # Position-specific time embedding using distances
        distances_expanded = _expand_chain_attr_to_seqlen(
            chain_attr=t, sizes=y_sizes, pad_value=0.0
        )
        ht = self.time_embedding(distances_expanded)
        h_y = h_y + ht

        # Create attention masks from sizes (1 for data, 0 for pad)
        x_attn_mask = (x != self.vocab.pad_idx).long()
        y_attn_mask = (y != self.vocab.pad_idx).long()

        # Add chain embeddings to h_y
        if self.chain_embedding is not None and chain_ids is not None:
            chain_attr_expanded_y = _expand_chain_attr_to_seqlen(
                chain_attr=chain_ids, sizes=y_sizes, pad_value=0
            )
            h_y = h_y + self.chain_embedding(chain_attr_expanded_y.long())

        if use_cache:
            assert isinstance(self.dec_layers[0], KV_CachedFlashMHADecoderBlock)
            assert y.shape[1] == 1, "When using cache, decoder input y should be of length 1"
            assert y_sizes.shape[1] == 1, "When using cache, y_sizes should have shape (B, 1)"
            assert torch.all(y_sizes == 1).item(), "When using cache, y_sizes should be all ones"

            # Used cached encoder and decoder key/values for generation
            for i, dec_layer in enumerate(self.dec_layers):
                h_y = dec_layer(x=h_y, x_attn_mask=y_attn_mask, y=None, y_attn_mask=x_attn_mask)

            y_logits = self.out_lm_head(h_y)
            return dict(enc_logits=None, dec_logits=y_logits)

        # Get encoder hidden representations
        h_x = self.enc_model.forward(x=x, x_sizes=x_sizes)

        # Add chain embeddings to h_x
        if self.chain_embedding is not None and chain_ids is not None:
            chain_attr_expanded_x = _expand_chain_attr_to_seqlen(
                chain_attr=chain_ids, sizes=x_sizes, pad_value=0
            )
            h_x = h_x + self.chain_embedding(chain_attr_expanded_x.long())

        # Encoder-decoder forward pass
        for i, enc_layer in enumerate(self.enc_layers):
            # Encoder layer
            h_x = enc_layer(x=h_x, x_attn_mask=x_attn_mask)

            # Decoder layers (with encoder-decoder alignment)
            if self.num_decoder_layers - self.num_encoder_layers + i >= 0:
                idx = self.num_decoder_layers - self.num_encoder_layers + i
                dec_layer = self.dec_layers[idx]
                # Decoder layer (args: x means main input, y is cross attended input)
                h_y = dec_layer(x=h_y, x_attn_mask=y_attn_mask, y=h_x, y_attn_mask=x_attn_mask)

        # Generate logits
        x_logits = self.out_lm_head(h_x)
        y_logits = self.out_lm_head(h_y)

        if torch.any(torch.isnan(x_logits)) or torch.any(torch.isnan(y_logits)):
            raise ValueError("NaN detected in logits")

        # Return dictionary format
        return dict(enc_logits=x_logits, dec_logits=y_logits)

    @torch.no_grad()
    def dec_perplexity(
        self,
        ts: Tensor,
        x_src: Tensor,
        y_src: Tensor,
        y_tgt: Tensor,
        x_sizes: Tensor,
        y_sizes: Tensor,
        chain_ids: Tensor,
    ) -> Tensor:
        """
        Computes the decoder perplexity on the target sequences given the source sequence and context.
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.forward(
                x_src, y_src, ts, x_sizes=x_sizes, y_sizes=y_sizes, chain_ids=chain_ids
            )

        y_logits = outputs["dec_logits"]
        y_logits = y_logits - torch.logsumexp(y_logits, dim=-1, keepdim=True)
        y_logits = y_logits.transpose(-1, -2)
        nll = F.cross_entropy(y_logits, y_tgt, ignore_index=self.vocab.pad_idx, reduction="none")

        y_tgt_mask = y_tgt != self.vocab.pad_idx
        nll_mean = (nll * y_tgt_mask).sum(dim=1) / y_tgt_mask.sum(dim=1)
        ppl = torch.exp(nll_mean)
        return ppl


class PEINTGenerator(PEINT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dec_layers = nn.ModuleList(
            [
                KV_CachedFlashMHADecoderBlock(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=4 * self.embed_dim,
                    attention_heads=self.num_heads,
                    add_bias_kv=False,
                    causal=self.causal_decoder,
                    dropout_p=self.dropout_p,
                    use_bias=self.use_bias,
                    max_encoder_seq_len=self.max_len,
                    max_decoder_seq_len=self.max_len,
                    layer_idx=l,
                )
                for l in range(self.num_decoder_layers)
            ]
        )

    @classmethod
    def from_peint(cls, peint_model: PEINT) -> "PEINTGenerator":
        kwargs = {
            "enc_model": peint_model.enc_model,
            "evo_vocab": peint_model.vocab,
            "embed_dim": peint_model.embed_dim,
            "num_heads": peint_model.num_heads,
            "num_chains": peint_model.num_chains,
            "num_encoder_layers": peint_model.num_encoder_layers,
            "num_decoder_layers": peint_model.num_decoder_layers,
            "max_len": peint_model.max_len,
            "dropout_p": peint_model.dropout_p,
            "use_chain_embedding": peint_model.use_chain_embedding,
            "use_attention_bias": peint_model.use_bias,
            "causal_decoder": peint_model.causal_decoder,
        }
        generator = cls(**kwargs)
        generator.load_state_dict(peint_model.state_dict())
        return generator

    def _reset_kv_cache(self):
        for dec_layer in self.dec_layers:
            dec_layer.cross_attn.kv_cache = None  # This will reset the cache
            dec_layer.self_attn.reset_kv_cache()

    @torch.no_grad()
    def dec_generate(
        self,
        ts: torch.Tensor,
        xs: torch.Tensor,
        x_sizes: torch.Tensor,
        y_sizes: torch.Tensor = None,  # use for fixed length decoding |x| = |y|
        chain_ids: torch.Tensor = None,
        max_decode_steps: int = None,
        no_special_toks: bool = True,
        temperature: float = 1.0,
        p: float = 1.0,
    ) -> Tensor:
        """Generate sequences using top p nucleus sampling"""
        batch_size = xs.size(0)
        device = xs.device

        # Limit max_decode_steps to account for both start and stop tokens
        max_context_len = self.max_len - (self.vocab.prepend_bos + self.vocab.append_eos)
        max_decode_steps = min(max_decode_steps or np.inf, max_context_len)

        for dec_layer in self.dec_layers:
            dec_layer.self_attn.init_kv_cache(batch_size, max_decode_steps)

        # Initialize output with start token
        y_decoded = torch.tensor([self.vocab.bos_idx]).unsqueeze(0).repeat(batch_size, 1).to(device)
        y_sizes_decoded = torch.zeros((batch_size, self.num_chains), dtype=torch.long).to(device)
        y_sizes_single = torch.ones((batch_size, 1), dtype=torch.long).to(device)
        y_chain_idx = torch.zeros((batch_size,), dtype=torch.long).to(device)
        y_sizes_decoded[:, y_chain_idx] = 1  # account for BOS token
        batch_indices = torch.arange(batch_size, device=device)
        eos_reached = torch.zeros(batch_size, dtype=torch.bool).to(device)

        # First forward pass to fill KV cache and get initial logits
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.forward(
                x=xs,
                y=y_decoded,
                t=ts[batch_indices, y_chain_idx],
                x_sizes=x_sizes,
                y_sizes=y_sizes_single,
                chain_ids=chain_ids,
                use_cache=False,
            )["dec_logits"]

        for _ in tqdm(range(max_decode_steps - 1)):
            logits = logits[:, -1, :] / temperature

            if no_special_toks:
                special_tok_idxs = [
                    self.vocab.bos_idx,
                    self.vocab.pad_idx,
                    self.vocab.mask_idx,
                    self.vocab.unk_idx,
                ]
                if y_sizes is not None:
                    special_tok_idxs.append(self.vocab.eos_idx)
                zero_idx = torch.tensor(special_tok_idxs, device=logits.device)
                logits[..., zero_idx] = -np.inf

            # sample next token
            next_token = sampling_function(logits, p=p)

            # append next_token to y_decoded
            next_token = torch.where(eos_reached.unsqueeze(-1), self.vocab.pad_idx, next_token)
            y_decoded = torch.cat([y_decoded, next_token], dim=1)

            # # gather tokens at eos_reached positions
            # eos_next_token = next_token[eos_reached]
            # assert torch.all(eos_next_token == self.vocab.pad_idx), "Tokens generated after EOS should be PAD"

            # update y_sizes_decoded
            y_sizes_decoded[batch_indices, y_chain_idx] += 1

            if y_sizes is None:
                # Move to next chain if separator token is reached
                finished_chains = next_token.squeeze(-1) == self.vocab.tokens_to_idx["."]
                y_chain_idx += finished_chains.long()
                eos_reached |= next_token.squeeze(-1) == self.vocab.eos_idx
                y_chain_idx = torch.clamp(y_chain_idx, max=self.num_chains - 1)
            else:
                # Move to next chain if chain length is reached according to y_sizes
                finished_chains = (
                    y_sizes_decoded[batch_indices, y_chain_idx]
                    == y_sizes[batch_indices, y_chain_idx]
                )
                y_chain_idx += finished_chains.long()
                eos_reached |= y_chain_idx >= self.num_chains
                y_chain_idx = torch.clamp(y_chain_idx, max=self.num_chains - 1)

            # stop if all sequences have reached eos
            if eos_reached.all():
                break

            # forward pass with updated token
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.forward(
                    x=xs,
                    y=next_token,
                    t=ts[batch_indices, y_chain_idx],
                    x_sizes=x_sizes,
                    y_sizes=y_sizes_single,
                    chain_ids=(
                        chain_ids[batch_indices, y_chain_idx] if chain_ids is not None else None
                    ),
                    use_cache=True,
                )["dec_logits"]

        self._reset_kv_cache()
        return y_decoded


def decode_sequence_from_toks(toks: np.ndarray, vocab: Vocab) -> str:
    tokens = []
    for tok in toks:
        if tok == vocab.bos_idx:
            continue
        if tok == vocab.eos_idx or tok == vocab.pad_idx:
            break
        tokens.append(vocab.token(tok))
    return "".join(tokens)


if __name__ == "__main__":
    from pathlib import Path

    from peint.data.datasets.codon import dataloader_from_transitions
    from peint.models.modules.peint_module import PEINTModule

    # Load trained joint model from checkpoint
    ckpt_dir = Path("/mfs1/u/stephenzlu/projects/peint/checkpoints")
    ckpt_path = ckpt_dir / "last.ckpt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # create a fresh peint model with the same hyperparameters as the training run
    vocab = CodonVocab.from_codons()
    esm_encoder = ESMCodonEncoder.from_pretrained(
        codon_vocab=vocab,
        codon_embed_dim=384,
    )
    net = PEINT(
        enc_model=esm_encoder,
        evo_vocab=vocab,
        embed_dim=1024,
        num_heads=16,
        num_chains=2,
        num_encoder_layers=3,
        num_decoder_layers=3,
        max_len=1022,
        dropout_p=0.0,
        use_chain_embedding=True,
        use_attention_bias=True,
        causal_decoder=True,
    )
    module = PEINTModule.load_from_checkpoint(ckpt_path, net=net, map_location=device, strict=False)
    module.net.in_embedding.codon_embedding.weight = module.net.enc_model.codon_embedding.weight
    module = module.eval()

    # load the test dataset
    datapath = Path("data/d4.txt")
    dataloader = dataloader_from_transitions(
        transitions=None, vocab=vocab, datapath=datapath, batch_size=32
    )

    # load the generator from the peint model
    generator = PEINTGenerator.from_peint(module.net).to(device)

    # for each transition in the test set, sample a child sequence using the model and compute its hamming distance + codon usage compared to the true child sequence
    real_child_nt_sequences = []
    sim_child_nt_sequences = []
    n_batches = 100

    for batch in tqdm(dataloader, desc="Inference"):
        batch = [b.to(device) for b in batch]
        [x_src, x_tgt, y_src, y_tgt, ts, chain_ids, x_sizes, y_sizes] = batch

        # get heavy chain lengths
        hc_lens = y_sizes[:, 0] * 3

        # decode the true child sequence using the vocab
        true_child_seqs = [
            decode_sequence_from_toks(y_tgt[i].cpu().numpy(), vocab) for i in range(y_tgt.size(0))
        ]
        true_hv_seqs, true_lt_seqs = zip(
            *[(seq[:hl], seq[hl:]) for seq, hl in zip(true_child_seqs, hc_lens)]
        )
        assert all(
            [len(tc) == (ys.sum().item() - 1) * 3 for tc, ys in zip(true_child_seqs, y_sizes)]
        )

        # sample a child sequence using the model
        y_decoded = generator.dec_generate(
            ts=ts, xs=x_src, x_sizes=x_sizes, y_sizes=y_sizes, chain_ids=chain_ids
        )
        sim_child_seqs = [
            decode_sequence_from_toks(y_decoded[i].cpu().numpy(), vocab)
            for i in range(y_decoded.size(0))
        ]
        sim_hv_seqs, sim_lt_seqs = zip(
            *[(seq[:hl], seq[hl:]) for seq, hl in zip(sim_child_seqs, hc_lens)]
        )
        assert all([len(tc) == len(sc) for tc, sc in zip(true_child_seqs, sim_child_seqs)])

        n_batches -= 1
        if n_batches == 0:
            break

    # breakpoint()
