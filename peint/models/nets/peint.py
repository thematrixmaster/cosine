from typing import List

import numpy as np
import torch
import torch.nn as nn
from esm.model.esm2 import ESM2
from esm.modules import RobertaLMHead
from torch import Tensor
from tqdm import tqdm

from evo.tokenization import Vocab
from peint.data.utils import RESIDUES
from peint.models.nets.esm2 import ESM2Flash, ESM2Model
from peint.models.nets.transformer import (
    ESMDecoderBlock,
    ESMEncoderBlock,
    FlashMHADecoderBlock,
    FlashMHAEncoderBlock,
    GeometricTimeEmbedder,
    MultiSequenceDecoderBlock,
    MultiSequenceEncoderBlock,
)
from peint.models.nets.utils import _create_sequence_mask, _expand_distances_to_seqlen

STANDARD_STATES = RESIDUES + ["<eos>"]

try:
    import flash_attn  # noqa

    ESM2Wrapper = ESM2Flash
    EncoderBlock = FlashMHAEncoderBlock
    DecoderBlock = FlashMHADecoderBlock
except ImportError:
    ESM2Wrapper = ESM2Model
    EncoderBlock = ESMEncoderBlock
    DecoderBlock = ESMDecoderBlock


def sampling_function(logits, p=0.9, argmax_sample=False):
    """
    Perform top p sampling on the given logits.

    Args:
    logits (torch.Tensor): Logits of shape [batch_size, vocab_size]
    p (float): Nucleus sampling parameter, default is 0.9

    Returns:
    torch.Tensor: Sampled token indices of shape [batch_size, 1]
    """

    probs = nn.functional.softmax(logits, dim=-1)

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


class PEINT(nn.Module):
    """Special version of the encoder-decoder transformer that builds on top of an ESM Model.
    The ESM model will first encode a sequence, and the final hidden representation will go into an encoder/decoder stack.
    The encoder/decoder stack will then work as a standard transformer.

    The ESM model is frozen and the transformer is trained on top of it.
    This also freezes the embedding layer and the LM head to stay in the same amino acid representation space.

    Time is encoded using sinusoidal positional encodings by bin.
    Positions are encoded using Rotary Embeddings (handled within the MHA modules).

    The entire model uses Flash Attention, including a rewritten version of the ESM model.
    """

    def __init__(
        self,
        esm_model: ESM2,
        evo_vocab: Vocab,
        embed_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        max_len=1022,
        dropout_p: float = 0.0,
        use_attention_bias: bool = True,
        finetune_esm: bool = False,
        tie_lm_heads: bool = True,
    ):
        super(PEINT, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        assert num_encoder_layers >= num_decoder_layers
        self.max_len = max_len

        self.esm = esm_model
        self.finetune_esm = finetune_esm
        self.tie_lm_heads = tie_lm_heads
        self.vocab = evo_vocab
        self.esm.eval() if not self.finetune_esm else self.esm.train()
        self.esm.requires_grad_(self.finetune_esm)  # freeze the ESM model
        self.dropout_p = dropout_p
        self.use_bias = use_attention_bias

        self.criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.vocab.pad_idx)

        # embedding layer from ESM
        self.embedding = nn.Embedding(len(self.vocab), embed_dim)
        self.embedding.load_state_dict(self.esm.embed_tokens.state_dict())
        self.embedding.requires_grad_(self.finetune_esm)  # freeze the ESM embedding layer

        self.time_embedding = GeometricTimeEmbedder(frequency_embedding_size=embed_dim)

        self.enc_layers = nn.ModuleList(
            [
                EncoderBlock(
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
                DecoderBlock(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=4 * self.embed_dim,
                    attention_heads=self.num_heads,
                    add_bias_kv=False,
                    dropout_p=self.dropout_p,
                    use_bias=self.use_bias,
                    layer_idx=l,
                )
                for l in range(num_decoder_layers)
            ]
        )

        self.enc_lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=len(self.vocab),
            weight=self.embedding.weight,
        )
        self.enc_lm_head.load_state_dict(self.esm.lm_head.state_dict())
        self.enc_lm_head.requires_grad_(self.finetune_esm)

        if self.tie_lm_heads:
            self.dec_lm_head = self.enc_lm_head  # use same lm head for encoder and decoder
        else:
            self.dec_lm_head = RobertaLMHead(
                embed_dim=self.embed_dim,
                output_dim=len(self.vocab),
                weight=self.embedding.weight,
            )
            self.dec_lm_head.load_state_dict(self.esm.lm_head.state_dict())
            self.dec_lm_head.requires_grad_(self.finetune_esm)

    def _compute_language_model_representations(self, x):
        # x already has CLS and EOS from dataloader
        # esm deals with the padding directly (doesn't need the mask even though we have it)

        res = self.esm(x, repr_layers=[self.esm.num_layers], need_head_weights=False)

        esm_s = res["representations"][self.esm.num_layers]  # just the final hidden state

        return esm_s

    def forward(self, x, y, t, x_pad_mask, y_pad_mask):
        # x: b x l, y: b x l, t: b x 1, x_pad_mask: b x l, y_pad_mask: b x l
        # embed y with t
        h_y = self.embedding(y)
        ht = self.time_embedding(t)
        ht = ht.expand_as(h_y)
        h_y = h_y + ht

        # get lm representations
        h_x = self._compute_language_model_representations(x)

        x_attn_mask = ~x_pad_mask  # now 1 means attend, 0 means don't
        y_attn_mask = ~y_pad_mask

        for i, enc_layer in enumerate(self.enc_layers):
            encoder_kwargs = {"x_padding_mask": x_pad_mask, "x_attn_mask": x_attn_mask}
            h_x = enc_layer(x=h_x, **encoder_kwargs)

            if self.num_decoder_layers - self.num_encoder_layers + i >= 0:
                decoder_kwargs = {
                    "x_padding_mask": y_pad_mask,
                    "y_padding_mask": x_pad_mask,
                    "x_attn_mask": y_attn_mask,
                    "y_attn_mask": x_attn_mask,
                }
                idx = self.num_decoder_layers - self.num_encoder_layers + i
                dec_layer = self.dec_layers[idx]
                h_y = dec_layer(x=h_y, y=h_x, **decoder_kwargs)

        x_logits = self.enc_lm_head(h_x)
        y_logits = self.dec_lm_head(h_y)

        if torch.any(torch.isnan(x_logits)) or torch.any(torch.isnan(y_logits)):
            raise ValueError("NaN detected in logits")

        return x_logits, y_logits

    def decode_sequences(self, decoded: torch.Tensor) -> List[str]:
        """Turns a batch of decoded sequences into a list of strings

        decoded: [B, L] tensor of decoded sequences
        """
        inv_vocab = {v: k for k, v in self.vocab.to_dict().items()}
        output_sequences = []
        for seq in decoded:
            decoded_str = "".join([inv_vocab.get(p.item()) for p in seq[1:]])  # remove cls
            eos_idx = decoded_str.find("<eos>")
            if eos_idx != -1:
                decoded_str = decoded_str[:eos_idx]
            output_sequences.append(decoded_str)
        return output_sequences

    def _precompute_encoder_states(
        self, x: torch.Tensor, x_pad_mask: torch.Tensor, return_logits=False
    ):
        """Precompute encoder states that don't change during generation"""
        # Get ESM representations
        h_x = self._compute_language_model_representations(x)
        x_attn_mask = ~x_pad_mask
        h_x_cached = []

        # Process through encoder layers
        for enc_layer in self.enc_layers:
            encoder_kwargs = {"x_padding_mask": x_pad_mask, "x_attn_mask": x_attn_mask}
            h_x = enc_layer(x=h_x, **encoder_kwargs)
            h_x_cached.append(h_x)

        if return_logits:
            x_logits = self.enc_lm_head(h_x)
            return x_logits, h_x_cached, x_attn_mask

        return h_x_cached, x_attn_mask

    def _decode_with_cached_encoder(
        self,
        y: torch.Tensor,
        t: torch.Tensor,
        h_x_cached: List[torch.Tensor],
        x_pad_mask: torch.Tensor,
        y_pad_mask: torch.Tensor,
        x_attn_mask: torch.Tensor,
        return_hidden_states=False,
    ) -> torch.Tensor:
        """Decode using pre-cached encoder states"""
        # Embed y with t
        h_y = self.embedding(y)
        ht = self.time_embedding(t)
        ht = ht.expand_as(h_y)
        h_y = h_y + ht
        h_y_cached = []

        y_attn_mask = ~y_pad_mask

        # Only run decoder layers that need cross-attention
        for i in range(self.num_decoder_layers):
            decoder_kwargs = {
                "x_padding_mask": y_pad_mask,
                "y_padding_mask": x_pad_mask,
                "x_attn_mask": y_attn_mask,
                "y_attn_mask": x_attn_mask,
            }
            idx = i + (self.num_encoder_layers - self.num_decoder_layers)
            dec_layer = self.dec_layers[i]
            h_x = h_x_cached[idx]
            h_y_cached.append(h_y)
            h_y = dec_layer(x=h_y, y=h_x, **decoder_kwargs)

        if return_hidden_states:
            return self.dec_lm_head(h_y), h_y_cached

        return self.dec_lm_head(h_y)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,  # <bos>x<eos> format
        t: torch.Tensor,
        max_decode_steps: int,
        device: torch.device,
        temperature: float = 1.0,
        p: float = 1.0,
    ) -> torch.Tensor:
        """Generate sequences using nucleus sampling with cached encoder states"""

        batch_size = x.size(0)
        x_pad_mask = x.eq(self.vocab.pad_idx)
        y_decoded = torch.tensor([self.vocab.bos_idx]).unsqueeze(0).repeat(batch_size, 1).to(device)
        eos_reached = torch.zeros(batch_size, dtype=torch.bool).to(device)

        # Precompute encoder states once
        h_x_cached, x_attn_mask = self._precompute_encoder_states(x, x_pad_mask)

        for _ in range(max_decode_steps):
            y_pad_mask = y_decoded.eq(self.vocab.pad_idx)

            # Use cached encoder states for decoding
            logits = self._decode_with_cached_encoder(
                y_decoded, t, h_x_cached, x_pad_mask, y_pad_mask, x_attn_mask
            )

            logits = logits[:, -1, :] / temperature
            # want to renormalize the probs away from extraneous states, this includes gaps
            zero_idx = torch.tensor(
                [self.vocab.index(t) for t in self.vocab.tokens if t not in STANDARD_STATES]
            )
            logits[..., zero_idx] = -np.inf

            next_tok = sampling_function(logits, p=p)
            next_tok[eos_reached] = self.vocab.pad_idx
            y_decoded = torch.cat([y_decoded, next_tok], dim=1)

            eos_reached |= next_tok.squeeze(-1) == self.vocab.eos_idx
            if eos_reached.all():
                break

        return y_decoded

    @torch.no_grad()
    def perplexity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,  # assumes <bos>y<eos> format
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Compute perplexity of (y, t) pairs given each x"""

        # Precompute encoder states once
        x_pad_mask = x.eq(self.vocab.pad_idx)
        h_x_cached, x_attn_mask = self._precompute_encoder_states(x, x_pad_mask)

        n_inputs = x.shape[0]
        n_targets = y.shape[0]
        assert y.shape[0] == t.shape[0]
        y_lengths = (y[:, 1:] != self.vocab.pad_idx).sum(dim=-1).float()

        # Initialize result tensor
        all_perplexities = torch.zeros(n_inputs, n_targets, device=x.device)

        for i in range(n_inputs):
            target_nlls = []

            for j in range(0, n_targets, batch_size):
                y_chunk = y[j : j + batch_size]
                t_chunk = t[j : j + batch_size]
                _bs = y_chunk.shape[0]
                y_pad_mask = y_chunk.eq(self.vocab.pad_idx)

                _h_x_cached = [h[i].unsqueeze(0).expand(_bs, -1, -1) for h in h_x_cached]
                _x_attn_mask = x_attn_mask[i].unsqueeze(0).expand(_bs, -1)
                _x_pad_mask = x_pad_mask[i : i + 1].expand(_bs, -1)

                y_logits = self._decode_with_cached_encoder(
                    y_chunk, t_chunk, _h_x_cached, _x_pad_mask, y_pad_mask, _x_attn_mask
                )

                y_nll = nn.functional.cross_entropy(
                    y_logits[:, :-1, :].transpose(-1, -2),
                    y_chunk[:, 1:],
                    ignore_index=self.vocab.pad_idx,
                    reduction="none",
                )
                y_tgt_mask = y_chunk[:, 1:] != self.vocab.pad_idx
                y_nll = (y_nll * y_tgt_mask.float()).sum(-1)
                target_nlls.append(y_nll)

            # Concatenate likelihoods for this input
            input_nlls = torch.cat(target_nlls, dim=0)
            # Compute perplexities for this input against all targets
            all_perplexities[i] = torch.exp(input_nlls / y_lengths)

        return all_perplexities.cpu().transpose(0, 1)  # shape: [n_targets, n_inputs]

    @classmethod
    def from_pretrained_esm2(cls, *args, **kwargs) -> "PEINT":
        """Create a PEINTModule from a pretrained ESM model."""
        import esm

        _esm_model, esm_vocab = esm.pretrained.esm2_t30_150M_UR50D()
        esm_model = ESM2Wrapper(
            num_layers=_esm_model.num_layers,
            embed_dim=_esm_model.embed_dim,
            attention_heads=_esm_model.attention_heads,
            alphabet="ESM-1b",
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
        return cls(esm_model=esm_model, evo_vocab=evo_vocab, *args, **kwargs)


class PIPET(PEINT):
    """
    Pipet: Protein interaction evolution in time

    Transformer that models P(y2, x2 | y1, y2, tx, ty).
    The model is trained to predict y2 and x2 given y1 and x2,
    where y1 and y2 are separated in evolution by ty, and x1 and x2 by tx
    The logic is handled in the encoder and decoder modules, as well as the data preprocessing.
    """

    def __init__(
        self,
        esm_model: ESM2,
        evo_vocab: Vocab,
        embed_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        max_len=1022,
        dropout_p: float = 0.0,
        use_attention_bias: bool = True,
        finetune_esm: bool = False,
        tie_lm_heads: bool = True,
        chain_break_token: str = ".",
        encoder_self_attn_type: str = "intra_inter",
        decoder_self_attn_type: str = "full",
        *args,
        **kwargs,
    ):
        super(PIPET, self).__init__(
            esm_model=esm_model,
            evo_vocab=evo_vocab,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            max_len=max_len,
            dropout_p=dropout_p,
            use_attention_bias=use_attention_bias,
            finetune_esm=finetune_esm,
            tie_lm_heads=tie_lm_heads,
        )
        self.chain_break_token = chain_break_token
        self.sep_idx = self.vocab.index(chain_break_token)
        self.enc_layers = nn.ModuleList(
            [
                MultiSequenceEncoderBlock(
                    attention_heads=num_heads,
                    embed_dim=embed_dim,
                    ffn_embed_dim=4 * embed_dim,
                    dropout_p=dropout_p,
                    layer_idx=i,
                    self_attn_type=encoder_self_attn_type,
                )
                for i in range(num_encoder_layers)
            ]
        )
        self.dec_layers = nn.ModuleList(
            [
                MultiSequenceDecoderBlock(
                    attention_heads=num_heads,
                    embed_dim=embed_dim,
                    ffn_embed_dim=4 * embed_dim,
                    dropout_p=dropout_p,
                    layer_idx=i,
                    self_attn_type=decoder_self_attn_type,
                )
                for i in range(num_decoder_layers)
            ]
        )

    def _compute_language_model_representations(
        self, x: Tensor, attn_mask_in_length: Tensor
    ) -> Tensor:
        """
        Compute ESM embedding separately for each sequence in the batch

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)
            attn_mask_in_length (torch.Tensor): Tensor of shape (batch_size, seq_len) containing
                the lengths of sequences in each batch.
                Non-zero values indicate the length of each sequence,
                while zeros represent padding.
        """
        bsz, seq_len = x.shape
        num_sequences = (attn_mask_in_length > 0).sum(dim=1)

        # Assert that all batch elements have the same number of sequences
        assert torch.all(
            num_sequences == num_sequences[0]
        ), "All batch elements must have the same number of sequences"

        num_sequences = num_sequences[0].item()
        combined_embedding = torch.zeros(
            (bsz, seq_len, self.embed_dim), dtype=torch.float, device=x.device
        )

        for seq_idx in range(num_sequences):
            # Shape (B, L), True for the residues of `seq_idx`th sequence of each batch element
            seq_mask = _create_sequence_mask(attn_mask_in_length, sequence_idx=seq_idx)
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

            # Add the current sequence embedding to the combined embedding
            combined_embedding += embedding * seq_mask.unsqueeze(-1)

        return combined_embedding

    def forward(
        self,
        enc_in: Tensor,
        dec_in: Tensor,
        enc_attn_mask: Tensor,
        dec_attn_mask: Tensor,
        distances: Tensor,
        **kwargs,
    ):
        """
        Args:
        enc_in: Shape (B, L), encoded tensor for x1 and y1
            each chain is prepended with <cls> and appended with <eos>
        dec_in: Shape (B, L), encoded tensor for x2 and y2 separated
            by chain break token.
        enc_lengths: Shape (B, 2), enc_lengths[b, i] is the length of
            chain i in batch element b.
        distances: Shape (B, 2), distances[b, i] is the distance for chain i
            in batch element b.
        """
        # Embed input tokens for decoder
        h_dec = self.embedding(dec_in)

        # Expand the distance to each position of the sequence, then embed and add
        distances_expanded = _expand_distances_to_seqlen(
            distances_in_length=distances, attn_mask_in_length=dec_attn_mask
        )

        h_t = self.time_embedding(distances_expanded)
        h_dec = h_dec + h_t

        # Get ESM representations for encoder input
        h_enc = self._compute_language_model_representations(enc_in, enc_attn_mask)

        dec_self_attn_weights_all = []
        for i, enc_layer in enumerate(self.enc_layers):
            h_enc = enc_layer(h_enc, enc_attn_mask)

            if self.num_decoder_layers - self.num_encoder_layers + i >= 0:
                idx = self.num_decoder_layers - self.num_encoder_layers + i
                dec_layer = self.dec_layers[idx]
                h_dec, dec_self_attn_weights = dec_layer(
                    h_dec, h_enc, dec_attn_mask, enc_attn_mask, **kwargs
                )
                dec_self_attn_weights_all.append(dec_self_attn_weights)

        enc_logits = self.enc_lm_head(h_enc)
        dec_logits = self.dec_lm_head(h_dec)

        outputs = dict(enc_logits=enc_logits, dec_logits=dec_logits)
        if kwargs.get("return_decoder_self_attn_weights", False):
            outputs["dec_self_attn_weights"] = dec_self_attn_weights_all
        return outputs

    def decode_sequences(self, decoded_toks: torch.Tensor) -> List[tuple]:
        """
        Turns a batch of decoded sequences into a list of tuples of strings,
        each tuple containing the two chains.

        Args:
            decoded_toks: [B, L] tensor of decoded sequences

        Returns:
            A list of tuples (chain1, chain2) for each sample in the batch.
        """
        inv_vocab = {v: k for k, v in self.vocab.to_dict().items()}
        output_sequences = []

        for seq in decoded_toks:
            # Convert to string, removing CLS token
            decoded_str = "".join([inv_vocab.get(p.item()) for p in seq[1:]])

            # Find chain break and EOS positions
            chain_break_idx = decoded_str.find(self.chain_break_token)
            eos_idx = decoded_str.find(self.vocab.eos_token)

            # Extract the two chains
            if chain_break_idx != -1:
                chain1 = decoded_str[:chain_break_idx]

                if eos_idx != -1 and eos_idx > chain_break_idx:
                    # Valid sequence with both chain break and EOS
                    chain2 = decoded_str[chain_break_idx + 1 : eos_idx]
                else:
                    # No EOS found after chain break
                    chain2 = decoded_str[chain_break_idx + 1 :]
            else:
                # No chain break found
                if eos_idx != -1:
                    chain1 = decoded_str[:eos_idx]
                else:
                    chain1 = decoded_str
                chain2 = ""

            output_sequences.append((chain1, chain2))

        return output_sequences

    @torch.no_grad()
    def generate(
        self,
        enc_in: torch.Tensor,
        enc_lengths: torch.Tensor,
        distances: torch.Tensor,
        max_decode_steps: int,
        device=torch.device,
        temperature: float = 1.0,
        p: float = 1.0,
    ):
        """
        Args:
            enc_in: Shape (B, L), encoded tensor for x1 and y1
                each chain is prepended with <cls> and appended with <eos>
            enc_lengths: Shape (B, 2), enc_lengths[b, i] is the length of
                chain i in batch element b.
            distances: Shape (B, 2), distances[b, i] is the distance for chain i
                in batch element b.
        """
        batch_size = enc_in.size(0)
        num_chains = 2  # Currently assume we have exactly 2 chains

        # Assign encoder attention mask, shape (B, L_enc)
        enc_attn_mask = torch.zeros_like(enc_in)
        enc_attn_mask[:, :num_chains] = enc_lengths

        # Shape (B, 1)
        decoded_toks = (torch.tensor([self.vocab.bos_idx]).unsqueeze(0).repeat(batch_size, 1)).to(
            device
        )

        # Track which chain we're currently generating (0 = first chain, 1 = second chain)
        current_chain_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Track the length of each chain, start with length = 1 for chain 0 (CLS token)
        dec_lengths = torch.zeros((batch_size, num_chains), dtype=torch.long, device=device)
        dec_lengths[:, 0] = 1

        # Track if all sequences have reached EOS for early stopping
        eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)
        eos_idx = self.vocab.eos_idx

        for i in tqdm(
            range(max_decode_steps),
            desc=f"Decoding: batch size={batch_size}, max steps={max_decode_steps}",
        ):
            # Decoder attention mask in length, shape (B, L_dec)
            dec_attn_mask = torch.zeros_like(decoded_toks)
            # Evolutionary distances tx and ty, shape (B, L_dec)
            distances_tensor = torch.zeros_like(decoded_toks, dtype=torch.float32)
            # Special case when we have only 1 token, dec_attn_mask and distances_tensor
            # need to be of shape (B, 1), but dec_lengths and distances are shape (B, 2)
            if i == 0:
                dec_attn_mask[:, 0] = dec_lengths[:, 0]
                distances_tensor[:, 0] = distances[:, 0]
            else:
                dec_attn_mask[:, :num_chains] = dec_lengths
                distances_tensor[:, :num_chains] = distances

            outputs = self(
                enc_in=enc_in,
                dec_in=decoded_toks,
                enc_attn_mask=enc_attn_mask,
                dec_attn_mask=dec_attn_mask,
                distances=distances_tensor,
            )
            logits = outputs["dec_logits"]

            # Scale by temperature, renormalize out other states
            logits = logits[:, -1, :] / temperature
            zero_idx = torch.tensor(
                [self.vocab.tokens_to_idx[t] for t in self.vocab.tokens if t not in STANDARD_STATES]
            )
            logits[..., zero_idx] = -np.inf

            # Hard constraints:
            # 1. First chain: EOS is not allowed
            first_chain_mask = current_chain_idx == 0
            logits[first_chain_mask, eos_idx] = -float("inf")

            # 2. Second chain: Chain break is not allowed
            second_chain_mask = current_chain_idx == 1
            logits[second_chain_mask, self.sep_idx] = -float("inf")

            # Sample next token, append to decoded sequence
            next_tok = sampling_function(logits, p=p)
            decoded_toks = torch.cat([decoded_toks, next_tok], dim=1)

            # Update which chain we're currently generating
            # If hit chain break, then start decoding the 2nd chain
            chain_breaks = next_tok.squeeze(-1) == self.sep_idx
            current_chain_idx = torch.where(
                chain_breaks, torch.ones_like(current_chain_idx), current_chain_idx
            )

            # Update lengths for appropriate chains
            for b in range(batch_size):
                if not eos_reached[b]:  # Only update if EOS not reached
                    dec_lengths[b, current_chain_idx[b]] += 1

            # Update EOS status
            new_eos = next_tok.squeeze(-1) == eos_idx
            eos_reached = eos_reached | new_eos
            if eos_reached.all():
                break

        return self.decode_sequences(decoded_toks)

    def _precompute_encoder_states(self, *args, **kwargs):
        raise NotImplementedError("PIPET does not support separate encoder state precomputation")

    def _decode_with_cached_encoder(self, *args, **kwargs):
        raise NotImplementedError(
            "PIPET does not support separate decoding with cached encoder states"
        )

    def perplexity(self, *args, **kwargs):
        raise NotImplementedError("PIPET does not support perplexity computation")
