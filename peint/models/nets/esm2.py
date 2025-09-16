from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from esm.model.esm2 import ESM2
from torch.nn import functional as F
from transformers.models.esm.modeling_esm import (
    EsmAttention,
    EsmIntermediate,
    EsmLayer,
    EsmOutput,
    EsmSelfAttention,
    EsmSelfOutput,
)

from peint.models.nets.transformer import (
    ESMEncoderBlock,
    FAEsmRotaryEmbedding,
    FlashMHAEncoderBlock,
)

##########################
# Flash Based ESM2 Model #
##########################


class ESM2Flash(ESM2):
    """ESM2 model with FlashAttention mechanism"""

    def __init__(
        self,
        num_layers: int = 30,
        embed_dim: int = 640,
        attention_heads: int = 20,
        alphabet="ESM-1b",
        token_dropout: bool = True,
        use_bias: bool = True,
        add_bias_kv: bool = False,
        **kwargs,
    ) -> None:
        self.add_bias_kv = add_bias_kv
        self.dropout_p = kwargs.get("dropout_p", 0.0)
        self.attention_heads = attention_heads
        self.use_bias = use_bias
        super(ESM2Flash, self).__init__(
            num_layers, embed_dim, attention_heads, alphabet, token_dropout
        )

    def _init_submodules(self):
        super()._init_submodules()
        self.layers = nn.ModuleList(
            [
                FlashMHAEncoderBlock(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=4 * self.embed_dim,
                    attention_heads=self.attention_heads,
                    add_bias_kv=self.add_bias_kv,
                    dropout_p=self.dropout_p,
                    use_bias=self.use_bias,
                    layer_idx=i,
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(self, x, repr_layers=[], **kwargs):
        h_x = self.embed_scale * self.embed_tokens(x)

        padding_mask = x.eq(self.padding_idx)

        if self.token_dropout:
            h_x.masked_fill_((x == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (x == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            h_x = h_x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            h_x = h_x * (
                1 - padding_mask.unsqueeze(-1).type_as(h_x)
            )  # multiplying by 0 the padding positions, not sure if actually needed

        hidden_representations = {}

        if 0 in repr_layers:
            hidden_representations[0] = h_x

        # mask is dealt with in the forward function of the layer
        encoder_kwargs = {
            "x_padding_mask": ~padding_mask
        }  # padding mask is inverted - 1 means attend

        for layer_idx, layer in enumerate(self.layers):
            h_x = layer(h_x, **encoder_kwargs)

            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = h_x

        h_x = self.emb_layer_norm_after(h_x)  # final_layer_norm
        h_x = self.lm_head(h_x)

        result = {"logits": h_x, "representations": hidden_representations}

        return result


############################################
# Non A100 version of the above model ######
############################################
class ESM2Model(ESM2):
    """ESM2 model"""

    def __init__(
        self,
        num_layers: int = 30,
        embed_dim: int = 640,
        attention_heads: int = 20,
        alphabet="ESM-1b",
        token_dropout: bool = True,
        use_bias: bool = True,
        add_bias_kv: bool = False,
        **kwargs,
    ) -> None:
        self.add_bias_kv = add_bias_kv
        self.dropout_p = kwargs.get("dropout_p", 0.0)
        self.attention_heads = attention_heads
        self.use_bias = use_bias
        super(ESM2Model, self).__init__(
            num_layers, embed_dim, attention_heads, alphabet, token_dropout
        )

    def _init_submodules(self):
        super()._init_submodules()
        self.layers = nn.ModuleList(
            [
                ESMEncoderBlock(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=4 * self.embed_dim,
                    attention_heads=self.attention_heads,
                    add_bias_kv=self.add_bias_kv,
                    dropout_p=self.dropout_p,
                    use_bias=self.use_bias,
                    layer_idx=i,
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(self, x, repr_layers=[], **kwargs):
        h_x = self.embed_scale * self.embed_tokens(x)

        padding_mask = x.eq(self.padding_idx)

        if self.token_dropout:
            h_x.masked_fill_((x == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (x == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            h_x = h_x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            h_x = h_x * (
                1 - padding_mask.unsqueeze(-1).type_as(h_x)
            )  # multiplying by 0 the padding positions, not sure if actually needed

        hidden_representations = {}

        if 0 in repr_layers:
            hidden_representations[0] = h_x

        for layer_idx, layer in enumerate(self.layers):
            h_x, attn = layer(h_x, attn_mask=padding_mask)

            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = h_x

        h_x = self.emb_layer_norm_after(h_x)  # final_layer_norm
        h_x = self.lm_head(h_x)

        result = {"logits": h_x, "representations": hidden_representations}

        return result


###################################
# Additional Useful ESM2 Components
###################################


class FAEsmSelfAttention(EsmSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.config = config
        if self.config.use_fa:
            self.rotary_embeddings = FAEsmRotaryEmbedding(dim=self.attention_head_size)

    def forward(self, **kwargs):
        if self.config.use_fa:
            return self.fa_forward(**kwargs)
        else:
            return self.sdpa_forward(**kwargs)

    def sdpa_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        query_layer = query_layer * self.attention_head_size**-0.5

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            raise NotImplementedError

        # Mask heads if we want to
        if head_mask is not None:
            raise NotImplementedError

        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()

        context_layer = F.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attn_mask=attention_mask, scale=1.0
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    def fa_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens,
        max_seqlen,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

        assert cu_seqlens is not None, "cu_seqlens must be provided for FlashAttention"
        assert max_seqlen is not None, "max_seqlen must be provided for FlashAttention"
        q = self.query(hidden_states) * self.attention_head_size**-0.5
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> n h d", h=self.num_attention_heads),
            (q, k, v),
        )
        qkv = torch.stack((q, k, v), dim=1)  # (n, 3, h, d)
        qkv = self.rotary_embeddings(qkv=qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        out = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, softmax_scale=1.0)
        out = rearrange(out, "n h d -> n (h d)")
        outputs = (out,)
        return outputs


class FAEsmAttention(EsmAttention):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.self = FAEsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        hidden_states_ln = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            hidden_states=hidden_states_ln,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class FAEsmLayer(EsmLayer):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = FAEsmAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = FAEsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
                    " with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = self.feed_forward_chunk(attention_output)

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs
