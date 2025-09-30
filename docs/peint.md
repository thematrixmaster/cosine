# PEINT: Unified Protein Evolution Modeling Architecture

## Executive Summary

This document describes the **PEINT** (Protein Evolution Inference Transformer) architecture, which provides a unified framework for both single-sequence evolution modeling `P(x2|x1, t)` and multi-sequence protein-protein interaction evolution `P(x2, y2|x1, y1, tx, ty)`. The architecture uses configuration-driven processing to handle both use cases within a single model implementation, eliminating the need for separate PEINT and PIPET classes.

---

## Table of Contents

1. [Unified Architecture Overview](#1-unified-architecture-overview)
2. [ESM Encoder Abstraction](#2-esm-encoder-abstraction)
3. [Single vs Multi-Sequence Processing](#3-single-vs-multi-sequence-processing)
4. [Size-Based Interface Design](#4-size-based-interface-design)
5. [Forward Pass Implementation](#5-forward-pass-implementation)
6. [Training Configuration](#6-training-configuration)
7. [Loss Computation](#7-loss-computation)
8. [Generation and Inference](#8-generation-and-inference)
9. [Attention Mechanism](#9-attention-mechanism)
10. [Key Architectural Insights](#10-key-architectural-insights)

---

## 1. Unified Architecture Overview

### PEINT: Single Model, Dual Capability
- **Base Class**: Direct `nn.Module` inheritance
- **Purpose**: Unified protein evolution modeling (single and multi-sequence)
- **Encoder**: `FlashMHAEncoderBlock` layers (supports both modes)
- **Decoder**: `FlashMHADecoderBlock` layers (supports both modes)
- **Processing Mode**: Controlled by `embed_x_per_chain` parameter
- **Constraint**: `num_encoder_layers >= num_decoder_layers`

**Key Configuration Parameters**:
```python
embed_x_per_chain: bool = False        # Enable multi-sequence ESM processing
chain_break_token: str = "."           # Separator between protein chains
use_attention_bias: bool = True        # Attention bias configuration
```

**Unified Architecture Components**:
```python
# Single set of layers that handle both single and multi-sequence inputs
self.enc_layers = nn.ModuleList([
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
])
```

### Architectural Simplification
The new architecture eliminates the need for separate PEINT/PIPET classes and specialized multi-sequence attention blocks, instead using:
- **Configuration-driven processing**: Single model handles both use cases
- **Standardized attention blocks**: Uses proven FlashMHA implementations
- **ESM encoder abstraction**: Clean separation of pretrained encoder logic

---

## 2. ESM Encoder Abstraction

### PretrainedEncoder Interface
The new architecture introduces a clean abstraction for pretrained encoders:

```python
class PretrainedEncoder(ABC, nn.Module):
    """Abstract base class for pretrained encoders."""

    def __init__(self, vocab: Vocab, embed_x_per_chain: bool = False):
        super(PretrainedEncoder, self).__init__()
        self.vocab = vocab
        self.embed_x_per_chain = embed_x_per_chain

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        pass

    @abstractmethod
    def get_in_embedding(self) -> nn.Embedding:
        pass

    @abstractmethod
    def get_out_lm_head(self) -> nn.Module:
        pass

    @abstractmethod
    def forward(self, x: Tensor, x_sizes: Tensor) -> Tensor:
        pass
```

### ESMEncoder Implementation
The `ESMEncoder` class provides a concrete implementation with flexible processing modes:

**Key Features**:
- **Dual processing modes**: Single sequence or per-chain embedding
- **Fine-tuning support**: Configurable parameter freezing
- **Consistent interface**: Standardized embedding and output layer access

**Processing Mode Selection**:
```python
def forward(self, x: Tensor, x_sizes: Tensor) -> Tensor:
    if self.embed_x_per_chain:
        # Multi-sequence processing: compute ESM per chain
        return self.forward_per_chain(x, x_sizes)
    else:
        # Single-sequence processing: process concatenated sequence
        res = self.esm(x, repr_layers=[self.esm.num_layers], need_head_weights=False)
        esm_s = res["representations"][self.esm.num_layers]
        return esm_s
```

**Per-Chain Processing**:
The `forward_per_chain` method enables multi-sequence capabilities by:
1. Processing each sequence separately through ESM
2. Combining embeddings while preserving sequence boundaries
3. Handling variable numbers of sequences per batch

---

## 3. Single vs Multi-Sequence Processing

### Single-Sequence Mode (`embed_x_per_chain=False`)
**Use Case**: Traditional protein evolution modeling `P(x2|x1, t)`
**ESM Processing**: Concatenated sequence processing through ESM
**Data Format**: Standard sequence pairs with simple masking
**Efficiency**: High - single ESM forward pass per sequence

**Processing Flow**:
```python
# Standard single-sequence processing
res = self.esm(x, repr_layers=[self.esm.num_layers], need_head_weights=False)
esm_s = res["representations"][self.esm.num_layers]
```

### Multi-Sequence Mode (`embed_x_per_chain=True`)
**Use Case**: Protein-protein interaction evolution `P(x2, y2|x1, y1, tx, ty)`
**ESM Processing**: Per-chain processing with sequence masking
**Data Format**: Chain-separated sequences with size-based masking
**Efficiency**: Lower - multiple ESM passes per sequence group

**Processing Flow**:
```python
# Per-chain processing with masking
for seq_idx in range(num_sequences):
    seq_mask = _create_sequence_mask(x_sizes, sequence_idx=seq_idx)
    x_seq = x.clone()
    x_seq.masked_fill_(~seq_mask, self.vocab.pad_idx)

    # Process individual sequence through ESM
    output = self.esm(x_seq, repr_layers=[self.esm.num_layers])
    embedding = output["representations"][self.esm.num_layers]

    # Accumulate in combined embedding
    combined_embedding += embedding * seq_mask.unsqueeze(-1)
```

**Chain Boundary Handling**:
- **Chain break tokens**: Maintain sequence separation during tokenization
- **Sequence masking**: Selective processing of individual chains
- **Embedding combination**: Additive combination preserving spatial structure

---

## 4. Size-Based Interface Design

### Unified Masking Interface
The new architecture standardizes on a size-based masking interface that replaces boolean attention masks:

**Size Tensors**: `x_sizes` and `y_sizes` store sequence lengths rather than boolean masks
**Utility Functions**: Centralized mask creation from size information
**Flexibility**: Supports variable numbers of sequences per batch element

**Key Interface Functions**:
```python
# Create attention masks from sizes (1 for data, 0 for pad)
x_attn_mask = _create_padding_mask(x_sizes)
y_attn_mask = _create_padding_mask(y_sizes)

# Create sequence-specific masks for multi-sequence processing
seq_mask = _create_sequence_mask(x_sizes, sequence_idx=seq_idx)

# Expand time distances to sequence length
distances_expanded = _expand_distances_to_seqlen(distances_in_length=t, attn_mask_in_length=y_sizes)
```

**Advantages of Size-Based Interface**:
1. **Memory efficiency**: Stores lengths instead of full boolean tensors
2. **Flexibility**: Easily supports variable sequence counts
3. **Consistency**: Single interface for both single and multi-sequence modes
4. **Utility**: Enables advanced masking patterns for multi-sequence attention

### Forward Pass Signature
The unified forward pass uses the size-based interface:

```python
def forward(
    self, x: Tensor, y: Tensor, t: Tensor, x_sizes: Tensor, y_sizes: Tensor
) -> Dict[str, Tensor]:
    """
    Unified forward pass using standardized size-based interface.

    Args:
        x: Encoder input sequences (batch_size, seq_len)
        y: Decoder input sequences (batch_size, seq_len)
        t: Time values (batch_size, num_chains)
        x_sizes: Sequence sizes for encoder (batch_size, seq_len)
        y_sizes: Sequence sizes for decoder (batch_size, seq_len)

    Returns:
        Dict with 'enc_logits' and 'dec_logits' keys
    """
```

---

## 5. Training Configuration

### Single-Sequence Configuration (`train_peint.yaml`)

```yaml
defaults:
  - override /data/dataset: peint
  - override /data/dataset@data.dataset_val: dms  # DMS validation
  - override /metrics: koenig_dms                 # Correlation metrics

net:
  num_heads: 20
  num_encoder_layers: 3
  num_decoder_layers: 3
  embed_x_per_chain: false      # Single-sequence mode
  chain_break_token: "."
  use_attention_bias: true
```

### Multi-Sequence Configuration (`train_pipet.yaml`)

```yaml
defaults:
  - override /data/dataset: pipet
  - override /metrics: none        # No specialized metrics yet

net:
  num_heads: 20
  num_encoder_layers: 3
  num_decoder_layers: 3
  embed_x_per_chain: true       # Multi-sequence mode
  chain_break_token: "."
  use_attention_bias: true
```

### Configuration Differences
The key difference between single and multi-sequence training is the `embed_x_per_chain` parameter:

- **`embed_x_per_chain: false`**: Uses concatenated sequence processing (traditional PEINT)
- **`embed_x_per_chain: true`**: Uses per-chain ESM processing (multi-sequence capabilities)

### Common Training Hyperparameters

Both configurations share identical optimization settings:
```yaml
trainer:
  max_steps: 300000
  val_check_interval: 4000
  precision: bf16

model:
  optimizer:
    lr: 0.0003
  weight_decay: 0.01
  scheduler:
    num_warmup_steps: 2000

data:
  batch_size: 16
```

---

## 6. Forward Pass Implementation

### Unified Forward Pass
The new PEINT architecture provides a single forward pass that handles both single and multi-sequence inputs:

```python
def forward(
    self, x: Tensor, y: Tensor, t: Tensor, x_sizes: Tensor, y_sizes: Tensor
) -> Dict[str, Tensor]:
    """
    Unified forward pass using standardized size-based interface.
    """
    # Embed decoder input with time
    h_y = self.in_embedding(y)

    # Position-specific time embedding using distances
    distances_expanded = _expand_distances_to_seqlen(distances_in_length=t, attn_mask_in_length=y_sizes)
    ht = self.time_embedding(distances_expanded)
    h_y = h_y + ht

    # Create attention masks from sizes (1 for data, 0 for pad)
    x_attn_mask = _create_padding_mask(x_sizes)
    y_attn_mask = _create_padding_mask(y_sizes)

    # Get pretrained encoder representations
    h_x = self.enc_model.forward(x=x, x_sizes=x_sizes)

    # Encoder-decoder forward pass
    for i, enc_layer in enumerate(self.enc_layers):
        # Encoder layer
        h_x = enc_layer(x=h_x, x_attn_mask=x_attn_mask)

        # Decoder layers (with encoder-decoder alignment)
        if self.num_decoder_layers - self.num_encoder_layers + i >= 0:
            idx = self.num_decoder_layers - self.num_encoder_layers + i
            dec_layer = self.dec_layers[idx]
            # Decoder layer (cross-attention to encoder)
            h_y = dec_layer(x=h_y, x_attn_mask=y_attn_mask, y=h_x, y_attn_mask=x_attn_mask)

    # Generate logits
    x_logits = self.out_lm_head(h_x)
    y_logits = self.out_lm_head(h_y)

    return dict(enc_logits=x_logits, dec_logits=y_logits)
```

### Key Implementation Features

1. **Size-Based Interface**: Uses `x_sizes` and `y_sizes` for flexible masking
2. **Position-Specific Time Embedding**: Supports per-position evolutionary distances
3. **ESM Encoder Abstraction**: Clean separation via `enc_model.forward()`
4. **Standard Attention Blocks**: Uses `FlashMHAEncoderBlock` and `FlashMHADecoderBlock`
5. **Encoder-Decoder Alignment**: Configurable layer alignment for cross-attention

### Processing Mode Differences

The same forward pass handles both modes through the ESM encoder:

**Single-Sequence Mode** (`embed_x_per_chain=False`):
- ESM processes concatenated sequences directly
- Standard attention masks from sequence sizes
- Efficient single-pass processing

**Multi-Sequence Mode** (`embed_x_per_chain=True`):
- ESM processes each chain separately via `forward_per_chain`
- Sequence-aware masking and embedding combination
- Higher computational cost but preserves chain structure

---

## 7. Loss Computation

### Unified Loss Implementation
The new PEINT architecture maintains the same dual-loss formulation for both single and multi-sequence modes:

```python
def model_step(self, batch):
    # Unified batch format with size-based interface
    [x_src, x_tgt, y_src, y_tgt, distances, x_sizes, y_sizes] = batch

    # Forward pass with size-based interface
    outputs = self(x_src, y_src, distances, x_sizes, y_sizes)
    enc_logits, dec_logits = outputs["enc_logits"], outputs["dec_logits"]

    # Identical loss computation for both modes
    mlm_loss = self.criterion(enc_logits.transpose(-1, -2), x_tgt)
    tlm_loss = self.criterion(dec_logits.transpose(-1, -2), y_tgt)

    loss = mlm_loss + tlm_loss
    return {
        "mlm_loss": mlm_loss, "tlm_loss": tlm_loss,
        "mlm_ppl": torch.exp(mlm_loss), "tlm_ppl": torch.exp(tlm_loss)
    }
```

### Loss Components
- **MLM Loss**: Masked language modeling loss on encoder sequences
- **TLM Loss**: Autoregressive language modeling loss on decoder sequences
- **Combined Loss**: Simple addition of both components

### Benefits of Unified Loss
1. **Consistency**: Same loss function regardless of processing mode
2. **Simplicity**: No mode-specific loss calculations
3. **Comparability**: Direct comparison between single and multi-sequence training
4. **Efficiency**: Single loss computation path

---

## 6. Generation and Inference

### PEINT Generation Capabilities

**Method**: `generate()` with nucleus sampling
```python
@torch.no_grad()
def generate(self, x, t, max_decode_steps, device, temperature=1.0, p=1.0):
    # Precompute encoder states (optimization)
    h_x_cached, x_attn_mask = self._precompute_encoder_states(x, x_pad_mask)

    # Autoregressive generation with cached encoder
    for step in range(max_decode_steps):
        logits = self._decode_with_cached_encoder(...)
        next_tok = sampling_function(logits, p=p)
        # Early stopping on EOS
```

**Additional Methods**:
- `perplexity()`: Efficient batch perplexity computation
- `_precompute_encoder_states()`: Encoder state caching for efficiency
- `decode_sequences()`: Convert tokens to amino acid strings

### PIPET Generation Capabilities

**Method**: `generate()` with multi-chain constraints
```python
@torch.no_grad()
def generate(self, enc_in, enc_lengths, distances, max_decode_steps, ...):
    # Track which chain is being generated (0 or 1)
    current_chain_idx = torch.zeros(batch_size, dtype=torch.long)

    # Hard constraints during generation:
    # 1. First chain: EOS not allowed
    # 2. Second chain: Chain break not allowed

    for step in range(max_decode_steps):
        # Apply constraints based on current chain
        if current_chain_idx == 0:
            logits[..., eos_idx] = -float("inf")
        elif current_chain_idx == 1:
            logits[..., sep_idx] = -float("inf")
```

**Key Features**:
- **Chain-aware generation**: Tracks and constrains generation per chain
- **Multi-chain output**: Returns `List[tuple]` of `(chain1, chain2)` pairs
- **Constraint enforcement**: Prevents invalid token sequences

**Limitations**:
- No encoder state caching (computational overhead)
- No perplexity computation method
- More complex generation logic

---

## 9. Attention Mechanism

### Unified Attention Architecture

The new PEINT architecture uses standard FlashMHA attention blocks that handle both single and multi-sequence inputs without specialized multi-sequence attention modules:

**Encoder Attention**: `FlashMHAEncoderBlock`
```python
# Standard Flash Multi-Head Attention encoder
self.enc_layers = nn.ModuleList([
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
])
```

**Decoder Attention**: `FlashMHADecoderBlock`
```python
# Standard Flash Multi-Head Attention decoder with cross-attention
self.dec_layers = nn.ModuleList([
    FlashMHADecoderBlock(
        embed_dim=self.embed_dim,
        ffn_embed_dim=4 * self.embed_dim,
        attention_heads=self.num_heads,
        add_bias_kv=False,
        dropout_p=self.dropout_p,
        use_bias=self.use_bias,
        layer_idx=l,
    )
    for l in range(num_decoder_layers)
])
```

### Multi-Sequence Capability Through Data Processing

Rather than using specialized attention modules, multi-sequence capabilities are achieved through:

1. **ESM Processing Mode**: `embed_x_per_chain` controls how sequences are processed
2. **Size-Based Masking**: Flexible attention masks from sequence size information
3. **Chain Boundary Preservation**: Maintained through tokenization and masking

### Advantages of Simplified Architecture

**Performance Benefits**:
- **Proven FlashAttention**: Uses optimized, well-tested attention implementations
- **Reduced Complexity**: Eliminates custom multi-sequence attention code
- **Better Maintainability**: Standard attention blocks are easier to debug and optimize

**Flexibility Benefits**:
- **Mode Switching**: Easy switching between single and multi-sequence modes
- **Standard Interface**: Compatible with existing transformer optimization techniques
- **Future Extensions**: Easy to incorporate new attention improvements

### Multi-Sequence Attention Through ESM Processing

The multi-sequence attention behavior is achieved implicitly through:

```python
# Per-chain ESM processing creates sequence-aware representations
def forward_per_chain(self, x: Tensor, x_sizes: Tensor):
    for seq_idx in range(num_sequences):
        # Process each chain separately
        seq_mask = _create_sequence_mask(x_sizes, sequence_idx=seq_idx)
        x_seq = x.clone()
        x_seq.masked_fill_(~seq_mask, self.vocab.pad_idx)

        # ESM encoding preserves chain structure
        output = self.esm(x_seq, repr_layers=[self.esm.num_layers])
        embedding = output["representations"][self.esm.num_layers]

        # Accumulate in combined embedding
        combined_embedding += embedding * seq_mask.unsqueeze(-1)
```

This approach provides multi-sequence modeling capabilities while using standard attention mechanisms.

---

## 8. Validation and Metrics

### PEINT Validation Strategy

**Primary Validation**: Standard evolutionary data split
**Secondary Validation**: **DMS Zero-Shot Evaluation**

**DMS Metrics Implementation**:
```python
# Correlation between model perplexities and experimental fitness scores
def compute_zero_shot_correlation(model, dms_data):
    perplexities = model.perplexity(x=parent_seqs, y=mutant_seqs, t=time)

    # Compute correlations
    spearman_corr = spearmanr(perplexities, experimental_fitness)
    pearson_corr = pearsonr(perplexities, experimental_fitness)

    return {"spearman": spearman_corr, "pearson": pearson_corr}
```

**Available Metrics**:
- `koenig_binding_heavy/spearman`: Spearman correlation for heavy chain binding
- `koenig_expression_heavy/pearson`: Pearson correlation for heavy chain expression
- Similar metrics for light chains across multiple DMS datasets

### PIPET Validation Strategy

**Current State**: Basic train/validation split only
```yaml
metrics: none  # No specialized validation metrics implemented
```

**Validation Gap**:
- No interaction-specific metrics
- No zero-shot evaluation for protein complexes
- Limited assessment of inter-chain modeling capabilities

**Potential Improvements**:
- Protein-protein interaction prediction benchmarks
- Complex formation energy correlation
- Interface residue contact prediction accuracy

---

## 9. Computational Complexity

### PEINT Computational Profile

**Attention Complexity**: `O(L²)` per sequence
**Memory Usage**: Linear in sequence length
**Optimization Features**:
- Encoder state caching for generation
- Batch perplexity computation
- Flash Attention integration

**Parallelization**: Standard batch processing

### PIPET Computational Profile

**Attention Complexity**: `O(L²)` with structured masking
**Memory Overhead**:
- Attention mask storage for variable sequence lengths
- Multi-sequence ESM computation
- Separate intra/inter attention parameter sets

**Computational Bottlenecks**:
- Per-sequence ESM encoding (not batched)
- Complex attention mask computations
- No encoder state caching during generation

**Optimization Opportunities**:
- Batch ESM computation across chains
- Precompute attention masks
- Implement encoder caching for PIPET generation

---

## 10. Key Architectural Insights

### 1. Unified Architecture Design
The new PEINT architecture represents a significant simplification:
```python
class PEINT(nn.Module):
    def __init__(self, enc_model: PretrainedEncoder, embed_x_per_chain: bool, ...):
        # Single model that handles both use cases
        self.enc_model = enc_model  # ESM processing abstraction
        self.enc_layers = nn.ModuleList([FlashMHAEncoderBlock(...)])  # Standard blocks
        self.dec_layers = nn.ModuleList([FlashMHADecoderBlock(...)])  # Standard blocks
```

**Benefits of Unified Design**:
- **Simplified Codebase**: Single model class instead of inheritance hierarchy
- **Consistent Interface**: Same forward pass signature for both modes
- **Easy Mode Switching**: Configuration-driven behavior changes
- **Reduced Maintenance**: Fewer specialized components to maintain

### 2. ESM Encoder Abstraction Innovation

The `PretrainedEncoder` abstraction provides clean separation of concerns:
- **Flexibility**: Easy to swap different pretrained models
- **Modularity**: ESM processing logic isolated from transformer architecture
- **Extensibility**: Can support future pretrained encoder types
- **Testing**: Easier to unit test individual components

### 3. Attention Mechanism Simplification

**Previous Approach**: Custom multi-sequence attention modules with complex masking
**New Approach**: Standard FlashMHA blocks with preprocessing-based multi-sequence support

**Advantages**:
- **Performance**: Proven FlashAttention optimizations
- **Reliability**: Well-tested standard attention implementations
- **Maintainability**: Eliminates complex custom attention code
- **Compatibility**: Works with existing transformer optimization techniques

### 4. Size-Based Interface Design

The transition from boolean masks to size-based interfaces provides:
- **Memory Efficiency**: Store lengths instead of full boolean tensors
- **Flexibility**: Dynamic sequence handling without fixed tensor shapes
- **Consistency**: Single interface pattern across the entire model
- **Scalability**: Efficient handling of variable-length multi-sequence data

### 5. Configuration-Driven Multi-Sequence Support

**Key Innovation**: `embed_x_per_chain` parameter controls processing mode
- **Single Source of Truth**: One parameter determines behavior across entire model
- **Easy Experimentation**: Simple config change switches between modes
- **Backward Compatibility**: Maintains existing single-sequence functionality
- **Future Extensions**: Framework for additional processing modes

### 6. Biological Structure Preservation

Multi-sequence capabilities are maintained through:
- **Chain-aware ESM processing**: Preserves protein boundaries during encoding
- **Size-based masking**: Maintains sequence structure information
- **Token-level control**: Chain break tokens preserve biological meaning

---

## Conclusion

The new unified PEINT architecture represents a significant architectural evolution that maintains the biological modeling capabilities while greatly simplifying the implementation:

### Key Improvements

1. **Architectural Simplification**: Single model class replaces inheritance hierarchy
2. **Standard Attention Blocks**: Eliminates custom multi-sequence attention modules
3. **ESM Abstraction**: Clean separation of pretrained encoder logic
4. **Size-Based Interface**: Efficient and flexible sequence handling
5. **Configuration-Driven**: Easy switching between processing modes

### Maintained Capabilities

1. **Single-Sequence Modeling**: Traditional protein evolution `P(x2|x1, t)`
2. **Multi-Sequence Modeling**: Protein interaction evolution `P(x2, y2|x1, y1, tx, ty)`
3. **ESM Integration**: Pretrained protein language model features
4. **Biological Structure**: Chain boundaries and protein complex topology

### Future Development Opportunities

1. **Performance Optimization**: Batch ESM computation for multi-sequence mode
2. **Validation Metrics**: Develop interaction-specific benchmarks
3. **Generation Enhancements**: Implement efficient constrained generation
4. **Encoder Extensions**: Support for additional pretrained models

This unified architecture provides a solid foundation for both single and multi-sequence protein evolution modeling while maintaining simplicity and extensibility.

---

*Last updated: September 2025*
*Analysis based on unified PEINT architecture with FlashMHA and ESM abstraction*
