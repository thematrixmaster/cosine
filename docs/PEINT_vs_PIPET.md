# PEINT vs PIPET: Comprehensive Training Pipeline Analysis

## Executive Summary

This document provides a detailed technical comparison between **PEINT** (Protein Evolution Inference Transformer) and **PIPET** (Protein Interaction Protein Evolution in Time) training pipelines. PEINT focuses on single-sequence evolution modeling `P(x2|x1, t)`, while PIPET extends this to protein-protein interaction evolution `P(x2, y2|x1, y1, tx, ty)`. The key architectural innovation in PIPET is the multi-sequence attention mechanism that can model both intra-protein and inter-protein dependencies.

---

## Table of Contents

1. [Model Architecture Differences](#1-model-architecture-differences)
2. [Data Processing and Tokenization](#2-data-processing-and-tokenization)
3. [Training Configuration Differences](#3-training-configuration-differences)
4. [Forward Pass Implementation](#4-forward-pass-implementation)
5. [Loss Computation](#5-loss-computation)
6. [Generation and Inference](#6-generation-and-inference)
7. [Attention Mechanism Analysis](#7-attention-mechanism-analysis)
8. [Validation and Metrics](#8-validation-and-metrics)
9. [Computational Complexity](#9-computational-complexity)
10. [Key Architectural Insights](#10-key-architectural-insights)

---

## 1. Model Architecture Differences

### PEINT Architecture
- **Base Class**: Direct `nn.Module` inheritance
- **Purpose**: Single-sequence protein evolution modeling
- **Encoder**: Standard `EncoderBlock` layers (3 layers default)
- **Decoder**: Standard `DecoderBlock` layers (3 layers default)
- **Attention**: Standard self-attention and cross-attention mechanisms
- **Constraint**: `num_encoder_layers >= num_decoder_layers`

**Key Components**:
```python
self.enc_layers = nn.ModuleList([
    EncoderBlock(embed_dim, ffn_embed_dim=4*embed_dim, ...)
    for l in range(num_encoder_layers)
])
```

### PIPET Architecture
- **Base Class**: Inherits from PEINT (`class PIPET(PEINT)`)
- **Purpose**: Multi-sequence protein interaction evolution
- **Encoder**: `MultiSequenceEncoderBlock` layers with configurable attention types
- **Decoder**: `MultiSequenceDecoderBlock` layers with attention type control

**Attention Type Options**:
- `"full"`: Full attention between all sequences
- `"intra_only"`: Attention only within each sequence
- `"intra_inter"`: **Decoupled intra and inter sequence attention (default for encoder)**

**Additional Parameters**:
```python
chain_break_token: str = "."           # Separator between protein chains
encoder_self_attn_type: str = "intra_inter"  # Multi-sequence attention type
decoder_self_attn_type: str = "full"   # Decoder attention configuration
```

**Key Components**:
```python
self.enc_layers = nn.ModuleList([
    MultiSequenceEncoderBlock(
        attention_heads=num_heads,
        self_attn_type=encoder_self_attn_type,
        ...
    )
    for i in range(num_encoder_layers)
])
```

---

## 2. Data Processing and Tokenization

### PEINT Data Pipeline

**Dataset Class**: `EncodedPEINTDataset`
- **Base**: Extends `EncodedCherriesDataset`
- **Input Format**: Single sequences `x1 → x2`
- **Masking Strategy**: MLM-style masking on input sequences
- **Time Encoding**: Single scalar time value `t` per sequence pair
- **Tokenization**: Standard sequence encoding: `<cls>sequence<eos>`

**Data Flow**:
```python
def __getitem__(self, index):
    x, y, t = super().__getitem__(index)
    x_src, x_tgt = mask_tensor(x, self.vocab, mask_prob=0.15)
    y_src, y_tgt = y[:-1], y[1:]  # Autoregressive shift
    return (x_src, x_tgt, y_src, y_tgt, t, x_pad_mask, y_pad_mask)
```

### PIPET Data Pipeline

**Dataset Class**: `EncodedPIPETDataset`
- **Base**: Extends `EncodedPIPETCherriesDataset` → `TorchWrapperDataset`
- **Input Format**: Multi-sequence pairs with chain separation
- **Chain Processing**: Explicit chain splitting and concatenation

**Chain Handling Logic**:
```python
def __getitem__(self, index):
    x, y, t = super().__getitem__(index)
    x_heavy, x_light = x.split(self.sep_token)  # Split by "."

    # Encode separately then concatenate
    x = torch.cat([x_heavy, x_light], dim=0)  # <cls>x1<eos><cls>x2<eos>

    # Create attention masks storing sequence lengths
    x_sizes[0], x_sizes[1] = x_heavy.size(0), x_light.size(0)
```

**Key Differences**:
1. **Sequence Structure**: PIPET maintains explicit chain boundaries
2. **Attention Masking**: Length-based masks instead of boolean masks
3. **Time Encoding**: Position-specific evolutionary distances
4. **Special Token Handling**: Preserves chain break tokens during masking

**Output Structure Comparison**:
```python
# PEINT Output
(x_src, x_tgt, y_src, y_tgt, t, x_pad_mask, y_pad_mask)

# PIPET Output
(x_src, x_tgt, y_src, y_tgt, distances, attn_mask_x_sizes, attn_mask_y_sizes)
```

---

## 3. Training Configuration Differences

### PEINT Configuration (`train_peint.yaml`)

```yaml
defaults:
  - override /data/dataset: peint
  - override /data/dataset@data.dataset_val: dms  # DMS validation
  - override /metrics: koenig_dms                 # Correlation metrics

net:
  num_heads: 20
  num_encoder_layers: 3
  num_decoder_layers: 3
  # Uses default attention mechanisms
```

**Validation Strategy**:
- Primary: Standard evolution data train/val split
- Secondary: **DMS zero-shot evaluation** for fitness correlation
- Metrics: Spearman/Pearson correlation with experimental data

### PIPET Configuration (`train_pipet.yaml`)

```yaml
defaults:
  - override /data/dataset: pipet
  - override /metrics: none        # No specialized metrics

net:
  num_heads: 20
  num_encoder_layers: 3
  num_decoder_layers: 3
  chain_break_token: "."
  decoder_self_attn_type: intra_inter  # Multi-sequence attention
```

**Validation Strategy**:
- Primary: Basic train/val split only
- **Gap**: No interaction-specific validation metrics

### Common Training Hyperparameters

Both models share identical optimization settings:
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

## 4. Forward Pass Implementation

### PEINT Forward Pass

**Single-Sequence Processing**:
```python
def forward(self, x, y, t, x_pad_mask, y_pad_mask):
    # Global time embedding
    h_y = self.embedding(y) + self.time_embedding(t).expand_as(h_y)

    # ESM encoding (single sequence)
    h_x = self._compute_language_model_representations(x)

    # Standard encoder-decoder architecture
    for enc_layer in self.enc_layers:
        h_x = enc_layer(h_x, x_padding_mask=x_pad_mask)

        # Cross-attention in decoder
        if decoder_active:
            h_y = dec_layer(x=h_y, y=h_x, **decoder_kwargs)
```

### PIPET Forward Pass

**Multi-Sequence Processing**:
```python
def forward(self, enc_in, dec_in, enc_attn_mask, dec_attn_mask, distances):
    # Position-wise time embedding
    distances_expanded = _expand_distances_to_seqlen(distances, dec_attn_mask)
    h_dec = self.embedding(dec_in) + self.time_embedding(distances_expanded)

    # Multi-sequence ESM encoding
    h_enc = self._compute_language_model_representations(enc_in, enc_attn_mask)

    # Multi-sequence encoder-decoder
    for enc_layer in self.enc_layers:
        h_enc = enc_layer(h_enc, enc_attn_mask)  # MultiSequenceEncoderBlock
```

**Key Implementation Differences**:

1. **Time Embedding Strategy**:
   - **PEINT**: Global time value broadcast to all positions
   - **PIPET**: Position-specific evolutionary distances per sequence

2. **ESM Integration**:
   - **PEINT**: Direct ESM encoding of full sequence
   - **PIPET**: Sequence-aware ESM encoding with attention mask handling

3. **Attention Mask Format**:
   - **PEINT**: Boolean masks (`True`/`False` for attend/don't attend)
   - **PIPET**: Length-based masks (stores actual sequence lengths)

---

## 5. Loss Computation

### PEINT Loss Implementation
```python
def model_step(self, batch):
    [x, x_targets, y, y_targets, t, x_pad_mask, y_pad_mask] = batch
    x_logits, y_logits = self(x, y, t, x_pad_mask, y_pad_mask)

    mlm_loss = self.criterion(x_logits.transpose(-1, -2), x_targets)
    tlm_loss = self.criterion(y_logits.transpose(-1, -2), y_targets)

    loss = mlm_loss + tlm_loss
    return {
        "mlm_loss": mlm_loss, "tlm_loss": tlm_loss,
        "mlm_ppl": torch.exp(mlm_loss), "tlm_ppl": torch.exp(tlm_loss)
    }
```

### PIPET Loss Implementation
```python
def model_step(self, batch):
    [enc_inputs, enc_targets, dec_inputs, dec_targets,
     distances, attn_mask_enc, attn_mask_dec] = batch

    outputs = self(enc_inputs, dec_inputs, attn_mask_enc, attn_mask_dec, distances)
    enc_logits, dec_logits = outputs["enc_logits"], outputs["dec_logits"]

    mlm_loss = self.criterion(enc_logits.transpose(-1, -2), enc_targets)
    tlm_loss = self.criterion(dec_logits.transpose(-1, -2), dec_targets)

    loss = mlm_loss + tlm_loss
    return identical_metrics_as_PEINT
```

**Similarity**: Both use identical dual-loss formulation (MLM + TLM)
**Difference**: PIPET processes structured multi-sequence data with attention masks

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

## 7. Attention Mechanism Analysis

### PEINT Attention Architecture

**Encoder Attention**: Standard self-attention within sequences
```python
# Standard transformer encoder block
class EncoderBlock:
    def forward(self, x, x_padding_mask):
        # Self-attention within sequence
        attn_output = self.self_attn(x, x, x, key_padding_mask=x_padding_mask)
```

**Decoder Attention**: Causal self-attention + cross-attention
```python
# Standard transformer decoder block
class DecoderBlock:
    def forward(self, x, y, x_padding_mask, y_padding_mask):
        # Causal self-attention
        x = self.self_attn(x, x, x, causal=True)
        # Cross-attention to encoder
        x = self.cross_attn(x, y, y, key_padding_mask=y_padding_mask)
```

**Limitation**: Cannot effectively model inter-protein interactions

### PIPET Attention Architecture

**Multi-Sequence Encoder Attention**: Three configurable modes

1. **`intra_only`**: Attention only within each protein chain
```python
# Restricts attention to same-chain positions only
mask = create_intra_chain_mask(attn_mask_in_length)
```

2. **`full`**: Full attention between all positions across chains
```python
# Standard attention across entire concatenated sequence
attn_weights = softmax(QK^T / sqrt(d_k))
```

3. **`intra_inter`**: **Decoupled intra-chain and inter-chain attention**
```python
class DecoupledIntraInterMultiSequenceSelfAttention:
    def forward(self, hidden_states, attention_mask_in_length):
        # Separate parameter sets for intra vs inter attention
        attn_intra = self.compute_attention_weights(..., which_attn="intra")
        attn_inter = self.compute_attention_weights(..., which_attn="inter")

        # Combine outputs
        return attn_intra + attn_inter
```

**Multi-Sequence Decoder Attention**: Similar tri-modal architecture for generation

**Advantages**:
- **Biological Relevance**: Separate modeling of intra-protein vs inter-protein dependencies
- **Flexibility**: Allows ablation studies on interaction importance
- **Scalability**: Efficient masking for variable-length sequences

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

### 1. Inheritance and Code Reuse
**PIPET extends PEINT** through selective component overriding:
```python
class PIPET(PEINT):
    def __init__(self, ...):
        super(PIPET, self).__init__(...)  # Inherit base functionality

        # Override only encoder/decoder layers
        self.enc_layers = nn.ModuleList([MultiSequenceEncoderBlock(...)])
        self.dec_layers = nn.ModuleList([MultiSequenceDecoderBlock(...)])
```

This design pattern enables:
- **Maximum code reuse** between single and multi-sequence models
- **Consistent training infrastructure** (optimizers, logging, etc.)
- **Easy ablation studies** by switching between attention types

### 2. Attention Mechanism Innovation

The **`intra_inter` attention mechanism** represents a key biological insight:
- **Intra-chain attention**: Models folding and local structure
- **Inter-chain attention**: Models binding interfaces and allostery
- **Decoupled parameters**: Allows independent optimization of each interaction type

### 3. Data Complexity Trade-offs

**PEINT**: Simple, efficient single-sequence processing
**PIPET**: Complex multi-sequence handling with biological structure preservation

The complexity increase enables modeling protein complexes but requires:
- More sophisticated data preprocessing
- Complex attention masking logic
- Careful sequence boundary management

### 4. Validation Strategy Differences

**PEINT's DMS validation** provides:
- Real experimental validation
- Quantitative fitness correlation metrics
- Zero-shot evaluation capabilities

**PIPET's validation gap** represents an opportunity:
- Need for interaction-specific benchmarks
- Complex formation energy predictions
- Interface contact prediction metrics

### 5. Generation Constraints and Biological Validity

**PIPET's constrained generation** enforces biological validity:
```python
# Prevent invalid sequences
if current_chain == 0:
    logits[eos_idx] = -inf    # First chain cannot end early
elif current_chain == 1:
    logits[sep_idx] = -inf    # Second chain cannot have chain breaks
```

This ensures generated sequences maintain proper protein complex structure.

---

## Conclusion

PIPET represents a sophisticated architectural extension of PEINT, specifically designed for protein-protein interaction evolution modeling. The key innovations are:

1. **Multi-sequence attention mechanisms** that capture both intra-chain and inter-chain dependencies
2. **Structured data processing** that preserves protein complex topology
3. **Constrained generation** that ensures biological validity

While PIPET adds significant complexity, it enables modeling of protein interaction evolution that is impossible with single-sequence approaches. The main areas for improvement are:

1. **Computational optimization** (encoder caching, batched ESM computation)
2. **Validation metrics** (interaction-specific benchmarks)
3. **Generation efficiency** (faster multi-chain decoding)

This analysis provides a foundation for future development and optimization of both modeling approaches.

---

*Last updated: January 2025*
*Analysis based on PEINT/PIPET codebase version with multi-sequence attention implementation*
