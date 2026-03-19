# Oracle-Guided Gillespie Sampling - Implementation Summary

## What Was Implemented

### 1. **Oracle Uncertainty Estimation** (`evo/evo/oracles/covid_oracle.py`)

Added uncertainty quantification to the CovidOracle:

- **Methods Added:**
  - `predict_with_uncertainty(sequence)` → (mean, variance)
  - `predict_batch_with_uncertainty(sequences)` → (means, variances)
  - `_predict_with_mc_dropout(sequences)` → (means, variances) using dropout sampling

- **Two Uncertainty Modes:**
  - **MC Dropout**: Runs N forward passes with dropout enabled during inference to estimate epistemic uncertainty
  - **Fixed Variance**: Uses a constant σ² as a hyperparameter (fallback option)

- **Fixes Applied:**
  - Added `.float()` conversion for bfloat16 → float32 compatibility with numpy
  - Properly toggles model between train/eval modes for MC Dropout

### 2. **Guided Gillespie Sampling** (`peint/models/nets/ctmc.py`)

Implemented classifier-guided sampling for CTMCs based on the theoretical framework:

**Main Method:**
```python
generate_with_gillespie(
    x, t, x_sizes,
    use_guidance=True,
    oracle=oracle,
    guidance_strength=1.0,
    use_taylor_approx=False,
    ...
)
```

**Theory:**
Modifies the CTMC rate matrix using Equation 8:
```
Q_z[x_i, x_j] = [2 * P(y > μ(x_i) | x_j)]^γ * Q[x_i, x_j]
```

Where:
- `P(y > μ(x_i) | x_j) = 1 - Φ((μ(x_i) - μ(x_j)) / σ(x_j))`
- `Φ` is the standard normal CDF
- `γ` is the guidance strength (higher = stronger guidance)

**Two Guidance Methods:**

1. **Exact Guidance** (`_apply_exact_guidance`):
   - Evaluates oracle on all L×V single-mutant sequences
   - Most accurate but computationally expensive
   - ~O(L×V) oracle calls per Gillespie step

2. **Taylor Approximation** (`_apply_taylor_guidance`):
   - Uses finite differences to approximate oracle gradients
   - Linearizes predictions: `μ(x_j) ≈ μ(x_i) + ∇μ · Δx`
   - Currently still O(L×V) but could be optimized to sample fewer positions

**Helper Methods:**
- `_sequence_to_string(seq)`: Converts tensors to AA strings, filtering to standard 20 AAs
- `_sequences_to_strings(seqs)`: Batch conversion

### 3. **Analysis Scripts**

#### `scripts/sample_and_score.py`
**Purpose:** Baseline unconditional sampling analysis

**What it does:**
- Samples N sequences from CTMC starting from oracle seeds
- Fixed branch length (t=0.1)
- Scores all samples with oracle
- Computes Δ (sampled_score - seed_score)
- Generates comprehensive visualizations

**Results from Run:**
- Mean Δ: -0.35 ± 0.58
- 30.5% improved, 69.5% declined
- Shows CTMC prior doesn't strongly align with oracle

#### `scripts/compare_guided_vs_unguided.py`
**Purpose:** Compare guided vs unguided sampling

**What it does:**
- Runs both unguided and guided sampling from same seeds
- Uses Taylor approximation for guidance
- MC Dropout for oracle uncertainty
- Statistical comparison (t-test)
- Side-by-side visualizations

**Parameters:**
- Branch length: 0.1
- Guidance strength: γ=2.0
- MC Dropout samples: 10
- 50 samples per condition per seed

**Currently Running...**

### 4. **Documentation**

- `scripts/CTMC_GUIDANCE_README.md`: Complete usage guide
- `scripts/test_ctmc_guidance.sh`: Quick test script
- `scripts/test_guidance_logic.py`: Mathematical validation (✅ passes)

## Key Implementation Details

### Sequence Filtering
The CTMC vocabulary includes special tokens (`<cls>`, `<pad>`, etc.) but the oracle only accepts standard 20 amino acids. Solution:

```python
valid_aas = set("ARNDCQEGHILKMFPSTWYV")
filtered_seq = "".join([tok for tok in seq if tok in valid_aas])
```

### Autocast for Model Inference
The CTMC model uses FlashAttention which requires fp16/bf16:

```python
with (torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16)):
    y = generator.generate_with_gillespie(..., use_guidance=True, oracle=oracle, guidance_strength=1.0, use_taylor_approx=False)
```

### Oracle Device Compatibility
- Oracle now runs on CUDA (SRU CUDA issue fixed by user)
- Added `.float()` conversions for bfloat16 → float32 numpy compatibility
- MC Dropout properly toggles train/eval modes

### Guidance Computation
At each Gillespie step:
1. Evaluate oracle on parent: `μ(x_i), σ(x_i)`
2. Generate all single mutants: `{x_j : j ∈ [L×V]}`
3. Evaluate oracle on mutants: `μ(x_j), σ(x_j)` for all j
4. Compute guidance weights: `w_j = [2 * Φ((μ(x_j) - μ(x_i)) / σ(x_j))]^γ`
5. Scale rates: `Q_guided[i,j] = w_j * Q[i,j]`
6. Sample next mutation from modified Q

## Performance Characteristics

### Computational Cost

**Per Gillespie Step:**
- CTMC forward pass: 1 call
- Oracle evaluations (exact): L×V calls
  - For L=120, V=33: ~4000 oracle calls
  - With MC Dropout (10 samples): ~40,000 forward passes
- Oracle evaluations (Taylor): L×V calls (same, but could be optimized)

**Per Trajectory:**
- Typical steps: 50-100 for t=0.1
- Total oracle calls: 200,000-400,000 per trajectory
- Batching helps but still expensive

### Speed
- Unguided sampling: ~1-2 it/s (Gillespie steps)
- Guided sampling: Slower due to oracle overhead
- Oracle on CUDA is much faster than CPU

## Files Modified/Created

**Modified:**
1. `evo/evo/oracles/covid_oracle.py`
   - Added uncertainty methods
   - Fixed bfloat16 compatibility
   - MC Dropout implementation

2. `peint/models/nets/ctmc.py`
   - Added `generate_with_gillespie(use_guidance=True)`
   - Implemented exact and Taylor guidance
   - Sequence conversion helpers

**Created:**
1. `scripts/sample_and_score.py` - Baseline analysis ✅
2. `scripts/compare_guided_vs_unguided.py` - Main comparison (running)
3. `scripts/ctmc_guidance.py` - Production script (not tested)
4. `scripts/test_guidance_logic.py` - Math validation ✅
5. `scripts/test_guidance_working.py` - Simple integration test
6. `scripts/CTMC_GUIDANCE_README.md` - Documentation
7. `scripts/test_ctmc_guidance.sh` - Test runner
8. `scripts/IMPLEMENTATION_SUMMARY.md` - This file

## Validation Status

✅ **Passed:**
- Mathematical logic tests (guidance weight computation)
- Oracle uncertainty methods work correctly
- Sequence conversion and filtering
- Unconditional baseline sampling (established metrics)

🔄 **In Progress:**
- Guided vs unguided comparison (currently running)

⏳ **Not Yet Tested:**
- Exact guidance (too slow for quick testing)
- Production script `ctmc_guidance.py`
- Very long trajectories (t > 0.5)
- Multiple guidance strengths

## Next Steps

1. **Immediate:** Wait for comparison results to see if guidance improves scores
2. **Optimization:** Reduce oracle calls in Taylor approximation by sampling fewer positions
3. **Experimentation:** Try different guidance strengths (γ = 0.5, 1.0, 2.0, 5.0)
4. **Validation:** Test exact vs Taylor accuracy trade-off
5. **Production:** Find oracle with native uncertainty (not MC Dropout)
6. **Scale:** Test on longer evolutionary times (t = 0.5, 1.0, 2.0)

## Known Issues & Limitations

1. **MC Dropout Uncertainty:** Model not trained for uncertainty, may not be calibrated
2. **Computational Cost:** Exact guidance is very expensive (L×V oracle calls per step)
3. **Taylor Approximation:** Currently not faster than exact (needs optimization)
4. **Fixed Variance:** Would work better with oracle that has native uncertainty
5. **Sequence Length Mismatch:** Filtering special tokens can change sequence length

## Usage Example

```python
from evo.oracles import get_oracle
from peint.models.nets.ctmc import NeuralCTMCGenerator

# Load oracle with MC Dropout
oracle = get_oracle("SARSCoV1", enable_mc_dropout=True, mc_samples=10)

# Guided sampling
y = generator.generate_with_gillespie(
    x=seed_tensor,
    t=torch.tensor([0.1]),
    x_sizes=seq_lengths,
    use_guidance=True,
    oracle=oracle,
    guidance_strength=2.0,
    use_taylor_approx=True,
)
```

## Theoretical Foundation

Based on:
- Nisonoff et al. "Unlocking Guidance for Discrete State-Space Diffusion and Flow Models" (arXiv:2406.01572)
- Bayesian inference on CTMC transition densities
- Classifier guidance adapted for continuous-time discrete-state processes
- Normal likelihood approximation with adaptive threshold (parent's mean)

The key insight: As branch length → 0, the posterior transition density is completely defined by the conditional rate matrix Q_z, which can be computed via Bayes' theorem using only oracle evaluations at discrete states.
