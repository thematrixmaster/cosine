# Oracle-Guided Gillespie Sampling for CTMC

This implementation provides oracle-guided sampling for Continuous-Time Markov Chain (CTMC) models using the theoretical framework from protein evolution inference.

## Overview

The guided Gillespie sampling algorithm biases the CTMC rate matrix Q towards sequences with higher predicted binding affinity (or other properties) using classifier guidance. This allows the model to generate sequences that are more likely to have desired properties without retraining.

### Theory

The guidance is based on the following equation (Equation 8):

```
Q_z[x_i, x_j] = [2 * P(y > μ(x_i) | x_j)]^γ * Q[x_i, x_j]
```

Where:
- `Q[x_i, x_j]` is the base CTMC rate from state i to state j
- `μ(x_i)` is the oracle's mean prediction for sequence x_i
- `P(y > μ(x_i) | x_j)` is the probability that x_j has higher predicted property than x_i
- `γ` is the guidance strength parameter (higher = stronger guidance)

The probability is computed using a normal likelihood:
```
P(y > μ(x_i) | x_j) = 1 - Φ((μ(x_i) - μ(x_j)) / σ(x_j))
```

Where Φ is the standard normal CDF.

## Implementation Components

### 1. Oracle Uncertainty Estimation (`evo/evo/oracles/covid_oracle.py`)

Extended the `CovidOracle` class to support uncertainty estimation:

- **MC Dropout**: Runs multiple forward passes with dropout enabled to estimate epistemic uncertainty
  - Note: May not work properly if the model wasn't trained with dropout
  - Enable with `enable_mc_dropout=True`, configure with `mc_samples=N`

- **Fixed Variance**: Uses a constant variance σ² across all predictions
  - More stable and recommended for now
  - Configure with `fixed_variance=σ²`

**New Methods:**
- `predict_with_uncertainty(sequence: str) -> tuple[float, float]`
- `predict_batch_with_uncertainty(sequences: list[str]) -> tuple[np.ndarray, np.ndarray]`

### 2. Guided Gillespie Sampling (`peint/models/nets/ctmc.py`)

Use `generate_with_gillespie(use_guidance=True)` to enable oracle guidance in the `NeuralCTMCGenerator` class.

**Key Features:**
- Oracle evaluation at every Gillespie step
- Two guidance modes: exact and Taylor approximation
- Compatible with existing CTMC infrastructure

**Parameters:**
- `oracle_fn`: Function that takes list of sequences and returns (means, variances)
- `guidance_strength`: Guidance parameter γ (default: 1.0)
- `use_taylor_approx`: Use Taylor approximation (faster but less accurate)
- Other parameters match `generate_with_gillespie()`

### 3. Guidance Methods

#### Exact Guidance (`_apply_exact_guidance`)

Evaluates the oracle on all L×V possible single-mutant sequences at each step:

**Pros:**
- Accurate oracle evaluation for each possible mutation
- Directly implements the theoretical guidance formula

**Cons:**
- Computationally expensive: O(L×V) oracle calls per step
- For L=100, V=20: 2000 oracle evaluations per step

**When to use:** When accuracy is critical and computational budget allows

#### Taylor Approximation (`_apply_taylor_guidance`)

Uses finite differences to approximate the oracle gradient and linearize predictions:

```
μ(x_j) ≈ μ(x_i) + ∇μ(x_i) · Δx
```

**Pros:**
- Still requires L×V oracle calls (for gradient estimation)
- More accurate than assuming oracle is locally linear
- Could be optimized to sample fewer positions

**Cons:**
- Currently not much faster than exact (needs optimization)
- Less accurate than exact evaluation

**Future optimization:** Sample only a subset of positions for gradient estimation

### 4. Sampling Script (`scripts/ctmc_guidance.py`)

Complete command-line interface for running guided sampling experiments.

## Usage

### Basic Usage

```bash
# Run with default settings (fixed variance, exact guidance)
uv run python scripts/ctmc_guidance.py \
    --oracle_name SARSCoV1 \
    --guidance_strength 1.0 \
    --num_trajectories 10

# Use Taylor approximation for speed
uv run python scripts/ctmc_guidance.py \
    --oracle_name SARSCoV1 \
    --guidance_strength 2.0 \
    --use_taylor \
    --num_trajectories 20

# Try MC Dropout for uncertainty (may not work well)
uv run python scripts/ctmc_guidance.py \
    --oracle_name SARSCoV2Beta \
    --mc_dropout \
    --mc_samples 10 \
    --guidance_strength 1.5
```

### Command-Line Arguments

**Model & Oracle:**
- `--model_path`: Path to CTMC checkpoint (default: uses hardcoded path)
- `--oracle_name`: Oracle variant (`SARSCoV1` or `SARSCoV2Beta`)
- `--device`: Device to run on (`cuda` or `cpu`)

**Guidance Parameters:**
- `--guidance_strength`: Guidance parameter γ (default: 1.0, higher = stronger)
- `--use_taylor`: Use Taylor approximation (faster but approximate)
- `--mc_dropout`: Enable MC Dropout for uncertainty
- `--fixed_variance`: Fixed variance σ² (default: 1.0)
- `--mc_samples`: Number of MC Dropout samples (default: 10)

**Sampling Parameters:**
- `--num_trajectories`: Trajectories per seed sequence (default: 10)
- `--target_time`: Evolutionary time (default: 1.0)
- `--max_steps`: Maximum Gillespie steps (default: 100)
- `--use_scalar_steps`: Use fixed step size instead of exponential holding times

**Other:**
- `--output_path`: Output directory for results
- `--no_iglm`: Disable IgLM likelihood weighting
- `--seed`: Random seed (default: 42)
- `--verbose`: Print detailed progress

### Output

The script generates:

1. **CSV file** (`guided_sampling_results.csv`):
   - Columns: seed info, guided/unguided sequences, fitness scores, mutations
   - One row per trajectory

2. **Visualization** (`guided_sampling_results.png`):
   - Fitness distribution histogram
   - Fitness vs. number of mutations scatter plot

3. **Console output**:
   - Per-seed summary statistics
   - Overall comparison of guided vs. unguided sampling

## Example Output

```
Loading model from: .../checkpoints/epoch_001.ckpt
Oracle: SARSCoV1
Guidance strength: 1.0

Found 2 seed sequences
  Seed 0: fitness=-0.916, len=124
  Seed 1: fitness=-0.912, len=126

================================================================================
Processing Seed 0: QVQLQESGPGLVKPSETLSLTCTVSGG... (fitness=-0.916)
================================================================================

Seed 0 Summary:
  Seed fitness: -0.916
  Guided:   mean=-0.523, std=0.142
  Unguided: mean=-0.821, std=0.198
  Improvement: 0.298

================================================================================
Overall Summary
================================================================================
Total trajectories: 20

Guided sampling:
  Mean fitness: -0.612 ± 0.187
  Mean mutations: 12.3 ± 3.5

Unguided sampling:
  Mean fitness: -0.868 ± 0.213
  Mean mutations: 11.8 ± 3.2

Improvement: 0.256
```

## Performance Considerations

### Computational Cost

For a sequence of length L=100 with vocabulary size V=20:

**Per Gillespie step:**
- **Exact guidance**: L×V = 2000 oracle evaluations
- **Taylor approximation**: L×V = 2000 oracle evaluations (current implementation)
- **CTMC forward pass**: 1 evaluation

**Per trajectory:**
- Typical: 50-100 Gillespie steps
- Total oracle calls: 100,000 - 200,000 per trajectory

**Optimization opportunities:**
1. Batch oracle evaluations more efficiently
2. Sample fewer positions for Taylor approximation
3. Cache oracle predictions between steps
4. Use adaptive guidance (stronger early, weaker late)

### Oracle Performance

The SARS-CoV oracle is relatively fast:
- ~100-200 sequences/second on GPU
- Batched evaluation is efficient
- Main bottleneck is generating mutant sequences

## Important Notes

### MC Dropout Limitations

**WARNING:** MC Dropout may not provide meaningful uncertainty estimates because:
1. The oracle model was trained with dropout but used in `eval()` mode during normal inference
2. Enabling dropout during inference may not reflect true epistemic uncertainty
3. The dropout was not calibrated for uncertainty estimation

**Recommendation:** Use fixed variance for now. If you need proper uncertainty:
- Find an oracle with ensemble predictions
- Train a Bayesian neural network oracle
- Use a Gaussian process oracle
- Calibrate the variance using validation data

### Variance Selection

The `fixed_variance` parameter acts as a **guidance temperature**:
- **Lower variance (e.g., 0.1)**: More confident oracle → stronger guidance
- **Higher variance (e.g., 10.0)**: Less confident oracle → weaker guidance
- **Default (1.0)**: Moderate confidence

You can tune this parameter to control how much you trust the oracle vs. the CTMC prior.

### Guidance Strength

The `guidance_strength` (γ) parameter:
- `γ = 0`: No guidance (equivalent to unguided sampling)
- `γ = 1`: Standard guidance (linear scaling)
- `γ > 1`: Stronger guidance (exponential emphasis on oracle)
- `γ < 1`: Weaker guidance (dampened oracle influence)

Recommended values: 0.5 - 2.0

## Region-Restricted Sampling

**New Feature (Added):** `generate_with_gillespie()` now supports restricting mutations to specific sequence regions (e.g., CDR regions of antibodies).

### Overview

The `mask` parameter allows you to constrain which positions can mutate during Gillespie sampling. This is particularly useful for:
- **Antibody optimization**: Restrict mutations to CDR (Complementarity-Determining Region) loops while preserving framework regions
- **Protein engineering**: Mutate only active sites or binding pockets
- **Controlled evolution**: Focus evolutionary search on specific domains

### Key Properties

1. **Holding time uses ALL sites**: The waiting time between mutations is calculated using the full rate matrix (masked + unmasked positions)
2. **Mutation sampling uses ONLY masked sites**: The position and token of the mutation are sampled only from allowed positions
3. **Compatible with guidance**: Works seamlessly with both exact and Taylor approximation guidance

### Usage

```python
from evo.antibody import create_region_masks
import torch

# Get CDR region mask for an antibody sequence
region_masks = create_region_masks(antibody_sequence, scheme="imgt")
cdr_mask = region_masks['CDR_overall']  # boolean numpy array (L,)

# Convert to tensor and add batch dimension
cdr_mask_tensor = torch.from_numpy(cdr_mask).unsqueeze(0).to(device)  # (1, L)

# Sample with CDR-only mutations (unguided)
y = generator.generate_with_gillespie(
    x=x,
    t=t,
    x_sizes=x_sizes,
    mask=cdr_mask_tensor,  # Restrict mutations to CDR regions
)

# Sample with CDR-only mutations (guided)
y_guided = generator.generate_with_gillespie(
    x=x,
    t=t,
    x_sizes=x_sizes,
    oracle=oracle,
    use_guidance=True,
    guidance_strength=2.0,
    use_taylor_approx=True,
    mask=cdr_mask_tensor,  # Restrict mutations to CDR regions
)
```

### Mask Format

- **Shape**: `(B, L)` boolean tensor matching input `x` shape
- **True**: Position can mutate
- **False**: Position cannot mutate
- **Default**: `None` (no restriction, all positions can mutate)

### Available Region Masks

The `evo.antibody.create_region_masks()` function provides:

- **Individual regions**: `CDR1`, `CDR2`, `CDR3`, `FR1`, `FR2`, `FR3`, `FR4`
- **Combined masks**: `CDR_overall` (all CDR regions), `FR_overall` (all framework regions)

Example:
```python
region_masks = create_region_masks(sequence, scheme="imgt")

# Use only CDR3 (most variable region)
cdr3_mask = torch.from_numpy(region_masks['CDR3']).unsqueeze(0).to(device)

# Use all framework regions
fr_mask = torch.from_numpy(region_masks['FR_overall']).unsqueeze(0).to(device)

# Combine multiple regions manually
cdr1_and_cdr3 = region_masks['CDR1'] | region_masks['CDR3']
custom_mask = torch.from_numpy(cdr1_and_cdr3).unsqueeze(0).to(device)
```

### Error Handling

**Shape validation:**
```python
# Raises ValueError if mask shape doesn't match x shape
ValueError: mask shape (1, 100) doesn't match x shape (1, 120). Expected shape: (1, 120)
```

**All positions masked:**
```python
# Raises ValueError if no valid mutation positions
ValueError: Sequences at batch indices [0] have no valid mutation positions
(all positions are masked). Cannot sample mutations.
```

### Testing

Run the test script to verify region masking functionality:

```bash
uv run python scripts/test_region_masking.py
```

This script tests:
1. Unguided sampling without mask (baseline)
2. Unguided sampling with CDR mask
3. Guided sampling with CDR mask
4. Edge case: all positions masked (error handling)
5. Edge case: wrong mask shape (error handling)

Expected output: All mutations should occur only within masked regions.

### Performance Notes

- **No overhead when mask=None**: Original performance is unchanged
- **Minimal overhead with mask**: Mask expansion and filtering is efficient (single tensor operation)
- **Holding time unaffected**: Total exit rate calculation remains the same
- **Compatible with batching**: Each sequence in a batch can have a different mask

## Future Improvements

1. **Oracle Selection:**
   - Integrate oracles with native uncertainty estimates
   - Support ensemble-based oracles
   - Add calibration methods for variance

2. **Efficiency:**
   - Optimize Taylor approximation to sample fewer positions
   - Implement gradient caching between steps
   - Add adaptive guidance scheduling

3. **Validation:**
   - Benchmark against wet lab data
   - Compare with fine-tuning approaches
   - Analyze guidance trajectory dynamics

4. **Features:**
   - Multi-objective guidance (multiple oracles)
   - Adaptive region masking (change mask during sampling)
   - Trajectory visualization and analysis tools

## Citation

If you use this implementation, please cite:

```bibtex
@article{nisonoff2024guided,
  title={Unlocking Guidance for Discrete State-Space Diffusion and Flow Models},
  author={Nisonoff, Hunter and others},
  journal={arXiv preprint arXiv:2406.01572},
  year={2024}
}
```

## Troubleshooting

**Problem:** Oracle predictions don't improve with guidance
- Try increasing `guidance_strength` (e.g., 2.0, 5.0)
- Check that oracle is loaded correctly and predictions make sense
- Verify oracle `higher_is_better=True` for the property you want

**Problem:** Generated sequences have too many/few mutations
- Adjust `target_time` (higher = more mutations)
- Adjust `max_steps` if time budget not being reached
- Check `use_scalar_steps` setting

**Problem:** Out of memory errors
- Reduce batch size in oracle evaluation
- Process sequences one at a time instead of batching
- Use CPU for oracle if GPU memory limited

**Problem:** Very slow execution
- Use `--use_taylor` flag (though not much faster currently)
- Reduce `--num_trajectories` or `--max_steps`
- Ensure CUDA is being used for both model and oracle

## Contact

For questions or issues, please contact the repository maintainer or open an issue on GitHub.
