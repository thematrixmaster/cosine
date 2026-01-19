# Batched Sampling Implementation for CTMC Guided Gillespie

## Summary of Changes

Successfully implemented batched sampling for CTMC guided Gillespie algorithm with support for:
- **Unguided sampling** (baseline)
- **Exact guidance** (evaluates oracle on all L×V mutants)
- **Taylor approximation guidance** (uses oracle gradients via autograd)

## Key Changes

### 1. Fixed Batching Mechanism

**Problem**: Script was calling methods with non-existent `num_samples` parameter

**Solution**: Manual input replication for proper batching
```python
# Replicate seed sequence for batch processing
x_batch = x.repeat(current_batch_size, 1)  # (1, L) -> (batch_size, L)
t_batch = t.repeat(current_batch_size)      # (1,) -> (batch_size,)
x_sizes_batch = x_sizes.repeat(current_batch_size)

y_batch = generator.generate_with_adapted_gillespie(
    x=x_batch, t=t_batch, x_sizes=x_sizes_batch,
    ...
)
```

### 2. Added Taylor Approximation Method

Implemented third sampling mode using gradient-based guidance (TAG from Nisonoff et al. 2025).

### 3. Added Argparse Configuration

```bash
python compare_guided_vs_unguided.py \
    --methods unguided exact_guided taylor_guided \
    --num-samples 100 \
    --batch-size 50 \
    --branch-length 0.1 \
    --guidance-strength 2.0
```

### 4. Fixed Oracle Parameter Mismatch

Changed from `oracle_fn`/`oracle_object` to single `oracle` parameter.

### 5. Updated Visualization for 3 Methods

All plots now dynamically handle 1-3 methods with proper coloring.

## Files Modified

1. **`scripts/compare_guided_vs_unguided.py`** - Main changes
2. **`evo/evo/oracles/base.py`** - Added missing `Tuple` import

## Files NOT Modified

**`peint/models/nets/ctmc.py`** - Already supports batching via B dimension

## Usage Examples

### Quick Test
```bash
python scripts/compare_guided_vs_unguided.py \
    --methods unguided taylor_guided \
    --num-samples 10
```

### Full Comparison
```bash
python scripts/compare_guided_vs_unguided.py \
    --methods unguided exact_guided taylor_guided \
    --num-samples 100 \
    --batch-size 50
```

## Testing

**IMPORTANT**: Must run on GPU node (model too large for CPU)

```bash
# Small test to verify implementation
python scripts/compare_guided_vs_unguided.py --num-samples 10
```
