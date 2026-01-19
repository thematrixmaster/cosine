# PEINT Documentation

This directory contains technical documentation for the PEINT project.

## Contents

### Core Documentation
- **[peint.md](peint.md)** - Main PEINT model documentation and architecture details

### Guided Sampling Methods
- **[ctmc_guidance.md](ctmc_guidance.md)** - Overview of oracle-guided CTMC sampling methods
- **[tag_implementation.md](tag_implementation.md)** - Taylor Series Approximation Guidance (TAG) implementation details
- **[batched_sampling.md](batched_sampling.md)** - Batched sampling implementation notes

## Guided Sampling Overview

The project implements oracle-guided protein evolution using Continuous-Time Markov Chains (CTMC):

1. **Exact Guidance**: Evaluates oracle on all L×V single mutants per step
   - Most accurate but computationally expensive
   - Uses deduplication and caching for efficiency

2. **Taylor Series Approximation Guidance (TAG)**: Gradient-based guidance
   - Approximates mutant scores using first-order Taylor expansion
   - ~40x faster than exact guidance with minimal performance trade-off
   - Requires differentiable oracle

3. **Unguided Sampling**: Baseline CTMC sampling without oracle guidance

### Key Scripts

Main analysis scripts are located in `/scripts`:

- `compare_branch_lengths.py` - Compare exact vs TAG guidance across branch lengths
- `compare_guided_vs_unguided.py` - Compare guided vs unguided sampling
- `oracle_guided_evolution.py` - General oracle-guided evolution framework

### Performance Optimizations

Recent optimizations include:

1. **Deterministic MC Dropout**: Global fixed dropout masks for reproducible predictions
2. **LRU Cache**: Caches oracle predictions (default size: 10,000)
3. **Deduplication**: Eliminates redundant mutant evaluations (up to 90% reduction)
4. **Vectorized Operations**: 40x speedup in guidance weight application

### Oracle Implementation

COVID-19 neutralization oracle (`evo/oracles/covid_oracle.py`):
- Dynamic seed loading from CSV files
- MC Dropout for uncertainty estimation
- Efficient caching with LRU eviction
- Supports SARSCoV1 and SARSCoV2Beta variants
