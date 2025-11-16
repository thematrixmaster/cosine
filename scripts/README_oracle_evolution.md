# Oracle-Guided Evolutionary Optimization

This script performs iterative genetic algorithm optimization using PEINT generative model and protein fitness oracles.

## Overview

The script implements a simple evolutionary algorithm that:
1. Starts with oracle seed sequences
2. Samples children sequences using PEINT with different evolutionary branch lengths
3. Scores all sequences with an oracle
4. Selects top sequences for the next generation
5. Repeats for multiple generations
6. Tracks and visualizes fitness improvement over time

## Usage

### Basic Command

```bash
python scripts/oracle_guided_evolution.py \
    --oracle clone \
    --generations 5 \
    --samples-per-parent 10 \
    --top-k 10 \
    --device cuda
```

### Arguments

- `--oracle`: Oracle to use for fitness evaluation
  - Choices: `clone`, `rand_0.0`, `rand_0.5`, `rand_1.0`, `SARSCoV1`, `SARSCoV2Beta`
  - Default: `clone`

- `--generations`: Number of generations to evolve
  - Default: 5

- `--samples-per-parent`: Number of children to sample from each parent per generation
  - Default: 10

- `--top-k`: Number of top sequences to keep each generation
  - Default: 10

- `--checkpoint`: Path to PEINT checkpoint
  - Default: `/accounts/projects/yss/stephen.lu/peint/logs/train/runs/2025-11-01_03-40-52/checkpoints/epoch_031.ckpt`

- `--device`: Device to use (`cuda` or `cpu`)
  - Default: `cuda`
  - Note: PEINT requires CUDA (uses flash-attention)

- `--branch-length-min`: Minimum branch length for sampling
  - Default: 1.0

- `--branch-length-max`: Maximum branch length for sampling
  - Default: 5.0

- `--output-dir`: Directory to save results
  - Default: `results/oracle_evolution`

### Example: Different Oracles

```bash
# LLM Oracle (lower is better)
python scripts/oracle_guided_evolution.py --oracle clone --generations 5

# Random Oracle with alpha=0.5
python scripts/oracle_guided_evolution.py --oracle rand_0.5 --generations 5

# COVID Oracle (higher is better)
python scripts/oracle_guided_evolution.py --oracle SARSCoV1 --generations 5
```

### Example: Longer Run

```bash
# More generations with more diversity
python scripts/oracle_guided_evolution.py \
    --oracle clone \
    --generations 10 \
    --samples-per-parent 20 \
    --top-k 15 \
    --branch-length-min 0.5 \
    --branch-length-max 10.0
```

## Algorithm Details

### Evolutionary Process

**Generation 0 (Initialization):**
- Start with oracle seed sequences (usually 2 sequences)
- Evaluate fitness of seed sequences

**Generation 1-N:**
1. **Sampling Phase:**
   - For each sequence in current population
   - Sample N children using PEINT generator
   - Branch lengths sampled uniformly from [min_t, max_t]
   - Total candidates = parents + all children

2. **Evaluation Phase:**
   - Score all candidates with oracle
   - Oracle evaluates heavy chain sequences only

3. **Selection Phase:**
   - Sort by fitness (direction depends on oracle)
   - Keep top-K sequences for next generation
   - Parents can be retained if they're still top-K

### PEINT Generation

The script uses PEINT to generate child sequences:
- **Input:** Parent heavy chain sequence
- **Process:**
  - Adds dummy light chain (PEINT requires paired sequences)
  - Samples children with different branch lengths
  - Extracts heavy chain from generated paired sequences
- **Output:** Child heavy chain sequences

Branch length controls evolutionary distance:
- Smaller branch lengths → children closer to parent
- Larger branch lengths → more divergent children

### Oracle Evaluation

Sequences are evaluated using the specified oracle:
- **LLM Oracle (`clone`)**: Cross-entropy loss, lower is better
- **Random Oracle (`rand_X`)**: Normalized loss, lower is better
- **COVID Oracles**: Neutralization logits, higher is better

Only heavy chain sequences are evaluated (as oracles are heavy-chain-specific).

## Output Files

The script generates several output files in the specified output directory:

### 1. Generation Statistics
**File:** `{oracle_name}_generation_stats.csv`

Contains statistics for each generation:
- `generation`: Generation number (0-N)
- `mean_fitness`: Mean fitness of population
- `std_fitness`: Standard deviation of fitness
- `best_fitness`: Best fitness in population
- `worst_fitness`: Worst fitness in population
- `population_size`: Number of sequences in population

### 2. All Sequences
**File:** `{oracle_name}_all_sequences.csv`

Contains all sequences ever sampled:
- `generation`: Generation when sampled
- `sequence`: Amino acid sequence
- `fitness`: Oracle fitness score

Useful for analyzing diversity and convergence.

### 3. Final Population
**File:** `{oracle_name}_final_population.csv`

Contains the final evolved population:
- `sequence`: Final sequences
- `fitness`: Final fitness scores

### 4. Fitness Evolution Plot
**File:** `{oracle_name}_fitness_evolution.png`

Two-panel figure showing:
- **Left:** Mean fitness over generations (with std deviation)
- **Right:** Best/worst/mean fitness over generations

## Implementation Details

### Sequence Format

- **Oracle seeds:** Heavy chain sequences (~132 amino acids)
- **PEINT input:** Paired sequences (heavy.light) with separator
- **PEINT output:** Paired sequences, heavy chain extracted
- **Oracle evaluation:** Heavy chain only

### Dummy Light Chain

Since PEINT was trained on paired sequences but oracles expect heavy chain only:
- Script adds constant dummy light chain: `DIQMTQSPSSLSASVGDRVTITC`
- PEINT generates paired sequences
- Heavy chain extracted before oracle evaluation
- Results reflect heavy chain fitness only

### Tensor Formats

PEINT requires specific tensor formats:
- `xs`: (batch, seq_len) - tokenized sequences
- `x_sizes`: (batch, seq_len) - chain sizes in first positions
- `y_sizes`: (batch, seq_len) - target chain sizes
- `chain_ids`: (batch, num_chains) - chain identifiers [1, 2]
- `ts`: (batch, num_chains) - branch lengths per chain

## Expected Results

### Typical Behavior

**Early Generations (1-3):**
- High diversity in population
- Mix of good and bad sequences
- Fitness may vary widely

**Middle Generations (3-5):**
- Population converges toward better sequences
- Diversity decreases
- Mean fitness improves

**Late Generations (5+):**
- Population stabilizes
- Best fitness plateaus
- Less improvement per generation

### Example Output

```
Generation 0
================================================================================
Population size: 2
Fitness stats:
  Mean: 0.339682
  Std:  0.111152
  Best: 0.228530

Generation 5
================================================================================
Population size: 10
Fitness stats:
  Mean: 0.245821
  Std:  0.015432
  Best: 0.221103
  Improvement: 0.007427 (lower is better)
```

## Tips for Best Results

### Population Size
- Larger `--top-k` → more diversity, slower convergence
- Smaller `--top-k` → faster convergence, risk of local minima

### Sampling Rate
- More `--samples-per-parent` → better exploration
- Fewer samples → faster runtime

### Branch Length
- Shorter range → local search, fine-tuning
- Longer range → global search, exploration

### Recommended Settings

**Fast Exploration:**
```bash
--generations 5 --samples-per-parent 5 --top-k 5 \
--branch-length-min 2.0 --branch-length-max 8.0
```

**Deep Optimization:**
```bash
--generations 10 --samples-per-parent 20 --top-k 15 \
--branch-length-min 0.5 --branch-length-max 5.0
```

**Fine-tuning:**
```bash
--generations 8 --samples-per-parent 15 --top-k 8 \
--branch-length-min 0.5 --branch-length-max 2.0
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `--samples-per-parent`
- Process parents sequentially instead of batched
- Use smaller checkpoint

### Slow Generation
- Check GPU utilization
- Reduce `--samples-per-parent`
- Use shorter sequences

### No Improvement
- Increase branch length range
- Increase `--samples-per-parent`
- Try different oracle
- Check oracle direction (higher vs lower is better)

## Future Extensions

Potential improvements to the algorithm:
- **Adaptive branch lengths:** Decrease over generations
- **Diversity maintenance:** Penalize similar sequences
- **Multi-objective:** Optimize multiple oracles simultaneously
- **Elitism:** Always keep best sequence from previous generation
- **Mutation:** Add random mutations to increase diversity
- **Crossover:** Combine parent sequences

## Citation

If you use this script, please cite:
- PEINT model and training framework
- Oracle models (CloneBO)
- Evolutionary optimization methodology
