from collections import defaultdict
from pathlib import Path

import os
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F

from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime

from peint.models.modules.ctmc_module import CTMCModule
from peint.models.nets.ctmc import NeuralCTMC, NeuralCTMCGenerator

from evo.tensor import collate_tensors
from evo.dataset import ComplexCherriesDataset, ComplexCherriesCollection
from evo.tokenization import Vocab

from tqdm import tqdm
tqdm.pandas()


def load_and_batchify_seed_sequences(vocab: Vocab, aho=True, number_of_seeds=5):
    df = pd.read_csv("/accounts/projects/yss/stephen.lu/peint-workspace/main/data/wyatt/aho/seed_aho_tuples.csv")
    heavy_seeds = df['fv_heavy_aho'].tolist()[:number_of_seeds]
    light_seeds = df['fv_light_aho'].tolist()[:number_of_seeds]

    if not aho: # remove gaps  if not using AHO alignment
        heavy_seeds = [seq.replace("-", "") for seq in heavy_seeds]
        light_seeds = [seq.replace("-", "") for seq in light_seeds]

    _xs = []
    _x_sizes = []
    for heavy_seed, light_seed in zip(heavy_seeds, light_seeds):
        combined_seed = f"{heavy_seed}.{light_seed}"
        x_sizes = torch.tensor([len(heavy_seed) + 1, len(light_seed) + 1], dtype=torch.long)
        x_sizes[0] += vocab.prepend_bos
        x_sizes[-1] += vocab.append_eos - 1
        xs = torch.from_numpy(vocab.encode_single_sequence(combined_seed))
        x_sizes = torch.nn.functional.pad(x_sizes, (0, len(xs) - len(x_sizes)), value=0)
        _xs.append(xs)
        _x_sizes.append(x_sizes)
    
    return collate_tensors(_xs), collate_tensors(_x_sizes), heavy_seeds, light_seeds


def load_model():
    # Aligned TR Rosetta Checkpoint
    # ckpt_dir = Path("/accounts/projects/yss/stephen.lu/peint-workspace/main/logs/train/runs/2025-12-12_03-54-53/checkpoints")
    # ckpt_path = ckpt_dir / "last.ckpt"

    # Antibody Jaffe Checkpoint
    ckpt_dir = Path("/accounts/projects/yss/stephen.lu/peint-workspace/main/logs/train/runs/2025-12-15_04-09-58/checkpoints")
    ckpt_path = ckpt_dir / "epoch_007.ckpt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    module = CTMCModule.load_from_checkpoint(ckpt_path, map_location=device, strict=False)
    module = module.eval()

    net: NeuralCTMC = module.net
    vocab: Vocab = net.vocab
    
    return net, vocab, device


def sample_for_t(generator: NeuralCTMCGenerator, xs: Tensor, x_sizes: Tensor, t: float, number_of_samples=100, method="matrix_exp"):
    sampled_heavy_chains = []
    sampled_light_chains = []

    batch_size = xs.size(0)
    hc_lens = x_sizes[:,0] - 2
    ts = torch.full((batch_size,), t, dtype=torch.float32, device=xs.device)

    for _ in tqdm(range(number_of_samples), desc=f"Sampling for t={t}"):
        with (torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16)):
            if method == "matrix_exp":
                y_decoded = generator.generate_with_independent_sites(t=ts, x=xs, x_sizes=x_sizes)
            elif method == "gillespie":
                y_decoded = generator.generate_with_fake_gillespie(t=ts, x=xs, x_sizes=x_sizes, use_scalar_steps=True, verbose=False)
            else:
                raise ValueError(f"Invalid method: {method}")

        # decode the sampled childsequences
        sim_child_seqs = [decode_sequence_from_toks(y_decoded[i].cpu().numpy()) for i in range(y_decoded.size(0))]
        sim_hv_seqs, sim_lt_seqs = zip(*[(seq[:hl], seq[hl+1:]) for seq, hl in zip(sim_child_seqs, hc_lens)])
        sampled_heavy_chains.extend(sim_hv_seqs)
        sampled_light_chains.extend(sim_lt_seqs)
    
    return sampled_heavy_chains, sampled_light_chains


def decode_sequence_from_toks(toks, skip_gap_tokens=False):
    tokens = []
    gap_idx = vocab.tokens_to_idx.get("-", -1)
    for tok in toks:
        if tok == vocab.bos_idx:
            continue
        if skip_gap_tokens and tok == gap_idx:
            continue
        if tok == vocab.eos_idx or tok == vocab.pad_idx:
            break
        tokens.append(vocab.token(tok))
    return "".join(tokens)    


def compute_hamming_distance(seq1, seq2):
    assert len(seq1) == len(seq2)
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


if __name__ == "__main__":

    N_SEEDS = 5
    METHOD = "matrix_exp"    # or matrix_exp
    N_SAMPLES_PER_SEED = 100
    TS = [0.0001, 0.001, 0.01]
    # TS = [5, 10, 20, 50]
    VERBOSE = False

    # Load the cached results directory where we will save all intermediate and final results
    paper_dir = Path("/scratch/users/stephen.lu/projects/protevo/paper")
    date = "antibody_samples"
    # date = datetime.now().strftime("%Y-%m-%d")
    results_dir = paper_dir / f"{date}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the model
    net, vocab, device = load_model()
    net = net.to(device)
    net = net.eval()
    
    # Load the generator
    generator = NeuralCTMCGenerator(neural_ctmc=net)
    
    # Load the seed aho sequences
    xs, x_sizes, heavy_seeds, light_seeds = load_and_batchify_seed_sequences(vocab, aho=True, number_of_seeds=N_SEEDS)
    xs = xs.to(device)
    x_sizes = x_sizes.to(device)

    # Sample from the model
    for t in TS:
        fv_heavy, fv_light = sample_for_t(
            generator,
            xs,
            x_sizes,
            t=t,
            number_of_samples=N_SAMPLES_PER_SEED,
            method=METHOD,
        )
        
        N_heavy_seeds = heavy_seeds * N_SAMPLES_PER_SEED
        N_light_seeds = light_seeds * N_SAMPLES_PER_SEED

        # save the samples to a csv file
        sample_df = pd.DataFrame({
            'fv_heavy_aho': fv_heavy,
            'fv_light_aho': fv_light,
            'fv_heavy_aho_seed': N_heavy_seeds,
            'fv_light_aho_seed': N_light_seeds
        })
        sample_df.to_csv(results_dir / f"{METHOD}_{N_SEEDS}_seeds_{N_SAMPLES_PER_SEED}_samples_{t}_time.csv", index=False)