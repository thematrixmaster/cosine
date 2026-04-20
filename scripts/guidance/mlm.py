"""
Guided Gibbs Sampling Baseline (Product of Experts)

Implements "Guided Sampling Humanization" (ArXiv:2412.04737) with a mutation constraint.
Samples residues from a Product of Experts: P(x) ~ P_MLM(x) * exp(lambda * Fitness(x))

Key Features:
- Proper Gibbs: Masks only 1 position at a time.
- Smart Constraints: Skips invalid sampling steps to prevent "freezing" at the boundary.
- Output: Matches HotFlip/GA CSV format exactly.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "evo"))

import argparse
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from oracle_counter import OracleCallCounter
from tqdm import tqdm

# Model Imports
from transformers import AutoTokenizer, EsmForMaskedLM

from evo.antibody import (
    compute_iglm_humanness,
    compute_oasis_humanness,
    create_region_masks,
)

# Oracle Imports
from evo.oracles import get_oracle

# ------------------------------------------------------------------------------
# Model Wrappers
# ------------------------------------------------------------------------------

AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")


class MLMOracleWrapper:
    def get_masked_logits(self, sequence: str, mask_idx: int) -> np.ndarray:
        raise NotImplementedError


class ESM2Wrapper(MLMOracleWrapper):
    def __init__(self, model_name: str, device: str):
        print(f"Loading ESM-2 model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
        self.model = EsmForMaskedLM.from_pretrained(f"facebook/{model_name}").to(device).eval()
        self.device = device
        self.vocab_map = {aa: self.tokenizer.convert_tokens_to_ids(aa) for aa in AA_VOCAB}

    def get_masked_logits(self, sequence: str, mask_idx: int) -> np.ndarray:
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        model_idx = mask_idx + 1  # Shift for CLS
        inputs["input_ids"][0, model_idx] = self.tokenizer.mask_token_id

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, model_idx]

        aa_logits = []
        for aa in AA_VOCAB:
            idx = self.vocab_map[aa]
            aa_logits.append(logits[idx].item())
        return np.array(aa_logits)


class AbLangWrapper(MLMOracleWrapper):
    def __init__(self, device: str):
        try:
            import ablang
        except ImportError:
            raise ImportError("AbLang not found. Please install via: pip install ablang")
        print("Loading AbLang (heavy chain)...")
        self.ablang = ablang.pretrained("heavy")
        self.ablang.freeze()
        self.tokenizer = self.ablang.tokenizer
        self.vocab_map = {aa: self.tokenizer.vocab_to_token[aa] for aa in AA_VOCAB}

    def get_masked_logits(self, sequence: str, mask_idx: int) -> np.ndarray:
        seq_list = list(sequence)
        seq_list[mask_idx] = "*"
        masked_seq = "".join(seq_list)

        res = self.ablang(masked_seq, mode="likelihood").squeeze(0)[1:-1]
        assert res.shape[0] == len(sequence), "Length mismatch in AbLang output."
        logits_full = res[mask_idx]

        aa_logits = []
        for aa in AA_VOCAB:
            idx = self.vocab_map.get(aa) - 1  # adjust for removed bos token at index 0
            if idx is not None:
                aa_logits.append(logits_full[idx])
            else:
                aa_logits.append(-100.0)
        return np.array(aa_logits)


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Guided Gibbs Sampling Baseline")

    # Oracle & Seed
    parser.add_argument(
        "--oracle", type=str, default="SARSCoV2Beta", choices=["SARSCoV1", "SARSCoV2Beta"]
    )
    parser.add_argument("--oracle-seed-idx", type=int, default=0)
    parser.add_argument(
        "--seed-seq",
        type=str,
        default=None,
        help="Custom seed sequence to optimize (overrides oracle-seed-idx if provided)",
    )

    # Model Params
    parser.add_argument("--mlm-model", type=str, default="esm2_t6_8M_UR50D")

    # Sampling Hyperparameters
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--sweeps", type=int, default=10, help="Increased sweeps for better convergence"
    )
    parser.add_argument("--guidance-strength", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Constraints
    parser.add_argument("--max-mutations", type=int, default=5)
    parser.add_argument(
        "--mask-region",
        type=str,
        default=None,
        choices=[
            "CDR1",
            "CDR2",
            "CDR3",
            "CDR_overall",
            "FR1",
            "FR2",
            "FR3",
            "FR4",
            "FR_overall",
            None,
        ],
        help="Restrict mutations to specific antibody region (default: None, allows all positions)",
    )

    # Tech
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-path", type=str, default=None)

    # New Flag for Local Fitness Approximation
    parser.add_argument(
        "--use-local-fitness",
        action="store_true",
        help="Precompute mutation scores from seed and reuse them (faster, assumes independence).",
    )

    return parser.parse_args()


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def load_oracle_binders(oracle_variant: str) -> pd.DataFrame:
    csv_files = {
        "SARSCoV1": "CovAbDab_heavy_binds SARS-CoV1.csv",
        "SARSCoV2Beta": "CovAbDab_heavy_binds SARS-CoV2_Beta.csv",
    }
    csv_path = Path("evo/evo/oracles/data") / csv_files[oracle_variant]
    if not csv_path.exists():
        csv_path = Path(__file__).parent.parent / "evo/evo/oracles/data" / csv_files[oracle_variant]
    df = pd.read_csv(csv_path)
    df.columns = ["sequence", "binding"]
    return df[df["binding"] == 1].reset_index(drop=True)


def select_oracle_seed(oracle_variant: str, seed_idx: int) -> str:
    binders = load_oracle_binders(oracle_variant)
    return binders.iloc[seed_idx]["sequence"]


def get_region_mask(sequence: str, region_name: str) -> np.ndarray:
    if region_name is None:
        return np.ones(len(sequence), dtype=bool)
    masks = create_region_masks(sequence, scheme="imgt")
    return masks[region_name]


def calculate_hamming_distance(seq1: str, seq2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


def calculate_internal_diversity(sequences: list[str]):
    """Calculate average pairwise Hamming distance within a list of sequences."""
    n = len(sequences)
    if n < 2:
        return 0.0, 0.0
    distances = []
    count = 0
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            distances.append(calculate_hamming_distance(sequences[i], sequences[j]))
            count += 1
    mean_distance = sum(distances) / count
    std_distance = np.std(distances)
    return mean_distance, std_distance


# ------------------------------------------------------------------------------
# Core Sampling Logic (Fixed)
# ------------------------------------------------------------------------------


def run_guided_gibbs_trajectory(
    start_seq: str,
    oracle,
    mlm_wrapper: MLMOracleWrapper,
    args,
    mask: np.ndarray,
    traj_idx: int,
    local_fitness_cache: dict = None,  # New argument for cached scores
) -> dict:

    current_seq = start_seq
    mutable_indices = np.where(mask)[0]

    for sweep in range(args.sweeps):
        # 1. Randomized Order
        np.random.shuffle(mutable_indices)

        for pos_idx in mutable_indices:
            # --- Check Constraints BEFORE Inference (The Fix) ---
            # Calculate current distance
            current_dist = calculate_hamming_distance(start_seq, current_seq)
            is_currently_mutated = current_seq[pos_idx] != start_seq[pos_idx]

            # If we are at the limit, we can ONLY touch positions that are already mutated.
            # Touching an unmutated position (and changing it) would go to Max+1 -> Reject.
            # So effectively, P(change) = 0 for unmutated positions when at limit.
            if current_dist >= args.max_mutations and not is_currently_mutated:
                continue

            # --- A. MLM Prior ---
            prior_logits = mlm_wrapper.get_masked_logits(current_seq, pos_idx)

            # --- B. Oracle Likelihood ---
            if local_fitness_cache is not None:
                # FAST PATH: Use precomputed scores relative to seed
                # This assumes independence (score depends only on mutation at pos_idx, ignoring background)
                scores = np.array(local_fitness_cache[pos_idx])
            else:
                # SLOW PATH: Compute exact fitness for current background
                candidates = []
                for aa in AA_VOCAB:
                    s_list = list(current_seq)
                    s_list[pos_idx] = aa
                    candidates.append("".join(s_list))

                # Batch score
                scores = oracle.predict_batch(candidates)[0]
                scores = np.array(scores)

            # --- C. Product of Experts ---
            combined_logits = (prior_logits / args.temperature) + (args.guidance_strength * scores)

            # Softmax
            combined_logits = combined_logits - np.max(combined_logits)
            probs = np.exp(combined_logits)
            probs = probs / probs.sum()

            # --- D. Sample ---
            chosen_idx = np.random.choice(len(AA_VOCAB), p=probs)
            proposed_aa = AA_VOCAB[chosen_idx]

            # Construct proposed sequence
            s_list = list(current_seq)
            s_list[pos_idx] = proposed_aa
            proposed_seq = "".join(s_list)

            # Final Safety Check (should barely ever trigger now due to logic above)
            dist = calculate_hamming_distance(start_seq, proposed_seq)
            if dist <= args.max_mutations:
                current_seq = proposed_seq
            # Else: Implicitly reject (keep current_seq)

    # Final Stats
    final_fitness = oracle.predict(current_seq, increment=False)[0]
    n_muts = calculate_hamming_distance(start_seq, current_seq)

    return {
        "seed_seq": start_seq,
        "sampled_seq": current_seq,
        "n_mutations": n_muts,
        "final_fitness": final_fitness,
        "trajectory_idx": traj_idx,
    }


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    seed = args.seed if args.seed is not None else int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"{'='*60}\nRunning Guided Gibbs Sampling (Product of Experts)\n{'='*60}")
    print(f"MLM: {args.mlm_model} | Sweeps: {args.sweeps} | Guidance: {args.guidance_strength}")
    print(f"Constraint: Max {args.max_mutations} mutations")
    print(f"Local Fitness Approx: {args.use_local_fitness}")

    if args.mlm_model.lower() == "ablang":
        mlm = AbLangWrapper(device=str(device))
    else:
        mlm = ESM2Wrapper(model_name=args.mlm_model, device=device)

    print(f"Loading Oracle: {args.oracle}")
    oracle = OracleCallCounter(
        get_oracle(
            args.oracle,
            enable_mc_dropout=True,
            mc_samples=10,
            mc_dropout_seed=42,
            use_iglm_weighting=False,
            device=device,
            precompute_seed_fitnesses=False,
        )
    )

    if args.seed_seq is not None:
        seed_seq = args.seed_seq
        print(f"Using custom seed sequence ({len(seed_seq)} aa)")
    else:
        seed_seq = select_oracle_seed(args.oracle, args.oracle_seed_idx)
    seed_fitness = oracle.predict(seed_seq, increment=False)[0]
    print(f"Seed Fitness: {seed_fitness:.4f}")

    mask = get_region_mask(seed_seq, args.mask_region)
    if args.mask_region:
        print(f"Constraining sampling to {args.mask_region} ({mask.sum()} residues)")

    # --------------------------------------------------------------------------
    # Precompute Local Fitness Map if requested
    # --------------------------------------------------------------------------
    local_fitness_cache = None
    if args.use_local_fitness:
        print("\nPrecomputing local fitness scores for all single mutations in mask...")
        local_fitness_cache = {}
        mutable_indices = np.where(mask)[0]

        # Prepare all single mutants for batch prediction
        all_candidates = []
        metadata = []  # stores (pos_idx, aa_idx_in_vocab)

        for pos_idx in mutable_indices:
            for i, aa in enumerate(AA_VOCAB):
                s_list = list(seed_seq)
                s_list[pos_idx] = aa
                cand_seq = "".join(s_list)
                all_candidates.append(cand_seq)
                metadata.append((pos_idx, i))

        # Batch predict (in one go, usually safe for small CDRs * 20 variants)
        # For larger sets, you might chunk this, but here it's typically < 400 sequences
        scores_flat = oracle.predict_batch(all_candidates)[0]

        # Structure into map: { pos_idx: [score_A, score_C, ...] }
        for (pos_idx, aa_idx), score in zip(metadata, scores_flat):
            if pos_idx not in local_fitness_cache:
                local_fitness_cache[pos_idx] = [0.0] * len(AA_VOCAB)
            local_fitness_cache[pos_idx][aa_idx] = score

        print(f"Precomputed {len(all_candidates)} scores.")

    results = []
    print(f"\nStarting {args.batch_size} trajectories...")
    start_time = time.time()

    for i in tqdm(range(args.batch_size)):
        res = run_guided_gibbs_trajectory(
            start_seq=seed_seq,
            oracle=oracle,
            mlm_wrapper=mlm,
            args=args,
            mask=mask,
            traj_idx=i,
            local_fitness_cache=local_fitness_cache,
        )
        res["delta_fitness"] = res["final_fitness"] - seed_fitness
        res["method"] = f"guided_gibbs_{args.mlm_model}"
        results.append(res)

    end_time = time.time()
    total_time = end_time - start_time
    time_per_sample = total_time / args.batch_size

    df = pd.DataFrame(results)

    # Add time column
    df["time"] = time_per_sample

    print("\nComputing metrics...")
    all_sequences = df["sampled_seq"].tolist()
    df["oasis_humanness"] = compute_oasis_humanness(all_sequences)
    df["iglm_humanness"] = compute_iglm_humanness(all_sequences)
    df["edit_dist_from_seed"] = df["n_mutations"]
    df["time"] = time_per_sample

    if args.output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = "esm2" if "esm" in args.mlm_model else "ablang"
        local_tag = "_local" if args.use_local_fitness else ""
        args.output_path = f"gibbs_{model_short}_{args.oracle}_seed{args.oracle_seed_idx}_sw{args.sweeps}_g{args.guidance_strength}{local_tag}_{ts}.csv"

    if args.save_csv:
        df.to_csv(args.output_path, index=False)

    # Final Summary
    n_seqs = args.batch_size
    avg_fitness = df["final_fitness"].mean()
    max_fitness = df["final_fitness"].max()
    unique_seqs = df["sampled_seq"].nunique()
    int_div_mean, int_div_std = calculate_internal_diversity(all_sequences)

    print(f"\n{'='*60}")
    if args.save_csv:
        print(f"Results saved to: {args.output_path}")
    print(f"Final Population Size: {len(df)}")
    print(f"Unique Sequences: {unique_seqs} / {args.batch_size}")
    print(f"Proportion Unique: {unique_seqs / args.batch_size:.4f}")
    print(f"Avg Fitness: {avg_fitness:.4f} +/- {df['final_fitness'].std():.4f}")
    print(f"Max Fitness: {max_fitness:.4f}")
    print(
        f"Avg OASIS Humanness: {df['oasis_humanness'].mean():.4f} +/- {df['oasis_humanness'].std():.4f}"
    )
    print(
        f"Avg IGLM Humanness: {df['iglm_humanness'].mean():.4f} +/- {df['iglm_humanness'].std():.4f}"
    )
    print(f"Delta Fitness (Avg): {avg_fitness - seed_fitness:.4f}")
    print(f"Delta Fitness (Max): {max_fitness - seed_fitness:.4f}")
    print(f"Internal Diversity (Avg Hamming Dist): {int_div_mean:.4f} +/- {int_div_std:.4f}")
    print(f"Avg Mutations: {df['n_mutations'].mean():.4f} +/- {df['n_mutations'].std():.4f}")
    print(f"Total Runtime: {total_time:.4f}s")
    print(f"Avg Time per Sample: {time_per_sample:.4f}s")
    print(f"Total oracle calls: {oracle.get_call_count()}")
    print(f"Avg oracle calls per sample: {oracle.get_call_count() / n_seqs:.4f}")


if __name__ == "__main__":
    main()
