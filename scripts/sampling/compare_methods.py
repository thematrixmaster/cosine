"""Compare CTMC sampling methods with oracle scoring.

Samples sequences using different guidance methods (unguided, exact_guided,
taylor_guided) across one or multiple branch lengths, scores with an oracle,
and saves publication-quality figures + CSVs.

Usage:
    # Compare guided vs unguided at a single branch length
    uv run python scripts/sampling/compare_methods.py \
        --model-path checkpoints/model.ckpt \
        --branch-lengths 0.5

    # Sweep over multiple branch lengths (Taylor vs Exact comparison)
    uv run python scripts/sampling/compare_methods.py \
        --model-path checkpoints/model.ckpt \
        --branch-lengths 0.1 0.5 1.0 \
        --methods exact_guided taylor_guided
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats

from cosine.models.modules.ctmc_module import CTMCModule
from cosine.models.nets.ctmc import NeuralCTMC, NeuralCTMCGenerator
from evo.oracles import CovidOracle, get_oracle
from evo.tokenization import Vocab

METHOD_COLORS = {
    "unguided": "#0173B2",
    "exact_guided": "#E69F00",
    "taylor_guided": "#029E73",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare CTMC sampling methods across branch lengths"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to CTMC model checkpoint"
    )
    parser.add_argument(
        "--branch-lengths",
        nargs="+",
        type=float,
        default=[0.5],
        help="Branch length(s) to test (default: 0.5)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["unguided", "exact_guided", "taylor_guided"],
        default=["unguided", "taylor_guided"],
        help="Methods to compare (default: unguided taylor_guided)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples per (branch_length, method, seed) (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for sampling (default: same as num-samples)",
    )
    parser.add_argument(
        "--exact-batch-size",
        type=int,
        default=5,
        help="Batch size for exact_guided (default: 5, it is slow)",
    )
    parser.add_argument(
        "--guidance-strength",
        type=float,
        default=2.0,
        help="Guidance strength γ for guided methods (default: 2.0)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Number of seed sequences to use (default: all available)",
    )
    parser.add_argument(
        "--num-mc-samples",
        type=int,
        default=10,
        help="Number of MC samples for oracle uncertainty (default: 10)",
    )
    parser.add_argument(
        "--oracle-chunk-size",
        type=int,
        default=5000,
        help="Max sequences per oracle call to avoid OOM (default: 5000)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="SARSCoV1",
        choices=["SARSCoV1", "SARSCoV2Beta"],
        help="Oracle variant (default: SARSCoV1)",
    )
    parser.add_argument(
        "--humanness",
        action="store_true",
        help="Compute OASIS humanness scores for antibody sequences",
    )
    parser.add_argument(
        "--figure-format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output figure format (default: pdf)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for raster formats (default: 300)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/sampling",
        help="Output directory for results (default: ./results/sampling)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: uses timestamp)"
    )
    return parser.parse_args()


def sample_sequences(
    generator,
    vocab,
    x,
    t,
    x_sizes,
    method,
    num_samples,
    batch_size,
    oracle,
    guidance_strength,
    oracle_chunk_size,
    device,
):
    """Sample sequences using the specified method. Returns (sequences, elapsed_seconds)."""
    sampled_seqs = []
    start_time = time.time()

    for batch_start in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - batch_start)
        x_batch = x.repeat(current_batch_size, 1)
        t_batch = t.repeat(current_batch_size)
        x_sizes_batch = x_sizes.repeat(current_batch_size)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if method == "unguided":
                y_batch = generator.generate_with_gillespie(
                    x=x_batch,
                    t=t_batch,
                    x_sizes=x_sizes_batch,
                    temperature=1.0,
                    no_special_toks=True,
                    max_decode_steps=1000,
                    use_scalar_steps=False,
                    verbose=False,
                )
            else:
                y_batch = generator.generate_with_gillespie(
                    x=x_batch,
                    t=t_batch,
                    x_sizes=x_sizes_batch,
                    oracle=oracle,
                    guidance_strength=guidance_strength,
                    temperature=1.0,
                    no_special_toks=True,
                    max_decode_steps=1000,
                    use_scalar_steps=False,
                    use_taylor_approx=(method == "taylor_guided"),
                    verbose=(batch_start == 0),
                    oracle_chunk_size=oracle_chunk_size,
                    use_guidance=True,
                )

        aa_set = set("ARNDCQEGHILKMFPSTWYV")
        for i in range(current_batch_size):
            y_str = "".join(
                vocab.token(idx.item()) for idx in y_batch[i] if vocab.token(idx.item()) in aa_set
            )
            sampled_seqs.append(y_str)

    return sampled_seqs, time.time() - start_time


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_fitness_boxplot(df, methods, branch_lengths, output_dir, fmt, dpi):
    """Boxplot of Δ fitness vs branch length (or per-method if single branch length)."""
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 11

    if len(branch_lengths) == 1:
        # Simple per-method boxplot
        fig, ax = plt.subplots(figsize=(3.0, 2.5))
        positions = np.arange(len(methods))
        bp = ax.boxplot(
            [df[df["method"] == m]["fitness_delta"].values for m in methods],
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=1.5, color="darkred"),
        )
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS.get(method, "#888888"))
            patch.set_alpha(0.7)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.set_xlabel("Method", fontsize=10)
        ax.set_ylabel("Δ Fitness", fontsize=10)
        ax.set_xticks(positions)
        ax.set_xticklabels([m.replace("_", " ").title() for m in methods], fontsize=9)
        ax.grid(True, alpha=0.2, linewidth=0.5)
    else:
        # Grouped boxplot by branch length
        plot_data = [
            {"Branch Length": bl, "Method": m.replace("_", " ").title(), "Δ Fitness": v}
            for bl in branch_lengths
            for m in methods
            for v in df[(df["branch_length"] == bl) & (df["method"] == m)]["fitness_delta"]
        ]
        plot_df = pd.DataFrame(plot_data)
        palette = {m.replace("_", " ").title(): METHOD_COLORS.get(m, "#888888") for m in methods}

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(
            data=plot_df,
            x="Branch Length",
            y="Δ Fitness",
            hue="Method",
            palette=palette,
            ax=ax,
            showfliers=False,
        )

        # Jittered individual points
        for i, bl in enumerate(branch_lengths):
            for j, method in enumerate(methods):
                method_name = method.replace("_", " ").title()
                subset = plot_df[
                    (plot_df["Branch Length"] == bl) & (plot_df["Method"] == method_name)
                ]
                if len(subset) > 0:
                    x_pos = i + (j - 0.5) * 0.25
                    ax.scatter(
                        np.random.normal(x_pos, 0.04, len(subset)),
                        subset["Δ Fitness"].values,
                        alpha=0.3,
                        s=20,
                        color=palette[method_name],
                        zorder=10,
                    )

        ax.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Branch Length", fontsize=13, fontweight="bold")
        ax.set_ylabel("Δ Fitness", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(pad=0.3)
    out = output_dir / f"fitness_comparison.{fmt}"
    plt.savefig(out, dpi=dpi, bbox_inches="tight", format=fmt)
    print(f"Saved {out}")
    plt.close()


def plot_runtime(runtime_df, methods, branch_lengths, output_dir, fmt, dpi):
    """Log-scale runtime comparison. Only produced when both exact and Taylor are run."""
    if "exact_guided" not in methods or "taylor_guided" not in methods:
        return

    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 11
    fig, ax = plt.subplots(figsize=(5, 4))

    agg = (
        runtime_df.groupby(["branch_length", "method"])["time_per_sample_seconds"]
        .mean()
        .reset_index()
    )
    x = np.arange(len(branch_lengths))
    width = 0.35
    colors = {
        "exact_guided": METHOD_COLORS["exact_guided"],
        "taylor_guided": METHOD_COLORS["taylor_guided"],
    }

    for offset, method, label in [
        (-width / 2, "exact_guided", "Exact"),
        (width / 2, "taylor_guided", "Taylor"),
    ]:
        times = [
            (
                agg[(agg["branch_length"] == bl) & (agg["method"] == method)][
                    "time_per_sample_seconds"
                ].values[0]
                if len(agg[(agg["branch_length"] == bl) & (agg["method"] == method)]) > 0
                else 0
            )
            for bl in branch_lengths
        ]
        bars = ax.bar(x + offset, times, width, label=label, color=colors[method], alpha=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.1,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    y_vals = [v for _, row in agg.iterrows() for v in [row["time_per_sample_seconds"]]]
    ax.set_yscale("log")
    ax.set_ylim(min(y_vals) * 0.5, max(y_vals) * 2.0)
    ax.set_xlabel("Branch Length", fontsize=13, fontweight="bold")
    ax.set_ylabel("Time per Sample (s, log scale)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{bl:.2f}" for bl in branch_lengths])
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3, which="both", axis="y")
    plt.tight_layout()

    out = output_dir / f"runtime_comparison.{fmt}"
    plt.savefig(out, dpi=dpi, bbox_inches="tight", format=fmt)
    print(f"Saved {out}")
    plt.close()


def plot_mutation_count(df, methods, output_dir, fmt, dpi):
    """Boxplot of mutation counts per method (single branch length only)."""
    if "num_mutations" not in df.columns:
        return
    sns.set_style("whitegrid")
    positions = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    bp = ax.boxplot(
        [df[df["method"] == m]["num_mutations"].values for m in methods],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=1.5, color="darkred"),
    )
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS.get(method, "#888888"))
        patch.set_alpha(0.7)
    ax.set_xlabel("Method", fontsize=10)
    ax.set_ylabel("Number of Mutations", fontsize=10)
    ax.set_xticks(positions)
    ax.set_xticklabels([m.replace("_", " ").title() for m in methods], fontsize=9)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    plt.tight_layout(pad=0.3)
    out = output_dir / f"num_mutations.{fmt}"
    plt.savefig(out, dpi=dpi, bbox_inches="tight", format=fmt)
    print(f"Saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    random_seed = args.seed if args.seed is not None else int(time.time())
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    branch_lengths = sorted(args.branch_lengths)
    methods = args.methods
    num_samples = args.num_samples
    default_batch_size = args.batch_size if args.batch_size is not None else num_samples

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bl_str = "_".join([f"{bl:.2f}".replace(".", "p") for bl in branch_lengths])
    method_str = "_".join([m.replace("_guided", "") for m in methods])
    exp_name = (
        f"{timestamp}_{args.variant}_bl{bl_str}_{method_str}_n{num_samples}_seed{random_seed}"
    )
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")

    # Load model
    print(f"Loading CTMC model from: {args.model_path}")
    module = CTMCModule.load_from_checkpoint(args.model_path, map_location=device, strict=False)
    net: NeuralCTMC = module.net
    vocab: Vocab = net.vocab
    net = net.eval().to(device)
    generator = NeuralCTMCGenerator(neural_ctmc=net)

    # Load oracle
    print(f"Loading oracle ({args.variant}) with MC Dropout ({args.num_mc_samples} samples)...")
    oracle: CovidOracle = get_oracle(
        args.variant,
        enable_mc_dropout=True,
        mc_samples=args.num_mc_samples,
        use_iglm_weighting=False,
        device=device,
    )

    # Select seed sequences
    all_seeds = list(oracle.seed_data.values())
    if args.num_seeds is not None:
        n = min(args.num_seeds, len(all_seeds))
        seed_sequences = [all_seeds[i] for i in sorted(random.sample(range(len(all_seeds)), n))]
        print(f"Using {n} of {len(all_seeds)} available seed sequences")
    else:
        seed_sequences = all_seeds
        print(f"Using all {len(all_seeds)} available seed sequences")

    if args.humanness:
        from evo.antibody import compute_oasis_humanness

        print("Computing humanness for seed sequences...")
        seed_h = compute_oasis_humanness([s["sequence"] for s in seed_sequences], chain="heavy")
        for i, h in enumerate(seed_h):
            seed_sequences[i]["humanness"] = h
        print(f"Seed humanness (mean): {np.mean(seed_h):.4f}")

    print(f"\n{'='*60}")
    print(f"Branch lengths:   {branch_lengths}")
    print(f"Methods:          {methods}")
    print(f"Samples/config:   {num_samples}")
    print(f"Guidance strength:{args.guidance_strength}")
    print(f"Seeds:            {len(seed_sequences)} | Random seed: {random_seed}")
    print(f"{'='*60}\n")

    def count_mutations(seq, seed):
        return sum(a != b for a, b in zip(seq, seed[: len(seq)]))

    all_results = []
    runtime_data = []
    total_start = time.time()

    for branch_length in branch_lengths:
        print(f"\n{'#'*60}\nBranch Length: {branch_length}\n{'#'*60}")

        for seed_idx, seed_data in enumerate(seed_sequences):
            seed_seq = seed_data["sequence"]
            seed_fitness = seed_data["fitness"]
            seed_humanness = seed_data.get("humanness")

            x = torch.tensor([vocab.tokens_to_idx[aa] for aa in seed_seq], device=device).unsqueeze(
                0
            )
            x_sizes = torch.tensor([len(seed_seq)], device=device)
            t = torch.tensor([branch_length], device=device)

            print(f"\nSeed {seed_idx}: fitness={seed_fitness:.4f}, len={len(seed_seq)}")

            for method in methods:
                batch_size = (
                    args.exact_batch_size if method == "exact_guided" else default_batch_size
                )
                print(f"  [{method}] sampling {num_samples} seqs (batch={batch_size})...")

                sampled_seqs, method_time = sample_sequences(
                    generator,
                    vocab,
                    x,
                    t,
                    x_sizes,
                    method,
                    num_samples,
                    batch_size,
                    oracle,
                    args.guidance_strength,
                    args.oracle_chunk_size,
                    device,
                )
                print(f"    {method_time:.1f}s ({method_time/num_samples:.2f}s/sample)")

                sampled_scores, _ = oracle.predict_batch(sampled_seqs)
                fitness_deltas = sampled_scores - seed_fitness
                mutations = [count_mutations(seq, seed_seq) for seq in sampled_seqs]

                humanness_scores = None
                humanness_time = 0.0
                if args.humanness:
                    from evo.antibody import compute_oasis_humanness

                    t0 = time.time()
                    humanness_scores = compute_oasis_humanness(sampled_seqs, chain="heavy")
                    humanness_time = time.time() - t0

                print(
                    f"    Δ fitness: {np.mean(fitness_deltas):+.4f} ± {np.std(fitness_deltas):.4f} "
                    f"| {(fitness_deltas > 0).mean()*100:.1f}% improved"
                )

                for i, (seq, score, fd, n_mut) in enumerate(
                    zip(sampled_seqs, sampled_scores, fitness_deltas, mutations)
                ):
                    row = {
                        "branch_length": branch_length,
                        "seed_idx": seed_idx,
                        "seed_seq": seed_seq,
                        "seed_fitness": seed_fitness,
                        "method": method,
                        "sampled_seq": seq,
                        "fitness": score,
                        "fitness_delta": fd,
                        "num_mutations": n_mut,
                        "guidance_strength": args.guidance_strength,
                    }
                    if args.humanness and humanness_scores is not None:
                        row["humanness"] = humanness_scores[i]
                        row["humanness_delta"] = humanness_scores[i] - (seed_humanness or 0.0)
                    all_results.append(row)

                runtime_data.append(
                    {
                        "branch_length": branch_length,
                        "seed_idx": seed_idx,
                        "method": method,
                        "num_samples": num_samples,
                        "sampling_time_seconds": method_time,
                        "humanness_time_seconds": humanness_time,
                        "time_per_sample_seconds": (method_time + humanness_time) / num_samples,
                    }
                )

    df = pd.DataFrame(all_results)
    runtime_df = pd.DataFrame(runtime_data)
    df.to_csv(exp_dir / "results.csv", index=False)
    runtime_df.to_csv(exp_dir / "runtime_data.csv", index=False)

    total_time = time.time() - total_start
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Summary statistics
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    summary = []
    for bl in branch_lengths:
        for method in methods:
            sub = df[(df["branch_length"] == bl) & (df["method"] == method)]
            if len(sub) == 0:
                continue
            rt = runtime_df[(runtime_df["branch_length"] == bl) & (runtime_df["method"] == method)]
            row = {
                "branch_length": bl,
                "method": method,
                "mean_fitness_delta": sub["fitness_delta"].mean(),
                "std_fitness_delta": sub["fitness_delta"].std(),
                "median_fitness_delta": sub["fitness_delta"].median(),
                "pct_improved": (sub["fitness_delta"] > 0).mean() * 100,
                "mean_mutations": sub["num_mutations"].mean(),
                "avg_time_per_sample": (
                    rt["time_per_sample_seconds"].mean() if len(rt) > 0 else float("nan")
                ),
            }
            summary.append(row)
            print(
                f"  bl={bl} | {method}: Δ={row['mean_fitness_delta']:+.4f} ± {row['std_fitness_delta']:.4f}, "
                f"{row['pct_improved']:.1f}% improved"
            )
    pd.DataFrame(summary).to_csv(exp_dir / "summary_stats.csv", index=False)

    # Statistical comparison (Taylor vs Exact)
    if "exact_guided" in methods and "taylor_guided" in methods:
        print(f"\n{'='*60}\nTAYLOR VS EXACT (two-sample t-test)\n{'='*60}")
        for bl in branch_lengths:
            exact_vals = df[(df["branch_length"] == bl) & (df["method"] == "exact_guided")][
                "fitness_delta"
            ]
            taylor_vals = df[(df["branch_length"] == bl) & (df["method"] == "taylor_guided")][
                "fitness_delta"
            ]
            if len(exact_vals) > 0 and len(taylor_vals) > 0:
                _, p = stats.ttest_ind(exact_vals, taylor_vals)
                d = (exact_vals.mean() - taylor_vals.mean()) / np.sqrt(
                    (exact_vals.std(ddof=1) ** 2 + taylor_vals.std(ddof=1) ** 2) / 2
                )
                sig = "p < 0.05" if p < 0.05 else "n.s."
                print(
                    f"  bl={bl}: exact={exact_vals.mean():+.4f}, taylor={taylor_vals.mean():+.4f}, "
                    f"p={p:.4e}, Cohen's d={d:.3f} ({sig})"
                )

    # Plots
    fmt, dpi = args.figure_format, args.dpi
    print(f"\nGenerating publication-quality figures ({fmt.upper()})...")
    plot_fitness_boxplot(df, methods, branch_lengths, exp_dir, fmt, dpi)
    plot_runtime(runtime_df, methods, branch_lengths, exp_dir, fmt, dpi)
    if len(branch_lengths) == 1:
        plot_mutation_count(df, methods, exp_dir, fmt, dpi)

    print(f"\nDone. All results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
