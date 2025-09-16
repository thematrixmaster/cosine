from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.distributions as dist
from clonebo.pools import get_oracle_and_pool
from evo.sequence import create_mutant_df

from peint.models.frameworks.peint import load_from_new_checkpoint


def sample_branch_lengths(rate=1, n_samples=64) -> torch.Tensor:
    """
    Sample branch lengths (ts) from an exponential distribution.
    :param mut_rate: Mutation rate.
    :param n_samples: Number of samples to draw.
    :return: Tensor of shape (n_samples,) containing sampled branch lengths.
    """
    ts = dist.Exponential(rate=rate).sample(sample_shape=(n_samples,))
    return torch.cumsum(ts, dim=0)


def load_peint_model():
    # ckpt_dir = Path("/scratch/users/milind_jagota/bcr/models/peint")
    # vh_ckpt_path = ckpt_dir / "heavy/epoch=28-step=12000.ckpt"
    # model = load_from_old_checkpoint(vh_ckpt_path)
    ckpt_dir = Path("/accounts/projects/yss/stephen.lu/protevo/plmr/logs/train/runs")
    vh_ckpt_path = ckpt_dir / "2025-08-08_15-36-04/checkpoints/epoch_089.ckpt"
    model = load_from_new_checkpoint(vh_ckpt_path)
    model = model.to(device="cuda")
    vocab = model.net.vocab
    return model, vocab


def load_clonebo_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("CloneBO/CloneLM-Heavy")
    tokenizer = AutoTokenizer.from_pretrained("CloneBO/CloneLM-Heavy")
    tokenizer.seq_sep_token = "[ClSep]"
    tokenizer.seq_sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.seq_sep_token)
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tokenizer


def main():
    # Load model
    # model, vocab = load_clonebo_model()
    model, vocab = load_peint_model()

    # Configure sampling parameters
    n_samples = 1
    mut_rate = 5  # expected duration between mutations

    # Load Sars-CoV-1 oracle and starting pool of labeled sequences
    (cost_func, labeled_seqs, labels, (start_mean, start_std), only_cdr) = get_oracle_and_pool(
        dict(oracle_name="SARSCoV1", start_ind=0, n_labelled_mut=2)
    )

    # Set and prepare start sequence
    seed_sequence = "".join(labeled_seqs[0].strip().split(" "))
    xs = torch.from_numpy(vocab.encode(seed_sequence))
    xs = xs.view(1, -1)

    # Sample branch lengths (ts) from an exponential distribution
    # ts = sample_branch_lengths(rate=mut_rate, n_samples=n_samples)  # (n_samples,)
    ts = torch.full((n_samples,), fill_value=5.0, device=xs.device)

    # Get all possible single site mutations
    mutants_df = create_mutant_df(seed_sequence)
    sequences = mutants_df["sequence"].unique().tolist()
    ys = torch.from_numpy(vocab.encode(sequences))  # (n_mutants, seq_len)

    # Broadcast ts and ys to match in shape
    ts = ts.unsqueeze(1).repeat(1, ys.shape[0]).view(-1, 1)  # (n_samples * n_mutants,)
    ys = (
        ys.unsqueeze(0).repeat(n_samples, 1, 1).view(-1, ys.shape[1])
    )  # (n_samples * n_mutants, seq_len)

    # Compute cost func for each mutant sequence
    mutant_labels = np.array([cost_func(seq) for seq in sequences])

    # Obtain sequence perplexities for each mutant y in ys under each (x, t) pair
    with torch.no_grad() and torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ppls = model.net.perplexity(
            x=xs.to(device=model.device),
            t=ts.to(device=model.device),
            y=ys.to(device=model.device),
            batch_size=64,
        )

    ppls = ppls.reshape(ys.shape[0], n_samples, -1).squeeze()  # (n_mutants, n_samples)

    # Add ppls and labels back to the mutants dataframe by matching sequences
    mutants_df["perplexity"] = np.nan
    mutants_df["label"] = np.nan
    for seq, ppl, lab in zip(sequences, ppls.numpy(), mutant_labels):
        mutants_df.loc[mutants_df["sequence"] == seq, "perplexity"] = ppl.min()
        mutants_df.loc[mutants_df["sequence"] == seq, "label"] = lab

    # Sort mutants by perplexity
    mutants_df = mutants_df.sort_values(by="perplexity").reset_index(drop=True)
    print(mutants_df.head())

    for mut_type in ["insertion", "deletion"]:
        print(f"Mutant type: {mut_type}")

        # Correlation plot between perplexity and label
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=mutants_df[mutants_df["type"] == mut_type], x="perplexity", y="label")
        plt.title(f"Perplexity vs SARSCoV1 Fitness Pred for {mut_type}")
        plt.xlabel("Perplexity")
        plt.ylabel("SARSCoV1 Fitness Pred")

        subset = mutants_df[mutants_df["type"] == mut_type]
        corr = subset["perplexity"].corr(subset["label"], method="spearman")
        print(f"Spearman correlation for type '{mut_type}': {corr:.4f}")

        # add to plot
        plt.text(
            0.05,
            0.95,
            f"Spearman: {corr:.4f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
        )
        plt.savefig(f"perplexity_vs_sarscov1_{mut_type}.png")

    # breakpoint()


if __name__ == "__main__":
    main()
