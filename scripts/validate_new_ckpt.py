#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from evo.dataset import CherriesDataset, EncodedPEINTDataset
from matplotlib import pyplot as plt
from tqdm import tqdm

from peint.data.datamodule import PLMRDataModule
from peint.models.frameworks.peint import load_from_new_checkpoint

# ## Load the checkpoint and test data
# In this first step, we load the model checkpoint and run inference on some test data. Our goal is to compute the log likelihoods of the parent-child pairs in the test data, to validate the model's performance.

# In[4]:

# Load models from old checkpoints that Milind provided
ckpt_dir = Path("/accounts/projects/yss/stephen.lu/protevo/plmr/logs/train/runs")
ckpt_paths = {
    "heavy": ckpt_dir / "2025-08-08_15-36-04/checkpoints/epoch_089.ckpt",
    # "light": ,
    # "joint": ,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modules = {k: load_from_new_checkpoint(v, device) for k, v in ckpt_paths.items()}
vocab = next(iter(modules.values())).net.vocab


# In[5]:

# Load held out dataset to evaluate the model on
data_dir = Path("/accounts/projects/yss/stephen.lu/protevo/plmr/data/wyatt/indels")
data_files = {
    "heavy": data_dir / "edges_heavy/d4.txt",
    "light": data_dir / "edges_light/d4.txt",
    "joint": data_dir / "edges_joint/d4.txt",
}

datasets = {
    k: EncodedPEINTDataset(
        dataset=CherriesDataset(data_file=v),
        vocab=vocab,
        mask_prob=0.15,
    )
    for k, v in data_files.items()
}
generators = {
    k: iter(
        PLMRDataModule(
            dataset=v,
            batch_size=32,
            shuffle=False,
        )._dataloader_template(dataset=v, training=False)
    )
    for k, v in datasets.items()
}

# In[6]:


# Calculate likelihoods for a few batches
n_batches = 100
ts, lls = [], defaultdict(list)

for i in tqdm(range(n_batches)):
    for j, seqtype in enumerate(modules.keys()):
        module = modules[seqtype]
        generator = generators[seqtype]

        batch = next(generator)
        batch = [b.to(device) for b in batch]
        [x, x_targets, y, y_targets, t, x_pad_mask, y_pad_mask] = batch

        yt_mask = y_targets != module.net.vocab.pad_idx  # actual values

        with torch.no_grad() and torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_logits, y_logits = module(x, y, t, x_pad_mask, y_pad_mask)

        # Calculate log probabilities predicted by the model
        y_logits = y_logits - torch.logsumexp(y_logits, dim=-1, keepdim=True)

        nll = F.cross_entropy(
            y_logits.transpose(-1, -2),
            y_targets,
            ignore_index=module.net.vocab.pad_idx,
            reduction="none",
        )  # keep unreduced to get per-site time likelihood

        ll = -nll * yt_mask.float()  # log likelihoods (bs, seq_len)
        ll = ll.sum(dim=-1)  # sum over sequence length (bs,)

        lls[seqtype].append(ll.detach().cpu().numpy())
        if j == 0:
            ts.append(t.squeeze().detach().cpu().numpy())

lls = {k: np.concatenate(v) for k, v in lls.items()}
ts = np.concatenate(ts)


# In[7]:

df_data = {"time": ts} | {f"{seqtype}_ll": lls[seqtype] for seqtype in lls.keys()}
df = pd.DataFrame(df_data)

if "heavy_ll" in df.columns and "light_ll" in df.columns:
    df["marginal_prod_ll"] = df.heavy_ll + df.light_ll

df["time_bin"] = df.time // df.time.quantile(0.15).astype(float)
binned_df = df.groupby("time_bin").mean()

# In[8]:

# Create subplot with histogram on x-axis
fig, (ax_main, ax_histx) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(6, 6))

# Main scatter plot
for seqtype in lls.keys():
    ax_main.scatter(binned_df.time, binned_df[f"{seqtype}_ll"], label=f"{seqtype} model")

if "marginal_prod_ll" in binned_df.columns:
    ax_main.scatter(binned_df.time, binned_df.marginal_prod_ll, label="Product of marginals")

ax_main.set_ylabel("LL")
ax_main.legend()

# Histogram on x-axis showing distribution of actual time values
ax_histx.hist(ts, bins=50, alpha=0.7, color="gray", edgecolor="black")
ax_histx.set_ylabel("Count")
ax_histx.set_xlabel("Time")

plt.tight_layout()
plt.savefig("ll_vs_time.png", dpi=300)
plt.clf()
