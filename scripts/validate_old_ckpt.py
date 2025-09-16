#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio.Seq import Seq
from evo.dataset import CherriesDataset, EncodedPEINTDataset
from matplotlib import pyplot as plt
from tqdm import tqdm

from peint.data.datamodule import PLMRDataModule
from peint.models.frameworks.peint import load_from_old_checkpoint

# ## Load the checkpoint and test data
# In this first step, we load the model checkpoint and run inference on some test data. Our goal is to compute the log likelihoods of the parent-child pairs in the test data, to validate the model's performance.

# In[4]:

# Load models from old checkpoints that Milind provided
ckpt_dir = Path("/scratch/users/milind_jagota/bcr/models/peint")
ckpt_paths = {
    "heavy": ckpt_dir / "heavy/epoch=28-step=12000.ckpt",
    "light": ckpt_dir / "light/epoch=46-step=20000.ckpt",
    "joint": ckpt_dir / "joint/epoch=28-step=12000.ckpt",
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modules = {k: load_from_old_checkpoint(v, device) for k, v in ckpt_paths.items()}
vocab = modules["joint"].net.vocab


# In[5]:

# Load held out dataset to evaluate the model on
data_dir = Path("/scratch/users/milind_jagota/bcr/data/heavylight/peint")
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
n_batches = 30
ts, lls = [], defaultdict(list)

for i in tqdm(range(n_batches)):
    for seqtype in ["heavy", "light", "joint"]:
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
        if seqtype == "joint":
            ts.append(t.squeeze().detach().cpu().numpy())

lls = {k: np.concatenate(v) for k, v in lls.items()}
ts = np.concatenate(ts)


# In[7]:

df = pd.DataFrame(
    {
        "time": ts,
        "joint_ll": lls["joint"],
        "heavy_ll": lls["heavy"],
        "light_ll": lls["light"],
    }
)
df["marginal_prod_ll"] = df.heavy_ll + df.light_ll
df["time_bin"] = df.time // 0.25
binned_df = df.groupby("time_bin").mean()


# In[8]:


plt.scatter(binned_df.time, binned_df.joint_ll, label="Joint heavy/light model")
plt.scatter(binned_df.time, binned_df.heavy_ll + binned_df.light_ll, label="Product of marginals")
plt.xlabel("Time")
plt.ylabel("LL")
plt.legend()
plt.savefig("ll_vs_time.png", dpi=300)
plt.clf()


# ## Sample on the test data
# In this step, we load a large family (tree) from the held-out donor, take a naive ancestor and sample forwards in time to obtain synthetic leaves. We then compare these to the real leaves from the original tree.

# In[68]:


# Read the donor data file and filter for sequences in the first clonal family with more than length 50
donor_data_file = Path("/scratch/users/milind_jagota/bcr/data/heavylight/peint/peint_df.csv.gz")
full_data = pd.read_csv(donor_data_file, compression="gzip")
full_data = full_data[full_data.sample_id == "d4"]
full_data = full_data.groupby("family").filter(lambda x: len(x) > 50)
families = pd.unique(full_data.family)
full_data = full_data[full_data.family == families[0]]


# In[69]:


# Extract the naive ancestor and corresponding leaves from the clonal family data
naive = full_data[full_data.edge_id.str.contains("naive")]
naive_h = str(Seq(naive["parent_heavy"].iloc[0]).translate())
naive_l = str(Seq(naive["parent_light"].iloc[0]).translate())

leaves = full_data[full_data.edge_id.str.contains("contig")][["child_heavy", "child_light"]]
leaves["child_heavy"] = leaves["child_heavy"].apply(lambda x: str(Seq(x).translate()))
leaves["child_light"] = leaves["child_light"].apply(lambda x: str(Seq(x).translate()))


# In[ ]:


# Generate samples from p(y|x, t) where x is the naive ancestor and t is a chosen branch length
n_samples = 100
batch_size = 1
n_batches = n_samples // batch_size
sim_t = 5.0
samples = []

sim_t = sim_t * torch.ones((batch_size, 1)).to(device)

heavy_input = torch.from_numpy(vocab.encode([naive_h] * batch_size)).to(device)
light_input = torch.from_numpy(vocab.encode([naive_l] * batch_size)).to(device)
joint_input = torch.from_numpy(vocab.encode([naive_h + 10 * "G" + naive_l] * batch_size)).to(device)

max_len = len(naive_h) + len(naive_l) + 10  # +10 for the added 'G's in joint input

for _ in tqdm(range(n_batches)):
    with torch.no_grad() and torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        samp = modules["joint"].net.generate(
            x=joint_input,
            t=sim_t,
            max_decode_steps=max_len + 1,  # +1 to account for the start token
            device=device,
            temperature=1.0,
            p=1.0,
        )
    samples.append(samp.cpu().numpy())

samples = np.concatenate(samples)
samples = vocab.decode(samples)


# In[ ]:


# calculate Hamming distances between ancestor and sampled leaves
samp_dists = []
for ind in range(n_batches):
    dist = (np.array(list(samples[ind])) != np.array(list(naive_h + 10 * "G" + naive_l))).sum()
    samp_dists.append(dist)


# In[82]:

with open("sampled.txt", "w") as f:
    to_write = ["samp{0}\t".format(i) + samples[i] for i in range(len(samples))]
    f.write("\n".join(to_write))

with open("real_leaves.txt", "w") as f:
    to_write = [
        "real{0}\t".format(i) + leaves.child_heavy.iloc[i] + (10 * "G") + leaves.child_light.iloc[i]
        for i in range(leaves.shape[0])
    ]
    f.write("\n".join(to_write))


# calculate distances between ancestor and actual leaves
real_dists = []
for ind in range(leaves.shape[0]):
    dist = (
        np.array(list(leaves.child_heavy.iloc[ind] + leaves.child_light.iloc[ind]))
        != np.array(list(naive_h + naive_l))
    ).sum()
    real_dists.append(dist)


# In[83]:
samp_dists = np.array(samp_dists)
real_dists = np.array(real_dists)


plt.hist(samp_dists, label="Sampled", alpha=0.5, density=True)
plt.hist(real_dists, label="Real", alpha=0.5, density=True)
plt.legend()
plt.savefig("dist_hist.png", dpi=300)
plt.clf()
