import numpy as np
import torch

import peint.models.frameworks.dfm as dfm
from evo.dataset import ComplexCherriesDataset, TorchWrapperDataset
from evo.tensor import collate_tensors
from evo.tokenization import Vocab


class Dataset(TorchWrapperDataset):
    def __init__(
        self,
        dataset: ComplexCherriesDataset,
        vocab: Vocab,
        sep_token: str = ".",
        permute_chain_order: bool = False,
        permute_method: str | None = "random",  # "random" or "reverse"
        *args,
        **kwargs,
    ):
        super().__init__(dataset=dataset, vocab=vocab, *args, **kwargs)
        assert sep_token in self.vocab.tokens, f"{sep_token} not in vocab"
        self.sep_token = sep_token
        self.permute_chain_order = permute_chain_order
        self.permute_method = permute_method
        self.scheduler = dfm.LinearScheduler()
        self.cond_path = dfm.FactorizedPath(self.scheduler, vocab_size=len(vocab))
        assert permute_method in [
            "random",
            "reverse",
            None,
        ], f"Invalid permute_method: {permute_method}"

    def __getitem__(self, index: int):
        xs, ys, ts, chain_ids = super().__getitem__(index)

        # Optionally permute the order of chains with some probability
        if self.permute_chain_order and len(xs) > 1:
            if self.permute_method == "reverse":
                perm = list(reversed(range(len(xs))))
            else:
                perm = np.random.permutation(len(xs))
            xs = [xs[i] for i in perm]
            ys = [ys[i] for i in perm]
            ts = [ts[i] for i in perm]
            chain_ids = [chain_ids[i] for i in perm]

        # for now, we expect x and y to have the same lengths
        assert all(
            [len(x) == len(y) for x, y in zip(xs, ys)]
        ), "x and y chains must have the same lengths"

        # create chain_ids tensor
        sizes = [len(x) for x in xs]
        cids = np.concatenate(
            [np.full(size, fill_value=cid) for cid, size in zip(chain_ids, sizes)], axis=0
        )
        pad_widths = [(0, 0)] * (cids.ndim - 1) + [(int(vocab.prepend_bos), int(vocab.append_eos))]
        cids = np.pad(cids, pad_width=pad_widths, mode="constant", constant_values=0)

        # tokenize x and y
        xs = torch.from_numpy(self.vocab.encode_single_sequence("".join(xs)))
        ys = torch.from_numpy(self.vocab.encode_single_sequence("".join(ys)))
        assert xs.shape[0] == ys.shape[0], "x and y must have the same length after tokenization"

        # interpolate x and y using a linear conditional path
        zs = self.cond_path.sample(
            x0=dfm.x2prob(xs, vocab_size=len(self.vocab)),
            x1=dfm.x2prob(ys, vocab_size=len(self.vocab)),
            t=torch.rand(1, device=xs.device),  # this t in [0, 1) is used for interpolation only
            vocab_size=len(self.vocab),
        )

        # breakpoint()

        # Handle both single time values and lists of time values
        if isinstance(ts, (int, float)):
            # Single time value - convert to tensor with single element per chain
            ts = torch.tensor([ts], dtype=torch.float32)
        elif all(isinstance(t, (int, float)) for t in ts):
            ts = torch.tensor(ts, dtype=torch.float32)
        else:
            ts = np.array(ts, dtype=object)

        return x_src, x_tgt, y_src, y_tgt, ts, chain_ids, x_sizes, y_sizes

    def collater(self, batch):
        x_src, x_tgt, y_src, y_tgt, ts, chain_ids, x_sizes, y_sizes = zip(*batch)
        return (
            collate_tensors(x_src, constant_value=self.vocab.pad_idx),
            collate_tensors(x_tgt, constant_value=self.vocab.pad_idx),
            collate_tensors(y_src, constant_value=self.vocab.pad_idx),
            collate_tensors(y_tgt, constant_value=self.vocab.pad_idx),
            collate_tensors(ts, constant_value=0.0),
            collate_tensors(chain_ids, constant_value=0),
            collate_tensors(x_sizes, constant_value=0),
            collate_tensors(y_sizes, constant_value=0),
        )


if __name__ == "__main__":
    from esm.data import Alphabet
    from tqdm import tqdm

    from ..datamodule import PLMRDataModule

    vocab = Vocab.from_esm_alphabet(Alphabet.from_architecture("ESM-1b"))

    data_file = "/accounts/projects/yss/stephen.lu/peint/data/dasm/edges/{}/v1jaffePairedCC.txt"

    _dataset = ComplexCherriesDataset(
        data_file=data_file.format("train"),
        min_t=0.0,
        sep_token=".",
        chain_id_offset=1,
    )
    dataset = PEINTFlowDataset(
        dataset=_dataset,
        vocab=vocab,
        sep_token=".",
        permute_chain_order=False,
    )

    # breakpoint()

    datamodule = PLMRDataModule(
        dataset=dataset,
        dataset_test=None,
        batch_size=16,
        num_workers=8,
        train_val_split=[0.95, 0.05],
        generator_seed=42,
        pin_memory=False,
        shuffle=True,
    )
    datamodule.setup(stage="fit")

    tr_dataloader = datamodule.train_dataloader()
    val_dataloaders = datamodule.val_dataloader()

    print(f"Dataset size: {len(dataset)}")
    print(f"Train dataset size: {len(datamodule.data_train)}")

    print(f"Number of validation datasets: {len(datamodule.data_val)}")
    print(f"Validation dataset size: {len(datamodule.data_val[0])}")

    for batch in tqdm(iter(tr_dataloader)):
        x_src, x_tgt, y_src, y_tgt, ts, chain_ids, x_sizes, y_sizes = batch
        print(batch)
        # breakpoint()
        break

    for batch in tqdm(iter(val_dataloaders[0])):
        x_src, x_tgt, y_src, y_tgt, ts, chain_ids, x_sizes, y_sizes = batch
        print(batch)
        # breakpoint()
        break

    # for batch in tqdm(iter(val_dataloaders[1])):
    #     x_src, y_src, y_tgt, mut_code, ts, chain_ids, x_sizes, y_sizes = batch
    #     print(batch)
    #     breakpoint()
    #     break

    # breakpoint()
