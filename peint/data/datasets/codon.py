import numpy as np
import torch

from evo.dataset import ComplexCherriesDataset, TorchWrapperDataset
from evo.tensor import collate_tensors, mask_tensor
from evo.tokenization import CodonVocab


class CodonDataset(TorchWrapperDataset):
    vocab: CodonVocab

    def __init__(
        self,
        dataset: ComplexCherriesDataset,
        vocab: CodonVocab,
        mask_prob: float = 0.15,
        random_token_prob: float = 0.1,
        leave_unmasked_prob: float = 0.1,
        embed_x_per_chain: bool = False,
        permute_chain_order: bool = False,
        permute_method: str | None = "random",  # "random" or "reverse"
        *args,
        **kwargs,
    ):
        super().__init__(dataset=dataset, vocab=vocab, *args, **kwargs)
        self._mask_prob = mask_prob
        self._random_token_prob = random_token_prob
        self._leave_unmasked_prob = leave_unmasked_prob
        self.embed_x_per_chain = embed_x_per_chain
        self.permute_chain_order = permute_chain_order
        self.permute_method = permute_method
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

        # convert chain id to tensor
        chain_ids = torch.tensor(chain_ids, dtype=torch.long)

        # x can be embedded either as separate chains or as one sequence
        if self.embed_x_per_chain:
            x_sizes = torch.tensor(
                [-(-len(x) // 3) + self.vocab.prepend_bos + self.vocab.append_eos for x in xs],
                dtype=torch.long,
            )
            xs = torch.cat(
                [torch.from_numpy(self.vocab.encode_single_sequence(x)) for x in xs], dim=0
            )
        else:
            x_sizes = torch.tensor([-(-len(x) // 3) for x in xs], dtype=torch.long)
            x_sizes[0] += self.vocab.prepend_bos  # add bos to first chain
            x_sizes[-1] += self.vocab.append_eos  # add eos to last chain
            xs = torch.from_numpy(self.vocab.encode_single_sequence("".join(xs)))

        # apply masking to xs while ignoring special tokens
        x_src, x_tgt = mask_tensor(
            xs,
            self.vocab,  # x_tgt gets pad tok at masked pos
            mask_prob=self._mask_prob,
            random_token_prob=self._random_token_prob,
            leave_unmasked_prob=self._leave_unmasked_prob,
        )

        # y is always embedded together as one sequence for autoregressive training
        y_sizes = torch.tensor([-(-len(y) // 3) for y in ys], dtype=torch.long)
        y_sizes[0] += self.vocab.prepend_bos  # add bos to first chain
        y_sizes[-1] += self.vocab.append_eos  # add eos to last chain
        ys = torch.from_numpy(self.vocab.encode_single_sequence("".join(ys)))

        y_src, y_tgt = ys[:-1], ys[1:]  # shift y for autoregressive training
        y_sizes[-1] -= 1  # adjust sizes since y_src is shifted and truncated by 1

        # Handle both single time values and lists of time values
        if isinstance(ts, (int, float)):
            # Single time value - convert to tensor with single element per chain
            ts = torch.tensor([ts], dtype=torch.float32)
        elif all(isinstance(t, (int, float)) for t in ts):
            ts = torch.tensor(ts, dtype=torch.float32)
        else:
            ts = np.array(ts, dtype=object)

        # Apply padding to the right of x_sizes, y_sizes to make them the same length as x_src, y_src
        x_sizes = torch.nn.functional.pad(x_sizes, (0, len(x_src) - len(x_sizes)), value=0)
        y_sizes = torch.nn.functional.pad(y_sizes, (0, len(y_src) - len(y_sizes)), value=0)

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
    from tqdm import tqdm

    from ..datamodule import PLMRDataModule

    vocab = CodonVocab.from_codons()

    data_dir = "/accounts/projects/yss/stephen.lu/peint/data/wyatt/subs/edges_joint/nt"
    data_file1 = f"{data_dir}/d1.txt"
    data_file2 = f"{data_dir}/d2.txt"
    data_file3 = f"{data_dir}/d3.txt"
    data_file4 = f"{data_dir}/d4.txt"

    datasets = []

    for data_file in [data_file1, data_file2, data_file3]:
        dataset = ComplexCherriesDataset(
            data_file=data_file,
            min_t=0.0,
            sep_token=".",
        )
        datasets.append(dataset)

    dataset = torch.utils.data.ConcatDataset(datasets)
    dataset = CodonDataset(
        dataset=dataset, vocab=vocab, embed_x_per_chain=True, permute_chain_order=True
    )

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
    val_dataloader = datamodule.val_dataloader()

    print(f"Dataset size: {len(dataset)}")
    print(f"Train dataset size: {len(datamodule.data_train)}")

    print(f"Number of validation datasets: {len(datamodule.data_val)}")
    print(f"Validation dataset size: {len(datamodule.data_val[0])}")

    for batch in tqdm(iter(tr_dataloader)):
        x_src, x_tgt, y_src, y_tgt, ts, chain_ids, x_sizes, y_sizes = batch
        # print(batch)
        # breakpoint()
        # break

    for batch in tqdm(iter(val_dataloader)):
        x_src, x_tgt, y_src, y_tgt, ts, chain_ids, x_sizes, y_sizes = batch
        # print(batch)
        # breakpoint()
        # break

    # breakpoint()
