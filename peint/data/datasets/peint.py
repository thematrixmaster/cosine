import numpy as np
import torch

from evo.dataset import ComplexCherriesDataset, TorchWrapperDataset
from evo.tensor import collate_tensors, mask_tensor
from evo.tokenization import Vocab


class EncodedPEINTDataset(TorchWrapperDataset):
    def __init__(
        self,
        dataset: ComplexCherriesDataset,
        vocab: Vocab,
        mask_prob: float = 0.15,
        random_token_prob: float = 0.1,
        leave_unmasked_prob: float = 0.1,
        sep_token: str = ".",
        embed_x_per_chain: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(dataset=dataset, vocab=vocab, *args, **kwargs)
        self._mask_prob = mask_prob
        self._random_token_prob = random_token_prob
        self._leave_unmasked_prob = leave_unmasked_prob
        assert sep_token in self.vocab.tokens, f"{sep_token} not in vocab"
        self.sep_token = sep_token
        self.embed_x_per_chain = embed_x_per_chain

    def __getitem__(self, index: int):
        xs, ys, ts = super().__getitem__(index)

        if self.embed_x_per_chain:
            x_sizes = torch.tensor(
                [len(x) + self.vocab.prepend_bos + self.vocab.append_eos for x in xs],
                dtype=torch.long,
            )
            xs = torch.cat(
                [torch.from_numpy(self.vocab.encode_single_sequence(x)) for x in xs], dim=0
            )
        else:
            x_sizes = torch.tensor(
                [len(x) + 1 for x in xs], dtype=torch.long
            )  # add 1 for sep token
            x_sizes[0] += self.vocab.prepend_bos  # add bos to first chain
            x_sizes[-1] += (
                self.vocab.append_eos - 1
            )  # add eos to last chain and subtract 1 for sep token
            xs = torch.from_numpy(self.vocab.encode_single_sequence(self.sep_token.join(xs)))

        # apply masking to xs while ignoring special tokens and separator tokens
        x_src, x_tgt = mask_tensor(
            xs,
            self.vocab,  # x_tgt gets pad tok at masked pos
            mask_prob=self._mask_prob,
            random_token_prob=self._random_token_prob,
            leave_unmasked_prob=self._leave_unmasked_prob,
            extra_special_tok_idx=[self.vocab.tokens_to_idx[self.sep_token]],
        )

        # y is always embedded together as one sequence with separator tokens for autoregressive training
        y_sizes = torch.tensor([len(y) + 1 for y in ys], dtype=torch.long)  # add 1 for sep token
        y_sizes[0] += self.vocab.prepend_bos  # add bos to first chain
        y_sizes[-1] += (
            self.vocab.append_eos - 1
        )  # add eos to last chain and subtract 1 for sep token
        ys = torch.from_numpy(self.vocab.encode_single_sequence(self.sep_token.join(ys)))

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

        return x_src, x_tgt, y_src, y_tgt, ts, x_sizes, y_sizes

    def collater(self, batch):
        x_src, x_tgt, y_src, y_tgt, ts, x_sizes, y_sizes = zip(*batch)
        return (
            collate_tensors(x_src, constant_value=self.vocab.pad_idx),
            collate_tensors(x_tgt, constant_value=self.vocab.pad_idx),
            collate_tensors(y_src, constant_value=self.vocab.pad_idx),
            collate_tensors(y_tgt, constant_value=self.vocab.pad_idx),
            collate_tensors(ts, constant_value=0.0),
            collate_tensors(x_sizes, constant_value=0.0),
            collate_tensors(y_sizes, constant_value=0.0),
        )


class EncodedDMSDataset(EncodedPEINTDataset):
    def __init__(self, t: float = 7.0, *args, **kwargs):
        super().__init__(
            mask_prob=0.0, random_token_prob=0.0, leave_unmasked_prob=0.0, *args, **kwargs
        )
        self.t = t

    def collater(self, batch):
        x_src, _, y_src, y_tgt, mut_strs, x_sizes, y_sizes = zip(*batch)
        ts = [torch.full_like(xs, fill_value=self.t, dtype=torch.float32) for xs in x_sizes]
        return (
            collate_tensors(x_src, constant_value=self.vocab.pad_idx),
            collate_tensors(y_src, constant_value=self.vocab.pad_idx),
            collate_tensors(y_tgt, constant_value=self.vocab.pad_idx),
            np.array(mut_strs, dtype=object),
            collate_tensors(ts, constant_value=0.0),
            collate_tensors(x_sizes, constant_value=0.0),
            collate_tensors(y_sizes, constant_value=0.0),
        )


if __name__ == "__main__":
    from esm.data import Alphabet
    from torch.utils.data import ConcatDataset
    from tqdm import tqdm

    from ..datamodule import PLMRDataModule

    vocab = Vocab.from_esm_alphabet(Alphabet.from_architecture("ESM-1b"))

    data_file1 = "/accounts/projects/yss/stephen.lu/peint/data/wyatt/all/edges_joint_pipet/d1.txt"
    data_file2 = "/accounts/projects/yss/stephen.lu/peint/data/wyatt/all/edges_joint_pipet/d2.txt"
    data_file3 = "/accounts/projects/yss/stephen.lu/peint/data/wyatt/all/edges_joint_pipet/d3.txt"
    data_file4 = "/accounts/projects/yss/stephen.lu/peint/data/wyatt/all/edges_joint_pipet/d4.txt"
    dms_file = "/accounts/projects/yss/stephen.lu/peint/data/wyatt/dms/mut_joint_pipet.txt"

    datasets = []
    for data_file in [data_file1, data_file2, data_file3]:
        dataset = ComplexCherriesDataset(
            data_file=data_file,
            cache_indices=False,
            min_t=5e-3,
            sep_token=".",
        )
        datasets.append(dataset)

    _dataset = ConcatDataset(datasets)
    dataset = EncodedPEINTDataset(
        dataset=_dataset,
        vocab=vocab,
        sep_token=".",
        mask_prob=0.15,
        random_token_prob=0.0,
        leave_unmasked_prob=0.0,
        embed_x_per_chain=True,
    )

    dms_dataset = ComplexCherriesDataset(
        data_file=dms_file,
        cache_indices=False,
        min_t=5e-3,
        sep_token=".",
    )
    dms_dataset = EncodedDMSDataset(
        dataset=dms_dataset,
        vocab=vocab,
        sep_token=".",
        t=7.0,
    )

    datamodule = PLMRDataModule(
        dataset=dataset,
        dataset_train=None,
        dataset_val=[dms_dataset],
        dataset_test=None,
        batch_size=32,
        num_workers=8,
        train_val_split=[0.95, 0.05],
        generator_seed=42,
        pin_memory=False,
    )
    datamodule.setup(stage="fit")

    tr_dataloader = datamodule.train_dataloader()
    val_dataloaders = datamodule.val_dataloader()

    print(f"Dataset size: {len(dataset)}")
    print(f"Train dataset size: {len(datamodule.data_train)}")

    print(f"Number of validation datasets: {len(datamodule.data_val)}")
    print(f"Validation dataset size: {len(datamodule.data_val[0])}")

    for batch in tqdm(iter(tr_dataloader)):
        x_src, x_tgt, y_src, y_tgt, ts, x_sizes, y_sizes = batch
        print(batch)
        # breakpoint()
        break

    for batch in tqdm(iter(val_dataloaders[0])):
        x_src, x_tgt, y_src, y_tgt, ts, x_sizes, y_sizes = batch
        print(batch)
        # breakpoint()
        break

    for batch in tqdm(iter(val_dataloaders[1])):
        x_src, y_src, y_tgt, mut_code, ts, x_sizes, y_sizes = batch
        print(batch)
        # breakpoint()
        break

    # breakpoint()
