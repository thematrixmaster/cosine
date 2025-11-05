import torch

from evo.dataset import ComplexCherriesDataset, TorchWrapperDataset
from evo.tensor import collate_tensors
from evo.tokenization import Vocab


class CTMCDataset(TorchWrapperDataset):
    def __init__(
        self,
        vocab: Vocab,
        dataset: ComplexCherriesDataset,
        sep_token: str = ".",
        *args,
        **kwargs,
    ):
        super().__init__(dataset=dataset, vocab=vocab, *args, **kwargs)
        self.sep_token = sep_token

    def __getitem__(self, index: int):
        xs, ys, ts, _ = super().__getitem__(index)

        # for now, we expect x and y to have the same lengths
        assert all(
            [len(x) == len(y) for x, y in zip(xs, ys)]
        ), "x and y chains must have the same lengths"

        # x is always embedded together as one sequence with a separator token
        x_sizes = torch.tensor([len(x) + 1 for x in xs], dtype=torch.long)
        x_sizes[0] += self.vocab.prepend_bos  # add bos to first chain
        x_sizes[-1] += (
            self.vocab.append_eos - 1
        )  # add eos to last chain and subtract 1 for sep token
        xs = torch.from_numpy(self.vocab.encode_single_sequence(self.sep_token.join(xs)))

        # y is always embedded together as one sequence for autoregressive training
        ys = torch.from_numpy(self.vocab.encode_single_sequence(self.sep_token.join(ys)))
        t = torch.tensor([ts[0]], dtype=torch.float32)

        # Apply padding to the right of x_sizes to match x_sizes
        x_sizes = torch.nn.functional.pad(x_sizes, (0, len(xs) - len(x_sizes)), value=0)

        return xs, ys, t, x_sizes

    def collater(self, batch):
        xs, ys, t, x_sizes = zip(*batch)
        return (
            collate_tensors(xs, constant_value=self.vocab.pad_idx),
            collate_tensors(ys, constant_value=self.vocab.pad_idx),
            collate_tensors(t, constant_value=0.0),
            collate_tensors(x_sizes, constant_value=0),
        )


if __name__ == "__main__":
    from esm.data import Alphabet
    from tqdm import tqdm

    from ..datamodule import PLMRDataModule

    vocab = Vocab.from_esm_alphabet(Alphabet.from_architecture("ESM-1b"))

    data_dir = "/accounts/projects/yss/stephen.lu/peint/data/wyatt/subs/edges_joint/aa"
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
    dataset = CTMCDataset(vocab=vocab, sep_token=".", dataset=dataset)

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

    for batch in tqdm(iter(tr_dataloader)):
        xs, ys, t, x_sizes = batch
        assert xs.size(1) == ys.size(1)

    for batch in tqdm(iter(val_dataloader)):
        xs, ys, t, x_sizes = batch
        assert xs.size(1) == ys.size(1)

    # breakpoint()
