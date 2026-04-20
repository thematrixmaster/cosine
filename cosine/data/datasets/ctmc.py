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
        random_cherry_order: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(dataset=dataset, vocab=vocab, *args, **kwargs)
        self.sep_token = sep_token
        self.random_cherry_order = random_cherry_order

    def __getitem__(self, index: int):
        xs, ys, ts, _ = super().__getitem__(index)

        # for now, we expect x and y to have the same lengths
        assert all(
            [len(x) == len(y) for x, y in zip(xs, ys)]
        ), "x and y chains must have the same lengths"

        # if random order is true, then permute order of x and y randomly
        if self.random_cherry_order:
            perm = torch.randperm(len(xs))
            xs = [xs[i] for i in perm]
            ys = [ys[i] for i in perm]

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
