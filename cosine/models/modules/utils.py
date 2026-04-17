from typing import Dict

import lightning as pl
import numpy as np
import wandb
from lightning.pytorch.loggers import WandbLogger

from evo.phylogeny import VALID_BINS


def log_wandb_scatter(
    trainer: pl.Trainer,
    arr_per_bin: Dict[float, np.ndarray],
    name: str = "time_bin_likelihood",
    xlabel: str = "Time",
    ylabel: str = "Mean per-site Likelihood",
    title: str = "Loglikelihood per time bin",
    transform_fn: callable = lambda v: np.exp(np.mean(v)),
):
    arrmeans = np.zeros(len(VALID_BINS))
    for k, v in arr_per_bin.items():
        arrmeans[k] = transform_fn(v)

    nonzero = np.nonzero(arrmeans)[0]
    xs = VALID_BINS[nonzero]
    ys = arrmeans[nonzero]

    data = [[x, y] for x, y in zip(xs, ys)]
    table = wandb.Table(data=data, columns=[xlabel, ylabel])

    # important for multi-gpu runs, doesn't seem to affect single gpu runs
    if trainer.global_rank == 0 and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.log(
            {
                name: wandb.plot.scatter(
                    table,
                    xlabel,
                    ylabel,
                    title=title,
                )
            }
        )
