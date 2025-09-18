from os import PathLike
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

from peint.metrics.api import Metric


class ZeroShotDMS(Metric):
    """Measures the correlation between the perplexity of sequences outputted by a language model
    and the experimentally measured fitness of these sequences via a DMS assay.
    """

    def __init__(
        self,
        datafile: PathLike,
        log_transform: bool = True,
        name="zero_shot_fitness_correlation",
    ):
        super().__init__(name)
        self.datafile = Path(datafile)
        self.log_transform = log_transform
        self.setup()

    def setup(self) -> None:
        df = pd.read_csv(self.datafile)
        assert {"mut", "fitness"}.issubset(
            set(df.columns)
        ), "datafile must contain 'mut' and 'fitness' columns"
        df.set_index("mut", inplace=True)
        if self.log_transform:
            df["fitness"] = np.log(df["fitness"])
        self.df = df

    def compute(self, outputs: List[List[Any]]):
        if len(outputs) <= 1:
            return {}
        all_ppls, all_fitness = [], []
        for _ppls, _muts in outputs[1]:
            for p, m in zip(_ppls, _muts):
                if m in self.df.index:
                    all_ppls.append(p)
                    all_fitness.append(self.df.loc[m, "fitness"])

        if len(all_ppls) < 2:
            return {}
        all_ppls = np.array(all_ppls)
        all_fitness = np.array(all_fitness)
        spearman_corr = pd.Series(all_ppls).corr(pd.Series(all_fitness), method="spearman")
        pearson_corr = pd.Series(all_ppls).corr(pd.Series(all_fitness), method="pearson")

        return {"spearman": spearman_corr, "pearson": pearson_corr}

    def aggregate(self, metrics):
        return metrics


# Example usage
if __name__ == "__main__":
    pass
