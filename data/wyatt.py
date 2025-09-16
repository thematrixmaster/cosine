from pathlib import Path

import pandas as pd


def main():
    df = pd.read_csv(
        Path(__file__).parent
        / "wyatt"
        / "wyatt-10x-1p5m_fs-all-prank_paired-merged_pcp_2025-06-24.csv"
    )
    sample_ids = df["sample_id"].unique()

    for sample_id in sample_ids:
        sample_df = df[df["sample_id"] == sample_id]
        parent_dna_heavy = df["parent_heavy"].tolist()
        parent_dna_light = df["parent_light"].tolist()
        # TODO


if __name__ == "__main__":
    main()
