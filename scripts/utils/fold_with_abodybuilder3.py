from typing import List

import numpy as np
import pandas as pd
import torch
from abodybuilder3.lightning_module import LitABB3
from abodybuilder3.stages.inference import compute_plddt
from abodybuilder3.utils import add_atom37_to_output, string_to_input
from tqdm import tqdm


def compute_abodybuilder3_plddt(
    model: LitABB3, heavy_chain: str, light_chain: str, device: str = "cuda"
) -> List[float]:
    """
    Computes PLDDT score for sequences using ABodyBuilder3.
    """
    ab_input = string_to_input(heavy=heavy_chain, light=light_chain)
    ab_input_batch = {
        key: (value.unsqueeze(0).to(device) if key not in ["single", "pair"] else value.to(device))
        for key, value in ab_input.items()
    }
    model.to(device)
    output = model(ab_input_batch, ab_input_batch["aatype"])
    output = add_atom37_to_output(output, ab_input["aatype"].to(device))
    plddt = compute_plddt(output["plddt"].detach().cpu()).mean().item()
    return plddt


if __name__ == "__main__":
    import os
    import sys

    assert len(sys.argv) > 1, "Please provide input CSV file path as argument."
    input_csv = sys.argv[1]
    assert input_csv.endswith(".csv"), "Input file must be a CSV."
    assert os.path.exists(input_csv), f"Input CSV file {input_csv} does not exist."

    input_df = pd.read_csv(input_csv)
    # sequence_col = "sampled_seq"
    sequence_col = "sequence"
    assert (
        sequence_col in input_df.columns
    ), f"Input CSV must contain '{sequence_col}' column for heavy chain sequences."

    # placeholder_light_chain = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
    placeholder_light_chain = "DIVMTQSPLSLPVTPGEPASISCRSSQSLLQNNGYNYLAWYLQKPGQSPQLLIYLSSTRASGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCMQSLQIPGTFGQGTRLEIK"

    path_to_checkpoint = (
        "/scratch/users/stephen.lu/projects/abodybuilder3/output/plddt-loss/best_second_stage.ckpt"
    )
    module = LitABB3.load_from_checkpoint(path_to_checkpoint)
    model = module.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plddts = []
    for idx, row in tqdm(
        input_df.iterrows(), total=len(input_df), desc="Computing ABodyBuilder3 PLDDT"
    ):
        # if 'abodybuilder3_plddt' in row and not pd.isna(row['abodybuilder3_plddt']):
        #     plddts.append(row['abodybuilder3_plddt'])
        #     continue
        heavy = row[sequence_col]
        light = placeholder_light_chain  # using placeholder light chain
        plddt = compute_abodybuilder3_plddt(model, heavy, light, device)
        plddts.append(plddt)

    input_df["abodybuilder3_plddt"] = plddts
    mean_plddt = np.mean(plddts)
    std_plddt = np.std(plddts)
    print(
        f"Computed ABodyBuilder3 PLDDT for {len(plddts)} sequences. Mean: {mean_plddt:.2f}, Std: {std_plddt:.2f}"
    )
    input_df.to_csv(input_csv, index=False)
