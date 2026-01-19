#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd


def process_oas_files(oas_dir, output_file):
    """
    Process all OAS CSV.gz files and convert to FASTA format.
    Only keeps human sequences and concatenates heavy and light chains with 10 glycines.
    """

    # Find all CSV.gz files
    csv_files = list(Path(oas_dir).glob("*.csv.gz"))

    print(f"Total number of CSV.gz files found: {len(csv_files)}")

    total_sequences = 0
    human_sequences = 0

    with open(output_file, "w") as fasta_out:
        for i, csv_file in enumerate(csv_files):
            print(f"Processing file {i+1}/{len(csv_files)}: {csv_file.name}")

            try:
                # Read metadata from the first row (header)
                metadata = ",".join(pd.read_csv(csv_file, nrows=0).columns)
                metadata = json.loads(metadata)
                species = metadata.get("species", "unknown").lower()

                # Skip if not human
                if species != "human":
                    print(f"  Skipping non-human species: {species}")
                    continue

                print(f"  Processing human data from: {csv_file.name}")

                # Read CSV data starting from second line
                df = pd.read_csv(csv_file, header=1)

                file_seq_count = 0
                for idx, row in df.iterrows():
                    total_sequences += 1
                    file_seq_count += 1

                    # Get heavy and light chain amino acid sequences
                    heavy_seq = str(row.get("sequence_alignment_aa_heavy", "")).strip()
                    light_seq = str(row.get("sequence_alignment_aa_light", "")).strip()

                    # Skip if either sequence is missing or NaN
                    if not heavy_seq or not light_seq or heavy_seq == "nan" or light_seq == "nan":
                        continue

                    # Check if sequences are productive
                    productive_heavy = str(row.get("productive_heavy", "")).upper()
                    productive_light = str(row.get("productive_light", "")).upper()

                    if productive_heavy != "T" or productive_light != "T":
                        continue

                    # Concatenate with 10 glycines
                    glycine_linker = "G" * 10
                    concatenated_seq = heavy_seq + glycine_linker + light_seq

                    # Create FASTA header
                    seq_id_heavy = str(
                        row.get("sequence_id_heavy", f"seq_{human_sequences+1}_heavy")
                    )
                    seq_id_light = str(
                        row.get("sequence_id_light", f"seq_{human_sequences+1}_light")
                    )

                    header = f">{csv_file.stem}_{seq_id_heavy}_{seq_id_light}"

                    # Write to FASTA file
                    fasta_out.write(f"{header}\n")
                    fasta_out.write(f"{concatenated_seq}\n")

                    human_sequences += 1

                    if file_seq_count % 1000 == 0:
                        print(f"    Processed {file_seq_count} sequences from this file")

                print(f"  Finished processing {csv_file.name}: {file_seq_count} total sequences")

            except Exception as e:
                print(f"  Error processing {csv_file.name}: {e}")
                continue

    print(f"\nSummary:")
    print(f"Total sequences examined: {total_sequences}")
    print(f"Human sequences written to FASTA: {human_sequences}")
    print(f"Output file: {output_file}")
    
    
def process_naive_oas_files(oas_dir, output_file="oas_heavy_ighm.fasta"):
    """
    Process all OAS CSV.gz files and convert to FASTA format.
    Only keeps human sequences that are IgM isotype (naive B-cells) from heavy chains with v_identity > 98%.
    """
    # Find all CSV.gz files
    csv_files = list(Path(oas_dir).glob("*.csv.gz"))

    print(f"Total number of CSV.gz files found: {len(csv_files)}")

    total_sequences = 0
    human_sequences = 0

    with open(output_file, "w") as fasta_out:
        for i, csv_file in enumerate(csv_files):
            if not csv_file.name.endswith("Heavy_IGHM.csv.gz"):
                print(f"Skipping file {csv_file.name} as it is not Heavy_IGHM")
                continue

            print(f"Processing file {i+1}/{len(csv_files)}: {csv_file.name}")
            try:
                # Read metadata from the first row (header)
                metadata = ",".join(pd.read_csv(csv_file, nrows=0).columns)
                metadata = json.loads(metadata)
                species = metadata.get("species", "unknown").lower()

                # Skip if not human
                # if species != "human":
                #     print(f"  Skipping non-human species: {species}")
                #     continue

                print(f"  Processing human data from: {csv_file.name}")

                # Read CSV data starting from second line
                df = pd.read_csv(csv_file, header=1)

                file_seq_count = 0
                for idx, row in df.iterrows():
                    total_sequences += 1
                    file_seq_count += 1

                    # Get heavy chain amino acid sequence
                    heavy_seq = str(row.get("sequence_alignment_aa", "")).strip()

                    # Skip if sequence is missing or NaN
                    if not heavy_seq or heavy_seq == "nan":
                        continue

                    # Check if sequences are productive
                    productive_heavy = str(row.get("productive", "")).upper()
                    if productive_heavy != "T":
                        continue

                    # Check v_identity > 98%
                    v_identity = row.get("v_identity", 0)
                    if v_identity < 98.0:
                        continue

                    # Create FASTA header
                    seq_id_heavy = str(
                        row.get("sequence_id", f"seq_{human_sequences+1}")
                    )

                    header = f">{csv_file.stem}_{seq_id_heavy}"

                    # Write to FASTA file
                    fasta_out.write(f"{header}\n")
                    fasta_out.write(f"{heavy_seq}\n")
                    human_sequences += 1

                    if file_seq_count % 1000 == 0:
                        print(f"    Processed {file_seq_count} sequences from this file")

                print(f"  Finished processing {csv_file.name}: {file_seq_count} total sequences")

            except Exception as e:
                print(f"  Error processing {csv_file.name}: {e}")
                continue

    print(f"\nSummary:")
    print(f"Total sequences examined: {total_sequences}")
    print(f"Human sequences written to FASTA: {human_sequences}")
    print(f"Output file: {output_file}")
    
    
def load_naive_sequences_from_oas():
    from datasets import load_dataset

    # 1. Connect to the OAS mirror on Hugging Face (streaming mode = no massive download)
    # We use a reliable mirror of the OAS data
    ds = load_dataset("bayer-science-for-better/oas_paired_human", split="train", streaming=True)

    naive_heavy_chains = []

    print("Fetching naive sequences...")
    for row in ds:
        # 2. Filter for 'IgM'. 
        # Biologically, Naive B-cells are IgM+. Class-switched (IgG/IgA) are matured.
        if row['isotype_heavy'] == 'IgM':

            # Optional: strict check for 0 mutations (if metadata allows), 
            # but IgM alone is the standard "simple" proxy for naive.
            seq = row['sequence_heavy_aa']

            # Basic validation to ensure it's a full sequence (not a fragment)
            if len(seq) > 100:
                naive_heavy_chains.append(seq)

        if len(naive_heavy_chains) >= 100:  # Grab 100 sequences
            break

    # 3. View results
    for i, seq in enumerate(naive_heavy_chains[:3]):
        print(f"\nNaive Seq {i+1}: {seq[:50]}...")
        
    return naive_heavy_chains


def main():
    oas_dir = "/scratch/users/stephen.lu/projects/protevo/data/naive_oas"
    output_file = "/scratch/users/stephen.lu/projects/protevo/data/naive_oas/oas.fasta"
    # process_oas_files(oas_dir, output_file)
    process_naive_oas_files(oas_dir, output_file=output_file)
    
    # naive_heavy_chains = load_naive_sequences_from_oas()
    breakpoint()


if __name__ == "__main__":
    main()
