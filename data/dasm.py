import os
from pathlib import Path
from evo.antibody import parallel_align_sequences


def extract_transitions_from_file(data_path: Path):
    # ignore first line of the file since it is the header
    with open(data_path, "r") as f:
        next(f)
        data = f.read()

    # split the data into lines
    is_paired = False
    lines = data.strip().split("\n")
    parent_chains, child_chains, branch_lengths = [], [], []
    for line in lines:
        # split the line into parts
        parts = line.split()
        branch_lengths.append(float(parts[2]))

        if is_paired or "." in parts[0]:
            parent_chains.append(parts[0].split("."))            
            child_chains.append(parts[1].split("."))
            is_paired = True
        else:
            parent_chains.append(parts[0])
            child_chains.append(parts[1])
    return parent_chains, child_chains, branch_lengths


def prepare_sequences_for_align(sequences):
    return [(f'seq_{i}', seq) for i, seq in enumerate(sequences)]


def extract_aligned_sequences(aligned_sequences):
    return [seq for _, seq in aligned_sequences]


def align_aho_sequences(parent_chains, child_chains):
    if isinstance(parent_chains[0], list):
        parent_heavy_chains, parent_light_chains = zip(*parent_chains)
        child_heavy_chains, child_light_chains = zip(*child_chains)
        aho_parent_heavy_chains = extract_aligned_sequences(parallel_align_sequences(prepare_sequences_for_align(parent_heavy_chains), n_jobs=16))
        aho_parent_light_chains = extract_aligned_sequences(parallel_align_sequences(prepare_sequences_for_align(parent_light_chains), n_jobs=16))
        aho_child_heavy_chains = extract_aligned_sequences(parallel_align_sequences(prepare_sequences_for_align(child_heavy_chains), n_jobs=16))
        aho_child_light_chains = extract_aligned_sequences(parallel_align_sequences(prepare_sequences_for_align(child_light_chains), n_jobs=16))
        # zip them back together to return in same format as input
        aho_parent_chains = list(zip(aho_parent_heavy_chains, aho_parent_light_chains))
        aho_child_chains = list(zip(aho_child_heavy_chains, aho_child_light_chains))
        return aho_parent_chains, aho_child_chains
    else:
        print(parent_chains[0])
        aho_parent_chains = extract_aligned_sequences(parallel_align_sequences(prepare_sequences_for_align(parent_chains), n_jobs=16))
        aho_child_chains = extract_aligned_sequences(parallel_align_sequences(prepare_sequences_for_align(child_chains), n_jobs=16))
        return aho_parent_chains, aho_child_chains


def save_transitions_to_file(transitions, data_path: Path):
    n_transitions = len(transitions)
    with open(data_path, "w") as f:
        f.write(f"{n_transitions} transitions\n")
        for parent_chains, child_chains, branch_length in transitions:
            if isinstance(parent_chains, list):
                f.write(f"{parent_chains[0]}.{parent_chains[1]} {child_chains[0]}.{child_chains[1]} {branch_length}\n")
            else:
                f.write(f"{parent_chains} {child_chains} {branch_length}\n")


if __name__ == "__main__":
    src_dir = Path("/scratch/users/stephen.lu/projects/protevo/data/dasm/edges")
    tgt_dir = Path("/scratch/users/stephen.lu/projects/protevo/data/dasm/edges_aho")

    os.makedirs(tgt_dir, exist_ok=True)

    # iterate over each subdirectory in src_dir and each file in each subdirectory
    for chain_type in os.listdir(src_dir):
        chain_type_dir = src_dir / chain_type
        tgt_chain_type_dir = tgt_dir / chain_type
        os.makedirs(tgt_chain_type_dir, exist_ok=True)
        for data_file in os.listdir(chain_type_dir):
            aligned_data_path = tgt_chain_type_dir / data_file
            print(f"Processing {data_file} and saving to {aligned_data_path}")

            # extract the transitions from the data file
            data_path = chain_type_dir / data_file
            parent_chains, child_chains, branch_lengths = extract_transitions_from_file(data_path)
            
            # align the sequences
            aho_parent_chains, aho_child_chains = align_aho_sequences(parent_chains, child_chains)

            # save the aligned sequences to a new file
            transitions = list(zip(aho_parent_chains, aho_child_chains, branch_lengths))
            save_transitions_to_file(transitions, aligned_data_path)
