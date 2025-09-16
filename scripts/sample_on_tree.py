from pathlib import Path

import pandas as pd
import torch
from Bio.Seq import Seq
from ete3 import Tree
from tqdm import tqdm

tqdm.pandas()

from peint.models.frameworks.peint import (
    load_from_old_checkpoint,
    simulate_evolution_with_rejection_sampling,
)


def df_to_ete3_tree(df):
    """
    Convert a dataframe of edges (parent_name, child_name, branch_length)
    into an ete3 Tree object in 'format=3'.

    The tree is assumed to have exactly one root: a node that does not appear as any child's name.
    """

    # Create a lookup for all nodes
    # We'll store node_name -> ete3.Tree instance
    nodes = {}

    # Create all nodes, but do not connect them yet
    for parent_name, child_name, branch_length in df[
        ["parent_name", "child_name", "branch_length"]
    ].itertuples(index=False):
        if parent_name not in nodes:
            nodes[parent_name] = Tree(name=parent_name)  # parent node
        if child_name not in nodes:
            nodes[child_name] = Tree(name=child_name)  # child node

    # Connect child to parent, setting the branch length (dist) on the child side
    # We'll also track who is the parent of whom in parent_of
    parent_of = {}
    for parent_name, child_name, branch_length in df[
        ["parent_name", "child_name", "branch_length"]
    ].itertuples(index=False):
        parent_node = nodes[parent_name]
        child_node = nodes[child_name]
        # Set branch length on the child
        child_node.dist = branch_length
        # Attach child to parent's children
        parent_node.add_child(child_node)
        parent_of[child_name] = parent_name

    # Find the root as the node that never appears as a child
    all_nodes = set(nodes.keys())
    all_children = set(parent_of.keys())
    root_candidates = all_nodes - all_children
    if len(root_candidates) != 1:
        raise ValueError(f"Expected exactly 1 root, found: {root_candidates}")
    root_name = root_candidates.pop()

    # Return the root as an ete3 Tree
    # By construction, nodes[root_name] is now the root and has all children below it
    tree = nodes[root_name]

    # If you want to ensure 'format=3' usage downstream, you can do:
    #   newick_str = tree.write(format=3)
    #   tree = Tree(newick_str, format=3)
    # But typically, once you have 'tree' as a Tree object, you can simply
    # use `tree.write(format=3)` or pass format=3 when you want to export.
    return tree


def main():
    data_dir = Path("/scratch/users/milind_jagota/bcr/data/heavylight/trees/")

    # heavy chain parent child relationships
    heavy = pd.read_csv(
        data_dir / "wyatt-10x-1p5m_paired-igh_fs-all_pcp_2024-11-21.csv.gz",
        compression="gzip",
        index_col=0,
    )

    # light chain parent child relationships (kappa and lambda, mutually exclusive)
    kappa = pd.read_csv(
        data_dir / "wyatt-10x-1p5m_paired-igk_fs-all_pcp_2024-11-21.csv.gz",
        compression="gzip",
        index_col=0,
    )
    lambd = pd.read_csv(
        data_dir / "wyatt-10x-1p5m_paired-igl_fs-all_pcp_2024-11-21.csv.gz",
        compression="gzip",
        index_col=0,
    )

    # Create unique family and edge identifiers for each parent-child relationship
    heavy["family"] = heavy["sample_id"] + "_" + heavy["family"].astype(str)
    kappa["family"] = kappa["sample_id"] + "_" + kappa["family"].astype(str)
    lambd["family"] = lambd["sample_id"] + "_" + lambd["family"].astype(str)

    heavy["edge_id"] = heavy["family"] + ";" + heavy["parent_name"] + ";" + heavy["child_name"]
    kappa["edge_id"] = kappa["family"] + ";" + kappa["parent_name"] + ";" + kappa["child_name"]
    lambd["edge_id"] = lambd["family"] + ";" + lambd["parent_name"] + ";" + lambd["child_name"]

    # Filter out kappa and lambda chains that are not present in the heavy chain
    # Filder out kappa and lambda chains that fail allelic exclusion
    keep_kappa = (kappa.edge_id.isin(heavy.edge_id)) & ~(kappa.edge_id.isin(lambd.edge_id))
    keep_lambd = (lambd.edge_id.isin(heavy.edge_id)) & ~(lambd.edge_id.isin(kappa.edge_id))
    kappa = kappa[keep_kappa]
    lambd = lambd[keep_lambd]

    # Filter out heavy chains that are not present in kappa or lambda chains
    keep_heavy = (heavy.edge_id.isin(kappa.edge_id)) | (heavy.edge_id.isin(lambd.edge_id))
    heavy = heavy[keep_heavy]

    # Merge heavy and light chains on edge_id
    merge_cols = ["sample_id", "family", "parent_name", "child_name", "edge_id"]
    keep_cols = [
        "parent",
        "child",
        "branch_length",
        "depth",
        "distance",
        "v_gene",
        "cdr1_codon_start",
        "cdr1_codon_end",
        "cdr2_codon_start",
        "cdr2_codon_end",
        "cdr3_codon_start",
        "cdr3_codon_end",
        "parent_is_naive",
        "child_is_leaf",
    ]

    keep_cols = merge_cols + keep_cols

    heavy_kappa = pd.merge(
        heavy[keep_cols],
        kappa[keep_cols],
        on=merge_cols,
        how="inner",
        suffixes=("_heavy", "_light"),
    )
    heavy_lambd = pd.merge(
        heavy[keep_cols],
        lambd[keep_cols],
        on=merge_cols,
        how="inner",
        suffixes=("_heavy", "_light"),
    )

    # Final dataframe with all parent to child (heavy+light) edges
    full_df = pd.concat([heavy_kappa, heavy_lambd], axis=0)

    # Select edges for a specific family (host + clonal family)
    family = "d4_203694-igk-203694"
    edges = full_df[full_df.family == family]

    # Translate the DNA sequences to amino acid sequences
    edges["parent_heavy"] = edges["parent_heavy"].progress_apply(lambda x: str(Seq(x).translate()))
    edges["parent_light"] = edges["parent_light"].progress_apply(lambda x: str(Seq(x).translate()))
    edges["child_heavy"] = edges["child_heavy"].progress_apply(lambda x: str(Seq(x).translate()))
    edges["child_light"] = edges["child_light"].progress_apply(lambda x: str(Seq(x).translate()))

    # Rename columns to match the expected format for ete3
    edges = edges.rename(columns={"branch_length_heavy": "branch_length"})
    edges["branch_length"] /= edges["branch_length"].mean()

    # Create a tree object from the edges by finding a root with no incoming edges and parent-child relationships
    tree = df_to_ete3_tree(edges[["parent_name", "child_name", "branch_length"]])

    # Find the heavy+light sequence of the root node (naive parent)
    root_data = edges[edges.parent_name == "naive"].iloc[0][["parent_heavy", "parent_light"]]
    root_seq = root_data.parent_heavy + ("G" * 10) + root_data.parent_light

    # Find the leaf nodes and their sequences
    leaf_data = edges[edges.child_name.str.contains("contig")][
        ["child_name", "child_heavy", "child_light"]
    ]
    leaf_data["true"] = leaf_data.child_heavy + ("G" * 10) + leaf_data.child_light

    tree.write()

    print(tree.get_ascii())

    model_checkpoint_path = Path(
        "/scratch/users/milind_jagota/bcr/models/peint/joint/epoch=28-step=12000.ckpt"
    )
    assert (
        model_checkpoint_path.exists()
    ), f"Model checkpoint {model_checkpoint_path} does not exist."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_from_old_checkpoint(
        checkpoint_path=model_checkpoint_path,
        device=device,
    )
    vocab = model.net.vocab

    simulation_args = {
        "model": model,
        "vocab": vocab,
        "root_sequence": root_seq,
        "tree": tree,
        "device": device,
        "max_decode_steps": len(root_seq) + 1,
        "max_batch_size": 128,
        "n_sequences": 4,
        "p_threshold": 1.0,
        "length_criterion": lambda x: x == len(root_seq),
        "likelihood_fn": None,
        "max_retries": 3,
        "seed": 42,
    }

    # Simulate evolution on the tree with rejection sampling based on sequence length
    for i in tqdm(range(50)):
        simulation_args["seed"] = i
        output = simulate_evolution_with_rejection_sampling(**simulation_args)
        sim_leaves = [output[node] for node in leaf_data.child_name]
        leaf_data["sim_{0}".format(i)] = sim_leaves

    leaf_data.to_csv("real_and_sim_leaves.csv", index=False)
    print(leaf_data)


if __name__ == "__main__":
    main()
