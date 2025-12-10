import os
from tqdm import tqdm
from pathlib import Path
from cherryml.io import convert_newick_to_CherryML_Tree


if __name__ == "__main__":
    tree_dir = Path("/accounts/projects/yss/stephen.lu/peint-workspace/main/data/lept/wyatt/trees")
    all_newick_files = list(tree_dir.glob("*.newick"))
    all_newick_files = sorted(all_newick_files)
    for newick_file in tqdm(all_newick_files, desc="Converting Newick to CherryML Tree"):
        print(f"Processing file: {newick_file}")
        cherryml_tree = convert_newick_to_CherryML_Tree(str(newick_file))
        output_file = newick_file.with_suffix(".cherryml")
        with open(output_file, "w") as f:
            f.write(str(cherryml_tree))
        print(f"Converted {newick_file} to {output_file}")
