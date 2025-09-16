from typing import TypeAlias

Protein: TypeAlias = str  # For now, we define a protein as a string of residues.

RESIDUES = [
    "A",  # Alanine
    "C",  # Cysteine
    "D",  # Aspartic acid
    "E",  # Glutamic acid
    "F",  # Phenylalanine
    "G",  # Glycine
    "H",  # Histidine
    "I",  # Isoleucine
    "K",  # Lysine
    "L",  # Leucine
    "M",  # Methionine
    "N",  # Asparagine
    "P",  # Proline
    "Q",  # Glutamine
    "R",  # Arginine
    "S",  # Serine
    "T",  # Threonine
    "V",  # Valine
    "W",  # Tryptophan
    "Y",  # Tyrosine
]

SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[BOS]": 2,
    "[EOS]": 3,
}

VOCAB = SPECIAL_TOKENS.copy() | {
    residue: idx + len(SPECIAL_TOKENS) for idx, residue in enumerate(RESIDUES)
}

PAD_TOKEN = VOCAB["[PAD]"]
BOS_TOKEN = VOCAB["[BOS]"]
EOS_TOKEN = VOCAB["[EOS]"]
