from evo.sequence import backtranslate, translate_sequence, rev_comp
from pyfaidx import Fasta
from Bio.Seq import Seq
from Bio.Data import CodonTable
import Levenshtein

SHANE_VGENE = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCAR"
SHANE_JGENE = "YFDYWGQGTLVTVSS"
SHANE_VGENE_START = 106675307
SHANE_JGENE_START = 105864260

hg38_fasta = '/scratch/users/milind_jagota/mutation_rate/genomes/hg38.fa'
fasta = Fasta(hg38_fasta)

def calculate_consensus_nt(consensus_aa: str, 
                           v_gene_chr: str, 
                           v_gene_start: int, 
                           v_gene_end: int, 
                           j_gene_chr: str, 
                           j_gene_start: int, 
                           j_gene_end: int, 
                           fasta: Fasta, 
                           v_gene_dir: str = "rev", 
                           j_gene_dir: str = "rev") -> str:
    if v_gene_dir != "rev" or j_gene_dir != "rev":
        raise ValueError("forward not supported")


    vgene_wt_nt = fasta[v_gene_chr][:].seq[v_gene_end:v_gene_start]
    vgene_wt_nt = rev_comp(vgene_wt_nt.upper())
    vgene_wt_aa = translate_sequence(vgene_wt_nt)

    jgene_wt_nt = fasta[j_gene_chr][:].seq[j_gene_end:j_gene_start]
    jgene_wt_nt = rev_comp(jgene_wt_nt.upper())
    jgene_wt_aa = translate_sequence(jgene_wt_nt)

    wt_str, consensus_str = vgene_wt_aa + jgene_wt_aa, consensus_aa
    wt_nt = vgene_wt_nt + jgene_wt_nt

    # 1. Split into a list of codons
    nt_codons = [wt_nt[i:i+3] for i in range(0, len(wt_nt), 3)]

    # 2. Get operations
    ops = Levenshtein.editops(wt_str, consensus_str)

    # 3. Sort by source index (src) descending. 
    # For identical src, sort by destination index (dest) descending to maintain order.
    ops.sort(key=lambda x: (x[1], x[2]), reverse=True)

    for op, src, dest in ops:
        if op == 'replace':
            nt_codons[src] = backtranslate(consensus_str[dest])
            # print(f"Replaced {wt_str[src]} with {consensus_str[dest]} at index {src}")
            
        elif op == 'insert':
            # List.insert(index, element) places element BEFORE the current index.
            # This matches how editops defines the 'src' index for insertions.
            new_codon = backtranslate(consensus_str[dest])
            nt_codons.insert(src, new_codon)
            # print(f"Inserted {consensus_str[dest]} at index {src}")
            
        elif op == 'delete':
            nt_codons.pop(src)
            # print(f"Deleted {wt_str[src]} at index {src}")

    # 4. Reconstruct the string
    final_wt_nt = "".join(nt_codons)

    # print(f"Final NT length: {len(final_wt_nt)}, AA length: {len(final_wt_nt)//3}")

    return final_wt_nt

shane_119_heavy_wt_nt = calculate_consensus_nt(
    consensus_aa="EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARYGYGYYYFDYWGQGTLVTVSS",
    v_gene_chr="chr14",
    v_gene_start=SHANE_VGENE_START,
    v_gene_end=SHANE_VGENE_START - len(SHANE_VGENE) * 3,
    j_gene_chr="chr14",
    j_gene_start=SHANE_JGENE_START,
    j_gene_end=SHANE_JGENE_START - len(SHANE_JGENE) * 3,
    fasta=fasta,
)

shane_120_heavy_wt_nt = calculate_consensus_nt(
    consensus_aa="EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARYGGYGYYYFDYWGQGTLVTVSS",
    v_gene_chr="chr14",
    v_gene_start=SHANE_VGENE_START,
    v_gene_end=SHANE_VGENE_START - len(SHANE_VGENE) * 3,
    j_gene_chr="chr14",
    j_gene_start=SHANE_JGENE_START,
    j_gene_end=SHANE_JGENE_START - len(SHANE_JGENE) * 3,
    fasta=fasta,
)

adams_wt_nt = backtranslate('E') + "gtcaaactggatgagactggaggaggcttggtgcaacctgggaggcccatgaaactctcctgtgttgcctctggattcacttttagtgactactggatgaactgggtccgccagtctccagagaaaggactggagtgggtagcacaaattagaaacaaaccttataattatgaaacatattattcagattctgtgaaaggcagattcaccatctcaagagatgattccaaaagtagtgtctacctgcaaatgaacaacttaagagttgaagacatgggtatctattactgtacgggttcttactatggtatggactactggggtcaaggaacctcagtcaccgtctcc"

# Dictionary of nucleotide wt sequences
nt_wt_seqs = {
    "koenig_heavy": "GAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCATTAGCGACTATTGGATACACTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTCGCAGGTATTACTCCTGCTGGTGGTTACACATACTACGCAGACTCCGTGAAGGGCCGGTTCACCATCTCCGCAGACACTTCCAAGAACACGGCGTATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCCGTATATTACTGTGCGAGATTCGTGTTCTTCCTGCCCTACGCCATGGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCA",
    "koenig_light": "GACATCCAGATGACCCAGTCTCCATCCTCCCTGTCTGCATCTGTAGGAGACAGAGTCACCATCACTTGCCGGGCAAGTCAGGACGTTAGCACCGCTGTAGCTTGGTATCAGCAGAAACCAGGGAAAGCCCCTAAGCTCCTGATCTATTCTGCATCCTTTTTGTATAGTGGGGTCCCATCAAGGTTCAGTGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAGTCTGCAACCTGAAGATTTTGCAACTTACTACTGTCAACAGAGTTACACTACCCCTCCCACGTTCGGCCAAGGGACCAAGGTGGAAATCAAACGT",
    "adams_heavy": adams_wt_nt.upper(),
    "shane_119_heavy": shane_119_heavy_wt_nt.upper(),
    "shane_120_heavy": shane_120_heavy_wt_nt.upper(),
    "petersen_222_heavy": "GAGGTGCAGCTGGTGGAGTCTGGGGGAGACTTGGTACAGCCGGGGGGGTCCCTGAGACTCTCCTGCGTAGTCTCTGGATTCACCTTCAGTACCTACAGTATGAACTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTTTCATACATCAGCAGTAGTAGTCTTAGTAGATACTACGCAGACTCTGTGAAGGGCCGATTCACCATCTCCAGAGACAACGCCAAGAACTCACTGTCTCTGCAACTGAACAGCCTGAGAGCCGAGGACACGGCTGTGTATTACTGTGTCAGAGGGAGCATCACCTGGCCCACCGAATATTACCTAGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCA",
    "petersen_222_light": "GAAATAGTGATGACGCAGTCCCCAGCCACCCTGTCTGTGTCTCCAGGGGAAAGAGCCACCCTCTCCTGCAGGGCCAGTCAGACTATAAGAAGCGACTTAGCCTGGTACCAGCAGAAACCTGGCCAGCCTCCCAGGCTCATCATCTATGGTGCATCCACCAGGGCCACTGGTATCCCAGCCAGGTTCAGTGGCAGTGGGTCTGGGACAGAGTTCACTCTCACCATCAGCAGCCTGCAGTCTGAAGATTCTGCAGTTTATTTCTGTCAGCAGTATAATAACTGGCCCCCCCTCACTTTCGGCGGAGGGACCAAGGTGGAGATCAAA",
    "petersen_319_heavy": "GAAGTGCAACTACAGGAGTCTGGGGGAGGTTTGGTTCGGCCTGGGGGGACCCTGAGACTCTCCTGTGCAGCCTCTGGATTCAGTTTTAGCAATTATAACATGTACTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTCTCAAGTATTAGTGGTAGTGGTCTTAGTACCTACTATGCAGACTCCGTGAAGGGTCGGTTCACCATCTCCAGAGACAAGTCCAAGAACACGGTGTATTTGCACATGAATAGCCTGCGAGCCGAGGACACGGCCCTATATTACTGTACGAAGGATTTTTCTACCTATATACCAATGACTGGTACCTTTGACTCCTGGGGCCAGGGAACCCAGGTCACCGTCTCCTCA",
    "petersen_319_light": "GAAATAGTGATGACCCAGTCTCCAGCCACCCTGTCTGTGTCTCCGGGGGAAAGAGCCACCCTCTCCTGCAGGGCCAGTCAGAGTGTTAACAGCAACTTAGCCTGGTACCAGCAGAGACCTGGCCAGGCTCCCAGGCTCCTCATCTATACTGCCTCCACCAGGGCCACTGGTATCCCAGCCAGGTTCAGTGGCAGTGGGTCTGGGACAGAGTTCACTCTCACCATCAGCAGCATACAGCCTGAAGATTTTGCAGTTTATTACTGTCAGCAGTATAGTAATTGGCCTCCGCTCACTTTCGGCGGAGGGACCAAGGTGGAGATCAAA",
}

def get_mutant(mutant: str, wildtype: str) -> str:
    assert len(mutant) == len(wildtype), "Mutant and wildtype sequences must be of the same length"
    different_indices = [i for i, (m, w) in enumerate(zip(mutant, wildtype)) if m != w]
    assert len(different_indices) <= 1, "There should be exactly one mutation"
    if len(different_indices) == 0:
        return ""
    idx = different_indices[0]
    wt = wildtype[idx]
    mt = mutant[idx]
    return f"{wt}{idx}{mt}"


def get_mutations(mutant: str, wildtype: str) -> str:
    """Get all mutations as comma-separated string (e.g., 'A105G,T107S').

    Unlike get_mutant(), this function supports any number of mutations.
    Returns empty string if sequences are identical.
    """
    assert len(mutant) == len(wildtype), "Mutant and wildtype sequences must be of the same length"
    different_indices = [i for i, (m, w) in enumerate(zip(mutant, wildtype)) if m != w]
    if len(different_indices) == 0:
        return ""
    mutations = [f"{wildtype[i]}{i}{mutant[i]}" for i in different_indices]
    return ",".join(mutations)