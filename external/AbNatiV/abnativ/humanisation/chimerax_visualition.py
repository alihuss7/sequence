# (c) 2023 Sormannilab and Aubin Ramon
#
# To visualise the humanised variants with Chimera X
#
# ============================================================================
import os
from typing import List, Dict, Tuple
import Bio
from Bio.PDB import PDBParser


AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V","SEC":"U","PYL":"O"
}

def _pdb_resname_to_1(resname: str) -> str:
    return AA3_TO_1.get(resname.upper().strip(), "X")

def _collect_chain_residues_aho(fp_pdb: str, chain_id: str) -> Dict[int, str]:
    """
    Read a (renumbered-to-AHo) PDB and return {resnum(AHo): one_letter_aa} for a given chain.
    Ignores insertion codes and non-standard residues.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("S", fp_pdb)
    model = structure[0]
    out = {}
    try:
        chain = model[chain_id]
    except KeyError:
        raise ValueError(f"Chain '{chain_id}' not found in {fp_pdb}")
    for res in chain.get_residues():
        het, resseq, icode = res.id  # e.g., (' ', 52, ' ')
        if het.strip():  # skip HETATM residues
            continue
        aa1 = _pdb_resname_to_1(res.get_resname())
        out[resseq] = aa1
    return out

def get_mutated_positions_from_aho_renumbered_pdbs(
    wt_pdb_aho: str, wt_chain: str,
    hum_pdb_aho: str, hum_chain: str
) -> List[Tuple[int, str, str]]:
    """
    Compare two AHo-renumbered PDBs and return list of (AHo_pos, wt_aa, hum_aa) where residues differ
    and both sides are standard amino acids.
    """
    wt_map = _collect_chain_residues_aho(wt_pdb_aho, wt_chain)
    hum_map = _collect_chain_residues_aho(hum_pdb_aho, hum_chain)
    mut_positions = []
    for pos in sorted(set(wt_map.keys()).intersection(hum_map.keys())):
        wt_aa, hum_aa = wt_map[pos], hum_map[pos]
        if wt_aa != "X" and hum_aa != "X" and wt_aa != hum_aa:
            mut_positions.append((pos, wt_aa, hum_aa))
    return mut_positions


def write_chimerax_cxc_visualisation(
    wt_pdb: str, wt_chain_h: str, wt_chain_l: str,
    hum_pdb: str, hum_chain_h: str, hum_chain_l: str,
    wt_pdb_crys: str, wt_chain_h_crys: str, wt_chain_l_crys: str,
    is_vhh: bool, 
    out_cxc_path: str,
    color_wt: str = "sandy brown",
    color_hum: str = "cornflower blue",
    color_hum_crys: str = "yellow green",
    color_mut_wt: str = "chocolate",
    color_mut_hum: str = "dark orchid",
    color_mut_wt_crys: str = "dark green",

):
    """
    Write a ChimeraX script (.cxc) that does the same as above using ChimeraX commands.
    Files need to be AHo aligned. 
    """
    lines = []
    lines.append(f'open "{os.path.basename(wt_pdb)}"')   # -> #1 in ChimeraX numbering
    lines.append(f'open "{os.path.basename(hum_pdb)}"')  # -> #2
    lines.append(f"color #1 {color_wt}")
    lines.append(f"color #2 {color_hum}")

    if wt_pdb_crys is not None: 
        lines.append(f'open "{os.path.basename(wt_pdb_crys)}"') 
        lines.append(f"color #3 {color_hum_crys}")

    lines.append("cartoon")
    h_muts = get_mutated_positions_from_aho_renumbered_pdbs(wt_pdb, wt_chain_h, hum_pdb, hum_chain_h)
    h_mutated_positions_aho = [pos for (pos, wt_aa, hum_aa) in h_muts]

    for pos in h_mutated_positions_aho:
        # style sticks; restrict to sidechain atoms with 'sidechain' selector
        lines.append(f"select #1/{wt_chain_h}:{pos}")
        lines.append(f"show sel atoms")
        lines.append(f"color sel {color_mut_wt}")
        lines.append(f"color sel byhetero")

        lines.append(f"select #2/{hum_chain_h}:{pos}")
        lines.append(f"show sel atoms")
        lines.append(f"color sel {color_mut_hum}")
        lines.append(f"color sel byhetero")

        if wt_pdb_crys is not None:
            lines.append(f"select #3/{wt_chain_h_crys}:{pos}")
            lines.append(f"show sel atoms")
            lines.append(f"color sel {color_mut_wt_crys}")
            lines.append(f"color sel byhetero")

    if not is_vhh:
        l_muts = get_mutated_positions_from_aho_renumbered_pdbs(wt_pdb, wt_chain_l, hum_pdb, hum_chain_l)
        l_mutated_positions_aho = [pos for (pos, wt_aa, hum_aa) in l_muts]

        for pos in l_mutated_positions_aho:
            # style sticks; restrict to sidechain atoms with 'sidechain' selector
            lines.append(f"select #1/{wt_chain_l}:{pos}")
            lines.append(f"show sel atoms")
            lines.append(f"color sel {color_mut_wt}")
            lines.append(f"color sel byhetero")

            lines.append(f"select #2/{hum_chain_l}:{pos}")
            lines.append(f"show sel atoms")
            lines.append(f"color sel {color_mut_hum}")
            lines.append(f"color sel byhetero")

            if wt_pdb_crys is not None:
                lines.append(f"select #3/{wt_chain_l_crys}:{pos}")
                lines.append(f"show sel atoms")
                lines.append(f"color sel {color_mut_wt_crys}")
                lines.append(f"color sel byhetero")
                

    lines.append(f"select H")
    lines.append(f"hide sel target a")
    lines.append(f"select clear")
    lines.append(f"set bgColor white")
    lines.append(f"graphics silhouettes true")

    lines.append("view")
    with open(out_cxc_path, "w") as f:
        f.write("\n".join(lines))
