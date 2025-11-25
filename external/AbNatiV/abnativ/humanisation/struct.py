# (c) 2023 Sormannilab and Aubin Ramon
#
# Util functions to model and superimpose structures, used in AbNatiV upon humanisation
#
# ============================================================================

from ..model.alignment.aho_consensus import cdr1_aho_indices, cdr2_aho_indices, cdr3_aho_indices, fr_aho_indices
from ..model.alignment.mybio import renumber_Fv_pdb_file

import numpy as np
import os
import Bio



def rmsd(native_atoms: np.array, model_atoms: np.array) -> float:
    """Compute the RMSD between two np arrays of Bio.PDB.Atom atoms."""
    dist = native_atoms - model_atoms
    RMSD = np.sqrt(np.sum(dist**2) / dist.shape[0])
    return RMSD



def merge_list(listo: list) -> np.array:
    """Merge all the lists of a list together into a numpy array"""
    l = []
    [l.extend(i) for i in listo]
    return np.array(l)




def get_atoms(
    model: Bio.PDB.Model,
    atoms_to_be_aligned: list,
    chain_id: str = "H",
    with_side_chain: bool = False,
    backbone_atom_types: list = ["CA", "C", "N"],
) -> dict:
    """Get all BIO Python.atoms of atoms that have an id in the list of ids of interest,
    useful for Bio.PDB.Superimposer() or to compute RMSDs. The output might be shorter
    than the list of id of interests then. But it keeps track of who is who in the keys of
    the output dictionary.

    Parameters
    ----------
        - model: Bio.PDB.Model,
        - atoms_to_be_aligned: list of int
            List of residues numbers to keep the atoms of. Would need to
            be AHo numbers for AHo numbered PDB files.
        - chain_id: str
        - with_side_chain: bool
            If True will consider every heavy atom of a residue.
        - backbone_atom_types: list of str
            List of atoms types to take keep. For backbone should be ['CA','C','N']

    Returns
    -------
        - a dict as follows: {(H,14): [atom_CA, atom_N]}"""

    ca_atoms = dict()
    chain = model[chain_id]

    for res in chain:
        # Check if residue number (.get_id()) is in the list
        if res.get_id()[1] in atoms_to_be_aligned:
            if with_side_chain:
                ca_atoms[(res.resname, res.get_id()[1])] = [
                    atom for atom in res if "H" not in atom.name
                ]
            else:
                ca_atoms[(res.resname, res.get_id()[1])] = [
                    res[atom_type] for atom_type in backbone_atom_types
                ]

    return ca_atoms




def compute_rmsd_of_interest(
    model1: Bio.PDB.Model,
    res_nbs_1: list,
    chain1: str,
    model2: Bio.PDB.Model,
    res_nbs_2: list,
    chain2: str,
    with_side_chain: bool = False,
    backbone_atom_types: list = ["CA", "C", "N"],
) -> float:
    
    """Compute the RMSD between two sets of residues of interest from two different models.
    Options to select side chain or how to define the backbone are available.

    Parameters
    ----------
        - model1: Bio.PDB.Model,
        - res_nbs_1: list of int
            List of residues numbers to keep the atoms of in model1. Would need to
            be AHo numbers for AHo numbered PDB files.
        - chain1: str
        - model2: Bio.PDB.Model,
        - res_nbs_2: list of int
            List of residues numbers to keep the atoms of in model2. Would need to
            be AHo numbers for AHo numbered PDB files.
        - chain2: str
        - with_side_chain: bool
            If True will consider every heavy atom of a residue.
        - backbone_atom_types: list of str
            List of atoms types to consider. Relevant only when with_side_chain=False).
             e.g., for backbone should be ['CA','C','N'].

    Returns
    -------
        - rmsd as a float"""

    # Remove residues that are not the same when compute on the side chain
    if with_side_chain:
        residues_in_1 = [res.get_id()[1] for res in model1[chain1]]
        residues_in_2 = [res.get_id()[1] for res in model2[chain2]]
        new_res_nbs_1, new_res_nbs_2 = list(), list()
        for res1, res2 in zip(res_nbs_1,res_nbs_2):
            if res1 in residues_in_1 and res2 in residues_in_2:
                if model1[chain1][res1].resname == model2[chain2][res2].resname:
                    new_res_nbs_1.append(res1)
                    new_res_nbs_2.append(res2)
        res_nbs_1, res_nbs_2 = new_res_nbs_1, new_res_nbs_2

    model1_atoms = merge_list(
        get_atoms(
            model1, res_nbs_1, chain1, with_side_chain, backbone_atom_types
        ).values()
    )
    model2_atoms = merge_list(
        get_atoms(
            model2, res_nbs_2, chain2, with_side_chain, backbone_atom_types
        ).values()
    )

    if len(model1_atoms) != len(model2_atoms):
        raise ValueError(
            f"Cannot compute RMSD. For the residue numbering given, Model1 has {len(model1_atoms)} selected atoms while Model2 has {len(model2_atoms)} selected atoms."
        )

    return round(rmsd(model1_atoms, model2_atoms), 4)



def superimpose_two_models(
    fixed_model: Bio.PDB.Model,
    chain_vh_fixed: str,
    chain_vl_fixed: str,
    res_nbs_vh_fixed: list,
    res_nbs_vl_fixed: list,
    moving_model: Bio.PDB.Model,
    chain_vh_moving: str,
    chain_vl_moving: str,
    res_nbs_vh_moving: list,
    res_nbs_vl_moving: list,
    is_VHH: bool,
    take_union_ng: bool,
    fp_save_aligned_model: str,
    with_side_chain: bool = False,
    backbone_atom_types: list = ["CA", "C", "N"],
) -> float:
    """
    Superimpose a moving model on a fixed model on a set of chosen residues (via their residue number).

    If models are AHo numbered, there is take_union_ng option to take the union sets of residues with the same AHo number.
    Useful when aligning on the framework region and want to take the union of non gaps residues. If models are not
    AHo numebred this will not work.

    Save the aligned version of the moving model in fp_save_aligned_model.

    Parameters
    ----------
        - fixed_model: Bio.PDB.Model,
        - chain_vh_fixed: str,
        - chain_vl_fixed: str,
        - res_nbs_vh_fixed: list of int
            List of residues numbers to keep the atoms of in fixed model heavy chain. Would need to
            be AHo numbers for AHo numbered PDB files.
        - res_nbs_vl_fixed: list of int
            List of residues numbers to keep the atoms of in fixed model light chain. Would need to
            be AHo numbers for AHo numbered PDB files.
        - moving_model: Bio.PDB.Model,
        - chain_vh_moving: str,
        - chain_vl_moving: str,
        - res_nbs_vh_moving: list of int
            List of residues numbers to keep the atoms of in moving model heavy chain. Would need to
            be AHo numbers for AHo numbered PDB files.
        - res_nbs_vl_moving: list of int
            List of residues numbers to keep the atoms of in moving model light chain. Would need to
            be AHo numbers for AHo numbered PDB files.
        - is_VHH: bool
            Will consider the light chain if False.
        - take_union_ng: bool
            If True, take the union of residues with the same res number. Useful when aligning numbered models,
            like for framework for isntance. If models are not AHo numebred this will not work.
        - fp_save_aligned_model: str
            File path where to save the aligned moving model.
        - with_side_chain: bool
            If True will consider every heavy atom of a residue when selecting the atoms.
        - backbone_atom_types: list of str
            List of atoms types to consider. Relevant only when with_side_chain=False).
             e.g., for backbone should be ['CA','C','N'].


    Returns
    -------
        - RMSD upon alignemnt on the set of given AHo positions.
    """

    # Get atom scaffolds
    fixed_vh_ca_atoms = get_atoms(
        fixed_model,
        res_nbs_vh_fixed,
        chain_vh_fixed,
        with_side_chain,
        backbone_atom_types,
    )
    moving_vh_ca_atoms = get_atoms(
        moving_model,
        res_nbs_vh_moving,
        chain_vh_moving,
        with_side_chain,
        backbone_atom_types,
    )

    if not is_VHH:
        fixed_vl_ca_atoms = get_atoms(
            fixed_model,
            res_nbs_vl_fixed,
            chain_vl_fixed,
            with_side_chain,
            backbone_atom_types,
        )
        moving_vl_ca_atoms = get_atoms(
            moving_model,
            res_nbs_vl_moving,
            chain_vl_moving,
            with_side_chain,
            backbone_atom_types,
        )

    # Will take only the Union set of no-gap scaffold residues
    if take_union_ng:
        union_fixed_atoms, union_moving_atoms = list(), list()

        for fixed_res_name, fixed_res_id in fixed_vh_ca_atoms:
            if (fixed_res_name, fixed_res_id) in moving_vh_ca_atoms:
                union_fixed_atoms.extend(
                    fixed_vh_ca_atoms[(fixed_res_name, fixed_res_id)]
                )
                union_moving_atoms.extend(
                    moving_vh_ca_atoms[(fixed_res_name, fixed_res_id)]
                )

        if not is_VHH:
            for fixed_res_name, fixed_res_id in fixed_vl_ca_atoms:
                if (fixed_res_name, fixed_res_id) in moving_vl_ca_atoms:
                    union_fixed_atoms.extend(
                        fixed_vl_ca_atoms[(fixed_res_name, fixed_res_id)]
                    )
                    union_moving_atoms.extend(
                        moving_vl_ca_atoms[(fixed_res_name, fixed_res_id)]
                    )

    else:
        union_fixed_atoms, union_moving_atoms = merge_list(
            fixed_vh_ca_atoms.values()
        ), merge_list(moving_vh_ca_atoms.values())

        if not is_VHH:
            union_fixed_atoms = np.append(
                union_fixed_atoms, merge_list(fixed_vl_ca_atoms.values())
            )
            union_moving_atoms = np.append(
                union_moving_atoms, merge_list(moving_vl_ca_atoms.values())
            )

    # Superimposition on set of
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(union_fixed_atoms, union_moving_atoms)
    super_imposer.apply(moving_model.get_atoms())

    align_rmsd = round(super_imposer.rms, 4)

    io = Bio.PDB.PDBIO()
    io.set_structure(moving_model)
    io.save(fp_save_aligned_model)

    return round(align_rmsd, 4)




def compute_cdr_displacement_between_two_structures(name_seq1:str, fp_pdb1: str, h_chain1: str, l_chain1: str, do_renumb1: bool,
                                            name_seq2:str, fp_pdb2:str, h_chain2: str, l_chain2: str, do_renumb2: bool,
                                            is_vhh: bool=True, is_abnativ2: bool=True,
                                            o_dir_pdb_models:str='./')->float:
    
    '''Superimpose two structures (models or crystals) on their scaffold and compute the CDR displacement (RSMD).

    Parameters:
    -----------
    name_seq1 : str
        Name or identifier of the first sequence/structure (used for labeling in output).
    fp_pdb1 : str
        File path to the first PDB structure.
    h_chain1 : str
        Heavy-chain identifier in the first structure (e.g. 'H' or 'A').
    l_chain1 : str
        Light-chain identifier in the first structure (ignored if `is_vhh=True`).
    do_renumb1 : bool
        Whether to renumber the first structure to the AHo numbering scheme.

    name_seq2 : str
        Name or identifier of the second sequence/structure (used for labeling in output).
    fp_pdb2 : str
        File path to the second PDB structure.
    h_chain2 : str
        Heavy-chain identifier in the second structure.
    l_chain2 : str
        Light-chain identifier in the second structure (ignored if `is_vhh=True`).
    do_renumb2 : bool
        Whether to renumber the second structure to the AHo numbering scheme.

    is_vhh : bool, optional
        Whether the input structures are nanobodies (single-domain VHHs). 
        If True, light-chain parameters are ignored. Default is True.
    is_abnativ2 : bool, optional
        If True, check for AHo CDR gaps during renumbering (used for compatibility with AbNatiV2 pipelines).
        Default is True.
    o_dir_pdb_models : str, optional
        Output directory where the aligned model will be saved as a PDB file.
        Default is the current directory ('./').

    Returns
    -------
    float or tuple of floats
        - For VHHs (`is_vhh=True`): Returns RMSD values for Scaffold, H-CDR1, H-CDR2, and H-CDR3.  
        - For Fvs (`is_vhh=False`): Returns RMSD values for Scaffold, H-CDR1–3 and L-CDR1–3.
    '''
    
    # Renumber in AHo numbering if needed, useful when one is a crystal structure
    if do_renumb1:
        aho_fp_pdb1 = os.path.join(os.path.dirname(fp_pdb1), os.path.basename(fp_pdb1).split('.')[0] + '_aho.pdb')
        renumber_Fv_pdb_file(fp_pdb1, h_chain1, l_chain1, is_VHH=is_vhh, scheme='AHo', outfilename=aho_fp_pdb1, check_AHo_CDR_gaps=is_abnativ2)
    else: aho_fp_pdb1 = fp_pdb1
    
    if do_renumb2:
        aho_fp_pdb2 = os.path.join(os.path.dirname(fp_pdb2), os.path.basename(fp_pdb2).split('.')[0] + '_aho.pdb')
        renumber_Fv_pdb_file(fp_pdb2, h_chain2, l_chain2, is_VHH=is_vhh, scheme='AHo', outfilename=aho_fp_pdb2, check_AHo_CDR_gaps=is_abnativ2)
    else: aho_fp_pdb2 = fp_pdb2

    model1 = Bio.PDB.PDBParser().get_structure(name_seq1, aho_fp_pdb1)[0]
    model2 = Bio.PDB.PDBParser().get_structure(name_seq2, aho_fp_pdb2)[0]

    # Superimpose the frameworks
    fp_save_aligned_model = os.path.join(o_dir_pdb_models,f'{name_seq2}_aligned_with_{name_seq1}.pdb')
    scaffold_rmsd = superimpose_two_models(model1, h_chain1, l_chain1, fr_aho_indices, fr_aho_indices,
                            model2, h_chain2, l_chain2, fr_aho_indices, fr_aho_indices,
                            is_VHH=is_vhh, take_union_ng=True,fp_save_aligned_model=fp_save_aligned_model)
    print(f'-> RMSD on scaffold upon alignment on scaffold:\n--> {scaffold_rmsd}A')

    # Compute RMSD on CDRs
    h_cdr1 = round(compute_rmsd_of_interest(model1, cdr1_aho_indices, h_chain1, model2, cdr1_aho_indices, h_chain2), 4)
    h_cdr2 = round(compute_rmsd_of_interest(model1, cdr2_aho_indices, h_chain1, model2, cdr2_aho_indices, h_chain2), 4)
    h_cdr3 = round(compute_rmsd_of_interest(model1, cdr3_aho_indices, h_chain1, model2, cdr3_aho_indices, h_chain2), 4)

    if not is_vhh:
        l_cdr1 = round(compute_rmsd_of_interest(model1, cdr1_aho_indices, l_chain1, model2, cdr1_aho_indices, l_chain2), 4)
        l_cdr2 = round(compute_rmsd_of_interest(model1, cdr2_aho_indices, l_chain1, model2, cdr2_aho_indices, l_chain2), 4)
        l_cdr3 = round(compute_rmsd_of_interest(model1, cdr3_aho_indices, l_chain1, model2, cdr3_aho_indices, l_chain2), 4)

        print(f'-> RMSD on CDRs upon alignment on scaffold:\n--> H-CDR1: {h_cdr1:.2f}A / H-CDR2: {h_cdr2:.2f}A / H-CDR3: {h_cdr3:.2f}A\n--> L-CDR1: {l_cdr1:.2f}A / L-CDR2: {l_cdr2:.2f}A / L-CDR3: {l_cdr3:.2f}A\n')
        return fp_save_aligned_model, scaffold_rmsd, h_cdr1, h_cdr2, h_cdr3, l_cdr1, l_cdr2, l_cdr3

    else:
        print(f'-> RMSD on CDRs upon alignment on scaffold:\n--> H-CDR1: {h_cdr1:.2f}A / H-CDR2: {h_cdr2:.2f}A / H-CDR3: {h_cdr3:.2f}A\n')
        return fp_save_aligned_model, scaffold_rmsd, h_cdr1, h_cdr2, h_cdr3
    