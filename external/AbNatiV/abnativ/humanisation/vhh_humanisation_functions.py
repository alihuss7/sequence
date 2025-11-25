# (c) 2023 Sormannilab and Aubin Ramon
#
# Humanisation pipelines of nanobodies of AbNatiV VH and VHH assesments
#
# ============================================================================

import os
import numpy as np

from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from ..model.alignment.mybio import print_Alignment_pap
from ..model.alignment.mybio import renumber_Fv_pdb_file
from ..model.scoring_functions import abnativ_scoring, fr_aho_indices

from .struct import compute_cdr_displacement_between_two_structures
from .chimerax_visualition import write_chimerax_cxc_visualisation
from .humanisation_utils import humanise_enhanced_sampling, humanise_exhaustive_sampling, predict_struct_vhh


def abnativ_vhh_humanisation(wt_seq: str, name_seq: str, nat_vh:str = 'VH', nat_vhh:str = 'VHH', output_dir: str='abnativ_humanisation', pdb_file: str=None, ch_id: str=None, 
                             allowed_user_positions:list=fr_aho_indices, is_brute: bool=False, threshold_abnativ_score:float=.98,
                             threshold_rasa_score:float=0.15, perc_allowed_decrease_vhh:float=2e-2, forbidden_mut:list=['C','M'],
                             a:float=.8,b:float=.2, seq_ref: str=None, name_seq_ref: str=None, verbose: bool=True) -> None: 
    '''Run AbNatiV humanisation pipeline on a nanobody sequence with a dual-control strategy 
    that aims to increase the AbNatiV VH-hummanness of a sequence
    while retaining its VHH-nativeness.

    Two sampling methods are available:
        - enhanced sampling (is_brute=False): iteratively explores
            the mutational space aiming for rapid convergence to generate a single humanised sequence.
        - exhaustive sampling (is_brute=True): It assesses all mutation combinations within 
            the available mutational space (PSSM-allowed mutations) and selects the best sequences (Pareto Front).

    See parameters for further details. 
    
    Parameters
    ----------
        - wt_seq: str
            Unaligned sequence string 
        - name_seq: str
        - nat_vh: str
            VH model type to compute the VH-nativeness on ['VH', 'VH2']
        - nat_vhh: str
            VHH model type to compute the VHH-nativeness on ['VHH', 'VHH2']
        - output_dir:str
            Directory where to save files
        - pdb_file: str
            Filepath to the pdb file of the wt_seq
        - ch_id: str
            Chain id of the pfb pdb_file
        - allowed_user_aho_positions: list of int
            List of AHo positions allowed by the user to make mutation on (default: framework positions)
        - is_brute: bool
            If True, runs brute method rather than the enhanced sampling method
        - threshold_abnativ_score: float 
            Bellow the AbNatiV VH threshold score, a position is considered as a liability
        - threshold_rasa_score: float 
            Above this threshold, the residue is considered solvent exposed and is considered for mutation
        - perc_allowed_decrease_vhh: float
            Maximun ΔVHH score decrease allowed for a mutation
        - forbidden_mut: list
            List of residues to ban for mutation i.e. ['C','M']
        - a: float
            Used in multi-objective selection function: aΔVH+bΔVHH
        - b: float
            Used in multi-objective selection function: aΔVH+bΔVHH
        - seq_ref: str 
            If None, does not plot any references in the profiles. If str, will plot it
        - name_seq_ref: str
        - verbose: bool
            
    Returns
    -------
        - dict with keys will be AHo numbers, values will be allowed substitutions 
            according to the PSSM criteria '''

    # Create folders
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    seq_dir = os.path.join(output_dir, name_seq)
    if not os.path.exists(seq_dir): os.makedirs(seq_dir)

    pdb_dir = os.path.join(seq_dir, 'structures')
    if not os.path.exists(pdb_dir): os.makedirs(pdb_dir)

    # Aligned WT sequence to take into account additional terminal residues added during the alignment process
    seq_records = [SeqRecord(Seq(wt_seq), id='single_seq')]
    wt_vh_seq_abnativ_df, wt_vh_profile_abnativ_df = abnativ_scoring(nat_vhh, seq_records, batch_size=1,mean_score_only=False, 
                                                                    do_align=True, is_VHH=True, verbose=False)
    al_wt_seq = ''.join(wt_vh_seq_abnativ_df['aligned_seq'])

    if '2' in nat_vh: is_abnativ2=True
    else: is_abnativ2=False

    # If provided, renumber PDB file into AHo numbering to compute RASA on
    if pdb_file is not None:
        aho_pdb_file = os.path.join(pdb_dir, os.path.basename(pdb_file).split('.')[0] + '_aho.pdb')
        try: 
            renumber_Fv_pdb_file(pdb_file, ch_id, None, is_VHH=True, scheme='AHo', outfilename=aho_pdb_file, check_AHo_CDR_gaps=is_abnativ2)
        except ValueError: 
            print('Difficulties to read PDB, please make sure it is as cleaned as possible or use the structure prediction option\
                    by writing "None" in the pdb_file option')
        pdb_file = aho_pdb_file

    # Run AbNatiV humanisation
    id_wt = name_seq + '_abnativ_wt'
    seq_records = list()
    seq_records.append(SeqRecord(Seq(al_wt_seq.replace('-','')), id=id_wt))
    
    if not is_brute: 
        if verbose: print(f'\n>> ENHANCED SAMPLING <<\n\n## {name_seq} ##\n')
        seq_hum =  humanise_enhanced_sampling(al_wt_seq.replace('-',''), name_seq, nat_vh, True, pdb_file, ch_id, seq_dir, allowed_user_positions,
                                        threshold_abnativ_score, threshold_rasa_score, perc_allowed_decrease_vhh=perc_allowed_decrease_vhh, forbidden_mut=forbidden_mut,
                                        a=a, b=b, seq_ref=seq_ref, name_seq_ref=name_seq_ref, verbose=verbose, nat_vhh=nat_vhh, pdb_dir=pdb_dir)
    
        id_hum = name_seq + '_abnativ_hum_enhanced'
        

    else: 
        if verbose: print(f'\n>> EXHAUSTIVE SAMPLING <<\n\n## {name_seq} ##\n')
        df_pareto_exhaustive = humanise_exhaustive_sampling(al_wt_seq.replace('-',''), nat_vh, True, name_seq, pdb_file, ch_id, seq_dir, pdb_dir, allowed_user_positions,
                                    threshold_abnativ_score, threshold_rasa_score, nat_vhh, perc_allowed_decrease_vhh, forbidden_mut,
                                    seq_ref, name_seq_ref)
        
        # Pre-process done on the higher number of mutations
        idx_best = df_pareto_exhaustive['count_mut'].idxmax()
        seq_hum, max_count = df_pareto_exhaustive.loc[idx_best, 'aligned_seq'].replace('-', ''), df_pareto_exhaustive.loc[idx_best, 'count_mut']
        id_hum = name_seq + f'_abnativ_hum_exhaustive_{max_count}'

    ### Pre-process the humanised results ###
    seq_records.append(SeqRecord(Seq(seq_hum), id=id_hum))

    # Final AbNatiV score and save
    vh_abnativ_df_mean, _ = abnativ_scoring(nat_vh, seq_records,  batch_size=1, mean_score_only=False, do_align=True, is_VHH=True)
    vhh_abnativ_df_mean, _ = abnativ_scoring(nat_vhh, seq_records, batch_size=1, mean_score_only=False, do_align=True, is_VHH=True)
    merge_abnativ_df = vh_abnativ_df_mean.merge(vhh_abnativ_df_mean, how='inner')

    # Print pap file 
    dict_all_seqs = {id_wt:wt_seq, id_hum:seq_hum}
    print_Alignment_pap(dict_all_seqs, os.path.join(seq_dir,f'{id_hum}.pap'), nchar_id=18)

    # Model WT/HUM structures
    wt_pred_pdb_file, wt_pred_ch_id, _, _, _, _ = predict_struct_vhh(al_wt_seq.replace('-',''), id_wt, pdb_dir, is_abnativ2=is_abnativ2)
    hum_pred_pdb_file, hum_pred_ch_id, _, _, _, _ = predict_struct_vhh(seq_hum, id_hum, pdb_dir, is_abnativ2=is_abnativ2)
    
    # Compute CDR-RMSD between the WT model and the humanised model
    fp_aligned_wt_model, scaffold, h_cdr1, h_cdr2, h_cdr3 = compute_cdr_displacement_between_two_structures(id_hum, hum_pred_pdb_file, hum_pred_ch_id, None, False,
                                                                id_wt, wt_pred_pdb_file, wt_pred_ch_id, None, False,
                                                                is_vhh=True, is_abnativ2=is_abnativ2, o_dir_pdb_models=pdb_dir)
                                                                
    
    merge_abnativ_df['RMSD (A)\nwith modelled WT\nof scaffold\nupon scaffold-superimposition'] = [np.nan, scaffold]
    merge_abnativ_df['RMSD (A)\nwith modelled WT\nof H-CDR1\nupon scaffold-superimposition'] = [np.nan, h_cdr1]
    merge_abnativ_df['RMSD (A)\nwith modelled WT\nof H-CDR2\nupon scaffold-superimposition'] = [np.nan, h_cdr2]
    merge_abnativ_df['RMSD (A)\nwith modelled WT\nof H-CDR3\nupon scaffold-superimposition'] = [np.nan, h_cdr3]

    # Compute CDR-RMSD between the WT crystal and the humanised model
    if pdb_file is not None:
        fp_aligned_wt_crys, scaffold_cryst, h_cdr1_cryst, h_cdr2_cryst, h_cdr3_cryst = compute_cdr_displacement_between_two_structures(id_hum, hum_pred_pdb_file, hum_pred_ch_id, None, False,
                                                                                                id_wt + '_cryst', pdb_file, ch_id, None, False,
                                                                                                is_vhh=True, is_abnativ2=is_abnativ2, o_dir_pdb_models=pdb_dir)

        merge_abnativ_df['RMSD (A)\nwith crystal WT\nof scaffold\nupon scaffold-superimposition'] = [np.nan, scaffold_cryst]
        merge_abnativ_df['RMSD (A)\nwith crystal WT\nof H-CDR1\nupon scaffold-superimposition'] = [np.nan, h_cdr1_cryst]
        merge_abnativ_df['RMSD (A)\nwith crystal WT\nof H-CDR2\nupon scaffold-superimposition'] = [np.nan, h_cdr2_cryst]
        merge_abnativ_df['RMSD (A)\nwith crystal WT\nof H-CDR3\nupon scaffold-superimposition'] = [np.nan, h_cdr3_cryst]
    else:
        fp_aligned_wt_crys = None

    merge_abnativ_df.to_csv(os.path.join(seq_dir, f'{id_hum}.csv'))

    # Print on chimera mutated residues 
    fp_chimerax_cxc = os.path.join(pdb_dir, f"{id_hum}_mut_chimerax.cxc")
    write_chimerax_cxc_visualisation(fp_aligned_wt_model, wt_pred_ch_id, None, 
                                        hum_pred_pdb_file, hum_pred_ch_id, None,
                                        fp_aligned_wt_crys, ch_id, None, # Crystal structure if present
                                        is_vhh=True, out_cxc_path=fp_chimerax_cxc)
    
    return merge_abnativ_df