# (c) 2023 Sormannilab and Aubin Ramon
#
# Humanisation pipelines of nanobodies of AbNatiV VH and VHH assesments
#
# ============================================================================

import os
import numpy as np 

from ..model.onehotencoder import alphabet
from ..model.alignment.mybio import print_Alignment_pap
from ..model.alignment.mybio import renumber_Fv_pdb_file
from ..model.scoring_functions import abnativ_scoring, abnativ_scoring_paired, fr_aho_indices

from .humanisation_utils import humanise_enhanced_sampling, humanise_exhaustive_sampling, predict_struct_vh_vl, humanise_enhanced_sampling_paired
from .struct import compute_cdr_displacement_between_two_structures
from .chimerax_visualition import write_chimerax_cxc_visualisation



from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

import pandas as pd

def abnativ_vh_vl_humanisation(vh_wt_seq: str, vl_wt_seq:str, name_seq: str= 'sequence', nat_vh:str='VH', nat_vl:str='VKappa', output_dir: str='abnativ_humanisation', pdb_file: str=None, ch_id_vh: str=None,
                               ch_id_vl: str=None, allowed_user_positions_h:list=fr_aho_indices, allowed_user_positions_l:list=fr_aho_indices, is_brute: bool=False, threshold_abnativ_score:float=.98,
                             threshold_rasa_score:float=0.15, forbidden_mut: list=['C','M'], seq_ref: str=None, name_seq_ref: str=None, verbose: bool=True) -> None: 
    '''Run AbNatiV humanisation pipeline on paired VH/VL Fv sequences that aims
      to increase the AbNatiV VH- and VL- hummanness of each sequence separately.

    Two sampling methods are available:
        - enhanced sampling (is_brute=False): iteratively explores
            the mutational space aiming for rapid convergence to generate a single humanised sequence.
        - exhaustive sampling (is_brute=True): It assesses all mutation combinations within 
            the available mutational space (PSSM-allowed mutations) and selects the best sequences (Pareto Front).

    See parameters for further details. 
    
    Parameters
    ----------
        - vh_wt_seq: str
            Unaligned heavy sequence string
        - vl_wt_seq: str
            Unaligned light sequence string
        - name_seq: str
        - nat_vh: str
            VH model type to compute the VH-nativeness on ['VH', 'VH2']
        - nat_vl: str
            VL model type to compute the VL-nativeness on ['VKappa', 'VLambda', 'VL2'] 
        - output_dir:str
            Directory where to save files
        - pdb_file: str
            Filepath to the pdb file of the wt_seq
        - ch_id_vh: str
            Chain id of the heavy chain in the pfb pdb_file
        - ch_id_vl: str
            Chain id of the light chain in the pfb pdb_file
        - allowed_user_aho_positions: list of int
            List of AHo positions allowed by the user to make mutation on (default: framework positions)
        - is_brute: bool
            If True, runs brute method rather than the enhanced sampling method
        - threshold_abnativ_score: float 
            Bellow the AbNatiV VH threshold score, a position is considered as a liability
        - threshold_rasa_score: float 
            Above this threshold, the residue is considered solvent exposed and is considered for mutation
        - forbidden_mut: list
            List of residues to ban for mutation i.e. ['C','M']
        - seq_ref: str 
            If None, does not plot any references in the profiles. If str, will plot it. 
        - name_seq_ref: str
        - verbose: bool
            
    Returns
    -------
        - None'''
    

    # Create folders
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    seq_dir = os.path.join(output_dir, name_seq)
    if not os.path.exists(seq_dir): os.makedirs(seq_dir)

    pdb_dir = os.path.join(seq_dir, 'structures')
    if not os.path.exists(pdb_dir): os.makedirs(pdb_dir)


    # Aligned WT sequence to take into account additional terminal residues added during the alignment process
    seq_records = [SeqRecord(Seq(vh_wt_seq), id='single_seq_vh')]
    wt_vh_seq_abnativ_df, _ = abnativ_scoring(nat_vh, seq_records, batch_size=1,mean_score_only=False, 
                                                                    do_align=True, is_VHH=True, verbose=False)
    al_vh_wt_seq = ''.join(wt_vh_seq_abnativ_df['aligned_seq'])

    seq_records = [SeqRecord(Seq(vl_wt_seq), id='single_seq_vl')]
    wt_vl_seq_abnativ_df, _ = abnativ_scoring(nat_vl, seq_records, batch_size=1,mean_score_only=False, 
                                                                    do_align=True, is_VHH=True, verbose=False)
    al_vl_wt_seq = ''.join(wt_vl_seq_abnativ_df['aligned_seq'])

    if '2' in nat_vh: is_abnativ2=True
    else: is_abnativ2=False

    # If provided, renumber PDB file into AHo numbering to compute RASA on
    if pdb_file is not None:
        aho_pdb_file = os.path.join(pdb_dir, os.path.basename(pdb_file).split('.')[0] + '_aho.pdb')
        try: 
            renumber_Fv_pdb_file(pdb_file, ch_id_vh, ch_id_vl, scheme='AHo', outfilename=aho_pdb_file, check_AHo_CDR_gaps=is_abnativ2)
        except ValueError: 
            print('Difficulties to read PDB, please make sure it is as cleaned as possible or use the structure prediction option\
                   by writing "None" in the pdb_file option')
        pdb_file = aho_pdb_file

    seq_records_vh = list()
    id_vh_wt = name_seq + '_abnativ_heavy_wt'
    seq_records_vh.append(SeqRecord(Seq(al_vh_wt_seq.replace('-','')), id=id_vh_wt))

    seq_records_vl = list()
    id_vl_wt = name_seq + '_abnativ_light_wt'
    seq_records_vl.append(SeqRecord(Seq(al_vl_wt_seq.replace('-','')), id=id_vl_wt))

    if not is_brute: # Run the enhanced pipeline

        # HEAVY HUMANISATION
        if verbose: print(f'\n>> ENHANCED SAMPLING <<\n\n## Humanise VH of {name_seq} ##\n')
        seq_hum_h =  humanise_enhanced_sampling(al_vh_wt_seq.replace('-',''), name_seq, nat_vh, False, pdb_file, ch_id_vh, seq_dir, allowed_user_positions_h,
                                        threshold_abnativ_score, threshold_rasa_score, forbidden_mut=forbidden_mut, seq_ref=seq_ref, name_seq_ref=name_seq_ref, 
                                        verbose=verbose, pdb_dir=pdb_dir, other_seq=al_vl_wt_seq.replace('-',''))
        id_hum_h = name_seq + '_abnativ_heavy_hum_enhanced'

        # LIGHT HUMANISATION
        if verbose: print(f'\n## Humanise VL of {name_seq} ##\n')
        seq_hum_l = humanise_enhanced_sampling(al_vl_wt_seq.replace('-',''), name_seq, nat_vl, False, pdb_file, ch_id_vl, seq_dir, allowed_user_positions_l,
                                        threshold_abnativ_score, threshold_rasa_score, forbidden_mut=forbidden_mut, seq_ref=seq_ref, name_seq_ref=name_seq_ref, 
                                        verbose=verbose, pdb_dir=pdb_dir, other_seq=al_vh_wt_seq.replace('-',''))
        id_hum_l = name_seq + '_abnativ_light_hum_enhanced'

        
    else: # Run the exhaustive pipeline
        if verbose: print(f'\n>> EXHAUSTIVE SAMPLING <<\n\n## Humanise VH of {name_seq} ##\n')
        df_pareto_exhaustive_h = humanise_exhaustive_sampling(al_vh_wt_seq.replace('-',''), nat_vh, False, name_seq + '_heavy', aho_pdb_file, ch_id_vh, seq_dir, pdb_dir, allowed_user_positions_h,
                                    threshold_abnativ_score, threshold_rasa_score, forbidden_mut=forbidden_mut,seq_ref=seq_ref, name_seq_ref=name_seq_ref, other_seq=al_vl_wt_seq.replace('-',''))
        
        if verbose: print(f'\n## Humanise VL of {name_seq} ##\n')
        df_pareto_exhaustive_l = humanise_exhaustive_sampling(al_vl_wt_seq.replace('-',''), nat_vl, False, name_seq + '_light', aho_pdb_file, ch_id_vl, seq_dir, pdb_dir, allowed_user_positions_l,
                                    threshold_abnativ_score, threshold_rasa_score, forbidden_mut=forbidden_mut,seq_ref=seq_ref, name_seq_ref=name_seq_ref, other_seq=al_vh_wt_seq.replace('-',''))

        # Pre-process done on the higher number of mutations
        idx_best_h = df_pareto_exhaustive_h['count_mut'].idxmax()
        seq_hum_h, max_count_h = df_pareto_exhaustive_h.loc[idx_best_h, 'aligned_seq'].replace('-', ''), df_pareto_exhaustive_h.loc[idx_best_h, 'count_mut']
        id_hum_h = name_seq + f'_abnativ_heavy_hum_exhaustive_{max_count_h}'
        
        idx_best_l = df_pareto_exhaustive_l['count_mut'].idxmax()
        seq_hum_l, max_count_l = df_pareto_exhaustive_l.loc[idx_best_l, 'aligned_seq'].replace('-', ''), df_pareto_exhaustive_l.loc[idx_best_l, 'count_mut']
        id_hum_l = name_seq + f'_abnativ_light_hum_exhaustive_{max_count_l}'
        
    seq_records_vh.append(SeqRecord(Seq(seq_hum_h), id=id_hum_h))  
    seq_records_vl.append(SeqRecord(Seq(seq_hum_l), id=id_hum_l))

    ### Pre-process the humanised results ###

    # Final AbNatiV score and save
    vh_abnativ_df_mean, _ = abnativ_scoring(nat_vh, seq_records_vh,  batch_size=2, mean_score_only=False, do_align=True, is_VHH=True)
    vl_abnativ_df_mean, _ = abnativ_scoring(nat_vl, seq_records_vl, batch_size=2, mean_score_only=False, do_align=True, is_VHH=True)

    # Print pap file 
    dict_all_seqs = {id_vh_wt:al_vh_wt_seq.replace('-',''), id_hum_h:seq_hum_h}
    print_Alignment_pap(dict_all_seqs, os.path.join(seq_dir,f'{id_hum_h}.pap'), nchar_id=18)

    dict_all_seqs = {id_vl_wt:al_vl_wt_seq.replace('-',''), id_hum_l:seq_hum_l}
    print_Alignment_pap(dict_all_seqs, os.path.join(seq_dir,f'{id_hum_l}.pap'), nchar_id=18)

    # Model WT/HUM structures
    aho_pdb_file_wt, ch_id_vh_wt, ch_id_vl_wt, _, _, _, _ = predict_struct_vh_vl(al_vh_wt_seq.replace('-',''), al_vl_wt_seq.replace('-',''), name_seq + '_abnativ_wt', pdb_dir, is_abnativ2=is_abnativ2, max_check_iter=1)
    aho_pdb_file_hum, ch_id_vh_hum, ch_id_vl_hum, _, _, _, _ = predict_struct_vh_vl(seq_hum_h, seq_hum_l, name_seq + '_abnativ_hum', pdb_dir, is_abnativ2=is_abnativ2, max_check_iter=1)
    
    # Compute CDR-RMSD between the WT model and the humanised model
    fp_aligned_wt_model, scaffold, h_cdr1, h_cdr2, h_cdr3, l_cdr1, l_cdr2, l_cdr3 = compute_cdr_displacement_between_two_structures(id_hum_h.replace('_heavy',''), aho_pdb_file_hum, ch_id_vh_hum, ch_id_vl_hum, False,
                                                                                                                       id_vh_wt.replace('_heavy',''), aho_pdb_file_wt, ch_id_vh_wt, ch_id_vl_wt, False,
                                                                                                                        is_vhh=False, is_abnativ2=is_abnativ2, o_dir_pdb_models=pdb_dir)
                                                    
    
    vh_abnativ_df_mean['RMSD (A)\nwith modelled WT\nof scaffold\nupon scaffold-superimposition'] = [np.nan, scaffold]
    vh_abnativ_df_mean['RMSD (A)\nwith modelled WT\nof H-CDR1\nupon scaffold-superimposition'] = [np.nan, h_cdr1]
    vh_abnativ_df_mean['RMSD (A)\nwith modelled WT\nof H-CDR2\nupon scaffold-superimposition'] = [np.nan, h_cdr2]
    vh_abnativ_df_mean['RMSD (A)\nwith modelled WT\nof H-CDR3\nupon scaffold-superimposition'] = [np.nan, h_cdr3]

    vl_abnativ_df_mean['RMSD (A)\nwith modelled WT\nof scaffold\nupon scaffold-superimposition'] = [np.nan, scaffold]
    vl_abnativ_df_mean['RMSD (A)\nwith modelled WT\nof L-CDR1\nupon scaffold-superimposition'] = [np.nan, l_cdr1]
    vl_abnativ_df_mean['RMSD (A)\nwith modelled WT\nof L-CDR2\nupon scaffold-superimposition'] = [np.nan, l_cdr2]
    vl_abnativ_df_mean['RMSD (A)\nwith modelled WT\nof L-CDR3\nupon scaffold-superimposition'] = [np.nan, l_cdr3]

    # Compute CDR-RMSD between the WT crystal and the humanised model
    if pdb_file is not None:
        fp_aligned_wt_crys, scaffold, h_cdr1, h_cdr2, h_cdr3, l_cdr1, l_cdr2, l_cdr3 = compute_cdr_displacement_between_two_structures(id_hum_h.replace('_heavy',''), aho_pdb_file_hum, ch_id_vh_hum, ch_id_vl_hum, False,
                                                                                                                           id_vh_wt.replace('_heavy','') + '_cryst', pdb_file, ch_id_vh, ch_id_vl, False,
                                                                                                                            is_vhh=False, is_abnativ2=is_abnativ2, o_dir_pdb_models=pdb_dir)
                                                    
        vh_abnativ_df_mean['RMSD (A)\nwith crystal WT\nof scaffold\nupon scaffold-superimposition'] = [np.nan, scaffold]
        vh_abnativ_df_mean['RMSD (A)\nwith crystal WT\nof H-CDR1\nupon scaffold-superimposition'] = [np.nan, h_cdr1]
        vh_abnativ_df_mean['RMSD (A)\nwith crystal WT\nof H-CDR2\nupon scaffold-superimposition'] = [np.nan, h_cdr2]
        vh_abnativ_df_mean['RMSD (A)\nwith crystal WT\nof H-CDR3\nupon scaffold-superimposition'] = [np.nan, h_cdr3]

        vl_abnativ_df_mean['RMSD (A)\nwith crystal WT\nof scaffold\nupon scaffold-superimposition'] = [np.nan, scaffold]
        vl_abnativ_df_mean['RMSD (A)\nwith crystal WT\nof L-CDR1\nupon scaffold-superimposition'] = [np.nan, l_cdr1]
        vl_abnativ_df_mean['RMSD (A)\nwith crystal WT\nof L-CDR2\nupon scaffold-superimposition'] = [np.nan, l_cdr2]
        vl_abnativ_df_mean['RMSD (A)\nwith crystal WT\nof L-CDR3\nupon scaffold-superimposition'] = [np.nan, l_cdr3]
    else: 
        fp_aligned_wt_crys = None
        
    vh_abnativ_df_mean.to_csv(os.path.join(seq_dir, f"{id_hum_h}.csv"))
    vl_abnativ_df_mean.to_csv(os.path.join(seq_dir, f"{id_hum_l}.csv"))


    # Print on chimera mutated residues 
    fp_chimerax_cxc = os.path.join(pdb_dir, f"{id_hum_h.replace('_heavy','')}_mut_chimerax.cxc")
    write_chimerax_cxc_visualisation(fp_aligned_wt_model, ch_id_vh_wt, ch_id_vl_wt, 
                                        aho_pdb_file_hum, ch_id_vh_hum, ch_id_vl_hum,
                                        fp_aligned_wt_crys, ch_id_vh, ch_id_vl, # If Cryst present 
                                        is_vhh=False, out_cxc_path=fp_chimerax_cxc)
    
    return vh_abnativ_df_mean, vl_abnativ_df_mean
    


def abnativ_vh_vl_humanisation_paired(vh_wt_seq: str, vl_wt_seq:str, name_seq: str= 'sequence', output_dir: str='abnativ_humanisation', pdb_file: str=None, ch_id_vh: str=None,
                               ch_id_vl: str=None, allowed_user_positions_h:list=fr_aho_indices, allowed_user_positions_l:list=fr_aho_indices, threshold_abnativ_score:float=.98,
                             threshold_rasa_score:float=0.15, percentage_pairing_decrease: float=0.1, a: float=10, b: float=1, forbidden_mut: list=['C','M'], verbose: bool=True) -> None: 
    '''Run AbNatiV humanisation pipeline on paired VH/VL Fv sequences that aims
      to increase the AbNatiV VH- and VL- hummanness jointly using the paired pAbNatiV2 model.

    One sampling method is only available:
        - enhanced sampling : iteratively explores
            the mutational space aiming for rapid convergence to generate a single humanised sequence.

    See parameters for further details. 
    
    Parameters
    ----------
        - vh_wt_seq: str
            Unaligned heavy sequence string
        - vl_wt_seq: str
            Unaligned light sequence string
        - name_seq: str
        - nat_vh: str
            VH model type to compute the VH-nativeness on ['VH', 'VH2']
        - nat_vl: str
            VL model type to compute the VL-nativeness on ['VKappa', 'VLambda', 'VL2'] 
        - output_dir:str
            Directory where to save files
        - pdb_file: str
            Filepath to the pdb file of the wt_seq
        - ch_id_vh: str
            Chain id of the heavy chain in the pfb pdb_file
        - ch_id_vl: str
            Chain id of the light chain in the pfb pdb_file
        - allowed_user_aho_positions_h: list of int
            List of AHo positions allowed by the user to make mutation on (default: framework positions)
        - allowed_user_aho_positions_l: list of int
            List of AHo positions allowed by the user to make mutation on (default: framework positions)
        - threshold_abnativ_score: float 
            Bellow the AbNatiV VH threshold score, a position is considered as a liability
        - threshold_rasa_score: float 
            Above this threshold, the residue is considered solvent exposed and is considered for mutation
        - percentage_pairing_decrease: float
            Max allowed decrease of pairing when testing mutations
        - a: float
            Used in multi-objective selection function: aΔVH+bΔPairing
        - b: float
            Used in multi-objective selection function: aΔVH+bΔPairing
        - forbidden_mut: list
            List of residues to ban for mutation i.e. ['C','M']
        - verbose: bool
            
    Returns
    -------
        - None.'''
    
    # Create folders
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    seq_dir = os.path.join(output_dir, name_seq)
    if not os.path.exists(seq_dir): os.makedirs(seq_dir)

    pdb_dir = os.path.join(seq_dir, 'structures')
    if not os.path.exists(pdb_dir): os.makedirs(pdb_dir)

    # If provided, renumber PDB file into AHo numbering to compute RASA on
    if pdb_file is not None:
        aho_pdb_file = os.path.join(pdb_dir, os.path.basename(pdb_file).split('.')[0] + '_aho.pdb')
        try: 
            renumber_Fv_pdb_file(pdb_file, ch_id_vh, ch_id_vl, scheme='AHo', outfilename=aho_pdb_file, check_AHo_CDR_gaps=True)
        except ValueError: 
            print('Difficulties to read PDB, please make sure it is as cleaned as possible or use the structure prediction option\
                   by writing "None" in the pdb_file option')
        pdb_file = aho_pdb_file

    # Run AbNatiV humanisation (enhanced pipeline only)
    if verbose: print(f'\n>> ENHANCED SAMPLING <<\n\n## Humanise VH/VL of {name_seq} ##\n')

    df_wt = pd.DataFrame({'ID': [name_seq], 'vh_seq': [vh_wt_seq], 'vl_seq': [vl_wt_seq]})
    df_hum =  humanise_enhanced_sampling_paired(df_wt, name_seq, pdb_file, ch_id_vh, ch_id_vl, seq_dir, allowed_user_positions_h, allowed_user_positions_l,
                                    threshold_abnativ_score, threshold_rasa_score, forbidden_mut=forbidden_mut, 
                                    verbose=verbose, pdb_dir=pdb_dir, percentage_pairing_decrease=percentage_pairing_decrease, a=a, b=b)
    
    id_hum, vh_seq_hum, vl_seq_hum = df_hum['ID'].iloc[0] + '_enhanced', df_hum['vh_seq'].iloc[0], df_hum['vl_seq'].iloc[0] 

    ### Pre-process the humanised results ###

    # Final AbNatiV score and save
    df_redo_scores = pd.DataFrame({'ID': [name_seq, id_hum], 'vh_seq': [vh_wt_seq, vh_seq_hum], 'vl_seq': [vl_wt_seq, vl_seq_hum]})
    abnativ_df_mean, abnativ_df_profile = abnativ_scoring_paired(df_redo_scores, batch_size=1, mean_score_only=False, do_align=True)

    # Print pap file 
    dict_all_seqs = {name_seq + '_vh': vh_wt_seq, id_hum + '_vh': vh_seq_hum}
    print_Alignment_pap(dict_all_seqs, os.path.join(seq_dir,f'{id_hum}_vh.pap'), nchar_id=18)

    dict_all_seqs = {name_seq + '_vl': vl_wt_seq, id_hum + '_vl': vl_seq_hum}
    print_Alignment_pap(dict_all_seqs, os.path.join(seq_dir,f'{id_hum}_vl.pap'), nchar_id=18)

    # Model WT/HUM structures
    aho_pdb_file_wt, ch_id_vh_wt, ch_id_vl_wt, _, _, _, _ = predict_struct_vh_vl(vh_wt_seq, vl_wt_seq, name_seq + '_abnativ_wt', pdb_dir, is_abnativ2=True, max_check_iter=1)
    aho_pdb_file_hum, ch_id_vh_hum, ch_id_vl_hum, _, _, _, _ = predict_struct_vh_vl(vh_seq_hum, vl_seq_hum, name_seq + '_abnativ_hum', pdb_dir, is_abnativ2=True, max_check_iter=1)
    
    # Compute CDR-RMSD between the WT model and the humanised model
    fp_aligned_wt_model, scaffold, h_cdr1, h_cdr2, h_cdr3, l_cdr1, l_cdr2, l_cdr3 = compute_cdr_displacement_between_two_structures(id_hum, aho_pdb_file_hum, ch_id_vh_hum, ch_id_vl_hum, False,
                                                                                                                       name_seq, aho_pdb_file_wt, ch_id_vh_wt, ch_id_vl_wt, False,
                                                                                                                        is_vhh=False, is_abnativ2=True, o_dir_pdb_models=pdb_dir)
                                                    
    
    abnativ_df_mean['RMSD (A)\nwith modelled WT\nof scaffold\nupon scaffold-superimposition'] = [np.nan, scaffold]
    abnativ_df_mean['RMSD (A)\nwith modelled WT\nof H-CDR1\nupon scaffold-superimposition'] = [np.nan, h_cdr1]
    abnativ_df_mean['RMSD (A)\nwith modelled WT\nof H-CDR2\nupon scaffold-superimposition'] = [np.nan, h_cdr2]
    abnativ_df_mean['RMSD (A)\nwith modelled WT\nof H-CDR3\nupon scaffold-superimposition'] = [np.nan, h_cdr3]

    abnativ_df_mean['RMSD (A)\nwith modelled WT\nof scaffold\nupon scaffold-superimposition'] = [np.nan, scaffold]
    abnativ_df_mean['RMSD (A)\nwith modelled WT\nof L-CDR1\nupon scaffold-superimposition'] = [np.nan, l_cdr1]
    abnativ_df_mean['RMSD (A)\nwith modelled WT\nof L-CDR2\nupon scaffold-superimposition'] = [np.nan, l_cdr2]
    abnativ_df_mean['RMSD (A)\nwith modelled WT\nof L-CDR3\nupon scaffold-superimposition'] = [np.nan, l_cdr3]

    # Compute CDR-RMSD between the WT crystal and the humanised model
    if pdb_file is not None:
        fp_aligned_wt_crys, scaffold, h_cdr1, h_cdr2, h_cdr3, l_cdr1, l_cdr2, l_cdr3 = compute_cdr_displacement_between_two_structures(id_hum, aho_pdb_file_hum, ch_id_vh_hum, ch_id_vl_hum, False,
                                                                                                                           name_seq + '_cryst', pdb_file, ch_id_vh, ch_id_vl, False,
                                                                                                                            is_vhh=False, is_abnativ2=True, o_dir_pdb_models=pdb_dir)
                                                    
        abnativ_df_mean['RMSD (A)\nwith crystal WT\nof scaffold\nupon scaffold-superimposition'] = [np.nan, scaffold]
        abnativ_df_mean['RMSD (A)\nwith crystal WT\nof H-CDR1\nupon scaffold-superimposition'] = [np.nan, h_cdr1]
        abnativ_df_mean['RMSD (A)\nwith crystal WT\nof H-CDR2\nupon scaffold-superimposition'] = [np.nan, h_cdr2]
        abnativ_df_mean['RMSD (A)\nwith crystal WT\nof H-CDR3\nupon scaffold-superimposition'] = [np.nan, h_cdr3]

        abnativ_df_mean['RMSD (A)\nwith crystal WT\nof scaffold\nupon scaffold-superimposition'] = [np.nan, scaffold]
        abnativ_df_mean['RMSD (A)\nwith crystal WT\nof L-CDR1\nupon scaffold-superimposition'] = [np.nan, l_cdr1]
        abnativ_df_mean['RMSD (A)\nwith crystal WT\nof L-CDR2\nupon scaffold-superimposition'] = [np.nan, l_cdr2]
        abnativ_df_mean['RMSD (A)\nwith crystal WT\nof L-CDR3\nupon scaffold-superimposition'] = [np.nan, l_cdr3]
    else: 
        fp_aligned_wt_crys = None
        
    abnativ_df_mean.to_csv(os.path.join(seq_dir, f"{id_hum}.csv"))


    # Print on chimera mutated residues 
    fp_chimerax_cxc = os.path.join(pdb_dir, f"{id_hum}_mut_chimerax.cxc")
    write_chimerax_cxc_visualisation(fp_aligned_wt_model, ch_id_vh_wt, ch_id_vl_wt, 
                                        aho_pdb_file_hum, ch_id_vh_hum, ch_id_vl_hum,
                                        fp_aligned_wt_crys, ch_id_vh, ch_id_vl, # If Cryst present 
                                        is_vhh=False, out_cxc_path=fp_chimerax_cxc)
    
    return abnativ_df_mean
