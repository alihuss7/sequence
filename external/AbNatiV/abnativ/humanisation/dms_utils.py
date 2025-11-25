# (c) 2023 Sormannilab and Aubin Ramon
#
# Util functions for Deep Mutational Scanning of a given sequence using AbNatiV scoring.
#
# ============================================================================

from ..model.scoring_functions import abnativ_scoring, abnativ_scoring_paired

import matplotlib.pyplot as plt
import pandas as pd 

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from collections import defaultdict


## ENHANCED SAMPLING METHOD ##

def plot_dms_average(dms_average_dependence:list, fp_save:str) -> None:
    '''Plot the average dependence profile on the mutation 
    of all other positions within the sequence of each position obtained
    via a prior deep mutational scanning study'''

    sns.set(font_scale = 1.1)
    sns.set_style('white', {'axes.spines.right':False, 'axes.spines.top': False,
                                'xtick.bottom': True, 'ytick.left': True})

    fig, ax= plt.subplots(figsize=(6,3))

    ax.plot(dms_average_dependence,color = 'C1',alpha=0.8,linewidth = 1.8)
    ax.tick_params(left=True, bottom=True)
    ax.set_xlabel('Residue position', fontsize=15)
    ax.set_ylabel('Average\ninterdependence', fontsize=15)

    plt.tight_layout()

    #Save plots
    plt.savefig(fp_save, dpi=300, bbox_inches='tight')
 

def get_no_gap_idx(al_seq: str) -> list:
    '''Get the gap indexs of an aligned sequence'''
    no_gap_idx = list()
    for k, res in enumerate(al_seq):
        if res != '-':
            no_gap_idx.append(k)
    return no_gap_idx


def compute_dms_map(seq:str, nat: str, is_VHH:bool, fp_folder_deep_mutations:str, 
                    alphabet:list, name_seq:str):
    '''Deep Mutational Scanning study of a given sequence. 

    For a given position, each of the other positions is individually mutated into 
    all available amino acid residues (19 possibilities). Across all mutated positions 
    and all available mutations, the differences between the AbNatiV score of the mutants 
    and of the WT are calculated. These differences are then averaged into a single value 
    quantifying the position under scrutiny dependence on the mutation 
    of all other positions within the sequence. 

    Parameters
    ----------
        - seq: str
            Unaligned string sequence
        - nat: str
            Type of AbNatiV nativeness to do the study on
        - is_VHH: bool
            If True, considers the VHH seed for the alignment, more suitable when aligning nanobody sequences
        - fp_folder_deep_mutations: str
            Path where to save the mutated sequences as a single .fa file in a this folder and figures
        - alphabet: list
            A list of string with the residues composing the sequences
        - name_seq: str 

    Returns
    -------
        - ng_dms_map: np.array
            [NO GAPS] a square matrix of length len(al_wt_seq) with:
                -> rows = how the mutation to this position affect the other ones
                -> columns = how this position is affected whent he other ones are mutated
        - ng_dms_avg_dep: list
            a list of lenght len(dms_map) with the average dependence for each position of dms_map
        - dms_map: np.array
            [NO GAPS] a square matrix of length len(al_wt_seq) with:
                -> rows = how the mutation to this position affect the other ones
                -> columns = how this position is affected whent he other ones are mutated
        - dms_avg_dep: list
            a list of length len(dms_map) with the average dependence for each position of dms_map
    '''

    #Score WT
    seq_records = [SeqRecord(Seq(seq), id='single_seq')]
    wt_vh_seq_abnativ_df, wt_vh_profile_abnativ_df = abnativ_scoring(nat, seq_records, batch_size=1, mean_score_only=False, 
                                                                    do_align=True, is_VHH=is_VHH, verbose=False)
    wt_vh_bnativ_profile = wt_vh_profile_abnativ_df[f'AbNatiV {nat} Residue Score'].to_list()
    al_wt_seq = ''.join(wt_vh_seq_abnativ_df['aligned_seq'])
    
    #Generate all mutations
    fp_mut_sata = deep_mutate_to_txt(al_wt_seq, fp_folder_deep_mutations, alphabet, name_seq)
    
    #Scoring all mutations of the DMS
    seq_records =  list(SeqIO.parse(fp_mut_sata, 'fasta'))
    dms_vh_seq_abnativ_df, dms_vh_profile_abnativ_df = abnativ_scoring(nat, seq_records, mean_score_only=False, do_align=False, is_VHH=is_VHH, verbose=True)

    #Remove DMS files
    # os.remove(fp_mut_sata)
    # os.rmdir(fp_folder_deep_mutations)
    
    pos_to_profiles = defaultdict(list)

    for uniq_id in set(dms_vh_profile_abnativ_df['seq_id']): # Running through profiles
        posi_being_mutated = int(uniq_id.split('_')[-2])
        vh_mut_profile = np.asarray(dms_vh_profile_abnativ_df[dms_vh_profile_abnativ_df['seq_id'] == uniq_id][f'AbNatiV {nat} Residue Score'].to_list(), dtype=float)
        pos_to_profiles[posi_being_mutated].append(vh_mut_profile)

    # Pre-convert WT profile to array once
    wt_vh_bnativ_profile = np.asarray(wt_vh_bnativ_profile, dtype=float)
    seq_len = len(al_wt_seq)

    # Compute mean profiles per position efficiently
    mean_profile_pposi = np.empty((seq_len, len(wt_vh_bnativ_profile)), dtype=float)
    no_gap_idx = get_no_gap_idx(al_wt_seq)

    for pos in range(seq_len):

        profiles = pos_to_profiles.get(pos, [])
        if len(profiles) == 0:
            # No mutants found for this position → fallback to WT
            mean_profile_pposi[pos] = wt_vh_bnativ_profile
        else:
            mean_profile_pposi[pos] = np.mean(profiles, axis=0)

    # Compute baseline & maps
    wt_baseline = np.tile(wt_vh_bnativ_profile, (seq_len, 1))
    dms_map = np.abs(mean_profile_pposi - wt_baseline)
    ng_dsm_map = dms_map[:, no_gap_idx][no_gap_idx, :]
    
    #Computing the median dependence when the other positions are mutated
    dms_avg_dep = give_averaged_position_dependence_dms(dms_map)
    ng_dms_avg_dep = give_averaged_position_dependence_dms(ng_dsm_map) #No gaps
    
    #Plot the average dependence profile
    fp_save = os.path.join(fp_folder_deep_mutations, f'{name_seq}_{nat}_avg_dms_dep.png')
    plot_dms_average(ng_dms_avg_dep, fp_save)

    return ng_dsm_map, ng_dms_avg_dep, dms_map, dms_avg_dep


def compute_dms_map_paired(df_wt:pd.DataFrame, fp_folder_deep_mutations:str, alphabet:list, name_seq:str):
    '''Deep Mutational Scanning study of a given sequence. 

    For a given position, each of the other positions is individually mutated into 
    all available amino acid residues (19 possibilities). Across all mutated positions 
    and all available mutations, the differences between the AbNatiV score of the mutants 
    and of the WT are calculated. These differences are then averaged into a single value 
    quantifying the position under scrutiny dependence on the mutation 
    of all other positions within the sequence. 

    Parameters
    ----------
        - df_pairs: pd.DataFrame
            Dataframe with paired sequences. ids in 'ID', vh_seq in 'vh_seq', vl_seq in 'vl_seq'
        - fp_folder_deep_mutations: str
            Path where to save the mutated sequences as a single .fa file in a this folder and figures
        - alphabet: list
            A list of string with the residues composing the sequences
        - name_seq: str 

    Returns
    -------
        - ng_dms_map: np.array
            [NO GAPS] a square matrix of length len(al_wt_seq) with:
                -> rows = how the mutation to this position affect the other ones
                -> columns = how this position is affected whent he other ones are mutated
        - ng_dms_avg_dep: list
            a list of lenght len(dms_map) with the average dependence for each position of dms_map
        - dms_map: np.array
            [NO GAPS] a square matrix of length len(al_wt_seq) with:
                -> rows = how the mutation to this position affect the other ones
                -> columns = how this position is affected whent he other ones are mutated
        - dms_avg_dep: list
            a list of length len(dms_map) with the average dependence for each position of dms_map
    '''

    #Score WT
    wt_vh_seq_abnativ_df, wt_vh_profile_abnativ_df = abnativ_scoring_paired(df_wt, batch_size=1, mean_score_only=False, do_align=True, verbose=False)
    wt_vh_bnativ_profile = wt_vh_profile_abnativ_df[f'AbNatiV VPaired2 Residue Score'].to_list()
    al_wt_seq_h = ''.join(wt_vh_seq_abnativ_df['aligned_seq_vh'])
    al_wt_seq_l = ''.join(wt_vh_seq_abnativ_df['aligned_seq_vl'])
    al_wt_seq = al_wt_seq_h + al_wt_seq_l

    #Generate all mutations
    fp_mut_sata_h = deep_mutate_to_txt(al_wt_seq_h, fp_folder_deep_mutations, alphabet, name_seq + '_h')
    fp_mut_sata_l = deep_mutate_to_txt(al_wt_seq_l, fp_folder_deep_mutations, alphabet, name_seq + '_l')

    seq_records_h =  list(SeqIO.parse(fp_mut_sata_h, 'fasta'))
    seq_records_l =  list(SeqIO.parse(fp_mut_sata_l, 'fasta'))

    list_ids, list_vh_dms, list_vl_dms = list(), list(), list()
    for l, rec in enumerate(seq_records_h):
        list_ids.append(rec.id)
        list_vh_dms.append(str(rec.seq))
        list_vl_dms.append(al_wt_seq_l)

    for l, rec in enumerate(seq_records_l):
        new_id = rec.id.split('_')
        new_id[-2] = str(int(new_id[-2]) + 149)
        list_ids.append('_'.join(new_id))
        list_vh_dms.append(al_wt_seq_h)
        list_vl_dms.append(str(rec.seq))

    df_dms = pd.DataFrame({'ID': list_ids, 'vh_seq': list_vh_dms, 'vl_seq': list_vl_dms})

    #Scoring all mutations of the DMS
    dms_vh_seq_abnativ_df, dms_vh_profile_abnativ_df = abnativ_scoring_paired(df_dms, mean_score_only=False, do_align=False, verbose=True)

    #Remove DMS files
    # os.remove(fp_mut_sata_h)
    # os.remove(fp_mut_sata_l)

    # Precompute a mapping from position -> list of mutation profiles
    pos_to_profiles = defaultdict(list)

    for uniq_id in set(dms_vh_profile_abnativ_df['seq_id']): # Running through profiles
        posi_being_mutated = int(uniq_id.split('_')[-2])
        vh_mut_profile = np.asarray(dms_vh_profile_abnativ_df[dms_vh_profile_abnativ_df['seq_id'] == uniq_id]['AbNatiV VPaired2 Residue Score'].to_list(), dtype=float)
        pos_to_profiles[posi_being_mutated].append(vh_mut_profile)

    # Pre-convert WT profile to array once
    wt_vh_bnativ_profile = np.asarray(wt_vh_bnativ_profile, dtype=float)
    seq_len = len(al_wt_seq)

    # Compute mean profiles per position efficiently
    mean_profile_pposi = np.empty((seq_len, len(wt_vh_bnativ_profile)), dtype=float)
    no_gap_idx = get_no_gap_idx(al_wt_seq)

    for pos in range(seq_len):

        profiles = pos_to_profiles.get(pos, [])
        if len(profiles) == 0:
            # No mutants found for this position → fallback to WT
            mean_profile_pposi[pos] = wt_vh_bnativ_profile
        else:
            mean_profile_pposi[pos] = np.mean(profiles, axis=0)

    # Compute baseline & maps
    wt_baseline = np.tile(wt_vh_bnativ_profile, (seq_len, 1))
    dms_map = np.abs(mean_profile_pposi - wt_baseline)
    ng_dsm_map = dms_map[:, no_gap_idx][no_gap_idx, :]
        

    #Computing the median dependence when the other positions are mutated
    dms_avg_dep = give_averaged_position_dependence_dms(dms_map)
    ng_dms_avg_dep = give_averaged_position_dependence_dms(ng_dsm_map) #No gaps
    
    #Plot the average dependence profile
    fp_save = os.path.join(fp_folder_deep_mutations, f'{name_seq}_avg_paired_dms_dep.png')
    plot_dms_average(ng_dms_avg_dep, fp_save)

    return ng_dsm_map, ng_dms_avg_dep, dms_map, dms_avg_dep


def give_averaged_position_dependence_dms(dms_map: np.array) -> list:
    '''Average at each position their dependence when everyother positions 
    are mutated (so no diag)

    Parameters
    ----------
        - dms_map : np.array
            Deep mutational scanning map, a square matrix with
                -> rows = how the mutation to this position affect the other ones
                -> columns = how this position is affected whent he other ones are mutated

    Returns
    -------
        - a list of lenght len(dms_map) 
        with the average dependence for each position of dms_map
    '''

    # Exclude the diagonal
    mask_diag=~np.eye(len(dms_map),dtype=bool)

    # loop on columns
    vals= [c[mask_diag.T[j]] for j,c in enumerate(dms_map.T)]
    means=[np.mean(va) for va in vals]

    return means

def deep_mutate_to_txt(seq_aligned_wt:str, save_deep_mut_folder:str, 
                       alphabet:list, name_seq:str) -> str:
    
    '''Deep mutate a given aligned sequence over the whole alphabet and save it 
    as a fasta file in a folder

    Parameters
    ----------
        - seq_aligned_wt: a string sequence
        - save_deep_mut_folder: a string path to save the mutated sequences as a single .fa file in a this folder
        - alphabet: a list of string with the residues composing the sequences
        - name_seq: str nme of the sequence

    Returns
    -------
        - the filename of the saved fasta file
    '''
    
    if not os.path.exists(save_deep_mut_folder):
        os.makedirs(save_deep_mut_folder)

    mutated_seqs = []
    mutated_seqs.append(seq_aligned_wt)
    len_seq_aligned = len(seq_aligned_wt)

    fp_mut_data =  os.path.join(save_deep_mut_folder, 'dms_' + name_seq + '.fa')
    
    with open(fp_mut_data, 'w') as f:
        for i in range(len_seq_aligned):
            single_mut_seq = list(seq_aligned_wt)
            wt_res = single_mut_seq[i]
            if wt_res != '-':
                for k, res in enumerate(alphabet):
                    if res != wt_res and res != '-':
                        single_mut_seq[i] = res
                        f.write(f'>{name_seq}_{i}_{k}\n')
                        f.write(''.join(single_mut_seq) + '\n')
    
    return fp_mut_data





