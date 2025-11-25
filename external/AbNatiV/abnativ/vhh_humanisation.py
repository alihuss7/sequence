# (c) 2023 Sormannilab and Aubin Ramon
#
# Humanisation of VHH sequences using AbNatiV VH and VHH assesments.
#
# ============================================================================

from .model.utils import is_protein
from .model.onehotencoder import alphabet
from .model.scoring_functions import fr_aho_indices
from .humanisation.vhh_humanisation_functions import abnativ_vhh_humanisation

import os
import argparse

from Bio import SeqIO

def run(args: argparse.Namespace):

    fp_fa_or_seq = args.input_filepath_or_seq

    # If input is a single sequence
    if not os.path.isfile(fp_fa_or_seq):
        if is_protein(fp_fa_or_seq, alphabet):
            wt_seq = fp_fa_or_seq
            name_seq = args.output_id

            abnativ_vhh_humanisation(wt_seq, name_seq, args.nat_vh, args.nat_vhh, args.output_directory, args.pdb_file, args.ch_id,
                            fr_aho_indices, args.is_Exhaustive, args.threshold_abnativ_score, args.threshold_rasa_score,
                            args.perc_allowed_decrease_vhh, args.forbidden_mut, args.a, args.b, verbose=True)
        else: 
            raise Exception(f'Can not find the input file {fp_fa_or_seq} or if you gave a single protein sequence make sure it is composed only of {alphabet}')
    
    # If input is a fasta file
    else:
        for record in SeqIO.parse(fp_fa_or_seq, 'fasta'): 
            wt_seq = str(record.seq)
            name_seq = record.id

            abnativ_vhh_humanisation(wt_seq, name_seq, args.nat_vh, args.nat_vhh, args.output_directory, args.pdb_file, args.ch_id,
                            fr_aho_indices, args.is_Exhaustive, args.threshold_abnativ_score, args.threshold_rasa_score,
                            args.perc_allowed_decrease_vhh, args.forbidden_mut, args.a, args.b, verbose=True)

    
    

    





