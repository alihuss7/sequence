# (c) 2023 Sormannilab and Aubin Ramon
#
# Humanisation of VH/VL sequences using pAbNatiV2 VH and VL paired assesments.
#
# ============================================================================


from .model.utils import is_protein
from .model.onehotencoder import alphabet
from .model.scoring_functions import fr_aho_indices
from .humanisation.vh_vl_humanisation_functions import abnativ_vh_vl_humanisation_paired

import argparse

def run(args: argparse.Namespace):

    if is_protein(args.input_seq_vh, alphabet) and is_protein(args.input_seq_vl, alphabet):

        abnativ_vh_vl_humanisation_paired(args.input_seq_vh, args.input_seq_vl, args.output_id, args.output_directory, args.pdb_file, args.ch_id_vh,
                                args.ch_id_vl, fr_aho_indices, fr_aho_indices, args.threshold_abnativ_score, args.threshold_rasa_score, 
                                args.perc_allowed_decrease_pairing, args.a, args.b, args.forbidden_mut, verbose=True)
        
    else: 
        raise Exception(f'The protein sequences you gave are not composed only of {alphabet}.')
