# (c) 2023 Sormannilab and Aubin Ramon
#
# Humanisation of VH/VL sequences using AbNatiV VH and VL assesments.
#
# ============================================================================

from .model.utils import is_protein
from .model.onehotencoder import alphabet
from .model.alignment.mybio import anarci_alignments_of_Fv_sequences_iter
from .model.scoring_functions import fr_aho_indices
from .humanisation.vh_vl_humanisation_functions import abnativ_vh_vl_humanisation
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

import argparse

def run(args: argparse.Namespace):

    if is_protein(args.input_seq_vh, alphabet) and is_protein(args.input_seq_vl, alphabet):

        if '2' not in args.nat_vl:
            # Find Fv chain type to use the adequate unpaired model from AbNatiV v1
            seq_records = [SeqRecord(Seq(args.input_seq_vl), id='light_chain')]
            VH, VK, VL, failed, mischtype = anarci_alignments_of_Fv_sequences_iter(seq_records, isVHH=False, verbose=False)

            if len(VK)>0:
                nat_vl = 'VKappa'
            elif len(VL)>0:
                nat_vl = 'VLambda'
            else:
                raise ValueError(f'Could not compute automtically the light_Fv_type (VKappa, VLambda) of the input light sequence')

        else:
            nat_vl = args.nat_vl

        abnativ_vh_vl_humanisation(args.input_seq_vh, args.input_seq_vl, args.output_id, args.nat_vh, nat_vl, args.output_directory, args.pdb_file, args.ch_id_vh,
                                args.ch_id_vl, fr_aho_indices, fr_aho_indices, args.is_Exhaustive, args.threshold_abnativ_score, args.threshold_rasa_score, 
                                args.forbidden_mut, verbose=True)

    else: 
        raise Exception(f'The protein sequences you gave are not composed only of {alphabet}.')