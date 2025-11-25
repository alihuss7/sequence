# (c) 2023 Sormannilab and Aubin Ramon
#
# Lightning testing of the AbNatiV model.
#
# ============================================================================

from .model.scoring_functions import abnativ_scoring_paired
from .model.onehotencoder import alphabet
import argparse
import os
import pandas as pd

from .model.utils import is_protein
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from .init import PRETRAINED_MODELS_DIR

def run(args: argparse.Namespace):

    # Check that models are downloaded
    if not os.path.exists(PRETRAINED_MODELS_DIR):
        raise Exception("Models not found. Please run 'abnativ update' to download the models.")

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    ## DATA SCORING ##
    batch_size = 128

    # Load the sequences
    if is_protein(args.vh, alphabet) and is_protein(args.vl, alphabet):
        df_pairs = pd.DataFrame({'ID': ['single_seq'], 'vh_seq': [args.vh], 'vl_seq': [args.vl]})
        col_id, col_vh, col_vl = 'ID', 'vh_seq', 'vl_seq'

    elif os.path.isfile(args.input_filepath_csv):
        df_pairs = pd.read_csv(args.input_filepath_csv)
        col_id, col_vh, col_vl = args.col_id, args.col_vh, args.col_vl

    else: raise Exception(f'Can not find the input file {args.input_filepath_or_seq} or if you gave a single vh and vl protein sequences make sure it is composed only of {alphabet}')

    abnativ_df_mean, abnativ_df_profile = abnativ_scoring_paired(df_pairs, col_id, col_vh, col_vl, batch_size, args.mean_score_only,
                                                          args.do_align, args.is_plotting_profiles, args.output_directory, args.output_id,
                                                          run_parall_al=args.ncpu)

    ## DATA SAVING ##
    print(f'\n-> Scores being saved in {args.output_directory}\n')
    abnativ_df_mean.to_csv(os.path.join(args.output_directory, f'{args.output_id}_abnativ_seq_scores.csv'))
    if not args.mean_score_only:
        abnativ_df_profile.to_csv(os.path.join(args.output_directory, f'{args.output_id}_abnativ_res_scores.csv'))
    if args.is_plotting_profiles:
        save_profile_fp = os.path.join(args.output_directory, f'{args.output_id}_profiles')
        print(f'\n-> Profile plots saved in {save_profile_fp}\n')






