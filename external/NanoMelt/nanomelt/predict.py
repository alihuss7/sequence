"""
 Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0

 Predicting nanobody apparent melting temperatures with NanoMelt
"""

from .model.nanomelt import NanoMeltPredPipe
from .model.embedding.fasta_embed import alphabet

import argparse
import os,sys

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from .model.modules.utils import is_protein





def run(args: argparse.Namespace):

    if args.do_align:
        import multiprocessing
        multiprocessing.set_start_method("fork", force=True)
        
    # Load the fasta file
    if os.path.isfile(args.input_filepath_or_seq):
        seq_records =  list(SeqIO.parse(args.input_filepath_or_seq, 'fasta'))
        if args.maximum_sequences_to_process > 0 and len(seq_records) > args.maximum_sequences_to_process :
            sys.stderr.write('\n**ERROR** Given args limit at maximum_sequences_to_process=%d, but input fasta file contains %d sequences!\n ABORTING RUN\n\n' % (args.maximum_sequences_to_process,len(seq_records)))
            return 1
    # If just sequence
    elif is_protein(args.input_filepath_or_seq, alphabet):
        seq_records = [SeqRecord(Seq(args.input_filepath_or_seq), id='single_seq')]

    # Run the model
    data = NanoMeltPredPipe(seq_records, args.do_align, args.ncpu, verbose=args.verbose)

    # Save data as .csv
    if not is_protein(args.input_filepath_or_seq, alphabet):
        data.to_csv(args.output_savefp, index=False)

    if is_protein(args.input_filepath_or_seq, alphabet):
        tm = list(data['NanoMelt Tm (C)'])[0]
        al_seq = list(data['Aligned Sequence'])[0]
        print(f'\n-> Sequence aligned as:\n')
        print(al_seq)
        print(f'\n\n---> Predicted apparent melting temperature -> {tm} (C)\n\n\n\n')
    else:
        print('\n\n---> NanoMelt prediction complete and results saved in %s\n\n\n\n' % (str(args.output_savefp).split('/')[-1]))