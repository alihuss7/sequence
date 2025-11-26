"""
 Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0

 Training nanobody apparent melting temperatures with NanoMelt
"""

from .model.embedding.fasta_embed import embed_fasta
from .model.final_retrain import PipeTrain
from .model.modules.clustering import get_k_medoids_clusters


import argparse
import os

from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

import pandas as pd



def run(args: argparse.Namespace):
    
    # Used for original NanoMelt
    list_submodels = [('SVR', 'esm1b'),
                ('SVR', 'esm2_t30'),
                ('GPR', 'vhse'),
                ('GPR', 'onehot'),
                ]
    
    # Load the csv file with the sequences and experimental methods
    if os.path.isfile(args.input_csv):
        input_data = pd.read_csv(args.input_csv)

        required_columns = ["Id", "Sequence", "Experimental method", "Measured apparent melting temperature (C)"]
        if not all(column in input_data.columns for column in required_columns):
            raise ValueError("Needs to have Id, Sequence, Experimental method \
                            and Measured apparent melting temperature (C) in the input .csv file.")
        
        seq_records = list()
        for id, seq in zip(input_data['Id'], input_data['Sequence']):
            seq_records.append(SeqRecord(Seq(seq), id=id))

    else: raise FileNotFoundError('Input file not found')

    batch_size = 128

    # Get the embeddings
    list_embs = set([submodel[1] for submodel in list_submodels])
    
    df_embed = embed_fasta(seq_records, list_embs, batch_size, do_align=args.do_align, ncpus=args.ncpu)

    # Clustering for 'expmeth_clust_kmedoids_blosum' stratification 
    labels_kmedoids_blosum = get_k_medoids_clusters(df_embed['al_seq'])
    df_embed['clust_kmedoids_blosum'] = labels_kmedoids_blosum

    labels_exp_kmedoids_blosum = list()
    for k, exp in enumerate(input_data['Experimental method']):
        labels_exp_kmedoids_blosum.append(exp + '_' + str(labels_kmedoids_blosum[k]))
    df_embed['expmeth_clust_kmedoids_blosum'] = labels_exp_kmedoids_blosum

    # Training
    PipeTrain(df_embed, strat_method='expmeth_clust_kmedoids_blosum', 
              list_submodels=list_submodels,folder_save=args.output_dir)
