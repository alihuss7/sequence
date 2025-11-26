"""
 Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0

 NanoMelt's model
"""


from .embedding.fasta_embed import embed_fasta

import numpy as np
import pandas as pd 
import os
import joblib

from tqdm import tqdm


import importlib.resources as pkg_resources
from sklearn.exceptions import InconsistentVersionWarning
import warnings

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

def predict(df_embed: pd.DataFrame, dict_submodels: dict,
			name_ens_model: str, folder_saved_models: str) -> list:
	'''
	Run the prediction from the ensemble model based on pre-computed embeded sequences.
	
	Parameters
    ----------
        - df_embed: pd.DataFrame
            Filepath to the fasta file of input sequences to compute the embeddings on.
        - dict_submodels: dict
            Dict of the pre-trained submodels. 
			Keys = name of saved submodel in folder_saved_models.
			Values = name of the embedding needed as inputs.
		- name_ens_model: str
			Name of the pre-trained ensemble model in folder_saved_models.
		- folder_saved_models: str
            Filepath tot he fodler with the saved models.

    Returns
    ----------
        - predictions: list
	'''
	
	# Predict for each embedding (retrained model)
	list_tms = []
	for sub_model in dict_submodels:
		fp_trained_model = os.path.join(folder_saved_models, sub_model + '.joblib')
		embeddings = df_embed[dict_submodels[sub_model]].values

		# Load model
		with open(fp_trained_model,"rb") as f:
			model = joblib.load(f)

		# Predict
		tms = model.predict(np.vstack(embeddings))
		assert len(tms) > 0, f"ERROR: Failed to predict for {dict_submodels[sub_model]}"
		list_tms.append(tms)
	
	# Combine embeddings
	list_tms = np.array(list_tms).T
	tms_features = np.vstack(list_tms)
	seq_features = np.vstack(df_embed['esm1b'].values)
	
	# Predict with ensemble
	fp_model_ensemb = os.path.join(folder_saved_models, name_ens_model + '.joblib')
	with open(fp_model_ensemb,"rb") as f:
		ensemb_model = joblib.load(f)
	pred = ensemb_model.predict(tms_features)

	return pred


def NanoMeltPredPipe(seq_records: list,
			 		do_align:bool, ncpus: int,  
					batch_size:int = 420, verbose:bool = False) -> pd.DataFrame:
	
	'''Run the full melting temperature prediciton pipeline (embeddding and prediction).
	
	Parameters
    ----------
        - seq_records: list
            List of SeqRecords from the BioPython package. seq = str(record.seq) / id = record.id
		- do_align: bool
			If True, will align all the sequences using ANARCI (as implemented in AbNatiV).
		- ncpus: int
			Number of CPUs to use to paralelise the alignement and the embedding step of the dataset.
		- batch_size: int
			Batch size used to compute the embeddings on the sequences. 
			To avoid filling up all the memory with the embeddings
			when scoring a lot of sequences.
		- verbose: bool
			If True, will print more information about every step.

    Returns
    ----------
        - A dataframe with the sequences and the corresponding predicted temperatures
			and predicted errors. 
	
	'''
	# Attention order of keys matters in dict_submodels, the same order of predictions will be given to the ensemble model
	# NanoMelt models as in the original paper, Keys = name of saved submodel in folder_saved_models / Values = name of the embedding needed as inputs.

	# NanoMelt final trained model  as in the original paper
	with pkg_resources.path('nanomelt.model.saved_models', 'NanoMelt_finalmodel') as model_path:
		folder_saved_models = str(model_path)

	dict_submodels = {
		'SVR_esm1b': 'esm1b',
		'SVR_esm2_t30': 'esm2_t30',
		'GPR_vhse': 'vhse',
		'GPR_onehot': 'onehot',
					}

	# Batch over seq_records not to overload memory usage
	all_predictions = list()
	all_al_seq, all_ids, all_seqs = [],[],[]
	for batch_idx in tqdm(range(0, len(seq_records), batch_size), disable=not verbose):
		# Embed fasta
		batch_end = min(batch_idx + batch_size, len(seq_records))
		if verbose: print(f'-> Processing the sequences {batch_idx}-{batch_end}')

		batch_seq_records = seq_records[batch_idx:batch_end]
		df_embed = embed_fasta(batch_seq_records, set(dict_submodels.values()), batch_size, 
						 do_align=do_align, ncpus=ncpus, verbose=verbose)

		# Predict
		predictions = predict(df_embed, dict_submodels, 'ridge_ensemb', folder_saved_models)
		all_predictions.extend(predictions)
		all_al_seq.extend(df_embed['al_seq'].tolist())
		all_ids.extend(df_embed['id'].tolist())
		all_seqs.extend(df_embed['seq'].tolist())

	all_predictions = [round(t, 2) for t in all_predictions]
	
	#data = {'ID': df_embed['id'], 'Aligned Sequence': df_embed['al_seq'], 
	#		 'Sequence': df_embed['seq'], "NanoMelt Tm (C)": all_predictions}
	data = {'ID': all_ids, 'Aligned Sequence': all_al_seq, 
		 'Sequence': all_seqs, "NanoMelt Tm (C)": all_predictions}
	return pd.DataFrame(data)

