"""
 Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0

 Generating corresponding embeddings for NanoMelt
"""

import esm
print('Loading ESM models')
esm_models_dict={'esm1b':esm.pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S'),
                    'esm2_t30':esm.pretrained.load_model_and_alphabet('esm2_t30_150M_UR50D'),
                    }
print('ESM loading complete\n')


import numpy as np
import time
from typing import Tuple

import pandas as pd
from pandas.api.types import CategoricalDtype


import ablang
import torch
import concurrent.futures
from ..modules.alignment.mybio import anarci_alignments_of_Fv_sequences_iter

from ImmuneBuilder import NanoBodyBuilder2
from antiberty import AntiBERTyRunner

from transformers import pipeline, RobertaTokenizer

from functools import wraps
from typing import List
from tqdm import tqdm

def time_it(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        verbose = kwargs.get('verbose', False)
        if verbose:
            print(f'Time taken by {func.__name__} is {time.time() - start}')

        return result
    return wrapper
    
#### SAVED INFO FOR EMBEDDINGS ####
# Alphabet for one hot
alphabet = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','-']

# From reference VHSE-based predictor 10.1371/journal.pone.0074506
vhse_scale_dict = {'A': [0.15, -1.11, -1.35, -0.92, 0.02, -0.91, 0.36, -0.48],
                    'C': [0.18, -1.67, -0.46, -0.21, 0.0, 1.2, -1.61, -0.19],
                    'D': [-1.15, 0.67, -0.41, -0.01, -2.68, 1.31, 0.03, 0.56],
                    'E': [-1.18, 0.4, 0.1, 0.36, -2.16, -0.17, 0.91, 0.02],
                    'F': [1.52, 0.61, 0.96, -0.16, 0.25, 0.28, -1.33, -0.2],
                    'G': [-0.2, -1.53, -2.63, 2.28, -0.53, -1.18, 2.01, -1.34],
                    'H': [-0.43, -0.25, 0.37, 0.19, 0.51, 1.28, 0.93, 0.65],
                    'I': [1.27, -0.14, 0.3, -1.8, 0.3, -1.61, -0.16, -0.13],
                    'K': [-1.17, 0.7, 0.7, 0.8, 1.64, 0.67, 1.63, 0.13],
                    'L': [1.36, 0.07, 0.36, -0.8, 0.22, -1.37, 0.08, -0.62],
                    'M': [1.01, -0.53, 0.43, 0.0, 0.23, 0.1, -0.86, -0.68],
                    'N': [-0.99, 0.0, -0.37, 0.69, -0.55, 0.85, 0.73, -0.8],
                    'P': [0.22, -0.17, -0.5, 0.05, -0.01, -1.34, -0.19, 3.56],
                    'Q': [-0.96, 0.12, 0.18, 0.16, 0.09, 0.42, -0.2, -0.41],
                    'R': [-1.47, 1.45, 1.24, 1.27, 1.55, 1.47, 1.3, 0.83],
                    'S': [-0.67, -0.86, -1.07, -0.41, -0.32, 0.27, -0.64, 0.11],
                    'T': [-0.34, -0.51, -0.55, -1.06, -0.06, -0.01, -0.79, 0.39],
                    'V': [0.76, -0.92, -0.17, -1.91, 0.22, -1.4, -0.24, -0.03],
                    'W': [1.5, 2.06, 1.79, 0.75, 0.75, -0.13, -1.01, -0.85],
                    'Y': [0.61, 1.6, 1.17, 0.73, 0.53, 0.25, -0.96, -0.52],
                    '-': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    }

if torch.cuda.is_available(): print("Transferred ESM model to GPU")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): print("Transferred ESM model to Apple MX GPU")


#### EMBEDDING FUNCTIONS ####
@time_it
def embed_esm_batch(indices: List[int], sequences: List[str], esm_models_dict: dict,
                    esm_model='esm2_t30_150M_UR50D', 
                    batch_size=128, verbose=False) -> Tuple[List[int], np.ndarray]:
    
    
    model, esm_alphabet = esm_models_dict[f'{esm_model}']
    batch_converter = esm_alphabet.get_batch_converter()

    id_layer = model.num_layers

    model.eval()  # disables drop-out for deterministic results
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model = model.to('mps')
        device = 'mps'
    else:
        device = 'cpu'

    batch_labels, batch_strs, batch_tokens = batch_converter([(str(i), seq) for i, seq in enumerate(sequences)])

    save_seq_encodings = list()
    with torch.no_grad():
        for batch_idx in range(0, len(sequences), batch_size):
            batch_end = min(batch_idx + batch_size, len(sequences))
            toks = batch_tokens[batch_idx:batch_end]
            strs_batch = batch_strs[batch_idx:batch_end]
            # if verbose: print(f"Processing batch {batch_idx + 1} ({len(strs_batch)} sequences)")

            toks = toks.to(device=device, non_blocking=True)
            results = model(toks, repr_layers=[id_layer], return_contacts=False)

            token_representations = results["representations"][id_layer].to(device="cpu")

            representations_per_seq = []
            for i, str_i in enumerate(strs_batch):
                representations_per_seq.append(token_representations[i, 1 : len(str_i) + 1].mean(0))

            seq_coding = np.array([tensor.numpy() for tensor in representations_per_seq], dtype='float64')
            save_seq_encodings.extend(seq_coding)

    return indices, save_seq_encodings, esm_model

@time_it
def embed_ablang_batch(indices: List[int], sequences: List[str], verbose=False) -> Tuple[List[int], np.ndarray]:
    heavy_ablang = ablang.pretrained("heavy") #Automatically uses CUDA if available
    heavy_ablang.freeze()
    seq_coding = np.array(heavy_ablang(sequences, mode='seqcoding'))
    return indices, seq_coding, 'ablang'

@time_it
def embed_nanobuilder_batch(indices: List[int], sequences: List[str], verbose=False) -> Tuple[List[int], List[np.ndarray]]:
    predictor = NanoBodyBuilder2()
    sliced_list_embed = []
    for seq in sequences:
        seq_dict = {'H': seq}
        nanobody = predictor.predict(seq_dict)
        seq_coding = nanobody.encodings[0].mean(0).cpu().numpy()
        sliced_list_embed.append(seq_coding)
    return indices, sliced_list_embed, 'nanobuilder'

@time_it
def embed_vhse_batch(indices: List[int], al_sequences: List[str], verbose=False) -> Tuple[List[int], List[np.ndarray]]:
    list_embedding = []
    for seq in al_sequences:
        seq_coding = np.concatenate([vhse_scale_dict[aa] for aa in seq], axis=None)
        list_embedding.append(seq_coding)
    return indices, list_embedding, 'vhse'


@time_it
def embed_antiberty_batch(indices: List[int], sequences: List[str], verbose=False) -> Tuple[List[int], List[np.ndarray]]:
    antiberty = AntiBERTyRunner() #Automatically uses CUDA if available
    list_tensor = antiberty.embed(sequences)
    list_embedding = [tensor[1:-1].mean(dim=0).cpu().numpy() for tensor in list_tensor]
    return indices, list_embedding, 'antiberty'

@time_it
def embed_nanobert_batch(indices: List[int], sequences: List[str], verbose=False) -> Tuple[List[int], List[np.ndarray]]:
    tokenizer = RobertaTokenizer.from_pretrained("NaturalAntibody/nanoBERT", return_tensors="pt")
    emb = pipeline('feature-extraction', model="NaturalAntibody/nanoBERT", tokenizer=tokenizer)
    list_embedding = [np.mean(emb(seq)[0][1:-1], axis=0) for seq in sequences]
    return indices, list_embedding, 'nanobert'

@time_it
def one_hot_align_batch(indices: List[int], al_sequences: List[str], verbose=False) -> Tuple[List[int], List[np.ndarray]]:
    sliced_list_embed = []
    for seq in al_sequences:
        res_coding = np.array((pd.get_dummies(pd.Series(list(seq)).astype(CategoricalDtype(categories=alphabet))))).astype(float)
        seq_coding = np.concatenate(res_coding, axis=None)
        sliced_list_embed.append(seq_coding)
    return indices, sliced_list_embed, 'onehot'


#### RUN FULL EMBEDDING ####
@time_it
def embed_fasta(seq_records, list_embs: list,
                batch_size: int=128, do_align: bool=True, ncpus: int=8,
                verbose: bool=False) -> pd.DataFrame:
    '''
    Take an input sequences and generate the corresponding embeddings.
    Align sequences with the module in AbNatiV if needed (used for onehot and vhse embeddings).
    Embedding generation is speed up via use of GPU when available and multiple threads. 
    (ESM, Ablang and AntiBerty are the longest ones but they can use GPUs).

    Parameters
    ----------
        - seq_records: list
            List of SeqRecords from the BioPython package. seq = str(record.seq) / id = record.id
        - list_embs: list[str]
            List of embeddings names to compute the embeddings of.
                i.e.: ['esm1b', 'esm2_t30', 'ablang', 'vhse', 'onehot', 'antiberty']
        - batch_size: int
        - do_align: bool
            If True, will align all the sequences using ANARCI (as implemented in AbNatiV).
        - ncpus: int
			Number of CPUs to use to paralelise the alignement and the embedding step of the dataset.
        - verbose: bool
			If True, will print more information about every step.

    Returns
    ----------
        - A dataframe with the saved sequences and embeddings. 
    '''

    # Load the data and align sequences if needed
    if type(seq_records) is not list and type(seq_records) is not tuple :
        seq_records=[seq_records]
    
    if type(seq_records[0]) is str :
        list_seq = [str(rec) for rec in seq_records]
        list_id = [str(j) for j in range(len(seq_records))]
    else :
        list_seq = [str(rec.seq) for rec in seq_records]
        list_id = [rec.id for rec in seq_records]

    if do_align:
        start_al = time.process_time()
        VH,_,_,_,_ = anarci_alignments_of_Fv_sequences_iter(seq_records, isVHH=True,
                                                       del_cyst_misalign=False, check_AHo_CDR_gaps=True, 
                                                       run_parallel=ncpus, verbose=verbose)
        recs = VH.to_recs()
        list_id = [rec.id for rec in recs]
        list_al_seq = [str(rec.seq) for rec in recs]
        list_seq = [seq.replace('-','') for seq in list_al_seq]
        if verbose: print('Time making alignment: ', time.process_time() - start_al)
    else:
        list_al_seq = list_seq
        list_seq = [seq.replace('-','') for seq in list_seq]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures1 = []

        if 'esm1b' in list_embs: futures1.append(executor.submit(embed_esm_batch, range(len(list_seq)), list_seq, esm_models_dict, 'esm1b', batch_size, verbose))
        if 'esm2_t30' in list_embs: futures1.append(executor.submit(embed_esm_batch, range(len(list_seq)), list_seq, esm_models_dict, 'esm2_t30', batch_size, verbose))

        all_esm1_embs, all_esm2_embs = dict(), dict()
        all_ablang_embs, all_nanobuilder_embs, all_antibery_embs = dict(), dict(), dict()
        all_nanobert_embs, all_vhse_embs, all_onehot_embs = dict(), dict(), dict() 

        # Futures2 not to send as threads as needed batches at the same time 
        if verbose: print('\nGenerating embeddings over batches')
        for i in tqdm(range(0, len(list_seq), batch_size), disable=True):
            batch_indices = list(range(i, min(i+batch_size, len(list_seq))))
            batch_seqs = list_seq[i:i+batch_size]
            batch_al_seqs = list_al_seq[i:i+batch_size]

            futures2 = []
            if 'ablang' in list_embs: futures2.append(executor.submit(embed_ablang_batch, batch_indices, batch_seqs, verbose))
            if 'vhse' in list_embs: futures2.append(executor.submit(embed_vhse_batch, batch_indices, batch_al_seqs, verbose))
            if 'antiberty' in list_embs: futures2.append(executor.submit(embed_antiberty_batch, batch_indices, batch_seqs, verbose))
            if 'onehot' in list_embs: futures2.append(executor.submit(one_hot_align_batch, batch_indices, batch_al_seqs, verbose))
            if 'nanobuilder' in list_embs: futures2.append(executor.submit(embed_nanobuilder_batch, batch_indices, batch_seqs, verbose))
            if 'nanobert' in list_embs: futures2.append(executor.submit(embed_nanobert_batch, batch_indices, batch_seqs, verbose))

            concurrent.futures.wait(futures2)

            # Collect results for the current batch
            for future in futures2:
                indices, result, embedding_type = future.result()
                if embedding_type == 'ablang':
                    all_ablang_embs.update(dict(zip(indices, result)))
                elif embedding_type == 'vhse':
                    all_vhse_embs.update(dict(zip(indices, result)))
                elif embedding_type == 'antiberty':
                    all_antibery_embs.update(dict(zip(indices, result)))
                elif embedding_type == 'onehot':
                    all_onehot_embs.update(dict(zip(indices, result)))
                elif embedding_type == 'nanobert':
                    all_nanobert_embs.update(dict(zip(indices, result)))
                elif embedding_type == 'nanobuilder':
                    all_nanobuilder_embs.update(dict(zip(indices, result)))

        for future in concurrent.futures.as_completed(futures1):
            indices, result, embedding_type = future.result()
            if embedding_type == 'esm1b':
                all_esm1_embs.update(dict(zip(indices,result)))
            elif embedding_type == 'esm2_t30':
                all_esm2_embs.update(dict(zip(indices,result)))

    df_embed = pd.DataFrame({
        'id': list_id,
        'al_seq': list_al_seq,
        'seq': list_seq})
    
    if 'esm1b' in list_embs: df_embed['esm1b'] = [all_esm1_embs[i] for i in range(len(list_seq))]
    if 'esm2_t30' in list_embs: df_embed['esm2_t30'] = [all_esm2_embs[i] for i in range(len(list_seq))]
    if 'ablang' in list_embs: df_embed['ablang'] = [all_ablang_embs[i] for i in range(len(list_seq))]
    if 'vhse' in list_embs: df_embed['vhse'] = [all_vhse_embs[i] for i in range(len(list_seq))]
    if 'antiberty' in list_embs: df_embed['antiberty'] = [all_antibery_embs[i] for i in range(len(list_seq))]
    if 'onehot' in list_embs: df_embed['onehot'] = [all_onehot_embs[i] for i in range(len(list_seq))]
    if 'nanobuilder' in list_embs: df_embed['nanobuilder'] = [all_nanobuilder_embs[i] for i in range(len(list_seq))]
    if 'nanobert' in list_embs: df_embed['nanobert'] = [all_nanobert_embs[i] for i in range(len(list_seq))]

    return df_embed

