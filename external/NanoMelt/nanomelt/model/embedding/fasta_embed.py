from functools import wraps
import concurrent.futures
import os
import torch
import pandas as pd
from typing import List, Tuple
import time



import numpy as np

from pandas.api.types import CategoricalDtype

from ..modules.alignment.mybio import anarci_alignments_of_Fv_sequences_iter

from ImmuneBuilder import NanoBodyBuilder2
from antiberty import AntiBERTyRunner
from transformers import pipeline, RobertaTokenizer
import ablang


from tqdm import tqdm


import esm
print('Loading ESM models')
esm_models_dict={'esm1b':esm.pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S'),
                    'esm2_t30':esm.pretrained.load_model_and_alphabet('esm2_t30_150M_UR50D')}

print('ESM loading complete\n')

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

if torch.cuda.is_available(): print("Will transfer ESM model to GPU")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): print("Will transfer ESM model to Apple MX GPU")



#### EMBEDDING FUNCTIONS ####
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

    print('Loading ESM data')
    batch_labels, batch_strs, batch_tokens = batch_converter(
    [(str(i), seq) for i, seq in enumerate(tqdm(sequences, total=len(sequences), disable=not verbose))]
            )       

    save_seq_encodings = list()
    with torch.no_grad():
        for batch_idx in range(0, len(sequences), batch_size):
            batch_end = min(batch_idx + batch_size, len(sequences))
            toks = batch_tokens[batch_idx:batch_end]
            strs_batch = batch_strs[batch_idx:batch_end]
            if verbose: print(f"Processing batch {batch_idx + 1} ({len(strs_batch)} sequences)")

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



def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

def _mps_barrier_and_trim():
    if _mps_available():
        try:
            torch.mps.synchronize()
            torch.mps.empty_cache()
        except Exception:
            pass

def _update_map(dct, key, idxs, vals):
    """Ensure key exists and update dct[key] with {idx: val} for each pair."""
    key = str(key).lower()
    if key not in dct:
        dct[key] = {}
    dct[key].update(dict(zip(idxs, vals)))

def _col_list_from_map(dct, key, N):
    """Build list of length N from map[key] (idx->val), filling missing with None."""
    key = str(key).lower()
    mp = dct.get(key, {})
    return [mp.get(i, None) for i in range(N)]

# If you had @time_it on your original function, keep it here
# @time_it
def embed_fasta(seq_records,
                list_embs: list,
                batch_size: int = 128,
                do_align: bool = True,
                ncpus: int = 8,
                is_VHH: bool = True,
                verbose: bool = False) -> pd.DataFrame:
    """
    Generate requested embeddings for the given sequences.

    Behaviour change (MPS only): PyTorch-backed embedders (ESM, Ablang, AntiBERTy, NanoBERT)
    are executed sequentially to avoid MPS thread-safety issues. CPU embedders
    (onehot, vhse, nanobuilder if CPU-only) may still run in threads.

    Returns
    -------
    pd.DataFrame with columns:
      - 'id'                : SeqRecord.id (post-alignment if do_align=True)
      - 'al_seq'            : aligned sequence (with gaps) if do_align else original
      - 'seq'               : ungapped sequence
      - one column per requested embedder (cells contain arrays/embeddings)
    """
    if not ncpus or ncpus <= 0:
        ncpus = os.cpu_count() or 1

    # ---- IDs and sequences ----
    list_id  = [rec.id for rec in seq_records]
    list_seq = [str(rec.seq) for rec in seq_records]

    # ---- Optional alignment (for alignment-based embedders) ----
    if do_align:
        t0 = time.process_time()
        VH, VK, VL, _, _ = anarci_alignments_of_Fv_sequences_iter(
            seq_records,
            isVHH=is_VHH,
            del_cyst_misalign=False,
            check_AHo_CDR_gaps=True,
            run_parallel=ncpus,
            verbose=verbose,
            check_terms=False
        )
        recs = VH.to_recs()
        recs.extend(VK.to_recs())
        recs.extend(VL.to_recs())
        list_id     = [rec.id for rec in recs]
        list_al_seq = [str(rec.seq) for rec in recs]
        # remove gaps for non-aligned embedders
        list_seq    = [s.replace('-', '') for s in list_al_seq]
        if verbose:
            print('Time making alignment:', time.process_time() - t0)
    else:
        list_al_seq = list_seq[:]                 # no gaps provided; keep original
        list_seq    = [s.replace('-', '') for s in list_seq]

    N = len(list_seq)
    all_indices = list(range(N))

    # ---- Storage dict: name -> {idx: embedding} ----
    all_cols = {}

    # ---- ESM models over the whole set (each call batches internally) ----
    wants_esm = [name for name in ('esm1b', 'esm2_t30', 'esm2_t33') if name in list_embs]
    if wants_esm:
        if _mps_available():
            # SERIAL on MPS
            for esm_name in wants_esm:
                idxs, encs, esm_model = embed_esm_batch(
                    all_indices, list_seq, esm_models_dict, esm_name,
                    batch_size=batch_size, verbose=verbose
                )
                colname = (esm_model or esm_name).lower()
                _update_map(all_cols, colname, idxs, encs)
                _mps_barrier_and_trim()
        else:
            # Thread-parallel on CUDA/CPU
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(wants_esm)) as ex:
                futs = {
                    ex.submit(
                        embed_esm_batch, all_indices, list_seq, esm_models_dict, esm_name,
                        batch_size, verbose
                    ): esm_name
                    for esm_name in wants_esm
                }
                for f in concurrent.futures.as_completed(futs):
                    idxs, encs, esm_model = f.result()
                    colname = (esm_model or futs[f]).lower()
                    _update_map(all_cols, colname, idxs, encs)

    # ---- Per-batch embedders ----
    gpu_batch = [name for name in ('ablang', 'antiberty', 'nanobert') if name in list_embs]
    cpu_batch = [name for name in ('vhse', 'onehot', 'nanobuilder') if name in list_embs]

    for start in range(0, N, batch_size):
        stop = min(start + batch_size, N)
        batch_idx      = list(range(start, stop))
        batch_seqs     = list_seq[start:stop]
        batch_al_seqs  = list_al_seq[start:stop]

        if _mps_available():
            # SERIAL for GPU-backed embedders (MPS)
            if 'ablang' in gpu_batch:
                idxs, encs, kind = embed_ablang_batch(batch_idx, batch_seqs, verbose)
                _update_map(all_cols, kind, idxs, encs)
                _mps_barrier_and_trim()
            if 'antiberty' in gpu_batch:
                idxs, encs, kind = embed_antiberty_batch(batch_idx, batch_seqs, verbose)
                _update_map(all_cols, kind, idxs, encs)
                _mps_barrier_and_trim()
            if 'nanobert' in gpu_batch:
                idxs, encs, kind = embed_nanobert_batch(batch_idx, batch_seqs, verbose)
                _update_map(all_cols, kind, idxs, encs)
                _mps_barrier_and_trim()

            # CPU embedders in threads
            if cpu_batch:
                to_submit = []
                if 'vhse' in cpu_batch:
                    to_submit.append(('vhse', embed_vhse_batch, batch_idx, batch_al_seqs, verbose))
                if 'onehot' in cpu_batch:
                    to_submit.append(('onehot', one_hot_align_batch, batch_idx, batch_al_seqs, verbose))
                if 'nanobuilder' in cpu_batch:
                    to_submit.append(('nanobuilder', embed_nanobuilder_batch, batch_idx, batch_seqs, verbose))

                if to_submit:  # <-- guard
                    num_workers = min(len(to_submit), ncpus)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
                        futs = [ex.submit(fn, idxs, seqs, v) for _, fn, idxs, seqs, v in to_submit]
                        for f in concurrent.futures.as_completed(futs):
                            idxs, encs, kind = f.result()
                            _update_map(all_cols, kind, idxs, encs)
        else:
            to_submit = []
            if 'ablang'   in gpu_batch: to_submit.append((embed_ablang_batch,   batch_idx, batch_seqs,    verbose))
            if 'antiberty' in gpu_batch: to_submit.append((embed_antiberty_batch, batch_idx, batch_seqs,    verbose))
            if 'nanobert' in gpu_batch: to_submit.append((embed_nanobert_batch,  batch_idx, batch_seqs,    verbose))
            if 'vhse'     in cpu_batch: to_submit.append((embed_vhse_batch,      batch_idx, batch_al_seqs, verbose))
            if 'onehot'   in cpu_batch: to_submit.append((one_hot_align_batch,   batch_idx, batch_al_seqs, verbose))
            if 'nanobuilder' in cpu_batch: to_submit.append((embed_nanobuilder_batch, batch_idx, batch_seqs, verbose))

            if to_submit:  # <-- guard so we never pass max_workers=0
                num_workers = min(len(to_submit), ncpus)
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
                    futs = [ex.submit(fn, idxs, seqs, v) for fn, idxs, seqs, v in to_submit]
                    for f in concurrent.futures.as_completed(futs):
                        idxs, encs, kind = f.result()
                        _update_map(all_cols, kind, idxs, encs)

    # ---- Assemble DataFrame in input order with expected column names ----
    df_embed = pd.DataFrame({
        'id':     list_id,
        'al_seq': list_al_seq,
        'seq':    list_seq,
    })

    # Add columns for whichever embedders were requested/populated
    for name in list_embs:
        col = name.lower()
        if col in all_cols and all_cols[col]:
            df_embed[col] = _col_list_from_map(all_cols, col, N)

    return df_embed




