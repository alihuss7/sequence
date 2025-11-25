# (c) 2023 Sormannilab and Aubin Ramon
#
# AbNatiV BERT-style masking OneHotEncoder Iterator.
#
# ============================================================================

from typing import Tuple
import numpy as np
import math
import random
import pandas as pd
from pandas.api.types import CategoricalDtype
import torch
import gc
from itertools import cycle

from Bio import SeqIO

alphabet = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "-",
]


def data_loader_masking_bert_onehot(
    list_seqs: list, batch_size: int, perc_masked_residues: float, is_masking: bool
) -> torch.utils.data.DataLoader:
    """
    Generate a Torch dataloader iterator from fp_data.

    Parameters
    ----------
    list_seq: lsit of str
        List of sequences to load.
    batch_size: int
    perc_masked_residues: float
        Ratio of residues to apply the BERT masking on (between 0 and 1).
    is_masking: bool

    """
    iterator = IterableMaskingBertOnehotDataset(
        list_seqs, perc_masked_residues=perc_masked_residues, is_masking=is_masking
    )
    loader = torch.utils.data.DataLoader(iterator, batch_size=batch_size, num_workers=0)
    return loader


def data_loader_masking_bert_onehot_fasta(
    fp_data: str,
    batch_size: int,
    perc_masked_residues: float,
    is_masking: bool,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """
    Generate a Torch dataloader iterator from fp_data. Useful for training without loading the whole dataset
    Useful for training.

    Parameters
    ----------
    fp_data: str
        Filepath to the fasta file with the sequences of interest.
    batch_size: int
    perc_masked_residues: float
        Ratio of residues to apply the BERT masking on (between 0 and 1).
    is_masking: bool
    num_workers: int
        For loading parallelisation.

    """
    iterator = IterableMaskingBertOnehotDatasetFasta(
        fp_data, perc_masked_residues=perc_masked_residues, is_masking=is_masking
    )
    loader = torch.utils.data.DataLoader(
        iterator, batch_size=batch_size, num_workers=num_workers
    )
    return loader


def data_loader_masking_bert_onehot_paired(
    df_pairs: pd.DataFrame,
    col_vh: str = "vh_seq",
    col_vl: str = "vl_seq",
    batch_size: int = 128,
    perc_masked_residues: float = 0.3,
    is_masking: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Generate a Torch dataloader iterator from fp_data. Useful for training without loading the whole dataset
    Useful for training.

    Parameters
    ----------
    - df_pairs: pd.DataFrame
        Dataframe with paired sequences
    - col_vh: str
        Name of the column with the heavy sequences
    - col_vl: str:
        Name of the column with the light sequences
    batch_size: int
    perc_masked_residues: float
        Ratio of residues to apply the BERT masking on (between 0 and 1).
    is_masking: bool
    num_workers: int
        For loading parallelisation.

    """
    iterator = IterableMaskingBertOnehotDatasetPaired(
        df_pairs,
        col_vh,
        col_vl,
        perc_masked_residues=perc_masked_residues,
        is_masking=is_masking,
    )
    loader = torch.utils.data.DataLoader(iterator, batch_size=batch_size, num_workers=0)
    return loader


def data_loader_masking_bert_onehot_csv_paired(
    fp_seq: str,
    col_vh_seq: str = "vh_seq",
    col_vl_seq: str = "vl_seq",
    chunk_size: int = 100000,
    batch_size: int = 128,
    perc_masked_residues: float = 0.3,
    is_masking: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """
    Generate a Torch dataloader iterator from fp_data. Useful for training without loading the whole dataset
    Useful for training.

    Parameters
    ----------
    fp_seq: str
        Filepath to the csv file  with the sequences of interest.
    col_vh_seq: str
        Name of the csv column of the heavy sequences.
    col_vl_seq: str
        Name of the csv column of the light sequences.
    chunk_size: int
        Chunk size for the iterative loading of the csv file with pandas.
    batch_size: int
    perc_masked_residues: float
        Ratio of residues to apply the BERT masking on (between 0 and 1).
    is_masking: bool
    num_workers: int
        For loading parallelisation.

    """
    iterator = IterableMaskingBertOnehotDatasetCSVPaired(
        fp_seq,
        perc_masked_residues=perc_masked_residues,
        is_masking=is_masking,
        col_vh_seq=col_vh_seq,
        col_vl_seq=col_vl_seq,
        chunk_size=chunk_size,
    )
    loader = torch.utils.data.DataLoader(
        iterator, batch_size=batch_size, num_workers=num_workers
    )
    return loader


def data_loader_masking_bert_onehot_csv_paired_crossmasking(
    fp_seq: str,
    col_vh_seq: str = "vh_seq",
    col_vl_seq: str = "vl_seq",
    chunk_size: int = 100000,
    batch_size: int = 128,
    perc_masked_residues: float = 0.3,
    is_masking: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """
    Generate a Torch dataloader iterator from fp_data. Useful for training without loading the whole dataset
    Useful for training.

    Parameters
    ----------
    fp_seq: str
        Filepath to the csv file  with the sequences of interest.
    col_vh_seq: str
        Name of the csv column of the heavy sequences.
    col_vl_seq: str
        Name of the csv column of the light sequences.
    chunk_size: int
        Chunk size for the iterative loading of the csv file with pandas.
    batch_size: int
    perc_masked_residues: float
        Ratio of residues to apply the BERT masking on (between 0 and 1).
    is_masking: bool
    num_workers: int
        For loading parallelisation.

    """
    iterator = IterableCrossMaskingBertOnehotDatasetCSVPaired(
        fp_seq,
        perc_masked_residues=perc_masked_residues,
        is_masking=is_masking,
        col_vh_seq=col_vh_seq,
        col_vl_seq=col_vl_seq,
        chunk_size=chunk_size,
    )
    loader = torch.utils.data.DataLoader(
        iterator, batch_size=batch_size, num_workers=num_workers
    )
    return loader


def data_loader_masking_bert_onehot_csv_paired_with_mismatch(
    fp_seq: str,
    mismatch_fp: str,
    col_vh_seq: str = "vh_seq",
    col_vl_seq: str = "vl_seq",
    chunk_size: int = 100_000,
    batch_size: int = 128,
    perc_masked_residues: float = 0.3,
    is_masking: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """
    Just like data_loader_masking_bert_onehot_csv_paired, but we also load
    mismatch sequences from `mismatch_fp` and yield them as the third item.
    """
    iterator = IterableMaskingBertOnehotDatasetCSVPairedWithMismatch(
        fp_seq,
        mismatch_fp,
        perc_masked_residues=perc_masked_residues,
        is_masking=is_masking,
        col_vh_seq=col_vh_seq,
        col_vl_seq=col_vl_seq,
        chunk_size=chunk_size,
    )
    loader = torch.utils.data.DataLoader(
        iterator, batch_size=batch_size, num_workers=num_workers
    )
    return loader




class IterableMaskingBertOnehotDatasetCSVPairedWithMismatch(
    torch.utils.data.IterableDataset
):
    def __init__(
        self,
        fp_seq: str,
        mismatch_fp: str,
        perc_masked_residues: float = 0.0,
        is_masking: bool = False,
        col_vh_seq: str = "vh_seq",
        col_vl_seq: str = "vl_seq",
        chunk_size: int = 100_000,
    ):
        self.fp_seq = fp_seq
        self.mismatch_fp = mismatch_fp
        self.perc_masked_residues = perc_masked_residues
        self.is_masking = is_masking
        self.col_vh_seq = col_vh_seq
        self.col_vl_seq = col_vl_seq
        self.chunk_size = chunk_size

        self.seq_df = pd.read_csv(self.fp_seq)
        self.mismatch_df = pd.read_csv(self.mismatch_fp)

        self.total_rows = len(self.seq_df)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start, end = 0, self.total_rows
        if worker_info:
            per_worker = int(math.ceil(self.total_rows / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, self.total_rows)

        for i in range(start, end):

            vh_tensor = torch_masking_BERT_onehot(
                getattr(self.seq_df.iloc[i], self.col_vh_seq), self.perc_masked_residues, self.is_masking
            )
            vl_tensor = torch_masking_BERT_onehot(
                getattr(self.seq_df.iloc[i], self.col_vl_seq), self.perc_masked_residues, self.is_masking
            )

            mismatch_vh_tensor = torch_masking_BERT_onehot(
                getattr(self.mismatch_df.iloc[i], self.col_vh_seq), self.perc_masked_residues, self.is_masking
            )
            mismatch_vl_tensor = torch_masking_BERT_onehot(
                getattr(self.mismatch_df.iloc[i], self.col_vl_seq), self.perc_masked_residues, self.is_masking
            )
 
            yield (vh_tensor, vl_tensor, mismatch_vh_tensor, mismatch_vl_tensor)



class IterableMaskingBertOnehotDataset(torch.utils.data.IterableDataset):
    """
    BERT-style masking onehot generator for all sequences given a fasta file.
    """

    def __init__(self, list_seqs=["QVQ"], perc_masked_residues=0.0, is_masking=False):
        self.list_seqs = list_seqs
        self.perc_masked_residues = perc_masked_residues
        self.is_masking = is_masking

    def __iter__(self) -> torch.utils.data.IterableDataset:
        for k, seq in enumerate(self.list_seqs):
            if len(seq) != 149:
                raise Exception(
                    f"Sequence number {k} is shorter than 149 characters. All sequences must be aligned with the AHo scheme."
                )
            yield torch_masking_BERT_onehot(
                seq,
                perc_masked_residues=self.perc_masked_residues,
                is_masking=self.is_masking,
            )


class IterableMaskingBertOnehotDatasetFasta(torch.utils.data.IterableDataset):
    """
    BERT-style masking onehot generator for all sequences given a fasta file.
    Useful for training.
    """

    def __init__(self, fp_seq="path.fa", perc_masked_residues=0.0, is_masking=False):
        self.fp_seq = fp_seq
        self.perc_masked_residues = perc_masked_residues
        self.is_masking = is_masking
        self.total_rows = sum(1 for _ in SeqIO.parse(fp_seq, "fasta"))

    def __iter__(self) -> torch.utils.data.IterableDataset:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single-process data loading
            iter_start = 0
            iter_end = self.total_rows
        else:  # In a worker process
            # Split workload
            per_worker = int(
                math.ceil(self.total_rows / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.total_rows)

        # Re-open the file and iterate over the assigned range
        with open(self.fp_seq, "r") as handle:
            for i, record in enumerate(SeqIO.parse(handle, "fasta")):
                if i < iter_start:
                    continue
                if i >= iter_end:
                    break
                yield torch_masking_BERT_onehot(
                    str(record.seq),
                    perc_masked_residues=self.perc_masked_residues,
                    is_masking=self.is_masking,
                )


class IterableMaskingBertOnehotDatasetCSVPaired(torch.utils.data.IterableDataset):
    """
    BERT-style masking onehot generator for all paired sequences given a csv file.
    The relevant column names can be provided.
    Chunk_size avoids the full memomry to be overloaded.
    Useful for training.
    """

    def __init__(
        self,
        fp_seq="path.csv",
        perc_masked_residues=0.0,
        is_masking=False,
        col_vh_seq: str = "vh_seq",
        col_vl_seq: str = "vl_seq",
        chunk_size: int = 100000,
    ):
        self.fp_seq = fp_seq
        self.perc_masked_residues = perc_masked_residues
        self.total_rows = sum(1 for _ in open(fp_seq)) - 1
        self.is_masking = is_masking

        self.col_vh_seq = col_vh_seq
        self.col_vl_seq = col_vl_seq
        self.chunk_size = chunk_size

    def __iter__(self) -> torch.utils.data.IterableDataset:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single-process data loading
            iter_start = 0
            iter_end = self.total_rows
        else:  # In a worker process
            # Split workload
            per_worker = int(math.ceil(self.total_rows / worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.total_rows)

        # Open CSV file and process assigned rows
        for chunk in pd.read_csv(self.fp_seq, chunksize=self.chunk_size):
            vh_seqs, vl_seqs = chunk[self.col_vh_seq], chunk[self.col_vl_seq]

            for i, (vh_seq, vl_seq) in enumerate(zip(vh_seqs, vl_seqs)):
                global_row_index = i + chunk.index.start
                if global_row_index < iter_start:
                    continue
                if global_row_index >= iter_end:
                    break

                vh_tensor = torch_masking_BERT_onehot(
                    vh_seq,
                    perc_masked_residues=self.perc_masked_residues,
                    is_masking=self.is_masking,
                )
                vl_tensor = torch_masking_BERT_onehot(
                    vl_seq,
                    perc_masked_residues=self.perc_masked_residues,
                    is_masking=self.is_masking,
                )

                yield vh_tensor, vl_tensor


class IterableMaskingBertOnehotDatasetPaired(torch.utils.data.IterableDataset):
    """
    BERT-style masking onehot generator for all paired sequences given a fasta file.
    The relevant column names can be provided.
    Chunk_size avoids the full memomry to be overloaded.
    Useful for training.
    """

    def __init__(
        self,
        df_pairs: pd.DataFrame,
        col_vh: str = "vh_seq",
        col_vl: str = "vl_seq",
        perc_masked_residues=0.0,
        is_masking=False,
    ):
        self.df_pairs = df_pairs
        self.col_vh = col_vh
        self.col_vl = col_vl

        self.perc_masked_residues = perc_masked_residues
        self.is_masking = is_masking

    def __iter__(self) -> torch.utils.data.IterableDataset:
        for vh_seq, vl_seq in zip(
            self.df_pairs[self.col_vh], self.df_pairs[self.col_vl]
        ):
            vh_tensor = torch_masking_BERT_onehot(
                vh_seq,
                perc_masked_residues=self.perc_masked_residues,
                is_masking=self.is_masking,
            )
            vl_tensor = torch_masking_BERT_onehot(
                vl_seq,
                perc_masked_residues=self.perc_masked_residues,
                is_masking=self.is_masking,
            )
            yield vh_tensor, vl_tensor


class IterableCrossMaskingBertOnehotDatasetCSVPaired(torch.utils.data.IterableDataset):
    """
    BERT-style masking onehot generator for all paired sequences given a fasta file.
    It will randomly mask the whole heavy or the light (perc_masked_residues times).
    The whole sequence is masked follwoing the same BERT sampling (80% dummy, 10% others, 10% true).
    The other chain doesn't get masked at all.
    The relevant column names can be provided.
    Chunk_size avoids the full memomry to be overloaded.
    Useful for training.
    """

    def __init__(
        self,
        fp_seq="path.csv",
        perc_masked_residues=0.0,
        is_masking=False,
        col_vh_seq: str = "vh_seq",
        col_vl_seq: str = "vl_seq",
        chunk_size: int = 100000,
    ):
        self.fp_seq = fp_seq
        self.perc_masked_residues = perc_masked_residues
        self.total_rows = sum(1 for _ in open(fp_seq)) - 1
        self.is_masking = is_masking

        self.col_vh_seq = col_vh_seq
        self.col_vl_seq = col_vl_seq
        self.chunk_size = chunk_size

    def __iter__(self) -> torch.utils.data.IterableDataset:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single-process data loading
            iter_start = 0
            iter_end = self.total_rows
        else:  # In a worker process
            # Split workload
            per_worker = int(math.ceil(self.total_rows / worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.total_rows)

        # Open CSV file and process assigned rows
        for chunk in pd.read_csv(self.fp_seq, chunksize=self.chunk_size):
            vh_seqs, vl_seqs = chunk[self.col_vh_seq], chunk[self.col_vl_seq]

            for i, (vh_seq, vl_seq) in enumerate(zip(vh_seqs, vl_seqs)):
                global_row_index = i + chunk.index.start
                if global_row_index < iter_start:
                    continue
                if global_row_index >= iter_end:
                    break

                is_chain_masked = np.random.choice(
                    [False, True],
                    p=[1 - self.perc_masked_residues, self.perc_masked_residues],
                )
                if is_chain_masked:
                    is_heavy = np.random.choice([False, True], p=[0.5, 0.5])
                    if is_heavy:
                        vh_tensor = torch_masking_BERT_onehot(
                            vh_seq, perc_masked_residues=1.0, is_masking=self.is_masking
                        )
                        vl_tensor = torch_masking_BERT_onehot(
                            vl_seq, perc_masked_residues=0, is_masking=False
                        )
                    else:
                        vh_tensor = torch_masking_BERT_onehot(
                            vh_seq, perc_masked_residues=0, is_masking=False
                        )
                        vl_tensor = torch_masking_BERT_onehot(
                            vl_seq, perc_masked_residues=1.0, is_masking=self.is_masking
                        )
                else:
                    vh_tensor = torch_masking_BERT_onehot(
                        vh_seq,
                        perc_masked_residues=self.perc_masked_residues,
                        is_masking=self.is_masking,
                    )
                    vl_tensor = torch_masking_BERT_onehot(
                        vl_seq,
                        perc_masked_residues=self.perc_masked_residues,
                        is_masking=self.is_masking,
                    )

                yield vh_tensor, vl_tensor


def torch_masking_BERT_onehot(
    seq: str,
    perc_masked_residues: float = 0.0,
    is_masking: bool = False,
    alphabet: list = alphabet,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    BERT-style masking on a one-hot encoding input. When a residue is masked, it is replaced
    by the dummie vector [1/21,...,1/21] of size 21. 80% of perc_masked_residues are masked,
    10% are replaced by another residue, 10% are left as they are.

    Parameters
    ----------
    seq: str
    perc_masked_residues: float
        Ratio of residues to apply the BERT masking on (between 0 and 1).
    is_masking: bool
        False for evaluation.
    alphabet: list
        List of string of the alphabet of residues used in the one hot encoder

    Returns
    -------
    onehot_seq: tensor
        One hot encoded input.
    m_tf_onehot_seq: tensor
        BERT masked one hot encoded input.

    """

    masked_alphabet = [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
        "-",
        "X" # Masking Token, only used in that one hot encoder alphabet, used at inference time, user defined
    ]
    
    # Value of masking token
    alphabet = masked_alphabet[:-1]
    masked_letter = [1 / len(alphabet)] * len(alphabet)

    # One Hot Encoding
    onehot_seq = np.array(
        (
            pd.get_dummies(
                pd.Series(list(seq)).astype(CategoricalDtype(categories=masked_alphabet))
            )
        )
    ).astype(float)

    ## Replace the masked tokens value
    # Find indices of "X" tokens
    mask_idx = masked_alphabet.index("X")
    mask_rows = onehot_seq[:, mask_idx] == 1

    # Replace those rows with uniform distribution over alphabet (excluding "X")
    onehot_seq = onehot_seq[:, :-1]   # drop "X" column entirely
    onehot_seq[mask_rows, :] = masked_letter

    onehot_seq = torch.tensor(onehot_seq, dtype=torch.float32)
    ln_seq = len(onehot_seq)

    m_tf_onehot_seq = onehot_seq.clone().detach()

    if is_masking:
        if perc_masked_residues > 1:
            raise NotImplementedError("Masking percentage should be between 0 and 1.")
        
        # MASKING
        nb_masking = math.floor(ln_seq * perc_masked_residues)
        nb_to_mask = math.floor(nb_masking * 0.8)  # 80% replace with mask token
        nb_to_replace = math.floor(nb_masking * 0.1)  # 10% replace with random residue

        if nb_to_mask != 0:

            rd_ids = torch.Tensor(
                random.sample(range(ln_seq), ln_seq)[: nb_to_mask + nb_to_replace]
            ).type(torch.int64)

            rd_alphabet_selection_to_replace = random.choices(alphabet, k=nb_to_replace)
            dummies_to_replace = np.array(
                (
                    pd.get_dummies(
                        pd.Series(rd_alphabet_selection_to_replace).astype(
                            CategoricalDtype(categories=alphabet)
                        )
                    )
                )
            )

            updates = np.array([masked_letter] * nb_to_mask)
            updates = torch.Tensor(np.concatenate((updates, dummies_to_replace)))

            m_tf_onehot_seq[rd_ids] = updates

    return onehot_seq, m_tf_onehot_seq
