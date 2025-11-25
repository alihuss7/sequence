# (c) 2023 Sormannilab and Aubin Ramon
#
# Scoring functions for the AbNatiV model.
#
# ============================================================================
import os

# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"   # allow CPU fallback for missing MPS ops

from .abnativ2_unpaired import AbNatiV_Model as AbNatiV2_Model
from .abnativ2_paired import AbNatiV_Paired_Model
from .abnativ import AbNatiV_Model

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .onehotencoder import (
    data_loader_masking_bert_onehot,
    data_loader_masking_bert_onehot_paired,
    alphabet,
)
from .alignment.mybio import anarci_alignments_of_Fv_sequences_iter
from .alignment.aho_consensus import (
    cdr1_aho_indices,
    cdr2_aho_indices,
    cdr3_aho_indices,
    fr_aho_indices,
)
from ..init import PRETRAINED_MODELS_DIR

from typing import Tuple
from collections import defaultdict
from tqdm import tqdm
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


def plot_abnativ_profile(res_scores, full_seq, name_seq, model_type, fig_fp_save):
    """
    Plot the AbNatiV score profile of a sequence.
    """
    sns.set(font_scale=2.2)
    sns.set_style(
        "white",
        {
            "axes.spines.right": False,
            "axes.spines.top": True,
            "axes.spines.bottom": False,
            "xtick.bottom": False,
            "xtick.top": True,
            "ytick.left": True,
            "xtick.labeltop": True,
        },
    )
    fig, ax = plt.subplots(figsize=(40, 8))

    # Plot
    ax.plot(res_scores, linewidth=5, alpha=0.65, color="darkorange", label=name_seq)

    # Add CDRs
    ax.axvspan(
        cdr1_aho_indices[0] - 1,
        cdr1_aho_indices[-1] - 1,
        alpha=0.06,
        color="forestgreen",
    )
    ax.axvspan(
        cdr2_aho_indices[0] - 1,
        cdr2_aho_indices[-1] - 1,
        alpha=0.06,
        color="forestgreen",
    )
    ax.axvspan(
        cdr3_aho_indices[0] - 1,
        cdr3_aho_indices[-1] - 1,
        alpha=0.06,
        color="forestgreen",
    )

    ax.xaxis.set_ticks(np.arange(0, len(full_seq), 1.0))
    ax.set_xticklabels(full_seq, fontsize=21)
    ax.tick_params(axis="x", which="major", pad=3)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(f"AbNatiV {model_type}\nResidue Score", fontsize=28, labelpad=15)
    ax.set_xlabel("Sequence", fontsize=28, labelpad=15)
    ax.xaxis.tick_top()

    plt.title(f"{name_seq} AbNatiV {model_type} profile", fontsize=31, pad=18)
    plt.tight_layout()
    plt.savefig(fig_fp_save, dpi=200, bbox_inches="tight")


def norm_by_length_no_gap(scores_pposi: list, onehot_encoding: list) -> list:
    """
    Sum the scores per position of a given sequence. Normalise the result by the
    number of residues (no gaps).

    Parameters
    ----------
        - scores_pposi : list
            Scores of the chosen positions
        - onehot_encoding : list
            One-hot encoding of the chosen given positions
    """
    length = len(onehot_encoding)
    idx = np.argmax(onehot_encoding, axis=1)
    nb_gaps = np.count_nonzero(idx == 20)
    length -= nb_gaps
    if length == 0:
        print("Length portion equals zero, too many gaps. Evaluates it as np.nan")
        norm = np.nan
    else:
        norm = np.sum(scores_pposi) / length
    return norm


def linear_rescaling(list_scores: list, t_new: int, t_r: int) -> list:
    """
    Linear rescaling of the scores to translate the threshold t_r (specific to each
    model type as defined in Methods) to the new threshold t_new.
    """
    rescaled = list()
    for x in list_scores:
        rescaled.append((t_new - 1) / (t_r - 1) * (x - 1) + 1)
    return rescaled


def get_abnativ_nativeness_scores(
    output_abnativ: dict, portion_indices: list, model_type: str
) -> list:
    """
    Give the AbNatiV nativeness scores for given positions (could be full) of the outputs of the
    AbNatiV model.

    Parameters
    ----------
        - output_abnativ : dict
            Output dict of sequences evaluated by AbNatiV
            i.e. {'fp':'/my/path/to/the/dataset.txt', 'recon_error_pbe': [0.2,0.3,0.4]}
        - portion_indices : list
            Position indices to score
            e.g., range(1,150) for the all sequence / range(27,43) for the CDR-1 (AHo numbering)
        - model_type : str
            VH, VHH, VKappa, VLambda

    Returns
    ----------
        - a list of the rescaled AbNatiV nativenees scores
    """

    humanness_scores = list()
    best_thresholds = {
        "VH": 0.988047,
        "VKappa": 0.992496,
        "VLambda": 0.985580,
        "VHH": 0.990973,
        "VH2": 0.993624,
        "VL2": 0.995629,
        "VHH2": 0.992695,
        "VPaired2": 0.991817,
        "VPairedH2": 0.992019,
        "VPairedL2": 0.993152,
        "VH2-Rhesus": 0.993892,
    }

    if "2" in model_type:
        output_data_to_compute_one = "mse_pposi"
    else:
        output_data_to_compute_one = "recon_error_pposi"

    # Convert AHo-position into list index
    portion_indices = np.array(portion_indices) - 1

    # Move tensors to NumPy before looping
    output_profiles = output_abnativ[output_data_to_compute_one].detach().cpu().numpy()
    input_profiles = output_abnativ["inputs"].detach().cpu().numpy()

    # Extract relevant portions in batch
    output_profiles_portion = output_profiles[:, portion_indices]
    input_profiles_portion = input_profiles[:, portion_indices]

    # Compute scores in a vectorized manner
    scores = np.array(
        [
            norm_by_length_no_gap(out, inp)
            for out, inp in zip(output_profiles_portion, input_profiles_portion)
        ]
    )

    # Apply exponential transformation
    humanness_scores.extend(np.exp(-scores))

    if model_type not in best_thresholds.keys():  # If scoring your own model
        rescaled_scores = humanness_scores

    else:
        rescaled_scores = linear_rescaling(
            humanness_scores, 0.8, best_thresholds[model_type]
        )

    return list(rescaled_scores)


@torch.inference_mode()
def abnativ_scoring(
    model_type: str,
    seq_records: list,
    batch_size: int = 1024,
    mean_score_only: bool = True,
    do_align: bool = True,
    is_VHH: bool = False,
    is_plotting_profiles: bool = False,
    output_dir: str = './',
    output_id: str = "antibody",
    verbose: bool = True,
    run_parall_al: int = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Infer on a list of seuqences, the AbNatiV loaded model with the
    selected model type. Returns a dataframe with the scored sequences.
    Alignement (AHo numbering) is performed if asked. If not asked, the sequences must have been
    beforehand aligned on the AHo scheme.

    Parameters
    ----------
        - model_type: str
            e.g., - VH, VHH, VKappa, VLambda (for default AbNatiV trained models),
                  - or, filepath to the custom checkpoint .ckpt (no linear rescaling will be applied)
        - seq_records: list
            List of SeqRecords from the BioPython package. seq = str(record.seq) / id = record.id
        - batch_size: int
        - mean_score_only: bool
            If True, provide only the mean nativeness score at the sequence level
        - do_align: bool
            If True, do the alignment with the AHo numbering by ANARCI #Coming update. A column
            will be added with the aligned_seq
        - is_VHH: bool
            If True, considers the VHH seed for the alignment, more suitable when aligning nanobody sequences
        - is_plotting_profiles: bool
            If True, plot the profile of every sequence
        - output_dir: str
            Filepath to the folder whre all files are saved
        - id: str
            Preffix of all saved files
        - verbose: bool
            If False, do not print anything except exceptions and errors
        - run_parall_al: int or bool
            If int, will parralelise the alignment on that int of cpus.
            If you don't want to parallelise, enter False

    Returns
    -------
        - df_mean: pd.DataFrame
            Dataframe composed of the id, the aligned sequence, the AbNatiV overall score,
            and in particular the CDR-1, CDR-2, CDR-3, Framework scores for a each sequences of the fasta file
        - df_profile: pd_DataFrame
            If not mean_score_only: Dataframe composed of the residue level Abnativ score with the score of each residue at each position.
            Else: Empty Dataframe

    """
    # Set the device
    if torch.cuda.is_available():
        device_type = "cuda"
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     device_type = 'mps'
    else:
        device_type = "cpu"

    if verbose:
        print(f"\nCalculations on device {device_type}\n")

    device = torch.device(device_type)

    model_dir = PRETRAINED_MODELS_DIR

    fr_trained_models = {  # AbNatiV1 models
        "VH": os.path.join(model_dir, "vh_model.ckpt"),
        "VHH": os.path.join(model_dir, "vhh_model.ckpt"),
        "VKappa": os.path.join(model_dir, "vkappa_model.ckpt"),
        "VLambda": os.path.join(model_dir, "vlambda_model.ckpt"),
        # AbNatiV2 models
        "VH2": os.path.join(model_dir, "vh2_model.ckpt"),
        "VL2": os.path.join(model_dir, "vl2_model.ckpt"),
        "VHH2": os.path.join(model_dir, "vhh2_model.ckpt"),
        # AbNatiV2-Zoo models
        "VH2-Rhesus": os.path.join(model_dir, "vh2_rhesus_model.ckpt"),
    }

    # Create folder if not existing
    if not os.path.exists(output_dir) and is_plotting_profiles:
        os.makedirs(output_dir)

    ## ALIGNMENT ##
    if do_align:
        if "2" in model_type:
            check_AHo_CDR_gaps = True
        else:
            check_AHo_CDR_gaps = False

        VH, VK, VL, failed, mischtype = anarci_alignments_of_Fv_sequences_iter(
            seq_records,
            isVHH=is_VHH,
            verbose=verbose,
            run_parallel=run_parall_al,
            check_AHo_CDR_gaps=check_AHo_CDR_gaps,
            del_cyst_misalign=False,
            check_terms=False,
        )

        recs = VH.to_recs()
        recs.extend(VK.to_recs())
        recs.extend(VL.to_recs())

        list_ids = [rec.id for rec in recs]
        list_al_seqs = [str(rec.seq) for rec in recs]
    else:
        list_al_seqs = [str(rec.seq) for rec in seq_records]
        list_ids = [rec.id for rec in seq_records]
        for k, al_seq in enumerate(list_al_seqs):
            if len(al_seq) != 149:
                raise Exception(
                    f"Sequence {k} is not aligned as expected (length={len(al_seq)}!=149), make sure all sequences are aligned on the AHo numbering."
                )

    list_seqs = [seq.replace("-", "") for seq in list_al_seqs]

    ## MODEL LOADING ##
    if model_type not in fr_trained_models.keys():
        try:
            if verbose:
                print(
                    f"\n### AbNatiV scoring of {output_id} from checkpoint {model_type} ###\n"
                )
            loaded_model = AbNatiV_Model.load_from_checkpoint(
                model_type, map_location=device
            )
            name_type = "Custom"
        except:
            print(
                f"Cannnot load the checkpoint {model_type}, you might use the default models: VH, VKappa, VLambda, or VHH."
            )

    else:
        if verbose:
            print(f"\n### AbNatiV {model_type}-ness scoring of {output_id} ###\n")
        if "2" in model_type:
            loaded_model = AbNatiV2_Model.load_from_checkpoint(
                fr_trained_models[model_type], map_location=device
            )
        else:
            loaded_model = AbNatiV_Model.load_from_checkpoint(
                fr_trained_models[model_type], map_location=device
            )
        name_type = model_type

    loaded_model.eval()
    loader = data_loader_masking_bert_onehot(
        list_al_seqs, batch_size, perc_masked_residues=0, is_masking=False
    )

    nb_of_iterations = math.ceil(len(list_ids) / batch_size)

    scored_data_dict_mean = defaultdict(list)
    scored_data_dict_mean.update(
        {"seq_id": list_ids, "input_seq": list_seqs, "aligned_seq": list_al_seqs}
    )

    scored_data_dict_profile = defaultdict(list)

    ## MODEL EVALUATION ##
    seen = 0
    for count, batch in enumerate(
        tqdm(loader, total=nb_of_iterations, disable=not verbose)
    ):
        batch = move_to_device(batch, device)
        output_abnativ = loaded_model(batch)

        # Sequence-level scores
        humanness_scores = get_abnativ_nativeness_scores(
            output_abnativ, range(1, 150), model_type
        )
        cdr1_scores = get_abnativ_nativeness_scores(
            output_abnativ, cdr1_aho_indices, model_type
        )
        cdr2_scores = get_abnativ_nativeness_scores(
            output_abnativ, cdr2_aho_indices, model_type
        )
        cdr3_scores = get_abnativ_nativeness_scores(
            output_abnativ, cdr3_aho_indices, model_type
        )
        fr_scores = get_abnativ_nativeness_scores(
            output_abnativ, fr_aho_indices, model_type
        )

        scored_data_dict_mean[f"AbNatiV {name_type} Score"].extend(humanness_scores)

        scored_data_dict_mean[f"AbNatiV CDR1-{name_type} Score"].extend(cdr1_scores)
        scored_data_dict_mean[f"AbNatiV CDR2-{name_type} Score"].extend(cdr2_scores)
        scored_data_dict_mean[f"AbNatiV CDR3-{name_type} Score"].extend(cdr3_scores)
        scored_data_dict_mean[f"AbNatiV FR-{name_type} Score"].extend(fr_scores)

        # Residue-level scores
        if not mean_score_only:
            batch_inputs = (
                output_abnativ["inputs"].cpu().numpy()
            )  # Move tensor to NumPy array
            batch_recon = output_abnativ["x_recon"].detach().cpu().numpy()

            err_key = "mse_pposi" if "2" in name_type else "recon_error_pposi"
            batch_errors = np.exp(
                -output_abnativ[err_key].detach().cpu().numpy()
            )  # Compute exp(-error)

            batch_size, seq_len, _ = (
                batch_recon.shape
            )  # Get batch size and sequence length

            seq_ids = list_ids[seen : seen + batch_size]  # Slice corresponding seq_ids
            seen += batch_size

            # Get max indices for amino acids
            aa_indices = np.argmax(batch_inputs, axis=2)  # Shape: (batch_size, seq_len)
            aa_chars = np.array(alphabet)[aa_indices]  # Convert indices to characters

            # Revert to 'X' for clarity in csv and png profiles
            mask_X = np.isclose(
                batch_inputs, 1.0 / len(alphabet), atol=1e-6, rtol=0
            ).all(axis=2)
            aa_chars[mask_X] = "X"

            # Flatten arrays for efficient DataFrame creation
            seq_id_expanded = np.repeat(seq_ids, seq_len)
            position_expanded = np.tile(np.arange(1, seq_len + 1), batch_size)
            aa_expanded = aa_chars.flatten()
            scores_expanded = batch_errors.flatten()

            scored_data_dict_profile["seq_id"].extend(seq_id_expanded)
            scored_data_dict_profile["AHo position"].extend(position_expanded)
            scored_data_dict_profile["aa"].extend(aa_expanded)
            scored_data_dict_profile[f"AbNatiV {name_type} Residue Score"].extend(
                scores_expanded
            )

            # Store reconstruction probabilities for each amino acid (one-hot encoded)
            for i, aa in enumerate(alphabet):
                scored_data_dict_profile[aa].extend(batch_recon[:, :, i].flatten())

            if is_plotting_profiles:
                dir_save = os.path.join(output_dir, f"{output_id}_profiles")
                os.makedirs(dir_save, exist_ok=True)  # Efficient directory creation

                for k, seq_id in enumerate(seq_ids):
                    clean_seq_id = seq_id.replace("/", "_")
                    fig_save_fp = os.path.join(
                        dir_save, f"{clean_seq_id}_abnativ_profile.png"
                    )
                    plot_abnativ_profile(
                        batch_errors[k], aa_chars[k], seq_id, name_type, fig_save_fp
                    )

    df_mean = pd.DataFrame.from_dict(scored_data_dict_mean)
    df_profile = pd.DataFrame.from_dict(scored_data_dict_profile)

    return df_mean, df_profile


def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


@torch.inference_mode()
def abnativ_scoring_paired(
    df_pairs: pd.DataFrame,
    col_id: str = "ID",
    col_vh: str = "vh_seq",
    col_vl: str = "vl_seq",
    batch_size: int = 256,
    mean_score_only: bool = True,
    do_align: bool = True,
    is_plotting_profiles: bool = False,
    output_dir: str = "./",
    output_id: str = "antibody",
    verbose: bool = True,
    run_parall_al: int = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Infer on a list of seuqences, the AbNatiV loaded model with the
    selected model type. Returns a dataframe with the scored sequences.
    Alignement (AHo numbering) is performed if asked. If not asked, the sequences must have been
    beforehand aligned on the AHo scheme.

    Parameters
    ----------
        - df_pairs: pd.DataFrame
            Dataframe with paired sequences
        - col_id: str
            Name of the column with the sequence IDs
        - col_vh: str
            Name of the column with the heavy sequences
        - col_vl: str:
            Name of the column with the light sequences
        - batch_size: int
        - mean_score_only: bool
            If True, provide only the mean nativeness score at the sequence level
        - do_align: bool
            If True, do the alignment with the AHo numbering by ANARCI #Coming update. A column
            will be added with the aligned_seq
        - is_plotting_profiles: bool
            If True, plot the profile of every sequence
        - output_dir: str
            Filepath to the folder whre all files are saved
        - id: str
            Preffix of all saved files
        - verbose: bool
            If False, do not print anything except exceptions and errors
        - run_parall_al: int or bool
            If int, will parralelise the alignment on that int of cpus.
            If you don't want to parallelise, enter False

    Returns
    -------
        - df_mean: pd.DataFrame
            Dataframe composed of the id, the aligned sequence, the AbNatiV overall score,
            and in particular the CDR-1, CDR-2, CDR-3, Framework scores for a each sequences of the fasta file
        - df_profile: pd_DataFrame
            If not mean_score_only: Dataframe composed of the residue level Abnativ score with the score of each residue at each position.
            Else: Empty Dataframe
            For paired sequences the returning profile concatenates the heavy and the light chains together.

    """
    # Set the device
    if torch.cuda.is_available():
        device_type = "cuda"
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     device_type = 'mps'
    else:
        device_type = "cpu"

    if verbose:
        print(f"\nCalculations on device {device_type}\n")

    device = torch.device(device_type)

    model_dir = PRETRAINED_MODELS_DIR

    # AbNatiV2 paired model
    fr_trained_model = os.path.join(model_dir, "vpaired2_model.ckpt")

    # Create folder if not existing
    if not os.path.exists(output_dir) and is_plotting_profiles:
        os.makedirs(output_dir)

    ## ALIGNMENT ENSURING DISCARDED PAIRED SEQUENCES WHEN ONLY ONE SEQUENCE PASSES THE ALIGNMENT ##
    if do_align:
        # Heavy
        seq_records_h = [
            SeqRecord(Seq(df_pairs[col_vh].iloc[i]), id=df_pairs[col_id].iloc[i])
            for i in range(len(df_pairs))
        ]
        VH, VK, VL, failed, mischtype = anarci_alignments_of_Fv_sequences_iter(
            seq_records_h,
            isVHH=False,
            verbose=verbose,
            run_parallel=run_parall_al,
            check_AHo_CDR_gaps=True,
            del_cyst_misalign=False,
            check_terms=False,
        )
        recs_h = VH.to_recs()
        recs_h.extend(VK.to_recs())
        recs_h.extend(VL.to_recs())

        # Light
        seq_records_l = [
            SeqRecord(Seq(df_pairs[col_vl].iloc[i]), id=df_pairs[col_id].iloc[i])
            for i in range(len(df_pairs))
        ]
        VH, VK, VL, failed, mischtype = anarci_alignments_of_Fv_sequences_iter(
            seq_records_l,
            isVHH=False,
            verbose=verbose,
            run_parallel=run_parall_al,
            check_AHo_CDR_gaps=True,
            del_cyst_misalign=False,
            check_terms=False,
        )
        recs_l = VH.to_recs()
        recs_l.extend(VK.to_recs())
        recs_l.extend(VL.to_recs())

        # Pair the alignment
        sequences = {"H": {}, "L": {}}
        for record in recs_h:
            sequences["H"][record.id] = str(record.seq)
        for record in recs_l:
            sequences["L"][record.id] = str(record.seq)

        paired_vh, paired_vl, ids = list(), list(), list()
        for seq_id in sequences["H"].keys():
            if seq_id in sequences["L"]:
                ids.append(seq_id)
                paired_vh.append(sequences["H"][seq_id])
                paired_vl.append(sequences["L"][seq_id])

        col_id_al, col_vh_al, col_vl_al = "ID", "vh_seq_aligned", "vl_seq_aligned"
        d = {col_id_al: ids, col_vh_al: paired_vh, col_vl_al: paired_vl}
        df_pairs_al = pd.DataFrame(d)

    else:
        df_pairs_al = df_pairs
        col_id_al, col_vh_al, col_vl_al = col_id, col_vh, col_vl
        for k, al_seq in enumerate(df_pairs_al[col_vh_al]):
            if len(al_seq) != 149:
                raise Exception(
                    f"Sequence {k} is not aligned as expected (length={len(al_seq)}!=149), make sure all sequences are aligned on the AHo numbering."
                )
        for k, al_seq in enumerate(df_pairs_al[col_vl_al]):
            if len(al_seq) != 149:
                raise Exception(
                    f"Sequence {k} is not aligned as expected (length={len(al_seq)}!=149), make sure all sequences are aligned on the AHo numbering."
                )

    ## MODEL LOADING ##
    loaded_model = AbNatiV_Paired_Model.load_from_checkpoint(
        fr_trained_model, map_location=device
    )
    loaded_model.eval()

    loader = data_loader_masking_bert_onehot_paired(
        df_pairs_al,
        col_vh_al,
        col_vl_al,
        batch_size,
        perc_masked_residues=0,
        is_masking=False,
    )

    nb_of_iterations = math.ceil(len(df_pairs_al) / batch_size)

    scored_data_dict_mean = defaultdict(list)
    scored_data_dict_mean.update(
        {
            "seq_id": df_pairs_al[col_id_al],
            "input_seq_vh": [seq.replace("-", "") for seq in df_pairs_al[col_vh_al]],
            "input_seq_vl": [seq.replace("-", "") for seq in df_pairs_al[col_vl_al]],
            "aligned_seq_vh": df_pairs_al[col_vh_al],
            "aligned_seq_vl": df_pairs_al[col_vl_al],
            "aligned_seq": [
                h + l for h, l in zip(df_pairs_al[col_vh_al], df_pairs_al[col_vl_al])
            ],
        }
    )

    scored_data_dict_profile = defaultdict(list)

    ## MODEL EVALUATION ##
    seen = 0
    for count, batch in enumerate(
        tqdm(loader, total=nb_of_iterations, disable=not verbose)
    ):
        batch = move_to_device(batch, device)
        output_abnativ = loaded_model(batch)

        # Sequence-level scores
        whole_humanness_scores = get_abnativ_nativeness_scores(
            output_abnativ, range(1, 299), "VPaired2"
        )
        pairing_pred_scores = (
            output_abnativ["pairing_pred"].cpu().detach().numpy().flatten()
        )

        h_humanness_scores = get_abnativ_nativeness_scores(
            output_abnativ, range(1, 150), "VPairedH2"
        )
        l_humanness_scores = get_abnativ_nativeness_scores(
            output_abnativ, range(150, 299), "VPairedL2"
        )

        h_cdr1_scores = get_abnativ_nativeness_scores(
            output_abnativ, cdr1_aho_indices, "VPairedH2"
        )
        h_cdr2_scores = get_abnativ_nativeness_scores(
            output_abnativ, cdr2_aho_indices, "VPairedH2"
        )
        h_cdr3_scores = get_abnativ_nativeness_scores(
            output_abnativ, cdr3_aho_indices, "VPairedH2"
        )
        h_fr_scores = get_abnativ_nativeness_scores(
            output_abnativ, fr_aho_indices, "VPairedH2"
        )

        l_cdr1_scores = get_abnativ_nativeness_scores(
            output_abnativ, np.array(cdr1_aho_indices) + 149, "VPairedL2"
        )
        l_cdr2_scores = get_abnativ_nativeness_scores(
            output_abnativ, np.array(cdr2_aho_indices) + 149, "VPairedL2"
        )
        l_cdr3_scores = get_abnativ_nativeness_scores(
            output_abnativ, np.array(cdr3_aho_indices) + 149, "VPairedL2"
        )
        l_fr_scores = get_abnativ_nativeness_scores(
            output_abnativ, np.array(fr_aho_indices) + 149, "VPairedL2"
        )

        scored_data_dict_mean[f"AbNatiV Heavy-Light Score"].extend(
            whole_humanness_scores
        )
        scored_data_dict_mean[f"AbNatiV Pairing Score (%)"].extend(pairing_pred_scores)

        scored_data_dict_mean[f"AbNatiV Heavy Score"].extend(h_humanness_scores)
        scored_data_dict_mean[f"AbNatiV Light Score"].extend(l_humanness_scores)

        scored_data_dict_mean[f"AbNatiV CDR1-Heavy Score"].extend(h_cdr1_scores)
        scored_data_dict_mean[f"AbNatiV CDR2-Heavy Score"].extend(h_cdr2_scores)
        scored_data_dict_mean[f"AbNatiV CDR3-Heavy Score"].extend(h_cdr3_scores)
        scored_data_dict_mean[f"AbNatiV FR-Heavy Score"].extend(h_fr_scores)

        scored_data_dict_mean[f"AbNatiV CDR1-Light Score"].extend(l_cdr1_scores)
        scored_data_dict_mean[f"AbNatiV CDR2-Light Score"].extend(l_cdr2_scores)
        scored_data_dict_mean[f"AbNatiV CDR3-Light Score"].extend(l_cdr3_scores)
        scored_data_dict_mean[f"AbNatiV FR-Light Score"].extend(l_fr_scores)

        # Residue-level scores
        if not mean_score_only:
            # Move tensors to NumPy in one step
            batch_inputs = output_abnativ["inputs"].cpu().numpy()
            batch_recon = output_abnativ["x_recon"].detach().cpu().numpy()
            batch_errors = np.exp(
                -output_abnativ["mse_pposi"].detach().cpu().numpy()
            )  # Precompute exp(-error)

            batch_size, seq_len, _ = batch_recon.shape  # Get dimensions

            seq_ids = (
                df_pairs_al[col_id_al].iloc[seen : seen + batch_size].to_numpy()
            )  # Slice corresponding seq_ids
            seen += batch_size

            # Get max indices for amino acids (faster than looping)
            aa_indices = np.argmax(batch_inputs, axis=2)
            aa_chars = np.array(alphabet)[aa_indices]

            # Revert to 'X' for clarity in csv and png profiles
            mask_X = np.isclose(
                batch_inputs, 1.0 / len(alphabet), atol=1e-6, rtol=0
            ).all(axis=2)
            aa_chars[mask_X] = "X"

            # Flatten arrays for efficient DataFrame creation
            seq_id_expanded = np.repeat(seq_ids, seq_len)
            position_expanded = np.tile(np.arange(1, seq_len + 1), batch_size)
            aa_expanded = aa_chars.flatten()
            scores_expanded = batch_errors.flatten()

            aho_labels = np.where(
                position_expanded > 149,
                [f"L-{pos - 149}" for pos in position_expanded],
                [f"H-{pos}" for pos in position_expanded],
            )

            scored_data_dict_profile["seq_id"].extend(seq_id_expanded)
            scored_data_dict_profile["AHo position"].extend(aho_labels)
            scored_data_dict_profile["aa"].extend(aa_expanded)
            scored_data_dict_profile[f"AbNatiV VPaired2 Residue Score"].extend(
                scores_expanded
            )

            # Store reconstruction probabilities for each amino acid
            for i, aa in enumerate(alphabet):
                scored_data_dict_profile[aa].extend(batch_recon[:, :, i].flatten())

            if is_plotting_profiles:
                dir_save = os.path.join(output_dir, f"{output_id}_profiles")
                os.makedirs(dir_save, exist_ok=True)  # Efficient directory creation

                for k, seq_id in enumerate(seq_ids):
                    clean_seq_id = seq_id.replace("/", "_")
                    fig_save_fp = os.path.join(
                        dir_save, f"{clean_seq_id}_abnativ_paired_heavy_profile.png"
                    )
                    plot_abnativ_profile(
                        batch_errors[k, :149],
                        aa_chars[k, :149],
                        seq_id,
                        "H-VPaired2",
                        fig_save_fp,
                    )

                    fig_save_fp = os.path.join(
                        dir_save, f"{clean_seq_id}_abnativ_paired_light_profile.png"
                    )
                    plot_abnativ_profile(
                        batch_errors[k, 149:],
                        aa_chars[k, 149:],
                        seq_id,
                        "L-VPaired2",
                        fig_save_fp,
                    )

    df_mean = pd.DataFrame.from_dict(scored_data_dict_mean)
    df_profile = pd.DataFrame.from_dict(scored_data_dict_profile)

    return df_mean, df_profile
