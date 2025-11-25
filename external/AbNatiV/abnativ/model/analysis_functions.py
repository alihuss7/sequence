import math
import os
from collections import defaultdict
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import pytorch_lightning as pl
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import textwrap

from .alignment.aho_consensus import (
    cdr1_aho_indices,
    cdr2_aho_indices,
    cdr3_aho_indices,
    fr_aho_indices,
)
from .alignment.mybio import anarci_alignments_of_Fv_sequences_iter
from .onehotencoder import (
    alphabet,
    data_loader_masking_bert_onehot_csv_paired,
    data_loader_masking_bert_onehot_fasta,
    data_loader_masking_bert_onehot_csv_paired_with_mismatch,
)


labelled_colors = {
    "pssm-generated": sns.color_palette("tab10")[1],
    "validation": sns.color_palette("tab10")[3],
    "mouse": sns.color_palette("tab10")[4],
    "human": sns.color_palette("tab10")[5],
    "rhesus": sns.color_palette("tab10")[6],
}

fontsizes = {
    "x_label": 16,
    "y_label": 16,
    "title": 20,
    "legend": 16,
}


def process_ROC(dataframes, labels, output_filepath, title, model_type=None):
    # Load the original dataset
    original_df = dataframes[0]
    original_scores = original_df[f"AbNatiV {model_type} Score"]

    # List of new datasets to compare
    new_dataframes = dataframes[1:]
    new_labels = labels[1:]

    # Initialize a plot
    plt.figure(figsize=(7, 6))

    roc_aucs = []

    for i, new_df in enumerate(new_dataframes):
        # Load the new dataset
        new_scores = new_df[f"AbNatiV {model_type} Score"]

        # Combine scores and create true labels
        combined_scores = pd.concat([original_scores, new_scores])

        true_labels = pd.Series([1] * len(original_scores) + [0] * len(new_scores))

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(true_labels, combined_scores)
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)

        label = new_labels[i]
        color = labelled_colors[label.lower()]

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f"{label}: {roc_auc:.3f} AUC", color=color, lw=3)

    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    # Set plot limits and labels
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    # Set plot limits and labels
    sns.despine()

    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.title(title)

    legend = plt.legend(loc="lower right", fontsize=16, frameon=False)

    if output_filepath:
        output_dir = os.path.dirname(output_filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save the plot as a PNG file
        plt.savefig(output_filepath, dpi=400, bbox_inches="tight")
    else:
        plt.show()

    plt.close()

    return roc_aucs


def process_PR(dataframes, labels, output_filepath=None, title="", model_type="None"):
    # Load the original dataset
    original_df = dataframes[0]
    original_scores = original_df[f"AbNatiV {model_type} Score"]

    # List of new datasets to compare
    new_dataframes = dataframes[1:]
    new_labels = labels[1:]

    # Initialize a plot
    plt.figure(figsize=(6, 6))
    plt.rcParams["font.family"] = "sans-serif"

    baselines = {}

    pr_aucs = []

    for i, new_df in enumerate(new_dataframes):
        new_scores = new_df[f"AbNatiV {model_type} Score"]

        # Combine scores and create true labels
        combined_scores = pd.concat([original_scores, new_scores])
        true_labels = pd.Series([1] * len(original_scores) + [0] * len(new_scores))

        # Calculate AUC
        precision, recall, thresholds = precision_recall_curve(
            true_labels, combined_scores
        )
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)

        f1_scores = 2 * (precision * recall) / (precision + recall)

        # Find the threshold that gives the maximum F1-score
        best_index = f1_scores.argmax()
        best_threshold = thresholds[best_index]
        best_f1 = f1_scores[best_index]

        label = new_labels[i]
        color = labelled_colors[label.lower()]
        # Plot ROC curve
        plt.plot(
            recall, precision, label=f"{label}: {pr_auc:.3f} AUC", color=color, lw=3
        )

        plt.scatter(
            recall[best_index],
            precision[best_index],
            color="black",
            label=f"Best Threshold {best_threshold:.4f} (F1 = {best_f1:.2f})",
        )
        print(label)
        print(f"Best Threshold {best_threshold}\n(F1 = {best_f1:.2f})\n")
        baseline = round(len(true_labels[true_labels == 1]) / len(true_labels), 3)
        if baseline not in baselines:
            baselines[baseline] = []
        baselines[baseline].append(label)

    for baseline, labels in baselines.items():
        if len(baselines) > 1:
            label = f'Baseline {" - ".join(labels)}: {baseline:.3f} AUC'
        else:
            label = f"Baseline: {baseline:.3f} AUC"
        color = (
            labelled_colors[labels[0].lower()]
            if len(baselines) > 1
            else (128 / 255, 128 / 255, 128 / 255, 0.5)
        )
        plt.plot(
            [0, 1], [baseline, baseline], linestyle="--", color=color, label=label, lw=2
        )

    # Set plot limits and labels
    plt.xlim([-0.025, 1.025])
    plt.ylim([-0.025, 1.025])

    # Set plot limits and labels
    sns.despine(right=True, top=True)

    plt.xlabel("Recall", fontsize=fontsizes["x_label"])
    plt.ylabel("Precision", fontsize=fontsizes["y_label"])
    wrapped_title = "\n".join(textwrap.wrap(title, width=20))
    plt.suptitle(wrapped_title, y=0.45, fontsize=fontsizes["title"])

    legend = plt.legend(loc="lower left", frameon=False, fontsize=fontsizes["legend"])

    if output_filepath:
        output_dir = os.path.dirname(output_filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save the plot as a PNG file
        plt.savefig(output_filepath, dpi=400, bbox_inches="tight")
    else:
        plt.show()

    plt.close()

    return pr_aucs


# Copied from scoring_functions to avoid circular imports!
# TODO: tidy and refactor
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
    best_thresholds = {}

    # Convert AHo-position into list index
    portion_indices = np.array(portion_indices) - 1

    # Score
    for k, profile in enumerate(output_abnativ["mse_pposi"]):
        score = norm_by_length_no_gap(
            profile[portion_indices].cpu().detach().numpy(),
            output_abnativ["inputs"][k][portion_indices].cpu().detach().numpy(),
        )
        humanness_scores.append(math.exp(-score))

    if model_type not in best_thresholds.keys():  # If scoring your own model
        rescaled_scores = humanness_scores

    else:
        rescaled_scores = linear_rescaling(
            humanness_scores, 0.8, best_thresholds[model_type]
        )

    return list(rescaled_scores)


def get_abnativ_scores_per_batch(
    model: pl.LightningModule,
    model_type: str,
    batch: int,
    scored_data_dict_mean: dict,
    aho_posi_whole=range(1, 150),
) -> dict:
    """
    Evaluate a batch of sequences using an AbNatiV model and update the score dictionary with nativeness scores.

    Parameters
    ----------
        - model : pl.LightningModule
            An instance of the AbNatiV_Model class used to evaluate the sequences.
        - model_type : str
            VH, VHH, VKappa, VLambda
        - batch : list
            A list of sequences to be evaluated.
        - scored_data_dict_mean : dict
            A dictionary to be updated with the new batch data. It should be a defaultdict(list) or similar structure.

    Returns
    -------
        - A dictionary containing the output of the sequences evaluated by AbNatiV.
            Example: {'fp': '/my/path/to/the/dataset.txt', 'recon_error_pbe': [0.2, 0.3, 0.4]}
    """

    output_abnativ = model(batch)
    name_type = model_type

    # Sequence-level scores
    humanness_scores = get_abnativ_nativeness_scores(
        output_abnativ, aho_posi_whole, model_type
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

    return output_abnativ


def evaluate_pssm(
    model: pl.LightningModule, model_type: str, fp_pssm: str, batch_size: int
):
    """
    Evaluate PSSM scores from the specified file path using the provided model and return the results as a pandas DataFrame.

    Parameters
    ----------
        - model : pl.LightningModule
            The model used to evaluate the PSSM scores.
        - model_type : str
            VH, VHH, VKappa, VLambda
        - fp_pssm : str
            File path to the PSSM data in FASTA format.
        - batch_size : int
            Batch size for loading the PSSM data.

    Returns
    -------
        - pd.DataFrame
            A DataFrame containing the evaluated PSSM scores.
    """
    scored_data_dict_mean = defaultdict(list)

    pssm_loader = data_loader_masking_bert_onehot_fasta(
        fp_pssm, batch_size, perc_masked_residues=0, is_masking=False
    )

    # TODO: Consider better device handling
    device = next(model.parameters()).device

    for batch in pssm_loader:
        batch = [tensor.to(device) for tensor in batch]
        output_abnativ = get_abnativ_scores_per_batch(
            model=model,
            model_type=model_type,
            batch=batch,
            scored_data_dict_mean=scored_data_dict_mean,
        )

    df_mean = pd.DataFrame.from_dict(scored_data_dict_mean)

    return df_mean


def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def evaluate_paired_pssm(
    model: pl.LightningModule,
    model_type: str,
    fp_pssm: str,
    batch_size: int,
    aho_posi_whole=range(1, 299),
):
    """
    Evaluate PSSM scores from the specified file path using the provided model and return the results as a pandas DataFrame.
    For paired sequences when loading the inputs. (using a paired csv file)
    Parameters
    ----------
        - model : pl.LightningModule
            The model used to evaluate the PSSM scores.
        - model_type : str
            VH, VHH, VKappa, VLambda
        - fp_pssm : str
            File path to the PSSM data in FASTA format.
        - batch_size : int
            Batch size for loading the PSSM data.

    Returns
    -------
        - pd.DataFrame
            A DataFrame containing the evaluated PSSM scores.
    """
    scored_data_dict_mean = defaultdict(list)

    pssm_loader = data_loader_masking_bert_onehot_csv_paired(
        fp_pssm, batch_size=batch_size, perc_masked_residues=0, is_masking=False
    )

    # TODO: Consider better device handling
    device = next(model.parameters()).device

    for batch in pssm_loader:
        batch = move_to_device(batch, device)
        output_abnativ = get_abnativ_scores_per_batch(
            model=model,
            model_type=model_type,
            batch=batch,
            scored_data_dict_mean=scored_data_dict_mean,
            aho_posi_whole=aho_posi_whole,
        )

    df_mean = pd.DataFrame.from_dict(scored_data_dict_mean)

    return df_mean


def evaluate_paired_pssm_with_mismatch(
    model: pl.LightningModule,
    model_type: str,
    fp_pssm: str,
    batch_size: int,
    aho_posi_whole=range(1, 299),
):
    """
    Evaluate PSSM scores from the specified file path using the provided model and return the results as a pandas DataFrame.
    For paired sequences when loading the inputs. (using a paired csv file)
    Parameters
    ----------
        - model : pl.LightningModule
            The model used to evaluate the PSSM scores.
        - model_type : str
            VH, VHH, VKappa, VLambda
        - fp_pssm : str
            File path to the PSSM data in FASTA format.
        - batch_size : int
            Batch size for loading the PSSM data.

    Returns
    -------
        - pd.DataFrame
            A DataFrame containing the evaluated PSSM scores.
    """
    scored_data_dict_mean = defaultdict(list)

    pssm_loader = data_loader_masking_bert_onehot_csv_paired_with_mismatch(
        fp_pssm,
        fp_pssm,
        batch_size=batch_size,
        perc_masked_residues=0,
        is_masking=False,
    )

    # TODO: Consider better device handling
    device = next(model.parameters()).device

    for batch in pssm_loader:
        batch = move_to_device(batch, device)
        output_abnativ = get_abnativ_scores_per_batch(
            model=model,
            model_type=model_type,
            batch=batch,
            scored_data_dict_mean=scored_data_dict_mean,
            aho_posi_whole=aho_posi_whole,
        )

    df_mean = pd.DataFrame.from_dict(scored_data_dict_mean)

    return df_mean


def evaluate_pairing_pred(
    model: pl.LightningModule,
    model_type: str,
    fp_pssm: str,
    batch_size: int,
    aho_posi_whole=range(1, 299),
):
    """
    Evaluate Pairing preiction for the top prediciton head
    Parameters
    ----------
        - model : pl.LightningModule
            The model used to evaluate the PSSM scores.
        - model_type : str
            VH, VHH, VKappa, VLambda
        - fp_pssm : str
            File path to the PSSM data in FASTA format.
        - batch_size : int
            Batch size for loading the PSSM data.

    Returns
    -------
        -  list
    """

    pssm_loader = data_loader_masking_bert_onehot_csv_paired(
        fp_pssm, batch_size=batch_size, perc_masked_residues=0, is_masking=False
    )

    # TODO: Consider better device handling
    device = next(model.parameters()).device

    saved_pairing_pred = list()
    for batch in pssm_loader:
        batch = move_to_device(batch, device)
        output_abnativ = model(batch)

        saved_pairing_pred += list(
            output_abnativ["pairing_pred"].cpu().detach().numpy()
        )

    return saved_pairing_pred


def evaluate_pairing_pred_with_mismatch(
    model: pl.LightningModule,
    model_type: str,
    fp_pssm: str,
    batch_size: int,
    aho_posi_whole=range(1, 299),
):
    """
    Evaluate Pairing preiction for the top prediciton head
    Parameters
    ----------
        - model : pl.LightningModule
            The model used to evaluate the PSSM scores.
        - model_type : str
            VH, VHH, VKappa, VLambda
        - fp_pssm : str
            File path to the PSSM data in FASTA format.
        - batch_size : int
            Batch size for loading the PSSM data.

    Returns
    -------
        -  list
    """

    pssm_loader = data_loader_masking_bert_onehot_csv_paired_with_mismatch(
        fp_pssm,
        fp_pssm,
        batch_size=batch_size,
        perc_masked_residues=0,
        is_masking=False,
    )

    # TODO: Consider better device handling
    device = next(model.parameters()).device

    saved_pairing_pred = list()
    for batch in pssm_loader:
        batch = move_to_device(batch, device)
        output_abnativ = model(batch)

        saved_pairing_pred += list(
            output_abnativ["pairing_pred"].cpu().detach().numpy()
        )

    return saved_pairing_pred


def evaluate_portion_score(
    model: pl.LightningModule,
    model_type: str,
    fp_database: str,
    batch_size: int,
    aho_selected_portion: list,
):
    """
    Evaluate portion of sequence of scores from the specified file path using the provided model and return the results as a pandas DataFrame.

    Parameters
    ----------
        - model : pl.LightningModule
            The model used to evaluate the PSSM scores.
        - model_type : str
            VH, VHH, VKappa, VLambda
        - fp_pssm : str
            File path to the PSSM data in FASTA format.
        - batch_size : int
            Batch size for loading the PSSM data.
        - aho_selected_portion : list of int
            List of AHo positions

    Returns
    -------
        - pd.DataFrame
            A DataFrame containing the evaluated PSSM scores.
    """
    scored_data_dict = defaultdict(list)

    pssm_loader = data_loader_masking_bert_onehot_fasta(
        fp_database, batch_size, perc_masked_residues=0, is_masking=False
    )

    # TODO: Consider better device handling
    device = next(model.parameters()).device

    for batch in pssm_loader:
        batch = [tensor.to(device) for tensor in batch]
        output_abnativ = model(batch)

        selected_scores = get_abnativ_nativeness_scores(
            output_abnativ, aho_selected_portion, model_type
        )

        scored_data_dict["AbNatiV selected portion score"].extend(selected_scores)

    df = pd.DataFrame.from_dict(scored_data_dict)

    return df


def calculate_pssm_aucs(
    model: pl.LightningModule,
    model_type: str,
    df_mean_val: pd.DataFrame,
    fp_pssm: str,
    batch_size: int,
    output_dir: str,
) -> Tuple[int, int]:
    """
    Evaluate PSSM scores, save the results, and calculate PR and ROC AUCs.

    This function performs the following steps:
    1. Randomly samples 10,000 examples from the validation scores if the dataset is larger.
    2. Evaluates PSSM scores from the provided file path.
    3. Saves the evaluated PSSM scores to 'pssm_scores.csv' in the specified directory.
    4. Processes and plots Precision-Recall (PR) curves for validation and PSSM-generated data.
    5. Processes and plots Receiver Operating Characteristic (ROC) curves for validation and PSSM-generated data.
    6. Returns the PR and ROC AUC scores.

    Parameters
    ----------
        - model : pl.LightningModule
            The model used to evaluate the PSSM scores.
        - model_type : str
            The type of model being used. Possible values are 'VH', 'VHH', 'VKappa', 'VLambda'.
        - df_mean_val : pd.DataFrame
            A DataFrame containing the mean validation scores.
        - fp_pssm : str
            File path to the PSSM data in FASTA format.
        - batch_size : int
            Batch size for loading the PSSM data.
        - output_dir : str
            The directory where the results will be saved.

    Returns
    -------
    tuple
        A tuple containing the PR AUC scores and ROC AUC scores.
    """

    # Random sample 10k examples from validation scores
    if len(df_mean_val) > 10000:
        df_mean_val = df_mean_val.sample(n=10000, random_state=11)

    # Evaluate PSSM scores from the provided file path and save the results in 'pssm_scores.csv' within the specified directory.
    df_mean_pssm = evaluate_pssm(model, model_type, fp_pssm, batch_size)
    df_mean_pssm.to_csv(f"{output_dir}/pssm_scores.csv")

    pr_aucs = process_PR(
        dataframes=[df_mean_val, df_mean_pssm],
        labels=["Validation", "PSSM-generated"],
        output_filepath=f"{output_dir}/PR_val_pssm.png",
        title="PR Validation-PSSM generated",
        model_type=model_type,
    )

    roc_aucs = process_ROC(
        dataframes=[df_mean_val, df_mean_pssm],
        labels=["Validation", "PSSM-generated"],
        output_filepath=f"{output_dir}/ROC_val_pssm.png",
        title="ROC Validation-PSSM generated",
        model_type=model_type,
    )

    return pr_aucs, roc_aucs


def calculate_pssm_aucs_paired(
    model: pl.LightningModule,
    model_type: str,
    df_mean_val: pd.DataFrame,
    fp_pssm: str,
    batch_size: int,
    output_dir: str,
) -> Tuple[int, int]:
    """
    Evaluate PSSM scores, save the results, and calculate PR and ROC AUCs. For paired sequences.

    This function performs the following steps:
    1. Randomly samples 10,000 examples from the validation scores if the dataset is larger.
    2. Evaluates PSSM scores from the provided file path.
    3. Saves the evaluated PSSM scores to 'pssm_scores.csv' in the specified directory.
    4. Processes and plots Precision-Recall (PR) curves for validation and PSSM-generated data.
    5. Processes and plots Receiver Operating Characteristic (ROC) curves for validation and PSSM-generated data.
    6. Returns the PR and ROC AUC scores.

    Parameters
    ----------
        - model : pl.LightningModule
            The model used to evaluate the PSSM scores.
        - model_type : str
            The type of model being used. Possible values are 'VH', 'VHH', 'VKappa', 'VLambda'.
        - df_mean_val : pd.DataFrame
            A DataFrame containing the mean validation scores.
        - fp_pssm : str
            File path to the PSSM data in FASTA format.
        - batch_size : int
            Batch size for loading the PSSM data.
        - output_dir : str
            The directory where the results will be saved.

    Returns
    -------
    tuple
        A tuple containing the PR AUC scores and ROC AUC scores.
    """

    # Random sample 10k examples from validation scores
    if len(df_mean_val) > 10000:
        df_mean_val = df_mean_val.sample(n=10000, random_state=11)
    if len(fp_pssm) > 10000:
        fp_pssm = fp_pssm.sample(n=10000, random_state=11)

    # Evaluate PSSM scores from the provided file path and save the results in 'pssm_scores.csv' within the specified directory.
    df_mean_pssm = evaluate_paired_pssm(model, model_type, fp_pssm, batch_size)
    df_mean_pssm.to_csv(f"{output_dir}/pssm_scores.csv")

    pr_aucs = process_PR(
        dataframes=[df_mean_val, df_mean_pssm],
        labels=["Validation", "PSSM-generated"],
        output_filepath=f"{output_dir}/PR_val_pssm.png",
        title="PR Validation-PSSM generated",
        model_type=model_type,
    )

    roc_aucs = process_ROC(
        dataframes=[df_mean_val, df_mean_pssm],
        labels=["Validation", "PSSM-generated"],
        output_filepath=f"{output_dir}/ROC_val_pssm.png",
        title="ROC Validation-PSSM generated",
        model_type=model_type,
    )

    return pr_aucs, roc_aucs


def calculate_pssm_aucs_paired_with_mismatch(
    model: pl.LightningModule,
    model_type: str,
    df_mean_val: pd.DataFrame,
    fp_pssm: str,
    batch_size: int,
    output_dir: str,
) -> Tuple[int, int]:
    """
    Evaluate PSSM scores, save the results, and calculate PR and ROC AUCs. For paired sequences.

    This function performs the following steps:
    1. Randomly samples 10,000 examples from the validation scores if the dataset is larger.
    2. Evaluates PSSM scores from the provided file path.
    3. Saves the evaluated PSSM scores to 'pssm_scores.csv' in the specified directory.
    4. Processes and plots Precision-Recall (PR) curves for validation and PSSM-generated data.
    5. Processes and plots Receiver Operating Characteristic (ROC) curves for validation and PSSM-generated data.
    6. Returns the PR and ROC AUC scores.

    Parameters
    ----------
        - model : pl.LightningModule
            The model used to evaluate the PSSM scores.
        - model_type : str
            The type of model being used. Possible values are 'VH', 'VHH', 'VKappa', 'VLambda'.
        - df_mean_val : pd.DataFrame
            A DataFrame containing the mean validation scores.
        - fp_pssm : str
            File path to the PSSM data in FASTA format.
        - batch_size : int
            Batch size for loading the PSSM data.
        - output_dir : str
            The directory where the results will be saved.

    Returns
    -------
    tuple
        A tuple containing the PR AUC scores and ROC AUC scores.
    """

    # Random sample 10k examples from validation scores
    if len(df_mean_val) > 10000:
        df_mean_val = df_mean_val.sample(n=10000, random_state=11)
    if len(fp_pssm) > 10000:
        fp_pssm = fp_pssm.sample(n=10000, random_state=11)

    # Evaluate PSSM scores from the provided file path and save the results in 'pssm_scores.csv' within the specified directory.
    df_mean_pssm = evaluate_paired_pssm_with_mismatch(
        model, model_type, fp_pssm, batch_size
    )
    df_mean_pssm.to_csv(f"{output_dir}/pssm_scores.csv")

    pr_aucs = process_PR(
        dataframes=[df_mean_val, df_mean_pssm],
        labels=["Validation", "PSSM-generated"],
        output_filepath=f"{output_dir}/PR_val_pssm.png",
        title="PR Validation-PSSM generated",
        model_type=model_type,
    )

    roc_aucs = process_ROC(
        dataframes=[df_mean_val, df_mean_pssm],
        labels=["Validation", "PSSM-generated"],
        output_filepath=f"{output_dir}/ROC_val_pssm.png",
        title="ROC Validation-PSSM generated",
        model_type=model_type,
    )

    return pr_aucs, roc_aucs
