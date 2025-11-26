"""Thin wrapper around the NanoKink prediction helpers."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd

from ._vendor import prepend_repo_path

prepend_repo_path("external", "NanoKink")

from nanokink import predict_kink_probabilities, results_to_dataframe  # type: ignore

DEFAULT_BATCH_SIZE = 512


def _normalize_sequences(
    sequences: Sequence[Tuple[str, str]],
) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    values: List[str] = []

    for idx, (sequence_id, sequence_value) in enumerate(sequences, start=1):
        cleaned_sequence = (sequence_value or "").strip().replace("\n", "")
        if not cleaned_sequence:
            continue

        ids.append(sequence_id or f"sequence_{idx}")
        values.append(cleaned_sequence)

    return ids, values


def run_nanokink_batch(
    sequences: Sequence[Tuple[str, str]],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    do_alignment: bool = True,
    calculate_confidence: bool = False,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run NanoKink predictions for ``sequences``.

    Returns a tuple of ``(success_df, failures)`` where ``success_df`` contains the
    successfully scored rows and ``failures`` lists the sequence IDs that errored.
    """

    if not sequences:
        raise ValueError("At least one sequence is required to call NanoKink.")

    names, cleaned_sequences = _normalize_sequences(sequences)
    if not cleaned_sequences:
        raise ValueError("All provided sequences were empty after cleaning.")

    effective_batch_size = max(1, min(batch_size, len(cleaned_sequences)))

    results = predict_kink_probabilities(
        cleaned_sequences,
        batch_size=effective_batch_size,
        do_alignment=do_alignment,
        calculate_confidence=calculate_confidence,
        verbose=verbose,
    )

    dataframe = results_to_dataframe(
        results,
        names=names,
        do_alignment=do_alignment,
    ).rename(columns={"name": "sequence_id"})

    failures: List[str] = []
    if "error" in dataframe.columns:
        error_mask = dataframe["error"].notna()
        if error_mask.any():
            failures = [
                f"{row.sequence_id}: {row.error}"
                for row in dataframe.loc[
                    error_mask, ["sequence_id", "error"]
                ].itertuples(index=False)
            ]
        dataframe = dataframe.loc[~error_mask].drop(columns=["error"], errors="ignore")

    return dataframe.reset_index(drop=True), failures


__all__ = ["run_nanokink_batch"]
