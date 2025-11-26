"""Thin wrapper around the NanoMelt prediction helpers."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ._vendor import prepend_repo_path

prepend_repo_path("external", "NanoMelt")

from nanomelt.model.nanomelt import NanoMeltPredPipe  # type: ignore

DEFAULT_CPUS = 2
DEFAULT_BATCH_SIZE = 420


def _normalize_sequences(
    sequences: Sequence[Tuple[str, str]],
) -> List[SeqRecord]:
    records: List[SeqRecord] = []

    for idx, (sequence_id, raw_value) in enumerate(sequences, start=1):
        cleaned_sequence = (raw_value or "").strip().replace("\n", "")
        if not cleaned_sequence:
            continue

        record_id = sequence_id or f"sequence_{idx}"
        records.append(SeqRecord(Seq(cleaned_sequence), id=record_id, description=""))

    return records


def run_nanomelt_batch(
    sequences: Sequence[Tuple[str, str]],
    *,
    do_alignment: bool = True,
    ncpus: int = DEFAULT_CPUS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run NanoMelt predictions for ``sequences``.

    Returns a tuple of ``(success_df, failures)`` where ``success_df`` contains the
    successfully scored rows and ``failures`` lists the sequence IDs that errored.
    """

    if not sequences:
        raise ValueError("At least one sequence is required to call NanoMelt.")

    seq_records = _normalize_sequences(sequences)
    if not seq_records:
        raise ValueError("All provided sequences were empty after cleaning.")

    dataframe = NanoMeltPredPipe(  # type: ignore[call-arg]
        seq_records,
        do_align=do_alignment,
        ncpus=max(1, ncpus),
        batch_size=max(1, batch_size),
        verbose=verbose,
    )

    if not isinstance(dataframe, pd.DataFrame):  # pragma: no cover - defensive
        dataframe = pd.DataFrame(dataframe)

    rename_map = {
        "ID": "sequence_id",
        "Sequence": "sequence",
        "Aligned Sequence": "aligned_sequence",
        "NanoMelt Tm (C)": "nanomelt_tm_c",
    }
    present_map = {
        src: dest for src, dest in rename_map.items() if src in dataframe.columns
    }
    dataframe = dataframe.rename(columns=present_map)

    if "sequence_id" not in dataframe.columns:
        dataframe.insert(0, "sequence_id", [record.id for record in seq_records])

    return dataframe.reset_index(drop=True), []


__all__ = ["run_nanomelt_batch"]
