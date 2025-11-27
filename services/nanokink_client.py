"""HTTP client for the managed NanoKink Cloud Run endpoint."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd

from requests import HTTPError

from .api_client import extract_failures, extract_results, post_json

DEFAULT_BATCH_SIZE = 512


def _normalize_sequences(
    sequences: Sequence[Tuple[str, str]],
) -> List[dict]:
    normalized: List[dict] = []

    for idx, (sequence_id, sequence_value) in enumerate(sequences, start=1):
        cleaned_sequence = (sequence_value or "").strip().replace("\n", "")
        if not cleaned_sequence:
            continue

        normalized.append(
            {
                "sequence_id": sequence_id or f"sequence_{idx}",
                "sequence": cleaned_sequence,
            }
        )

    return normalized


def run_nanokink_batch(
    sequences: Sequence[Tuple[str, str]],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    do_alignment: bool = True,
    calculate_confidence: bool = False,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run NanoKink predictions for ``sequences`` via the remote API."""

    if not sequences:
        raise ValueError("At least one sequence is required to call NanoKink.")

    payload_sequences = _normalize_sequences(sequences)
    if not payload_sequences:
        raise ValueError("All provided sequences were empty after cleaning.")

    payload = {
        "sequences": payload_sequences,
        "batch_size": max(1, batch_size),
        "do_alignment": do_alignment,
        "calculate_confidence": calculate_confidence,
        "verbose": verbose,
    }

    try:
        response = post_json("nanokink", payload)
    except HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise RuntimeError("NanoKink API endpoint is unavailable.") from exc
        raise
    failures = extract_failures(response)
    results = extract_results(response)

    dataframe = pd.DataFrame(results)
    if not dataframe.empty:
        dataframe = dataframe.rename(columns={"name": "sequence_id"})
        if "sequence_id" not in dataframe.columns:
            dataframe.insert(
                0,
                "sequence_id",
                [item.get("sequence_id") for item in payload_sequences],
            )

    return dataframe.reset_index(drop=True), failures


__all__ = ["run_nanokink_batch"]
