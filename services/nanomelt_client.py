"""HTTP client for the managed NanoMelt Cloud Run endpoint."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd
from requests import HTTPError

from .api_client import post_json


def _normalize_sequences(
    sequences: Sequence[Tuple[str, str]],
) -> List[dict]:
    records: List[dict] = []

    for idx, (sequence_id, raw_value) in enumerate(sequences, start=1):
        cleaned_sequence = (raw_value or "").strip().replace("\n", "")
        if not cleaned_sequence:
            continue

        records.append(
            {
                "sequence_id": sequence_id or f"sequence_{idx}",
                "sequence": cleaned_sequence,
            }
        )

    return records


def run_nanomelt_batch(
    sequences: Sequence[Tuple[str, str]],
) -> Tuple[pd.DataFrame, List[str]]:
    """Run NanoMelt predictions for ``sequences`` using the remote API."""

    if not sequences:
        raise ValueError("At least one sequence is required to call NanoMelt.")

    seq_records = _normalize_sequences(sequences)
    if not seq_records:
        raise ValueError("All provided sequences were empty after cleaning.")

    results: List[dict] = []
    failures: List[str] = []

    for record in seq_records:
        payload = {"sequence": record["sequence"]}
        try:
            response = post_json("nanomelt", payload)
        except HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                failures.append(
                    f"{record['sequence_id']}: NanoMelt API endpoint is unavailable."
                )
            else:
                failures.append(f"{record['sequence_id']}: {exc}")
            continue
        except Exception as exc:  # pragma: no cover - surfaced in UI
            failures.append(f"{record['sequence_id']}: {exc}")
            continue

        prediction = response.get("prediction") if isinstance(response, dict) else None
        if not isinstance(prediction, dict):
            failures.append(
                f"{record['sequence_id']}: NanoMelt response missing prediction."
            )
            continue

        row = {
            "sequence_id": record["sequence_id"],
            "sequence": response.get("sequence", record["sequence"]),
        }
        row.update(prediction)
        results.append(row)

    dataframe = pd.DataFrame(results)
    rename_map = {
        "ID": "sequence_id",
        "Sequence": "sequence",
        "Aligned Sequence": "aligned_sequence",
        "NanoMelt Tm (C)": "nanomelt_tm_c",
    }
    present_map = {
        src: dest for src, dest in rename_map.items() if src in dataframe.columns
    }
    if present_map:
        dataframe = dataframe.rename(columns=present_map)

    return dataframe.reset_index(drop=True), failures


__all__ = ["run_nanomelt_batch"]
