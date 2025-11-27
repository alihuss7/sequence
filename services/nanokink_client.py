"""HTTP client for the managed NanoKink Cloud Run endpoint."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd
from requests import HTTPError

from .api_client import post_json


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
    do_alignment: bool = True,
    calculate_confidence: bool = False,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run NanoKink predictions by submitting one request per sequence."""

    if not sequences:
        raise ValueError("At least one sequence is required to call NanoKink.")

    seq_records = _normalize_sequences(sequences)
    if not seq_records:
        raise ValueError("All provided sequences were empty after cleaning.")

    results: List[dict] = []
    failures: List[str] = []

    for record in seq_records:
        payload = {"sequence": record["sequence"]}
        payload.update(
            {
                "do_alignment": do_alignment,
                "calculate_confidence": calculate_confidence,
                "verbose": verbose,
            }
        )

        try:
            response = post_json("nanokink", payload)
        except HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                failures.append(
                    f"{record['sequence_id']}: NanoKink API endpoint is unavailable."
                )
            else:
                failures.append(f"{record['sequence_id']}: {exc}")
            continue
        except Exception as exc:  # pragma: no cover - surfaced in UI
            failures.append(f"{record['sequence_id']}: {exc}")
            continue

        if not isinstance(response, dict):
            failures.append(
                f"{record['sequence_id']}: NanoKink returned a non-JSON payload."
            )
            continue

        details = response.get("details")
        details_data = details if isinstance(details, dict) else {}

        row = {
            "sequence_id": record["sequence_id"],
            "sequence": response.get("sequence", record["sequence"]),
            "tool": response.get("tool"),
            "probability": response.get("probability"),
            "raw_score": response.get("raw_score"),
        }
        for key, value in details_data.items():
            row[f"details_{key}"] = value
        results.append(row)

    dataframe = pd.DataFrame(results)

    return dataframe.reset_index(drop=True), failures


__all__ = ["run_nanokink_batch"]
