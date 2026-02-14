"""HTTP client for the managed NbFrame endpoint."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

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


def _flatten_response(sequence_id: str, sequence: str, response: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"sequence_id": sequence_id, "sequence": sequence}

    for key in ("prediction", "result", "scores", "probabilities", "thresholds"):
        block = response.get(key)
        if isinstance(block, dict):
            for nested_key, nested_value in block.items():
                if not isinstance(nested_value, (dict, list, tuple, set)):
                    row[nested_key] = nested_value

    for key, value in response.items():
        if key in {"prediction", "result", "scores", "probabilities", "thresholds"}:
            continue
        if not isinstance(value, (dict, list, tuple, set)):
            row[key] = value

    row["sequence"] = response.get("sequence", row["sequence"])
    row["sequence_id"] = response.get("sequence_id", row["sequence_id"])
    if "probability" not in row and "prob_kinked" in row:
        row["probability"] = row["prob_kinked"]
    return row


def run_nbframe_batch(
    sequences: Sequence[Tuple[str, str]],
    *,
    kinked_threshold: float = 0.70,
    extended_threshold: float = 0.40,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run NbFrame sequence predictions via the remote API."""

    if not sequences:
        raise ValueError("At least one sequence is required to call NbFrame.")

    if extended_threshold > kinked_threshold:
        raise ValueError("Extended threshold cannot be greater than kinked threshold.")

    seq_records = _normalize_sequences(sequences)
    if not seq_records:
        raise ValueError("All provided sequences were empty after cleaning.")

    results: List[dict] = []
    failures: List[str] = []

    for record in seq_records:
        payload: Dict[str, Any] = {
            "sequence": record["sequence"],
            "sequence_id": record["sequence_id"],
            "kinked_threshold": kinked_threshold,
            "extended_threshold": extended_threshold,
            "mode": "sequence",
        }

        try:
            response = post_json("nbframe", payload)
        except HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                failures.append(
                    f"{record['sequence_id']}: NbFrame API endpoint is unavailable."
                )
            else:
                failures.append(f"{record['sequence_id']}: {exc}")
            continue
        except Exception as exc:  # pragma: no cover - surfaced in UI
            failures.append(f"{record['sequence_id']}: {exc}")
            continue

        if not isinstance(response, dict):
            failures.append(
                f"{record['sequence_id']}: NbFrame returned a non-JSON payload."
            )
            continue

        results.append(
            _flatten_response(record["sequence_id"], record["sequence"], response)
        )

    dataframe = pd.DataFrame(results)
    return dataframe.reset_index(drop=True), failures


__all__ = ["run_nbframe_batch"]
