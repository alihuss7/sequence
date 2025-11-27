"""Client helpers for remotely hosted AbNatiV nativeness scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from requests import HTTPError

from .api_client import extract_failures, post_json


@dataclass
class AbnativResult:
    sequence_id: str
    nativeness_score: float
    residue_scores_path: Optional[str]
    raw_sequence_payload: Any


def _resolve_nativeness_score(
    scores_block: Dict[str, Any],
    nativeness_type: str,
) -> float:
    candidate_keys = [
        "nativeness_score",
        "AbNatiV Score",
        f"AbNatiV {nativeness_type} Score",
    ] + [
        key
        for key in scores_block
        if key.lower().startswith("abnativ") and key.lower().endswith("score")
    ]

    for key in candidate_keys:
        if key in scores_block:
            return float(scores_block[key])

    available = ", ".join(sorted(scores_block)) or "<none>"
    raise RuntimeError(
        "AbNatiV response missing a recognised nativeness score. "
        f"Available keys: {available}."
    )


def run_abnativ(
    sequence: str,
    *,
    nativeness_type: str = "VH2",
    output_id: str = "streamlit_sequence",
    do_align: bool = True,
    is_vhh: bool = False,
) -> AbnativResult:
    """Score a single sequence via the managed AbNatiV Cloud Run API."""

    cleaned_sequence = sequence.strip().replace("\n", "")
    if not cleaned_sequence:
        raise ValueError("Sequence must be a non-empty string")

    payload = {
        "sequence": cleaned_sequence,
        "sequence_id": output_id,
        "nativeness_type": nativeness_type,
        "do_align": do_align,
        "is_vhh": is_vhh,
    }

    try:
        response = post_json("abnativ", payload)
    except HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise RuntimeError("AbNatiV API endpoint is unavailable.") from exc
        raise
    failures = extract_failures(response)
    if failures:
        raise RuntimeError("; ".join(failures))

    scores_block = response.get("scores") if isinstance(response, dict) else None
    if not isinstance(scores_block, dict):
        raise RuntimeError("AbNatiV did not return a scores payload.")

    nativeness_value = _resolve_nativeness_score(scores_block, nativeness_type)

    return AbnativResult(
        sequence_id=response.get("sequence_id", output_id),
        nativeness_score=nativeness_value,
        residue_scores_path=None,
        raw_sequence_payload=response,
    )
