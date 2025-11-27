"""Shared helpers for talking to the managed sequence services."""

from __future__ import annotations

import os
from typing import Any, Dict

import requests

DEFAULT_BASE_URL = os.environ.get("SEQUENCE_LIBRARIES_URL")
_REQUEST_TIMEOUT = (10, 120)  # connect, read


def _base_url() -> str:
    raw = os.environ.get("SEQUENCE_LIBRARIES_URL", DEFAULT_BASE_URL or "")
    if not raw:
        raise RuntimeError(
            "Set the SEQUENCE_LIBRARIES_URL environment variable to point at your managed Sequence services."
        )
    return raw.rstrip("/")


def _headers() -> Dict[str, str]:
    return {"Content-Type": "application/json"}


def post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any] | Any:
    """Send a JSON request to ``path`` and return the decoded payload."""

    url = f"{_base_url()}/{path.lstrip('/')}"
    response = requests.post(
        url,
        json=payload,
        headers=_headers(),
        timeout=_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - defensive
        raise RuntimeError("Sequence service returned a non-JSON response.") from exc


def extract_results(payload: Any) -> list[Dict[str, Any]]:
    """Normalise service responses into a list of result dictionaries."""

    if isinstance(payload, dict):
        if "results" in payload and isinstance(payload["results"], list):
            return payload["results"]
        return [payload]

    if isinstance(payload, list):
        return payload

    raise RuntimeError("Sequence service returned an unexpected payload shape.")


def extract_failures(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        failures = payload.get("failures", [])
        if isinstance(failures, list):
            return [str(item) for item in failures]
    return []


__all__ = ["post_json", "extract_results", "extract_failures", "DEFAULT_BASE_URL"]
