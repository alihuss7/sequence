"""Shared helpers for talking to the managed sequence services."""

from __future__ import annotations

import os
import time
from typing import Any, Dict

import requests
from requests import RequestException, Response

DEFAULT_BASE_URL = os.environ.get("SEQUENCE_LIBRARIES_URL")
_REQUEST_TIMEOUT = (10, 120)  # connect, read
_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
_MAX_ATTEMPTS = max(1, int(os.environ.get("SEQUENCE_API_MAX_ATTEMPTS", "3")))
_BACKOFF_SECONDS = max(0.0, float(os.environ.get("SEQUENCE_API_BACKOFF_SECONDS", "1.0")))


def _base_url() -> str:
    raw = os.environ.get("SEQUENCE_LIBRARIES_URL", DEFAULT_BASE_URL or "")
    if not raw:
        raise RuntimeError(
            "Set the SEQUENCE_LIBRARIES_URL environment variable to point at your managed Sequence services."
        )
    return raw.rstrip("/")


def _headers() -> Dict[str, str]:
    return {"Content-Type": "application/json"}


def _response_preview(response: Response, limit: int = 500) -> str:
    body = (response.text or "").strip().replace("\n", " ")
    if not body:
        return ""
    return body[:limit]


def _raise_http_error(response: Response, url: str) -> None:
    status = response.status_code
    reason = response.reason or "HTTP error"
    message = f"{status} {reason} for url: {url}"
    preview = _response_preview(response)
    if preview:
        message = f"{message} | response: {preview}"
    raise requests.HTTPError(message, response=response, request=response.request)


def post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any] | Any:
    """Send a JSON request to ``path`` and return the decoded payload."""

    url = f"{_base_url()}/{path.lstrip('/')}"
    last_request_error: RequestException | None = None
    response: Response | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=_headers(),
                timeout=_REQUEST_TIMEOUT,
            )
        except RequestException as exc:
            last_request_error = exc
            if attempt >= _MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Request failed after {_MAX_ATTEMPTS} attempt(s) for url: {url}"
                ) from exc
            time.sleep(_BACKOFF_SECONDS * attempt)
            continue

        if response.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_ATTEMPTS:
            time.sleep(_BACKOFF_SECONDS * attempt)
            continue

        break

    if response is None:
        if last_request_error is not None:
            raise RuntimeError(
                f"Request failed after {_MAX_ATTEMPTS} attempt(s) for url: {url}"
            ) from last_request_error
        raise RuntimeError(f"No response received for url: {url}")

    if not response.ok:
        _raise_http_error(response, url)

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
