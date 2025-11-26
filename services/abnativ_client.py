"""Thin wrapper around the upstream AbNatiV scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import os
import sys


def _maybe_add_vendored_abnativ() -> None:
    """Ensure the vendored AbNatiV tree is importable when pip install didn't run."""

    repo_root = Path(__file__).resolve().parents[1]
    vendored_root = repo_root / "external" / "AbNatiV"
    if vendored_root.exists():
        vendored_path = str(vendored_root)
        if vendored_path not in sys.path:
            sys.path.insert(0, vendored_path)


def _maybe_add_vendored_anarci() -> None:
    """Expose the bundled ANARCI package (lib/python) when not pip-installed."""

    repo_root = Path(__file__).resolve().parents[1]
    anarci_root = repo_root / "external" / "ANARCI" / "lib" / "python"
    if anarci_root.exists():
        anarci_path = str(anarci_root)
        if anarci_path not in sys.path:
            sys.path.insert(0, anarci_path)


_maybe_add_vendored_anarci()
_maybe_add_vendored_abnativ()

from abnativ import init as abnativ_init
from abnativ.model.scoring_functions import abnativ_scoring

_DEFAULT_MODEL_FILES = {
    "VH": "vh_model.ckpt",
    "VH2": "vh2_model.ckpt",
    "VH2_RHESUS": "vh2_rhesus_model.ckpt",
    "VHH": "vhh_model.ckpt",
    "VHH2": "vhh2_model.ckpt",
    "VL2": "vl2_model.ckpt",
    "VL": "vl2_model.ckpt",
    "VLAMBDA": "vlambda_model.ckpt",
    "VLAMBDA2": "vl2_model.ckpt",
    "VKAPPA": "vkappa_model.ckpt",
    "VKAPPA2": "vkappa2_model.ckpt",
    "VPAIRED2": "vpaired2_model.ckpt",
}


def _normalize_model_key(nativeness_type: str) -> Optional[str]:
    """Return the normalized default model key or None when custom ckpts are used."""

    if not nativeness_type:
        return None

    candidate = nativeness_type.strip()
    if not candidate:
        return None

    if (
        candidate.lower().endswith(".ckpt")
        or "/" in candidate
        or os.path.sep in candidate
    ):
        # Treat anything that looks like an explicit path as a user-provided checkpoint
        return None

    return candidate.replace("-", "_").upper()


def _ensure_pretrained_model(nativeness_type: str) -> None:
    """Download the default checkpoint on demand when it is not cached."""

    normalized = _normalize_model_key(nativeness_type)
    if not normalized:
        return

    filename = _DEFAULT_MODEL_FILES.get(normalized)
    if not filename:
        return

    model_dir = _abnativ_home() / "models" / "pretrained_models"
    model_path = model_dir / filename
    if model_path.exists():
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[abnativ] Downloading missing checkpoint '{filename}' to {model_dir}")
    try:
        abnativ_init.ensure_zenodo_models(
            [filename],
            do_force_update=False,
            model_dir=str(model_dir),
        )
    except Exception as exc:  # pragma: no cover - network/host dependent
        raise RuntimeError(
            f"Unable to fetch checkpoint '{filename}' automatically. "
            "Ensure outbound network access is available or run 'abnativ init' manually."
        ) from exc

    if not model_path.exists():  # pragma: no cover - defensive
        raise RuntimeError(
            f"Checkpoint download for '{filename}' did not complete; file still missing at {model_path}."
        )


from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


@dataclass
class AbnativResult:
    sequence_id: str
    nativeness_score: float
    residue_scores_path: Optional[Path]
    raw_sequence_df: Any
    raw_residue_df: Any


def _abnativ_home() -> Path:
    env_home = os.environ.get("ABNATIV_HOME")
    return (
        Path(env_home).expanduser()
        if env_home
        else (Path.home() / ".abnativ").expanduser()
    )


def _resolve_nativeness_column(scores_df) -> str:
    candidate_names = [
        "nativeness_score",
        "AbNatiV Score",
    ] + [
        col
        for col in scores_df.columns
        if col.lower().startswith("abnativ") and col.lower().endswith("score")
    ]

    for column in candidate_names:
        if column in scores_df.columns:
            return column

    available = ", ".join(scores_df.columns)
    raise RuntimeError(
        "AbNatiV response is missing a recognizable nativeness column. "
        f"Columns returned: {available or 'none'}."
    )


def run_abnativ(
    sequence: str,
    *,
    nativeness_type: str = "VH2",
    output_dir: Optional[Path] = None,
    output_id: str = "streamlit_sequence",
    do_align: bool = True,
    is_vhh: bool = False,
) -> AbnativResult:
    """Score a single sequence with AbNatiV."""

    cleaned_sequence = sequence.strip().replace("\n", "")
    if not cleaned_sequence:
        raise ValueError("Sequence must be a non-empty string")

    seq_record = SeqRecord(Seq(cleaned_sequence), id=output_id, description="")

    target_dir = Path(output_dir) if output_dir else _abnativ_home()
    target_dir.mkdir(parents=True, exist_ok=True)

    _ensure_pretrained_model(nativeness_type)

    scores_df, profiles_df = abnativ_scoring(
        model_type=nativeness_type,
        seq_records=[seq_record],
        batch_size=32,
        mean_score_only=False,
        do_align=do_align,
        is_VHH=is_vhh,
        output_dir=str(target_dir),
        output_id=output_id,
        run_parall_al=4,
    )

    if scores_df is None or scores_df.empty:
        raise RuntimeError(
            "AbNatiV did not return any sequence-level scores. "
            "This usually means ANARCI filtered the sequence."
        )

    nativeness_column = _resolve_nativeness_column(scores_df)
    nativeness_value = float(scores_df.iloc[0][nativeness_column])

    residue_scores_dir = target_dir / f"{output_id}_profiles"
    residue_scores_path = residue_scores_dir if residue_scores_dir.exists() else None

    return AbnativResult(
        sequence_id=output_id,
        nativeness_score=nativeness_value,
        residue_scores_path=residue_scores_path,
        raw_sequence_df=scores_df,
        raw_residue_df=profiles_df,
    )
