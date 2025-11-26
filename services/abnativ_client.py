"""Thin wrapper around the upstream AbNatiV scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import os
import sys


def _prepend_if_exists(path: Path) -> None:
    """Insert ``path`` into sys.path so vendored deps are importable."""

    str_path = str(path)
    if path.exists() and str_path not in sys.path:
        sys.path.insert(0, str_path)


def _ensure_vendored_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _prepend_if_exists(repo_root / "external" / "AbNatiV")
    _prepend_if_exists(repo_root / "external" / "ANARCI" / "lib" / "python")


_ensure_vendored_dependencies()

from abnativ.model.scoring_functions import abnativ_scoring
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
