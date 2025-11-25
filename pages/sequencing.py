"""Sequencing page content."""

from __future__ import annotations

import csv
import io
from typing import List, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas should be part of Streamlit stack
    pd = None

import streamlit as st

try:
    from services.abnativ_client import run_abnativ
except ImportError:  # pragma: no cover - AbNatiV is optional in dev
    run_abnativ = None


RESULT_DF_KEY = "sequencing_results_df"
RESULT_CSV_KEY = "sequencing_results_csv"
DOWNLOAD_COUNTER_KEY = "sequencing_download_counter"


def _parse_csv_sequences(uploaded_file) -> List[Tuple[str, str]]:
    text = uploaded_file.getvalue().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("Uploaded CSV must include headers with a sequence column.")

    normalized = {name.lower(): name for name in reader.fieldnames}
    sequence_col = next(
        (
            original
            for key, original in normalized.items()
            if key in {"sequence", "heavy_chain", "heavychain", "vh", "vh_sequence"}
        ),
        None,
    )
    if not sequence_col:
        raise ValueError(
            "CSV needs a sequence column named 'sequence', 'heavy_chain', or 'vh'."
        )

    id_col = next(
        (
            original
            for key, original in normalized.items()
            if key in {"id", "name", "sequence_id"}
        ),
        None,
    )

    sequences: List[Tuple[str, str]] = []
    for idx, row in enumerate(reader, start=1):
        seq_value = (row.get(sequence_col) or "").strip()
        if not seq_value:
            continue
        sequence_id = (
            (row.get(id_col) or f"csv_sequence_{idx}")
            if id_col
            else f"csv_sequence_{idx}"
        )
        sequences.append((sequence_id, seq_value))

    if not sequences:
        raise ValueError("No valid sequences were found in the uploaded CSV.")

    return sequences


def _gather_sequences(
    heavy_chain_sequence: str, uploaded_file
) -> List[Tuple[str, str]]:
    sequences: List[Tuple[str, str]] = []

    if uploaded_file is not None:
        sequences.extend(_parse_csv_sequences(uploaded_file))

    manual_sequence = heavy_chain_sequence.strip().replace("\n", "")
    if manual_sequence:
        sequences.append(("manual_sequence", manual_sequence))

    if not sequences:
        raise ValueError("Provide a sequence in the text area or upload a CSV file.")

    return sequences


def _reset_results_state() -> None:
    st.session_state.pop(RESULT_DF_KEY, None)
    st.session_state.pop(RESULT_CSV_KEY, None)


def _next_download_key() -> str:
    current = st.session_state.get(DOWNLOAD_COUNTER_KEY, 0) + 1
    st.session_state[DOWNLOAD_COUNTER_KEY] = current
    return f"download_csv_{current}"


def render():
    """Render the sequencing page."""
    st.header("Sequencing")

    if pd is None:
        st.error(
            "Pandas is required to display sequencing results. Install pandas and restart the app."
        )
        return

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Input")
        heavy_chain_sequence = st.text_area(
            "Heavy Chain Sequence",
            height=150,
            placeholder="Enter heavy chain sequence here...",
        )

        uploaded_file = st.file_uploader(
            "Upload CSV File (Optional)",
            type=["csv"],
            help="Upload a CSV file with sequences",
        )

        model_selection = st.radio(
            "Select Model", options=["AbNatiV", "Nanokink"], index=0
        )

        run_button = st.button("Run", type="primary", use_container_width=True)

    with right_col:
        st.subheader("Output")
        output_placeholder = st.empty()
        download_placeholder = st.empty()

    def _render_output(df, csv_data):
        with output_placeholder:
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Output will appear here after running the model.")

        with download_placeholder:
            st.download_button(
                label="Download CSV",
                data=csv_data or "",
                file_name="abnativ_results.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=csv_data is None,
                key=_next_download_key(),
            )

    st.session_state[DOWNLOAD_COUNTER_KEY] = 0
    _render_output(
        st.session_state.get(RESULT_DF_KEY),
        st.session_state.get(RESULT_CSV_KEY),
    )

    if not run_button:
        return

    if model_selection != "AbNatiV":
        _reset_results_state()
        st.warning("Nanokink integration is not available yet.")
        return

    if run_abnativ is None:
        st.error(
            "AbNatiV integration is unavailable. Ensure the submodule is installed."
        )
        return

    try:
        sequences = _gather_sequences(heavy_chain_sequence, uploaded_file)
    except ValueError as exc:
        _reset_results_state()
        st.error(str(exc))
        return

    successes: List[Tuple[str, float]] = []
    failures: List[str] = []

    with st.spinner("Running AbNatiV..."):
        for sequence_id, sequence_value in sequences:
            try:
                result = run_abnativ(
                    sequence_value,
                    nativeness_type="VH2",
                    output_id=sequence_id,
                )
            except Exception as exc:
                failures.append(f"{sequence_id}: {exc}")
                continue

            successes.append((sequence_id, result.nativeness_score))

    if not successes:
        _reset_results_state()
        _render_output(None, None)
        st.error("AbNatiV processing failed for all sequences.")
        if failures:
            st.caption("Failure details")
            st.code("\n".join(failures))
        return

    results_df = pd.DataFrame(successes, columns=["sequence_id", "nativeness_score"])
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_value = csv_buffer.getvalue()

    st.session_state[RESULT_DF_KEY] = results_df
    st.session_state[RESULT_CSV_KEY] = csv_value

    _render_output(results_df, csv_value)

    st.success(f"Processed {len(successes)} sequence(s) with AbNatiV.")
    if failures:
        st.warning("Some sequences failed")
        st.code("\n".join(failures))
