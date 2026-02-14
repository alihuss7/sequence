"""Home page content."""

import streamlit as st


def render():
    """Render the home page."""
    st.markdown(
        """
        <style>
        .home-heading {
            font-size: 2.75rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .home-subheading {
            font-size: 1.1rem;
            color: #5c5c5c;
            margin-bottom: 2rem;
        }
        .model-card {
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background-color: #ffffff;
        }
        .model-card h3 {
            margin-top: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='home-heading'>Sormanni Sequencing Platform</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='home-subheading'>Explore experimental models powering the Sequencing workspace."
        " Bring your VH/VL sequences to AbNatiV, run end-to-end structure prediction with NbForge,"
        " classify CDR3 conformations with NbFrame, and estimate nanobody thermostability with NanoMelt.</div>",
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            """
            <div class='model-card'>
                <h3>AbNatiV</h3>
                <p>AbNatiV is our production nativeness engine. Highlights:</p>
                <ul>
                    <li><strong>Architecture:</strong> Vector-quantized VAE trained on &gt;20M sequences with VH/VL/VHH heads.</li>
                    <li><strong>Outputs:</strong> Global nativeness, CDR/Framework breakdowns, and residue-level heatmaps.</li>
                    <li><strong>Sequencing tab:</strong> Accepts VH sequences or CSV uploads, performs ANARCI alignment, executes AbNatiV locally, and returns a sortable table plus CSV export.</li>
                </ul>
                <p>Use AbNatiV to triage candidate panels before downstream screening and humanisation steps.</p>
                <p><em>Status: Fully operational for VH2 scoring; VL/VHH workflows planned.</em></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container():
        st.markdown(
            """
            <div class='model-card'>
                <h3>NbForge</h3>
                <p>
                    NbForge is our CLI-first nanobody structure prediction workflow exposed as a managed API. It performs
                    VHH-domain checks and trimming, writes AHo-numbered outputs, and can optionally include integrated
                    NbFrame sequence and structure scores.
                </p>
                <p><em>Status: Integrated for sequence-to-structure inference.</em></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container():
        st.markdown(
            """
            <div class='model-card'>
                <h3>NbFrame</h3>
                <p>
                    NbFrame predicts whether a VHH CDR3 is kinked, extended, or uncertain. Use the Sequencing tab to batch
                    classify sequences with configurable probability thresholds and export the results for screening.
                </p>
                <p><em>Status: Integrated for sequence-based classification.</em></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container():
        st.markdown(
            """
            <div class='model-card'>
                <h3>NanoMelt</h3>
                <p>
                    NanoMelt is a semi-supervised ensemble that stacks SVRs and GPRs on top of diverse nanobody embeddings
                    (one-hot, VHSE, ESM-1b, ESM-2) to estimate apparent melting temperatures. Use the Sequencing tab to
                    batch score nanobodies, keeping the CC BY-NC-SA 4.0 license in mind for non-commercial usage.
                </p>
                <p><em>Status: Experimental; GPU acceleration recommended for embedding throughput.</em></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
