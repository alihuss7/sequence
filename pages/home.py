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
        " Bring your VH/VL sequences to AbNatiV today; Nanokink arrives soon.</div>",
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
                <h3>Nanokink</h3>
                <p>
                    Nanokink focuses on nanobody stability and developability metrics. Integration is underway;
                    this space will ultimately expose run controls, tunable thresholds, and visualization for
                    Nanokink scorecards once the model is available.
                </p>
                <p><em>Status: API + weights pending delivery from research team.</em></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
