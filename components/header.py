"""Header component with logo and navigation."""

from functools import lru_cache
import base64

import streamlit as st

from assets import image_data


@lru_cache(maxsize=1)
def _logo_data_uri() -> str:
    """Return inline data URI for the Sormanni logo."""
    encoded = base64.b64encode(image_data.sormanni_logo_png_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_header():
    """Render the header with logo and navigation tabs."""
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown(
            f'<img src="{_logo_data_uri()}" width="150" alt="Sormanni logo" />',
            unsafe_allow_html=True,
        )

    with col2:
        nav_cols = st.columns([2.7, 0.8, 0.8, 0.8, 0.8])

        with nav_cols[0]:
            st.write("")

        with nav_cols[1]:
            if st.button(
                "Home",
                key="home",
                type=(
                    "primary" if st.session_state.active_page == "Home" else "secondary"
                ),
            ):
                st.session_state.active_page = "Home"
                st.query_params["page"] = "Home"
                st.rerun()

        with nav_cols[2]:
            if st.button(
                "Sequencing",
                key="seq",
                type=(
                    "primary"
                    if st.session_state.active_page == "Sequencing"
                    else "secondary"
                ),
            ):
                st.session_state.active_page = "Sequencing"
                st.query_params["page"] = "Sequencing"
                st.rerun()

        with nav_cols[3]:
            st.markdown(
                """
                <a href="https://gitlab.developers.cam.ac.uk/ch/sormanni" target="_blank" style="text-decoration: none;">
                    <button style="background-color: transparent; border: none; color: #262730; font-size: 1rem; padding: 0.5rem 1rem; cursor: pointer;">
                        Database
                    </button>
                </a>
                """,
                unsafe_allow_html=True,
            )

        with nav_cols[4]:
            if st.button(
                "Contact Us",
                key="contact",
                type=(
                    "primary"
                    if st.session_state.active_page == "Contact Us"
                    else "secondary"
                ),
            ):
                st.session_state.active_page = "Contact Us"
                st.query_params["page"] = "Contact Us"
                st.rerun()

    st.markdown("---")
