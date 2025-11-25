"""Footer component."""

import streamlit as st


def render_footer():
    """Render the footer with copyright information."""
    st.markdown(
        """
        <div class="footer">
            <div class="footer-content">
                © Copyright Sormanni 2025
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
