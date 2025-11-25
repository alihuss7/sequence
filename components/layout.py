import streamlit as st


def section_card(title, content):
    """Display a section card."""
    st.markdown(f"### {title}")
    st.write(content)
