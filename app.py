import importlib
from io import BytesIO

import streamlit as st
from PIL import Image
from assets import image_data
from style.styles import apply_styles
from components.header import render_header
from pages import home, sequencing, database, contact_us

home = importlib.reload(home)
sequencing = importlib.reload(sequencing)
database = importlib.reload(database)
contact_us = importlib.reload(contact_us)

FAVICON_IMAGE = Image.open(BytesIO(image_data.favicon_png_bytes()))

st.set_page_config(
    page_title="Sormanni Sequencing", page_icon=FAVICON_IMAGE, layout="wide"
)

if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

query_params = st.query_params
if "page" in query_params:
    st.session_state.active_page = query_params["page"]

apply_styles()

render_header()

if st.session_state.active_page == "Home":
    home.render()
elif st.session_state.active_page == "Sequencing":
    sequencing.render()
elif st.session_state.active_page == "Database":
    database.render()
elif st.session_state.active_page == "Contact Us":
    contact_us.render()
