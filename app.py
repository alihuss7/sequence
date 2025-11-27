import importlib
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image
from assets import image_data
from style.styles import apply_styles
from components.header import render_header
from pages import home, sequencing, database, contact_us

ABNATIV_CACHE_DIR = Path(os.environ.get("ABNATIV_HOME", "/data/.abnativ"))
ABNATIV_INIT_ERROR = None


def ensure_abnativ_models() -> None:
    """Download AbNatiV weights into a persistent cache if missing."""
    global ABNATIV_INIT_ERROR

    os.environ["ABNATIV_HOME"] = str(ABNATIV_CACHE_DIR)
    ABNATIV_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if any(ABNATIV_CACHE_DIR.iterdir()):
        return

    init_env = os.environ.copy()
    init_env["ABNATIV_HOME"] = str(ABNATIV_CACHE_DIR)
    init_env["HOME"] = str(ABNATIV_CACHE_DIR.parent)
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "abnativ.__main__",
                "init",
            ],
            check=True,
            env=init_env,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        ABNATIV_INIT_ERROR = "AbNatiV CLI not found; ensure the package is installed."
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        detail_msg = f" Details: {detail}" if detail else ""
        ABNATIV_INIT_ERROR = f"Failed to initialise AbNatiV models: {exc}{detail_msg}"


ensure_abnativ_models()

home = importlib.reload(home)
sequencing = importlib.reload(sequencing)
database = importlib.reload(database)
contact_us = importlib.reload(contact_us)

FAVICON_IMAGE = Image.open(BytesIO(image_data.favicon_png_bytes()))

st.set_page_config(
    page_title="Sormanni Sequencing", page_icon=FAVICON_IMAGE, layout="wide"
)

if ABNATIV_INIT_ERROR:
    st.error(ABNATIV_INIT_ERROR)

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
