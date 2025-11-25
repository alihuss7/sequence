"""Custom CSS styles for the application."""

import streamlit as st


def apply_styles():
    """Apply custom CSS styles to the application."""
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] {
            height: 100%;
        }

        main .block-container {
            padding-bottom: 8rem;
        }

        header {visibility: hidden;}
        
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.5rem 2rem;
            background-color: #ffffff;
            border-bottom: 1px solid #e0e0e0;
            margin-bottom: 2rem;
        }
        
        .brand {
            font-size: 2rem;
            font-weight: bold;
            color: #262730;
        }
        
        .stButton > button {
            background-color: transparent;
            border: none;
            color: #262730;
            font-size: 1rem;
            padding: 0.5rem 1rem;
            cursor: pointer;
        }
        
        .stButton > button:hover {
            color: #6C3BAA;
            background-color: transparent;
        }
        
        .stButton > button:focus {
            box-shadow: none;
            background-color: transparent;
        }
        
        .stButton > button[kind="primary"] {
            color: #6C3BAA;
            font-weight: bold;
        }
        
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #ffffff;
            color: #262730;
            text-align: center;
            padding: 1rem 2rem;
            font-size: 0.9rem;
        }
        
        .footer-content {
            max-width: 1200px;
            margin: 0 auto;
            border-top: 1px solid #e0e0e0;
            padding-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
