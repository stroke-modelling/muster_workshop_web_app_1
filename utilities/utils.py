"""
Useful functions.
"""
import streamlit as st


def print_progress(text):
    st.write(f':green-background[🪵:heavy_check_mark: {text}]')


def print_progress_loc(p, log_loc):
    """Wrapper for print_progress, checks if st.container given."""
    try:
        with log_loc:
            # Streamlit container given.
            print_progress(p)
    except AttributeError:
        # No container given.
        print_progress(p)
