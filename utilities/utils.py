"""
Useful functions.
"""
import streamlit as st
import plotly.graph_objs as go


def print_progress(text):
    st.write(f':green-background[:heavy_check_mark: {text}]')


def print_progress_loc(p, log_loc):
    """Wrapper for print_progress, checks if st.container given."""
    try:
        with log_loc:
            # Streamlit container given.
            print_progress(p)
    except AttributeError:
        # No container given.
        print_progress(p)


def update_plotly_font_sizes(fig):
    s = 16
    fig.update_layout(
        title=dict(font=dict(size=20)),
        xaxis=dict(title_font=dict(size=s), tickfont=dict(size=s)),
        yaxis=dict(title_font=dict(size=s), tickfont=dict(size=s)),
        xaxis2=dict(title_font=dict(size=s), tickfont=dict(size=s)),
        yaxis2=dict(title_font=dict(size=s), tickfont=dict(size=s)),
        xaxis3=dict(title_font=dict(size=s), tickfont=dict(size=s)),
        yaxis3=dict(title_font=dict(size=s), tickfont=dict(size=s)),
        legend_title=dict(font=dict(size=s)),
        legend=dict(font=dict(size=s)),
        hoverlabel=dict(font=dict(size=s)),
    )
    fig.update_annotations(font_size=s)  # includes subplot_titles

    return fig
