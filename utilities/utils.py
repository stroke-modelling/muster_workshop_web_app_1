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


def make_formatted_time_str(t):
    try:
        t_new = (f'{int(60*(t/60)//60):d}hr ' +
                 f'{int(60*(t/60)%60):02d}min')
    except ValueError:
        # t is NaN.
        t_new = '~'
    return t_new


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


def write_text_from_file(filename, head_lines_to_skip=0):
    """
    Write text from 'filename' into streamlit.
    Skip a few lines at the top of the file using head_lines_to_skip.
    """
    # Open the file and read in the contents,
    # skipping a few lines at the top if required.
    with open(filename, 'r', encoding="utf-8") as f:
        text_to_print = f.readlines()[head_lines_to_skip:]

    # Turn the list of all of the lines into one long string
    # by joining them up with an empty '' string in between each pair.
    text_to_print = ''.join(text_to_print)

    # Write the text in streamlit.
    st.markdown(f"""{text_to_print}""")


def page_setup():
    # ----- Page setup -----
    # The following options set up the display in the tab in your browser. 
    # Set page to widescreen must be first call to st.
    st.set_page_config(
        page_title='OPTIMIST & MUSTER',
        page_icon=':ambulance:',
        # layout='wide'
        )
    # n.b. this can be set separately for each separate page if you like.


def set_inputs_changed(b=True):
    st.session_state['inputs_changed'] = b


def set_rerun_region_summaries(b=True):
    st.session_state['rerun_region_summaries'] = b


def set_rerun_map(b=True):
    st.session_state['rerun_maps'] = b


def set_rerun_full_results(b=True):
    st.session_state['rerun_full_results'] = b


def set_rerun_lsoa_units_times(b=True):
    st.session_state['rerun_lsoa_units_times'] = b
    if b:
        # Also flag any inputs changed:
        set_inputs_changed()
