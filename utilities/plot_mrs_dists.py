"""
Set up and plot mRS distributions.
"""
import stroke_outcome  # for reference dists
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def load_reference_mrs_dists():
    mrs_dists_ref, mrs_dists_ref_notes = (
        stroke_outcome.outcome_utilities.import_mrs_dists_from_file())

    nlvo_no_treatment = mrs_dists_ref.loc['no_treatment_nlvo'].values
    nlvo_no_treatment_noncum = np.diff(nlvo_no_treatment, prepend=0.0)

    lvo_no_treatment = mrs_dists_ref.loc['no_treatment_lvo'].values
    lvo_no_treatment_noncum = np.diff(lvo_no_treatment, prepend=0.0)

    dist_dict = {
        'nlvo_no_treatment': nlvo_no_treatment,
        'nlvo_no_treatment_noncum': nlvo_no_treatment_noncum,
        'lvo_no_treatment': lvo_no_treatment,
        'lvo_no_treatment_noncum': lvo_no_treatment_noncum,
    }
    return dist_dict


def setup_for_mrs_dist_bars(
        bar_option,
        scenario_dict,
        df_mrs_by_isdn,
        df_mrs_by_icb,
        df_mrs_by_nearest_ivt,
        df_mrs_national
        ):
    # Set up where the data should come from -
    # which of the input dataframes, and which column within it.
    # Also keep a copy of the name of the selected region.
    if bar_option.startswith('ISDN: '):
        df = df_mrs_by_isdn
        str_region_type = 'ISDN'
        str_selected_region = bar_option.split('ISDN: ')[-1]
    elif bar_option.startswith('ICB: '):
        df = df_mrs_by_icb
        str_region_type = 'ICB'
        str_selected_region = bar_option.split('ICB: ')[-1]
    elif bar_option.startswith('Nearest unit: '):
        df = df_mrs_by_nearest_ivt
        str_region_type = 'Nearest unit'
        str_selected_region = bar_option.split('Nearest unit: ')[-1]
    else:
        df = df_mrs_national
        str_region_type = 'National'
        str_selected_region = bar_option.split('National: ')[-1]

    occ_type = scenario_dict['stroke_type']
    treat_type = scenario_dict['treatment_type']

    col = f'{occ_type}_drip_ship_{treat_type}_mrs_dists'  # temp - scenario name still in here
    col2 = f'{occ_type}_redirect_{treat_type}_mrs_dists'  # temp - scenario name still in here

    # Prettier formatting for the plot title:
    col_pretty = ''.join([
        f'{scenario_dict["stroke_type_str"]}, ',
        f'{scenario_dict["treatment_type_str"]}'
        ])

    # Gather mRS distributions.
    # Selected region:
    dist_noncum = df.loc[str_selected_region, f'{col}_noncum']
    dist_cum = df.loc[str_selected_region, col]
    dist_std = df.loc[str_selected_region, f'{col}_noncum_std']

    # Redirect:
    dist2_noncum = df.loc[str_selected_region, f'{col2}_noncum']
    dist2_cum = df.loc[str_selected_region, col2]
    dist2_std = df.loc[str_selected_region, f'{col2}_noncum_std']

    # National data:
    dist_national_noncum = (
        df_mrs_national.loc[df_mrs_national.index[0], f'{col}_noncum'])
    dist_national_cum = df_mrs_national.loc[df_mrs_national.index[0], col]
    dist_national_std = (
        df_mrs_national.loc[df_mrs_national.index[0], f'{col}_noncum_std'])

    # No-treatment data:
    dist_dict = load_reference_mrs_dists()
    if 'nlvo' in occ_type:
        dist_ref_noncum = dist_dict['nlvo_no_treatment_noncum']
        dist_ref_cum = dist_dict['nlvo_no_treatment']
    else:
        dist_ref_noncum = dist_dict['lvo_no_treatment_noncum']
        dist_ref_cum = dist_dict['lvo_no_treatment']

    # Place all data and setup for plot into this dictionary:
    mrs_lists_dict = {
        'No treatment': {
            'noncum': dist_ref_noncum,
            'cum': dist_ref_cum,
            'std': None,
            'colour': 'red',
            'linestyle': 'dot',
        },
        # 'National': {
        #     'noncum': dist_national_noncum,
        #     'cum': dist_national_cum,
        #     'std': dist_national_std,
        #     'colour': 'green',
        #     'linestyle': None,
        # },
        # # if str_selected_region is 'National',
        # # then the following entry overwrites previous:
        'Drip-and-ship': {
            'noncum': dist_noncum,
            'cum': dist_cum,
            'std': dist_std,
            'colour': 'blue',
            'linestyle': 'dash',
        },
        # if str_selected_region is 'National',
        # then the following entry overwrites previous:
        'Redirect': {
            'noncum': dist2_noncum,
            'cum': dist2_cum,
            'std': dist2_std,
            'colour': 'magenta',
            'linestyle': 'dash',
        },
    }
    return mrs_lists_dict, str_selected_region, col_pretty


def plot_mrs_bars(mrs_lists_dict, title_text=''):
    # fig = go.Figure()
    subplot_titles = [
        'Discharge disability<br>probability distribution',
        'Cumulative probability<br>of discharge disability'
    ]

    fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles)

    for label, mrs_dict in mrs_lists_dict.items():

        fig.add_trace(go.Bar(
            x=[*range(7)],
            y=mrs_dict['noncum'],
            error_y=dict(
                type='data',
                array=mrs_dict['std'],
                visible=True),
            name=label,
            marker_color=mrs_dict['colour'],
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[*range(7)],
            y=mrs_dict['cum'],
            name=label,
            marker_color=mrs_dict['colour'],
            mode='lines',
            line=dict(dash=mrs_dict['linestyle'])
            ), row=1, col=2)

    fig.update_layout(barmode='group')
    fig.update_layout(title=title_text)
    fig.update_xaxes(title_text='Discharge disability (mRS)', row=1, col='all')
    fig.update_yaxes(title_text='Probability', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative probability', row=1, col=2)

    # fig.show()
    st.plotly_chart(fig)
