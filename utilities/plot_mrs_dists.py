"""
Set up and plot mRS distributions.
"""
import stroke_outcome  # for reference dists
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import utilities.calculations as calc


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
        df_nearest_units,
        df_mrs,
        scenarios=['drip_ship', 'redirect']
        ):
    # Set up where the data should come from -
    # which of the input dataframes, and which column within it.
    # Also keep a copy of the name of the selected region.
    if bar_option.startswith('ISDN: '):
        str_region_type = 'ISDN'
        str_selected_region = bar_option.split('ISDN: ')[-1]
        col_region = 'isdn'
    elif bar_option.startswith('ICB: '):
        str_region_type = 'ICB'
        str_selected_region = bar_option.split('ICB: ')[-1]
        col_region = 'icb'
    elif bar_option.startswith('Nearest unit: '):
        str_region_type = 'Nearest unit'
        str_selected_region = bar_option.split('Nearest unit: ')[-1]
        col_region = 'nearest_ivt_unit_name'
    else:
        str_region_type = 'National'
        str_selected_region = 'National'
        col_region = ''

    col_vals = [str_selected_region]

    # Create the data for this region:
    df = calc.group_mrs_dists_by_region(
        df_mrs, df_nearest_units, col_region=col_region, col_vals=col_vals)

    occ_type = scenario_dict['stroke_type']
    treat_type = scenario_dict['treatment_type']

    col = f'{occ_type}_{scenarios[0]}_{treat_type}_mrs_dists'
    col2 = f'{occ_type}_{scenarios[1]}_{treat_type}_mrs_dists'

    # Prettier formatting for the plot title:
    col_pretty = ''.join([
        f'{scenario_dict["stroke_type_str"]}, ',
        f'{scenario_dict["treatment_type_str"]}'
        ])

    # Gather mRS distributions.
    # Selected region:
    dist_noncum = df.loc[str_selected_region, [f'{col}_noncum_{i}' for i in range(7)]].values
    dist_cum = df.loc[str_selected_region, [f'{col}_{i}' for i in range(7)]].values
    dist_std = df.loc[str_selected_region, [f'{col}_noncum_std_{i}' for i in range(7)]].values

    # Redirect:
    dist2_noncum = df.loc[str_selected_region, [f'{col2}_noncum_{i}' for i in range(7)]].values
    dist2_cum = df.loc[str_selected_region, [f'{col2}_{i}' for i in range(7)]].values
    dist2_std = df.loc[str_selected_region, [f'{col2}_noncum_std_{i}' for i in range(7)]].values

    # # National data:
    # dist_national_noncum = (
    #     df_mrs_national.loc[df_mrs_national.index[0], f'{col}_noncum'])
    # dist_national_cum = df_mrs_national.loc[df_mrs_national.index[0], col]
    # dist_national_std = (
    #     df_mrs_national.loc[df_mrs_national.index[0], f'{col}_noncum_std'])

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
        scenarios[0]: {
            'noncum': dist_noncum,
            'cum': dist_cum,
            'std': dist_std,
            'colour': 'blue',
            'linestyle': 'dash',
        },
        # if str_selected_region is 'National',
        # then the following entry overwrites previous:
        scenarios[1]: {
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