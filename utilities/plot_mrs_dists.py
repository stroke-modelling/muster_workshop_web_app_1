"""
Set up and plot mRS distributions.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import utilities.calculations as calc
from utilities.utils import load_reference_mrs_dists


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

    # Seaborn-colorblind colours:
    # #0072b2  blue
    # #009e73  green
    # #d55e00  red
    # #cc79a7  pink
    # #f0e442  yellow
    # #56b4e9  light blue

    # Place all data and setup for plot into this dictionary:
    mrs_lists_dict = {
        'No treatment': {
            'noncum': dist_ref_noncum,
            'cum': dist_ref_cum,
            'std': None,
            'colour': 'grey',
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
            'colour': '#0072b2',
            'linestyle': 'dash',
        },
        # if str_selected_region is 'National',
        # then the following entry overwrites previous:
        scenarios[1]: {
            'noncum': dist2_noncum,
            'cum': dist2_cum,
            'std': dist2_std,
            'colour': '#56b4e9',
            'linestyle': 'dashdot',
        },
    }
    return mrs_lists_dict, str_selected_region, col_pretty


def plot_mrs_bars(mrs_lists_dict, title_text=''):
    # fig = go.Figure()
    subplot_titles = [
        'Discharge disability<br>probability distribution',
        'Cumulative probability<br>of discharge disability'
    ]

    fig = make_subplots(rows=2, cols=1, subplot_titles=subplot_titles)

    for label, mrs_dict in mrs_lists_dict.items():

        fig.add_trace(go.Bar(
            x=[*range(7)],
            y=mrs_dict['noncum'],
            error_y=dict(
                type='data',
                array=mrs_dict['std'],
                visible=True),
            name=label,
            legendgroup=1,
            marker_color=mrs_dict['colour'],
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[*range(7)],
            y=mrs_dict['cum'],
            name=label,
            legendgroup=2,
            marker_color=mrs_dict['colour'],
            mode='lines',
            line=dict(dash=mrs_dict['linestyle'])
            ), row=2, col=1)

    fig.update_layout(barmode='group')
    fig.update_layout(title=title_text)
    for row in [1, 2]:  # 'all' doesn't work for some reason
        fig.update_xaxes(
            title_text='Discharge disability (mRS)',
            # Ensure that all mRS ticks are shown:
            tickmode='linear',
            tick0=0,
            dtick=1,
            row=row, col=1
            )
    fig.update_yaxes(title_text='Probability', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative probability', row=2, col=1)

    # Figure setup.
    fig.update_layout(
        # width=1200,
        height=700,
        margin_t=100,
        )

    # fig.show()
    st.plotly_chart(fig, use_container_width=True)
