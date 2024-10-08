"""
Set up and plot mRS distributions.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np

import utilities.calculations as calc
from utilities.utils import load_reference_mrs_dists


def setup_for_mrs_dist_bars(
        bar_option,
        scenario_dict,
        df_nearest_units,
        df_mrs,
        input_dict,
        scenarios=['drip_ship', 'redirect']
        ):
    # Set up where the data should come from -
    # which of the input dataframes, and which column within it.
    # Also keep a copy of the name of the selected region.
    if bar_option.startswith('ISDN: '):
        str_selected_region = bar_option.split('ISDN: ')[-1]
        col_region = 'isdn'
    elif bar_option.startswith('ICB: '):
        str_selected_region = bar_option.split('ICB: ')[-1]
        col_region = 'icb'
    elif bar_option.startswith('Ambulance service: '):
        str_selected_region = bar_option.split('Ambulance service: ')[-1]
        col_region = 'ambo22'
    elif bar_option.startswith('Nearest unit: '):
        str_selected_region = bar_option.split('Nearest unit: ')[-1]
        col_region = 'nearest_ivt_unit_name'
    else:
        str_selected_region = 'National'
        col_region = ''

    col_vals = [str_selected_region]

    # Create the data for this region:
    df = calc.group_mrs_dists_by_region(
        df_mrs, df_nearest_units, col_region=col_region, col_vals=col_vals)

    occ_type = scenario_dict['stroke_type']
    treat_type = scenario_dict['treatment_type']

    col = f'{scenarios[0]}_{occ_type}_{treat_type}_mrs_dists'
    col2 = f'{scenarios[1]}_{occ_type}_{treat_type}_mrs_dists'

    # Prettier formatting for the plot title:
    col_pretty = ''.join([
        f'{scenario_dict["stroke_type_str"]}, ',
        f'{scenario_dict["treatment_type_str"]}'
        ])

    # No-treatment data:
    dist_dict = load_reference_mrs_dists()
    if 'nlvo' in occ_type:
        dist_ref_noncum = dist_dict['nlvo_no_treatment_noncum']
        dist_ref_cum = dist_dict['nlvo_no_treatment']
    elif 'lvo' in occ_type:
        dist_ref_noncum = dist_dict['lvo_no_treatment_noncum']
        dist_ref_cum = dist_dict['lvo_no_treatment']
    else:
        # Combined stroke types.
        # Scale and sum the nLVO and LVO dists.
        scale_nlvo = input_dict['prop_nlvo']
        scale_lvo = input_dict['prop_lvo']

        dist_ref_noncum = (
            (dist_dict['nlvo_no_treatment_noncum'] * scale_nlvo) +
            (dist_dict['lvo_no_treatment_noncum'] * scale_lvo)
        )
        dist_ref_cum = np.cumsum(dist_ref_noncum)

    # Gather mRS distributions.
    try:
        # Selected region:
        dist_noncum = df.loc[str_selected_region,
                            [f'{col}_noncum_{i}' for i in range(7)]].values
        dist_cum = df.loc[str_selected_region,
                        [f'{col}_{i}' for i in range(7)]].values
        dist_std = df.loc[str_selected_region,
                        [f'{col}_noncum_std_{i}' for i in range(7)]].values

        # Redirect:
        dist2_noncum = df.loc[str_selected_region,
                            [f'{col2}_noncum_{i}' for i in range(7)]].values
        dist2_cum = df.loc[str_selected_region,
                        [f'{col2}_{i}' for i in range(7)]].values
        dist2_std = df.loc[str_selected_region,
                        [f'{col2}_noncum_std_{i}' for i in range(7)]].values
    except KeyError:
        # The data doesn't exist.
        if (('mt' in treat_type) & ('ivt' not in treat_type)):
            # MT-only. Use reference data.
            # Selected region:
            dist_noncum = dist_ref_noncum
            dist_cum = dist_ref_cum
            dist_std = None
            # Redirect:
            dist2_noncum = dist_ref_noncum
            dist2_cum = dist_ref_cum
            dist2_std = None
        else:
            # Use IVT-only data.
            col = f'{scenarios[0]}_{occ_type}_ivt_mrs_dists'
            col2 = f'{scenarios[1]}_{occ_type}_ivt_mrs_dists'
            # Selected region:
            dist_noncum = df.loc[
                str_selected_region,
                [f'{col}_noncum_{i}' for i in range(7)]
                ].values
            dist_cum = df.loc[
                str_selected_region,
                [f'{col}_{i}' for i in range(7)]
                ].values
            dist_std = df.loc[
                str_selected_region,
                [f'{col}_noncum_std_{i}' for i in range(7)]
                ].values

            # Redirect:
            dist2_noncum = df.loc[
                str_selected_region,
                [f'{col2}_noncum_{i}' for i in range(7)]
                ].values
            dist2_cum = df.loc[
                str_selected_region,
                [f'{col2}_{i}' for i in range(7)]
                ].values
            dist2_std = df.loc[
                str_selected_region,
                [f'{col2}_noncum_std_{i}' for i in range(7)]
                ].values

    # Display names for the data:
    display_name_dict = {
        'drip_ship': 'Usual care',
        'redirect': 'Redirection',
        'msu': 'MSU'
    }
    # Pick out the nicer-formatted names if they exist
    # or use the current names if not.
    try:
        display0 = display_name_dict[scenarios[0]]
    except KeyError:
        display0 = scenarios[0]
    try:
        display1 = display_name_dict[scenarios[1]]
    except KeyError:
        display1 = scenarios[1]

    # Seaborn-colorblind colours:
    # #0072b2  blue
    # #009e73  green
    # #d55e00  red
    # #cc79a7  pink
    # #f0e442  yellow
    # #56b4e9  light blue

    # Place all data and setup for plot into this dictionary.
    # The keys are used for the legend labels.
    mrs_lists_dict = {
        'No treatment': {
            'noncum': dist_ref_noncum,
            'cum': dist_ref_cum,
            'std': None,
            'colour': 'grey',
            'linestyle': 'dot',
        },
        display0: {
            'noncum': dist_noncum,
            'cum': dist_cum,
            'std': dist_std,
            'colour': '#0072b2',
            'linestyle': 'dash',
        },
        display1: {
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

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=subplot_titles, shared_xaxes=True)
    fig.update_layout(xaxis_showticklabels=True)

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
    # Bump the second half of the legend downwards:
    # (bump amount is eyeballed based on fig height)
    fig.update_layout(legend_tracegroupgap=240)

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
        margin_t=150,
        )

    # Options for the mode bar.
    # (which doesn't appear on touch devices.)
    plotly_config = {
        # Mode bar always visible:
        # 'displayModeBar': True,
        # Plotly logo in the mode bar:
        'displaylogo': False,
        # Remove the following from the mode bar:
        'modeBarButtonsToRemove': [
            # 'zoom',
            # 'pan',
            'select',
            # 'zoomIn',
            # 'zoomOut',
            'autoScale',
            'lasso2d'
            ],
        # Options when the image is saved:
        'toImageButtonOptions': {'height': None, 'width': None},
        }
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)
