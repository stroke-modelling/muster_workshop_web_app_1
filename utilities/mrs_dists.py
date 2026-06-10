"""
Display mRS distributions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

from utilities.utils import update_plotly_font_sizes


def set_up_mrs_labels():
    # Label for redir scenario on the mRS results:
    label_redir = ''.join([
        'Redirection available (mix of usual care,<br>',
        'redirection approved, redirection rejected)'
        ])
    options_labels = {
        'usual_care': 'Usual care',
        'redir_allowed': label_redir,
        'redir_accept': 'Only redirected patients',
        'no_treatment': 'No treatment',
    }
    scenario_labels = {
        'usual_care': 'Usual care',
        'redir_allowed':  'Redir.',
        'redir_accepted_only': 'Only redirected patients',
        'no_treatment': 'No treatment',
        'diff_redir_allowed_minus_usual_care': 'Diff.',
    }
    scenario_help = {
        'usual_care': None,
        'redir_allowed': ''.join([
            'Redirection available (mix of usual care, ',
            'redirection approved, redirection rejected).'
        ]),
        'redir_accepted_only': None,
        'no_treatment': None,
        'diff_redir_allowed_minus_usual_care': (
            'Difference between redirection available and usual care.'),
        }
    outcome_labels = {
        'mrs_0-2': 'Percentage with mRS<2',
        'mrs_shift': 'Average change in mRS score',
        'utility_shift': 'Change in utility',
    }
    outcome_formats = {
        'mrs_0-2': '%.1f%%',
        'mrs_shift': '%+.2f',
        'utility_shift': '%+.2f',
    }
    bar_colours = {
        'usual_care': '#0072b2',
        'redir_allowed': '#56b4e9',
        'redir_accept': '#009e73',
        'no_treatment': 'grey',
    }
    return (options_labels, scenario_labels, scenario_help, outcome_labels, outcome_formats, bar_colours)


def calculate_mrs_bars(
        df: pd.DataFrame,
        region: str,
        lsoa_subset: str,
        pops: pd.DataFrame,
        dict_no_treatment_outcomes: dict,
        bar_colours: dict,
        options_labels: dict,
        ):
    """
    
    """
    # Where to pick out mRS data from the outcomes df:
    cols_mrs = [f'mrs_dists_{i}' for i in range(7)]
    cols_mrs_noncum = [c.replace('dists_', 'dists_noncum_') for c in cols_mrs]
    cols_mrs_std = [f'{c}_std' for c in cols_mrs]

    # Calculate "no treatment" data.
    # Should have the same total proportions of nLVO
    # and LVO in both the usual care and redir groups,
    # with only the details of who goes where differing,
    # so only calculate one set of no-treatment mRS dists.
    df_no_treat = calculate_no_treatment_mrs(
        pops, dict_no_treatment_outcomes)

    df_u = df['usual_care'][lsoa_subset]
    df_r = df['redir_allowed'][lsoa_subset]
    df_a = df['redir_accepted_only'][lsoa_subset]
    try:
        df_u = df_u.loc[region]
        df_r = df_r.loc[region]
        df_a = df_a.loc[region]
        selected_region_is_mt_unit = False
    except KeyError:
        # This is an MT unit and the LSOA subset excludes patients
        # nearest MT units.
        selected_region_is_mt_unit = True
    if selected_region_is_mt_unit:
        pass
    else:
        # Calculate summary values for metrics:
        df_d = df_r - df_u
        dict_metrics = {}
        for key in ['mrs_0-2', 'mrs_shift', 'utility_shift']:
            p = 100 if key == 'mrs_0-2' else 1
            dict_metrics[key] = {
                'usual_care': df_u[key] * p,
                'redir_allowed': df_r[key] * p,
                'redir_accepted_only': df_a[key] * p,
                'diff_redir_allowed_minus_usual_care': df_d[key] * p
            }

        # Gather mRS distributions:
        dict_mrs_bars = {
            'usual_care': {
                'noncum': df_u[cols_mrs_noncum],
                'cum': df_u[cols_mrs],
                'std': df_u[cols_mrs_std],
                'colour': bar_colours['usual_care'],
                'label': options_labels['usual_care'],
            },
            'redir_allowed': {
                'noncum': df_r[cols_mrs_noncum],
                'cum': df_r[cols_mrs],
                'std': df_r[cols_mrs_std],
                'colour': bar_colours['redir_allowed'],
                'label': options_labels['redir_allowed']
            },
            'redir_accept': {
                'noncum': df_a[cols_mrs_noncum],
                'cum': df_a[cols_mrs],
                'std': df_a[cols_mrs_std],
                'colour': bar_colours['redir_accept'],
                'label': options_labels['redir_accept']
            },
            'no_treatment': {
                'noncum': df_no_treat[cols_mrs_noncum],
                'cum': df_no_treat[cols_mrs],
                'colour': bar_colours['no_treatment'],
                'label': options_labels['no_treatment']
            },
        }
    return dict_mrs_bars, dict_metrics


def calculate_no_treatment_mrs(pops: pd.DataFrame,
                               dict_nt_outcomes: dict):
    """
    Calculate no-treatment mRS distribution for this patient mix.

    Combine the nLVO and LVO no-treatment distributions in the
    proportions given in the population data.

    Inputs
    ------
    pops             - pd.DataFrame. Contains the numbers of
                       patients with nLVO and with LVO.
    dict_nt_outcomes - dict. Contains no-treatment mRS dists
                       for nLVO and LVO patients separately.

    Returns
    -------
    df_no_treat - pd.Series. Mixed no-treatment mRS distribution.
    """
    cols_mrs = [f'mrs_dists_{i}' for i in range(7)]
    cols_mrs_noncum = [c.replace('dists_', 'dists_noncum_') for c in cols_mrs]

    # Pick out stroke type proportions:
    prop_nlvo = pops[pops.index.str.startswith('nlvo')].sum()
    prop_lvo = pops[pops.index.str.startswith('lvo')].sum()
    prop_both = pops[pops.index.str.startswith('lvo') |
                     pops.index.str.startswith('nlvo')].sum()

    # Weighted sum of mRS distributions:
    df_no_treat = (
        ((prop_nlvo / prop_both) * dict_nt_outcomes['nlvo_no_treatment']) +
        ((prop_lvo / prop_both) * dict_nt_outcomes['lvo_no_treatment'])
    )
    df_no_treat[cols_mrs_noncum] = np.diff(df_no_treat[cols_mrs], prepend=0.0)
    # Round values:
    df_no_treat[cols_mrs_noncum] = np.round(df_no_treat[cols_mrs_noncum], 3)
    # Convert to series:
    df_no_treat = df_no_treat.squeeze()
    return df_no_treat


def plot_mrs_bars(mrs_lists_dict: dict, key: str = None):
    """
    Plot mRS distribution bar chart.

    Inputs
    ------
    mrs_lists_dict - dict. Data and kwargs for each set of data
                     to be plotted. Keys include 'noncum', 'cum',
                     'std', 'label', 'colour', 'linestyle'.
    key            - str. Key for plotly_chart widget.
    """
    fig = go.Figure()

    for label, mrs_dict in mrs_lists_dict.items():
        if 'std' in mrs_dict.keys():
            error_y = dict(
                type='data',
                array=100.0*mrs_dict['std'],
                visible=True
                )
        else:
            error_y = None
        fig.add_trace(go.Bar(
            x=[*range(7)],
            y=100.0*mrs_dict['noncum'],
            error_y=error_y,
            name=mrs_dict['label'],
            marker_color=mrs_dict['colour'],
            ))

    fig.update_layout(barmode='group')

    fig.update_layout(xaxis_showticklabels=True)
    fig.update_xaxes(
        title_text='Discharge disability (mRS)',
        # Ensure that all mRS ticks are shown:
        tickmode='linear',
        tick0=0,
        dtick=1,
        )
    fig.update_yaxes(title_text='Probability (%)')
    fig.update_layout(legend=dict(
        yanchor='top',
        y=-0.4,
        yref='paper',
        xanchor='center',
        x=0.5,
        orientation='h'
    ))
    # Figure setup.
    fig.update_layout(height=350, margin_t=0)
    fig = update_plotly_font_sizes(fig)
    fig.update_layout(title='')
    # Turn off legend click events
    # (default is click on legend item, remove that item from the plot)
    fig.update_layout(legend_itemclick=False)

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
    st.plotly_chart(
        fig,
        config=plotly_config,
        key=key
        )
