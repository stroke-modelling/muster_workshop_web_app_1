"""
Set up and plot mRS distributions.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW  # for mRS dist stats

import stroke_maps.load_data

import utilities.calculations as calc
from utilities.utils import load_reference_mrs_dists


def load_no_treatment_mrs_dists(occ_type, scale_nlvo=None, scale_lvo=None):
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
        dist_ref_noncum = (
            (dist_dict['nlvo_no_treatment_noncum'] * scale_nlvo) +
            (dist_dict['lvo_no_treatment_noncum'] * scale_lvo)
        )
        dist_ref_cum = np.cumsum(dist_ref_noncum)
    return dist_ref_cum, dist_ref_noncum


def pick_out_region_name(bar_option):
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
    return str_selected_region, col_region


def find_lsoa_names_to_keep(
        df_mrs_usual_care,
        col_to_mask_mrs,
        col_region='',
        str_selected_region=''
        ):
    # First limit the LSOA available to only those that benefit.
    try:
        lsoa_names_benefit = df_mrs_usual_care.loc[
            df_mrs_usual_care[col_to_mask_mrs] > 0.0].index.values
    except KeyError:
        # This column doesn't exist.
        lsoa_names_benefit = df_mrs_usual_care.index

    # Then limit the LSOA available to only those that are in the
    # selected region.
    if col_region == '':
        # Just keep everything.
        lsoa_names = df_mrs_usual_care.index
    else:
        mask_nearest = (
            df_mrs_usual_care[col_region] == str_selected_region)
        # Which LSOA are in this catchment area?
        lsoa_names = df_mrs_usual_care.loc[mask_nearest].index.values

    # Combine this with the previous mask:
    lsoa_names = list(set(lsoa_names) & set(lsoa_names_benefit))
    return lsoa_names


def find_total_mrs_for_unique_times(
        dict_of_dfs,
        lsoa_names,
        treat_type,
        occ_type,
        df_mrs_ivt,
        df_mrs_mt,
        ):
    mrs_lists_dict = {}
    # Limit the dataframes to only these LSOA:
    for key, df in dict_of_dfs.items():
        # Keep only these LSOA rows.
        df = df.loc[df.index.isin(lsoa_names)].copy()
        # Jettison the LSOA names:
        df = df.reset_index().drop('lsoa', axis='columns')
        # Store the result:
        mrs_lists_dict[key] = df

    if treat_type == 'ivt':
        # Only look at the IVT columns.
        # Which mRS dist columns do we want?
        dist_cols = [f'{occ_type}_{treat_type}_mrs_dists_noncum_{i}'
                     for i in range(7)]
        # Merge in these mRS dists:
        for key, df in mrs_lists_dict.items():
            # How many admissions are there for each treatment time?
            df = df[['Admissions', 'time_to_ivt']].groupby('time_to_ivt').sum()
            # Merge in the mRS dists:
            df = pd.merge(
                df.reset_index(),
                df_mrs_ivt[['time_to_ivt'] + dist_cols],
                on='time_to_ivt', how='left'
            )
            # Store the result:
            mrs_lists_dict[key] = df

    elif treat_type == 'mt':
        # Only look at the MT columns.
        # Which mRS dist columns do we want?
        dist_cols = [f'{occ_type}_{treat_type}_mrs_dists_noncum_{i}'
                     for i in range(7)]
        # Merge in these mRS dists:
        for key, df in mrs_lists_dict.items():
            # How many admissions are there for each treatment time?
            df = df[['Admissions', 'time_to_mt']].groupby('time_to_mt').sum()
            # Merge in the mRS dists:
            df = pd.merge(
                df.reset_index(),
                df_mrs_mt[['time_to_mt'] + dist_cols],
                on='time_to_mt', how='left'
            )
            # Store the result:
            mrs_lists_dict[key] = df
    else:
        # Pull in a mix of IVT and MT data.
        cols_group = ['time_to_mt', 'time_to_ivt', 'lvo_ivt_better_than_mt']
        # Set up column names:
        dist_cols = [f'{occ_type}_{treat_type}_mrs_dists_noncum_{i}'
                     for i in range(7)]
        dist_mt_cols = [f'{occ_type}_mt_mrs_dists_noncum_{i}'
                        for i in range(7)]
        dist_ivt_cols = [f'{occ_type}_ivt_mrs_dists_noncum_{i}'
                         for i in range(7)]

        # Merge in these mRS dists:
        for key, df in mrs_lists_dict.items():
            # How many patients have each combination of time to treatment
            # and treatment souce (IVT or MT dist)?
            df = df.groupby(cols_group).sum().reset_index()

            mask_ivt_better = df['lvo_ivt_better_than_mt'] == True
            # Initially copy over all MT results:
            df = pd.merge(
                df, df_mrs_mt[['time_to_mt'] + dist_mt_cols],
                on='time_to_mt', how='left'
            )
            # Rename "mt" to "ivt_mt":
            df = df.rename(columns=dict(zip(dist_mt_cols, dist_cols)))
            # Now copy over all IVT results:
            df = pd.merge(
                df, df_mrs_ivt[['time_to_ivt'] + dist_ivt_cols],
                on='time_to_ivt', how='left'
            )
            # Update the "ivt_mt" column where IVT is better:
            df.loc[mask_ivt_better, dist_cols] = df.loc[
                mask_ivt_better, dist_ivt_cols].values
            # Now remove the separate IVT columns:
            df = df.drop(dist_ivt_cols, axis='columns')

            # Store the result:
            mrs_lists_dict[key] = df
    return mrs_lists_dict, dist_cols


def calculate_average_mrs_dists(df, cols_here):
    # Split list of values into one column per mRS band
    # and keep one row per LSOA.
    vals = df[cols_here].copy()

    # Create stats from these data:
    weighted_stats = DescrStatsW(
        vals, weights=df['Admissions'], ddof=0)
    # Means (one value per mRS):
    means = weighted_stats.mean
    # Standard deviations (one value per mRS):
    stds = weighted_stats.std

    # Round these values:
    means = np.round(means, 3)
    stds = np.round(stds, 3)
    # Cumulative probability from the mean bins:
    cumulatives = np.cumsum(means)

    # Return:
    return means, cumulatives, stds


@st.cache_data
def calculate_average_mrs(
        occ_type,
        treat_type,
        col_region,
        str_selected_region,
        col_to_mask_mrs,
        dict_of_dfs,
        df_mrs_ivt,
        df_mrs_mt,
        input_dict,
        ):
    # Store results in here:
    dict_averaged = dict([(k, {}) for k in ['no_treatment'] + list(dict_of_dfs.keys())])

    # Find reference mRS distributions (no treatment):
    try:
        prop_nlvo = input_dict['prop_nlvo']
        prop_lvo = input_dict['prop_lvo']
    except KeyError:
        prop_nlvo = None
        prop_lvo = None
    dist_ref_cum, dist_ref_noncum = load_no_treatment_mrs_dists(
        occ_type, prop_nlvo, prop_lvo)
    # Store no-treatment data:
    dict_averaged['no_treatment'] = {
        'noncum': dist_ref_noncum,
        'cum': dist_ref_cum,
        'std':  None
    }

    # Decide whether to use no-treatment dists or to
    # fish dists out of the big mRS lists.
    use_ref_data = False
    if (occ_type == 'nlvo') & (treat_type == 'mt'):
        use_ref_data = True
        # mRS dists don't exist. Use reference data.
        for key in list(dict_averaged.keys()):
            dict_averaged[key]['noncum'] = dist_ref_noncum
            dict_averaged[key]['cum'] = dist_ref_cum
            dict_averaged[key]['std'] = None
    elif (occ_type == 'nlvo') & ('mt' in treat_type):
        # Use IVT data instead of IVT & MT.
        treat_type = 'ivt'

    if use_ref_data is False:
        lsoa_names = find_lsoa_names_to_keep(
            dict_of_dfs['usual_care'],
            col_to_mask_mrs,
            col_region,
            str_selected_region
            )
        dict_of_dfs, dist_cols = find_total_mrs_for_unique_times(
            dict_of_dfs,
            lsoa_names,
            treat_type,
            occ_type,
            df_mrs_ivt,
            df_mrs_mt,
            )
        for key, df in dict_of_dfs.items():
            dist_noncum, dist_cum, dist_std = (
                calculate_average_mrs_dists(df, dist_cols))
            # Store in results dict:
            dict_averaged[key]['noncum'] = dist_noncum
            dict_averaged[key]['cum'] = dist_cum
            dict_averaged[key]['std'] = dist_std

    return dict_averaged


def setup_for_mrs_dist_bars(dict_averaged):
    # Display names for the data:
    display_name_dict = {
        'usual_care': 'Usual care',
        'drip_ship': 'Usual care',
        'redirect': 'Redirection',
        'redirection_considered': 'Redirection considered',
        'redirection_rejected': 'Redirection rejected',
        'redirection_approved': 'Redirection approved',
        'msu': 'MSU',
        'no_treatment': 'No treatment'
    }

    # Seaborn-colorblind colours:
    # #0072b2  blue
    # #009e73  green
    # #d55e00  red
    # #cc79a7  pink
    # #f0e442  yellow
    # #56b4e9  light blue
    colours = ['grey', '#0072b2', '#56b4e9', '#009e73', '#f0e442']
    linestyles = [None, 'dash', 'dot', 'dashdot']
    i = 0

    format_dicts = {}

    for key, df in dict_averaged.items():
        # Pick out the nicer-formatted name if it exists
        # or use the current name if not.
        try:
            display0 = display_name_dict[key]
        except KeyError:
            display0 = key

        dict_here = {
            'noncum': df['noncum'],
            'cum': df['cum'],
            'std': df['std'],
            'colour': colours[i],
            'linestyle': linestyles[i],
        }
        format_dicts[display0] = dict_here
        i += 1

    return format_dicts


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
    # Give enough of a top margin that the main title doesn't
    # clash with the top subplot title.
    fig.update_layout(
        # width=1200,
        height=700,
        margin_t=150,
        )

    return fig
