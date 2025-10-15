"""
Set up and plot mRS distributions.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW  # for mRS dist stats
from pandas.api.types import is_numeric_dtype

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
        occ_types,
        df_mrs_ivt,
        df_mrs_mt,
        multi_scens=[]
        ):

    df_mrs_ivt = df_mrs_ivt.copy()
    df_mrs_mt = df_mrs_mt.copy()

    mrs_lists_dict = {}
    # Limit the dataframes to only these LSOA:
    for key, df in dict_of_dfs.items():
        # Keep only these LSOA rows.
        df = df.loc[df.index.isin(lsoa_names)].copy()
        # Jettison the LSOA names:
        df = df.reset_index().drop('lsoa', axis='columns')
        # Drop string columns:
        df = df[df.columns[[is_numeric_dtype(df[c]) for c in df.columns]]]
        # Store the result:
        mrs_lists_dict[key] = df

    dict_dfs_to_store = {}
    all_dist_cols = []

    for occ_type in occ_types:
        if treat_type in ['ivt', 'mt']:
            # Only look at the the matching columns.
            df_mrs = df_mrs_ivt if treat_type == 'ivt' else df_mrs_mt
            # Which mRS dist columns do we want?
            dist_cols = [f'{occ_type}_{treat_type}_mrs_dists_noncum_{i}'
                         for i in range(7)]
            if ((treat_type == 'mt') & (occ_type == 'nlvo')):
                # Use no-treatment data.
                dist_dict = load_reference_mrs_dists()
                dist_ref_noncum = dist_dict['nlvo_no_treatment_noncum']
                df_mrs[dist_cols] = dist_ref_noncum
            col_time = f'time_to_{treat_type}'
            if len(multi_scens) == 0:
                cols_time = [col_time]
                suffixes = ['']
            else:
                cols_time = [f'{s}_time_to_{treat_type}' for s in multi_scens]
                suffixes = [f'_{s}' for s in multi_scens]
            # Merge in these mRS dists:
            for key, df in mrs_lists_dict.items():
                # How many admissions are there for each treatment time?
                df = df[['Admissions'] + cols_time].groupby(cols_time).sum()
                df = df.reset_index()
                all_new_cols = []
                for c, col_here in enumerate(cols_time):
                    # Merge in the mRS dists:
                    df = pd.merge(
                        df,
                        df_mrs[[col_time] + dist_cols],
                        left_on=col_here, right_on=col_time, how='left',
                    )
                    dist_cols_suff = [d+suffixes[c] for d in dist_cols]
                    df = df.rename(columns=dict(
                        zip(dist_cols, dist_cols_suff)))
                    all_new_cols += dist_cols_suff
                    if col_here != col_time:
                        df = df.drop(col_time, axis='columns')
                df = df.set_index(cols_time)
                try:
                    dict_dfs_to_store[key] = pd.merge(
                        dict_dfs_to_store[key], df[all_new_cols],
                        left_index=True, right_index=True
                        )
                except KeyError:
                    dict_dfs_to_store[key] = df
            # Store the result:
            all_dist_cols += all_new_cols
        else:
            # Pull in a mix of IVT and MT data.
            if len(multi_scens) == 0:
                cols_time = [('time_to_ivt', 'time_to_mt')]
                cols_group = ['time_to_mt', 'time_to_ivt',
                              'lvo_ivt_better_than_mt']
                cols_better = ['lvo_ivt_better_than_mt']
                suffixes = ['']
            else:
                cols_time = []
                cols_group = []
                cols_better = []
                suffixes = []
                for s in multi_scens:
                    cols_time += [(f'{s}_time_to_ivt', f'{s}_time_to_mt')]
                    cols_group += [f'{s}_time_to_mt', f'{s}_time_to_ivt',
                                   f'{s}_lvo_ivt_better_than_mt']
                    cols_better += [f'{s}_lvo_ivt_better_than_mt']
                    suffixes += [f'_{s}']
            # Set up column names:
            dist_cols = [f'{occ_type}_{treat_type}_mrs_dists_noncum_{i}'
                         for i in range(7)]
            dist_mt_cols = [f'{occ_type}_mt_mrs_dists_noncum_{i}'
                            for i in range(7)]
            dist_ivt_cols = [f'{occ_type}_ivt_mrs_dists_noncum_{i}'
                             for i in range(7)]

            # Merge in these mRS dists:
            for key, df in mrs_lists_dict.items():
                # How many patients have each combination of time to
                # treatment and treatment souce (IVT or MT dist)?
                df = df.groupby(cols_group).sum().reset_index()

                all_new_cols = []
                for c, (col_ivt, col_mt) in enumerate(cols_time):
                    dist_cols_here = [f'{d}{suffixes[c]}' for d in dist_cols]
                    col_better = cols_better[c]
                    if occ_type == 'lvo':
                        mask_ivt_better = df[col_better] == True
                        # Initially copy over all MT results:
                        # Note, MUSTER special case for different time
                        # to MT depending on whether IVT given should
                        # already have been applied and saved as
                        # "time to mt" before this function.
                        df = pd.merge(
                            df, df_mrs_mt[['time_to_mt'] + dist_mt_cols],
                            left_on=col_mt, right_on='time_to_mt', how='left'
                        )
                        if col_mt != 'time_to_mt':
                            df = df.drop('time_to_mt', axis='columns')
                        # Rename "mt" to "ivt_mt":
                        df = df.rename(columns=dict(
                            zip(dist_mt_cols, dist_cols_here)))
                        # Now copy over all IVT results:
                        df = pd.merge(
                            df, df_mrs_ivt[['time_to_ivt'] + dist_ivt_cols],
                            left_on=col_ivt, right_on='time_to_ivt',
                            how='left',
                        )
                        if col_ivt != 'time_to_ivt':
                            df = df.drop('time_to_ivt', axis='columns')
                        # Update the "ivt_mt" column where IVT is better:
                        df.loc[mask_ivt_better, dist_cols_here] = (
                            df.loc[mask_ivt_better, dist_ivt_cols].values)
                        # Now remove the separate IVT columns:
                        df = df.drop(dist_ivt_cols, axis='columns')
                    else:
                        # nLVO. Just use the IVT data.
                        # Copy over all IVT results:
                        df = pd.merge(
                            df, df_mrs_ivt[['time_to_ivt'] + dist_ivt_cols],
                            left_on=col_ivt, right_on='time_to_ivt',
                            how='left',
                        )
                        if col_ivt != 'time_to_ivt':
                            df = df.drop('time_to_ivt', axis='columns')
                        # Rename "ivt" to "ivt_mt":
                        df = df.rename(columns=dict(
                            zip(dist_ivt_cols, dist_cols_here)))
                    all_new_cols += dist_cols_here
                df = df.set_index(cols_group)
                try:
                    dict_dfs_to_store[key] = pd.merge(
                        dict_dfs_to_store[key], df[all_new_cols],
                        left_index=True, right_index=True
                        )
                except KeyError:
                    dict_dfs_to_store[key] = df

                # Store the result:
                all_dist_cols += all_new_cols

    # Store the result:
    for key, df in dict_dfs_to_store.items():
        dict_dfs_to_store[key] = df.reset_index()
    return dict_dfs_to_store, all_dist_cols


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
    colours = ['grey', '#0072b2', '#56b4e9', '#009e73', '#cc79a7']
    linestyles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
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
