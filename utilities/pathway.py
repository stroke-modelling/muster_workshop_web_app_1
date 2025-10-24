"""
Pathway timings.
"""
import streamlit as st
import pandas as pd
import numpy as np

from utilities.utils import print_progress_loc


def select_pathway_timings(use_col):
    """
    This version creates a long list of number inputs.

    TO DO another day - set these reference values up in fixed_params.
    Default values from median onset to arrival times document
    (Mike Allen, 23rd April 2024):
    onset_to_call: 79,
    call_to_ambulance_arrival_time: 18,
    ambulance_on_scene_time: 29,
    """
    # Load in pathway timings from file:
    df_pathway = pd.read_csv('./data/pathway_timings.csv', index_col='name')
    # Limit to the keys we need:
    df_pathway = df_pathway[df_pathway[use_col] == 1]
    # Make a new column to store the user-selected value.
    df_pathway['value'] = df_pathway['default'].copy()

    # Draw each widget separately:
    for key in df_pathway.index:
        d = df_pathway.loc[key]
        # Convert types to match step to make sure that numeric
        # dtypes match, e.g. all int or all float.
        df_pathway.loc[key, 'value'] = st.number_input(
            d['label'],
            value=d['default'].astype(type(d['step'])),
            min_value=d['min_value'].astype(type(d['step'])),
            max_value=d['max_value'].astype(type(d['step'])),
            step=d['step'],
            help=f"Reference value: {d['default']}",
            # key=key
            )
    cols_to_keep = ['label', 'value']
    return df_pathway[cols_to_keep]


def calculate_treatment_times_without_travel(
        df_pathway, scenarios, _log=True, _log_loc=None
        ):
    """
    df_pathway: index is variable name, columns are label and value.
    """
    # Turn into a series:
    d = df_pathway['value']
    r = pd.Series()

    for s in scenarios:
        if ('usual_care' in s) | ('prehospdiag' in s):
            # All "usual care" scenarios share these initial timings:
            t_ambo = [
                'process_time_call_ambulance',
                'process_time_ambulance_response',
                'process_ambulance_on_scene_duration',
            ]
            # Differences in times between treatment types and transfers:
            t_ivt = t_ambo + [
                'process_time_arrival_to_needle'
            ]
            t_mt_transfer = t_ambo + [
                'transfer_time_delay',
                'process_time_transfer_arrival_to_puncture',
            ]
            t_mt_no_transfer = t_ambo + [
                'process_time_arrival_to_puncture',
            ]
            if 'prehospdiag' in s:
                p = 'process_ambulance_on_scene_diagnostic_duration'
                t_ivt.append(p)
                t_mt_transfer.append(p)
                t_mt_no_transfer.append(p)
            # Sum all the pathway steps:
            r[f'{s}_time_to_ivt'] = d[t_ivt].sum()
            r[f'{s}_time_to_mt_transfer'] = d[t_mt_transfer].sum()
            r[f'{s}_time_to_mt_no_transfer'] = d[t_mt_no_transfer].sum()
        else:
            # MSU.
            t_ambo = [
                'process_time_call_ambulance',
                'process_msu_dispatch',
            ]
            t_ivt = t_ambo + [
                'process_msu_thrombolysis',
                ]
            t_mt_after_ivt = t_ambo + [
                'process_msu_thrombolysis',
                'process_msu_on_scene_post_thrombolysis',
                'process_time_msu_arrival_to_puncture',
            ]
            t_mt_no_ivt = t_ambo + [
                'process_msu_on_scene_no_thrombolysis',
                'process_time_msu_arrival_to_puncture',
            ]
            # Sum all the pathway steps:
            r[f'{s}_time_to_ivt'] = d[t_ivt].sum()
            r[f'{s}_time_to_mt_after_ivt'] = d[t_mt_after_ivt].sum()
            r[f'{s}_time_to_mt_no_ivt'] = d[t_mt_no_ivt].sum()

    if _log:
        p = 'Calculated treatment times without travel.'
        print_progress_loc(p, _log_loc)
    return r


def calculate_treatment_times(
        df,
        unique_travel_ivt,
        unique_travel_mt,
        _log=True,
        _log_loc=None,
        ):
    """
    df: df_treatment_times_without_travel
    """

    keys_ivt_without_travel = [t for t in df.index if 'time_to_ivt' in t]
    keys_mt_without_travel = [t for t in df.index if 'time_to_mt' in t]
    times_ivt = df[keys_ivt_without_travel].values
    times_mt = df[keys_mt_without_travel].values

    unique_ivt = sorted(list(set(sum(
        [list(np.array(unique_travel_ivt) + t) for t in times_ivt], []))))
    unique_mt = sorted(list(set(sum(
        [list(np.array(unique_travel_mt) + t) for t in times_mt], []))))

    if _log:
        p = 'Calculated unique treatment times.'
        print_progress_loc(p, _log_loc)
    return unique_ivt, unique_mt


def find_unique_treatment_time_pairs(
        dict_time_pairs, series_treatment_times_without_travel,
        _log=True, _log_loc=None,
        ):
    """
    """
    ivt_times = [
        t for t in series_treatment_times_without_travel.keys()
        if ('time_to_ivt' in t)
        ]
    mt_times = [
        t for t in series_treatment_times_without_travel.keys()
        if ('time_to_mt' in t)
        ]

    list_dfs = []
    for k, df_times in dict_time_pairs.items():
        for ivt_time in ivt_times:
            t_ivt = series_treatment_times_without_travel[ivt_time]
            for mt_time in mt_times:
                t_mt = series_treatment_times_without_travel[mt_time]
                df_times['time_to_ivt'] = df_times['travel_for_ivt'] + t_ivt
                df_times['time_to_mt'] = df_times['travel_for_mt'] + t_mt
                list_dfs.append(df_times[['time_to_ivt', 'time_to_mt']].copy())
    # Find unique pairs of treatment times:
    df_treat = (pd.concat(list_dfs, axis='rows').drop_duplicates()
                .sort_values(['time_to_ivt', 'time_to_mt']))

    if _log:
        p = 'Found unique treatment time pairs.'
        print_progress_loc(p, _log_loc)
    return df_treat
