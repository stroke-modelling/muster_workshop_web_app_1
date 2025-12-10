"""
Pathway timings.
"""
import streamlit as st
import pandas as pd
import numpy as np

from utilities.utils import print_progress_loc, set_inputs_changed, \
    make_formatted_time_str
import utilities.plot_timeline as timeline


def select_pathway_timings(use_col: str, containers: list):
    """
    Display widget for each pathway timing, store in dataframe.

    Default values from median onset to arrival times document
    (Mike Allen, 23rd April 2024):
    onset_to_call: 79,
    call_to_ambulance_arrival_time: 18,
    ambulance_on_scene_time: 29,

    Inputs
    ------
    use_col    - str. Either 'optimist' or 'muster'. Limits the drawn
                 input widgets to only the relevant variables. The
                 reference dataframe contains a mix of Optimist and
                 Muster data.
    containers - list. List of containers to draw the input widgets in.
                 Currently placement is hard-coded and assumes two
                 (Optimist) or three (Muster) containers in list.

    Returns
    -------
    df_pathway - pd.DataFrame. User-selected pathway timings.
    """
    # Load in pathway timings from file:
    df_pathway = pd.read_csv('./data/pathway_timings.csv', index_col='name')
    # Limit to the keys we need:
    df_pathway = df_pathway[df_pathway[use_col] == 1]
    # Make a new column to store the user-selected value.
    df_pathway['value'] = df_pathway['default'].copy()

    # Draw each widget separately:
    for key in df_pathway.index:
        if 'ambulance' in key:
            c = containers[0]
        elif ('msu' in key) & ('arrival' not in key):
            c = containers[2]
        else:
            c = containers[1]

        d = df_pathway.loc[key]
        with c:
            # Convert types to match step to make sure that numeric
            # dtypes match, e.g. all int or all float.
            df_pathway.loc[key, 'value'] = st.number_input(
                d['label'],
                value=d['default'].astype(type(d['step'])),
                min_value=d['min_value'].astype(type(d['step'])),
                max_value=d['max_value'].astype(type(d['step'])),
                step=d['step'],
                format='%0.0f',
                help=f"Reference value: {d['default']}",
                on_change=set_inputs_changed,
                )
    cols_to_keep = ['label', 'value']
    return df_pathway[cols_to_keep]


def calculate_treatment_times_without_travel(
        d: pd.Series,
        scenarios: list,
        _log: bool = True,
        _log_loc: st.container = None
        ):
    """
    Calculate treatment times excluding travel from pathway steps.

    Check whether to include ambulance response. Want to use this
    fixed value in OPTIMIST but not in MUSTER.

    Inputs
    ------
    d         - pd.Series. Pathway timings.
    scenarios - list. Names of the scenarios to calculate treatment
                times for. Expecting usual_care, prehospdiag, and msu.
    _log      - bool. Whether to print log message.
    _log_loc  - st.container or None. Where to print log message.

    Returns
    -------
    r - pd.Series. Calculated treatment times for the given scenarios.
    """
    # Store results in here:
    r = pd.Series()

    # Check whether to include travel time of ambulance
    # before it reaches the patient. Want to include this time
    # for OPTIMIST (where the time is always the same) but not
    # for MUSTER (where the time depends on the MSU base).
    use_ambo_response = not any(['msu' in s for s in scenarios])

    for s in scenarios:
        if ('usual_care' in s) | ('prehospdiag' in s):
            # All "usual care" scenarios share these initial timings:
            t_ambo = [
                'process_time_call_ambulance',
                'process_ambulance_on_scene_duration',
            ]
            if use_ambo_response:
                t_ambo.append('process_time_ambulance_response')
            if 'prehospdiag' in s:
                p = 'process_ambulance_on_scene_diagnostic_duration'
                t_ambo.append(p)
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
        df: pd.DataFrame,
        unique_travel_ivt: list,
        unique_travel_mt: list,
        _log: bool = True,
        _log_loc: st.container = None,
        ):
    """
    Calculate the unique times to treatment with IVT and MT.

    Inputs
    ------
    df                - pd.DataFrame. Treatment times without travel.
    unique_travel_ivt - list. Unique total travel times to IVT unit.
    unique_travel_mt  - list. Unique total travel times to MT unit.
    _log              - bool. Whether to print log message.
    _log_loc          - st.container or None. Where to print log
                        message.

    Returns
    -------
    unique_ivt - list. Unique times to treatment with IVT.
    unique_mt  - list. Same as above but for times to MT.
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
        dict_time_pairs: dict,
        series_treatment_times_without_travel: pd.Series,
        _log: bool = True,
        _log_loc: st.container = None,
        ):
    """
    Find unique pairs of treatment times to IVT and to MT.

    Inputs
    ------
    dict_time_pairs                       - dict. Unique pairs of
                                            travel times to IVT and MT
                                            units.
    series_treatment_times_without_travel - pd.Series. Treatment times
                                            without travel for all
                                            scenarios.
    _log                                  - bool. Whether to print log
                                            message.
    _log_loc                              - st.container or None. Where
                                            to print log message.

    Returns
    -------
    df_treat - pd.DataFrame. The unique pairs of time to treatment with
               IVT and with MT.
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
                if 'travel_ambo_response' in df_times.columns:
                    # For MUSTER. Separate ambulance response time
                    # depending on whether usual care or MSU.
                    # ? December 2025 - why aren't we also adding the MSU response time here? TO DO
                    df_times['time_to_ivt'] += df_times['travel_ambo_response']
                    df_times['time_to_mt'] += df_times['travel_ambo_response']
                list_dfs.append(df_times[['time_to_ivt', 'time_to_mt']].copy())
    # Find unique pairs of treatment times:
    df_treat = (pd.concat(list_dfs, axis='rows', ignore_index=True)
                .drop_duplicates()
                .sort_values(['time_to_ivt', 'time_to_mt']))

    if _log:
        p = 'Found unique treatment time pairs.'
        print_progress_loc(p, _log_loc)
    return df_treat


def calculate_treatment_times_each_lsoa_scenarios(
        df_lsoa_units_times: pd.DataFrame,
        series_treatment_times: pd.Series,
        _log: bool = True,
        _log_loc: st.container = None,
        _test: bool = False
        ):
    """
    Calculate treatment times for each LSOA in OPTIMIST scenarios.

    Inputs
    ------
    df_lsoa_units_times    - pd.DataFrame. Each LSOA's assigned stroke
                             units and travel times.
    series_treatment_times - pd.Series. Times to treatment for each
                             scenario excluding travel times.
    _log                   - bool. Whether to print log message.
    _log_loc               - st.container or None. Where to print log
                             message.
    _test                  - bool. Whether to run sanity checks.

    Returns
    -------
    df_lsoa_units_times - pd.DataFrame. The input dataframe with extra
                          times for treatment in each scenario.
    """
    # LSOA-level scenario timings.
    mask = df_lsoa_units_times['transfer_required']
    # Usual care:
    s = 'usual_care'
    df_lsoa_units_times[f'{s}_ivt'] = (
        df_lsoa_units_times['nearest_ivt_time'] +
        series_treatment_times[f'{s}_time_to_ivt']
    )
    df_lsoa_units_times.loc[mask, f'{s}_mt'] = (
        df_lsoa_units_times['nearest_ivt_then_mt_time'] +
        series_treatment_times[f'{s}_time_to_mt_transfer']
    )
    df_lsoa_units_times.loc[~mask, f'{s}_mt'] = (
        df_lsoa_units_times['nearest_ivt_then_mt_time'] +
        series_treatment_times[f'{s}_time_to_mt_no_transfer']
    )
    # Redirection approved:
    s = 'redirection_approved'
    df_lsoa_units_times[f'{s}_ivt'] = (
        df_lsoa_units_times['nearest_mt_time'] +
        series_treatment_times['prehospdiag_time_to_ivt']
    )
    df_lsoa_units_times[f'{s}_mt'] = (
        df_lsoa_units_times['nearest_mt_time'] +
        series_treatment_times['prehospdiag_time_to_mt_no_transfer']
    )
    # Redirection rejected:
    s = 'redirection_rejected'
    df_lsoa_units_times[f'{s}_ivt'] = (
        df_lsoa_units_times['nearest_ivt_time'] +
        series_treatment_times['prehospdiag_time_to_ivt']
    )
    df_lsoa_units_times.loc[mask, f'{s}_mt'] = (
        df_lsoa_units_times['nearest_ivt_then_mt_time'] +
        series_treatment_times['prehospdiag_time_to_mt_transfer']
    )
    df_lsoa_units_times.loc[~mask, f'{s}_mt'] = (
        df_lsoa_units_times['nearest_ivt_then_mt_time'] +
        series_treatment_times['prehospdiag_time_to_mt_no_transfer']
    )
    if _test:
        # How many unique combinations of these times are there?
        # Default setup, roughly 33,000 LSOA and 9,000 combos.
        # Worth calculating for unique combinations of times and
        # then admissions-weighting.
        st.write(len(df_lsoa_units_times))
        scens = ['usual_care', 'redirection_approved', 'redirection_rejected']
        treats = ['ivt', 'mt']
        cols_treat_scen = [f'{s}_{t}' for s in scens for t in treats]
        cols_treat_scen = [c for c in cols_treat_scen
                           if c in df_lsoa_units_times.columns]
        st.write(len(
            df_lsoa_units_times[cols_treat_scen].drop_duplicates()))

    if _log:
        p = 'Found treatment times by LSOA for base scenarios.'
        print_progress_loc(p, _log_loc)
    return df_lsoa_units_times


def calculate_treatment_times_each_lsoa_scenarios_muster(
        df_lsoa_units_times: pd.DataFrame,
        series_treatment_times: pd.Series,
        _log: bool = True,
        _log_loc: st.container = None,
        _test: bool = False
        ):
    """
    Calculate treatment times for each LSOA in MUSTER scenarios.

    Inputs
    ------
    df_lsoa_units_times    - pd.DataFrame. Each LSOA's assigned stroke
                             units and travel times.
    series_treatment_times - pd.Series. Times to treatment for each
                             scenario excluding travel times.
    _log                   - bool. Whether to print log message.
    _log_loc               - st.container or None. Where to print log
                             message.
    _test                  - bool. Whether to run sanity checks.

    Returns
    -------
    df_lsoa_units_times - pd.DataFrame. The input dataframe with extra
                          times for treatment in each scenario.
    """
    # LSOA-level scenario timings.
    mask = df_lsoa_units_times['transfer_required']
    # Usual care:
    s = 'usual_care'
    df_lsoa_units_times[f'{s}_ivt'] = (
        df_lsoa_units_times['ambo_response_then_nearest_ivt_time'] +
        series_treatment_times[f'{s}_time_to_ivt']
    )
    df_lsoa_units_times.loc[mask, f'{s}_mt'] = (
        df_lsoa_units_times['ambo_response_then_nearest_ivt_then_mt_time'] +
        series_treatment_times[f'{s}_time_to_mt_transfer']
    )
    df_lsoa_units_times.loc[~mask, f'{s}_mt'] = (
        df_lsoa_units_times['ambo_response_then_nearest_mt_time'] +
        series_treatment_times[f'{s}_time_to_mt_no_transfer']
    )
    # MSU (IVT):
    s = 'msu_ivt'
    df_lsoa_units_times[f'{s}_ivt'] = (
        df_lsoa_units_times['msu_response_time'] +
        series_treatment_times['msu_time_to_ivt']
    )
    df_lsoa_units_times[f'{s}_mt'] = (
        df_lsoa_units_times['msu_response_then_mt_time'] +
        series_treatment_times['msu_time_to_mt_after_ivt']
    )
    # MSU (no IVT):
    s = 'msu_no_ivt'
    # Don't calculate any times for IVT in this scenario.
    df_lsoa_units_times[f'{s}_mt'] = (
        df_lsoa_units_times['msu_response_then_mt_time'] +
        series_treatment_times['msu_time_to_mt_no_ivt']
    )

    if _test:
        # How many unique combinations of these times are there?
        # Default setup, roughly 33,000 LSOA and 9,000 combos.
        # Worth calculating for unique combinations of times and
        # then admissions-weighting.
        st.write(len(df_lsoa_units_times))
        scens = ['usual_care', 'msu_ivt', 'msu_no_ivt']
        treats = ['ivt', 'mt']
        cols_treat_scen = [f'{s}_{t}' for s in scens for t in treats]
        cols_treat_scen = [c for c in cols_treat_scen
                           if c in df_lsoa_units_times.columns]
        st.write(len(
            df_lsoa_units_times[cols_treat_scen].drop_duplicates()))

    if _log:
        p = 'Found treatment times by LSOA for base scenarios.'
        print_progress_loc(p, _log_loc)
    return df_lsoa_units_times


def draw_timeline(
        df_pathway_steps: pd.DataFrame,
        series_treatment_times_without_travel: pd.Series,
        use_msu: bool = False
        ):
    """
    Calculate some extra data and then draw the timeline.

    Inputs
    ------
    df_pathway_steps                      - pd.DataFrame. Each timing
                                            step in the pathway.
    series_treatment_times_without_travel - pd.Series. Total times for
                                            summary labels.
    use_msu                               - bool. Whether to set up
                                            and draw MUSTER bits.
    """
    # ----- Timeline -----
    # Calculate some extra keys:
    df_pathway_steps.loc['onset'] = 0
    df_pathway_steps.loc['arrival_ivt_only'] = 0
    df_pathway_steps.loc['arrival_ivt_mt'] = 0
    df_pathway_steps.loc['needle_to_door_out'] = (
        df_pathway_steps.loc['transfer_time_delay', 'value'] -
        df_pathway_steps.loc['process_time_arrival_to_needle', 'value']
    )
    df_pathway_steps.loc['needle_to_puncture'] = (
        df_pathway_steps.loc['process_time_arrival_to_puncture', 'value'] -
        df_pathway_steps.loc['process_time_arrival_to_needle', 'value']
    )
    if use_msu:
        # Times for MUSTER:
        df_pathway_steps.loc['arrival_ambo'] = 0
        df_pathway_steps.loc['arrival_msu'] = 0
    else:
        # Times for OPTIMIST:
        df_pathway_steps.loc['ambo_arrival_to_prehospdiag'] = (
            df_pathway_steps.loc[
                'process_ambulance_on_scene_duration', 'value']
        )

    df_treats = pd.read_csv('./data/timeline_treatment_time_lookup.csv')
    # Only keep the rows we need:
    if use_msu:
        df_treats = df_treats.loc[df_treats['muster'] == 1]
    else:
        df_treats = df_treats.loc[df_treats['optimist'] == 1]
    df_treats = df_treats.drop(['optimist', 'muster'], axis='columns')

    # Format the times more nicely for display:
    for i in df_treats.index:
        s = df_treats.loc[i, 'source_time']
        if s != 'none':
            t = series_treatment_times_without_travel[s]
            # Convert minutes to hour-minute strings:
            df_treats.loc[i, 'min'] = t
            df_treats.loc[i, 'hr_min'] = make_formatted_time_str(t)
        else:
            df_treats.loc[i, 'min'] = np.NaN
            df_treats.loc[i, 'hr_min'] = '-'

    timeline.make_timeline_fig(df_pathway_steps, df_treats, use_msu=use_msu)
