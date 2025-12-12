"""
Calculate outcomes.
"""
import streamlit as st
import pandas as pd
import numpy as np

import stroke_outcome  # for reference dists
import classes.model_module as model

from utilities.utils import print_progress_loc, set_rerun_map


def select_outcome_type():
    """
    User selection of which outcome to show on the maps.

    This function shows nicer labels for display on the app.

    Returns
    -------
    outcome_type - str. The name of the selected outcome as it appears
                   in the outcome dataframes.
    """
    outcome_type_dict = {
        'utility_shift': 'Added utility',
        'mrs_0-2': 'mRS <= 2',
    }

    def f(key):
        return outcome_type_dict[key]
    # Outcome type input:
    outcome_type = st.radio(
        'Outcome measure',
        ['utility_shift', 'mrs_0-2'],
        index=0,  # 'added utility' as default
        format_func=f,
        # horizontal=True
        on_change=set_rerun_map
    )
    return outcome_type


def load_no_treatment_outcomes(
        _log: bool = True, _log_loc: st.container = None):
    """
    Load the no-treatment mRS distributions.

    Inputs
    ------
    _log     - bool. Whether to print log message.
    _log_loc - st.container or None. Where to print log message.

    Returns
    -------
    dict_no_treatment_outcomes - dict. Keys for nLVO and LVO, values
                                 are Series with mRS distributions
                                 and changes from no treatment (for
                                 consistency with outcome results).
    """
    # mRS distributions:
    mrs_dists_ref = (
        stroke_outcome.outcome_utilities.import_mrs_dists_from_file())
    label_dict = {
        'no_treatment_nlvo': 'nlvo_no_treatment',
        'no_treatment_lvo': 'lvo_no_treatment'
        }

    dict_no_treatment_outcomes = {}
    for dist_key, new_key in label_dict.items():
        dist = mrs_dists_ref.loc[dist_key].values
        # Place in Series:
        s = pd.Series(dist, index=[f'mrs_dists_{i}' for i in range(7)])
        # Calculate other outcomes.
        s['mrs_0-2'] = dist[2]
        # Changes from "no treatment":
        s['mrs_shift'] = 0.0
        s['utility_shift'] = 0.0
        # Store:
        dict_no_treatment_outcomes[new_key] = pd.DataFrame(s).transpose()

    if _log:
        p = 'Loaded reference no-treatment outcomes.'
        print_progress_loc(p, _log_loc)
    return dict_no_treatment_outcomes


def calculate_unique_outcomes(
        times_to_ivt: list,
        times_to_mt: list,
        _log: bool = True,
        _log_loc: st.container = None
        ):
    """
    Calculate outcomes for the unique treatment times.

    Run the continuous model from the stroke-outcome package.

    Inputs
    ------
    times_to_ivt - list. Unique onset to thrombolysis times.
    times_to_mt  - list. Unique onset to thrombectomy times.
    _log         - bool. Whether to print log message.
    _log_loc     - st.container or None. Where to print log message.

    Returns
    -------
    d - dict. One key per population (nLVO IVT, LVO IVT, LVO MT).
        Each value is a dataframe with the keys: time_to_ivt or
        time_to_mt; mrs_0-2, mrs_shift, utility_shift,
        mrs_dists_i for i in 0 to 6.
    """
    times = {'ivt': list(np.array(times_to_ivt, dtype=float)),
             'mt': list(np.array(times_to_mt, dtype=float))}
    # Run results for IVT and for MT separately.
    outcomes_by_stroke_type = {
        'ivt': run_outcome_model_for_unique_times_ivt(times['ivt']),
        'mt': run_outcome_model_for_unique_times_mt(times['mt']),
    }
    # Gather results for combinations of these:
    pops = ['nlvo_ivt', 'lvo_ivt', 'lvo_mt']
    outcome_measures = ['mrs_0-2', 'mrs_shift', 'utility_shift']

    d = {}
    for pop in pops:
        t = pop.split('_')[1]
        # Outcomes:
        keys = [f'{pop}_each_patient_{m}' for m in outcome_measures]
        vals = [outcomes_by_stroke_type[t][k] for k in keys]
        # mRS dists:
        key_mrs = f'{pop}_each_patient_mrs_dist_post_stroke'
        vals_mrs = outcomes_by_stroke_type[t][key_mrs]
        cols_mrs = [f'{pop}_mrs_dists_{i}' for i in range(7)]
        # Gather data:
        arrs = [
            np.array(times[t]).reshape(len(times[t]), 1),
            np.array(vals).T,
            vals_mrs,
            ]
        df = pd.DataFrame(
            np.hstack(arrs),
            columns=[f'time_to_{t}'] + keys + cols_mrs
        )
        df = np.round(df, 5)
        # Remove "each patient" and occlusion/treatment from columns:
        new_cols = [c.replace('_each_patient', '').replace(f'{pop}_', '')
                    for c in df.columns]
        df = df.rename(columns=dict(zip(df.columns, new_cols)))
        d[pop] = df

    if _log:
        p = 'Calculated unique outcomes for base scenarios.'
        print_progress_loc(p, _log_loc)
    return d


def run_outcome_model_for_unique_times_ivt(times_to_ivt: np.array):
    """
    Run the stroke-outcome model for IVT only for unique times.

    Inputs
    ------
    times_to_ivt - list. Unique onset to thrombolysis times.

    Returns
    -------
    outcomes_by_stroke_type_ivt_only - dict. Results from the outcome
                                       model for nLVO and LVO.
    """
    # IVT:
    # Set up input table for stroke outcome package.
    outcome_inputs_df_ivt_only = pd.DataFrame()
    # Provide a dummy stroke type code for now -
    # it will be overwritten when the outcomes are calculated so that
    # both nLVO and LVO results are calculated from one scenario.
    outcome_inputs_df_ivt_only['stroke_type_code'] = (
        np.repeat(1, len(times_to_ivt)))
    # Set everyone to receive only IVT.
    outcome_inputs_df_ivt_only['ivt_chosen_bool'] = 1
    outcome_inputs_df_ivt_only['mt_chosen_bool'] = 0
    # And include the times we've just calculated:
    outcome_inputs_df_ivt_only['onset_to_needle_mins'] = times_to_ivt
    outcome_inputs_df_ivt_only['onset_to_puncture_mins'] = np.NaN

    # Run the outcome model for just these times.
    outcomes_by_stroke_type_ivt_only = model.run_outcome_model(
        outcome_inputs_df_ivt_only).copy()
    return outcomes_by_stroke_type_ivt_only


def run_outcome_model_for_unique_times_mt(times_to_mt: np.array):
    """
    Run the stroke-outcome model for MT only for unique times.

    Inputs
    ------
    times_to_mt - list. Unique onset to thrombectomy times.

    Returns
    -------
    outcomes_by_stroke_type_mt_only - dict. Results from the outcome
                                      model for nLVO and LVO.
    """
    # MT:
    # Set up input table for stroke outcome package.
    outcome_inputs_df_mt_only = pd.DataFrame()
    # Provide a dummy stroke type code for now -
    # it will be overwritten when the outcomes are calculated so that
    # both nLVO and LVO results are calculated from one scenario.
    outcome_inputs_df_mt_only['stroke_type_code'] = (
        np.repeat(1, len(times_to_mt)))
    # Set everyone to receive only MT.
    outcome_inputs_df_mt_only['ivt_chosen_bool'] = 0
    outcome_inputs_df_mt_only['mt_chosen_bool'] = 1
    # And include the times we've just calculated:
    outcome_inputs_df_mt_only['onset_to_needle_mins'] = np.NaN
    outcome_inputs_df_mt_only['onset_to_puncture_mins'] = times_to_mt

    # Run the outcome model for just these times.
    outcomes_by_stroke_type_mt_only = model.run_outcome_model(
        outcome_inputs_df_mt_only).copy()
    return outcomes_by_stroke_type_mt_only


def flag_lvo_ivt_better_than_mt(
        outcomes_lvo_ivt: pd.DataFrame,
        outcomes_lvo_mt: pd.DataFrame,
        unique_treatment_pairs: pd.DataFrame,
        _log: bool = True,
        _log_loc: st.container = None,
        ):
    """
    Find where IVT is better than MT for all pairs of treatment times.

    Inputs
    ------
    outcomes_lvo_ivt       - pd.DataFrame. Base outcomes for LVO
                             treated with IVT only.
    outcomes_lvo_mt        - pd.DataFrame. Base outcomes for LVO
                             treated with MT only.
    unique_treatment_pairs - pd.DataFrame. Pairs of times to treatment
                             with IVT and with MT.
    _log                   - bool. Whether to print log message.
    _log_loc               - st.container or None. Where to print log
                             message.

    Returns
    -------
    df - pd.DataFrame. Rows for unique times to treatment for IVT
         and for MT, and columns with comparison outcome measure
         mRS<=2 and a flag for whether IVT is better than MT.
    """
    outcomes_t = {'ivt': outcomes_lvo_ivt, 'mt': outcomes_lvo_mt}
    out = 'mrs_0-2'

    df = unique_treatment_pairs.copy()
    for t, outcomes in outcomes_t.items():
        # Pick out outcomes for this pair of times:
        df = pd.merge(df, outcomes[[f'time_to_{t}', out]],
                      on=f'time_to_{t}', how='left')
        df = df.rename(columns={out: f'{out}_{t}'})
    # Find where LVO IVT is better than LVO MT:
    df['ivt_better'] = (df[f'{out}_mt'] < df[f'{out}_ivt'])

    if _log:
        p = '''Marked unique treatment time pairs where LVO with IVT
        is better than MT.'''
        print_progress_loc(p, _log_loc)
    return df


def combine_lvo_ivt_mt_outcomes(
        df_lvo_ivt: pd.DataFrame,
        df_lvo_mt: pd.DataFrame,
        df_lvo_ivt_mt_better: pd.DataFrame,
        _log=True,
        _log_loc=None,
        ):
    """
    Gather outcomes for LVO IVT&MT, checking which treatment is better.

    Inputs
    ------
    df_lvo_ivt           - pd.DataFrame. Base outcomes for LVO
                           treated with IVT only.
    df_lvo_mt            - pd.DataFrame. Base outcomes for LVO
                           treated with MT only.
    df_lvo_ivt_mt_better - pd.DataFrame. Pairs of treatment times with
                           IVT and MT and flag for when IVT is better.
    _log                 - bool. Whether to print log message.
    _log_loc             - st.container or None. Where to print log
                           message.

    Returns
    -------
    df - pd.DataFrame. Outcomes for all pairs of times to treatment
         with IVT and MT where either the IVT or MT outcomes are used
         depending on which is better.
    """
    # Combine IVT&MT. Start with MT results and replace with
    # the IVT results when IVT is better than MT.
    df = df_lvo_ivt_mt_better.copy()
    df = pd.merge(
        df, df_lvo_mt,
        on='time_to_mt', how='left'
        )
    # Remove MT results where IVT is better:
    cols_outcomes = [c for c in df_lvo_mt.columns if 'time' not in c]
    df.loc[df['ivt_better'], cols_outcomes] = pd.NA
    # Substitute in the IVT results:
    df = (
        df.set_index('time_to_ivt').combine_first(
            df_lvo_ivt.set_index('time_to_ivt'))
        ).reset_index().set_index(['time_to_ivt', 'time_to_mt'])
    if _log:
        p = 'Gathered outcomes for LVO with both IVT and MT.'
        print_progress_loc(p, _log_loc)
    return df
