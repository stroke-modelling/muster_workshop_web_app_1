"""
Calculate outcomes.
"""
import streamlit as st
import pandas as pd
import numpy as np

import classes.model_module as model

from utilities.utils import print_progress_loc


def calculate_unique_outcomes(
        times_to_ivt, times_to_mt,
        _log=True, _log_loc=None
        ):
    """
    Run the outcome model for only the unique treatment times
    instead of one row per LSOA.
    """
    times = {'ivt': times_to_ivt, 'mt': times_to_mt}
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
        # Create non-cumulative mRS dists:
        vals_mrs_n = np.diff(vals_mrs, prepend=0.0)
        cols_mrs_n = [c.replace('dists_', 'dists_noncum_') for c in cols_mrs]
        # Gather data:
        arrs = [
            np.array(times[t]).reshape(len(times[t]), 1),
            np.array(vals).T,
            vals_mrs,
            vals_mrs_n
            ]
        df = pd.DataFrame(
            np.hstack(arrs),
            columns=[f'time_to_{t}'] + keys + cols_mrs + cols_mrs_n
        )
        df = np.round(df, 5)
        # Remove "each patient" and occlusion/treatment from columns:
        new_cols = [c.replace('_each_patient', '').replace(f'{pop}_', '')
                    for c in df.columns]
        r = dict(zip(df.columns, new_cols))
        df = df.rename(columns=r)
        d[pop] = df

    if _log:
        p = 'Calculated unique outcomes for base scenarios.'
        print_progress_loc(p, _log_loc)
    return d


def run_outcome_model_for_unique_times_ivt(times_to_ivt):
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


def run_outcome_model_for_unique_times_mt(times_to_mt):
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
        outcomes_lvo_ivt, outcomes_lvo_mt,
        unique_treatment_pairs,
        _log=True, _log_loc=None,
        ):
    """
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
        p = 'Marked unique treatment time pairs where LVO with IVT is better than MT in base scenarios.'
        print_progress_loc(p, _log_loc)
    return df
