
# ----- Imports -----
import streamlit as st
from importlib_resources import files
import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW  # for mRS dist stats
import geopandas

# Load region shapes:
import stroke_maps.load_data

# For running outcomes:
from classes.geography_processing import Geoprocessing
# from classes.model import Model
import classes.model_module as model

# Custom functions:
from utilities.maps import dissolve_polygons_by_value
# Containers:


@st.cache_data
def calculate_geography(df_unit_services):
    try:
        geo = st.session_state['geo']
    except KeyError:
        # Process and save geographic data (only needed when hospital data changes)
        geo = Geoprocessing(
            limit_to_england=True
            )

    # Update units:
    geo.df_unit_services = df_unit_services
    geo.update_unit_services()

    # Rerun geography:
    geo.run()

    # Reset index because Model expects a column named 'lsoa':
    df_geo = geo.get_combined_data().copy(deep=True).reset_index()

    # Cache the geo class so that on the next run all of the big
    # data files are not loaded in another time.
    st.session_state['geo'] = geo

    return df_geo.copy()


# ##########################################
# ##### TREATMENT TIMES WITHOUT TRAVEL #####
# ##########################################
def calculate_times_to_treatment_without_travel_usual_care(input_dict):
    # Usual care:
    # Time to IVT:
    usual_care_time_to_ivt = (
        input_dict['process_time_call_ambulance'] +
        input_dict['process_time_ambulance_response'] +
        input_dict['process_ambulance_on_scene_duration'] +
        input_dict['process_time_arrival_to_needle']
        )
    # Separate MT timings required depending on whether transfer
    # needed.
    # Timings for units needing transfers:
    usual_care_mt_transfer = (
        input_dict['process_time_call_ambulance'] +
        input_dict['process_time_ambulance_response'] +
        input_dict['process_ambulance_on_scene_duration'] +
        input_dict['transfer_time_delay'] +
        input_dict['process_time_transfer_arrival_to_puncture']
        )
    # Timings for units that do not need transfers:
    usual_care_mt_no_transfer = (
        input_dict['process_time_call_ambulance'] +
        input_dict['process_time_ambulance_response'] +
        input_dict['process_ambulance_on_scene_duration'] +
        input_dict['process_time_arrival_to_puncture']
        )

    # Gather these into a dictionary:
    d = {
        'usual_care_time_to_ivt': usual_care_time_to_ivt,
        'usual_care_mt_transfer': usual_care_mt_transfer,
        'usual_care_mt_no_transfer': usual_care_mt_no_transfer,
    }
    return d


def calculate_times_to_treatment_without_travel_msu(input_dict):
    # MSU:
    # Time to IVT:
    msu_time_to_ivt = (
        input_dict['process_time_call_ambulance'] +
        input_dict['process_msu_dispatch'] +
        input_dict['process_msu_thrombolysis']
        )
    # Time to MT after IVT and...:
    msu_time_to_mt = (
        input_dict['process_time_call_ambulance'] +
        input_dict['process_msu_dispatch'] +
        input_dict['process_msu_thrombolysis'] +
        input_dict['process_msu_on_scene_post_thrombolysis'] +
        input_dict['process_time_msu_arrival_to_puncture']
        )
    # ... after no IVT:
    msu_time_to_mt_no_ivt = (
        input_dict['process_time_call_ambulance'] +
        input_dict['process_msu_dispatch'] +
        input_dict['process_msu_on_scene_no_thrombolysis'] +
        input_dict['process_time_msu_arrival_to_puncture']
        )

    # Gather these into a dictionary:
    d = {
        'msu_time_to_ivt': msu_time_to_ivt,
        'msu_time_to_mt': msu_time_to_mt,
        'msu_time_to_mt_no_ivt': msu_time_to_mt_no_ivt,
    }
    return d


def calculate_times_to_treatment_without_travel_prehospdiag(input_dict):
    # Usual care:
    # Time to IVT:
    time_to_ivt = (
        input_dict['process_time_call_ambulance'] +
        input_dict['process_time_ambulance_response'] +
        input_dict['process_ambulance_on_scene_duration'] +
        input_dict['process_ambulance_on_scene_diagnostic_duration'] +
        input_dict['process_time_arrival_to_needle']
        )
    # Separate MT timings required depending on whether transfer
    # needed.
    # Timings for units needing transfers:
    mt_transfer = (
        input_dict['process_time_call_ambulance'] +
        input_dict['process_time_ambulance_response'] +
        input_dict['process_ambulance_on_scene_duration'] +
        input_dict['process_ambulance_on_scene_diagnostic_duration'] +
        input_dict['transfer_time_delay'] +
        input_dict['process_time_transfer_arrival_to_puncture']
        )
    # Timings for units that do not need transfers:
    mt_no_transfer = (
        input_dict['process_time_call_ambulance'] +
        input_dict['process_time_ambulance_response'] +
        input_dict['process_ambulance_on_scene_duration'] +
        input_dict['process_ambulance_on_scene_diagnostic_duration'] +
        input_dict['process_time_arrival_to_puncture']
        )

    # Gather these into a dictionary:
    d = {
        'prehospdiag_time_to_ivt': time_to_ivt,
        'prehospdiag_mt_transfer': mt_transfer,
        'prehospdiag_mt_no_transfer': mt_no_transfer,
    }
    return d


def calculate_treatment_times_each_lsoa(df_travel_times, treatment_time_dict):
    df_travel_times['usual_care_ivt'] = (
        treatment_time_dict['usual_care_time_to_ivt'] +
        df_travel_times['nearest_ivt_time']
    )
    # First set MT times assuming no transfer...
    df_travel_times['usual_care_mt'] = (
        treatment_time_dict['usual_care_mt_no_transfer'] +
        df_travel_times['nearest_ivt_time']
    )
    # ... then update the values that need a transfer:
    mask_usual_care_transfer = (df_travel_times['transfer_required'] == True)
    df_travel_times.loc[mask_usual_care_transfer, 'usual_care_mt'] = (
        treatment_time_dict['usual_care_mt_transfer'] +
        df_travel_times.loc[mask_usual_care_transfer, 'nearest_ivt_time'] +
        df_travel_times.loc[mask_usual_care_transfer, 'transfer_time']
    )
    # MSU IVT:
    df_travel_times['msu_ivt'] = (
        treatment_time_dict['msu_time_to_ivt'] +
        df_travel_times['nearest_msu_time']
    )
    # Time to MT when IVT is given:
    df_travel_times['msu_mt_with_ivt'] = (
        treatment_time_dict['msu_time_to_mt'] +
        df_travel_times['nearest_msu_time'] +
        df_travel_times['nearest_mt_time']
    )
    # Time to MT when IVT is not given:
    df_travel_times['msu_mt_no_ivt'] = (
        treatment_time_dict['msu_time_to_mt_no_ivt'] +
        df_travel_times['nearest_msu_time'] +
        df_travel_times['nearest_mt_time']
    )
    # Round all of these times to the nearest minute:
    cols_treatment_time = [
        'usual_care_ivt',
        'usual_care_mt',
        'msu_ivt',
        'msu_mt_with_ivt',
        'msu_mt_no_ivt'
    ]
    df_travel_times[cols_treatment_time] = np.round(
        df_travel_times[cols_treatment_time], 0)
    return df_travel_times


def calculate_treatment_times_each_lsoa_prehospdiag(
        df_travel_times, treatment_time_dict
        ):
    # Usual care:
    df_travel_times['usual_care_ivt'] = (
        treatment_time_dict['usual_care_time_to_ivt'] +
        df_travel_times['nearest_ivt_time']
    )
    # First set MT times assuming no transfer...
    df_travel_times['usual_care_mt'] = (
        treatment_time_dict['usual_care_mt_no_transfer'] +
        df_travel_times['nearest_ivt_time']
    )
    # ... then update the values that need a transfer:
    mask_usual_care_transfer = (df_travel_times['transfer_required'] == True)
    df_travel_times.loc[mask_usual_care_transfer, 'usual_care_mt'] = (
        treatment_time_dict['usual_care_mt_transfer'] +
        df_travel_times.loc[mask_usual_care_transfer, 'nearest_ivt_time'] +
        df_travel_times.loc[mask_usual_care_transfer, 'transfer_time']
    )

    # Redirection rejected:
    df_travel_times['redirection_rejected_ivt'] = (
        treatment_time_dict['prehospdiag_time_to_ivt'] +
        df_travel_times['nearest_ivt_time']
    )
    # First set MT times assuming no transfer...
    df_travel_times['redirection_rejected_mt'] = (
        treatment_time_dict['prehospdiag_mt_no_transfer'] +
        df_travel_times['nearest_ivt_time']
    )
    # ... then update the values that need a transfer:
    mask_usual_care_transfer = (df_travel_times['transfer_required'] == True)
    df_travel_times.loc[mask_usual_care_transfer, 'redirection_rejected_mt'] = (
        treatment_time_dict['prehospdiag_mt_transfer'] +
        df_travel_times.loc[mask_usual_care_transfer, 'nearest_ivt_time'] +
        df_travel_times.loc[mask_usual_care_transfer, 'transfer_time']
    )

    # Redirection approved:
    # Nobody needs a transfer for MT.
    df_travel_times['redirection_approved_ivt'] = (
        treatment_time_dict['prehospdiag_time_to_ivt'] +
        df_travel_times['nearest_mt_time']
    )
    # First set MT times assuming no transfer...
    df_travel_times['redirection_approved_mt'] = (
        treatment_time_dict['prehospdiag_mt_no_transfer'] +
        df_travel_times['nearest_mt_time']
    )

    # Round all of these times to the nearest minute:
    cols_treatment_time = [
        'usual_care_ivt',
        'usual_care_mt',
        'redirection_rejected_ivt',
        'redirection_rejected_mt',
        'redirection_approved_ivt',
        'redirection_approved_mt',
    ]
    df_travel_times[cols_treatment_time] = np.round(
        df_travel_times[cols_treatment_time], 0)
    return df_travel_times


# ##########################################
# ##### BUILD INPUTS FOR OUTCOME MODEL #####
# ##########################################
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


def convert_outcome_dicts_to_df_outcomes(
        times_to_ivt,
        times_to_mt,
        outcomes_by_stroke_type_ivt_only,
        outcomes_by_stroke_type_mt_only
        ):
    # Outcome columns and numbers of decimal places to round to:
    dict_outcomes_dp = {
        'mrs_0-2': 3,
        'mrs_shift': 3,
        # 'utility': 3,
        'utility_shift': 3,
    }
    occlusions = ['nlvo', 'lvo']
    outcome_measures = list(dict_outcomes_dp.keys())
    # Gather outcomes for these times:
    keys_for_df_ivt = [f'{o}_ivt_each_patient_{m}' for o in occlusions
                       for m in outcome_measures]
    df_outcomes_ivt = pd.DataFrame(
        [times_to_ivt] + [outcomes_by_stroke_type_ivt_only[k]
                          for k in keys_for_df_ivt],
        index=['time_to_ivt'] + keys_for_df_ivt
    ).transpose()

    keys_for_df_mt = [f'lvo_mt_each_patient_{m}' for m in outcome_measures]
    df_outcomes_mt = pd.DataFrame(
        [times_to_mt] + [outcomes_by_stroke_type_mt_only[k]
                         for k in keys_for_df_mt],
        index=['time_to_mt'] + keys_for_df_mt
    ).transpose()

    # Round data:
    for suff, dp in dict_outcomes_dp.items():
        for df in [df_outcomes_mt, df_outcomes_ivt]:
            cols = [c for c in df.columns if c.endswith(suff)]
            df[cols] = np.round(df[cols], dp)

    # Remove "each patient" from column names:
    r_ivt = dict(zip(
        df_outcomes_ivt.columns,
        [c.replace('_each_patient', '') for c in df_outcomes_ivt.columns]
    ))
    r_mt = dict(zip(
        df_outcomes_mt.columns,
        [c.replace('_each_patient', '') for c in df_outcomes_mt.columns]
    ))
    df_outcomes_ivt = df_outcomes_ivt.rename(columns=r_ivt)
    df_outcomes_mt = df_outcomes_mt.rename(columns=r_mt)

    return df_outcomes_ivt, df_outcomes_mt


def convert_outcome_dicts_to_df_mrs(
        times_to_ivt,
        times_to_mt,
        outcomes_by_stroke_type_ivt_only,
        outcomes_by_stroke_type_mt_only
        ):
    # Gather mRS distributions for these times:
    keys_for_df_mrs_ivt = [
        'nlvo_ivt_each_patient_mrs_dist_post_stroke',
        'lvo_ivt_each_patient_mrs_dist_post_stroke',
    ]
    column_names = (
        [f'nlvo_ivt_mrs_dists_{i}' for i in range(7)] +
        [f'lvo_ivt_mrs_dists_{i}' for i in range(7)]
    )

    df_mrs_ivt = pd.DataFrame(
        np.hstack(
            [np.array(times_to_ivt).reshape((len(times_to_ivt), 1))] +
            [outcomes_by_stroke_type_ivt_only[k] for k in keys_for_df_mrs_ivt]
            ),
        columns=['time_to_ivt'] + column_names
    )

    keys_for_df_mrs_mt = [
        'lvo_mt_each_patient_mrs_dist_post_stroke',
    ]
    column_names = (
        [f'lvo_mt_mrs_dists_{i}' for i in range(7)]
    )
    df_mrs_mt = pd.DataFrame(
        np.hstack(
            [np.array(times_to_mt).reshape((len(times_to_mt), 1))] +
            [outcomes_by_stroke_type_mt_only[k] for k in keys_for_df_mrs_mt]
            ),
        columns=['time_to_mt'] + column_names
    )

    # Remove "each patient" from column names:
    df_mrs_ivt = df_mrs_ivt.rename(columns=dict(zip(
        df_mrs_ivt.columns,
        [c.replace('_each_patient', '') for c in df_mrs_ivt.columns]
    )))
    df_mrs_mt = df_mrs_mt.rename(columns=dict(zip(
        df_mrs_mt.columns,
        [c.replace('_each_patient', '') for c in df_mrs_mt.columns]
    )))

    # Calculate non-cumulative probabilities:
    # Current columns:
    col_names_nlvo_ivt = (
        [f'nlvo_ivt_mrs_dists_{i}' for i in range(7)])
    col_names_lvo_ivt = (
        [f'lvo_ivt_mrs_dists_{i}' for i in range(7)])
    col_names_lvo_mt = (
        [f'lvo_mt_mrs_dists_{i}' for i in range(7)])
    # Columns to be created:
    col_names_noncum_nlvo_ivt = (
        [f'nlvo_ivt_mrs_dists_noncum_{i}' for i in range(7)])
    col_names_noncum_lvo_ivt = (
        [f'lvo_ivt_mrs_dists_noncum_{i}' for i in range(7)])
    col_names_noncum_lvo_mt = (
        [f'lvo_mt_mrs_dists_noncum_{i}' for i in range(7)])
    # Create the new data:
    df_mrs_ivt[col_names_noncum_nlvo_ivt] = np.diff(
        df_mrs_ivt[col_names_nlvo_ivt].values, prepend=0.0)
    df_mrs_ivt[col_names_noncum_lvo_ivt] = np.diff(
        df_mrs_ivt[col_names_lvo_ivt].values, prepend=0.0)
    df_mrs_mt[col_names_noncum_lvo_mt] = np.diff(
        df_mrs_mt[col_names_lvo_mt].values, prepend=0.0)
    return df_mrs_ivt, df_mrs_mt


# ####################################
# ##### GATHER FULL LSOA RESULTS #####
# ####################################
def build_full_lsoa_outcomes_from_unique_time_results(
            df_travel_times,
            df_outcomes_ivt,
            df_outcomes_mt,
            scenarios=[],
        ):
    outcome_measures = ['mrs_shift', 'utility_shift', 'mrs_0-2']
    cols_lvo_ivt_mt = [f'lvo_ivt_mt_{m}' for m in outcome_measures]
    cols_lvo_ivt = [f'lvo_ivt_{m}' for m in outcome_measures]
    cols_lvo_mt = [f'lvo_mt_{m}' for m in outcome_measures]

    # Apply these results to the treatment times in each scenario.
    df_lsoa = df_travel_times.copy().reset_index()

    keys_for_df_ivt = [k for k in df_outcomes_ivt.columns if 'time' not in k]
    keys_for_df_mt = [k for k in df_outcomes_mt.columns if 'time' not in k]

    for s in scenarios:
        # IVT:
        df_lsoa = pd.merge(
            df_lsoa, df_outcomes_ivt,
            left_on=f'{s}_ivt', right_on='time_to_ivt', how='left'
            ).drop('time_to_ivt', axis='columns')
        df_lsoa = df_lsoa.rename(columns=dict(
            zip(keys_for_df_ivt, [f'{s}_{k}' for k in keys_for_df_ivt])))
        # MT:
        df_lsoa = pd.merge(
            df_lsoa, df_outcomes_mt,
            left_on=f'{s}_mt', right_on='time_to_mt', how='left'
            ).drop('time_to_mt', axis='columns')
        df_lsoa = df_lsoa.rename(columns=dict(
            zip(keys_for_df_mt, [f'{s}_{k}' for k in keys_for_df_mt])))
        # IVT & MT:
        cols_s_lvo_ivt_mt = [f'{s}_{k}' for k in cols_lvo_ivt_mt]
        cols_s_lvo_ivt = [f'{s}_{k}' for k in cols_lvo_ivt]
        cols_s_lvo_mt = [f'{s}_{k}' for k in cols_lvo_mt]
        if s == 'msu':
            # Special case for MSU data. Use the times for IVT then MT.
            # The time to MT depends on whether IVT was given.
            # Set up the rename first.
            keys_msu_ivt_mt_from_df_mt = (
                [f'{s}_{k}'.replace('_mt_', '_ivt_mt_')
                 for k in keys_for_df_mt]
            )
            r = dict(zip(keys_for_df_mt, keys_msu_ivt_mt_from_df_mt))
            df_lsoa = pd.merge(
                df_lsoa, df_outcomes_mt.rename(columns=r),
                left_on='msu_ivt_mt', right_on='time_to_mt', how='left'
                ).drop('time_to_mt', axis='columns')
        else:
            # Initially copy over the MT data:
            df_lsoa[cols_s_lvo_ivt_mt] = df_lsoa[cols_s_lvo_mt].copy()
        # Find whether IVT is better than MT:
        mask_s_ivt_better = (
            df_lsoa[f'{s}_lvo_mt_mrs_0-2'] <
            df_lsoa[f'{s}_lvo_ivt_mrs_0-2']
        )
        # Store this mask for later use of picking out mRS dists:
        df_lsoa[f'{s}_lvo_ivt_better_than_mt'] = mask_s_ivt_better
        # Update outcomes for those patients:
        # (keep the .values because the column names don't match)
        df_lsoa.loc[mask_s_ivt_better, cols_s_lvo_ivt_mt] = (
            df_lsoa.loc[mask_s_ivt_better, cols_s_lvo_ivt].values)

    return df_lsoa


def gather_pdeath_from_unique_time_results(
            df_lsoa,
            df_mrs_ivt,
            df_mrs_mt,
            scenarios=[],
        ):
    """
    df_lsoa already contains masks for where IVT better than MT
    for each scenario.
    """
    # Apply these results to the treatment times in each scenario.

    for s in scenarios:
        # Copy over probability of death:
        # IVT:
        df_lsoa = pd.merge(
            df_lsoa, df_mrs_ivt[
                ['time_to_ivt', 'nlvo_ivt_mrs_dists_noncum_6', 'lvo_ivt_mrs_dists_noncum_6']],
            left_on=f'{s}_ivt', right_on='time_to_ivt', how='left'
            ).drop('time_to_ivt', axis='columns')
        df_lsoa = df_lsoa.rename(columns={
            'nlvo_ivt_mrs_dists_noncum_6': f'{s}_probdeath_nlvo_ivt',
            'lvo_ivt_mrs_dists_noncum_6': f'{s}_probdeath_lvo_ivt',
            })
        # MT:
        df_lsoa = pd.merge(
            df_lsoa, df_mrs_mt[['time_to_mt', 'lvo_mt_mrs_dists_noncum_6']],
            left_on=f'{s}_mt', right_on='time_to_mt', how='left'
            ).drop('time_to_mt', axis='columns')
        df_lsoa = df_lsoa.rename(columns={
            'lvo_mt_mrs_dists_noncum_6': f'{s}_probdeath_lvo_mt'})
        # IVT & MT:
        if s == 'msu':
            # Special case for MSU data. Use the times for IVT then MT.
            # The time to MT depends on whether IVT was given.
            df_lsoa = pd.merge(
                df_lsoa, df_mrs_mt[['time_to_mt', 'lvo_mt_mrs_dists_noncum_6']],
                left_on='msu_ivt_mt', right_on='time_to_mt', how='left'
                ).drop('time_to_mt', axis='columns')
            df_lsoa = df_lsoa.rename(columns={
                'lvo_mt_mrs_dists_noncum_6': f'{s}_probdeath_lvo_ivt_mt'})
        else:
            # Initially copy over the MT data:
            df_lsoa[f'{s}_probdeath_lvo_ivt_mt'] = (
                df_lsoa[f'{s}_probdeath_lvo_mt'].copy())
        # Update values where IVT is better than MT:
        # (keep the .values because the column names don't match)
        mask = df_lsoa[f'{s}_lvo_ivt_better_than_mt']
        df_lsoa.loc[mask, f'{s}_probdeath_lvo_ivt_mt'] = (
            df_lsoa.loc[mask, f'{s}_probdeath_lvo_ivt'].values)
    return df_lsoa


# ###########################
# ##### AVERAGE RESULTS #####
# ###########################
@st.cache_data
def group_results_by_region(df_lsoa, df_unit_services, df_lsoa_regions):
    df_lsoa = df_lsoa.copy()
    # ----- LSOAs for grouping results -----
    # Merge in other region info.
    df_lsoa = pd.merge(
        df_lsoa, df_lsoa_regions,
        on='lsoa', how='left'
        )
    # Replace some zeros with NaN:
    mask = df_lsoa['transfer_required']
    df_lsoa.loc[~mask, 'transfer_time'] = pd.NA

    # Remove string columns and columns that won't make sense
    # when averaged (e.g. admissions numbers).
    # (temporary - I don't know how else to groupby a df with
    # some object columns)
    cols_to_drop = [
        'lsoa',
        'lsoa_code',
        'nearest_ivt_unit_name',
        'nearest_mt_unit',
        'transfer_unit',
        'nearest_msu_unit',
        'nearest_mt_unit_name',
        'transfer_unit_name',
        'nearest_msu_unit_name',
        'short_code',
        'country',
        'England',
        'Admissions'
        ]
    # Only keep cols that exist (sometimes have MSU, sometimes not):
    cols_to_drop = [c for c in cols_to_drop if c in df_lsoa.columns]
    df_lsoa = df_lsoa.drop(cols_to_drop, axis='columns')

    df_nearest_ivt = group_results_by_nearest_ivt(df_lsoa, df_unit_services)

    # # Drop extra columns that won't make sense when averaged.
    # cols_to_drop = ['transfer_required']
    # # Only keep cols that exist:
    # cols_to_drop = [c for c in cols_to_drop if c in df_lsoa.columns]
    # df_lsoa = df_lsoa.drop(cols_to_drop, axis='columns')

    df_icb = group_results_by_icb(df_lsoa)
    df_isdn = group_results_by_isdn(df_lsoa)
    df_ambo = group_results_by_ambo(df_lsoa)

    # Repeat for only patients who require a transfer:
    df_masked = df_lsoa.loc[df_lsoa['transfer_required'] == 1].copy()
    df_benefit_icb = group_results_by_icb(df_masked)
    df_benefit_isdn = group_results_by_isdn(df_masked)
    df_benefit_ambo = group_results_by_ambo(df_masked)
    df_benefit_nearest_ivt = group_results_by_nearest_ivt(df_masked, df_unit_services)

    return df_icb, df_isdn, df_nearest_ivt, df_ambo, df_benefit_icb, df_benefit_isdn, df_benefit_nearest_ivt, df_benefit_ambo


def group_results_by_icb(df_lsoa):
    # Glob results by ICB:
    df_icb = df_lsoa.copy()
    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with
    # some object columns)
    df_icb = df_icb.drop([
        'nearest_ivt_unit',
        'region',
        'region_type',
        'region_code',
        'icb_code',
        'isdn',
        'ambo22',
        ], axis='columns')
    # Average:
    df_icb = df_icb.groupby('icb').mean()

    # Round the values.
    # Outcomes:
    cols_outcome = [c for c in df_icb.columns if (
        (c.endswith('utility_shift')) |
        (c.endswith('mrs_0-2')) | (c.endswith('mrs_shift'))
        )]
    df_icb[cols_outcome] = np.round(df_icb[cols_outcome], 3)

    # Times:
    cols_time = [c for c in df_icb.columns if 'time' in c]
    df_icb[cols_time] = np.round(df_icb[cols_time], 2)
    return df_icb.copy()


def group_results_by_isdn(df_lsoa):
    # Glob results by ISDN:
    df_isdn = df_lsoa.copy()
    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with
    # some object columns)
    df_isdn = df_isdn.drop([
        'nearest_ivt_unit',
        'region',
        'region_type',
        'region_code',
        'icb',
        'icb_code',
        'ambo22',
        ], axis='columns')
    # Average:
    df_isdn = df_isdn.groupby('isdn').mean()

    # Round the values.
    # Outcomes:
    cols_outcome = [c for c in df_isdn.columns if (
        (c.endswith('utility_shift')) |
        (c.endswith('mrs_0-2')) | (c.endswith('mrs_shift'))
        )]
    df_isdn[cols_outcome] = np.round(df_isdn[cols_outcome], 3)

    # Times:
    cols_time = [c for c in df_isdn.columns if 'time' in c]
    df_isdn[cols_time] = np.round(df_isdn[cols_time], 2)
    return df_isdn.copy()


def group_results_by_ambo(df_lsoa):
    # Glob results by ISDN:
    df_ambo = df_lsoa.copy()
    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with
    # some object columns)
    df_ambo = df_ambo.drop([
        'nearest_ivt_unit',
        'region',
        'region_type',
        'region_code',
        'icb',
        'icb_code',
        'isdn',
        ], axis='columns')
    # Average:
    df_ambo = df_ambo.groupby('ambo22').mean()

    # Round the values.
    # Outcomes:
    cols_outcome = [c for c in df_ambo.columns if (
        (c.endswith('utility_shift')) |
        (c.endswith('mrs_0-2')) | (c.endswith('mrs_shift'))
        )]
    df_ambo[cols_outcome] = np.round(df_ambo[cols_outcome], 3)

    # Times:
    cols_time = [c for c in df_ambo.columns if 'time' in c]
    df_ambo[cols_time] = np.round(df_ambo[cols_time], 2)
    return df_ambo.copy()


def group_results_by_nearest_ivt(df_lsoa, df_unit_services):
    # Glob results by nearest IVT unit:
    df_nearest_ivt = df_lsoa.copy()
    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with
    # some object columns)
    df_nearest_ivt = df_nearest_ivt.drop([
        'region',
        'region_type',
        'region_code',
        'icb',
        'icb_code',
        'isdn',
        'ambo22',
        ], axis='columns')
    # Average:
    df_nearest_ivt = df_nearest_ivt.groupby('nearest_ivt_unit').mean()
    # Merge back in the unit names:
    df_nearest_ivt = pd.merge(
        df_unit_services['ssnap_name'],
        df_nearest_ivt, how='right', left_on='Postcode', right_index=True)

    # Round the values.
    # Outcomes:
    cols_outcome = [c for c in df_nearest_ivt.columns if (
        (c.endswith('utility_shift')) |
        (c.endswith('mrs_0-2')) | (c.endswith('mrs_shift'))
        )]
    df_nearest_ivt[cols_outcome] = np.round(df_nearest_ivt[cols_outcome], 3)

    # Times:
    cols_time = [c for c in df_nearest_ivt.columns if 'time' in c]
    df_nearest_ivt[cols_time] = np.round(df_nearest_ivt[cols_time], 2)
    return df_nearest_ivt.copy()


# #############################
# ##### mRS DISTRIBUTIONS #####
# #############################


# ###########################
# ##### AVERAGE RESULTS #####
# ###########################

def combine_results_by_diff(
        df_lsoa,
        scenario_types,
        combine_mrs_dists=False
        ):
    df1 = pd.DataFrame(index=df_lsoa.index)
    df2 = pd.DataFrame(index=df_lsoa.index)
    # Column names for these new DataFrames:
    cols_combo = []
    cols_scen1 = []
    cols_scen2 = []

    # scenario_types = ['redirect', 'drip_ship']
    occlusion_types = ['nlvo', 'lvo', 'combo']
    treatment_types = ['ivt', 'mt', 'ivt_mt']
    if combine_mrs_dists:
        outcome_types = ['mrs_dists_noncum']  # not cumulative
    else:
        outcome_types = ['mrs_0-2', 'mrs_shift', 'utility_shift']

    oto_combos = [(occ, tre, out) for out in outcome_types
                  for tre in treatment_types for occ in occlusion_types]
    for (occ, tre, out) in oto_combos:
        if combine_mrs_dists:
            cols_mrs_scen1 = [
                f'{scenario_types[0]}_{occ}_{tre}_{out}_{i}'
                for i in range(7)]
            cols_mrs_scen2 = [
                f'{scenario_types[1]}_{occ}_{tre}_{out}_{i}'
                for i in range(7)]
            try:
                data_nlvo = df_lsoa[cols_mrs_scen1]
                data_exists = True
            except KeyError:
                data_exists = False

            if data_exists:
                cols_combo += [f'{occ}_{tre}_{out}_{i}' for i in range(7)]
                cols_scen1 += cols_mrs_scen1
                cols_scen2 += cols_mrs_scen2
        else:
            # Existing column names:
            col_scen1 = f'{scenario_types[0]}_{occ}_{tre}_{out}'
            col_scen2 = f'{scenario_types[1]}_{occ}_{tre}_{out}'
            # New column name for the diff data:
            col_diff = f'{occ}_{tre}_{out}'
            try:
                data_scen1 = df_lsoa[col_scen1]
                data_scen2 = df_lsoa[col_scen2]
                data_exists = True
            except KeyError:
                # This combination doesn't exist
                # (e.g. nLVO with MT).
                data_exists = False

            if data_exists:
                cols_combo.append(col_diff)
                cols_scen1.append(col_scen1)
                cols_scen2.append(col_scen2)
            else:
                pass

    # Pick out the data from the original dataframe:
    df1 = df_lsoa[cols_scen1].copy()
    df2 = df_lsoa[cols_scen2].copy()
    # Rename columns so they match:
    df1.columns = cols_combo
    df2.columns = cols_combo

    # Create new dataframe from combining the two separate ones:
    combo_data = df1 - df2

    if combine_mrs_dists:
        # Make cumulative probabilities:
        prefixes = sorted(list(set(
            ['_'.join(c.split('_')[:-1]) for c in cols_combo])))
        for col in prefixes:
            cols_here = [f'{col}_{i}' for i in range(7)]
            col_cumsum = col.split('_noncum')[0]
            cols_cumsum_here = [f'{col_cumsum}_{i}' for i in range(7)]
            # Cumulative probability:
            cumulatives = np.cumsum(combo_data[cols_here], axis=1)
            combo_data[cols_cumsum_here] = cumulatives

    # Round the values:
    dp = 3
    combo_data[combo_data.columns] = np.round(
        combo_data[combo_data.columns], dp)

    # Update column names to mark them as combined:
    combo_cols = [
        ''.join([
            # f"{col.split('_')[0]}_",
            f'diff_{scenario_types[0]}_minus_{scenario_types[1]}_',
            f"{'_'.join(col.split('_'))}"
            ])
        for col in combo_data.columns]
    combo_data.columns = combo_cols

    # Merge this new data into the starting dataframe:
    df_lsoa = pd.merge(df_lsoa, combo_data,
                       left_index=True, right_index=True)

    return df_lsoa.copy()


def replace_str_in_list(l, str_old, str_new):
    ind_col = l.index(str_old)
    l.remove(str_old)
    l.insert(ind_col, str_new)
    return l


def build_columns_combine_occlusions(
        treatment_list=['ivt', 'mt', 'ivt_mt'],
        outcome_list=['mrs_0-2', 'mrs_shift', 'utility_shift'],
        scenario_list=[],  # ['drip_ship', 'mothership', 'redirect'],
        dummy_col='zero',
        ):
    """
    All occlusion types in separate lists.
    """
    # Generate all combos of these lists for nLVO:
    cols_nlvo = [f'nlvo_{t}_{o}' for t in treatment_list
                 for o in outcome_list]
    if len(scenario_list) > 0:
        cols_nlvo = [f'{s}_{col}' for col in cols_nlvo for s in scenario_list]
    # Make versions for LVO and combo columns:
    cols_lvo = [c.replace('nlvo', 'lvo') for c in cols_nlvo]
    cols_combo = [c.replace('nlvo', 'combo') for c in cols_nlvo]

    # Update invalid columns.
    # For nLVO with MT, point to data with no change from no treatment:
    cols_nlvo_mt = [c for c in cols_nlvo if (('mt' in c) & ('ivt' not in c))]
    for col in cols_nlvo_mt:
        if 'shift' in col:
            # Point to dummy zero data instead, no change:
            cols_nlvo = replace_str_in_list(cols_nlvo, col, dummy_col)
        else:
            # Point to no-treatment data instead:
            o = np.array(outcome_list)[[(o in col) for o in outcome_list]][0]
            col_nlvo = f'nlvo_no_treatment_{o}'  # Ignore scenario
            cols_nlvo = replace_str_in_list(cols_nlvo, col, col_nlvo)
    # For nLVO with IVT&MT, point to IVT-only data:
    cols_nlvo_ivt_mt = [c for c in cols_nlvo if ('ivt_mt' in c)]
    for col in cols_nlvo_ivt_mt:
        col_nlvo = col.replace('ivt_mt', 'ivt')
        cols_nlvo = replace_str_in_list(cols_nlvo, col, col_nlvo)
    return cols_nlvo, cols_lvo, cols_combo


def build_columns_combine_occlusions_mrs(
        treatment_list=['ivt', 'mt', 'ivt_mt'],
        outcome_list=['mrs_dists_noncum'],
        scenario_list=[],
        dummy_col='nlvo_no_treatment_noncum',
        ):
    # Generate all combos of these lists for nLVO:
    cols_nlvo = [
        f'nlvo_{t}_{o}_{m}'
        for t in treatment_list
        for o in outcome_list
        for m in range(7)  # mRS score
        ]
    if len(scenario_list) > 0:
        cols_nlvo = [f'{s}_{col}' for col in cols_nlvo for s in scenario_list]

    # Make versions for LVO and combo columns:
    cols_lvo = [c.replace('nlvo', 'lvo') for c in cols_nlvo]
    cols_combo = [c.replace('nlvo', 'combo') for c in cols_nlvo]

    # Update invalid columns.
    # For nLVO with MT, point to data with no change from no treatment:
    cols_nlvo_mt = [c for c in cols_nlvo if (('mt' in c) & ('ivt' not in c))]
    for col in cols_nlvo_mt:
        # Point to no-treatment data instead:
        m = col.split('_')[-1]  # mRS score
        cols_nlvo = replace_str_in_list(cols_nlvo, col, f'{dummy_col}_{m}')
    # For nLVO with IVT&MT, point to IVT-only data:
    cols_nlvo_ivt_mt = [c for c in cols_nlvo if ('ivt_mt' in c)]
    for col in cols_nlvo_ivt_mt:
        col_nlvo = col.replace('ivt_mt', 'ivt')
        cols_nlvo = replace_str_in_list(cols_nlvo, col, col_nlvo)
    return cols_nlvo, cols_lvo, cols_combo


def build_columns_combine_redir(
        treatment_list=['ivt', 'mt', 'ivt_mt'],
        outcome_list=['mrs_0-2', 'mrs_shift', 'utility_shift'],
        occlusion_list=['nlvo', 'lvo'],
        scenario_list=['redirection_rejected', 'redirection_approved'],
        combo_name='redirection_considered',
        dummy_col='zero',
        ):
    """
    All occlusion types in one list.
    """
    # Generate all combos of these lists for redir reject:
    cols_rr = [f'{scenario_list[0]}_{v}_{t}_{o}' for v in occlusion_list
               for t in treatment_list for o in outcome_list]

    # # Update invalid columns.
    # # For nLVO with MT, point to data with no change from no treatment:
    # cols_nlvo_mt = [c for c in cols_rr if (('nlvo' in c) & ('mt' in c) & ('ivt' not in c))]
    # for col in cols_nlvo_mt:
    #     if 'shift' in col:
    #         # Point to dummy zero data instead, no change:
    #         cols_rr = replace_str_in_list(cols_rr, col, dummy_col)
    #     else:
    #         # Point to no-treatment data instead:
    #         o = np.array(treatment_list)[[(t in col) for t in treatment_list]]
    #         col_nlvo = f'nlvo_no_treatment_{o}'  # Ignore scenario
    #         cols_rr = replace_str_in_list(cols_rr, col, col_nlvo)
    # # For nLVO with IVT&MT, point to IVT-only data:
    # cols_nlvo_ivt_mt = [c for c in cols_rr if (('nlvo' in c) & ('ivt_mt' in c))]
    # for col in cols_nlvo_ivt_mt:
    #     col_nlvo = col.replace('ivt_mt', 'ivt')
    #     ind_col = cols_rr.index(col)
    #     cols_rr.remove(col)
    #     cols_rr.insert(ind_col, col_nlvo)

    # Make versions for redir approve:
    cols_ra = [c.replace(scenario_list[0], scenario_list[1])
               for c in cols_rr]
    # Make versions for redir considered:
    cols_combo = [c.replace(scenario_list[0], combo_name)
                  for c in cols_rr]

    return cols_rr, cols_ra, cols_combo


def keep_existing_cols(df, cols_1, cols_2, cols_combo):

    # Remove sets of columns where any data is missing:
    exists_1 = [(c in df) for c in cols_1]
    exists_2 = [(c in df) for c in cols_2]
    exists_both = np.array(exists_1) & np.array(exists_2)

    cols_1 = list(np.array(cols_1)[exists_both])
    cols_2 = list(np.array(cols_2)[exists_both])
    cols_combo = list(np.array(cols_combo)[exists_both])
    return cols_1, cols_2, cols_combo


def combine_results(
        df_lsoa,
        cols_1,
        cols_2,
        cols_combo,
        props_list,
        round_dp=3
        ):
    """
    Make two new dataframes, one with all the column 1 data
    and one with all the column 2 data, and with the same column
    names. Then subtract one dataframe from the other
    and merge the results back into the main one.
    This is more efficient than calculating and creating
    each new column individually.
    """
    # Pick out the data from the original dataframe and rename columns
    # so they match in both dataframes.
    # Don't rename using a dict because sometimes multiple columns
    # are defined using the same dummy column, and the rename
    # will give all of those the same final name.
    # Pick out the data from the original dataframe:
    df1 = df_lsoa[cols_1].copy()
    df2 = df_lsoa[cols_2].copy()
    # Rename columns so they match:
    df1.columns = cols_combo
    df2.columns = cols_combo

    # Simple addition: x% of column 1 plus y% of column 2.
    # Create new dataframe from combining the two separate ones:
    combo_data = df1 * props_list[0] + df2 * props_list[1]

    # Round the values:
    combo_data[combo_data.columns] = np.round(
        combo_data[combo_data.columns], round_dp)

    # Merge this new data into the starting dataframe:
    df_lsoa = pd.merge(df_lsoa, combo_data, left_index=True, right_index=True)

    return df_lsoa


def load_or_calculate_region_outlines(
        outline_name,
        df_lsoa,
        col_lhs='nearest_ivt_unit_name',
        col_rhs='nearest_mt_unit_name',
        ):
    """
    Don't replace these outlines with stroke-maps!
    These versions match the simplified LSOA shapes.
    """
    # Load in another gdf:

    if outline_name == 'ISDN':
        load_gdf_catchment = True
        outline_file = './data/outline_isdns.geojson'
        outline_names_col = 'isdn'
    elif outline_name == 'ICB':
        load_gdf_catchment = True
        outline_file = './data/outline_icbs.geojson'
        outline_names_col = 'icb'  # to display
    elif outline_name == 'Ambulance service':
        load_gdf_catchment = True
        outline_file = './data/outline_ambo22s.geojson'
        outline_names_col = 'ambo22'  # to display
    elif outline_name == 'Nearest service':
        load_gdf_catchment = False
        outline_names_col = 'Nearest service'

        # Make catchment area polygons:
        gdf_catchment_lhs = dissolve_polygons_by_value(
            df_lsoa.copy().reset_index()[['lsoa', col_lhs]],
            col=col_lhs,
            load_msoa=True
            )
        gdf_catchment_lhs = gdf_catchment_lhs.rename(
            columns={col_lhs: 'Nearest service'})

        gdf_catchment_rhs = dissolve_polygons_by_value(
            df_lsoa.copy().reset_index()[['lsoa', col_rhs]],
            col=col_rhs,
            load_msoa=True
            )
        gdf_catchment_rhs = gdf_catchment_rhs.rename(
            columns={col_rhs: 'Nearest service'})

    if load_gdf_catchment:
        gdf_catchment_lhs = geopandas.read_file(outline_file)
        # Convert to British National Grid:
        gdf_catchment_lhs = gdf_catchment_lhs.to_crs('EPSG:27700')
        # st.write(gdf_catchment['geometry'])
        # # Make geometry valid:
        # gdf_catchment['geometry'] = [
        #     make_valid(g) if g is not None else g
        #     for g in gdf_catchment['geometry'].values
        #     ]
        gdf_catchment_rhs = gdf_catchment_lhs.copy()

    # Make colour transparent:
    gdf_catchment_lhs['colour'] = 'rgba(0, 0, 0, 0)'
    gdf_catchment_rhs['colour'] = 'rgba(0, 0, 0, 0)'
    # Make a dummy column for the legend entry:
    gdf_catchment_lhs['outline_type'] = outline_name
    gdf_catchment_rhs['outline_type'] = outline_name

    gdf_catchment_pop = gdf_catchment_lhs.copy()
    return (
        outline_names_col,
        gdf_catchment_lhs,
        gdf_catchment_rhs,
        gdf_catchment_pop
    )
