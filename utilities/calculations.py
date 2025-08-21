
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
from utilities.utils import load_reference_mrs_dists
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


# @st.cache_data
def calculate_outcomes(dict_outcome_inputs, df_unit_services, geodata):
    """

    # Run the outcomes with the selected pathway:
    """
    # Set up model
    model = Model(dict_outcome_inputs, geodata)
    # Run model
    model.run()

    # df_lsoa, df_mrs = model.run(geodata, dict_outcome_inputs)

    # # TO DO - merge in the geographical data to the outcome results.

    df_lsoa = model.get_full_results().copy(deep=True)
    df_lsoa.index.names = ['lsoa']
    df_lsoa.columns.names = ['property']

    df_mrs = model.get_full_mrs_dists().copy(deep=True)
    df_mrs.index.names = ['lsoa']
    df_mrs.columns.names = ['property']

    del model

    # Copy stroke unit names over. Currently has only postcodes.
    cols_postcode = ['nearest_ivt_unit', 'nearest_mt_unit',
                     'transfer_unit', 'nearest_msu_unit']
    for col in cols_postcode:
        if col in df_lsoa.columns:
            df_lsoa = pd.merge(
                df_lsoa, df_unit_services['ssnap_name'],
                left_on=col, right_index=True, how='left'
                )
            df_lsoa = df_lsoa.rename(columns={'ssnap_name': f'{col}_name'})
            # Reorder columns so name appears next to postcode.
            i = df_lsoa.columns.tolist().index(col)
            df_lsoa = df_lsoa[
                [*df_lsoa.columns[:i], f'{col}_name', *df_lsoa.columns[i:-1]]]

    # Outcome columns and numbers of decimal places:
    dict_outcomes_dp = {
        'mrs_0-2': 3,
        'mrs_shift': 3,
        # 'utility': 3,
        'utility_shift': 3,
    }
    for suff, dp in dict_outcomes_dp.items():
        cols = [c for c in df_lsoa.columns if c.endswith(suff)]
        df_lsoa[cols] = np.round(df_lsoa[cols], dp)


    return df_lsoa.copy(), df_mrs.copy()
    # return 'plop', 'cats'
    # return df_mrs.copy()
    # return df_lsoa.copy()


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
def run_outcome_model_for_unique_times(times_to_ivt, times_to_mt):
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
    outcomes_by_stroke_type_ivt_only = model.run_outcome_model(
        outcome_inputs_df_ivt_only).copy()
    outcomes_by_stroke_type_mt_only = model.run_outcome_model(
        outcome_inputs_df_mt_only).copy()
    return (outcomes_by_stroke_type_ivt_only,
            outcomes_by_stroke_type_mt_only)


def convert_outcome_dicts_to_df_outcomes(
        times_to_ivt,
        times_to_mt,
        outcomes_by_stroke_type_ivt_only,
        outcomes_by_stroke_type_mt_only
        ):
    # Gather outcomes for these times:
    keys_for_df_ivt = [
        'nlvo_ivt_each_patient_mrs_shift',
        'nlvo_ivt_each_patient_utility_shift',
        'nlvo_ivt_each_patient_mrs_0-2',
        'lvo_ivt_each_patient_mrs_shift',
        'lvo_ivt_each_patient_utility_shift',
        'lvo_ivt_each_patient_mrs_0-2',
    ]
    df_outcomes_ivt = pd.DataFrame(
        [times_to_ivt] + [outcomes_by_stroke_type_ivt_only[k] for k in keys_for_df_ivt],
        index=['time_to_ivt'] + keys_for_df_ivt
    ).transpose()

    keys_for_df_mt = [
        'lvo_mt_each_patient_mrs_shift',
        'lvo_mt_each_patient_utility_shift',
        'lvo_mt_each_patient_mrs_0-2',
    ]
    df_outcomes_mt = pd.DataFrame(
        [times_to_mt] + [outcomes_by_stroke_type_mt_only[k] for k in keys_for_df_mt],
        index=['time_to_mt'] + keys_for_df_mt
    ).transpose()

    # Outcome columns and numbers of decimal places:
    dict_outcomes_dp = {
        'mrs_0-2': 3,
        'mrs_shift': 3,
        # 'utility': 3,
        'utility_shift': 3,
    }
    for suff, dp in dict_outcomes_dp.items():
        for df in [df_outcomes_mt, df_outcomes_ivt]:
            cols = [c for c in df.columns if c.endswith(suff)]
            df[cols] = np.round(df[cols], dp)

    # Remove "each patient" from column names:
    df_outcomes_ivt = df_outcomes_ivt.rename(columns=dict(zip(
        df_outcomes_ivt.columns, [c.replace('_each_patient', '') for c in df_outcomes_ivt.columns]
    )))
    df_outcomes_mt = df_outcomes_mt.rename(columns=dict(zip(
        df_outcomes_mt.columns, [c.replace('_each_patient', '') for c in df_outcomes_mt.columns]
    )))
    keys_for_df_ivt = [k.replace('_each_patient', '') for k in keys_for_df_ivt]
    keys_for_df_mt = [k.replace('_each_patient', '') for k in keys_for_df_mt]

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


def make_outcome_inputs_usual_care(
        pathway_dict,
        df_travel_times,
        time_to_ivt_without_travel=-1000.0,
        time_to_mt_with_transfer_without_travel=-1000.0,
        time_to_mt_without_transfer_without_travel=-1000.0,
        ):
    # If any of the treatment times are below zero, assume
    # they need to be calculated afresh from the dict.
    recalculate_times = (
        (time_to_ivt_without_travel < 0.0) |
        (time_to_mt_with_transfer_without_travel < 0.0) |
        (time_to_mt_without_transfer_without_travel < 0.0)
        )
    if recalculate_times:
        time_dict = calculate_times_to_treatment_without_travel_usual_care(
            pathway_dict)
        time_to_ivt_without_travel = time_dict['usual_care_time_to_ivt']
        time_to_mt_with_transfer_without_travel = (
            time_dict['usual_care_mt_transfer'])
        time_to_mt_without_transfer_without_travel = (
            time_dict['usual_care_mt_no_transfer'])
    # Time to IVT:
    time_to_ivt = (
        time_to_ivt_without_travel +
        df_travel_times['nearest_ivt_time'].values
        )

    # Separate MT timings required depending on whether transfer
    # needed. Mask for which rows of data need a transfer:
    mask_transfer = np.where(df_travel_times['transfer_required'].values)
    # Timings for units needing transfers:
    mt_transfer = (
        time_to_mt_with_transfer_without_travel +
        df_travel_times['nearest_ivt_time'].values +
        df_travel_times['transfer_time'].values
        )
    # Timings for units that do not need transfers:
    mt_no_transfer = (
        time_to_mt_without_transfer_without_travel +
        df_travel_times['nearest_ivt_time'].values
        )
    # Combine the two sets of MT times:
    time_to_mt = mt_no_transfer
    time_to_mt[mask_transfer] = mt_transfer[mask_transfer]

    # Set up input table for stroke outcome package.
    outcome_inputs_df = pd.DataFrame()
    # Provide a dummy stroke type code for now -
    # it will be overwritten when the outcomes are calculated so that
    # both nLVO and LVO results are calculated from one scenario.
    outcome_inputs_df['stroke_type_code'] = np.repeat(1, len(df_travel_times))
    # Set everyone to receive both treatment types.
    outcome_inputs_df['ivt_chosen_bool'] = 1
    outcome_inputs_df['mt_chosen_bool'] = 1
    # And include the times we've just calculated:
    outcome_inputs_df['onset_to_needle_mins'] = time_to_ivt
    outcome_inputs_df['onset_to_puncture_mins'] = time_to_mt
    # Also store LSOA name - not needed by outcome model,
    # but useful later for matching these results to their LSOA.
    outcome_inputs_df['LSOA'] = df_travel_times.index

    return outcome_inputs_df.copy()


def make_outcome_inputs_redirection_rejected(pathway_dict, df_travel_times):
    """
    These timings are the same as for usual care but with the extra
    time for diagnostic.
    """
    # Time to IVT:
    time_to_ivt = (
        pathway_dict['process_time_call_ambulance'] +
        pathway_dict['process_time_ambulance_response'] +
        pathway_dict['process_ambulance_on_scene_duration'] +
        pathway_dict['process_ambulance_on_scene_diagnostic_duration'] +
        df_travel_times['nearest_ivt_time'].values +
        pathway_dict['process_time_arrival_to_needle']
        )

    # Separate MT timings required depending on whether transfer
    # needed. Mask for which rows of data need a transfer:
    mask_transfer = np.where(df_travel_times['transfer_required'].values)
    # Timings for units needing transfers:
    mt_transfer = (
        pathway_dict['process_time_call_ambulance'] +
        pathway_dict['process_time_ambulance_response'] +
        pathway_dict['process_ambulance_on_scene_duration'] +
        pathway_dict['process_ambulance_on_scene_diagnostic_duration'] +
        df_travel_times['nearest_ivt_time'].values +
        pathway_dict['transfer_time_delay'] +
        df_travel_times['transfer_time'].values +
        pathway_dict['process_time_transfer_arrival_to_puncture']
        )
    # Timings for units that do not need transfers:
    mt_no_transfer = (
        pathway_dict['process_time_call_ambulance'] +
        pathway_dict['process_time_ambulance_response'] +
        pathway_dict['process_ambulance_on_scene_duration'] +
        pathway_dict['process_ambulance_on_scene_diagnostic_duration'] +
        df_travel_times['nearest_ivt_time'].values +
        pathway_dict['process_time_arrival_to_puncture']
        )
    # Combine the two sets of MT times:
    time_to_mt = mt_no_transfer
    time_to_mt[mask_transfer] = mt_transfer[mask_transfer]

    # Set up input table for stroke outcome package.
    outcome_inputs_df = pd.DataFrame()
    # Provide a dummy stroke type code for now -
    # it will be overwritten when the outcomes are calculated so that
    # both nLVO and LVO results are calculated from one scenario.
    outcome_inputs_df['stroke_type_code'] = np.repeat(1, len(df_travel_times))
    # Set everyone to receive both treatment types.
    outcome_inputs_df['ivt_chosen_bool'] = 1
    outcome_inputs_df['mt_chosen_bool'] = 1
    # And include the times we've just calculated:
    outcome_inputs_df['onset_to_needle_mins'] = time_to_ivt
    outcome_inputs_df['onset_to_puncture_mins'] = time_to_mt
    # Also store LSOA name - not needed by outcome model,
    # but useful later for matching these results to their LSOA.
    outcome_inputs_df['LSOA'] = df_travel_times.index

    return outcome_inputs_df


def make_outcome_inputs_redirection_approved(pathway_dict, df_travel_times):
    """
    """
    # Time to IVT:
    time_to_ivt = (
        pathway_dict['process_time_call_ambulance'] +
        pathway_dict['process_time_ambulance_response'] +
        pathway_dict['process_ambulance_on_scene_duration'] +
        pathway_dict['process_ambulance_on_scene_diagnostic_duration'] +
        df_travel_times['nearest_mt_time'].values +
        pathway_dict['process_time_arrival_to_needle']
        )

    # Time to MT: this time nobody has a transfer.
    time_to_mt = (
        pathway_dict['process_time_call_ambulance'] +
        pathway_dict['process_time_ambulance_response'] +
        pathway_dict['process_ambulance_on_scene_duration'] +
        pathway_dict['process_ambulance_on_scene_diagnostic_duration'] +
        df_travel_times['nearest_mt_time'].values +
        pathway_dict['process_time_arrival_to_puncture']
        )

    # Set up input table for stroke outcome package.
    outcome_inputs_df = pd.DataFrame()
    # Provide a dummy stroke type code for now -
    # it will be overwritten when the outcomes are calculated so that
    # both nLVO and LVO results are calculated from one scenario.
    outcome_inputs_df['stroke_type_code'] = np.repeat(1, len(df_travel_times))
    # Set everyone to receive both treatment types.
    outcome_inputs_df['ivt_chosen_bool'] = 1
    outcome_inputs_df['mt_chosen_bool'] = 1
    # And include the times we've just calculated:
    outcome_inputs_df['onset_to_needle_mins'] = time_to_ivt
    outcome_inputs_df['onset_to_puncture_mins'] = time_to_mt
    # Also store LSOA name - not needed by outcome model,
    # but useful later for matching these results to their LSOA.
    outcome_inputs_df['LSOA'] = df_travel_times.index

    return outcome_inputs_df


def make_outcome_inputs_msu(
        pathway_dict,
        df_travel_times,
        time_to_ivt_without_travel=-1000.0,
        time_to_mt_without_travel=-1000.0,
        msu_gives_ivt=True
        ):
    # If any of the treatment times are below zero, assume
    # they need to be calculated afresh from the dict.
    recalculate_times = (
        (time_to_ivt_without_travel < 0.0) |
        (time_to_mt_without_travel < 0.0)
        )
    if recalculate_times:
        time_dict = calculate_times_to_treatment_without_travel_msu(
            pathway_dict)
        time_to_ivt_without_travel = time_dict['msu_time_to_ivt']
        time_to_mt_with_ivt_without_travel = time_dict['msu_time_to_mt']
        time_to_mt_without_ivt_without_travel = (
            time_dict['msu_time_to_mt_no_ivt'])
        if msu_gives_ivt:
            time_to_mt_without_travel = time_to_mt_with_ivt_without_travel
        else:
            time_to_mt_without_travel = time_to_mt_without_ivt_without_travel

    # To shorten lines:
    s = pathway_dict['scale_msu_travel_times']

    # Time to IVT:
    time_to_ivt = (
        time_to_ivt_without_travel +
        (df_travel_times['nearest_msu_time'].values * s)
        )

    # Time to MT:
    # If required, everyone goes directly to the nearest MT unit.
    time_to_mt = (
        time_to_mt_without_travel +
        (df_travel_times['nearest_msu_time'].values * s) +
        (df_travel_times['nearest_mt_time'].values * s)
        )

    # Bonus times - not needed for outcome model but calculated anyway.
    msu_occupied_treatment = (
        pathway_dict['process_msu_dispatch'] +
        (df_travel_times['nearest_msu_time'].values * s) +
        pathway_dict['process_msu_thrombolysis'] +
        pathway_dict['process_msu_on_scene_post_thrombolysis'] +
        (df_travel_times['nearest_mt_time'].values * s)
        )

    msu_occupied_no_treatment = (
        pathway_dict['process_msu_dispatch'] +
        (df_travel_times['nearest_msu_time'].values * s) +
        pathway_dict['process_msu_on_scene_no_thrombolysis'] +
        (df_travel_times['nearest_mt_time'].values * s)
        )
    # Where does the MSU go after leaving the scene?
    # Always to MT unit? But patients might not need MT...
    # Does return it to base though.

    # Set up input table for stroke outcome package.
    outcome_inputs_df = pd.DataFrame()
    # Provide a dummy stroke type code for now -
    # it will be overwritten when the outcomes are calculated so that
    # both nLVO and LVO results are calculated from one scenario.
    outcome_inputs_df['stroke_type_code'] = np.repeat(1, len(df_travel_times))
    # Set everyone to receive both treatment types.
    outcome_inputs_df['ivt_chosen_bool'] = 1
    outcome_inputs_df['mt_chosen_bool'] = 1
    # And include the times we've just calculated:
    outcome_inputs_df['onset_to_needle_mins'] = time_to_ivt
    outcome_inputs_df['onset_to_puncture_mins'] = time_to_mt
    # Also store LSOA name - not needed by outcome model,
    # but useful later for matching these results to their LSOA.
    outcome_inputs_df['LSOA'] = df_travel_times.index
    # Also store bonus times:
    outcome_inputs_df['msu_occupied_ivt'] = msu_occupied_treatment
    outcome_inputs_df['msu_occupied_no_ivt'] = msu_occupied_no_treatment

    return outcome_inputs_df.copy()

# ####################################
# ##### GATHER FULL LSOA RESULTS #####
# ####################################

def build_full_lsoa_outcomes_from_unique_time_results(
            df_travel_times,
            df_outcomes_ivt,
            df_outcomes_mt,
            df_mrs_ivt,
            df_mrs_mt,
    ):
    # Apply these results to the treatment times in each scenario.
    df_lsoa = df_travel_times.copy().reset_index()


    # Usual care:
    # IVT:
    df_lsoa = pd.merge(
        df_lsoa, df_outcomes_ivt,
        left_on='usual_care_ivt', right_on='time_to_ivt', how='left'
        )
    keys_for_df_ivt = [k for k in df_outcomes_ivt.columns if 'time' not in k]
    keys_usual_care_from_df_ivt = [f'usual_care_{k}' for k in keys_for_df_ivt]
    df_lsoa = df_lsoa.drop('time_to_ivt', axis='columns')
    df_lsoa = df_lsoa.rename(columns=dict(
        zip(keys_for_df_ivt, keys_usual_care_from_df_ivt)))
    # MT:
    df_lsoa = pd.merge(
        df_lsoa, df_outcomes_mt,
        left_on='usual_care_mt', right_on='time_to_mt', how='left'
        )
    keys_for_df_mt = [k for k in df_outcomes_mt.columns if 'time' not in k]
    keys_usual_care_from_df_mt = [f'usual_care_{k}' for k in keys_for_df_mt]
    df_lsoa = df_lsoa.drop('time_to_mt', axis='columns')
    df_lsoa = df_lsoa.rename(columns=dict(
        zip(keys_for_df_mt, keys_usual_care_from_df_mt)))
    # IVT & MT:
    cols_lvo_ivt_mt = [
        'lvo_ivt_mt_mrs_shift',
        'lvo_ivt_mt_utility_shift',
        'lvo_ivt_mt_mrs_0-2',
    ]
    cols_lvo_ivt = [
        'lvo_ivt_mrs_shift',
        'lvo_ivt_utility_shift',
        'lvo_ivt_mrs_0-2',
    ]
    cols_lvo_mt = [
        'lvo_mt_mrs_shift',
        'lvo_mt_utility_shift',
        'lvo_mt_mrs_0-2'
    ]
    cols_usual_care_lvo_ivt_mt = [f'usual_care_{k}' for k in cols_lvo_ivt_mt]
    cols_usual_care_lvo_ivt = [f'usual_care_{k}' for k in cols_lvo_ivt]
    cols_usual_care_lvo_mt = [f'usual_care_{k}' for k in cols_lvo_mt]
    # Initially copy over the MT data:
    df_lsoa[cols_usual_care_lvo_ivt_mt] = df_lsoa[cols_usual_care_lvo_mt].copy()
    # Find whether IVT is better than MT:
    mask_usual_care_ivt_better = (
        df_lsoa['usual_care_lvo_mt_mrs_0-2'] <
        df_lsoa['usual_care_lvo_ivt_mrs_0-2']
    )
    # Store this mask for later use of picking out mRS dists:
    df_lsoa['usual_care_lvo_ivt_better_than_mt'] = mask_usual_care_ivt_better
    # Update outcomes for those patients:
    # (keep the .values because the column names don't match)
    df_lsoa.loc[mask_usual_care_ivt_better, cols_usual_care_lvo_ivt_mt] = (
        df_lsoa.loc[mask_usual_care_ivt_better, cols_usual_care_lvo_ivt].values)

    # Copy over probability of death:
    # IVT:
    df_lsoa = pd.merge(
        df_lsoa, df_mrs_ivt[
            ['time_to_ivt', 'nlvo_ivt_mrs_dists_noncum_6', 'lvo_ivt_mrs_dists_noncum_6']],
        left_on='usual_care_ivt', right_on='time_to_ivt', how='left'
        )
    df_lsoa = df_lsoa.drop('time_to_ivt', axis='columns')
    df_lsoa = df_lsoa.rename(columns={
        'nlvo_ivt_mrs_dists_noncum_6': 'usual_care_probdeath_nlvo_ivt',
        'lvo_ivt_mrs_dists_noncum_6': 'usual_care_probdeath_lvo_ivt',
        })
    # MT:
    df_lsoa = pd.merge(
        df_lsoa, df_mrs_mt[['time_to_mt', 'lvo_mt_mrs_dists_noncum_6']],
        left_on='usual_care_mt', right_on='time_to_mt', how='left'
        )
    df_lsoa = df_lsoa.drop('time_to_mt', axis='columns')
    df_lsoa = df_lsoa.rename(columns={
        'lvo_mt_mrs_dists_noncum_6': 'usual_care_probdeath_lvo_mt'})
    # IVT & MT:
    # Initially copy over the MT data:
    df_lsoa['usual_care_probdeath_lvo_ivt_mt'] = (
        df_lsoa['usual_care_probdeath_lvo_mt'].copy())
    # Update values where IVT is better than MT:
    # (keep the .values because the column names don't match)
    df_lsoa.loc[mask_usual_care_ivt_better, 'usual_care_probdeath_lvo_ivt_mt'] = (
        df_lsoa.loc[mask_usual_care_ivt_better, 'usual_care_probdeath_lvo_ivt'].values)


    # MSU:
    # IVT:
    df_lsoa = pd.merge(
        df_lsoa, df_outcomes_ivt,
        left_on='msu_ivt', right_on='time_to_ivt', how='left'
        )
    keys_msu_from_df_ivt = [f'msu_{k}' for k in keys_for_df_ivt]
    df_lsoa = df_lsoa.drop('time_to_ivt', axis='columns')
    df_lsoa = df_lsoa.rename(columns=dict(
        zip(keys_for_df_ivt, keys_msu_from_df_ivt)))
    # MT only:
    df_lsoa = pd.merge(
        df_lsoa, df_outcomes_mt,
        left_on='msu_mt_no_ivt', right_on='time_to_mt', how='left'
        )
    keys_msu_from_df_mt = [f'msu_{k}' for k in keys_for_df_mt]
    df_lsoa = df_lsoa.drop('time_to_mt', axis='columns')
    df_lsoa = df_lsoa.rename(columns=dict(
        zip(keys_for_df_mt, keys_msu_from_df_mt)))
    # IVT & MT:
    # First copy over MT results:
    cols_msu_lvo_ivt_mt = [f'msu_{k}' for k in cols_lvo_ivt_mt]
    cols_msu_lvo_ivt = [f'msu_{k}' for k in cols_lvo_ivt]
    cols_msu_lvo_mt = [f'msu_{k}' for k in cols_lvo_mt]
    df_lsoa = pd.merge(
        df_lsoa, df_outcomes_mt,
        left_on='msu_mt_with_ivt', right_on='time_to_mt', how='left'
        )
    keys_msu_ivt_mt_from_df_mt = [f'msu_{k}'.replace('_mt_', '_ivt_mt_') for k in keys_for_df_mt]
    df_lsoa = df_lsoa.drop('time_to_mt', axis='columns')
    df_lsoa = df_lsoa.rename(columns=dict(
        zip(keys_for_df_mt, keys_msu_ivt_mt_from_df_mt)))
    # Find whether IVT is better than MT:
    mask_msu_ivt_better = (
        df_lsoa['msu_lvo_ivt_mt_mrs_0-2'] <
        df_lsoa['msu_lvo_ivt_mrs_0-2']
    )
    # Store this mask for later use of picking out mRS dists:
    df_lsoa['msu_lvo_ivt_better_than_mt'] = mask_msu_ivt_better
    # Update outcomes for those patients:
    # (keep the .values because the column names don't match)
    df_lsoa.loc[mask_msu_ivt_better, cols_msu_lvo_ivt_mt] = (
        df_lsoa.loc[mask_msu_ivt_better, cols_msu_lvo_ivt].values)

    # Copy over probability of death:
    # IVT:
    df_lsoa = pd.merge(
        df_lsoa, df_mrs_ivt[
            ['time_to_ivt', 'nlvo_ivt_mrs_dists_noncum_6', 'lvo_ivt_mrs_dists_noncum_6']],
        left_on='msu_ivt', right_on='time_to_ivt', how='left'
        )
    df_lsoa = df_lsoa.drop('time_to_ivt', axis='columns')
    df_lsoa = df_lsoa.rename(columns={
        'nlvo_ivt_mrs_dists_noncum_6': 'msu_probdeath_nlvo_ivt',
        'lvo_ivt_mrs_dists_noncum_6': 'msu_probdeath_lvo_ivt',
        })
    # MT:
    df_lsoa = pd.merge(
        df_lsoa, df_mrs_mt[['time_to_mt', 'lvo_mt_mrs_dists_noncum_6']],
        left_on='msu_mt_no_ivt', right_on='time_to_mt', how='left'
        )
    df_lsoa = df_lsoa.drop('time_to_mt', axis='columns')
    df_lsoa = df_lsoa.rename(columns={
        'lvo_mt_mrs_dists_noncum_6': 'msu_probdeath_lvo_mt'})
    # IVT & MT:
    # Initially copy over the MT data:
    df_lsoa['msu_probdeath_lvo_ivt_mt'] = (
        df_lsoa['msu_probdeath_lvo_mt'].copy())
    # Update values where IVT is better than MT:
    # (keep the .values because the column names don't match)
    df_lsoa.loc[mask_msu_ivt_better, 'msu_probdeath_lvo_ivt_mt'] = (
        df_lsoa.loc[mask_msu_ivt_better, 'msu_probdeath_lvo_ivt'].values)
    
    return df_lsoa


def build_full_lsoa_outcomes_from_unique_time_results_optimist(
            df_travel_times,
            df_outcomes_ivt,
            df_outcomes_mt,
            df_mrs_ivt,
            df_mrs_mt,
    ):
    # Apply these results to the treatment times in each scenario.
    df_lsoa = df_travel_times.copy().reset_index()


    # Usual care:
    # IVT:
    df_lsoa = pd.merge(
        df_lsoa, df_outcomes_ivt,
        left_on='usual_care_ivt', right_on='time_to_ivt', how='left'
        )
    keys_for_df_ivt = [k for k in df_outcomes_ivt.columns if 'time' not in k]
    keys_usual_care_from_df_ivt = [f'usual_care_{k}' for k in keys_for_df_ivt]
    df_lsoa = df_lsoa.drop('time_to_ivt', axis='columns')
    df_lsoa = df_lsoa.rename(columns=dict(
        zip(keys_for_df_ivt, keys_usual_care_from_df_ivt)))
    # MT:
    df_lsoa = pd.merge(
        df_lsoa, df_outcomes_mt,
        left_on='usual_care_mt', right_on='time_to_mt', how='left'
        )
    keys_for_df_mt = [k for k in df_outcomes_mt.columns if 'time' not in k]
    keys_usual_care_from_df_mt = [f'usual_care_{k}' for k in keys_for_df_mt]
    df_lsoa = df_lsoa.drop('time_to_mt', axis='columns')
    df_lsoa = df_lsoa.rename(columns=dict(
        zip(keys_for_df_mt, keys_usual_care_from_df_mt)))
    # IVT & MT:
    cols_lvo_ivt_mt = [
        'lvo_ivt_mt_mrs_shift',
        'lvo_ivt_mt_utility_shift',
        'lvo_ivt_mt_mrs_0-2',
    ]
    cols_lvo_ivt = [
        'lvo_ivt_mrs_shift',
        'lvo_ivt_utility_shift',
        'lvo_ivt_mrs_0-2',
    ]
    cols_lvo_mt = [
        'lvo_mt_mrs_shift',
        'lvo_mt_utility_shift',
        'lvo_mt_mrs_0-2'
    ]
    cols_usual_care_lvo_ivt_mt = [f'usual_care_{k}' for k in cols_lvo_ivt_mt]
    cols_usual_care_lvo_ivt = [f'usual_care_{k}' for k in cols_lvo_ivt]
    cols_usual_care_lvo_mt = [f'usual_care_{k}' for k in cols_lvo_mt]
    # Initially copy over the MT data:
    df_lsoa[cols_usual_care_lvo_ivt_mt] = df_lsoa[cols_usual_care_lvo_mt].copy()
    # Find whether IVT is better than MT:
    mask_usual_care_ivt_better = (
        df_lsoa['usual_care_lvo_mt_mrs_0-2'] <
        df_lsoa['usual_care_lvo_ivt_mrs_0-2']
    )
    # Store this mask for later use of picking out mRS dists:
    df_lsoa['usual_care_lvo_ivt_better_than_mt'] = mask_usual_care_ivt_better
    # Update outcomes for those patients:
    # (keep the .values because the column names don't match)
    df_lsoa.loc[mask_usual_care_ivt_better, cols_usual_care_lvo_ivt_mt] = (
        df_lsoa.loc[mask_usual_care_ivt_better, cols_usual_care_lvo_ivt].values)

    # Copy over probability of death:
    # IVT:
    df_lsoa = pd.merge(
        df_lsoa, df_mrs_ivt[
            ['time_to_ivt', 'nlvo_ivt_mrs_dists_noncum_6', 'lvo_ivt_mrs_dists_noncum_6']],
        left_on='usual_care_ivt', right_on='time_to_ivt', how='left'
        )
    df_lsoa = df_lsoa.drop('time_to_ivt', axis='columns')
    df_lsoa = df_lsoa.rename(columns={
        'nlvo_ivt_mrs_dists_noncum_6': 'usual_care_probdeath_nlvo_ivt',
        'lvo_ivt_mrs_dists_noncum_6': 'usual_care_probdeath_lvo_ivt',
        })
    # MT:
    df_lsoa = pd.merge(
        df_lsoa, df_mrs_mt[['time_to_mt', 'lvo_mt_mrs_dists_noncum_6']],
        left_on='usual_care_mt', right_on='time_to_mt', how='left'
        )
    df_lsoa = df_lsoa.drop('time_to_mt', axis='columns')
    df_lsoa = df_lsoa.rename(columns={
        'lvo_mt_mrs_dists_noncum_6': 'usual_care_probdeath_lvo_mt'})
    # IVT & MT:
    # Initially copy over the MT data:
    df_lsoa['usual_care_probdeath_lvo_ivt_mt'] = (
        df_lsoa['usual_care_probdeath_lvo_mt'].copy())
    # Update values where IVT is better than MT:
    # (keep the .values because the column names don't match)
    df_lsoa.loc[mask_usual_care_ivt_better, 'usual_care_probdeath_lvo_ivt_mt'] = (
        df_lsoa.loc[mask_usual_care_ivt_better, 'usual_care_probdeath_lvo_ivt'].values)


    # Redirection rejected:
    # IVT:
    df_lsoa = pd.merge(
        df_lsoa, df_outcomes_ivt,
        left_on='redirection_rejected_ivt', right_on='time_to_ivt', how='left'
        )
    keys_for_df_ivt = [k for k in df_outcomes_ivt.columns if 'time' not in k]
    keys_redirection_rejected_from_df_ivt = [f'redirection_rejected_{k}' for k in keys_for_df_ivt]
    df_lsoa = df_lsoa.drop('time_to_ivt', axis='columns')
    df_lsoa = df_lsoa.rename(columns=dict(
        zip(keys_for_df_ivt, keys_redirection_rejected_from_df_ivt)))
    # MT:
    df_lsoa = pd.merge(
        df_lsoa, df_outcomes_mt,
        left_on='redirection_rejected_mt', right_on='time_to_mt', how='left'
        )
    keys_for_df_mt = [k for k in df_outcomes_mt.columns if 'time' not in k]
    keys_redirection_rejected_from_df_mt = [f'redirection_rejected_{k}' for k in keys_for_df_mt]
    df_lsoa = df_lsoa.drop('time_to_mt', axis='columns')
    df_lsoa = df_lsoa.rename(columns=dict(
        zip(keys_for_df_mt, keys_redirection_rejected_from_df_mt)))
    # IVT & MT:
    cols_lvo_ivt_mt = [
        'lvo_ivt_mt_mrs_shift',
        'lvo_ivt_mt_utility_shift',
        'lvo_ivt_mt_mrs_0-2',
    ]
    cols_lvo_ivt = [
        'lvo_ivt_mrs_shift',
        'lvo_ivt_utility_shift',
        'lvo_ivt_mrs_0-2',
    ]
    cols_lvo_mt = [
        'lvo_mt_mrs_shift',
        'lvo_mt_utility_shift',
        'lvo_mt_mrs_0-2'
    ]
    cols_redirection_rejected_lvo_ivt_mt = [f'redirection_rejected_{k}' for k in cols_lvo_ivt_mt]
    cols_redirection_rejected_lvo_ivt = [f'redirection_rejected_{k}' for k in cols_lvo_ivt]
    cols_redirection_rejected_lvo_mt = [f'redirection_rejected_{k}' for k in cols_lvo_mt]
    # Initially copy over the MT data:
    df_lsoa[cols_redirection_rejected_lvo_ivt_mt] = df_lsoa[cols_redirection_rejected_lvo_mt].copy()
    # Find whether IVT is better than MT:
    mask_redirection_rejected_ivt_better = (
        df_lsoa['redirection_rejected_lvo_mt_mrs_0-2'] <
        df_lsoa['redirection_rejected_lvo_ivt_mrs_0-2']
    )
    # Store this mask for later use of picking out mRS dists:
    df_lsoa['redirection_rejected_lvo_ivt_better_than_mt'] = mask_redirection_rejected_ivt_better
    # Update outcomes for those patients:
    # (keep the .values because the column names don't match)
    df_lsoa.loc[mask_redirection_rejected_ivt_better, cols_redirection_rejected_lvo_ivt_mt] = (
        df_lsoa.loc[mask_redirection_rejected_ivt_better, cols_redirection_rejected_lvo_ivt].values)

    # Copy over probability of death:
    # IVT:
    df_lsoa = pd.merge(
        df_lsoa, df_mrs_ivt[
            ['time_to_ivt', 'nlvo_ivt_mrs_dists_noncum_6', 'lvo_ivt_mrs_dists_noncum_6']],
        left_on='redirection_rejected_ivt', right_on='time_to_ivt', how='left'
        )
    df_lsoa = df_lsoa.drop('time_to_ivt', axis='columns')
    df_lsoa = df_lsoa.rename(columns={
        'nlvo_ivt_mrs_dists_noncum_6': 'redirection_rejected_probdeath_nlvo_ivt',
        'lvo_ivt_mrs_dists_noncum_6': 'redirection_rejected_probdeath_lvo_ivt',
        })
    # MT:
    df_lsoa = pd.merge(
        df_lsoa, df_mrs_mt[['time_to_mt', 'lvo_mt_mrs_dists_noncum_6']],
        left_on='redirection_rejected_mt', right_on='time_to_mt', how='left'
        )
    df_lsoa = df_lsoa.drop('time_to_mt', axis='columns')
    df_lsoa = df_lsoa.rename(columns={
        'lvo_mt_mrs_dists_noncum_6': 'redirection_rejected_probdeath_lvo_mt'})
    # IVT & MT:
    # Initially copy over the MT data:
    df_lsoa['redirection_rejected_probdeath_lvo_ivt_mt'] = (
        df_lsoa['redirection_rejected_probdeath_lvo_mt'].copy())
    # Update values where IVT is better than MT:
    # (keep the .values because the column names don't match)
    df_lsoa.loc[mask_redirection_rejected_ivt_better, 'redirection_rejected_probdeath_lvo_ivt_mt'] = (
        df_lsoa.loc[mask_redirection_rejected_ivt_better, 'redirection_rejected_probdeath_lvo_ivt'].values)


    # Redirection approved:
    # IVT:
    df_lsoa = pd.merge(
        df_lsoa, df_outcomes_ivt,
        left_on='redirection_approved_ivt', right_on='time_to_ivt', how='left'
        )
    keys_for_df_ivt = [k for k in df_outcomes_ivt.columns if 'time' not in k]
    keys_redirection_approved_from_df_ivt = [f'redirection_approved_{k}' for k in keys_for_df_ivt]
    df_lsoa = df_lsoa.drop('time_to_ivt', axis='columns')
    df_lsoa = df_lsoa.rename(columns=dict(
        zip(keys_for_df_ivt, keys_redirection_approved_from_df_ivt)))
    # MT:
    df_lsoa = pd.merge(
        df_lsoa, df_outcomes_mt,
        left_on='redirection_approved_mt', right_on='time_to_mt', how='left'
        )
    keys_for_df_mt = [k for k in df_outcomes_mt.columns if 'time' not in k]
    keys_redirection_approved_from_df_mt = [f'redirection_approved_{k}' for k in keys_for_df_mt]
    df_lsoa = df_lsoa.drop('time_to_mt', axis='columns')
    df_lsoa = df_lsoa.rename(columns=dict(
        zip(keys_for_df_mt, keys_redirection_approved_from_df_mt)))
    # IVT & MT:
    cols_lvo_ivt_mt = [
        'lvo_ivt_mt_mrs_shift',
        'lvo_ivt_mt_utility_shift',
        'lvo_ivt_mt_mrs_0-2',
    ]
    cols_lvo_ivt = [
        'lvo_ivt_mrs_shift',
        'lvo_ivt_utility_shift',
        'lvo_ivt_mrs_0-2',
    ]
    cols_lvo_mt = [
        'lvo_mt_mrs_shift',
        'lvo_mt_utility_shift',
        'lvo_mt_mrs_0-2'
    ]
    cols_redirection_approved_lvo_ivt_mt = [f'redirection_approved_{k}' for k in cols_lvo_ivt_mt]
    cols_redirection_approved_lvo_ivt = [f'redirection_approved_{k}' for k in cols_lvo_ivt]
    cols_redirection_approved_lvo_mt = [f'redirection_approved_{k}' for k in cols_lvo_mt]
    # Initially copy over the MT data:
    df_lsoa[cols_redirection_approved_lvo_ivt_mt] = df_lsoa[cols_redirection_approved_lvo_mt].copy()
    # Find whether IVT is better than MT:
    mask_redirection_approved_ivt_better = (
        df_lsoa['redirection_approved_lvo_mt_mrs_0-2'] <
        df_lsoa['redirection_approved_lvo_ivt_mrs_0-2']
    )
    # Store this mask for later use of picking out mRS dists:
    df_lsoa['redirection_approved_lvo_ivt_better_than_mt'] = mask_redirection_approved_ivt_better
    # Update outcomes for those patients:
    # (keep the .values because the column names don't match)
    df_lsoa.loc[mask_redirection_approved_ivt_better, cols_redirection_approved_lvo_ivt_mt] = (
        df_lsoa.loc[mask_redirection_approved_ivt_better, cols_redirection_approved_lvo_ivt].values)

    # Copy over probability of death:
    # IVT:
    df_lsoa = pd.merge(
        df_lsoa, df_mrs_ivt[
            ['time_to_ivt', 'nlvo_ivt_mrs_dists_noncum_6', 'lvo_ivt_mrs_dists_noncum_6']],
        left_on='redirection_approved_ivt', right_on='time_to_ivt', how='left'
        )
    df_lsoa = df_lsoa.drop('time_to_ivt', axis='columns')
    df_lsoa = df_lsoa.rename(columns={
        'nlvo_ivt_mrs_dists_noncum_6': 'redirection_approved_probdeath_nlvo_ivt',
        'lvo_ivt_mrs_dists_noncum_6': 'redirection_approved_probdeath_lvo_ivt',
        })
    # MT:
    df_lsoa = pd.merge(
        df_lsoa, df_mrs_mt[['time_to_mt', 'lvo_mt_mrs_dists_noncum_6']],
        left_on='redirection_approved_mt', right_on='time_to_mt', how='left'
        )
    df_lsoa = df_lsoa.drop('time_to_mt', axis='columns')
    df_lsoa = df_lsoa.rename(columns={
        'lvo_mt_mrs_dists_noncum_6': 'redirection_approved_probdeath_lvo_mt'})
    # IVT & MT:
    # Initially copy over the MT data:
    df_lsoa['redirection_approved_probdeath_lvo_ivt_mt'] = (
        df_lsoa['redirection_approved_probdeath_lvo_mt'].copy())
    # Update values where IVT is better than MT:
    # (keep the .values because the column names don't match)
    df_lsoa.loc[mask_redirection_approved_ivt_better, 'redirection_approved_probdeath_lvo_ivt_mt'] = (
        df_lsoa.loc[mask_redirection_approved_ivt_better, 'redirection_approved_probdeath_lvo_ivt'].values)

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

    return df_icb, df_isdn, df_nearest_ivt, df_ambo


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
@st.cache_data
def group_mrs_dists_by_region(df_lsoa, nearest_ivt_units, **kwargs):
    df_lsoa = df_lsoa.copy()

    # ----- LSOAs for grouping results -----
    # Merge in other region info.

    # Load region info for each LSOA:
    # Relative import from package files:
    df_lsoa_regions = stroke_maps.load_data.lsoa_region_lookup()
    df_lsoa_regions = df_lsoa_regions.reset_index()
    df_lsoa = pd.merge(
        df_lsoa, df_lsoa_regions,
        left_on='lsoa', right_on='lsoa', how='left'
        )

    # Load further region data linking SICBL to other regions:
    df_regions = stroke_maps.load_data.region_lookup()
    df_regions = df_regions.reset_index()
    # Drop columns already in df_lsoa:
    df_regions = df_regions.drop(['region', 'region_type'], axis='columns')
    df_lsoa = pd.merge(
        df_lsoa, df_regions,
        left_on='region_code', right_on='region_code', how='left'
        )

    # Load ambulance service data:
    df_lsoa_ambo = stroke_maps.load_data.ambulance_lsoa_lookup()
    df_lsoa_ambo = df_lsoa_ambo.reset_index()
    # Merge in:
    df_lsoa = pd.merge(
        df_lsoa, df_lsoa_ambo[['LSOA11NM', 'ambo22']],
        left_on='lsoa', right_on='LSOA11NM', how='left'
        ).drop('LSOA11NM', axis='columns')

    # Merge in nearest IVT units:
    df_lsoa = pd.merge(df_lsoa, nearest_ivt_units,
                       left_on='lsoa', right_index=True)

    df = group_mrs_dists_by_column(df_lsoa, **kwargs)
    return df.copy()


def group_mrs_dists_by_column(df_lsoa, col_region='', col_vals=[]):
    # Glob results by column values:
    df_lsoa = df_lsoa.copy()

    if len(col_region) > 0:
        if len(col_vals) == 0:
            col_vals = sorted(list(set(df_lsoa[col_region])))
        use_all = False
    else:
        col_vals = ['National']
        use_all = True

    cols_to_combine = [c for c in df_lsoa.columns if
                       (('dist' in c) & ('noncum' in c))]

    df_by_region = pd.DataFrame()
    for val in col_vals:
        if use_all:
            mask = [True] * len(df_lsoa)
        else:
            mask = (df_lsoa[col_region] == val)
        df_region = df_lsoa.loc[mask, ['Admissions'] + cols_to_combine].copy()

        # Admissions:
        df_by_region.loc[val, 'Admissions'] = df_region['Admissions'].sum()

        # Remove repeats from all the mRS bands:
        cols_each_scen = sorted(list(set(
            ['_'.join(c.split('_')[:-1]) for c in cols_to_combine])))

        # Stats:
        for c, col in enumerate(cols_each_scen):
            cols_here = [c for c in df_region.columns if c.startswith(col)]
            cols_cumulative = [c.replace('_noncum', '') for c in cols_here]
            cols_std = [
                c.replace('_noncum_', '_noncum_std_') for c in cols_here]

            # Split list of values into one column per mRS band
            # and keep one row per LSOA.
            vals = df_region[cols_here].copy()

            # Create stats from these data:
            weighted_stats = DescrStatsW(
                vals, weights=df_region['Admissions'], ddof=0)
            # Means (one value per mRS):
            means = weighted_stats.mean
            # Standard deviations (one value per mRS):
            stds = weighted_stats.std
            # Cumulative probability from the mean bins:
            cumulatives = np.cumsum(means)

            # Store in results df:
            df_by_region.loc[val, cols_here] = means
            df_by_region.loc[val, cols_cumulative] = cumulatives
            df_by_region.loc[val, cols_std] = stds

    # Round the values.
    # Outcomes:
    cols_to_round = [c for c in df_by_region.columns if 'dist' in c]
    df_by_region[cols_to_round] = np.round(df_by_region[cols_to_round], 3)
    return df_by_region.copy()


# ###########################
# ##### AVERAGE RESULTS #####
# ###########################
def combine_results_by_occlusion_type(
        df_lsoa,
        prop_dict,
        combine_mrs_dists=False,
        scenario_list=[]
        ):
    """
    Make two new dataframes, one with all the column 1 data
    and one with all the column 2 data, and with the same column
    names. Then subtract one dataframe from the other
    and merge the results back into the main one.
    This is more efficient than calculating and creating
    each new column individually.
    """
    # Use a copy of the input dataframe so we can temporarily
    # add columns for no-treatment:
    df_lsoa_to_update = df_lsoa.copy()

    # Simple addition: x% of column 1 plus y% of column 2.
    # Column names for these new DataFrames:
    cols_combo = []
    cols_nlvo = []
    cols_lvo = []

    # Don't combine treatment types for now
    # (no nLVO with MT data).
    if len(scenario_list) > 0:
        pass
    else:
        scenario_list = ['drip_ship', 'mothership', 'redirect']
    treatment_list = ['ivt', 'mt', 'ivt_mt']
    if combine_mrs_dists:
        outcome_list = ['mrs_dists_noncum']  # not cumulative
    else:
        outcome_list = ['mrs_0-2', 'mrs_shift', 'utility_shift']

    # For no-treatment mRS distributions:
    dist_dict = load_reference_mrs_dists()

    for s in scenario_list:
        for t in treatment_list:
            for o in outcome_list:
                if combine_mrs_dists:
                    cols_mrs_lvo = [f'{s}_lvo_{t}_{o}_{i}' for i in range(7)]

                    if t == 'mt':
                        cols_mrs_nlvo = [
                            f'{s}_nlvo_{t}_{o}_{i}' for i in range(7)]
                        if 'noncum' in o:
                            dist = dist_dict['nlvo_no_treatment_noncum']
                        else:
                            dist = dist_dict['nlvo_no_treatment']
                        # Add this data to the starting dataframe:
                        df_lsoa_to_update[cols_mrs_nlvo] = dist
                    else:
                        cols_mrs_nlvo = [
                            f'{s}_nlvo_ivt_{o}_{i}' for i in range(7)]
                    try:
                        data_nlvo = df_lsoa_to_update[cols_mrs_nlvo]
                        data_exists = True
                    except KeyError:
                        data_exists = False

                    if data_exists:
                        cols_here = [f'{s}_combo_{t}_{o}_{i}' for i in range(7)]
                        cols_combo += cols_here
                        cols_nlvo += cols_mrs_nlvo
                        cols_lvo += cols_mrs_lvo

                else:
                    if t == 'mt':
                        if o in ['mrs_shift', 'utility_shift']:
                            col_nlvo = f'{s}_nlvo_ivt_{o}'
                            df_lsoa_to_update[col_nlvo] = 0.0
                        else:
                            col_nlvo = f'nlvo_no_treatment_{o}'
                    else:
                        col_nlvo = f'{s}_nlvo_ivt_{o}'

                    # col_nlvo = f'nlvo_{s}_{t}_{o}'
                    col_lvo = f'{s}_lvo_{t}_{o}'
                    try:
                        data_nlvo = df_lsoa_to_update[col_nlvo]
                        data_exists = True
                    except KeyError:
                        data_exists = False

                    if data_exists:
                        cols_combo.append(f'{s}_combo_{t}_{o}')
                        cols_nlvo.append(col_nlvo)
                        cols_lvo.append(col_lvo)

    # Pick out the data from the original dataframe:
    df1 = df_lsoa_to_update[cols_nlvo].copy()
    df2 = df_lsoa_to_update[cols_lvo].copy()
    # Rename columns so they match:
    df1.columns = cols_combo
    df2.columns = cols_combo

    # Create new dataframe from combining the two separate ones:
    combo_data = (
        df1 * prop_dict['nlvo'] +
        df2 * prop_dict['lvo']
    )

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
    # # Update column names to mark them as combined:
    # combo_data.columns = [
    #     '_'.join(col.split('_')[0], 'combo', col.split('_')[1:])
    #     f'combo_{col}' for col in combo_data.columns]
    # Merge this new data into the starting dataframe:
    df_lsoa = pd.merge(df_lsoa, combo_data, left_index=True, right_index=True)

    return df_lsoa.copy()


def combine_results_by_redirection(
        df_lsoa,
        redirect_dict,
        combine_mrs_dists=False
        ):
    # Simple addition: x% of column 1 plus y% of column 2.

    prop_lvo_redirected = redirect_dict['sensitivity']
    prop_nlvo_redirected = (1.0 - redirect_dict['specificity'])

    prop_dict = {
        'nlvo': prop_nlvo_redirected,
        'lvo': prop_lvo_redirected,
    }

    # Don't combine treatment types for now
    # (no nLVO with MT data).
    occlusion_list = ['nlvo', 'lvo']
    treatment_list = ['ivt', 'mt', 'ivt_mt']
    if combine_mrs_dists:
        outcome_list = ['mrs_dists_noncum']  # not cumulative
    else:
        outcome_list = ['mrs_0-2', 'mrs_shift', 'utility_shift']

    for v in occlusion_list:
        # Column names for these new DataFrames:
        cols_combo = []
        cols_rr = []
        cols_ra = []
        prop = prop_dict[v]
        for t in treatment_list:
            for o in outcome_list:
                if combine_mrs_dists:
                    cols_mrs_rr = [
                        f'redirection_rejected_{v}_{t}_{o}_{i}' for i in range(7)]
                    cols_mrs_ra = [
                        f'redirection_approved_{v}_{t}_{o}_{i}' for i in range(7)]
                    try:
                        data_nlvo = df_lsoa[cols_mrs_rr]
                        data_exists = True
                    except KeyError:
                        data_exists = False

                    if data_exists:
                        cols_here = [
                            f'redirection_considered_{v}_{t}_{o}_{i}' for i in range(7)]
                        cols_combo += cols_here
                        cols_rr += cols_mrs_rr
                        cols_ra += cols_mrs_ra
                else:
                    col_rr = f'redirection_rejected_{v}_{t}_{o}'
                    col_ra = f'redirection_approved_{v}_{t}_{o}'
                    try:
                        df_lsoa[col_rr]
                        data_exists = True
                    except KeyError:
                        data_exists = False

                    if data_exists:
                        cols_combo.append(f'redirection_considered_{v}_{t}_{o}')
                        cols_rr.append(col_rr)
                        cols_ra.append(col_ra)

        # Pick out the data from the original dataframe:
        df1 = df_lsoa[cols_rr].copy()
        df2 = df_lsoa[cols_ra].copy()
        # Rename columns so they match:
        df1.columns = cols_combo
        df2.columns = cols_combo

        # Create new dataframe from combining the two separate ones:
        combo_data = (
            df1 * (1.0 - prop) +
            df2 * prop
        )

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

        # Merge this new data into the starting dataframe:
        df_lsoa = pd.merge(df_lsoa, combo_data,
                           left_index=True, right_index=True)

    return df_lsoa.copy()


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

    for occ in occlusion_types:
        for tre in treatment_types:
            for out in outcome_types:
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
                        cols_here = [f'{occ}_{tre}_{out}_{i}'
                                     for i in range(7)]
                        cols_combo += cols_here
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
