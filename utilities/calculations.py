
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
from classes.model import Model

# Custom functions:
from utilities.utils import load_reference_mrs_dists
from utilities.maps import dissolve_polygons_by_value
# Containers:


def calculate_geography(df_unit_services):
    # Process and save geographic data (only needed when hospital data changes)
    geo = Geoprocessing(
        df_unit_services=df_unit_services,
        limit_to_england=True
        )
    geo.run()

    # Reset index because Model expects a column named 'lsoa':
    geo.combined_data = geo.combined_data.reset_index()
    return geo


@st.cache_data
def calculate_outcomes(dict_outcome_inputs, df_unit_services, geodata):
    """

    # Run the outcomes with the selected pathway:
    """
    # Set up model
    model = Model(dict_outcome_inputs, geodata)
    # Run model
    model.run()

    # TO DO - merge in the geographical data to the outcome results.

    df_lsoa = model.full_results.copy()
    df_lsoa.index.names = ['lsoa']
    df_lsoa.columns.names = ['property']

    # # No-treatment data:
    # dist_dict = load_reference_mrs_dists()

    # Copy stroke unit names over. Currently has only postcodes.
    cols_postcode = ['nearest_ivt_unit', 'nearest_mt_unit',
                     'transfer_unit', 'nearest_msu_unit']
    for col in cols_postcode:
        if col in df_lsoa.columns:
            df_lsoa = pd.merge(
                df_lsoa, df_unit_services['stroke_team'],
                left_on=col, right_index=True, how='left'
                )
            df_lsoa = df_lsoa.rename(columns={'stroke_team': f'{col}_name'})
            # Reorder columns so name appears next to postcode.
            i = df_lsoa.columns.tolist().index(col)
            df_lsoa = df_lsoa[
                [*df_lsoa.columns[:i], f'{col}_name', *df_lsoa.columns[i:-1]]]

    # # Make a copy of nLVO IVT results for nLVO IVT+MT results:
    # cols_ivt_mt = [c for c in df_lsoa.columns if 'ivt_mt' in c]
    # for col in cols_ivt_mt:
    #     # Find the equivalent column for nLVO:
    #     col_nlvo = col.replace('lvo', 'nlvo')
    #     # Find the equivalent column for ivt-only:
    #     col_ivt = col_nlvo.replace('ivt_mt', 'ivt')
    #     # Copy over the data:
    #     df_lsoa[col_nlvo] = df_lsoa[col_ivt]
    # # Set the nLVO MT results to be the nLVO no-treatment results:
    # cols_mt = [c for c in df_lsoa.columns if
    #            (('_mt_' in c) & ('_ivt_' not in c))]
    # for col in cols_mt:
    #     # Find the equivalent column for nLVO:
    #     col_nlvo = col.replace('lvo', 'nlvo')
    #     if (('utility_shift' in col_nlvo) | ('mrs_shift' in col_nlvo)):
    #         # No change from non-treatment.
    #         df_lsoa[col_nlvo] = 0.0
    #     elif 'utility' in col_nlvo:
    #         df_lsoa[col_nlvo] = df_lsoa['nlvo_no_treatment_utility']
    #     elif 'mrs_0-2' in col_nlvo:
    #         df_lsoa[col_nlvo] = df_lsoa['nlvo_no_treatment_mrs_0-2']

    # TO DO - the results df contains a mix of scenarios
    # (drip and ship, mothership, msu) in the column names.
    # Pull them out and put them into 'scenario' header.
    # Also need to do something with separate nlvo, lvo, treatment types
    # because current setup just wants some averaged added utility outcome
    # rather than split by stroke type.

    # Outcome columns and numbers of decimal places:
    dict_outcomes_dp = {
        'mrs_0-2': 3,
        'mrs_shift': 3,
        # 'utility': 3,
        'utility_shift': 3,
    }
    for suff, dp in dict_outcomes_dp.items():
        cols = [c for c in df_lsoa.columns if c.endswith(suff)]
        for col in cols:
            df_lsoa[col] = np.round(df_lsoa[col], dp)

    df_mrs = model.full_mrs_dists.copy()
    df_mrs.index.names = ['lsoa']
    df_mrs.columns.names = ['property']

    # TO DO - more carefully pick out the 7 mRS columns and update the
    # 7 reference mRS values.

    # # Make a copy of nLVO IVT results for nLVO IVT+MT results:
    # cols_ivt_mt = [c for c in df_mrs.columns if 'ivt_mt' in c]
    # # Find the col names without _mrs at the end:
    # cols_ivt_mt_prefixes = sorted(list(set(
    #     ['_'.join(c.split('_')[:-1]) for c in cols_ivt_mt])))
    # for col in cols_ivt_mt_prefixes:
    #     # Find the equivalent column for nLVO:
    #     col_nlvo = col.replace('lvo', 'nlvo')
    #     # Find the equivalent column for ivt-only:
    #     col_ivt = col_nlvo.replace('ivt_mt', 'ivt')

    #     # Add mRS suffixes back on:
    #     cols_nlvo = [f'{col_nlvo}_{i}' for i in range(7)]
    #     cols_ivt = [f'{col_ivt}_{i}' for i in range(7)]
    #     # Copy over the data:
    #     df_mrs[cols_nlvo] = df_mrs[cols_ivt]
    # # Set the nLVO MT results to be the nLVO no-treatment results:
    # cols_mt = [c for c in df_mrs.columns if
    #            (('_mt_' in c) & ('_ivt_' not in c))]
    # # Find the col names without _mrs at the end:
    # cols_mt_prefixes = sorted(list(set(
    #     ['_'.join(c.split('_')[:-1]) for c in cols_mt])))
    # for col in cols_mt_prefixes:
    #     # Find the equivalent column for nLVO:
    #     col_nlvo = col.replace('lvo', 'nlvo')
    #     # Add the suffixes back in:
    #     cols_nlvo = [f'{col_nlvo}_{i}' for i in range(7)]
    #     if 'noncum' in col_nlvo:
    #         dist = dist_dict['nlvo_no_treatment_noncum']
    #     else:
    #         dist = dist_dict['nlvo_no_treatment']
    #     # Copy over the data:
    #     df_mrs[cols_nlvo] = dist

    return df_lsoa, df_mrs


# ##########################################
# ##### BUILD INPUTS FOR OUTCOME MODEL #####
# ##########################################
def make_outcome_inputs_usual_care(pathway_dict, df_travel_times):
    # Time to IVT:
    time_to_ivt = (
        pathway_dict['process_time_call_ambulance'] +
        pathway_dict['process_time_ambulance_response'] +
        pathway_dict['process_ambulance_on_scene_duration'] +
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


def make_outcome_inputs_msu(pathway_dict, df_travel_times):
    # Time to IVT:
    time_to_ivt = (
        pathway_dict['process_time_call_ambulance'] +
        pathway_dict['process_msu_dispatch'] +
        (df_travel_times['nearest_msu_time'].values *
         pathway_dict['scale_msu_travel_times']) +
        pathway_dict['process_msu_thrombolysis']
        )

    # Time to MT:
    # If required, everyone goes directly to the nearest MT unit.
    time_to_mt = (
        pathway_dict['process_time_call_ambulance'] +
        pathway_dict['process_msu_dispatch'] +
        (df_travel_times['nearest_msu_time'].values *
         pathway_dict['scale_msu_travel_times']) +
        pathway_dict['process_msu_thrombolysis']  +
        pathway_dict['process_msu_on_scene_post_thrombolysis'] +
        (df_travel_times['nearest_mt_time'].values *
         pathway_dict['scale_msu_travel_times']) +
        pathway_dict['process_time_msu_arrival_to_puncture']
        )

    # Bonus times - not needed for outcome model but calculated anyway.
    msu_occupied_treatment = (
        pathway_dict['process_msu_dispatch'] +
        (df_travel_times['nearest_msu_time'].values *
         pathway_dict['scale_msu_travel_times']) +
        pathway_dict['process_msu_thrombolysis'] +
        pathway_dict['process_msu_on_scene_post_thrombolysis'] +
        (df_travel_times['nearest_mt_time'].values *
         pathway_dict['scale_msu_travel_times'])
        )

    msu_occupied_no_treatment = (
        pathway_dict['process_msu_dispatch'] +
        (df_travel_times['nearest_msu_time'].values *
         pathway_dict['scale_msu_travel_times']) +
        pathway_dict['process_msu_on_scene_no_thrombolysis']
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
    # Also store bonus times:
    outcome_inputs_df['msu_occupied_treatment'] = msu_occupied_treatment
    outcome_inputs_df['msu_occupied_no_treatment'] = msu_occupied_no_treatment

    return outcome_inputs_df


# ###########################
# ##### AVERAGE RESULTS #####
# ###########################
@st.cache_data
def group_results_by_region(df_lsoa, df_unit_services):
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

    # Replace some zeros with NaN:
    mask = df_lsoa['transfer_required']
    df_lsoa.loc[~mask, 'transfer_time'] = pd.NA

    # Remove string columns:
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
        'country'
        ]
    # Only keep cols that exist (sometimes have MSU, sometimes not):
    cols_to_drop = [c for c in cols_to_drop if c in df_lsoa.columns]
    df_lsoa = df_lsoa.drop(cols_to_drop, axis='columns')

    df_nearest_ivt = group_results_by_nearest_ivt(df_lsoa, df_unit_services)
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
    for col in cols_outcome:
        df_icb[col] = np.round(df_icb[col], 3)

    # Times:
    cols_time = [c for c in df_icb.columns if 'time' in c]
    for col in cols_time:
        df_icb[col] = np.round(df_icb[col], 2)
    return df_icb


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
    for col in cols_outcome:
        df_isdn[col] = np.round(df_isdn[col], 3)

    # Times:
    cols_time = [c for c in df_isdn.columns if 'time' in c]
    for col in cols_time:
        df_isdn[col] = np.round(df_isdn[col], 2)
    return df_isdn


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
    for col in cols_outcome:
        df_ambo[col] = np.round(df_ambo[col], 3)

    # Times:
    cols_time = [c for c in df_ambo.columns if 'time' in c]
    for col in cols_time:
        df_ambo[col] = np.round(df_ambo[col], 2)
    return df_ambo


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
        df_unit_services['stroke_team'],
        df_nearest_ivt, how='right', left_on='Postcode', right_index=True)

    # Round the values.
    # Outcomes:
    cols_outcome = [c for c in df_nearest_ivt.columns if (
        (c.endswith('utility_shift')) |
        (c.endswith('mrs_0-2')) | (c.endswith('mrs_shift'))
        )]
    for col in cols_outcome:
        df_nearest_ivt[col] = np.round(df_nearest_ivt[col], 3)

    # Times:
    cols_time = [c for c in df_nearest_ivt.columns if 'time' in c]
    for col in cols_time:
        df_nearest_ivt[col] = np.round(df_nearest_ivt[col], 2)
    return df_nearest_ivt


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
    return df


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
    for col in cols_to_round:
        df_by_region[col] = np.round(df_by_region[col], 3)
    return df_by_region


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
    for col in combo_data.columns:
        combo_data[col] = np.round(combo_data[col], dp)
    # # Update column names to mark them as combined:
    # combo_data.columns = [
    #     '_'.join(col.split('_')[0], 'combo', col.split('_')[1:])
    #     f'combo_{col}' for col in combo_data.columns]
    # Merge this new data into the starting dataframe:
    df_lsoa = pd.merge(df_lsoa, combo_data, left_index=True, right_index=True)

    return df_lsoa


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
        for col in combo_data.columns:
            combo_data[col] = np.round(combo_data[col], dp)

        # Merge this new data into the starting dataframe:
        df_lsoa = pd.merge(df_lsoa, combo_data,
                           left_index=True, right_index=True)

    return df_lsoa


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
    for col in combo_data.columns:
        combo_data[col] = np.round(combo_data[col], dp)

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

    return df_lsoa


def load_or_calculate_region_outlines(outline_name, df_lsoa):
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
            df_lsoa.copy().reset_index()[['lsoa', 'nearest_ivt_unit_name']],
            col='nearest_ivt_unit_name',
            load_msoa=True
            )
        gdf_catchment_lhs = gdf_catchment_lhs.rename(
            columns={'nearest_ivt_unit_name': 'Nearest service'})

        gdf_catchment_rhs = dissolve_polygons_by_value(
            df_lsoa.copy().reset_index()[['lsoa', 'nearest_mt_unit_name']],
            col='nearest_mt_unit_name',
            load_msoa=True
            )
        gdf_catchment_rhs = gdf_catchment_rhs.rename(
            columns={'nearest_mt_unit_name': 'Nearest service'})

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
