"""
Geography.
"""
# ----- Imports -----
import streamlit as st
import pandas as pd
import numpy as np

import stroke_maps.load_data

from classes.geography_processing import Geoprocessing
from utilities.utils import print_progress_loc


# ----- Functions -----
def import_stroke_unit_services(
        use_msu=True,
        keep_only_ivt_mt=False,
        keep_only_england=True
        ):
    """
    """
    # Set up stroke unit services (IVT, MT, MSU).
    df_unit_services = stroke_maps.load_data.stroke_unit_region_lookup()

    # Rename columns to match what the rest of the model here wants.
    df_unit_services.index.name = 'Postcode'
    df_unit_services = df_unit_services.rename(columns={
        'use_ivt': 'Use_IVT',
        'use_mt': 'Use_MT',
        'use_msu': 'Use_MSU',
    })

    if keep_only_ivt_mt:
        # Remove stroke units that don't offer IVT or MT:
        mask = (
            (df_unit_services['Use_IVT'] == 1) |
            (df_unit_services['Use_MT'] == 1)
        )
        df_unit_services = df_unit_services.loc[mask].copy()
    else:
        pass

    if keep_only_england:
        # Limit to England:
        mask = df_unit_services['country'] == 'England'
        df_unit_services = df_unit_services.loc[mask].copy()
        # Remove Wales:
        df_unit_services = df_unit_services.loc[
            df_unit_services['region_type'] != 'LHB'].copy()
    else:
        pass

    # Limit the units list to only units in the travel time matrix:
    df_travel = pd.read_csv(
        './data/inter_hospital_time_calibrated.csv',
        index_col='from_postcode'
        )
    units_allowed = df_travel.index.values
    mask_allowed = df_unit_services.index.isin(units_allowed)
    df_unit_services = df_unit_services[mask_allowed].copy()

    # Limit which columns to show:
    cols_to_keep = ['ssnap_name', 'Use_IVT', 'Use_MT', 'isdn']
    df_unit_services = df_unit_services[cols_to_keep]

    if use_msu:
        df_unit_services['Use_MSU'] = df_unit_services['Use_MT'].copy()

    # Sort by ISDN name for nicer display:
    df_unit_services = df_unit_services.sort_values(['isdn', 'ssnap_name'])

    return df_unit_services


def load_lsoa_region_lookups():
    # Load region info for each LSOA:
    # Relative import from package files:
    df_lsoa_regions = stroke_maps.load_data.lsoa_region_lookup()
    df_lsoa_regions = df_lsoa_regions.reset_index()

    # Load further region data linking SICBL to other regions:
    df_regions = stroke_maps.load_data.region_lookup()
    df_regions = df_regions.reset_index()
    # Drop columns already in df_lsoa:
    df_regions = df_regions.drop(['region', 'region_type'], axis='columns')
    df_lsoa_regions = pd.merge(
        df_lsoa_regions, df_regions,
        left_on='region_code', right_on='region_code', how='left'
        )

    # Load ambulance service data:
    df_lsoa_ambo = stroke_maps.load_data.ambulance_lsoa_lookup()
    df_lsoa_ambo = df_lsoa_ambo.reset_index()
    # Merge in:
    df_lsoa_regions = pd.merge(
        df_lsoa_regions, df_lsoa_ambo[['LSOA11NM', 'ambo22']],
        left_on='lsoa', right_on='LSOA11NM', how='left'
        ).drop('LSOA11NM', axis='columns')
    return df_lsoa_regions


def select_unit_services():
    """
    """
    # Load stroke unit data from file:
    df = import_stroke_unit_services(
        use_msu=False,
        keep_only_ivt_mt=False,
        keep_only_england=True
        )

    # Display this as an editable dataframe:
    df_unit_services = st.data_editor(
        df,
        disabled=['postcode', 'ssnap_name', 'isdn'],
        # height=180  # limit height to show fewer rows
        # Make columns display as checkboxes instead of 0/1 ints:
        column_config={
            'Use_IVT': st.column_config.CheckboxColumn(),
            'Use_MT': st.column_config.CheckboxColumn(),
        },
        )
    return df_unit_services


@st.cache_data
def find_nearest_units_each_lsoa(df_unit_services, _log=True, _log_loc=None):
    """

    Result
    ------
    df_geo - pd.Dataframe. Columns 'LSOA', 'nearest_ivt_unit',
             'nearest_ivt_time', 'nearest_mt_unit', 'nearest_mt_time',
             'transfer_unit', 'transfer_required', 'transfer_time',
             'nearest_msu_unit', 'nearest_msu_time', 'Admissions',
             'nearest_ivt_then_mt_time'
    """
    try:
        geo = st.session_state['geo']
    except KeyError:
        # Process and save geographic data
        # (only needed when hospital data changes)
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
    # Round travel times to nearest minute.
    # +1e-5 to make all 0.5 times round up to next minute.
    cols_times = ['nearest_ivt_time', 'nearest_mt_time', 'transfer_time',
                  'nearest_msu_time']
    df_geo[cols_times] = np.round(df_geo[cols_times] + 1e-5, 0)
    # Separate column for separate travel time including transfer:
    df_geo['nearest_ivt_then_mt_time'] = (
        df_geo['nearest_ivt_time'] + df_geo['transfer_time'])
    # Cache the geo class so that on the next run all of the big
    # data files are not loaded in another time.
    st.session_state['geo'] = geo

    if _log:
        p = 'Assigned LSOA to nearest units.'
        print_progress_loc(p, _log_loc)
    return df_geo


@st.cache_data
def find_unique_travel_times(
        df_times,
        cols_ivt=['nearest_ivt_time', 'nearest_mt_time'],
        cols_mt=['nearest_ivt_then_mt_time', 'nearest_mt_time'],
        cols_pairs={
            'transfer': ('nearest_ivt_time', 'nearest_ivt_then_mt_time'),
            'no_transfer': ('nearest_mt_time', 'nearest_mt_time')
        },
        _log=True,
        _log_loc=None
        ):
    """

    """
    # IVT can either be at the nearest unit or at the MT unit if
    # redirected. MT is always at the MT unit, either travelling
    # there directly or going via the IVT-only unit.
    times_to_ivt = sorted(list(set(df_times[cols_ivt].values.flatten())))
    times_to_mt = sorted(list(set(df_times[cols_mt].values.flatten())))

    # Find all pairs of times.
    # Combinations are: IVT at nearest unit, then MT after transfer;
    #                   IVT and MT at nearest MT unit.
    all_pairs = {}
    for label, pair in cols_pairs.items():
        pairs_here = df_times[list(pair)].drop_duplicates()
        # Don't use rename dictionary because can have duplicate
        # column names in pair.
        pairs_here.columns = ['travel_for_ivt', 'travel_for_mt']
        all_pairs[label] = pairs_here

    if _log:
        p = 'Gathered unique travel times.'
        print_progress_loc(p, _log_loc)
    return times_to_ivt, times_to_mt, all_pairs


@st.cache_data
def find_region_admissions_by_unique_travel_times(
        df_lsoa_units_times, keep_only_england=True, unique_travel=True,
        _log=True, _log_loc=None
        ):
    """
    df_lsoa_units_times includes admissions data.

    Multiple layers in this dictionary.
    + dict_region_unique_times
      + all_patients
      + nearest_unit_no_mt
        + usual_care_ivt
        + usual_care_mt
        + redirection_ivt
        + redirection_mt
    """
    # Load in LSOA-region lookup:
    df_lsoa_regions = load_lsoa_region_lookups()
    # Columns: 'lsoa', 'lsoa_code', 'region', 'region_code',
    # 'region_type', 'short_code', 'country', 'icb', 'icb_code',
    # 'isdn', 'ambo22'.
    if keep_only_england:
        mask_eng = df_lsoa_regions['region_type'] == 'SICBL'
        df_lsoa_regions = df_lsoa_regions.loc[mask_eng].copy()

    # Merge in admissions and timings data:
    cols_to_merge = ['LSOA', 'Admissions', 'transfer_required']
    if unique_travel:
        cols_to_merge += ['nearest_ivt_time', 'nearest_mt_time',
                          'nearest_ivt_then_mt_time']
    else:
        scens = ['usual_care', 'redirection_approved',
                 'redirection_rejected']
        treats = ['ivt', 'mt']
        cols_treat_scen = [f'{s}_{t}' for s in scens for t in treats]
        cols_to_merge += cols_treat_scen

    df_lsoa_regions = pd.merge(
        df_lsoa_regions, df_lsoa_units_times[cols_to_merge],
        left_on='lsoa', right_on='LSOA', how='right'
        )
    # Calculate this separately for each region type.
    region_types = ['region', 'icb', 'isdn', 'ambo22']
    dict_region_unique_times = {}
    masks = {'all_patients': slice(None),
             'nearest_unit_no_mt': df_lsoa_regions['transfer_required']}

    if unique_travel:
        # For usual care, all IVT is at "nearest ivt unit"
        # and all MT is at "nearest MT unit" after "time to nearest
        # ivt unit plus transfer time" (for no transfer, time is
        # zero). Under redirection, all IVT and all MT is at
        # "nearest MT unit".
        time_cols_dict = {
            'usual_care_ivt': ['nearest_ivt_time'],
            'usual_care_mt': ['nearest_ivt_then_mt_time'],
            'redirection_ivt': ['nearest_mt_time'],
            'redirection_mt': ['nearest_mt_time']
            }
        for mask_label, mask in masks.items():
            dict_region_unique_times[mask_label] = {}
            df_here = df_lsoa_regions.loc[mask]
            for region_type in region_types:
                dict_region_unique_times[mask_label][region_type] = {}
                for time_label, time_cols in time_cols_dict.items():

                    cols = [region_type, 'Admissions'] + time_cols
                    df = df_here[cols].groupby(
                        [region_type, *time_cols]).sum()
                    # df has columns for region, time, and admissions.
                    # Change to index of time, one column per region,
                    # values of admissions:
                    df = (df.unstack(time_cols).transpose()
                          .reset_index().set_index(time_cols)
                          .drop('level_0', axis='columns')
                          )
                    dict_region_unique_times[
                        mask_label][region_type][time_label] = df

        if _log:
            p = 'Found total admissions with each unique travel time per region.'
            print_progress_loc(p, _log_loc)
    else:
        # Unique treatment time combinations.
        # For usual care, all IVT is at "nearest ivt unit"
        # and all MT is at "nearest MT unit" after "time to nearest
        # ivt unit plus transfer time" (for no transfer, time is
        # zero). Under redirection, all IVT and all MT is at
        # "nearest MT unit".
        for mask_label, mask in masks.items():
            dict_region_unique_times[mask_label] = {}
            df_here = df_lsoa_regions.loc[mask]
            for region_type in region_types:
                cols = [region_type, 'Admissions'] + cols_treat_scen
                df = df_here[cols].groupby(
                    [region_type, *cols_treat_scen]).sum()
                # df has columns for region, time, and admissions.
                # Change to index of time, one column per region,
                # values of admissions:
                df = (df.unstack(cols_treat_scen).transpose()
                      .reset_index().set_index(cols_treat_scen)
                      .drop('level_0', axis='columns')
                      )
                dict_region_unique_times[
                    mask_label][region_type] = df

        if _log:
            p = 'Found total admissions with each set of unique treatment times per region.'
            print_progress_loc(p, _log_loc)
    return dict_region_unique_times
