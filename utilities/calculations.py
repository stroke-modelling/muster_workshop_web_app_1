
# ----- Imports -----
import streamlit as st
from importlib_resources import files
import pandas as pd

# For running outcomes:
from classes.geography_processing import Geoprocessing
from classes.model import Model
from classes.scenario import Scenario

# Custom functions:
# Containers:


@st.cache_data
def calculate_outcomes(input_dict, df_unit_services):
    """

    # Run the outcomes with the selected pathway:
    """
    # Feed input parameters into Scenario:
    scenario = Scenario({
        'name': 1,
        'limit_to_england': True,
        **input_dict
    })

    # Process and save geographic data (only needed when hospital data changes)
    geo = Geoprocessing(
        df_unit_services=df_unit_services
        )
    geo.run()

    # Reset index because Model expects a column named 'msoa':
    geo.combined_data = geo.combined_data.reset_index()

    # Set up model
    model = Model(
        scenario=scenario,
        geodata=geo.combined_data
        )

    # Run model
    model.run()

    df_lsoa = model.full_results.copy()
    df_lsoa.index.names = ['lsoa']
    df_lsoa.columns.names = ['property']

    # TO DO - the results df contains a mix of scenarios
    # (drip and ship, mothership, msu) in the column names.
    # Pull them out and put them into 'scenario' header.
    # Also need to do something with separate nlvo, lvo, treatment types
    # because current setup just wants some averaged added utility outcome
    # rather than split by stroke type.
    return df_lsoa


@st.cache_data
def group_results_by_region(df_lsoa, df_unit_services):
    df_lsoa = df_lsoa.copy()
    # ----- LSOAs for grouping results -----
    # Merge in other region info.

    # Load region info for each LSOA:
    # Relative import from package files:
    path_to_file = files('stroke_maps.data').joinpath('regions_lsoa_ew.csv')
    df_lsoa_regions = pd.read_csv(path_to_file)  # , index_col=[0, 1])
    df_lsoa = pd.merge(
        df_lsoa, df_lsoa_regions,
        left_on='lsoa', right_on='lsoa', how='left'
        )

    # Load further region data linking SICBL to other regions:
    path_to_file = files('stroke_maps.data').joinpath('regions_ew.csv')
    df_regions = pd.read_csv(path_to_file)  # , index_col=[0, 1])
    # Drop columns already in df_lsoa:
    df_regions = df_regions.drop(['region', 'region_type'], axis='columns')
    df_lsoa = pd.merge(
        df_lsoa, df_regions,
        left_on='region_code', right_on='region_code', how='left'
        )

    # Replace some zeros with NaN:
    mask = df_lsoa['transfer_required']
    df_lsoa.loc[~mask, 'transfer_time'] = pd.NA

    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with some object columns)
    df_lsoa = df_lsoa.drop([
        'lsoa',
        'lsoa_code',
        'nearest_mt_unit',
        'transfer_unit',
        'nearest_msu_unit',
        'short_code',
        'country'
        ], axis='columns')

    df_nearest_ivt = group_results_by_nearest_ivt(df_lsoa, df_unit_services)
    df_icb = group_results_by_icb(df_lsoa)
    df_isdn = group_results_by_isdn(df_lsoa)

    return df_icb, df_isdn, df_nearest_ivt


def group_results_by_icb(df_lsoa):
    # Glob results by ICB:
    df_icb = df_lsoa.copy()
    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with some object columns)
    df_icb = df_icb.drop([
        'nearest_ivt_unit',
        'region',
        'region_type',
        'region_code',
        'icb_code',
        'isdn'
        ], axis='columns')
    # Average:
    df_icb = df_icb.groupby('icb').mean()
    return df_icb


def group_results_by_isdn(df_lsoa):
    # Glob results by ISDN:
    df_isdn = df_lsoa.copy()
    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with some object columns)
    df_isdn = df_isdn.drop([
        'nearest_ivt_unit',
        'region',
        'region_type',
        'region_code',
        'icb',
        'icb_code'
        ], axis='columns')
    # Average:
    df_isdn = df_isdn.groupby('isdn').mean()
    return df_isdn


def group_results_by_nearest_ivt(df_lsoa, df_unit_services):
    # Glob results by nearest IVT unit:
    df_nearest_ivt = df_lsoa.copy()
    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with some object columns)
    df_nearest_ivt = df_nearest_ivt.drop([
        'region',
        'region_type',
        'region_code',
        'icb',
        'icb_code',
        'isdn'
        ], axis='columns')
    # Average:
    df_nearest_ivt = df_nearest_ivt.groupby('nearest_ivt_unit').mean()
    # Merge back in the unit names:
    df_nearest_ivt = pd.merge(
        df_unit_services['stroke_team'],
        df_nearest_ivt, how='right', left_on='Postcode', right_index=True)
    return df_nearest_ivt


def convert_lsoa_to_msoa_results(df_lsoa):
    # Convert LSOA to MSOA:
    df_lsoa_to_msoa = pd.read_csv('data/lsoa_to_msoa.csv')
    df_lsoa = df_lsoa.reset_index()
    df_msoa = pd.merge(
        df_lsoa,
        df_lsoa_to_msoa[['lsoa11nm', 'msoa11cd', 'msoa11nm']],
        left_on='lsoa', right_on='lsoa11nm', how='left'
        )
    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with some object columns)
    df_msoa = df_msoa.drop([
        'lsoa', 'nearest_ivt_unit', 'nearest_mt_unit', 'transfer_unit',
        'nearest_msu_unit', 'lsoa11nm', 'msoa11nm'
        ], axis='columns')
    # Aggregate by MSOA:
    df_msoa = df_msoa.groupby('msoa11cd').mean()
    # df_msoa = df_msoa.set_index('msoa11cd')
    # Merge the MSOA names back in and set the index to (msoa_code, msoa):
    df_msoa = df_msoa.reset_index()
    df_msoa = pd.merge(
        df_msoa, df_lsoa_to_msoa[['msoa11cd', 'msoa11nm']],
        left_on='msoa11cd', right_on='msoa11cd', how='left'
        )
    # Remove duplicate rows:
    df_msoa = df_msoa.drop_duplicates()
    df_msoa = df_msoa.rename(columns={'msoa11cd': 'msoa_code', 'msoa11nm': 'msoa'})
    df_msoa = df_msoa.set_index(['msoa', 'msoa_code'])

    return df_msoa
