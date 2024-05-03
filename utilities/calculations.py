
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

    # Make a copy of nLVO IVT results for nLVO IVT+MT results:
    cols_ivt_mt = [c for c in df_lsoa.columns if 'ivt_mt' in c]
    for col in cols_ivt_mt:
        # Find the equivalent column for nLVO:
        col_nlvo = col.replace('lvo', 'nlvo')
        # Find the equivalent column for ivt-only:
        col_ivt = col_nlvo.replace('ivt_mt', 'ivt')
        # Copy over the data:
        df_lsoa[col_nlvo] = df_lsoa[col_ivt]

    # TO DO - the results df contains a mix of scenarios
    # (drip and ship, mothership, msu) in the column names.
    # Pull them out and put them into 'scenario' header.
    # Also need to do something with separate nlvo, lvo, treatment types
    # because current setup just wants some averaged added utility outcome
    # rather than split by stroke type.

    # Change to float16 to preserve very few significant figures:
    import numpy as np
    cols_float = df_lsoa.select_dtypes([float]).columns
    for col in cols_float:
        df_lsoa[col] = df_lsoa[col].astype(np.float16)

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
    cols_to_drop = [
        'lsoa',
        'lsoa_code',
        'nearest_mt_unit',
        'transfer_unit',
        'nearest_msu_unit',
        'short_code',
        'country'
        ]
    # Only keep cols that exist (sometimes have MSU, sometimes not):
    cols_to_drop = [c for c in cols_to_drop if c in df_lsoa.columns]
    df_lsoa = df_lsoa.drop(cols_to_drop, axis='columns')

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

    # Keep a copy of the nearest IVT units:
    df_nearest_unit = df_msoa[['msoa11cd', 'nearest_ivt_unit']].copy()
    # Remove duplicate rows:
    df_nearest_unit = df_nearest_unit.drop_duplicates()

    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with some object columns)
    cols_to_drop = [
        'lsoa', 'nearest_ivt_unit', 'nearest_mt_unit', 'transfer_unit',
        'nearest_msu_unit', 'lsoa11nm', 'msoa11nm'
        ]
    # Only keep cols that exist (sometimes have MSU, sometimes not):
    cols_to_drop = [c for c in cols_to_drop if c in df_msoa.columns]
    df_msoa = df_msoa.drop(cols_to_drop, axis='columns')

    # Special cases -
    # For each column, keep a copy of MSOA codes where all LSOAs
    # have zero values.
    # ----

    # Aggregate by MSOA:
    df_msoa = df_msoa.groupby('msoa11cd').mean()
    # df_msoa = df_msoa.set_index('msoa11cd')

    # Change to float16 to preserve very few significant figures:
    import numpy as np
    cols_float = df_msoa.select_dtypes([float]).columns
    for col in cols_float:
        df_msoa[col] = df_msoa[col].astype(np.float16)

    # Merge the MSOA names back in:
    df_msoa = df_msoa.reset_index()
    df_msoa = pd.merge(
        df_msoa, df_lsoa_to_msoa[['msoa11cd', 'msoa11nm']],
        left_on='msoa11cd', right_on='msoa11cd', how='left'
        )
    # Remove duplicate rows:
    df_msoa = df_msoa.drop_duplicates()
    # Merge the nearest IVT unit back in.
    df_msoa = pd.merge(
        df_msoa, df_nearest_unit,
        left_on='msoa11cd', right_on='msoa11cd', how='left'
        )
    # Set the index to (msoa_code, msoa)
    df_msoa = df_msoa.rename(columns={'msoa11cd': 'msoa_code', 'msoa11nm': 'msoa'})
    df_msoa = df_msoa.set_index(['msoa', 'msoa_code'])

    return df_msoa


def combine_results_by_occlusion_type(df_lsoa, prop_dict):
    """
    Make two new dataframes, one with all the column 1 data
    and one with all the column 2 data, and with the same column
    names. Then subtract one dataframe from the other
    and merge the results back into the main one.
    This is more efficient than calculating and creating
    each new column individually.
    """
    # Simple addition: x% of column 1 plus y% of column 2.
    df1 = pd.DataFrame(index=df_lsoa.index)  # nLVO
    df2 = pd.DataFrame(index=df_lsoa.index)  # LVO
    # Column names for these new DataFrames:
    cols = []

    # Don't combine treatment types for now
    # (no nLVO with MT data).
    scenario_list = ['drip_ship', 'mothership', 'redirect']
    treatment_list = ['ivt', 'mt', 'ivt_mt']
    outcome_list = ['mrs_0-2', 'mrs_shift', 'utility', 'utility_shift']

    for s in scenario_list:
        for t in treatment_list:
            for o in outcome_list:
                # if t == 'mt':
                #     if o in ['mrs_shift', 'utility_shift']:
                #         data_nlvo = 0.0
                #         data_exists = True
                #     else:
                #         col_nlvo = f'nlvo_no_treatment_{o}'
                #         data_nlvo = df_lsoa[col_nlvo]
                #         data_exists = True
                # else:
                # col_nlvo = f'nlvo_{s}_ivt_{o}'
                col_nlvo = f'nlvo_{s}_{t}_{o}'
                try:
                    data_nlvo = df_lsoa[col_nlvo]
                    data_exists = True
                except KeyError:
                    data_exists = False
                col_lvo = f'lvo_{s}_{t}_{o}'

                if data_exists:
                    cols.append(f'{s}_{t}_{o}')
                    df1[f'{s}_{t}_{o}'] = data_nlvo
                    df2[f'{s}_{t}_{o}'] = df_lsoa[col_lvo]
    # Create new dataframe from combining the two separate ones:
    combo_data = (
        df1 * prop_dict['nlvo'] +
        df2 * prop_dict['lvo']
    )
    # Update column names to mark them as combined:
    combo_data.columns = [f'combo_{col}' for col in cols]
    # Merge this new data into the starting dataframe:
    df_lsoa = pd.merge(df_lsoa, combo_data, left_index=True, right_index=True)
    return df_lsoa


def combine_results_by_redirection(df_lsoa, redirect_dict):
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
    outcome_list = ['mrs_0-2', 'mrs_shift', 'utility', 'utility_shift']

    for v in occlusion_list:    
        df1 = pd.DataFrame(index=df_lsoa.index)  # mothership
        df2 = pd.DataFrame(index=df_lsoa.index)  # drip-and-ship
        # Column names for these new DataFrames:
        cols = []
        prop = prop_dict[v]
        for t in treatment_list:
            for o in outcome_list:
                col_drip_ship = f'{v}_drip_ship_{t}_{o}'
                col_mothership = f'{v}_mothership_{t}_{o}'
                try:
                    df_lsoa[col_drip_ship]
                    data_exists = True
                except KeyError:
                    data_exists = False

                if data_exists:
                    cols.append(f'{t}_{o}')
                    df1[f'{t}_{o}'] = df_lsoa[col_drip_ship]
                    df2[f'{t}_{o}'] = df_lsoa[col_mothership]
        # Create new dataframe from combining the two separate ones:
        combo_data = (
            df1 * prop +
            df2 * (1.0 - prop)
        )
        # Update column names to mark them as combined:
        combo_data.columns = [f'{v}_redirect_{col}' for col in cols]

        # Merge this new data into the starting dataframe:
        df_lsoa = pd.merge(df_lsoa, combo_data,
                           left_index=True, right_index=True)
    return df_lsoa


def combine_results_by_diff(df_lsoa):
    df1 = pd.DataFrame(index=df_lsoa.index)
    df2 = pd.DataFrame(index=df_lsoa.index)
    # Column names for these new DataFrames:
    cols = []

    scenario_types = ['redirect', 'drip_ship']
    occlusion_types = ['nlvo', 'lvo', 'combo']
    treatment_types = ['ivt', 'mt', 'ivt_mt']
    outcome_types = ['mrs_0-2', 'mrs_shift', 'utility', 'utility_shift']

    for occ in occlusion_types:
        for tre in treatment_types:
            for out in outcome_types:
                # Existing column names:
                col_scen1 = f'{occ}_{scenario_types[0]}_{tre}_{out}'
                col_scen2 = f'{occ}_{scenario_types[1]}_{tre}_{out}'
                # New column name for the diff data:
                col_diff = f'{occ}_{tre}_{out}'
                # col_zero_bool = ''.join([
                #     f'{occ}_',
                #     f'diff_{scenario_types[0]}_minus_{scenario_types[1]}',
                #     f'_{tre}_{out}_iszero'
                # ])
                try:
                    data_scen1 = df_lsoa[col_scen1]
                    data_scen2 = df_lsoa[col_scen2]
                    data_exists = True
                except KeyError:
                    # This combination doesn't exist
                    # (e.g. nLVO with MT).
                    data_exists = False

                if data_exists:
                    cols.append(col_diff)
                    df1[col_diff] = data_scen1
                    df2[col_diff] = data_scen2

                    # zero_bool = (data_scen1 - data_scen2) == 0.0
                    # df_lsoa[col_zero_bool] = zero_bool
                else:
                    pass

    # Create new dataframe from combining the two separate ones:
    combo_data = df1 - df2
    # Update column names to mark them as combined:
    combo_cols = [
        ''.join([
            f"{col.split('_')[0]}_",
            f'diff_{scenario_types[0]}_minus_{scenario_types[1]}_',
            f"{'_'.join(col.split('_')[1:])}"
            ])
        for col in cols]
    combo_data.columns = combo_cols

    # Merge this new data into the starting dataframe:
    df_lsoa = pd.merge(df_lsoa, combo_data,
                       left_index=True, right_index=True)
    return df_lsoa
