
# ----- Imports -----
import streamlit as st
from importlib_resources import files
import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW  # for mRS dist stats

# For running outcomes:
from classes.geography_processing import Geoprocessing
from classes.model import Model
from classes.scenario import Scenario

# Custom functions:
from utilities.utils import load_reference_mrs_dists
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

    # No-treatment data:
    dist_dict = load_reference_mrs_dists()

    # Copy stroke unit names over. Currently has only postcodes.
    for col in ['nearest_ivt_unit', 'nearest_mt_unit', 'transfer_unit', 'nearest_msu_unit']:
        if col in df_lsoa.columns:
            df_lsoa = pd.merge(
                df_lsoa, df_unit_services['stroke_team'],
                left_on=col, right_index=True, how='left'
                )
            df_lsoa = df_lsoa.rename(columns={'stroke_team': f'{col}_name'})
            # Reorder columns so name appears next to postcode.
            i = df_lsoa.columns.tolist().index(col)
            df_lsoa = df_lsoa[[*df_lsoa.columns[:i], f'{col}_name', *df_lsoa.columns[i:-1]]]

    # Make a copy of nLVO IVT results for nLVO IVT+MT results:
    cols_ivt_mt = [c for c in df_lsoa.columns if 'ivt_mt' in c]
    for col in cols_ivt_mt:
        # Find the equivalent column for nLVO:
        col_nlvo = col.replace('lvo', 'nlvo')
        # Find the equivalent column for ivt-only:
        col_ivt = col_nlvo.replace('ivt_mt', 'ivt')
        # Copy over the data:
        df_lsoa[col_nlvo] = df_lsoa[col_ivt]
    # Set the nLVO MT results to be the nLVO no-treatment results:
    cols_mt = [c for c in df_lsoa.columns if (('_mt_' in c) & ('_ivt_' not in c))]
    for col in cols_mt:
        # Find the equivalent column for nLVO:
        col_nlvo = col.replace('lvo', 'nlvo')
        if (('utility_shift' in col_nlvo) | ('mrs_shift' in col_nlvo)):
            # No change from non-treatment.
            df_lsoa[col_nlvo] = 0.0
        elif 'utility' in col_nlvo:
            df_lsoa[col_nlvo] = df_lsoa['nlvo_no_treatment_utility']
        elif 'mrs_0-2' in col_nlvo:
            df_lsoa[col_nlvo] = df_lsoa['nlvo_no_treatment_mrs_0-2']

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
        'utility': 3,
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

    # Make a copy of nLVO IVT results for nLVO IVT+MT results:
    cols_ivt_mt = [c for c in df_mrs.columns if 'ivt_mt' in c]
    # Find the col names without _mrs at the end:
    cols_ivt_mt_prefixes = sorted(list(set(['_'.join(c.split('_')[:-1]) for c in cols_ivt_mt])))
    for col in cols_ivt_mt_prefixes:
        # Find the equivalent column for nLVO:
        col_nlvo = col.replace('lvo', 'nlvo')
        # Find the equivalent column for ivt-only:
        col_ivt = col_nlvo.replace('ivt_mt', 'ivt')

        # Add mRS suffixes back on:
        cols_nlvo = [f'{col_nlvo}_{i}' for i in range(7)]
        cols_ivt = [f'{col_ivt}_{i}' for i in range(7)]
        # Copy over the data:
        df_mrs[cols_nlvo] = df_mrs[cols_ivt]
    # Set the nLVO MT results to be the nLVO no-treatment results:
    cols_mt = [c for c in df_mrs.columns if (('_mt_' in c) & ('_ivt_' not in c))]
    # Find the col names without _mrs at the end:
    cols_mt_prefixes = sorted(list(set(['_'.join(c.split('_')[:-1]) for c in cols_mt])))
    for col in cols_mt_prefixes:
        # Find the equivalent column for nLVO:
        col_nlvo = col.replace('lvo', 'nlvo')
        # Add the suffixes back in:
        cols_nlvo = [f'{col_nlvo}_{i}' for i in range(7)]
        if 'noncum' in col_nlvo:
            dist = dist_dict['nlvo_no_treatment_noncum']
        else:
            dist = dist_dict['nlvo_no_treatment']
        # Copy over the data:
        df_mrs[cols_nlvo] = dist

    return df_lsoa, df_mrs


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

    # Round the values.
    # Outcomes:
    cols_outcome = [c for c in df_icb.columns if (
        (c.endswith('utility')) | (c.endswith('utility_shift')) |
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

    # Round the values.
    # Outcomes:
    cols_outcome = [c for c in df_isdn.columns if (
        (c.endswith('utility')) | (c.endswith('utility_shift')) |
        (c.endswith('mrs_0-2')) | (c.endswith('mrs_shift'))
        )]
    for col in cols_outcome:
        df_isdn[col] = np.round(df_isdn[col], 3)

    # Times:
    cols_time = [c for c in df_isdn.columns if 'time' in c]
    for col in cols_time:
        df_isdn[col] = np.round(df_isdn[col], 2)
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
        'isdn',
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
        (c.endswith('utility')) | (c.endswith('utility_shift')) |
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

    # cols_mrs_dist = [col for col in df_lsoa.columns if
    #                  (('mrs_dist' in col) & (col.endswith('noncum')))]

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

    # Merge in nearest IVT units:
    df_lsoa = pd.merge(df_lsoa, nearest_ivt_units, left_on='lsoa', right_index=True)

    # # Replace some zeros with NaN:
    # mask = df_lsoa['transfer_required']
    # df_lsoa.loc[~mask, 'transfer_time'] = pd.NA

    # # Scale the mRS distributions by admissions:
    # for col in cols_mrs_dist:
    #     df_lsoa[f'{col}_by_admissions'] = df_lsoa['Admissions'] * df_lsoa[col].apply(lambda x: np.array(x))

    df = group_mrs_dists_by_column(df_lsoa, **kwargs)
    # df_national = group_mrs_dists_by_column(df_lsoa)
    # df_nearest_ivt = group_mrs_dists_by_column(df_lsoa, 'nearest_ivt_unit_name')
    # df_icb = group_mrs_dists_by_column(df_lsoa, 'icb')
    # df_isdn = group_mrs_dists_by_column(df_lsoa, 'isdn')

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
    # cols_cumulative = [c.replace('_noncum_', '') for c in cols_to_combine]
    # cols_std = [c.replsace('_noncum_', '_noncum_std_') for c in cols_to_combine]

    # cols_for_df = ['Admissions']
    # for i in range(len(cols_to_combine)):
    #     cols_for_df += [cols_to_combine[i], cols_cumulative[i], cols_std[i]]

    df_by_region = pd.DataFrame() # columns=cols_for_df)
    for val in col_vals:
        if use_all:
            mask = [True] * len(df_lsoa)
        else:
            mask = (df_lsoa[col_region] == val)
        df_region = df_lsoa.loc[mask, ['Admissions'] + cols_to_combine].copy()

        # Admissions:
        df_by_region.loc[val, 'Admissions'] = df_region['Admissions'].sum()

        # Remove repeats from all the mRS bands:
        cols_each_scen = sorted(list(set(['_'.join(c.split('_')[:-1]) for c in cols_to_combine])))
        # st.write(df_lsoa.columns)
        # st.write(cols_each_scen)

        # Stats:
        for c, col in enumerate(cols_each_scen):
            cols_here = [c for c in df_region.columns if c.startswith(col)]
            cols_cumulative = [c.replace('_noncum', '') for c in cols_here]
            cols_std = [c.replace('_noncum_', '_noncum_std_') for c in cols_here]

            # Split list of values into one column per mRS band
            # and keep one row per LSOA.
            vals = df_region[cols_here].copy()
            # # Make one long column of all mRS values (no more lists):
            # # Need to explicitly turn them into floats here
            # # otherwise np.std() will throw an error.
            # vals = vals.explode().astype(float)
            # # Reshape into one row per LSOA:
            # vals = vals.values.reshape(len(df_region), 7)

            # Create stats from these data:
            weighted_stats = DescrStatsW(vals, weights=df_region['Admissions'], ddof=0)
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

            # Sanity check:
            # st.write(np.sum(means))
            # These mean values do sum to 1 (+/- floating point error).


        # # First calculate means.
        # # Add all mRS distributions together...
        # df_here = df_region.copy().sum(axis='rows')
        # # ... and divide by the total number of admissions:
        # df_here[cols_to_combine] = (
        #     df_here[cols_to_combine] / df_here['Admissions'].sum())
        # df_by_region.loc[val, cols_to_combine] = df_here


        # # Now calculate standard deviations.
        # for c, col in enumerate(cols_to_combine):
        #     # Only take one column of mRS dists at a time.
        #     df_here = df_region[col].copy()
        #     # Make one long column of all mRS values (no more lists):
        #     # Need to explicitly turn them into floats here
        #     # otherwise np.std() will throw an error.
        #     df_here = df_here.explode().astype(float)
        #     # Reshape into one row per LSOA:
        #     vals_here = df_here.values.reshape(len(df_region), 7)
        #     # Take the standard deviations:
        #     vals_here = np.std(vals_here, axis=0)
        #     # Recombine into a single column:
        #     vals_here = vals_here.tolist()
        #     # Place into the results:
        #     df_by_region.at[val, cols_std[c]] = vals_here

    # # Rename columns:
    # dict_rename = dict(zip(cols_to_combine, [c.split('_by_admissions')[0] for c in cols_to_combine]))
    # df_by_region = df_by_region.rename(columns=dict_rename)

    # Round the values.
    # Outcomes:
    cols_to_round = [c for c in df_by_region.columns if 'dist' in c]
    for col in cols_to_round:
        df_by_region[col] = np.round(df_by_region[col], 3)
    return df_by_region


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
    # TO DO - currently this is causing MSOAs to be assigned to multiple nearest units - sort it out please.
    df_nearest_unit = df_nearest_unit.drop_duplicates()

    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with some object columns)
    cols_to_drop = [
        'lsoa',
        'nearest_ivt_unit',
        'nearest_mt_unit',
        'transfer_unit',
        'nearest_msu_unit',
        'lsoa11nm',
        'msoa11nm',
        'nearest_ivt_unit_name',
        'nearest_mt_unit_name',
        'transfer_unit_name',
        'nearest_msu_unit_name',
        ]
    # Only keep cols that exist (sometimes have MSU, sometimes not):
    cols_to_drop = [c for c in cols_to_drop if c in df_msoa.columns]
    df_msoa = df_msoa.drop(cols_to_drop, axis='columns')

    # Aggregate by MSOA:
    df_msoa = df_msoa.groupby('msoa11cd').mean()
    # df_msoa = df_msoa.set_index('msoa11cd')

    # Convert back to float:
    for col in df_msoa.columns:
        try:
            df_msoa[col] = pd.to_numeric(df_msoa[col])
        except ValueError:
            pass

    # # Change to float16 to preserve very few significant figures:
    # import numpy as np
    # cols_float = df_msoa.select_dtypes([float]).columns
    # for col in cols_float:
    #     df_msoa[col] = df_msoa[col].astype(np.float16)

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


# ###########################
# ##### AVERAGE RESULTS #####
# ###########################
def combine_results_by_occlusion_type(df_lsoa, prop_dict, combine_mrs_dists=False):
    """
    Make two new dataframes, one with all the column 1 data
    and one with all the column 2 data, and with the same column
    names. Then subtract one dataframe from the other
    and merge the results back into the main one.
    This is more efficient than calculating and creating
    each new column individually.
    """
    # Simple addition: x% of column 1 plus y% of column 2.
    # Column names for these new DataFrames:
    cols_combo = []
    cols_nlvo = []
    cols_lvo = []

    # Don't combine treatment types for now
    # (no nLVO with MT data).
    scenario_list = ['drip_ship', 'mothership', 'redirect']
    treatment_list = ['ivt', 'mt', 'ivt_mt']
    if combine_mrs_dists:
        outcome_list = ['mrs_dists_noncum']  # not cumulative
    else:
        outcome_list = ['mrs_0-2', 'mrs_shift', 'utility', 'utility_shift']

    for s in scenario_list:
        for t in treatment_list:
            for o in outcome_list:
                if combine_mrs_dists:
                    cols_mrs_lvo = [f'lvo_{s}_{t}_{o}_{i}' for i in range(7)]

                    if t == 'ivt_mt':
                        cols_mrs_nlvo = [f'nlvo_{s}_ivt_{o}_{i}' for i in range(7)]
                    else:
                        cols_mrs_nlvo = [f'nlvo_{s}_{t}_{o}_{i}' for i in range(7)]
                    try:
                        data_nlvo = df_lsoa[cols_mrs_nlvo]
                        data_exists = True
                    except KeyError:
                        data_exists = False

                    if data_exists:
                        cols_here = [f'{s}_{t}_{o}_{i}' for i in range(7)]
                        cols_combo += cols_here
                        cols_nlvo += cols_mrs_nlvo
                        cols_lvo += cols_mrs_lvo

                else:

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
                    col_lvo = f'lvo_{s}_{t}_{o}'
                    try:
                        data_nlvo = df_lsoa[col_nlvo]
                        data_exists = True
                    except KeyError:
                        data_exists = False

                    if data_exists:
                        cols_combo.append(f'{s}_{t}_{o}')
                        cols_nlvo.append(col_nlvo)
                        cols_lvo.append(col_lvo)

    # Pick out the data from the original dataframe:
    df1 = df_lsoa[cols_nlvo].copy()
    df2 = df_lsoa[cols_lvo].copy()
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
        prefixes = sorted(list(set(['_'.join(c.split('_')[:-1]) for c in cols_combo])))
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
    combo_data.columns = [f'combo_{col}' for col in combo_data.columns]
    # Merge this new data into the starting dataframe:
    df_lsoa = pd.merge(df_lsoa, combo_data, left_index=True, right_index=True)

    return df_lsoa


def combine_results_by_redirection(df_lsoa, redirect_dict, combine_mrs_dists=False):
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
        outcome_list = ['mrs_0-2', 'mrs_shift', 'utility', 'utility_shift']

    for v in occlusion_list:
        # Column names for these new DataFrames:
        cols_combo = []
        cols_drip_ship = []
        cols_mothership = []
        prop = prop_dict[v]
        for t in treatment_list:
            for o in outcome_list:
                if combine_mrs_dists:
                    cols_mrs_drip_ship = [f'{v}_drip_ship_{t}_{o}_{i}' for i in range(7)]
                    cols_mrs_mothership = [f'{v}_mothership_{t}_{o}_{i}' for i in range(7)]
                    try:
                        data_nlvo = df_lsoa[cols_mrs_drip_ship]
                        data_exists = True
                    except KeyError:
                        data_exists = False

                    if data_exists:
                        cols_here = [f'{v}_redirect_{t}_{o}_{i}' for i in range(7)]
                        cols_combo += cols_here
                        cols_drip_ship += cols_mrs_drip_ship
                        cols_mothership += cols_mrs_mothership
                else:
                    col_drip_ship = f'{v}_drip_ship_{t}_{o}'
                    col_mothership = f'{v}_mothership_{t}_{o}'
                    try:
                        df_lsoa[col_drip_ship]
                        data_exists = True
                    except KeyError:
                        data_exists = False

                    if data_exists:
                        cols_combo.append(f'{v}_redirect_{t}_{o}')
                        cols_drip_ship.append(col_drip_ship)
                        cols_mothership.append(col_mothership)

        # Pick out the data from the original dataframe:
        df1 = df_lsoa[cols_drip_ship].copy()
        df2 = df_lsoa[cols_mothership].copy()
        # Rename columns so they match:
        df1.columns = cols_combo
        df2.columns = cols_combo

        # Create new dataframe from combining the two separate ones:
        combo_data = (
            df1 * prop +
            df2 * (1.0 - prop)
        )

        if combine_mrs_dists:
            # Make cumulative probabilities:
            prefixes = sorted(list(set(['_'.join(c.split('_')[:-1]) for c in cols_combo])))
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
        # combo_data.columns = [f'{v}_redirect_{col}' for col in cols_combo]

        # Merge this new data into the starting dataframe:
        df_lsoa = pd.merge(df_lsoa, combo_data,
                           left_index=True, right_index=True)

    return df_lsoa


def combine_results_by_diff(df_lsoa, combine_mrs_dists=False):
    df1 = pd.DataFrame(index=df_lsoa.index)
    df2 = pd.DataFrame(index=df_lsoa.index)
    # Column names for these new DataFrames:
    cols_combo = []
    cols_scen1 = []
    cols_scen2 = []

    scenario_types = ['redirect', 'drip_ship']
    occlusion_types = ['nlvo', 'lvo', 'combo']
    treatment_types = ['ivt', 'mt', 'ivt_mt']
    if combine_mrs_dists:
        outcome_types = ['mrs_dists_noncum']  # not cumulative
    else:
        outcome_types = ['mrs_0-2', 'mrs_shift', 'utility', 'utility_shift']

    for occ in occlusion_types:
        for tre in treatment_types:
            for out in outcome_types:
                if combine_mrs_dists:
                    cols_mrs_scen1 = [f'{occ}_{scenario_types[0]}_{tre}_{out}_{i}' for i in range(7)]
                    cols_mrs_scen2 = [f'{occ}_{scenario_types[1]}_{tre}_{out}_{i}' for i in range(7)]
                    try:
                        data_nlvo = df_lsoa[cols_mrs_scen1]
                        data_exists = True
                    except KeyError:
                        data_exists = False

                    if data_exists:
                        cols_here = [f'{occ}_{tre}_{out}_{i}' for i in range(7)]
                        cols_combo += cols_here
                        cols_scen1 += cols_mrs_scen1
                        cols_scen2 += cols_mrs_scen2
                else:
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
                        cols_combo.append(col_diff)
                        cols_scen1.append(col_scen1)
                        cols_scen2.append(col_scen2)

                        # zero_bool = (data_scen1 - data_scen2) == 0.0
                        # df_lsoa[col_zero_bool] = zero_bool
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
        prefixes = sorted(list(set(['_'.join(c.split('_')[:-1]) for c in cols_combo])))
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
            f"{col.split('_')[0]}_",
            f'diff_{scenario_types[0]}_minus_{scenario_types[1]}_',
            f"{'_'.join(col.split('_')[1:])}"
            ])
        for col in combo_data.columns]
    combo_data.columns = combo_cols

    # Merge this new data into the starting dataframe:
    df_lsoa = pd.merge(df_lsoa, combo_data,
                       left_index=True, right_index=True)

    return df_lsoa
