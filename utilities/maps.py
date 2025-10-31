"""
Draw maps of England.
"""
import streamlit as st
import numpy as np
import pandas as pd
import os

from utilities.utils import print_progress_loc
from utilities.regions import load_lsoa_demog


def draw_units_map(df_unit_services, df_lsoa_units_times):
    pass


def select_map_data(df_subgroups):
    # Pick an outcome population from the already-selected subgroups.
    dict_labels = dict(df_subgroups['label'])

    def f(label):
        """Display layer with nice name instead of key."""
        return dict_labels[label]
    # Pick a layer to use for calculating population results:
    key = st.selectbox(
        'Choose a population to show on the maps.',
        options=df_subgroups.index,
        format_func=f,
        index=0,
        )
    label = df_subgroups.loc[key, 'label']
    return key, label


@st.cache_data
def load_lsoa_raster_lookup():
    path_to_raster = os.path.join('data', 'rasterise_geojson_lsoa11cd_eng.csv')
    path_to_raster_info = os.path.join('data', 'rasterise_geojson_fid_eng_transform_dict.csv')
    # Load LSOA name to code lookup:
    path_to_lsoa_lookup = os.path.join('data', 'lsoa_to_msoa.csv')
    df_lsoa_lookup = pd.read_csv(path_to_lsoa_lookup)
    df_lsoa_lookup = df_lsoa_lookup[['lsoa11cd', 'lsoa11nm']].rename(
        columns={'lsoa11cd': 'LSOA11CD', 'lsoa11nm': 'LSOA11NM'})

    #
    df_raster = pd.read_csv(path_to_raster)
    df_raster = df_raster.rename(columns={'LSOA11CD': 'LSOA11CD_props'})
    transform_dict = pd.read_csv(path_to_raster_info, header=None).set_index(0)[1].to_dict()
    # Merge LSOA codes into time data:
    df_raster = pd.merge(df_raster, df_lsoa_lookup[['LSOA11NM', 'LSOA11CD']], left_on='LSOA11CD_majority', right_on='LSOA11CD', how='left')

    # Manually remove Isles of Scilly:
    df_raster = df_raster.loc[~(df_raster['LSOA11CD_majority'] == 'E01019077')]
    # Calculate how much of each pixel contains land (as opposed to sea
    # or other countries):
    area_each_pixel = transform_dict['pixel_size']**2.0
    df_raster['total_area_covered'] = df_raster['area_total'] / area_each_pixel
    # Pick out pixels that mostly contain no land:
    mask_sea = (df_raster['total_area_covered'] <= (1.0 / 3.0))
    # Remove the data here so that they're not shown.
    df_raster = df_raster.loc[~mask_sea]
    return df_raster, transform_dict


def convert_df_to_2darray(df_raster, data_col, transform_dict):
    # Make a 1D array with all pixels, not just valid ones:
    raster_arr_maj = np.full(
        int(transform_dict['height'] * transform_dict['width']), np.NaN)
    # Update the values of valid pixels:
    if data_col is None:
        # This should happen when nLVO + MT is selected.
        pass
    else:
        raster_arr_maj[df_raster['i'].values] = df_raster[data_col].values
    # Reshape into rectangle:
    raster_arr_maj = raster_arr_maj.reshape(
        (int(transform_dict['width']), int(transform_dict['height']))).transpose()
    return raster_arr_maj


def gather_map_df(
        df_usual,
        df_redir,
        df_lsoa_units_times,
        cols_map=['utility_shift'],
        _log=True, _log_loc=None,
        ):
    """
    Gather LSOA-level outcomes for usual care, redir, and diff.
    """
    scenarios = ['usual_care', 'redirection_approved', 'redirection_rejected']
    treats = ['ivt', 'mt']
    cols_time = [f'{s}_{t}' for s in scenarios for t in treats]

    cols_to_keep = ['LSOA']

    df_results = df_lsoa_units_times[cols_to_keep + cols_time].copy()
    df_results = df_results.set_index(cols_time).copy()

    df_results = pd.merge(df_results, df_usual[cols_map], left_index=True,
                          right_index=True, how='left')
    rename_dict = dict([(c, f'{c}_usual_care') for c in cols_map])
    df_results = df_results.rename(columns=rename_dict)
    df_results = pd.merge(df_results, df_redir[cols_map], left_index=True,
                          right_index=True, how='left')
    rename_dict = dict([(c, f'{c}_redir') for c in cols_map])
    df_results = df_results.rename(columns=rename_dict)

    df_results = df_results.reset_index()
    df_results = df_results.drop(cols_time, axis='columns')
    df_results = df_results.set_index('LSOA')

    for col in cols_map:
        df_results[f'{col}_redir_minus_usual_care'] = (
            df_results[f'{col}_redir'] -
            df_results[f'{col}_usual_care']
        )

    # For population map. Load in LSOA-level demographic data:
    df_demog = load_lsoa_demog()
    df_results = pd.merge(df_results, df_demog.set_index('LSOA'),
                          left_index=True, right_index=True, how='left')

    if _log:
        p = 'Gathered data for maps.'
        print_progress_loc(p, _log_loc)
    return df_results


def gather_map_data(
        df_maps,
        column_colours,
        column_colours_diff,
        column_pop,
        _log=True, _log_loc=None,
        ):
    # Load LSOA geometry:
    df_raster, transform_dict = load_lsoa_raster_lookup()

    df_raster = pd.merge(df_raster, df_maps.reset_index(),
                         left_on='LSOA11NM', right_on='LSOA', how='left')

    burned_lhs = convert_df_to_2darray(df_raster, column_colours,
                                       transform_dict)
    burned_rhs = convert_df_to_2darray(df_raster, column_colours_diff,
                                       transform_dict)
    burned_pop = convert_df_to_2darray(df_raster, column_pop,
                                       transform_dict)
    if _log:
        p = 'Created map arrays.'
        print_progress_loc(p, _log_loc)
    return burned_lhs, burned_rhs, burned_pop


def gather_map_arrays(df_usual, df_redir, df_lsoa_units_times, _log_loc=None):
    """Wrapper for gather map into df then reshape to arrays."""
    df_maps = gather_map_df(
        df_usual,
        df_redir,
        df_lsoa_units_times,
        _log_loc=_log_loc
        )

    map_arrs = gather_map_data(
        df_maps,
        'utility_shift_usual_care',
        'utility_shift_redir_minus_usual_care',
        'population_density',
        _log_loc=_log_loc
        )
    return map_arrs


def plot_maps(maps_arrs):
    pass
