"""
Draw maps of England.
"""
import streamlit as st
import numpy as np
import pandas as pd
import os

from utilities.utils import print_progress_loc, set_rerun_map


def select_map_data(df_subgroups: pd.DataFrame):
    """
    Pick which subgroup to show on the outcome maps.

    This function includes formatting so that the subgroup names are
    displayed more nicely.

    Inputs
    ------
    df_subgroups - pd.DataFrame. Index is the highlighted subgroup
                   keys, one column 'label' has nice display names.

    Returns
    -------
    key   - str. Name of the selected subgroup.
    label - str. Display name of the selected subgroup.
    """
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
        on_change=set_rerun_map
        )
    label = df_subgroups.loc[key, 'label']
    return key, label


@st.cache_data
def load_lsoa_raster_lookup():
    """
    Load the data that links LSOA to the raster map pixels.

    Cache this data for all users because it can never be altered
    by the user.

    Returns
    -------
    df_raster      - pd.DataFrame. Contains the pixel index and its
                     representative LSOA for only the pixels that are
                     mostly covered by land. The pixel index can later
                     be converted to a row and column in the array
                     since the size of the grid is determined in
                     advance.
    transform_dict - dict. Spatial information for transforming the
                     raster array to British National Grid coordinates,
                     e.g. coordinates of leftmost pixel.
                     Keys: xmin, ymin, xmax, ymax, pixel_size, width,
                     height, im_xmax, im_ymin.
                     The im keys mark the bounds of the transformed
                     pixels as opposed to the actual extent of the
                     original LSOA shapes.
    """
    path_to_raster = os.path.join('data', 'rasterise_geojson_lsoa11cd_eng.csv')
    path_to_raster_info = os.path.join(
        'data', 'rasterise_geojson_fid_eng_transform_dict.csv')
    # Load LSOA name to code lookup:
    path_to_lsoa_lookup = os.path.join('data', 'lsoa_to_msoa.csv')
    df_lsoa_lookup = pd.read_csv(path_to_lsoa_lookup)
    df_lsoa_lookup = df_lsoa_lookup[['lsoa11cd', 'lsoa11nm']].rename(
        columns={'lsoa11cd': 'LSOA11CD', 'lsoa11nm': 'LSOA11NM'})

    #
    df_raster = pd.read_csv(path_to_raster)
    df_raster = df_raster.rename(columns={'LSOA11CD': 'LSOA11CD_props'})
    transform_dict = (
        pd.read_csv(path_to_raster_info, header=None)
        .set_index(0)[1].to_dict())
    # Merge LSOA codes into time data:
    df_raster = pd.merge(
        df_raster, df_lsoa_lookup[['LSOA11NM', 'LSOA11CD']],
        left_on='LSOA11CD_majority', right_on='LSOA11CD', how='left'
        )

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


def convert_df_to_2darray(
        df_raster: pd.DataFrame, data_col: str, transform_dict: dict):
    """
    Convert a dataframe of LSOA to a raster map for plotting.

    Make a new empty 1D array. Update the pixel indices in df_raster
    for the given LSOA data. Then reshape into a 2D grid using the
    reference grid size.

    Inputs
    ------
    df_raster      - pd.DataFrame. Contains pixel indices and the
                     values that they will be set to in the array.
    data_col       - str. Column containing the data to show.
    transform_dict - dict. Contains size and bound information for the
                     raster image.

    Returns
    -------
    raster_arr_maj - np.array. A 2D grid making a map of England with
                     the given data.
    """
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
        (int(transform_dict['width']), int(transform_dict['height']))
        ).transpose()
    return raster_arr_maj


def gather_map_df(
        df_usual: pd.DataFrame,
        df_unusual: pd.DataFrame,
        df_lsoa_units_times: pd.DataFrame,
        cols_time: list,
        cols_map: list = ['utility_shift'],
        map_labels: list = ['usual_care', 'redir'],
        _log: bool = True,
        _log_loc: st.container = None,
        ):
    """
    Gather LSOA-level outcomes for usual, non-usual, and difference.

    For OPTIMIST, df_unusual is mix of usual care, redirection
    approved, and redirection rejected. For MUSTER it is a mix of usual
    care, MSU with IVT, and MSU with no IVT.

    Inputs
    ------
    df_usual            - pd.DataFrame. Outcomes of usual scenario.
    df_unusual          - pd.DataFrame. Outcomes of non-usual scenario.
    df_lsoa_units_times - pd.DataFrame. Treatment times for each LSOA.
    cols_time           - list. Treatment time columns. Should match
                          index columns of df_usual and df_redir.
    cols_map            - list. Which data columns to gather.
    map_labels          - list. How to label the usual and non-usual
                          data in the resulting columns.
    _log                - bool. Whether to print log message.
    _log_loc            - st.container or None. Where to print log
                          message.

    Returns
    -------
    df_results - pd.DataFrame. LSOA-level outcome data for showing on
                 maps.
    """
    # Gather the LSOA names and treatment times:
    cols_to_keep = ['LSOA']
    df_results = df_lsoa_units_times[cols_to_keep + cols_time].copy()
    df_results = df_results.set_index(cols_time).copy()
    # Merge in results for usual care...
    df_results = pd.merge(df_results, df_usual[cols_map], left_index=True,
                          right_index=True, how='left')
    rename_dict = dict([(c, f'{c}_{map_labels[0]}') for c in cols_map])
    df_results = df_results.rename(columns=rename_dict)
    # ... and non-usual care (redirection or MSU):
    df_results = pd.merge(df_results, df_unusual[cols_map], left_index=True,
                          right_index=True, how='left')
    rename_dict = dict([(c, f'{c}_{map_labels[1]}') for c in cols_map])
    df_results = df_results.rename(columns=rename_dict)
    # Only keep LSOA names and outcomes:
    df_results = df_results.reset_index()
    df_results = df_results.drop(cols_time, axis='columns')
    df_results = df_results.set_index('LSOA')
    # Calculate difference between redir scenario and usual care:
    for col in cols_map:
        df_results[f'{col}_{map_labels[1]}_minus_{map_labels[0]}'] = (
            df_results[f'{col}_{map_labels[1]}'] -
            df_results[f'{col}_{map_labels[0]}']
        )

    if _log:
        p = 'Gathered data for maps.'
        print_progress_loc(p, _log_loc)
    return df_results


def gather_map_data(
        df_raster: pd.DataFrame,
        transform_dict: dict,
        df_maps: pd.DataFrame,
        cols: list,
        _log: bool = True,
        _log_loc: st.container = None,
        ):
    """
    Wrapper for convert_df_to_2darray() to create multiple map arrays.

    Inputs
    ------
    df_raster      - pd.DataFrame. Contains pixel indices and the
                     values that they will be set to in the array.
    transform_dict - dict. Contains size and bound information for the
                     raster image.
    df_maps        - pd.DataFrame. Contains LSOA-level data for the
                     maps.
    cols           - list. Columns of data to turn into maps.
    _log           - bool. Whether to print log message.
    _log_loc       - st.container or None. Where to print log message.

    Returns
    -------
    arrs - list. List of map arrays for the chosen columns of data.
    """
    df_raster = df_raster.copy()
    df_raster = pd.merge(df_raster, df_maps.reset_index(),
                         left_on='LSOA11NM', right_on='LSOA', how='left')
    arrs = []
    for c in cols:
        arrs.append(convert_df_to_2darray(df_raster, c, transform_dict))
    if _log:
        p = 'Created map array.'
        print_progress_loc(p, _log_loc)
    return arrs


def gather_map_arrays(
        df_usual: pd.DataFrame,
        df_unusual: pd.DataFrame,
        df_lsoa_units_times: pd.DataFrame,
        df_raster: pd.DataFrame,
        transform_dict: dict,
        map_labels: list = ['usual_care', 'redir'],
        scenarios: list = ['usual_care', 'redirection_approved',
                           'redirection_rejected'],
        col_map: str = 'utility_shift',
        _log_loc: st.container = None
        ):
    """
    Wrapper for gather map into df then reshape to arrays.

    Inputs
    ------
    df_usual            - pd.DataFrame. Outcomes of usual scenario.
    df_unusual          - pd.DataFrame. Outcomes of non-usual scenario.
    df_lsoa_units_times - pd.DataFrame. Treatment times for each LSOA.
    df_raster           - pd.DataFrame. Contains pixel indices and the
                          values that they will be set to in the array.
    transform_dict      - dict. Contains size and bound information for
                          the raster image.
    map_labels          - list. How to label the two scenarios for the
                          map data.
    scenarios           - list. Use to build up a list of columns
                          containing treatment times.
    col_map             - str. The source data type for the maps.
    _log_loc            - st.container or None. Where to print log
                          message.

    Returns
    -------
    arr_dict  - dict. Arrays for each of the maps.
    vlim_dict - dict. Data limits for each of the maps.
    """
    # Build up the columns containing treatment times:
    treats = ['ivt', 'mt']
    cols_time = [f'{s}_{t}' for s in scenarios for t in treats]
    # Special case for muster:
    if 'msu_no_ivt' in scenarios:
        cols_time.remove('msu_no_ivt_ivt')

    df_maps = gather_map_df(
        df_usual,
        df_unusual,
        df_lsoa_units_times,
        cols_time,
        cols_map=[col_map],
        map_labels=map_labels,
        _log_loc=_log_loc
        )

    # Gather the data in these columns...
    cols = [
        f'{col_map}_{map_labels[0]}',
        f'{col_map}_{map_labels[1]}_minus_{map_labels[0]}',
        ]
    # ... and rename them to these columns:
    cols_out = [c.replace(f'{col_map}_', '') for c in cols]

    arrs = gather_map_data(
        df_raster, transform_dict, df_maps, cols, _log_loc=_log_loc
        )
    arr_dict = dict(zip(cols_out, arrs))

    # Store a copy of the min and max values in the data:
    vlim_dict = {}
    for i, c in enumerate(cols):
        vlim_dict[cols_out[i]] = {}
        vlim_dict[cols_out[i]]['data_max'] = df_maps[c].max()
        vlim_dict[cols_out[i]]['data_min'] = df_maps[c].min()
    return arr_dict, vlim_dict


def gather_pop_map(df_raster: pd.DataFrame, transform_dict: dict):
    """
    Create the array for the population map.

    Inputs
    ------
    df_raster      - pd.DataFrame. Contains pixel indices and the
                     values that they will be set to in the array.
    transform_dict - dict. Contains size and bound information for
                     the raster image.

    Returns
    -------
    arrs[0]   - np.array. Data in the map.
    vlim_dict - dict. Data limits.
    """
    # For population map. Load in LSOA-level demographic data:
    df_demog = pd.read_csv(os.path.join('data', 'LSOA_popdens.csv'))
    # Store data limits:
    vlim_dict = {'pop': {}}
    vlim_dict['pop']['data_max'] = df_demog['population_density'].max()
    vlim_dict['pop']['data_min'] = df_demog['population_density'].min()
    # Gather the array to plot:
    arrs = gather_map_data(
        df_raster,
        transform_dict,
        df_demog,
        ['population_density'],
        _log=False
        )
    return arrs[0], vlim_dict
