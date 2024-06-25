import numpy as np
from itertools import product
import pandas as pd
from importlib_resources import files
import stroke_outcome  # for reference dists


def find_multiindex_column_names(gdf, **kwargs):
    """
    Find the full column name to match a partial column name.

    Example usage:
    find_multiindex_column_name(gdf, scenario='any', property='geometry')

    Inputs
    ------
    gdf    - GeoDataFrame.
    kwargs - in format level_name=column_name for column level names
             in the gdf column MultiIndex.

    Returns
    -------
    cols - list or str or tuple. The column name(s) matching the
           requested names in those levels.
    """
    masks = [
        gdf.columns.get_level_values(level).isin(col_list)
        for level, col_list in kwargs.items()
    ]
    mask = np.all(masks, axis=0)
    cols = gdf.columns[mask]
    if len(cols) == 1:
        cols = cols.values[0]
    elif len(cols) == 0:
        cols = ''  # Should throw up a KeyError when used to index.
    return cols


def convert_results_dict_to_multiindex_df(results_dict):
    # Most results are built up of combinations of these strings:
    column_strings = {
        'occlusion': ['lvo', 'nlvo'],
        'pathway': ['drip_ship', 'mothership', 'msu'],
        'treatment': ['ivt', 'mt', 'ivt_mt'],
        'outcome': ['mrs_0-2', 'mrs_shift', 'utility', 'utility_shift']
    }
    # So build a series of multiindex headers to split this all up.
    column_combo_tuples = [x for x in product(*column_strings.values())]

    # Prepare a DataFrame to store the reformatted data in:
    df_multi = pd.DataFrame(index=[0]).astype(float)
    # For each tuple in turn, check if it exists in the input
    # results dictionary, and if it does then add the data to the df.
    for tup in column_combo_tuples:
        key = '_'.join(tup)
        try:
            val = results_dict[key]
            df_multi[(tup)] = val
        except KeyError:
            # Probably a combo that doesn't exist like nLVO with MT.
            pass

    # Currently df_multi contains column names that are tuples.
    # Split the tuples into multiple levels with a MultiIndex:
    df_multi.columns = pd.MultiIndex.from_tuples(df_multi.columns)

    # Set the names of the column levels:
    df_multi.columns.names = column_strings.keys()
    return df_multi


def take_subset_by_column_level_values(df_results, **kwargs):
    # Pick a thing, convert rest to table:
    cols = find_multiindex_column_names(
        df_results, **kwargs)
    # Take only this subset of columns:
    df_here = df_results[cols]
    # Drop the useless headers:
    headers_to_drop = []
    # for header in df_here.columns.names:
    for header in list(kwargs.keys()):
        columns = df_here.columns.get_level_values(header)
        if len(columns.unique().values) == 1:
            headers_to_drop.append(header)
    for header in headers_to_drop:
        df_here.columns = df_here.columns.droplevel(header)
    return df_here


def convert_row_to_table(df_here, index_cols_list):
    # Give the current useless index a name:
    df_here.index.name = 'pants'

    # Check that the requested index columns exist:
    index_cols_list = [i for i in index_cols_list
                       if i in df_here.columns.names]
    df_here = df_here.T.unstack(index_cols_list).T

    # Remove useless extra index column:
    df_here.index = df_here.index.droplevel('pants')
    return df_here


def make_outline_england_wales_full():
    """Similar to stroke-maps."""
    from stroke_maps.geo import import_geojson
    # All region polygons:
    gdf_list = []
    gdf_boundaries_regions_e = import_geojson('SICBL22NM')
    gdf_list.append(gdf_boundaries_regions_e)
    gdf_boundaries_regions_w = import_geojson('LHB20NM')
    gdf_list.append(gdf_boundaries_regions_w)
    # Combine:
    gdf_boundaries_regions = pd.concat(gdf_list, axis='rows')

    gdf_boundaries_regions['ind'] = 0
    gdf_boundaries_regions = gdf_boundaries_regions.dissolve(by='ind')
    # Save:
    gdf_boundaries_regions.to_file('data/outline_england_wales.geojson')


def make_outline_england_wales():
    """Similar to stroke-maps."""
    from stroke_maps.geo import import_geojson
    import os
    from shapely.validation import make_valid  # for fixing dodgy polygons
    import shapely

    # All msoa shapes:
    gdf = import_geojson(
        'MSOA11NM',
        path_to_file=os.path.join('data', 'outline_msoa11cds.geojson')#)'MSOA_V3_reduced_simplified.geojson')
        )
    
    # Limit to England:
    mask = gdf.index.str.startswith('E')
    gdf = gdf.loc[mask].copy()
    # Make geometry valid:
    gdf['geometry'] = [
        make_valid(g) if g is not None else g
        for g in gdf['geometry'].values
        ]

    # Combine:
    gdf['ind'] = 0
    gdf = gdf.dissolve(by='ind')

    # Reduce precision
    gdf.geometry = shapely.set_precision(gdf.geometry, grid_size=0.001)

    # Save:
    gdf.to_file('data/outline_england.geojson')


def make_outline_icbs(col='icb'):
    """Similar to stroke-maps."""
    import streamlit as st
    from stroke_maps.geo import import_geojson
    import os
    from shapely.validation import make_valid  # for fixing dodgy polygons

    # # All msoa shapes:
    # gdf = import_geojson(
    #     'LSOA11NM',
    #     path_to_file=os.path.join('data', 'LSOA_V3_reduced_simplified.geojson')
    #     )
    import geopandas
    gdf = geopandas.read_file(os.path.join('data', 'LSOA_V3_reduced_simplified.geojson'))
    # gdf = geopandas.read_file(os.path.join('data', 'LSOA.geojson'))
    st.write(gdf.crs)
    gdf = gdf.set_index('LSOA11CD')
    # Limit to England:
    mask = gdf.index.str.startswith('E')
    gdf = gdf.loc[mask].copy()
    # Make geometry valid:
    gdf['geometry'] = [
        make_valid(g) if g is not None else g
        for g in gdf['geometry'].values
        ]

    # Load region info for each LSOA:
    # Relative import from package files:
    path_to_file = files('stroke_maps.data').joinpath('regions_lsoa_ew.csv')
    df_lsoa_regions = pd.read_csv(path_to_file)  # , index_col=[0, 1])

    # Load further region data linking SICBL to other regions:
    path_to_file = files('stroke_maps.data').joinpath('regions_ew.csv')
    df_regions = pd.read_csv(path_to_file)  # , index_col=[0, 1])
    # Drop columns already in df_lsoa:
    df_regions = df_regions.drop(['region', 'region_type'], axis='columns')
    df_lsoa = pd.merge(
        df_lsoa_regions, df_regions,
        left_on='region_code', right_on='region_code', how='left'
        )
    # # Link LSOA to MSOA:
    # df_lsoa_to_msoa = pd.read_csv('data/lsoa_to_msoa.csv')
    # df_lsoa = pd.merge(
    #     df_lsoa_to_msoa, df_lsoa, left_on='lsoa11cd', right_on='lsoa_code', how='right'
    # )
    gdf = pd.merge(gdf, df_lsoa, left_on='LSOA11NM', right_on='lsoa', how='left')

    # Combine:
    gdf = gdf.dissolve(by=col)

    # Save:
    gdf.to_file(f'data/outline_{col}s.geojson')


def make_outline_msoa_from_lsoa():
    """
    
    For consistency - ISDN and ICB outlines are best made from LSOA,
    so make the MSOA outlines from the same LSOA outlines
    to prevent gaps appearing where there are differences.
    """
    import streamlit as st
    from stroke_maps.geo import import_geojson
    import os
    from shapely.validation import make_valid  # for fixing dodgy polygons

    # # All msoa shapes:
    # gdf = import_geojson(
    #     'LSOA11NM',
    #     path_to_file=os.path.join('data', 'LSOA_V3_reduced_simplified.geojson')
    #     )
    import geopandas
    gdf = geopandas.read_file(os.path.join('data', 'LSOA_V3_reduced_simplified.geojson'))
    # gdf = geopandas.read_file(os.path.join('data', 'LSOA.geojson'))
    st.write(gdf.crs)
    gdf = gdf.set_index('LSOA11CD')
    # Limit to England:
    mask = gdf.index.str.startswith('E')
    gdf = gdf.loc[mask].copy()
    # Make geometry valid:
    gdf['geometry'] = [
        make_valid(g) if g is not None else g
        for g in gdf['geometry'].values
        ]

    # Link LSOA to MSOA:
    gdf = gdf.reset_index()
    df_lsoa_to_msoa = pd.read_csv('data/lsoa_to_msoa.csv')
    gdf = pd.merge(gdf, df_lsoa_to_msoa, left_on='LSOA11CD', right_on='lsoa11cd', how='left')

    # Combine:
    col = 'msoa11cd'
    gdf = gdf.dissolve(by=col)
    gdf = gdf.reset_index()

    # Only keep code and name columns:
    gdf = gdf[['msoa11cd', 'msoa11nm', 'geometry']]
    gdf = gdf.rename(columns={'msoa11cd': 'MSOA11CD', 'msoa11nm': 'MSOA11NM'})
    gdf = gdf.set_index('MSOA11CD')

    # Save:
    gdf.to_file(f'data/outline_{col}s.geojson')


def make_outline_lsoa_limit_to_england():
    """
    
    For consistency - ISDN and ICB outlines are best made from LSOA,
    so make the MSOA outlines from the same LSOA outlines
    to prevent gaps appearing where there are differences.
    """
    import streamlit as st
    from stroke_maps.geo import import_geojson
    import os
    from shapely.validation import make_valid  # for fixing dodgy polygons

    # # All msoa shapes:
    # gdf = import_geojson(
    #     'LSOA11NM',
    #     path_to_file=os.path.join('data', 'LSOA_V3_reduced_simplified.geojson')
    #     )
    import geopandas
    gdf = geopandas.read_file(os.path.join('data', 'LSOA_V3_reduced_simplified.geojson'))
    # gdf = geopandas.read_file(os.path.join('data', 'LSOA.geojson'))
    st.write(gdf.crs)
    gdf = gdf.set_index('LSOA11CD')
    # Limit to England:
    mask = gdf.index.str.startswith('E')
    gdf = gdf.loc[mask].copy()
    # Make geometry valid:
    gdf['geometry'] = [
        make_valid(g) if g is not None else g
        for g in gdf['geometry'].values
        ]

    # Save:
    gdf.to_file(f'data/outline_lsoa11cds.geojson')


def load_reference_mrs_dists():
    mrs_dists_ref, mrs_dists_ref_notes = (
        stroke_outcome.outcome_utilities.import_mrs_dists_from_file())

    nlvo_no_treatment = mrs_dists_ref.loc['no_treatment_nlvo'].values
    nlvo_no_treatment_noncum = np.diff(nlvo_no_treatment, prepend=0.0)

    lvo_no_treatment = mrs_dists_ref.loc['no_treatment_lvo'].values
    lvo_no_treatment_noncum = np.diff(lvo_no_treatment, prepend=0.0)

    dist_dict = {
        'nlvo_no_treatment': nlvo_no_treatment,
        'nlvo_no_treatment_noncum': nlvo_no_treatment_noncum,
        'lvo_no_treatment': lvo_no_treatment,
        'lvo_no_treatment_noncum': lvo_no_treatment_noncum,
    }
    return dist_dict
