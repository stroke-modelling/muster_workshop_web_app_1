import numpy as np
from itertools import product
import pandas as pd


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
