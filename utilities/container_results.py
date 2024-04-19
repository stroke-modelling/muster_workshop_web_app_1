"""
All of the content for the Results section.
"""
import numpy as np
import streamlit as st

# For running outcomes:
from classes.geography_processing import Geoprocessing
from classes.model import Model
from classes.scenario import Scenario


def split_results_dict_by_pathway(results_dict):
    # Split results by pathway:
    results_drip_ship = dict([(key, value) for key, value in results_dict.items()
                            if 'drip_ship' in key])
    results_mothership = dict([(key, value) for key, value in results_dict.items()
                            if 'mothership' in key])
    results_msu = dict([(key, value) for key, value in results_dict.items()
                        if 'msu' in key])
    # Everything else:
    results_all_keys = results_dict.keys()
    results_all_keys -= results_drip_ship.keys()
    results_all_keys -= results_mothership.keys()
    results_all_keys -= results_msu.keys()
    results_all = {key: results_dict[key] for key in results_all_keys}

    # Remove the pathway name from the dict keys.
    # Assume that keys never end with pathway name so there is always
    # a 'pathway_' in the key somewhere.
    def rename_keys(dict, str_to_replace, replace_with=''):
        # Keep a copy of the original keys to prevent RuntimeError
        # when the keys are updated during the loop.
        keys = list(dict.keys())
        for key in keys:
            new_key = key.replace(str_to_replace, replace_with)
            dict[new_key] = dict.pop(key)
        return dict

    results_drip_ship = rename_keys(results_drip_ship, 'drip_ship_')
    results_mothership = rename_keys(results_mothership, 'mothership_')
    results_msu = rename_keys(results_msu, 'msu_')

    new_dict = {
        'all': results_all,
        'drip_ship': results_drip_ship,
        'mothership': results_mothership,
        'msu': results_msu,
    }

    return new_dict


def make_multiindex_stroke_type(df_columns, split_list):
    # Start with 'lvo_ivt_mt' because 'lvo_ivt' starts the same,
    # checking first for 'lvo_ivt' would also pick up 'lvo_ivt_mt'.
    new_cols = np.array([[''] * len(df_columns), df_columns])
    for s in split_list:
        cols = np.array(new_cols[1])
        cols_split = [c.split(s) for c in cols]
        for c, col_list in enumerate(cols_split):
            if len(col_list) > 1:
                new_col_name = ''.join(col_list)
                # Remove final underscores:
                if s[-1] == '_':
                    s = s[:-1] 
                if new_col_name[-1] == '_':
                    new_col_name = new_col_name[:-1]
                # Store in the list:
                new_cols[0][c] = s
                new_cols[1][c] = new_col_name
    return new_cols


def make_column_style_dict(cols, format='%.3f'):
    style_dict = dict([
        [col, st.column_config.NumberColumn(format=format)]
        for col in cols]
        )
    return style_dict


# @st.cache_data
def make_outcomes(input_dict, df_unit_services):
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

    return df_lsoa
