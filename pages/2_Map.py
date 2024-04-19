"""
MUSTER results display app.

Because a long app quickly gets out of hand,
try to keep this document to mostly direct calls to streamlit to write
or display stuff. Use functions in other files to create and
organise the stuff to be shown. In this example, most of the work is
done in functions stored in files named container_(something).py
"""
# ----- Imports -----
import streamlit as st
from importlib_resources import files
import pandas as pd

from stroke_maps.geo import check_scenario_level
# Custom functions:
import utilities.utils as utils
import utilities.maps as maps
# Containers:
import utilities.container_inputs as inputs
import utilities.container_results as results


# @st.cache_data
def main_calculations(input_dict, df_unit_services):
    # Run the outcomes with the selected pathway:
    df_lsoa = results.make_outcomes(input_dict, df_unit_services)

    # TO DO - the results df contains a mix of scenarios
    # (drip and ship, mothership, msu) in the column names.
    # Pull them out and put them into 'scenario' header.
    # Also need to do something with separate nlvo, lvo, treatment types
    # because current setup just wants some averaged added utility outcome
    # rather than split by stroke type.

    # --- MSOAs for geography ---
    df_msoa = inputs.convert_lsoa_to_msoa_results(df_lsoa)

    # Check whether the input DataFrames have a 'scenario' column level.
    # This is required for talking to stroke-maps package.
    # If not, add one now with a placeholder scenario name.
    df_msoa = check_scenario_level(df_msoa)

    # Merge outcome and geography:
    gdf_boundaries_msoa = maps._load_geometry_msoa(df_msoa)

    # --- LSOAs for grouping results ---
    # Merge in other region info.

    # Load region info for each LSOA:
    # Relative import from package files:
    path_to_file = files('stroke_maps.data').joinpath('regions_lsoa_ew.csv')
    df_lsoa_regions = pd.read_csv(path_to_file)  # , index_col=[0, 1])
    df_lsoa = pd.merge(df_lsoa, df_lsoa_regions, left_on='lsoa', right_on='lsoa', how='left')

    # Load further region data linking SICBL to other regions:
    path_to_file = files('stroke_maps.data').joinpath('regions_ew.csv')
    df_regions = pd.read_csv(path_to_file)  # , index_col=[0, 1])
    # Drop columns already in df_lsoa:
    df_regions = df_regions.drop(['region', 'region_type'], axis='columns')
    df_lsoa = pd.merge(df_lsoa, df_regions, left_on='region_code', right_on='region_code', how='left')

    st.markdown('### Results by LSOA')
    st.write(df_lsoa)

    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with some object columns)
    df_lsoa = df_lsoa.drop([
        'lsoa', 'lsoa_code', 'nearest_ivt_unit', 'nearest_mt_unit', 'transfer_unit',
        'nearest_msu_unit', 'short_code', 'country'
        ], axis='columns')

    # Glob results by ICB:
    df_icb = df_lsoa.copy()
    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with some object columns)
    df_icb = df_icb.drop([
        'region', 'region_type', 'region_code', 'icb_code', 'isdn'
        ], axis='columns')
    # Average:
    df_icb = df_icb.groupby('icb').mean()

    # Glob results by ISDN:
    df_isdn = df_lsoa.copy()
    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with some object columns)
    df_isdn = df_isdn.drop([
        'region', 'region_type', 'region_code', 'icb', 'icb_code'
        ], axis='columns')
    # Average:
    df_isdn = df_isdn.groupby('isdn').mean()

    st.markdown('### Results by ISDN')
    st.write(df_isdn)

    st.markdown('### Results by ICB')
    st.write(df_icb)

    return gdf_boundaries_msoa


# ###########################
# ##### START OF SCRIPT #####
# ###########################
# page_setup()
st.set_page_config(
    page_title='MUSTER',
    page_icon=':ambulance:',
    layout='wide'
    )

# Make containers:
# +-----------------------+
# |    container_intro    |
# +-----------------------+
# |     container_map     |
# +-----------------------+
# | container_shared_data |
# +-----------------------+
# | container_params_here |
# +-----------------------+
# |  container_outcomes   |
# +-----------------------+
container_intro = st.container()
container_map_inputs = st.container()
container_map = st.empty()
container_shared_data = st.container()
container_params_here = st.container()
container_outcomes = st.container()

# ###########################
# ########## SETUP ##########
# ###########################

# Draw the input selection boxes in this function:
with st.sidebar:
    st.header('Pathway inputs')
    input_dict = inputs.select_parameters()

# Set up stroke unit services (IVT, MT, MSU).
from stroke_maps.catchment import Catchment
catchment = Catchment()
df_unit_services = catchment.get_unit_services()
# Remove Wales:
df_unit_services = df_unit_services.loc[df_unit_services['region_type'] != 'LHB'].copy()
df_unit_services_full = df_unit_services.copy()
# Limit which columns to show:
cols_to_keep = [
    'stroke_team',
    'use_ivt',
    'use_mt',
    'use_msu',
    # 'transfer_unit_postcode',  # to add back in later if stroke-maps replaces geography_processing class
    # 'region',
    # 'icb',
    'isdn'
]
df_unit_services = df_unit_services[cols_to_keep]
# Change 1/0 columns to bool for formatting:
cols_use = ['use_ivt', 'use_mt', 'use_msu']
df_unit_services[cols_use] = df_unit_services[cols_use].astype(bool)
# Display and store any changes from the user:
df_unit_services = st.data_editor(df_unit_services, disabled=['postcode', 'stroke_team', 'isdn'])

# Restore dtypes:
df_unit_services[cols_use] = df_unit_services[cols_use].astype(int)

# Update the full data (for maps) with the changes:
cols_to_merge = cols_use  # + ['transfer_unit_postcode']
df_unit_services_full = df_unit_services_full.drop(cols_to_merge, axis='columns')
df_unit_services_full = pd.merge(
    df_unit_services_full,
    df_unit_services[cols_to_merge].copy(),
    left_index=True, right_index=True, how='left'
    )

# Rename columns to match what the rest of the model here wants.
df_unit_services.index.name = 'Postcode'
df_unit_services = df_unit_services.rename(columns={
    'use_ivt': 'Use_IVT',
    'use_mt': 'Use_MT',
    'use_msu': 'Use_MSU',
})

with container_map_inputs:
    cols = st.columns(6)  # make more columns than needed to space closer
    scenario_dict = inputs.select_scenario(cols)

# If the requested data is nLVO + MT, stop now.
stop_bool = (
    (scenario_dict['stroke_type'] == 'nlvo') &
    ('mt' in scenario_dict['treatment_type'])
)
if stop_bool:
    st.warning('No data for nLVO with MT.')
    st.stop()

scenario_types = ['drip_ship', 'mothership', 'msu']
# Draw a blank map in a container and then replace the contents with
# this intended map once it's finished being drawn
with container_map:
    maps.plotly_blank_maps(scenario_types)

colour_dict = inputs.set_up_colours(scenario_dict | {'scenario_type': 'not diff'})

gdf_boundaries_msoa = main_calculations(input_dict, df_unit_services)

# Find geometry column for plot function:
col_geo = utils.find_multiindex_column_names(
    gdf_boundaries_msoa, property=['geometry'])


columns_colours = [
    '_'.join([
        scenario_dict['stroke_type'],
        scenario_type,
        scenario_dict['treatment_type'],
        scenario_dict['outcome_type']
    ])
    for scenario_type in scenario_types
    ]

maps.plotly_many_maps(
    gdf_boundaries_msoa,
    columns_colours,
    column_geometry=col_geo,
    v_bands=colour_dict['v_bands'],
    v_bands_str=colour_dict['v_bands_str'],
    colour_map=colour_dict['colour_map'],
    subplot_titles=scenario_types,
    legend_title=f'v: {scenario_dict["outcome_type_str"]}',
    container_map=container_map,
    df_units=df_unit_services_full
)

st.stop()
