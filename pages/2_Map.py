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

from stroke_maps.geo import check_scenario_level
# Custom functions:
import utilities.utils as utils
import utilities.maps as maps
# Containers:
import utilities.container_inputs as inputs
import utilities.container_results as results


@st.cache_data
def main_calculations(input_dict):
    # Run the outcomes with the selected pathway:
    df_lsoa = results.make_outcomes(input_dict)

    st.markdown('### Results by LSOA')
    st.write(df_lsoa)

    # TO DO - the results df contains a mix of scenarios
    # (drip and ship, mothership, msu) in the column names.
    # Pull them out and put them into 'scenario' header.
    # Also need to do something with separate nlvo, lvo, treatment types
    # because current setup just wants some averaged added utility outcome
    # rather than split by stroke type.

    df_msoa = inputs.convert_lsoa_to_msoa_results(df_lsoa)

    # Check whether the input DataFrames have a 'scenario' column level.
    # This is required for talking to stroke-maps package.
    # If not, add one now with a placeholder scenario name.
    df_msoa = check_scenario_level(df_msoa)

    # Merge outcome and geography:
    gdf_boundaries_msoa = maps._load_geometry_msoa(df_msoa)
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

gdf_boundaries_msoa = main_calculations(input_dict)

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
    container_map=container_map
)

st.stop()
