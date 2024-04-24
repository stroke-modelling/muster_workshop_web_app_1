"""
OPTIMIST results display app.

Because a long app quickly gets out of hand,
try to keep this document to mostly direct calls to streamlit to write
or display stuff. Use functions in other files to create and
organise the stuff to be shown. In this example, most of the work is
done in functions stored in files named container_(something).py
"""
# ----- Imports -----
import streamlit as st

# Custom functions:
import utilities.calculations as calc
import utilities.maps as maps
# Containers:
import utilities.container_inputs as inputs


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
# +-----------------------------+
# |       container_intro       |
# +-----------------------------+
# |        container_map        |
# +-----------------------------+
# |    container_map_inputs     |
# +-----------------------------+
# |  container_results_tables   |
# +-----------------------------+
# |  container_select_outcome   |
# +-----------------------------+
container_intro = st.container()
with st.sidebar:
    container_inputs = st.container()
    container_unit_services = st.container()
container_map = st.empty()
container_map_inputs = st.container()
container_results_tables = st.container()
container_select_outcome = st.container()

# ###########################
# ########## SETUP ##########
# ###########################

with container_intro:
    st.markdown('# Benefit in outcomes from redirection')

# ----- User inputs -----
with container_inputs:
    with st.form('Model setup'):
        st.header('Pathway inputs')
        input_dict = inputs.select_parameters_optimist()

        st.header('Stroke unit services')
        st.markdown('Update which services the stroke units provide:')
        df_unit_services, df_unit_services_full = (
            inputs.select_stroke_unit_services())
        submitted = st.form_submit_button('Submit')

with container_map_inputs:
    cols = st.columns(2)
with container_select_outcome:
    st.markdown('### Alternative outcome measure for map')
    st.markdown('Try these if you dare.')
    scenario_dict = inputs.select_scenario(
        containers=[container_select_outcome] + cols
        )

# If the requested data is nLVO + MT, stop now.
try:
    stop_bool = ((scenario_dict['stroke_type'] in ['nlvo', 'combo']) &
                 ('mt' in scenario_dict['treatment_type']))
except KeyError:
    stop_bool = False
if stop_bool:
    st.warning('No data for nLVO with MT.')
    st.stop()


# ----- Setup for plots -----
# Which scenarios will be shown in the maps:
# (in this order)
scenario_types = ['drip_ship', 'diff_redirect_minus_drip_ship']

legend_title = ''.join([
    f'v: {scenario_dict["outcome_type_str"]};<br>',
    'd: Benefit of redirection over drip-and-ship'
    ])

# Which subplots to draw which units on:
# Each entry is [row number, column number].
# In plotly, the first row is 1 and first column is 1.
# The order in which they are drawn (and so which markers appear
# on top) is the order of this dictionary.
unit_subplot_dict = {
    'ivt': [[1, 1], [1, 2]],  # both maps
    'mt': [[1, 1], [1, 2]]    # both maps
}

# Draw a blank map in a container and then replace the contents with
# this intended map once it's finished being drawn
with container_map:
    maps.plotly_blank_maps(scenario_types, n_blank=2)

# ----- Main calculations -----
# Process LSOA and calculate outcomes:
df_lsoa = calc.calculate_outcomes(input_dict, df_unit_services)

# Remove the MSU data:
cols_to_remove = [c for c in df_lsoa.columns if 'msu' in c]
df_lsoa = df_lsoa.drop(cols_to_remove, axis='columns')
# Current setup means that MSU data is calculated with default
# values even if we don't explicitly give it any parameters.

# Extra calculations for redirection:
# Combine drip-and-ship and mothership results in proportions given:
redirect_dict = {
    'sensitivity': input_dict['sensitivity'],
    'specificity': input_dict['specificity'],
}
df_lsoa = calc.combine_results_by_redirection(df_lsoa, redirect_dict)

# Make combined nLVO + LVO data in the proportions given:
prop_dict = {
    'nlvo': input_dict['prop_nlvo'],
    'lvo': input_dict['prop_lvo']
}
df_lsoa = calc.combine_results_by_occlusion_type(df_lsoa, prop_dict)

# Calculate diff - redirect minus drip-ship:
df_lsoa = calc.combine_results_by_diff(df_lsoa)

gdf_boundaries_msoa = maps.combine_geography_with_outcomes(df_lsoa)
df_icb, df_isdn, df_nearest_ivt = calc.group_results_by_region(
    df_lsoa, df_unit_services)


with container_results_tables:
    results_tabs = st.tabs([
        'Results by IVT unit catchment',
        'Results by ISDN',
        'Results by ICB',
        'Full results by LSOA'
        ])

    # Set some columns to bool for nicer display:
    cols_bool = ['transfer_required', 'England']
    for col in cols_bool:
        for df in [df_icb, df_isdn, df_nearest_ivt, df_lsoa]:
            df[col] = df[col].astype(bool)

    with results_tabs[0]:
        st.markdown('### Results by nearest IVT unit')
        st.markdown('Results are the mean values of all LSOA in each IVT unit catchment area.')
        st.dataframe(df_nearest_ivt)

    with results_tabs[1]:
        st.markdown('### Results by ISDN')
        st.markdown('Results are the mean values of all LSOA in each ISDN.')
        st.dataframe(df_isdn)

    with results_tabs[2]:
        st.markdown('### Results by ICB')
        st.markdown('Results are the mean values of all LSOA in each ICB.')
        st.dataframe(df_icb)

    with results_tabs[3]:
        st.markdown('### Results by LSOA')
        st.dataframe(df_lsoa)

# ----- Colour setup -----
# Give the scenario dict a dummy 'scenario_type' entry
# so that the right colour map and colour limits are picked.
colour_dict = inputs.set_up_colours(
    scenario_dict | {'scenario_type': 'not diff'})
colour_diff_dict = inputs.set_up_colours(
    scenario_dict | {'scenario_type': 'diff'}, v_name='d')
# Find the names of the columns that contain the data
# that will be shown in the colour maps.
columns_colours = [
    '_'.join([
        scenario_dict['stroke_type'],
        scenario_type,
        scenario_dict['treatment_type'],
        scenario_dict['outcome_type']
    ])
    for scenario_type in scenario_types
    ]
colour_dict['column'] = columns_colours[0]
colour_diff_dict['column'] = columns_colours[1]

# Make one combined GeoDataFrame of all of the separate maps
# that will be used across all subplots.
gdf_polys, combo_colour_map = maps.create_combo_gdf_for_plotting(
    gdf_boundaries_msoa,
    colour_dicts=[colour_dict, colour_diff_dict],
    subplot_titles=scenario_types,
    legend_title=legend_title,
)

maps.plotly_many_maps(
    gdf_polys,
    combo_colour_map,
    subplot_titles=scenario_types,
    legend_title=legend_title,
    container_map=container_map,
    df_units=df_unit_services_full,
    unit_subplot_dict=unit_subplot_dict
)
