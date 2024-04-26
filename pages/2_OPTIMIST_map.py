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
    page_title='OPTIMIST',
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
            inputs.select_stroke_unit_services(use_msu=False))
        submitted = st.form_submit_button('Submit')

with container_map_inputs:
    cols = st.columns(2)
with container_select_outcome:
    st.markdown('### Alternative outcome measure for map')
    st.markdown('Try these if you dare.')
    scenario_dict = inputs.select_scenario(
        containers=[container_select_outcome] + cols,
        use_combo_stroke_types=True
        )



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

# If the requested data is nLVO + MT, stop now.
try:
    stop_bool = ((scenario_dict['stroke_type'] in ['nlvo']) &
                 (scenario_dict['treatment_type'] == 'mt'))
except KeyError:
    stop_bool = False
if stop_bool:
    with container_map_inputs:
        st.warning('No data for nLVO with MT.')
        st.stop()

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

# # Create hospital catchment areas from this MSOA geography data.
# cols = [('nearest_ivt_unit', 'scenario'), ('geometry', 'any')]
# import pandas as pd
# gdf_catchment = maps.find_geometry_ivt_catchment(pd.DataFrame(gdf_boundaries_msoa[cols]))
# # Save:
# gdf_catchment.to_file(f'data/outline_nearest_ivt.geojson')



# Make dummy polygons:
gdf_dummy, combo_colour_map = maps.create_dummy_colour_gdf(
    [colour_dict, colour_diff_dict])

# Left-hand subplot colours:
# For each colour scale and data column combo,
# merge polygons that fall into the same colour band.
gdf_lhs = maps.dissolve_polygons_by_colour(
    gdf_boundaries_msoa,
    colour_dict['column'],
    colour_dict['v_bands'],
    colour_dict['v_bands_str'],
    combo_colour_map
    )

# Right-hand subplot colours:
gdf_rhs = maps.dissolve_polygons_by_colour(
    gdf_boundaries_msoa,
    colour_diff_dict['column'],
    colour_diff_dict['v_bands'],
    colour_diff_dict['v_bands_str'],
    combo_colour_map
    )

# Region outlines:
# Load in another gdf:
import geopandas
from shapely.validation import make_valid  # for fixing dodgy polygons
gdf_catchment = geopandas.read_file('./data/outline_isdns.geojson')
# Make geometry valid:
gdf_catchment['geometry'] = [
    make_valid(g) if g is not None else g
    for g in gdf_catchment['geometry'].values
    ]
# st.write(gdf_catchment)
# Make colour transparent:
gdf_catchment['colour'] = 'rgba(0, 0, 0, 0)'
gdf_catchment['outline_type'] = 'ISDN'

# Stroke unit scatter markers:
# traces_units = maps.create_stroke_team_markers(df_unit_services_full)

# Convert gdf polygons to xy cartesian coordinates:
for gdf in [gdf_dummy, gdf_lhs, gdf_rhs, gdf_catchment]:
    x_list, y_list = maps.convert_shapely_polys_into_xy(gdf)
    gdf['x'] = x_list
    gdf['y'] = y_list


import plotly.graph_objects as go
from plotly.subplots import make_subplots
# ----- Plotting -----
fig = make_subplots(rows=1, cols=2)

# Add each row of the dataframe separately.
# Scatter the edges of the polygons and use "fill" to colour
# within the lines.
gdf = gdf_dummy
for i in gdf.index:
    fig.add_trace(go.Scatter(
        x=gdf.loc[i, 'x'],
        y=gdf.loc[i, 'y'],
        mode='lines',
        fill="toself",
        fillcolor=gdf.loc[i, 'colour'],
        line_width=0,
        hoverinfo='skip',
        name=gdf.loc[i, 'colour_str'],
        ), row='all', col='all'
        )
    
gdf = gdf_lhs
for i in gdf.index:
    fig.add_trace(go.Scatter(
        x=gdf.loc[i, 'x'],
        y=gdf.loc[i, 'y'],
        mode='lines',
        fill="toself",
        fillcolor=gdf.loc[i, 'colour'],
        line_width=0,
        hoverinfo='skip',
        name=gdf.loc[i, 'colour_str'],
        showlegend=False
        ), row='all', col=1
        )    

gdf = gdf_rhs
for i in gdf.index:
    fig.add_trace(go.Scatter(
        x=gdf.loc[i, 'x'],
        y=gdf.loc[i, 'y'],
        mode='lines',
        fill="toself",
        fillcolor=gdf.loc[i, 'colour'],
        line_width=0,
        hoverinfo='skip',
        name=gdf.loc[i, 'colour_str'],
        showlegend=False
        ), row='all', col=2
        )

gdf = gdf_catchment
# I can't for the life of me get hovertemplate working here
# for mysterious reasons, so just stick to "text" for hover info.
for i in gdf.index:
    fig.add_trace(go.Scatter(
        x=gdf.loc[i, 'x'],
        y=gdf.loc[i, 'y'],
        mode='lines',
        fill="toself",
        fillcolor=gdf.loc[i, 'colour'],
        line_color='black',
        name=gdf.loc[i, 'outline_type'],
        text=gdf.loc[i, 'isdn'],
        hoverinfo="text",
        ), row='all', col='all'
        )

# Equivalent to pyplot set_aspect='equal':
fig.update_yaxes(col=1, scaleanchor='x', scaleratio=1)
fig.update_yaxes(col=2, scaleanchor='x2', scaleratio=1)

# Shared pan and zoom settings:
fig.update_xaxes(matches='x')
fig.update_yaxes(matches='y')

# Remove axis ticks:
fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

# --- Stroke unit scatter markers ---
if len(unit_subplot_dict) > 0:
    # # Add a blank trace to put a gap in the legend.
    # Stupid? Yes. Works? Also yes.
    # Make sure the name isn't the same as any other blank name
    # already set, e.g. in combo_colour_dict, or this repeat
    # entry will be deleted later.
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        marker={'color': 'rgba(0,0,0,0)'},
        name=' ' * 10
    ))

    # Create the scatter traces for the stroke units...
    traces = maps.create_stroke_team_markers(df_unit_services_full)

    # ... and THEN add traces to the subplots.
    for service, grid_lists in unit_subplot_dict.items():
        for grid_list in grid_lists:
            row = grid_list[0]
            col = grid_list[1]
            fig.add_trace(traces[service], row=row, col=col)

# Remove repeat legend names:
# from https://stackoverflow.com/a/62162555
names = set()
fig.for_each_trace(
    lambda trace:
        trace.update(showlegend=False)
        if (trace.name in names) else names.add(trace.name))
# This makes sure that if multiple maps use the exact same
# colours and labels, the labels only appear once in the legend.

# Figure setup.
fig.update_layout(
    width=1200,
    height=700,
    margin_t=40,
    margin_b=0
    )

# Disable clicking legend to remove trace:
fig.update_layout(legend_itemclick=False)
fig.update_layout(legend_itemdoubleclick=False)

if container_map is None:
    container_map = st.container()
with container_map:
    st.plotly_chart(fig)

# st.plotly_chart(fig)


# # Make one combined GeoDataFrame of all of the separate maps
# # that will be used across all subplots.
# gdf_polys, combo_colour_map = maps.create_combo_gdf_for_plotting(
#     gdf_boundaries_msoa,
#     colour_dicts=[colour_dict, colour_diff_dict],
#     subplot_titles=scenario_types,
#     legend_title=legend_title,
#     gdf_catchment=gdf_catchment
# )

# maps.plotly_many_maps(
#     gdf_polys,
#     combo_colour_map,
#     subplot_titles=scenario_types,
#     legend_title=legend_title,
#     container_map=container_map,
#     df_units=df_unit_services_full,
#     unit_subplot_dict=unit_subplot_dict
# )
